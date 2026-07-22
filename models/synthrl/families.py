"""The SynthRL benchmark families: the ``BaseDeepModel`` wrappers.

SynthRL (Shin & Lee, IJCAI-25) is a transformer encoder-decoder trained in stages: a
parameter-loss pretraining stage (SynthRL-p), an in-domain reinforcement-learning stage
(SynthRL-i), and an out-of-domain RL stage (SynthRL-o). This module holds the family
wrappers; the network is :class:`SynthRLNetwork` and the loss recipes live in
``lightning_module.py``.

Two things differ from every other family here. The network emits **class logits**, not
the ML-side vector, so ``predict`` decodes through :class:`SynthRLRepresentation` rather
than ``ParameterSpace.ml_vector_to_synth_dict``. And the parameter loss is the paper's
Gaussian-smoothed per-parameter cross-entropy, so the injected :class:`ParameterLoss` is
ignored (as in the flow-matching family).

Two stages land here: **SynthRL-p** (the RL-free parameter stage) and **SynthRL-i** (the
in-domain RL stage, warm-started from a SynthRL-p checkpoint). SynthRL-o (out-of-domain
RL) is deferred, gated on D-FAMILIES. Both stages are fully evaluable through the registry
+ ``Evaluator`` -- ``predict`` is identical and VST-free for both.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
from torch import nn

from dataset.render_backends import RenderSettings
from dataset.torch_dataset import RenderedCorpusDataset
from models.base_deep_model import BaseDeepModel
from models.presetgen_vae.network import measure_corpus_mel_db_range
from models.synthrl.network import SynthRLNetwork
from models.synthrl.representation import (
    DEFAULT_LABEL_SMOOTHING_SIGMA,
    DEFAULT_NUM_BINS,
    SynthRLRepresentation,
)
from models.synthrl.reward import RewardWeights
from models.training.checkpoint import load_checkpoint
from models.training.config import TrainingConfig
from models.training.loss import ParameterLoss
from synth.parameter_space import ParameterSpace


class BaseSynthRLModel(BaseDeepModel):
    """Shared plumbing for the SynthRL families: front-end, network, representation.

    Holds the mel/STFT + transformer constructor knobs and the SynthRL-local
    discretization knobs (``num_bins`` / ``label_smoothing_sigma``). Builds the common
    ``architecture_hparams``: ``class_counts`` from the representation, render length and
    sample rate read from the corpus (D-SELFDESC), and the mel-dB normalization endpoints
    measured over the train corpus (D-MELNORM), all folded in so ``load`` rebuilds the
    identical network + representation offline. Not registered itself.
    """

    def __init__(
        self,
        num_bins: int = DEFAULT_NUM_BINS,
        label_smoothing_sigma: float = DEFAULT_LABEL_SMOOTHING_SIGMA,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 257,
        mel_fmin: float = 30.0,
        mel_fmax: float = 11000.0,
        spectrogram_min_db: float = -120.0,
        spectrogram_max_db: float = 0.0,
        d_model: int = 256,
        num_conv_layers: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
        default_root_dir: str = "lightning_logs",
        init_from_checkpoint: Optional[str] = None,
    ) -> None:
        super().__init__(default_root_dir=default_root_dir)
        self._init_from_checkpoint = init_from_checkpoint
        self._num_bins = num_bins
        self._label_smoothing_sigma = label_smoothing_sigma
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._win_length = win_length
        self._n_mels = n_mels
        self._mel_fmin = mel_fmin
        self._mel_fmax = mel_fmax
        self._spectrogram_min_db = spectrogram_min_db
        self._spectrogram_max_db = spectrogram_max_db
        self._d_model = d_model
        self._num_conv_layers = num_conv_layers
        self._num_encoder_layers = num_encoder_layers
        self._num_decoder_layers = num_decoder_layers
        self._num_heads = num_heads
        self._feedforward_dim = feedforward_dim
        self._dropout = dropout
        # Built in _build_architecture_hparams, consumed by _build_lightning_module (same fit call).
        self._training_representation: Optional[SynthRLRepresentation] = None
        self._training_render_settings: Optional[RenderSettings] = None
        self._training_sample_rate: Optional[int] = None

    @staticmethod
    def _corpus_sample_rate(train_dataset: RenderedCorpusDataset) -> int:
        """Read the render sample rate from the corpus's ``run_summary.json``."""
        with open(train_dataset.corpus_dir / "run_summary.json") as summary_file:
            return int(json.load(summary_file)["sample_rate"])

    @staticmethod
    def _corpus_render_settings(train_dataset: RenderedCorpusDataset) -> RenderSettings:
        """Read the corpus's render contract (for re-rendering in the RL stage)."""
        with open(train_dataset.corpus_dir / "run_summary.json") as summary_file:
            return RenderSettings(**json.load(summary_file)["render_settings"])

    def _warm_start_network(self, network: nn.Module, architecture_hparams: Dict[str, Any]) -> None:
        """Load a prior SynthRL checkpoint's weights before training, if configured.

        SynthRL-i warm-starts from a SynthRL-p checkpoint. The network is byte-identical
        when the two stages share the corpus and discretization knobs, so ``load_state_dict``
        (strict) both restores the weights and guards against a mismatched checkpoint.
        """
        if self._init_from_checkpoint is None:
            return
        payload = load_checkpoint(Path(self._init_from_checkpoint))
        network.load_state_dict(payload["state_dict"])

    def _build_architecture_hparams(
        self, train_dataset: RenderedCorpusDataset, parameter_space: ParameterSpace
    ) -> Dict[str, Any]:
        representation = SynthRLRepresentation(
            parameter_space, num_bins=self._num_bins,
            label_smoothing_sigma=self._label_smoothing_sigma,
        )
        self._training_representation = representation

        example_audio, _ = train_dataset[0]
        sample_rate = self._corpus_sample_rate(train_dataset)
        # Stashed for the RL stage's in-loop re-rendering (ignored by the parameter stage).
        self._training_sample_rate = sample_rate
        self._training_render_settings = self._corpus_render_settings(train_dataset)
        min_db, max_db = measure_corpus_mel_db_range(
            train_dataset, sample_rate=sample_rate, n_fft=self._n_fft,
            hop_length=self._hop_length, win_length=self._win_length, n_mels=self._n_mels,
            mel_fmin=self._mel_fmin, mel_fmax=self._mel_fmax, db_floor=self._spectrogram_min_db,
        )
        return {
            "class_counts": representation.class_counts,
            "num_bins": self._num_bins,
            "label_smoothing_sigma": self._label_smoothing_sigma,
            "num_audio_samples": int(example_audio.shape[-1]),
            "sample_rate": sample_rate,
            "n_fft": self._n_fft,
            "hop_length": self._hop_length,
            "win_length": self._win_length,
            "n_mels": self._n_mels,
            "mel_fmin": self._mel_fmin,
            "mel_fmax": self._mel_fmax,
            "spectrogram_min_db": min_db,
            "spectrogram_max_db": max_db,
            "d_model": self._d_model,
            "num_conv_layers": self._num_conv_layers,
            "num_encoder_layers": self._num_encoder_layers,
            "num_decoder_layers": self._num_decoder_layers,
            "num_heads": self._num_heads,
            "feedforward_dim": self._feedforward_dim,
            "dropout": self._dropout,
        }

    def _build_network(self, architecture_hparams: Dict[str, Any]) -> nn.Module:
        return SynthRLNetwork(
            class_counts=architecture_hparams["class_counts"],
            num_audio_samples=architecture_hparams["num_audio_samples"],
            sample_rate=architecture_hparams["sample_rate"],
            n_fft=architecture_hparams["n_fft"],
            hop_length=architecture_hparams["hop_length"],
            win_length=architecture_hparams["win_length"],
            n_mels=architecture_hparams["n_mels"],
            mel_fmin=architecture_hparams["mel_fmin"],
            mel_fmax=architecture_hparams["mel_fmax"],
            spectrogram_min_db=architecture_hparams["spectrogram_min_db"],
            spectrogram_max_db=architecture_hparams["spectrogram_max_db"],
            d_model=architecture_hparams["d_model"],
            num_conv_layers=architecture_hparams["num_conv_layers"],
            num_encoder_layers=architecture_hparams["num_encoder_layers"],
            num_decoder_layers=architecture_hparams["num_decoder_layers"],
            num_heads=architecture_hparams["num_heads"],
            feedforward_dim=architecture_hparams["feedforward_dim"],
            dropout=architecture_hparams["dropout"],
        )

    def _representation(self) -> SynthRLRepresentation:
        """Rebuild the representation from the saved hparams (for decoding at predict)."""
        hparams = self._architecture_hparams
        return SynthRLRepresentation(
            self._parameter_space, num_bins=hparams["num_bins"],
            label_smoothing_sigma=hparams["label_smoothing_sigma"],
        )

    def predict(self, audio: torch.Tensor) -> Dict[str, float]:
        """Predict a synth-side dict for one waveform ``[num_samples]``.

        Runs the network to flat class logits, then argmax-decodes each per-parameter
        head to a synth-side dict through the representation (continuous heads decode to
        their bin center, categorical heads to their option). Overrides the base decode,
        which expects an ML-side vector.
        """
        if self._network is None or self._parameter_space is None:
            raise RuntimeError("Model must be fit (or loaded) before predict.")
        self._network.eval()
        audio = audio.to(next(self._network.parameters()).device)
        with torch.no_grad():
            logits = self._network(audio.unsqueeze(0))
        class_vector = logits.squeeze(0).cpu().numpy()
        return self._representation().class_logits_to_synth_dict(class_vector)


class SynthRLp(BaseSynthRLModel):
    """The paper's ``SynthRL-p`` model (stage 1): parameter loss only, no RL.

    The full SynthRL transformer trained by the Gaussian-smoothed per-parameter
    cross-entropy (:class:`SynthRLParameterRegressor`). This is the RL-free stage and the
    warm-start checkpoint the in-domain RL stage (SynthRL-i) later loads.
    """

    def _build_lightning_module(
        self, network: nn.Module, parameter_loss: ParameterLoss, training_config: TrainingConfig
    ):
        # Lazy: the training-only Lightning stack (D-FRAMEWORK) stays off the eval path.
        # The injected ParameterLoss is ignored -- SynthRL uses its own class-CE loss.
        from models.synthrl.lightning_module import SynthRLParameterRegressor

        return SynthRLParameterRegressor(
            network, self._training_representation, training_config.optimizer
        )


class SynthRLi(BaseSynthRLModel):
    """The paper's ``SynthRL-i`` model (stage 2): in-domain reinforcement learning.

    Warm-starts from a SynthRL-p checkpoint (``init_from_checkpoint``) and fine-tunes the
    same network with REINFORCE against a re-rendered audio-similarity reward
    (:class:`SynthRLReinforceRegressor`). Training re-renders with the real Dexed via the
    parallel fresh-process backend, so a SynthRL-i *training* run needs the plugin (a
    documented, training-only deviation from D-SELFDESC). ``predict`` is unchanged --
    it decodes class logits through the representation, no VST -- so the eval path is
    identical to SynthRL-p.

    RL hyperparameters come from the shared ``TrainingConfig``'s ``rl`` section
    (:class:`~models.training.config.RLConfig`), read at build time like InverSynth II's
    ``loss.audio_loss_weight``: the reward weights, the per-target PER buffer capacity,
    per-step sample count and pre-fill pass count, the parameter-loss -> RL curriculum ramp
    length (epochs), the render engine, and the render-worker count. Only ``backend_factory``
    (a test seam for injecting a fake renderer) stays a constructor argument.
    """

    def __init__(
        self,
        *,
        backend_factory: Optional[Callable[[], object]] = None,
        **base_kwargs: Any,
    ) -> None:
        super().__init__(**base_kwargs)
        self._backend_factory = backend_factory

    def _build_lightning_module(
        self, network: nn.Module, parameter_loss: ParameterLoss, training_config: TrainingConfig
    ):
        # Lazy: the training-only Lightning + render stack stays off the eval path.
        from models.synthrl.lightning_module import SynthRLReinforceRegressor

        rl = training_config.rl
        reward_weights = RewardWeights(
            spectrogram=rl.reward_spectrogram_weight,
            spectral_convergence=rl.reward_spectral_convergence_weight,
            mfcc=rl.reward_mfcc_weight,
        )
        return SynthRLReinforceRegressor(
            network,
            self._training_representation,
            training_config.optimizer,
            render_settings=self._training_render_settings,
            sample_rate=self._training_sample_rate,
            renderer=rl.renderer,
            num_render_workers=rl.num_render_workers,
            reward_weights=reward_weights,
            buffer_capacity=rl.buffer_capacity,
            samples_per_target=rl.samples_per_target,
            prefill_epochs=rl.prefill_epochs,
            ramp_epochs=rl.ramp_epochs,
            seed=training_config.seed,
            backend_factory=self._backend_factory,
        )
