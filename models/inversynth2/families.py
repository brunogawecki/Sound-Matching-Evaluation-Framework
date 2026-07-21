"""The InverSynth II benchmark families: the ``BaseDeepModel`` wrappers.

InverSynth II (Barkan et al., ISMIR 2023) fills this benchmark's **neural-proxy** family
slot -- a peer paper approach alongside the discriminative (Sound2Synth) and generative
(preset-gen-vae) families, not a baseline. The paper stacks three models, built here in stages
under the paper's own names:

- ``IS``      -- encoder, parameters-loss only (Stage 1).
- ``IS2xITF`` -- ``IS`` plus a differentiable neural synthesizer-proxy and an audio loss during
  training, but **without** inference-time finetuning. The "x" reads *excluding* ITF (Stage 2).
- ``IS2``     -- the full model, ``IS2xITF`` **with** per-sample inference-time finetuning (Stage 3).

The synthesizer-proxy (Stages 2-3) is a training-only component: it supplies gradients for the
audio loss and never touches evaluation. ``predict`` always returns a synth-dict and the
``Evaluator`` re-renders with the real Dexed (D-EVAL / D-REPRO).
"""
from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from dataset.torch_dataset import RenderedCorpusDataset
from models.base_deep_model import BaseDeepModel
from models.inversynth2.network import InverSynthEncoderNetwork, IS2Network
from models.presetgen_vae.network import measure_corpus_mel_db_range
from models.training.config import LossConfig, TrainingConfig
from models.training.loss import ParameterLoss
from synth.parameter_space import ParameterSpace


class BaseInverSynthModel(BaseDeepModel):
    """Shared front-end plumbing for the InverSynth II families.

    Holds the mel/STFT constructor knobs and builds the front-end ``architecture_hparams`` common
    to every stage: ``ml_dimension``, render length and sample rate read from the corpus
    (D-SELFDESC), and the mel-dB normalization endpoints measured over the train corpus
    (D-MELNORM), all folded in so ``load`` rebuilds the identical network offline (no VST, no
    Lightning). Concrete stages add their network-specific hparams and their Lightning recipe.
    Not registered itself.
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 257,
        mel_fmin: float = 30.0,
        mel_fmax: float = 11000.0,
        spectrogram_min_db: float = -120.0,
        spectrogram_max_db: float = 0.0,
        dropout: float = 0.3,
        default_root_dir: str = "lightning_logs",
    ) -> None:
        super().__init__(default_root_dir=default_root_dir)
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._win_length = win_length
        self._n_mels = n_mels
        self._mel_fmin = mel_fmin
        self._mel_fmax = mel_fmax
        self._spectrogram_min_db = spectrogram_min_db
        self._spectrogram_max_db = spectrogram_max_db
        self._dropout = dropout

    @staticmethod
    def _corpus_sample_rate(train_dataset: RenderedCorpusDataset) -> int:
        """Read the render sample rate from the corpus's ``run_summary.json``."""
        with open(train_dataset.corpus_dir / "run_summary.json") as summary_file:
            return int(json.load(summary_file)["sample_rate"])

    def _front_end_hparams(
        self, train_dataset: RenderedCorpusDataset, parameter_space: ParameterSpace
    ) -> Dict[str, Any]:
        """The mel/STFT + corpus-derived hparams shared by every InverSynth II network."""
        example_audio, _ = train_dataset[0]
        sample_rate = self._corpus_sample_rate(train_dataset)
        min_db, max_db = measure_corpus_mel_db_range(
            train_dataset, sample_rate=sample_rate, n_fft=self._n_fft,
            hop_length=self._hop_length, win_length=self._win_length, n_mels=self._n_mels,
            mel_fmin=self._mel_fmin, mel_fmax=self._mel_fmax, db_floor=self._spectrogram_min_db,
        )
        return {
            "ml_dimension": parameter_space.ml_dimension,
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
            "dropout": self._dropout,
        }


class IS(BaseInverSynthModel):
    """The paper's ``IS`` model (Stage 1): a spectrogram -> parameters encoder, params loss only.

    The reference's strided-CNN encoder emitting the ML-side vector through ``ParameterSpace``,
    trained by the stock :class:`LightningRegressor` (:class:`ParameterLoss` only, no audio loss).
    """

    def _build_network(self, architecture_hparams: Dict[str, Any]) -> nn.Module:
        return InverSynthEncoderNetwork(
            ml_dimension=architecture_hparams["ml_dimension"],
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
            dropout=architecture_hparams["dropout"],
        )

    def _build_architecture_hparams(
        self, train_dataset: RenderedCorpusDataset, parameter_space: ParameterSpace
    ) -> Dict[str, Any]:
        return self._front_end_hparams(train_dataset, parameter_space)

    def _build_lightning_module(
        self, network: nn.Module, parameter_loss: ParameterLoss, training_config: TrainingConfig
    ):
        # Lazy: the training-only Lightning stack (D-FRAMEWORK) stays off the eval path.
        from models.training.lightning_module import LightningRegressor

        return LightningRegressor(network, parameter_loss, training_config.optimizer)


class IS2xITF(BaseInverSynthModel):
    """The paper's ``IS2`` model without inference-time finetuning (Stage 2; "x" = *excluding* ITF).

    ``IS``'s encoder plus a training-only differentiable synthesizer-proxy (:class:`IS2Network`),
    trained by :class:`LightningIS2Regressor` on the paper's combined loss: parameters loss +
    ``lambda`` * proxy audio loss (its Eq. 4). The proxy supplies gradients only -- the saved
    checkpoint carries both encoder and proxy weights (Stage 3 ITF needs the proxy), but
    ``predict`` runs the encoder alone and the ``Evaluator`` re-renders with the real Dexed.
    ``proxy_dropout`` is the decoder's dropout; ``lambda`` is the config's ``loss.audio_loss_weight``.
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 257,
        mel_fmin: float = 30.0,
        mel_fmax: float = 11000.0,
        spectrogram_min_db: float = -120.0,
        spectrogram_max_db: float = 0.0,
        dropout: float = 0.3,
        proxy_dropout: float = 0.3,
        default_root_dir: str = "lightning_logs",
    ) -> None:
        super().__init__(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            mel_fmin=mel_fmin,
            mel_fmax=mel_fmax,
            spectrogram_min_db=spectrogram_min_db,
            spectrogram_max_db=spectrogram_max_db,
            dropout=dropout,
            default_root_dir=default_root_dir,
        )
        self._proxy_dropout = proxy_dropout

    def _build_network(self, architecture_hparams: Dict[str, Any]) -> nn.Module:
        return IS2Network(
            ml_dimension=architecture_hparams["ml_dimension"],
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
            dropout=architecture_hparams["dropout"],
            proxy_dropout=architecture_hparams["proxy_dropout"],
        )

    def _build_architecture_hparams(
        self, train_dataset: RenderedCorpusDataset, parameter_space: ParameterSpace
    ) -> Dict[str, Any]:
        hparams = self._front_end_hparams(train_dataset, parameter_space)
        hparams["proxy_dropout"] = self._proxy_dropout
        return hparams

    def _build_lightning_module(
        self, network: nn.Module, parameter_loss: ParameterLoss, training_config: TrainingConfig
    ):
        # Lazy: the training-only Lightning stack (D-FRAMEWORK) stays off the eval path.
        from models.inversynth2.lightning_module import LightningIS2Regressor

        return LightningIS2Regressor(
            network, parameter_loss, training_config.optimizer, training_config.loss
        )


class IS2(IS2xITF):
    """The paper's full ``IS2`` model (Stage 3): ``IS2xITF`` plus inference-time finetuning (ITF).

    Training is identical to :class:`IS2xITF` (same :class:`IS2Network`, same combined loss). The
    only difference is at prediction: ITF adapts the encoder to each test example (the paper's Eq.
    6). For a test waveform ``x_t`` the proxy ``d_phi*`` is frozen and the encoder ``theta*`` is
    fine-tuned for ``itf_steps`` alternations to minimize::

        L_t + itf_regularization_weight * L_B

    with ``L_t = MAE(proxy(encoder(x_t)), spectrogram(x_t))`` the self-supervised audio loss on the
    test example, and ``L_B`` the Eq. 4 combined loss (:class:`ParameterLoss` + ``lambda`` * proxy
    audio loss) over a random batch of a training pool cached at ``fit`` time -- the regularizer that
    keeps the encoder from forgetting its training and overfitting the single test example. The best
    step is kept and the encoder is restored to ``theta*`` afterwards, so ITF is per-sample and
    leaves the model unchanged for the next call.

    **ITF step selection (the paper's real-synth monitor).** The paper monitors the *real*
    synthesizer to select the best ITF step and to fall back to ``theta*``. ``predict`` gets that
    monitor through :meth:`set_itf_render_callback`: when the ``Evaluator`` (which owns the corpus
    render contract, D-EVAL) injects a ``synth_dict -> waveform`` callback, each step is scored by
    rendering the current prediction with the real Dexed and taking ``MAE`` between its mel-dB
    spectrogram and the target's -- the paper's ``L_t^f``. Only the *gradient* still flows through
    the frozen proxy (the synth is not differentiable); the real synth is used solely to pick the
    step and to keep ``theta*`` when no alternation improves. The Evaluator reuses one render process
    for this monitor (fast, small voice-state leak) while the final *scored* render stays fresh-
    process (D-REPRO); the monitor is only a selection heuristic, so the leak is acceptable.

    **VST-free fallback.** With no callback set (a direct ``predict`` off the Evaluator, or the
    VST-free cluster), selection falls back to the **proxy** audio loss ``L_t``: the lowest-proxy-loss
    step wins, ``theta*`` is kept if none improves. Because proxy and synth differ, a proxy-selected
    step is not guaranteed to lower the real-synth metric the ``Evaluator`` reports -- the discrepancy
    the real-synth monitor above removes.

    ``itf_steps`` / ``itf_batch_size`` / ``itf_regularization_weight`` (``lambda_B``) /
    ``itf_learning_rate`` are ITF knobs with the paper's defaults; ``itf_pool_size`` bounds how many
    training examples the checkpoint carries for ``L_B``.
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 257,
        mel_fmin: float = 30.0,
        mel_fmax: float = 11000.0,
        spectrogram_min_db: float = -120.0,
        spectrogram_max_db: float = 0.0,
        dropout: float = 0.3,
        proxy_dropout: float = 0.3,
        itf_steps: int = 30,
        itf_batch_size: int = 64,
        itf_learning_rate: float = 1.0e-4,
        itf_regularization_weight: float = 1.0,
        itf_pool_size: int = 256,
        itf_seed: int = 0,
        default_root_dir: str = "lightning_logs",
    ) -> None:
        super().__init__(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            mel_fmin=mel_fmin,
            mel_fmax=mel_fmax,
            spectrogram_min_db=spectrogram_min_db,
            spectrogram_max_db=spectrogram_max_db,
            dropout=dropout,
            proxy_dropout=proxy_dropout,
            default_root_dir=default_root_dir,
        )
        self._itf_steps = int(itf_steps)
        self._itf_batch_size = int(itf_batch_size)
        self._itf_learning_rate = float(itf_learning_rate)
        self._itf_regularization_weight = float(itf_regularization_weight)
        self._itf_pool_size = int(itf_pool_size)
        self._itf_seed = int(itf_seed)

        # Training-derived; set at fit time and restored from the checkpoint at load time.
        self._itf_pool_audio: Optional[torch.Tensor] = None
        self._itf_pool_targets: Optional[torch.Tensor] = None
        self._itf_audio_loss_weight = 1.0  # lambda of Eq. 4, used inside L_B
        self._itf_continuous_loss = "mse"
        self._itf_categorical_loss_weight = 1.0

        # Optional real-synth ITF monitor: a ``synth_dict -> waveform`` callback the Evaluator
        # injects at eval time (paper's L_t^f). None -> selection falls back to the proxy loss.
        self._itf_render_callback: Optional[Callable[[Dict[str, float]], np.ndarray]] = None

    def set_itf_render_callback(
        self, render_callback: Optional[Callable[[Dict[str, float]], np.ndarray]]
    ) -> None:
        """Give ITF a real synthesizer to monitor step selection with (the paper's ``L_t^f``).

        ``render_callback`` maps an estimated synth-side dict to a mono waveform at the corpus
        sample rate (the caller merges the corpus default parameters and owns the render contract,
        D-EVAL). The ``Evaluator`` sets this to a reused-process Dexed renderer before scoring and
        clears it afterwards; pass ``None`` to restore the VST-free proxy monitor.
        """
        self._itf_render_callback = render_callback

    # -- fit: train as IS2xITF, then cache the ITF pool + Eq. 4 loss knobs ----
    def fit(
        self,
        train_dataset: RenderedCorpusDataset,
        validation_dataset: Optional[RenderedCorpusDataset] = None,
        config: Optional[Dict[str, object]] = None,
    ) -> None:
        super().fit(train_dataset, validation_dataset, config)
        training_config = TrainingConfig.from_dict(config)
        # ITF's L_B reuses the exact Eq. 4 recipe, so remember its loss knobs (persisted below).
        self._itf_audio_loss_weight = float(training_config.loss.audio_loss_weight)
        self._itf_continuous_loss = training_config.loss.continuous_loss
        self._itf_categorical_loss_weight = float(training_config.loss.categorical_loss_weight)
        self._capture_itf_pool(train_dataset, training_config.seed)

    def _capture_itf_pool(self, train_dataset: RenderedCorpusDataset, seed: int) -> None:
        """Stash a random slice of the training set for L_B (bounded, so the checkpoint stays small)."""
        pool_size = min(self._itf_pool_size, len(train_dataset))
        generator = torch.Generator().manual_seed(int(seed))
        indices = torch.randperm(len(train_dataset), generator=generator)[:pool_size].tolist()
        self._itf_pool_audio = torch.stack([train_dataset[i][0] for i in indices])
        self._itf_pool_targets = torch.stack([train_dataset[i][1] for i in indices])

    # -- checkpoint: carry the pool + loss knobs so load() can run ITF offline -
    def _extra_checkpoint_state(self) -> Optional[Dict[str, Any]]:
        if self._itf_pool_audio is None or self._itf_pool_targets is None:
            return None
        return {
            "itf_pool_audio": self._itf_pool_audio.cpu(),
            "itf_pool_targets": self._itf_pool_targets.cpu(),
            "itf_audio_loss_weight": self._itf_audio_loss_weight,
            "itf_continuous_loss": self._itf_continuous_loss,
            "itf_categorical_loss_weight": self._itf_categorical_loss_weight,
        }

    def _restore_extra_checkpoint_state(self, extra_state: Optional[Dict[str, Any]]) -> None:
        if extra_state is None:
            raise RuntimeError(
                "IS2 checkpoint is missing its ITF training pool; the model must be fit as IS2 "
                "(an IS2xITF checkpoint does not carry the pool ITF needs)."
            )
        self._itf_pool_audio = extra_state["itf_pool_audio"]
        self._itf_pool_targets = extra_state["itf_pool_targets"]
        self._itf_audio_loss_weight = float(extra_state["itf_audio_loss_weight"])
        self._itf_continuous_loss = extra_state["itf_continuous_loss"]
        self._itf_categorical_loss_weight = float(extra_state["itf_categorical_loss_weight"])

    # -- predict: per-sample inference-time finetuning (Eq. 6) ----------------
    def predict(self, audio: torch.Tensor) -> Dict[str, float]:
        """Adapt the encoder to ``audio`` via ITF, then decode to a synth-side dict.

        Featurizes the target once, then -- inside a session that restores the model afterwards --
        fine-tunes the encoder to this one target (:meth:`_finetune_to_target`, the paper's Eq. 6),
        decodes the best step's prediction, and lets the session restore ``theta*``. So ITF is
        per-sample and leaves the model unchanged for the next call. Still returns a synth-dict --
        the ``Evaluator`` re-renders it fresh-process with the real Dexed.
        """
        if self._network is None or self._parameter_space is None:
            raise RuntimeError("Model must be fit (or loaded) before predict.")
        if self._itf_pool_audio is None or self._itf_pool_targets is None:
            raise RuntimeError("IS2 has no ITF training pool; fit or load the model first.")

        encoder = self._network.encoder
        device = next(self._network.parameters()).device
        with torch.no_grad():
            target_spectrogram = encoder.mel_db_spectrogram(audio.to(device).unsqueeze(0))

        with self._itf_encoder_session():
            best_encoder_state = self._finetune_to_target(target_spectrogram, device)
            encoder.load_state_dict(best_encoder_state)
            with torch.no_grad():
                vector = encoder.forward_from_spectrogram(target_spectrogram).squeeze(0).cpu().numpy()

        return self._parameter_space.ml_vector_to_synth_dict(vector)

    @contextmanager
    def _itf_encoder_session(self):
        """Run the enclosed ITF block with the model guaranteed restored afterwards.

        Puts the network in eval mode (so weight updates leave BatchNorm/dropout running stats
        untouched and forwards are batch-size-independent), freezes the proxy ``phi*``, and snapshots
        the encoder ``theta*``. On exit ``theta*`` is reloaded and the proxy unfrozen -- even if a
        step raises -- so per-sample finetuning never leaks into the next call.
        """
        encoder = self._network.encoder
        proxy = self._network.proxy
        self._network.eval()
        saved_encoder_state = copy.deepcopy(encoder.state_dict())
        saved_proxy_grads = [parameter.requires_grad for parameter in proxy.parameters()]
        for parameter in proxy.parameters():
            parameter.requires_grad_(False)
        try:
            yield
        finally:
            encoder.load_state_dict(saved_encoder_state)
            for parameter, requires_grad in zip(proxy.parameters(), saved_proxy_grads):
                parameter.requires_grad_(requires_grad)

    def _finetune_to_target(
        self, target_spectrogram: torch.Tensor, device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """The ITF loop (Eq. 6): the encoder ``state_dict`` that best matches ``target_spectrogram``.

        Runs ``itf_steps`` Adam alternations on the encoder minimizing ``L_t + lambda_B * L_B``:
        :meth:`_proxy_audio_loss` on the target (``L_t``, the finetuning signal) plus
        :func:`regularizer_loss` over a fresh random training-pool batch (``L_B``, the anchor that
        stops the encoder overfitting the single target). Each step is scored by
        :func:`score_current_encoder` -- the real synth when a render callback is set (the paper's
        ``L_t^f``), the proxy otherwise -- and the best-scoring step is kept, with ``theta*`` itself
        the fallback when no step improves. Must run inside :meth:`_itf_encoder_session`.
        """
        encoder = self._network.encoder
        proxy = self._network.proxy
        parameter_loss = ParameterLoss(
            self._parameter_space,
            LossConfig(
                continuous_loss=self._itf_continuous_loss,
                categorical_loss_weight=self._itf_categorical_loss_weight,
            ),
        ).to(device)

        # Featurization is param-free: featurize the pool once, re-run only the CNN + head per step.
        with torch.no_grad():
            pool_spectrograms = encoder.mel_db_spectrogram(self._itf_pool_audio.to(device))
        pool_targets = self._itf_pool_targets.to(device)
        pool_count = pool_spectrograms.shape[0]
        batch_size = min(self._itf_batch_size, pool_count)
        batch_generator = torch.Generator().manual_seed(self._itf_seed)

        def regularizer_loss() -> torch.Tensor:
            """L_B: the Eq. 4 combined loss over a fresh random training-pool batch."""
            indices = torch.randint(
                pool_count, (batch_size,), generator=batch_generator
            ).to(device)
            batch_spectrograms = pool_spectrograms.index_select(0, indices)
            batch_predictions = encoder.forward_from_spectrogram(batch_spectrograms)
            parameter_term = parameter_loss(
                batch_predictions, pool_targets.index_select(0, indices)
            )["loss"]
            audio_term = F.l1_loss(proxy(batch_predictions), batch_spectrograms)
            return parameter_term + self._itf_audio_loss_weight * audio_term

        def score_current_encoder() -> float:
            """Step-selection score of the encoder's current weights: real synth if a callback is set, else proxy."""
            with torch.no_grad():
                prediction = encoder.forward_from_spectrogram(target_spectrogram)
                if self._itf_render_callback is None:
                    return F.l1_loss(proxy(prediction), target_spectrogram).item()
                synth_dict = self._parameter_space.ml_vector_to_synth_dict(
                    prediction.squeeze(0).cpu().numpy()
                )
                rendered = torch.from_numpy(
                    np.asarray(self._itf_render_callback(synth_dict), dtype=np.float32)
                ).to(device).unsqueeze(0)
                return F.l1_loss(encoder.mel_db_spectrogram(rendered), target_spectrogram).item()

        best_encoder_state = copy.deepcopy(encoder.state_dict())
        best_loss = score_current_encoder()
        optimizer = torch.optim.Adam(encoder.parameters(), lr=self._itf_learning_rate)
        for _ in range(self._itf_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = (
                self._proxy_audio_loss(target_spectrogram)
                + self._itf_regularization_weight * regularizer_loss()
            )
            loss.backward()
            optimizer.step()

            step_loss = score_current_encoder()
            if step_loss < best_loss:
                best_loss = step_loss
                best_encoder_state = copy.deepcopy(encoder.state_dict())
        return best_encoder_state

    def _proxy_audio_loss(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """L_t: the frozen proxy's reconstruction MAE of the encoder's prediction for ``spectrogram``."""
        encoder = self._network.encoder
        proxy = self._network.proxy
        prediction = encoder.forward_from_spectrogram(spectrogram)
        return F.l1_loss(proxy(prediction), spectrogram)
