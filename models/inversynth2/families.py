"""The InverSynth II benchmark families: the ``BaseDeepModel`` wrappers.

InverSynth II (Barkan et al., ISMIR 2023) fills this benchmark's **neural-proxy** family
slot -- a peer paper approach alongside the discriminative (Sound2Synth) and generative
(preset-gen-vae) families, not a baseline. The paper stacks three models, built here in stages
under the paper's own names:

- ``IS``      -- encoder, parameters-loss only (this file, Stage 1).
- ``IS2xITF`` -- ``IS`` plus a differentiable neural synthesizer-proxy and an audio loss during
  training, but **without** inference-time finetuning. The "x" reads *excluding* ITF (Stage 2).
- ``IS2``     -- the full model, ``IS2xITF`` **with** per-sample inference-time finetuning (Stage 3).

The synthesizer-proxy (Stages 2-3) is a training-only component: it supplies gradients for the
audio loss and never touches evaluation. ``predict`` always returns a synth-dict and the
``Evaluator`` re-renders with the real Dexed (D-EVAL / D-REPRO).
"""
from __future__ import annotations

import json
from typing import Any, Dict

from torch import nn

from dataset.torch_dataset import RenderedCorpusDataset
from models.base_deep_model import BaseDeepModel
from models.inversynth2.network import InverSynthEncoderNetwork
from models.presetgen_vae.network import measure_corpus_mel_db_range
from models.training.config import TrainingConfig
from models.training.loss import ParameterLoss
from synth.parameter_space import ParameterSpace


class IS(BaseDeepModel):
    """The paper's ``IS`` model (Stage 1): a spectrogram -> parameters encoder, params loss only.

    The reference's strided-CNN encoder emitting the ML-side vector through ``ParameterSpace``,
    trained by the stock :class:`LightningRegressor` (:class:`ParameterLoss` only, no audio
    loss). The mel/STFT knobs are constructor arguments; ``ml_dimension``, ``num_audio_samples``
    and ``sample_rate`` are read from the corpus at ``fit`` time and the mel-dB normalization
    endpoints are measured over the train corpus (D-MELNORM), all folded into
    ``architecture_hparams`` so ``load`` can rebuild the exact network before restoring weights
    (no VST, no Lightning). Reading the render length + sample rate from the corpus (not the
    constructor) keeps the network aligned with the self-describing corpus (D-SELFDESC).
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

    @staticmethod
    def _corpus_sample_rate(train_dataset: RenderedCorpusDataset) -> int:
        """Read the render sample rate from the corpus's ``run_summary.json``."""
        with open(train_dataset.corpus_dir / "run_summary.json") as summary_file:
            return int(json.load(summary_file)["sample_rate"])

    def _build_architecture_hparams(
        self, train_dataset: RenderedCorpusDataset, parameter_space: ParameterSpace
    ) -> Dict[str, Any]:
        """The network hparams for build + checkpoint.

        Render length + sample rate come from the corpus (D-SELFDESC); the mel-dB normalization
        endpoints are measured over the train corpus (D-MELNORM), overriding the constructor
        defaults, so ``load`` rebuilds the identical front-end offline.
        """
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

    def _build_lightning_module(
        self, network: nn.Module, parameter_loss: ParameterLoss, training_config: TrainingConfig
    ):
        # Lazy: the training-only Lightning stack (D-FRAMEWORK) stays off the eval path.
        from models.training.lightning_module import LightningRegressor

        return LightningRegressor(network, parameter_loss, training_config.optimizer)
