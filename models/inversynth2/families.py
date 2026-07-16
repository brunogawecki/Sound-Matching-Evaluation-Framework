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

import json
from typing import Any, Dict

from torch import nn

from dataset.torch_dataset import RenderedCorpusDataset
from models.base_deep_model import BaseDeepModel
from models.inversynth2.network import InverSynthEncoderNetwork, IS2Network
from models.presetgen_vae.network import measure_corpus_mel_db_range
from models.training.config import TrainingConfig
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
