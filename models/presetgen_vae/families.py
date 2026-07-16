"""The preset-gen-vae benchmark families: the ``BaseDeepModel`` wrappers around
:class:`PresetGenVAENetwork`.

Two registry entries share one base: :class:`PresetGenVAEMLPRegressor` (the paper's MLP
regression) and :class:`PresetGenVAEFlowRegressor` (its flow regression) -- the two models
the paper reports, i.e. the "MLP" and "Flow" rows of its Table 1. They differ **only** in the
regressor head; both carry the latent RealNVP flow of the paper's Figure 1, and both train
through the same ``LightningVAERegressor`` recipe (reconstruction + beta-latent + parameters).

Following Table 1, both also pin ``latent_dimension`` to ``ml_dimension``: the flow head needs
it (invertibility), and the paper deliberately gives the MLP model the same latent width so the
MLP-vs-flow comparison is not confounded by latent size (its section 3.4.2).
"""
from __future__ import annotations

import json
from typing import Any, Dict

from torch import nn

from dataset.torch_dataset import RenderedCorpusDataset
from models.base_deep_model import BaseDeepModel
from models.presetgen_vae.network import PresetGenVAENetwork, measure_corpus_mel_db_range
from models.training.config import TrainingConfig
from models.training.loss import ParameterLoss
from synth.parameter_space import ParameterSpace


class BasePresetGenVAERegressor(BaseDeepModel):
    """Shared :class:`BaseDeepModel` family wrapping :class:`PresetGenVAENetwork`.

    The paper's ``FlowVAE``: a spectrogram autoencoder with a RealNVP flow on its latent and a
    regressor head, trained by :class:`LightningVAERegressor` (reconstruction + beta-latent +
    parameters). The mel/STFT/encoder/latent-flow/regressor knobs are constructor arguments;
    ``ml_dimension``, ``num_audio_samples`` and ``sample_rate`` are read from the corpus at
    ``fit`` time and folded into ``architecture_hparams`` so ``load`` can rebuild the exact
    network before restoring weights (no VST, no Lightning). Reading the render length + sample
    rate from the corpus (not the constructor) keeps the network aligned with the self-describing
    corpus (D-SELFDESC). ``latent_dimension`` is likewise not a constructor knob: it is pinned to
    ``ml_dimension`` at ``fit`` time, per Table 1. Concrete families pin the head via
    ``_regressor_architecture``: :class:`PresetGenVAEMLPRegressor` (``mlp``) and
    :class:`PresetGenVAEFlowRegressor` (``flow``). Not registered itself.
    """

    _regressor_architecture: str

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
        encoder_dropout: float = 0.3,
        latent_flow_layers: int = 6,
        latent_flow_hidden_features: int = 300,
        regressor_hidden_layers: int = 3,
        regressor_hidden_width: int = 1024,
        regressor_dropout: float = 0.4,
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
        self._encoder_dropout = encoder_dropout
        self._latent_flow_layers = latent_flow_layers
        self._latent_flow_hidden_features = latent_flow_hidden_features
        self._regressor_hidden_layers = regressor_hidden_layers
        self._regressor_hidden_width = regressor_hidden_width
        self._regressor_dropout = regressor_dropout

    def _build_network(self, architecture_hparams: Dict[str, Any]) -> nn.Module:
        return PresetGenVAENetwork(
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
            latent_dimension=architecture_hparams["latent_dimension"],
            encoder_dropout=architecture_hparams["encoder_dropout"],
            latent_flow_layers=architecture_hparams["latent_flow_layers"],
            latent_flow_hidden_features=architecture_hparams["latent_flow_hidden_features"],
            regressor_architecture=architecture_hparams["regressor_architecture"],
            regressor_hidden_layers=architecture_hparams["regressor_hidden_layers"],
            regressor_hidden_width=architecture_hparams["regressor_hidden_width"],
            regressor_dropout=architecture_hparams["regressor_dropout"],
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
        defaults, so ``load`` rebuilds the identical front-end offline. ``latent_dimension`` is
        pinned to the ML-side width for both families (Table 1).
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
            "latent_dimension": parameter_space.ml_dimension,
            "encoder_dropout": self._encoder_dropout,
            "latent_flow_layers": self._latent_flow_layers,
            "latent_flow_hidden_features": self._latent_flow_hidden_features,
            "regressor_architecture": self._regressor_architecture,
            "regressor_hidden_layers": self._regressor_hidden_layers,
            "regressor_hidden_width": self._regressor_hidden_width,
            "regressor_dropout": self._regressor_dropout,
        }

    def _build_lightning_module(
        self, network: nn.Module, parameter_loss: ParameterLoss, training_config: TrainingConfig
    ):
        # Lazy: the training-only Lightning stack (D-FRAMEWORK) stays off the eval path.
        from models.presetgen_vae.lightning_module import LightningVAERegressor

        return LightningVAERegressor(
            network, parameter_loss, training_config.optimizer, training_config.loss
        )


class PresetGenVAEMLPRegressor(BasePresetGenVAERegressor):
    """The paper's MLP-regression model, Table 1's "MLP" rows: a ``3l1024`` MLP head off zK.

    Three hidden layers of 1024 plus the output layer -- the paper's "4-layers MLP with 1024
    hidden units", batch-norm and dropout on the first two (its section 3.4.2). Constructor
    defaults are the paper's.
    """

    _regressor_architecture = "mlp"


class PresetGenVAEFlowRegressor(BasePresetGenVAERegressor):
    """The paper's flow-regression model, Table 1's "Flow" rows: a ``realnvp_6l300`` RealNVP
    head used feed-forward off zK.

    Same VAE (and same latent flow) as :class:`PresetGenVAEMLPRegressor`; only the head differs.
    ``regressor_hidden_layers`` / ``regressor_hidden_width`` mean coupling layers / hidden
    features here. The head is invertible, so it also *requires* the latent to be exactly as wide
    as the ML-side vector -- which the base already guarantees (the paper's build-time assert).
    """

    _regressor_architecture = "flow"

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
        encoder_dropout: float = 0.3,
        latent_flow_layers: int = 6,
        latent_flow_hidden_features: int = 300,
        regressor_hidden_layers: int = 6,
        regressor_hidden_width: int = 300,
        regressor_dropout: float = 0.4,
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
            encoder_dropout=encoder_dropout,
            latent_flow_layers=latent_flow_layers,
            latent_flow_hidden_features=latent_flow_hidden_features,
            regressor_hidden_layers=regressor_hidden_layers,
            regressor_hidden_width=regressor_hidden_width,
            regressor_dropout=regressor_dropout,
            default_root_dir=default_root_dir,
        )
