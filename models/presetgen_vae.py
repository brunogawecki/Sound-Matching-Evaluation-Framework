"""preset-gen-vae port (Le Vaillant et al., DAFx 2021; paper_repos/preset-gen-vae).

A staged reimplementation of the paper's ``FlVAE2`` model as a ``BaseDeepModel``
family that predicts this framework's D1 parameter space through ``ParameterSpace``
(not the paper's 144-param / ``all<=32`` scheme), so it stays comparable to the other
model families and trains through the existing harness and ``ParameterLoss``.

**Stage 1 (this module):** the audio->params path only -- a faithful mel-dB front-end
and the paper's ``speccnn8l1_bn`` spectrogram encoder, feeding a deterministic latent
and an MLP regressor head. This is a plain regressor (no autoencoding), the baseline the
VAE must beat. Stage 2 adds the spectrogram decoder + reconstruction/KL loss; Stage 3
adds the latent RealNVP flow. The network is deliberately structured
(front-end -> CNN -> latent -> regressor) so those stages slot in without reshaping it.

The head emits **raw** outputs (continuous floats + categorical logits), matching the
``ParameterSpace`` / ``ParameterLoss`` contract -- the paper's ``PresetActivation``
(Hardtanh + softmax) is intentionally dropped, exactly as ``Sound2SynthSpectrogramNetwork``
does. ``ParameterLoss`` applies softmax/cross-entropy to categorical blocks itself.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn

from dataset.torch_dataset import RenderedCorpusDataset
from models.base_deep_model import BaseDeepModel
from models.training.checkpoint import network_state_dict_from_lightning_checkpoint
from models.training.config import TrainingConfig
from models.training.loss import ParameterLoss

# Fixed architecture constants, faithful to the paper's speccnn8l1_bn encoder.
_NEGATIVE_SLOPE = 0.1  # LeakyReLU slope used throughout the paper's CNN
_LOG_EPSILON = 1e-7  # floor inside log10 before dB conversion


def _conv2d_block(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: int,
    use_batch_norm: bool,
) -> nn.Sequential:
    """One conv layer of the paper's ``layer.Conv2D``: conv -> LeakyReLU -> (BN).

    Batch-norm is applied *after* the activation (the paper's ``batch_norm='after'``),
    and omitted on the first and last conv layers (``batch_norm=None`` there).
    """
    block = nn.Sequential()
    block.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
    block.add_module("act", nn.LeakyReLU(_NEGATIVE_SLOPE))
    if use_batch_norm:
        block.add_module("bn", nn.BatchNorm2d(out_channels))
    return block


def _build_spectrogram_cnn() -> nn.Sequential:
    """The paper's ``speccnn8l1_bn`` single-channel encoder CNN (enc1..enc8).

    Eight strided convolutions taking a 1-channel spectrogram to 1024 feature maps.
    No batch-norm on the first (enc1) and last (enc8) layers, per the paper.
    """
    return nn.Sequential(
        _conv2d_block(1, 8, (5, 5), (2, 2), 2, use_batch_norm=False),  # enc1
        _conv2d_block(8, 16, (4, 4), (2, 2), 2, use_batch_norm=True),  # enc2
        _conv2d_block(16, 32, (4, 4), (2, 2), 2, use_batch_norm=True),  # enc3
        _conv2d_block(32, 64, (4, 4), (2, 2), 2, use_batch_norm=True),  # enc4
        _conv2d_block(64, 128, (4, 4), (2, 2), 2, use_batch_norm=True),  # enc5
        _conv2d_block(128, 256, (4, 4), (2, 2), 2, use_batch_norm=True),  # enc6
        _conv2d_block(256, 512, (4, 4), (2, 2), 2, use_batch_norm=True),  # enc7 (4x4conv)
        _conv2d_block(512, 1024, (1, 1), (1, 1), 0, use_batch_norm=False),  # enc8 (1x1conv)
    )


def _build_regressor(
    dim_z: int,
    ml_dimension: int,
    hidden_layers: int,
    hidden_width: int,
    dropout: float,
) -> nn.Sequential:
    """The paper's ``MLPRegression`` (e.g. ``3l1024``) minus its ``PresetActivation``.

    ``hidden_layers`` fully-connected layers of ``hidden_width``, each with ReLU; the
    first ``hidden_layers - 1`` also carry BatchNorm + Dropout (the paper omits both on
    the two deepest layers). Ends in a plain ``Linear`` to ``ml_dimension`` raw outputs.
    """
    if hidden_layers < 1:
        raise ValueError(f"regressor hidden_layers must be >= 1, got {hidden_layers}.")
    model = nn.Sequential()
    for layer_index in range(hidden_layers):
        in_features = dim_z if layer_index == 0 else hidden_width
        model.add_module(f"fc{layer_index + 1}", nn.Linear(in_features, hidden_width))
        if layer_index < hidden_layers - 1:
            model.add_module(f"bn{layer_index + 1}", nn.BatchNorm1d(hidden_width))
            model.add_module(f"drp{layer_index + 1}", nn.Dropout(dropout))
        model.add_module(f"act{layer_index + 1}", nn.ReLU())
    model.add_module(f"fc{hidden_layers + 1}", nn.Linear(hidden_width, ml_dimension))
    return model


class PresetGenVaeNetwork(nn.Module):
    """Raw audio ``[batch, num_samples]`` -> ML-side vector ``[batch, ml_dimension]``.

    A mel-dB front-end (STFT -> mel filterbank -> dB -> min-max to [-1, 1]) feeds the
    paper's ``speccnn8l1_bn`` CNN, a latent projection, and an MLP regressor. The CNN's
    flattened output size is inferred once at construction from ``num_audio_samples``
    (the render length), so the network is fully determined by its hparams and rebuilds
    identically for checkpoint loading.
    """

    def __init__(
        self,
        ml_dimension: int,
        num_audio_samples: int,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 257,
        mel_fmin: float = 30.0,
        mel_fmax: float = 11000.0,
        spectrogram_min_db: float = -120.0,
        spectrogram_max_db: float = 0.0,
        dim_z: int = 256,
        encoder_dropout: float = 0.3,
        regressor_hidden_layers: int = 3,
        regressor_hidden_width: int = 1024,
        regressor_dropout: float = 0.4,
    ) -> None:
        super().__init__()
        if spectrogram_max_db <= spectrogram_min_db:
            raise ValueError("spectrogram_max_db must exceed spectrogram_min_db.")
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._win_length = win_length
        self._n_mels = n_mels
        self._min_db = float(spectrogram_min_db)
        self._max_db = float(spectrogram_max_db)

        # Deterministic, non-persistent buffers (follow .to(device), out of state_dict).
        self.register_buffer("_window", torch.hann_window(win_length), persistent=False)
        mel_filterbank = self._build_mel_filterbank(sample_rate, n_fft, n_mels, mel_fmin, mel_fmax)
        self.register_buffer("_mel_filterbank", mel_filterbank, persistent=False)

        self.spectrogram_cnn = _build_spectrogram_cnn()
        # Infer the flattened CNN output size from a dummy render-length input.
        cnn_output_items = self._infer_cnn_output_items(num_audio_samples)
        # Deterministic latent projection (Stage 2 replaces dim_z -> 2*dim_z + reparam).
        self.encoder_mlp = nn.Sequential(
            nn.Dropout(encoder_dropout), nn.Linear(cnn_output_items, dim_z)
        )
        self.regressor = _build_regressor(
            dim_z, ml_dimension, regressor_hidden_layers, regressor_hidden_width, regressor_dropout
        )

    @staticmethod
    def _build_mel_filterbank(
        sample_rate: int, n_fft: int, n_mels: int, fmin: float, fmax: float
    ) -> torch.Tensor:
        """A ``[n_mels, 1 + n_fft // 2]`` mel filterbank (librosa, un-normalized)."""
        import librosa  # heavy import kept off module load

        filterbank = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, norm=None
        )
        return torch.from_numpy(np.asarray(filterbank, dtype=np.float32))

    def _mel_db_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Audio ``[batch, num_samples]`` -> normalized mel-dB ``[batch, 1, n_mels, frames]``."""
        # torch.stft needs float32/64; under bf16-mixed autocast the input arrives bf16.
        audio = audio.float()
        complex_stft = torch.stft(
            audio,
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            win_length=self._win_length,
            window=self._window,
            center=True,
            pad_mode="reflect",
            return_complex=True,
        )
        magnitude = complex_stft.abs()  # [batch, freq, frames]
        mel_magnitude = torch.matmul(self._mel_filterbank, magnitude)  # [batch, n_mels, frames]
        decibels = 20.0 * torch.log10(mel_magnitude + _LOG_EPSILON)
        decibels = torch.clamp(decibels, min=self._min_db, max=self._max_db)
        # Min-max to [-1, 1] over the fixed dB range (Stage 2 may swap in corpus stats).
        normalized = 2.0 * (decibels - self._min_db) / (self._max_db - self._min_db) - 1.0
        return normalized.unsqueeze(1)  # add the single input channel

    def _infer_cnn_output_items(self, num_audio_samples: int) -> int:
        with torch.no_grad():
            dummy_audio = torch.zeros(1, num_audio_samples)
            dummy_spectrogram = self._mel_db_spectrogram(dummy_audio)
            cnn_output = self.spectrogram_cnn(dummy_spectrogram)
        return int(cnn_output.numel())

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        spectrogram = self._mel_db_spectrogram(audio)
        features = self.spectrogram_cnn(spectrogram)
        flattened = torch.flatten(features, start_dim=1)
        latent = self.encoder_mlp(flattened)
        return self.regressor(latent)


class PresetGenVaeRegressor(BaseDeepModel):
    """The :class:`BaseDeepModel` family wrapping :class:`PresetGenVaeNetwork` (Stage 1).

    Only ``_build_network`` and ``fit`` are family-specific (save/load/predict are inherited).
    The mel/STFT/encoder/regressor knobs are constructor arguments (paper ``speccnn8l1_bn`` +
    ``3l1024`` defaults); ``ml_dimension``, ``num_audio_samples`` and ``sample_rate`` are read
    from the corpus at ``fit`` time and folded into ``architecture_hparams`` so ``load`` can
    rebuild the exact network before restoring weights (no VST, no Lightning). Reading the
    render length + sample rate from the corpus (not the constructor) keeps the network aligned
    with the self-describing corpus (D-SELFDESC) instead of a possibly-mismatched fixed value.

    Stage 1 trains through the existing :class:`LightningRegressor` + :class:`ParameterLoss`,
    exactly like :class:`Sound2SynthSpectrogramRegressor`; the VAE reconstruction/KL and the
    latent flow arrive in Stages 2 and 3.
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
        dim_z: int = 256,
        encoder_dropout: float = 0.3,
        regressor_hidden_layers: int = 3,
        regressor_hidden_width: int = 1024,
        regressor_dropout: float = 0.4,
        default_root_dir: str = "lightning_logs",
    ) -> None:
        super().__init__()
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._win_length = win_length
        self._n_mels = n_mels
        self._mel_fmin = mel_fmin
        self._mel_fmax = mel_fmax
        self._spectrogram_min_db = spectrogram_min_db
        self._spectrogram_max_db = spectrogram_max_db
        self._dim_z = dim_z
        self._encoder_dropout = encoder_dropout
        self._regressor_hidden_layers = regressor_hidden_layers
        self._regressor_hidden_width = regressor_hidden_width
        self._regressor_dropout = regressor_dropout
        self._default_root_dir = default_root_dir

    def _build_network(self, architecture_hparams: Dict[str, Any]) -> nn.Module:
        return PresetGenVaeNetwork(
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
            dim_z=architecture_hparams["dim_z"],
            encoder_dropout=architecture_hparams["encoder_dropout"],
            regressor_hidden_layers=architecture_hparams["regressor_hidden_layers"],
            regressor_hidden_width=architecture_hparams["regressor_hidden_width"],
            regressor_dropout=architecture_hparams["regressor_dropout"],
        )

    @staticmethod
    def _corpus_sample_rate(train_dataset: RenderedCorpusDataset) -> int:
        """Read the render sample rate from the corpus's ``run_summary.json``."""
        with open(train_dataset.corpus_dir / "run_summary.json") as summary_file:
            return int(json.load(summary_file)["sample_rate"])

    def fit(
        self,
        train_dataset: RenderedCorpusDataset,
        validation_dataset: Optional[RenderedCorpusDataset] = None,
        config: Optional[Dict[str, object]] = None,
    ) -> None:
        # Lazy: the training-only Lightning stack (D-FRAMEWORK) stays off the eval path.
        import lightning.pytorch as pl

        from models.training.data_module import CorpusDataModule
        from models.training.lightning_module import LightningRegressor
        from models.training.trainer_factory import build_trainer

        training_config = TrainingConfig.from_dict(config)
        pl.seed_everything(training_config.seed, workers=True)

        parameter_space = train_dataset.parameter_space
        # Render length + sample rate come from the corpus, not the constructor (D-SELFDESC).
        example_audio, _ = train_dataset[0]
        architecture_hparams = {
            "ml_dimension": parameter_space.ml_dimension,
            "num_audio_samples": int(example_audio.shape[-1]),
            "sample_rate": self._corpus_sample_rate(train_dataset),
            "n_fft": self._n_fft,
            "hop_length": self._hop_length,
            "win_length": self._win_length,
            "n_mels": self._n_mels,
            "mel_fmin": self._mel_fmin,
            "mel_fmax": self._mel_fmax,
            "spectrogram_min_db": self._spectrogram_min_db,
            "spectrogram_max_db": self._spectrogram_max_db,
            "dim_z": self._dim_z,
            "encoder_dropout": self._encoder_dropout,
            "regressor_hidden_layers": self._regressor_hidden_layers,
            "regressor_hidden_width": self._regressor_hidden_width,
            "regressor_dropout": self._regressor_dropout,
        }
        network = self._build_network(architecture_hparams)
        parameter_loss = ParameterLoss(parameter_space, training_config.loss)
        lightning_regressor = LightningRegressor(network, parameter_loss, training_config.optimizer)
        data_module = CorpusDataModule(
            train_dataset, validation_dataset, training_config.data, seed=training_config.seed
        )

        will_validate = data_module.will_validate
        monitor = "val_loss" if will_validate else "train_loss"
        trainer = build_trainer(
            training_config,
            default_root_dir=self._default_root_dir,
            monitor=monitor,
            run_validation=will_validate,
        )

        run_metadata = {
            "architecture": type(self).__name__,
            "dataset": train_dataset.corpus_dir.name,
        }
        for logger in trainer.loggers:
            logger.log_hyperparams({**run_metadata, **training_config.to_dict()})
        trainer.fit(lightning_regressor, datamodule=data_module)

        # Load the best checkpoint's weights, then register the trained network.
        best_path = trainer.checkpoint_callback.best_model_path
        if best_path:
            network.load_state_dict(network_state_dict_from_lightning_checkpoint(best_path))
        self._set_trained_network(network, architecture_hparams, parameter_space)
