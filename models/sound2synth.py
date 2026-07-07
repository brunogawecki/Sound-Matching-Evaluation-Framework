"""Sound2Synth-lineage models (Chen et al., 2022; github.com/Sound2Synth/Sound2Synth).

Discriminative spectrogram->parameters regressor (issue #19), the first real deep
family. A VGG11-BN conv net over a log-power STFT of the target audio, emitting the
ML-side parameter vector (continuous floats + categorical logits) through this
framework's ``ParameterSpace`` contract rather than Sound2Synth's binning scheme, so it
trains through the existing harness and ``ParameterLoss``.

``Sound2SynthSpectrogramNetwork`` is the plain ``nn.Module`` (STFT featurisation lives in
``forward``, per D-REPR); ``Sound2SynthSpectrogramRegressor`` trains it. Lightning is
imported lazily in ``fit`` so the eval path (load/predict) needs no training deps
(D-FRAMEWORK).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from dataset.torch_dataset import RenderedCorpusDataset
from models.base_deep_model import BaseDeepModel
from models.training.checkpoint import network_state_dict_from_lightning_checkpoint
from models.training.config import TrainingConfig
from models.training.loss import ParameterLoss

# Fixed architecture constants: LeakyReLU slope and the log-power floor.
_NEGATIVE_SLOPE = 0.01
_LOG_EPSILON = 1e-5


def _convolution_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """A 3x3 conv + BatchNorm + LeakyReLU block (Sound2Synth's ``CONV`` helper)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(_NEGATIVE_SLOPE),
    )


class Sound2SynthSpectrogramNetwork(nn.Module):
    """Raw audio ``[batch, num_samples]`` -> ML-side vector ``[batch, ml_dimension]``.

    A log-power STFT front-end feeds a VGG11-BN backbone matching Sound2Synth's
    ``ConvBackbone``. Only the head is ours: an MLP to ``ml_dimension`` raw outputs
    (continuous floats + categorical logits) that :class:`ParameterLoss` consumes.
    """

    def __init__(
        self,
        ml_dimension: int,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        head_hidden_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._win_length = win_length
        # Non-persistent: deterministic from win_length, follows .to(device), kept out of state_dict.
        self.register_buffer("_window", torch.hann_window(win_length), persistent=False)

        # VGG11-BN backbone matching Sound2Synth's ConvBackbone: channels
        # 1->64->128->256->256->512->512->512->512, max-pool after blocks 1, 2, 4, 6 (no 5th),
        # then AdaptiveMaxPool2d((2,2)) -> fixed 512x2x2 = 2048 regardless of spectrogram size.
        self.convolutional_backbone = nn.Sequential(
            _convolution_block(1, 64),
            nn.MaxPool2d(2, 2),
            _convolution_block(64, 128),
            nn.MaxPool2d(2, 2),
            _convolution_block(128, 256),
            _convolution_block(256, 256),
            nn.MaxPool2d(2, 2),
            _convolution_block(256, 512),
            _convolution_block(512, 512),
            nn.MaxPool2d(2, 2),
            _convolution_block(512, 512),
            _convolution_block(512, 512),
            nn.AdaptiveMaxPool2d((2, 2)),
        )
        self.embedding = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.LeakyReLU(_NEGATIVE_SLOPE),
        )
        # Our head, not the paper's grouped-FC classifier: a plain MLP to the flat ML-side vector.
        self.head = nn.Sequential(
            nn.Linear(2048, head_hidden_dim),
            nn.LeakyReLU(_NEGATIVE_SLOPE),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, ml_dimension),
        )

    def _log_power_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """``log(|STFT|^2 + eps)`` as ``[batch, 1, frequency, frames]``."""
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
        power = complex_stft.abs() ** 2  # [batch, frequency, frames]
        log_power = torch.log(power + _LOG_EPSILON)
        return log_power.unsqueeze(1)  # add the single input channel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrogram = self._log_power_spectrogram(x)
        features = self.convolutional_backbone(spectrogram)
        flattened = torch.flatten(features, start_dim=1)  # [batch, 2048]
        embedding = self.embedding(flattened)
        return self.head(embedding)


class Sound2SynthSpectrogramRegressor(BaseDeepModel):
    """The :class:`BaseDeepModel` family wrapping :class:`Sound2SynthSpectrogramNetwork`.

    Only ``_build_network`` and ``fit`` are family-specific (save/load/predict are inherited).
    The spectrogram + architecture knobs flow into ``architecture_hparams`` at ``fit`` time so
    ``load`` can rebuild the exact network before restoring weights (no VST, no Lightning).
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        head_hidden_dim: int = 512,
        dropout: float = 0.3,
        default_root_dir: str = "lightning_logs",
    ) -> None:
        super().__init__()
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._win_length = win_length
        self._head_hidden_dim = head_hidden_dim
        self._dropout = dropout
        self._default_root_dir = default_root_dir

    def _build_network(self, architecture_hparams: Dict[str, Any]) -> nn.Module:
        return Sound2SynthSpectrogramNetwork(
            ml_dimension=architecture_hparams["ml_dimension"],
            n_fft=architecture_hparams["n_fft"],
            hop_length=architecture_hparams["hop_length"],
            win_length=architecture_hparams["win_length"],
            head_hidden_dim=architecture_hparams["head_hidden_dim"],
            dropout=architecture_hparams["dropout"],
        )

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
        architecture = type(self).__name__
        dataset = train_dataset.corpus_dir.name
        pl.seed_everything(training_config.seed, workers=True)

        parameter_space = train_dataset.parameter_space
        architecture_hparams = {
            "ml_dimension": parameter_space.ml_dimension,
            "n_fft": self._n_fft,
            "hop_length": self._hop_length,
            "win_length": self._win_length,
            "head_hidden_dim": self._head_hidden_dim,
            "dropout": self._dropout,
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

        run_metadata = {"architecture": architecture, "dataset": dataset}
        for logger in trainer.loggers:
            logger.log_hyperparams({**run_metadata, **training_config.to_dict()})
        trainer.fit(lightning_regressor, datamodule=data_module)

        # Load the best checkpoint's weights, then register the trained network.
        best_path = trainer.checkpoint_callback.best_model_path
        if best_path:
            network.load_state_dict(network_state_dict_from_lightning_checkpoint(best_path))
        self._set_trained_network(network, architecture_hparams, parameter_space)
