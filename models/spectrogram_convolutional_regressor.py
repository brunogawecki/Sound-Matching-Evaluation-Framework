"""Discriminative spectrogram->parameters regressor (issue #19, Sound2Synth lineage).

The first real deep family: a VGG11-BN convolutional network over a log-power STFT
of the target audio, predicting the ML-side parameter vector. It recreates the STFT
``ConvBackbone`` branch of Sound2Synth (Chen et al., 2022;
``github.com/Sound2Synth/Sound2Synth``) -- the ``main`` branch only, deliberately the
lowest-risk architecture -- but emits through this framework's own ``ParameterSpace``
contract (regression for continuous params, logits for categorical blocks) rather than
Sound2Synth's quantise-everything-into-bins scheme, so it plugs straight into the
existing training harness and ``ParameterLoss``.

Featurisation lives inside ``forward`` (D-REPR: the dataset yields raw waveforms). The
spectrogram is computed with ``torch.stft`` -- differentiable, device-aware, and part
of core ``torch`` (no ``torchaudio`` dependency). ``SpectrogramConvolutionalNetwork``
is the plain ``nn.Module``; ``SpectrogramConvolutionalRegressor`` is the
:class:`BaseDeepModel` family that trains it through the harness. The ``fit`` wiring
mirrors ``tests/tiny_deep_model.py`` (the harness's reference wiring).

Lightning (and the harness modules that import it) is imported lazily inside ``fit`` so
this module -- and hence the eval-path ``load``/``predict`` inherited from
:class:`BaseDeepModel` -- stays importable without the training-only dependency
(D-FRAMEWORK); ``models/__init__`` imports this module eagerly.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn

from dataset.torch_dataset import RenderedCorpusDataset
from models.base_deep_model import BaseDeepModel
from models.training.checkpoint import network_state_dict_from_lightning_checkpoint
from models.training.config import TrainingConfig
from models.training.loss import ParameterLoss

# A conv-stack schedule (torchvision-VGG style): each int is a conv layer's output
# channel count, "M" is a 2x2 max-pool. This is the VGG11 schedule Sound2Synth's
# ``ConvBackbone`` uses -- 8 conv layers, 5 pools, channels 1->64->...->512.
ConvConfig = List[Union[int, str]]
DEFAULT_CONV_CONFIG: ConvConfig = [
    64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M",
]

# Fixed architecture constants (not tuned per-instance): the adaptive-pool output is
# 2x2 (so the flattened width is last_channels * 4, independent of spectrogram size),
# LeakyReLU's slope, and the log-power floor. Sound2Synth used ``power=2.0`` then a log.
_ADAPTIVE_POOL_SIZE = 2
_NEGATIVE_SLOPE = 0.01
_LOG_EPSILON = 1e-5


class SpectrogramConvolutionalNetwork(nn.Module):
    """Raw audio ``[batch, num_samples]`` -> ML-side vector ``[batch, ml_dimension]``.

    A log-power STFT front-end feeds a VGG11-BN conv stack; adaptive max-pooling to a
    fixed ``2x2`` grid makes the fully-connected sizes independent of the spectrogram
    dimensions. The output is raw (no softmax/sigmoid): continuous slots are raw floats
    and categorical blocks are logits, exactly what :class:`ParameterLoss` and
    ``ParameterSpace.ml_vector_to_synth_dict`` consume.
    """

    def __init__(
        self,
        ml_dimension: int,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        conv_config: Optional[ConvConfig] = None,
        embedding_dim: int = 512,
        head_hidden_dim: int = 512,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._win_length = win_length
        # Non-persistent: deterministic from win_length, rebuilt in __init__, and it
        # follows the module across ``.to(device)`` (kept out of the saved state_dict).
        self.register_buffer("_window", torch.hann_window(win_length), persistent=False)

        conv_config = list(conv_config) if conv_config is not None else list(DEFAULT_CONV_CONFIG)
        self.features = self._build_conv_stack(conv_config)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((_ADAPTIVE_POOL_SIZE, _ADAPTIVE_POOL_SIZE))

        last_channels = [item for item in conv_config if isinstance(item, int)][-1]
        flattened_dim = last_channels * _ADAPTIVE_POOL_SIZE * _ADAPTIVE_POOL_SIZE
        self.embedding = nn.Sequential(
            nn.Linear(flattened_dim, embedding_dim),
            nn.LeakyReLU(_NEGATIVE_SLOPE),
        )
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, head_hidden_dim),
            nn.LeakyReLU(_NEGATIVE_SLOPE),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, ml_dimension),
        )

    @staticmethod
    def _build_conv_stack(conv_config: ConvConfig) -> nn.Sequential:
        """Build the VGG-style Conv2d+BN+LeakyReLU / MaxPool stack from a schedule."""
        layers: List[nn.Module] = []
        in_channels = 1
        for item in conv_config:
            if item == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                out_channels = int(item)
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.LeakyReLU(_NEGATIVE_SLOPE))
                in_channels = out_channels
        return nn.Sequential(*layers)

    def _log_power_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """``log(|STFT|^2 + eps)`` as ``[batch, 1, frequency, frames]``."""
        # ``.float()``: torch.stft needs float32/64 -- under bf16-mixed autocast the
        # input would otherwise arrive as bf16 and error. The conv stack downstream is
        # free to autocast normally.
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

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        spectrogram = self._log_power_spectrogram(audio)
        features = self.features(spectrogram)
        pooled = self.adaptive_pool(features)
        flattened = torch.flatten(pooled, start_dim=1)
        embedding = self.embedding(flattened)
        return self.head(embedding)


class SpectrogramConvolutionalRegressor(BaseDeepModel):
    """The :class:`BaseDeepModel` family wrapping :class:`SpectrogramConvolutionalNetwork`.

    Only ``_build_network`` and ``fit`` are family-specific; ``save``/``load``/``predict``
    are inherited. The spectrogram + architecture knobs are stored on the instance and
    flow into ``architecture_hparams`` at ``fit`` time, so ``load`` can rebuild the exact
    network before restoring weights (no VST, no Lightning).
    """

    def __init__(
        self,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        conv_config: Optional[ConvConfig] = None,
        embedding_dim: int = 512,
        head_hidden_dim: int = 512,
        dropout: float = 0.3,
        default_root_dir: str = "lightning_logs",
    ) -> None:
        super().__init__()
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._win_length = win_length
        self._conv_config = list(conv_config) if conv_config is not None else list(DEFAULT_CONV_CONFIG)
        self._embedding_dim = embedding_dim
        self._head_hidden_dim = head_hidden_dim
        self._dropout = dropout
        self._default_root_dir = default_root_dir

    def _build_network(self, architecture_hparams: Dict[str, Any]) -> nn.Module:
        return SpectrogramConvolutionalNetwork(
            ml_dimension=architecture_hparams["ml_dimension"],
            n_fft=architecture_hparams["n_fft"],
            hop_length=architecture_hparams["hop_length"],
            win_length=architecture_hparams["win_length"],
            conv_config=architecture_hparams["conv_config"],
            embedding_dim=architecture_hparams["embedding_dim"],
            head_hidden_dim=architecture_hparams["head_hidden_dim"],
            dropout=architecture_hparams["dropout"],
        )

    def fit(
        self,
        train_dataset: RenderedCorpusDataset,
        validation_dataset: Optional[RenderedCorpusDataset] = None,
        config: Optional[Dict[str, object]] = None,
    ) -> None:
        # Lazy: the training-only Lightning stack (D-FRAMEWORK) is imported here so the
        # module stays importable on the eval path (load/predict need no Lightning).
        import lightning.pytorch as pl

        from models.training.data_module import CorpusDataModule
        from models.training.lightning_module import LightningRegressor
        from models.training.trainer_factory import build_trainer

        training_config = TrainingConfig.from_dict(config)
        pl.seed_everything(training_config.seed, workers=True)

        parameter_space = train_dataset.parameter_space
        architecture_hparams = {
            "ml_dimension": parameter_space.ml_dimension,
            "n_fft": self._n_fft,
            "hop_length": self._hop_length,
            "win_length": self._win_length,
            "conv_config": list(self._conv_config),
            "embedding_dim": self._embedding_dim,
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
        trainer.fit(lightning_regressor, datamodule=data_module)

        # Export step: load the best .ckpt's network weights, then register the network.
        best_path = trainer.checkpoint_callback.best_model_path
        if best_path:
            network.load_state_dict(network_state_dict_from_lightning_checkpoint(best_path))
        self._set_trained_network(network, architecture_hparams, parameter_space)
