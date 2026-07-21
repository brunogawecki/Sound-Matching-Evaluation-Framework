"""The InverSynth II networks: the encoder (``IS``) and the encoder + synthesizer-proxy (``IS2``).

Ports the strided-CNN encoder of ``paper_repos/InverSynth2/Python code/FM Synth/encdec.py``
(``Encoder``) and drives the mixed continuous+categorical head through this framework's
``ParameterSpace`` contract. The mel-dB front-end is reused from the preset-gen-vae port
(``models.presetgen_vae.network``): 257 mels, n_fft 1024, hop 256, normalized to [-1, 1],
which matches InverSynth II's DX7 spectrogram configuration almost exactly. Featurization
lives inside ``forward`` (D-REPR).

``InverSynthEncoderNetwork`` is the ``IS`` model: a spectrogram -> parameters regressor with
no synthesizer-proxy. :class:`InverSynthProxyNetwork` is the paper's differentiable neural
synthesizer-proxy (Decoder), a training-only component; :class:`IS2Network` pairs the two for
the ``IS2xITF`` / ``IS2`` stages. The proxy never touches evaluation -- ``IS2Network.forward``
is the encoder alone, so ``BaseDeepModel.predict`` re-renders with the real Dexed unchanged.

The reference hardcodes the flattened-feature width (``Linear(12288, ...)``) for the FM/TAL
input size. That is fragile across render lengths, so this port probes the conv stack with a
dummy render-length input at build time and sizes the head's ``Linear`` from the result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch import nn

# Reuse the preset-gen-vae mel-dB front-end (the deliberate cross-paper reuse the plan calls
# for). These helpers are engine-agnostic and paper-independent. ``_build_decoder_cnn`` /
# ``_center_crop_or_pad`` are that port's shape-robust mirror of this same CNN, so the Stage 2
# proxy reuses them rather than re-porting the reference Decoder's hardcoded reshape dims.
from models.presetgen_vae.network import (
    _build_decoder_cnn,
    _build_mel_filterbank,
    _center_crop_or_pad,
    _compute_mel_db_spectrogram,
)

# Fixed architecture constant, faithful to the reference encoder.
_NEGATIVE_SLOPE = 0.1  # LeakyReLU slope used throughout InverSynth's CNN


def _conv2d_block(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int],
    stride: Tuple[int, int],
    padding: int,
    use_batch_norm: bool,
) -> nn.Sequential:
    """One conv layer of the reference ``Encoder``: conv -> LeakyReLU -> (BN).

    Batch-norm is applied after the activation and omitted on the first conv layer, faithful
    to ``encdec.py::Encoder.enc_nn``.
    """
    block = nn.Sequential()
    block.add_module("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
    block.add_module("act", nn.LeakyReLU(_NEGATIVE_SLOPE))
    if use_batch_norm:
        block.add_module("bn", nn.BatchNorm2d(out_channels))
    return block


def _build_encoder_cnn() -> nn.Sequential:
    """The reference ``Encoder`` convolutional body (enc_nn + features_mixer_cnn).

    Six strided convolutions (1 -> 8 -> 16 -> 32 -> 64 -> 128 -> 256; first kernel 5x5, the rest
    4x4, all stride 2), then the feature mixer (256 -> 512 stride-2 4x4, then a 512 -> 2048 1x1
    conv). No batch-norm on the very first conv, per the reference. Ends at 2048 feature maps.
    """
    return nn.Sequential(
        _conv2d_block(1, 8, (5, 5), (2, 2), 2, use_batch_norm=False),
        _conv2d_block(8, 16, (4, 4), (2, 2), 2, use_batch_norm=True),
        _conv2d_block(16, 32, (4, 4), (2, 2), 2, use_batch_norm=True),
        _conv2d_block(32, 64, (4, 4), (2, 2), 2, use_batch_norm=True),
        _conv2d_block(64, 128, (4, 4), (2, 2), 2, use_batch_norm=True),
        _conv2d_block(128, 256, (4, 4), (2, 2), 2, use_batch_norm=True),
        # features_mixer_cnn
        _conv2d_block(256, 512, (4, 4), (2, 2), 2, use_batch_norm=True),
        _conv2d_block(512, 2048, (1, 1), (1, 1), 0, use_batch_norm=False),
    )


class InverSynthEncoderNetwork(nn.Module):
    """Raw audio ``[batch, num_samples]`` -> ML-side vector ``[batch, ml_dimension]``.

    A normalized mel-dB front-end feeds the reference's strided-CNN encoder; the head is a
    ``Dropout -> Linear -> BatchNorm1d`` MLP to ``ml_dimension`` raw outputs (continuous floats
    + categorical logits) that :class:`ParameterLoss` consumes. The final ``BatchNorm1d`` is
    ported faithfully from ``encdec.py::Encoder.mlp``.

    ``spectrogram_min_db`` / ``spectrogram_max_db`` are the corpus-measured [-1, 1] endpoints
    (D-MELNORM), passed in at build time so ``load`` rebuilds the identical front-end offline.
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
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if spectrogram_max_db <= spectrogram_min_db:
            raise ValueError("spectrogram_max_db must exceed spectrogram_min_db.")
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._win_length = win_length
        self._min_db = float(spectrogram_min_db)
        self._max_db = float(spectrogram_max_db)

        # Deterministic, non-persistent buffers (follow .to(device), out of state_dict).
        self.register_buffer("_window", torch.hann_window(win_length), persistent=False)
        mel_filterbank = _build_mel_filterbank(sample_rate, n_fft, n_mels, mel_fmin, mel_fmax)
        self.register_buffer("_mel_filterbank", mel_filterbank, persistent=False)

        self.encoder_cnn = _build_encoder_cnn()
        # Probe the deepest feature-map shape from a dummy render-length input, rather than the
        # reference's hardcoded 12288 (its FM/TAL input size). Keeping the full shape (not just
        # the flattened width) also lets the Stage 2 proxy decoder invert this exact feature map.
        self._cnn_output_shape, self._target_spectrogram_size = self._infer_cnn_output_shape(
            num_audio_samples
        )
        cnn_output_items = int(np.prod(self._cnn_output_shape))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(cnn_output_items, ml_dimension),
            nn.BatchNorm1d(ml_dimension),
        )

    @property
    def cnn_output_shape(self) -> Tuple[int, int, int]:
        """The encoder CNN's deepest feature-map shape ``(channels, height, width)``."""
        return self._cnn_output_shape

    @property
    def target_spectrogram_size(self) -> Tuple[int, int]:
        """The mel-dB spectrogram size ``(n_mels, frames)`` the proxy decoder reconstructs."""
        return self._target_spectrogram_size

    def mel_db_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """The normalized mel-dB spectrogram the encoder featurizes ``audio`` into.

        Public accessor for the param-free front-end. Stage 3 ITF computes the training
        pool's spectrograms once with this and then re-runs only the CNN + head (which do
        carry the fine-tuned weights) via :meth:`forward_from_spectrogram`.
        """
        return _compute_mel_db_spectrogram(
            audio, self._window, self._mel_filterbank,
            self._n_fft, self._hop_length, self._win_length, self._min_db, self._max_db,
        )

    def forward_from_spectrogram(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Prediction from a precomputed mel-dB spectrogram (skips the STFT front-end)."""
        features = self.encoder_cnn(spectrogram)
        return self.head(torch.flatten(features, start_dim=1))

    def _infer_cnn_output_shape(
        self, num_audio_samples: int
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int]]:
        with torch.no_grad():
            dummy_spectrogram = self.mel_db_spectrogram(torch.zeros(1, num_audio_samples))
            cnn_output = self.encoder_cnn(dummy_spectrogram)
        channels, height, width = cnn_output.shape[1:]
        target_size = (int(dummy_spectrogram.shape[-2]), int(dummy_spectrogram.shape[-1]))
        return (int(channels), int(height), int(width)), target_size

    def forward_with_spectrogram(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The prediction and the mel-dB spectrogram it was computed from.

        Stage 2's training step needs that input spectrogram as the proxy's reconstruction
        target, so it is returned here instead of being recomputed with a second STFT.
        """
        spectrogram = self.mel_db_spectrogram(audio)
        return self.forward_from_spectrogram(spectrogram), spectrogram

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        prediction, _ = self.forward_with_spectrogram(audio)
        return prediction


class InverSynthProxyNetwork(nn.Module):
    """The paper's differentiable neural synthesizer-proxy ``d_phi`` (``encdec.py::Decoder``).

    A transposed-CNN mapping the encoder's predicted ML-side vector back to a normalized mel-dB
    spectrogram, so an audio (spectrogram) loss can supply gradients to the encoder during
    training (Stage 2 onward). It is **training-only**: ``predict`` re-renders with the real
    Dexed, so the proxy never runs at evaluation.

    The reference ``Decoder`` hardcodes ``Linear(104, 24576)`` and ``view(-1, 2048, 3, 4)`` for
    its FM/TAL shape. This port instead reuses the preset-gen-vae port's shape-robust mirror
    decoder (``_build_decoder_cnn`` inverts the identical CNN topology), sizing the un-mixer
    ``Linear`` from the encoder's probed feature-map shape and cropping/padding the output to the
    exact ``(n_mels, frames)`` target, so the audio loss is well-defined at any render length.

    The proxy consumes the encoder's **raw** ML-side vector (continuous floats + categorical
    logits), not the paper's activated preset (Hardtanh + softmax) -- the same raw contract the
    encoder head emits for :class:`ParameterLoss`. The initial ``Linear`` absorbs the scale.
    """

    def __init__(
        self,
        ml_dimension: int,
        cnn_output_shape: Tuple[int, int, int],
        target_spectrogram_size: Tuple[int, int],
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self._cnn_output_shape = cnn_output_shape
        self._target_spectrogram_size = target_spectrogram_size
        cnn_output_items = int(np.prod(cnn_output_shape))
        # Reference Decoder.mlp: Linear -> Dropout (dropout after, unlike the encoder head).
        self.mlp = nn.Sequential(nn.Linear(ml_dimension, cnn_output_items), nn.Dropout(dropout))
        self.decoder_cnn = _build_decoder_cnn(unmixer_in_channels=cnn_output_shape[0])

    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        mixed_features = self.mlp(prediction).view(-1, *self._cnn_output_shape)
        reconstruction = self.decoder_cnn(mixed_features)
        return _center_crop_or_pad(reconstruction, *self._target_spectrogram_size)


@dataclass(frozen=True)
class IS2NetworkOutput:
    """Everything the Stage 2 training loss needs from one ``forward_training`` pass.

    ``proxy_spectrogram`` is the proxy's reconstruction of ``target_spectrogram`` (the encoder's
    mel-dB input); the audio loss is their MAE. ``prediction`` feeds :class:`ParameterLoss`.
    """

    prediction: torch.Tensor  # [batch, ml_dimension] raw floats + categorical logits
    proxy_spectrogram: torch.Tensor  # [batch, 1, n_mels, frames] in [-1, 1]
    target_spectrogram: torch.Tensor  # [batch, 1, n_mels, frames] in [-1, 1]


class IS2Network(nn.Module):
    """The encoder + training-only synthesizer-proxy of the ``IS2xITF`` / ``IS2`` stages.

    ``forward(audio)`` is the encoder alone (the whole eval path), so the inherited
    ``BaseDeepModel.predict`` is unchanged and the proxy is skipped at inference.
    ``forward_training(audio)`` additionally runs the proxy and returns everything the combined
    parameters + audio loss needs. Both networks are optimized by the single optimizer, jointly,
    as the paper trains them (its Eq. 4). The proxy is sized from the encoder's probed feature-map
    shape, so it inverts that exact map.
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
        dropout: float = 0.3,
        proxy_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = InverSynthEncoderNetwork(
            ml_dimension=ml_dimension,
            num_audio_samples=num_audio_samples,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            mel_fmin=mel_fmin,
            mel_fmax=mel_fmax,
            spectrogram_min_db=spectrogram_min_db,
            spectrogram_max_db=spectrogram_max_db,
            dropout=dropout,
        )
        self.proxy = InverSynthProxyNetwork(
            ml_dimension=ml_dimension,
            cnn_output_shape=self.encoder.cnn_output_shape,
            target_spectrogram_size=self.encoder.target_spectrogram_size,
            dropout=proxy_dropout,
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Eval path: the encoder only (the proxy is training-only)."""
        return self.encoder(audio)

    def forward_training(self, audio: torch.Tensor) -> IS2NetworkOutput:
        """Full pass for the training loss: prediction + proxy reconstruction + its target."""
        prediction, target_spectrogram = self.encoder.forward_with_spectrogram(audio)
        proxy_spectrogram = self.proxy(prediction)
        return IS2NetworkOutput(
            prediction=prediction,
            proxy_spectrogram=proxy_spectrogram,
            target_spectrogram=target_spectrogram,
        )
