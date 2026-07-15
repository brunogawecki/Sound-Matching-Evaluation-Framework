"""The InverSynth II encoder network (Stage 1: the ``IS`` model).

Ports the strided-CNN encoder of ``paper_repos/InverSynth2/Python code/FM Synth/encdec.py``
(``Encoder``) and drives the mixed continuous+categorical head through this framework's
``ParameterSpace`` contract. The mel-dB front-end is reused from the preset-gen-vae port
(``models.presetgen_vae.network``): 257 mels, n_fft 1024, hop 256, normalized to [-1, 1],
which matches InverSynth II's DX7 spectrogram configuration almost exactly. Featurization
lives inside ``forward`` (D-REPR).

``InverSynthEncoderNetwork`` is the ``IS`` model: a spectrogram -> parameters regressor with
no synthesizer-proxy. Stage 2 (``IS2xITF``) adds the proxy decoder and the audio loss; the
proxy is a training-only component, so the encoder here stays the whole eval path.

The reference hardcodes the flattened-feature width (``Linear(12288, ...)``) for the FM/TAL
input size. That is fragile across render lengths, so this port probes the conv stack with a
dummy render-length input at build time and sizes the head's ``Linear`` from the result.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import nn

# Reuse the preset-gen-vae mel-dB front-end (the deliberate cross-paper reuse the plan calls
# for). These helpers are engine-agnostic and paper-independent.
from models.presetgen_vae.network import _build_mel_filterbank, _compute_mel_db_spectrogram

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
        # Probe the flattened feature width from a dummy render-length input, rather than the
        # reference's hardcoded 12288 (its FM/TAL input size).
        cnn_output_items = self._infer_cnn_output_items(num_audio_samples)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(cnn_output_items, ml_dimension),
            nn.BatchNorm1d(ml_dimension),
        )

    def _mel_db_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        return _compute_mel_db_spectrogram(
            audio, self._window, self._mel_filterbank,
            self._n_fft, self._hop_length, self._win_length, self._min_db, self._max_db,
        )

    def _infer_cnn_output_items(self, num_audio_samples: int) -> int:
        with torch.no_grad():
            dummy_spectrogram = self._mel_db_spectrogram(torch.zeros(1, num_audio_samples))
            cnn_output = self.encoder_cnn(dummy_spectrogram)
        return int(np.prod(cnn_output.shape[1:]))

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        spectrogram = self._mel_db_spectrogram(audio)
        features = self.encoder_cnn(spectrogram)
        flattened = torch.flatten(features, start_dim=1)
        return self.head(flattened)
