"""The SynthRL network: mel front-end + strided conv + transformer encoder + DETR decoder.

Ports the architecture of SynthRL (Shin & Lee, IJCAI-25, §3.2): a mel-spectrogram
is reduced by ``num_conv_layers`` strided 2D convolutions, flattened into a token
sequence with a 2D sinusoidal positional encoding, and passed through
``num_encoder_layers`` transformer encoder layers to a feature map ``z``. A DETR-style
decoder holds one learnable **query per parameter**; ``num_decoder_layers`` transformer
decoder layers (self-attention over the queries, cross-attention onto ``z``) turn the
queries into per-parameter vectors, and a per-parameter projection head emits that
parameter's class logits. ``forward`` returns the flat class-logit vector laid out by
:class:`SynthRLRepresentation` (``[batch, total_class_dimension]``).

The mel-dB front-end is reused from the preset-gen-vae port (the same one InverSynth II
uses): 257 mels, n_fft 1024, hop 256, min-max normalized to [-1, 1]. Featurization lives
inside ``forward`` (D-REPR). The conv stack's output spatial size is probed from a dummy
render-length input at build time, so the positional encoding fits any render length.

The strided-conv reducer, the DETR cross-attention decoder, and the per-parameter class
heads are the genuinely new pieces; the mel front-end and transformer layers are
reused / standard. The network emits **class logits**, not the framework's ML-side
vector, so the SynthRL families decode through the representation, not
``ml_vector_to_synth_dict`` (see ``families.py``).
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
from torch import nn

# Reuse the preset-gen-vae mel-dB front-end (the same cross-paper reuse InverSynth II makes).
from models.presetgen_vae.network import _build_mel_filterbank, _compute_mel_db_spectrogram

_NEGATIVE_SLOPE = 0.1  # LeakyReLU slope for the conv reducer


def _build_2d_sincos_positional_encoding(d_model: int, height: int, width: int) -> torch.Tensor:
    """Fixed DETR-style 2D sinusoidal positional encoding ``[d_model, height, width]``.

    Half the channels encode the row (mel) axis, half the column (time) axis, each a
    standard sine/cosine positional encoding. ``d_model`` must be divisible by 4.
    """
    if d_model % 4 != 0:
        raise ValueError(f"d_model must be divisible by 4 for 2D positional encoding, got {d_model}.")
    half = d_model // 2
    frequency = torch.exp(
        torch.arange(0, half, 2, dtype=torch.float32) * (-math.log(10000.0) / half)
    )
    encoding = torch.zeros(d_model, height, width)
    row = torch.arange(height, dtype=torch.float32).unsqueeze(1)  # [height, 1]
    column = torch.arange(width, dtype=torch.float32).unsqueeze(1)  # [width, 1]
    row_angles = row * frequency  # [height, half/2]
    column_angles = column * frequency  # [width, half/2]
    encoding[0:half:2] = torch.sin(row_angles).t().unsqueeze(-1).expand(-1, height, width)
    encoding[1:half:2] = torch.cos(row_angles).t().unsqueeze(-1).expand(-1, height, width)
    encoding[half::2] = torch.sin(column_angles).t().unsqueeze(-2).expand(-1, height, width)
    encoding[half + 1 :: 2] = torch.cos(column_angles).t().unsqueeze(-2).expand(-1, height, width)
    return encoding


def _build_conv_reducer(num_conv_layers: int, d_model: int) -> nn.Sequential:
    """A stack of ``num_conv_layers`` stride-2 conv blocks, then a 1x1 projection to ``d_model``.

    Each block halves both spatial dimensions (3x3, stride 2, pad 1) and applies
    LeakyReLU + batch-norm (batch-norm omitted on the first block, following the peer
    CNN front-ends). Channels ramp 1 -> 32 -> 64 -> ... (doubling, capped at 256); a
    final 1x1 conv projects to ``d_model`` so the tokens carry the model dimension.
    """
    if num_conv_layers < 1:
        raise ValueError(f"num_conv_layers must be >= 1, got {num_conv_layers}.")
    layers = nn.Sequential()
    in_channels = 1
    out_channels = 32
    for index in range(num_conv_layers):
        block = nn.Sequential()
        block.add_module("conv", nn.Conv2d(in_channels, out_channels, (3, 3), (2, 2), 1))
        block.add_module("act", nn.LeakyReLU(_NEGATIVE_SLOPE))
        if index > 0:
            block.add_module("bn", nn.BatchNorm2d(out_channels))
        layers.add_module(f"conv_block_{index}", block)
        in_channels = out_channels
        out_channels = min(out_channels * 2, 256)
    layers.add_module("project", nn.Conv2d(in_channels, d_model, (1, 1), (1, 1), 0))
    return layers


class SynthRLNetwork(nn.Module):
    """Raw audio ``[batch, num_samples]`` -> flat class logits ``[batch, total_class_dimension]``.

    ``class_counts`` is the per-parameter head widths from :class:`SynthRLRepresentation`
    (subset order); the emitted logits concatenate the heads in that order, matching the
    representation's ``class_slices``. ``spectrogram_min_db`` / ``spectrogram_max_db`` are
    the corpus-measured [-1, 1] endpoints (D-MELNORM), passed at build time so ``load``
    rebuilds the identical front-end offline.
    """

    def __init__(
        self,
        class_counts: List[int],
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
        d_model: int = 256,
        num_conv_layers: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        feedforward_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if spectrogram_max_db <= spectrogram_min_db:
            raise ValueError("spectrogram_max_db must exceed spectrogram_min_db.")
        self._class_counts = list(class_counts)
        self._n_fft = n_fft
        self._hop_length = hop_length
        self._win_length = win_length
        self._min_db = float(spectrogram_min_db)
        self._max_db = float(spectrogram_max_db)

        # Deterministic, non-persistent front-end buffers (follow .to(device), out of state_dict).
        self.register_buffer("_window", torch.hann_window(win_length), persistent=False)
        self.register_buffer(
            "_mel_filterbank",
            _build_mel_filterbank(sample_rate, n_fft, n_mels, mel_fmin, mel_fmax),
            persistent=False,
        )

        self.conv_reducer = _build_conv_reducer(num_conv_layers, d_model)
        feature_height, feature_width = self._infer_feature_map_size(num_audio_samples)
        self.register_buffer(
            "_positional_encoding",
            _build_2d_sincos_positional_encoding(d_model, feature_height, feature_width),
            persistent=False,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, num_heads, feedforward_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, enable_nested_tensor=False
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, num_heads, feedforward_dim, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        # One learnable query vector per parameter (the DETR object queries).
        self.parameter_queries = nn.Parameter(torch.randn(len(self._class_counts), d_model) * 0.02)
        # Per-parameter projection heads emitting each parameter's class logits.
        self.class_heads = nn.ModuleList([nn.Linear(d_model, count) for count in self._class_counts])

    def mel_db_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """The normalized mel-dB spectrogram ``[batch, 1, n_mels, frames]`` the conv reducer sees."""
        return _compute_mel_db_spectrogram(
            audio, self._window, self._mel_filterbank,
            self._n_fft, self._hop_length, self._win_length, self._min_db, self._max_db,
        )

    def _infer_feature_map_size(self, num_audio_samples: int) -> Tuple[int, int]:
        with torch.no_grad():
            dummy = self.mel_db_spectrogram(torch.zeros(1, num_audio_samples))
            feature_map = self.conv_reducer(dummy)
        return int(feature_map.shape[-2]), int(feature_map.shape[-1])

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        spectrogram = self.mel_db_spectrogram(audio)  # [batch, 1, n_mels, frames]
        feature_map = self.conv_reducer(spectrogram)  # [batch, d_model, height, width]
        feature_map = feature_map + self._positional_encoding.unsqueeze(0)
        tokens = feature_map.flatten(2).transpose(1, 2)  # [batch, height*width, d_model]
        memory = self.encoder(tokens)  # [batch, num_tokens, d_model]

        batch_size = audio.shape[0]
        queries = self.parameter_queries.unsqueeze(0).expand(batch_size, -1, -1)
        decoded = self.decoder(queries, memory)  # [batch, num_parameters, d_model]
        logits = [head(decoded[:, position, :]) for position, head in enumerate(self.class_heads)]
        return torch.cat(logits, dim=1)  # [batch, total_class_dimension]
