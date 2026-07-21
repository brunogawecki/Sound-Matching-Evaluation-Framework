"""The audio conditioning encoder: a from-scratch Audio Spectrogram Transformer.

Faithful port of ``AudioSpectrogramTransformer`` (+ ``PatchEmbed`` /
``PositionalEncoding``) from ``synth-permutations``' ``components/transformer.py``: a
pre-norm ViT over overlapping 16x16 mel-spectrogram patches whose learned "embed"
query tokens attention-pool the sequence into ``num_conditioning_outputs``
conditioning vectors -- one per vector-field layer. This is the paper's *conditioning
encoder*, shared by both CNF families; it is not the AST regression baseline (that is
a separate model, not ported). No pretrained weights.

The only contract change from the reference is ``input_channels=1`` (the framework
renders mono, D3); the mel front-end that produces the input spectrogram lives in
:mod:`models.flow_matching.network`, and ``spectrogram_shape`` is computed there from
the corpus render contract instead of being hardcoded to Surge's ``(128, 401)``.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """A learned additive positional encoding over a fixed number of positions."""

    def __init__(self, d_model: int, num_positions: int) -> None:
        super().__init__()
        # The reference's "norm0.02" init, the variant its AST config uses.
        self.positional_embedding = nn.Parameter(torch.randn(1, num_positions, d_model) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.positional_embedding[:, : x.shape[1], :]


class PatchEmbed(nn.Module):
    """ViT-style convolutional patch embedding with AST's overlapping stride.

    Zero-pads the spectrogram up to the next whole patch in each axis, then projects
    ``patch_size`` x ``patch_size`` patches (stride ``patch_stride``) to ``d_model``
    and flattens them into a token sequence. ``num_tokens`` is measured empirically
    from a dummy forward, exactly like the reference.
    """

    def __init__(
        self,
        patch_size: int,
        patch_stride: int,
        input_channels: int,
        d_model: int,
        spectrogram_shape: Tuple[int, int],
    ) -> None:
        super().__init__()
        if patch_stride >= patch_size:
            raise ValueError("patch_stride must be smaller than patch_size (overlapping patches).")
        num_mels, num_frames = spectrogram_shape
        mel_padding = (patch_stride - (num_mels - patch_size)) % patch_stride
        time_padding = (patch_stride - (num_frames - patch_size)) % patch_stride
        # (left, right, top, bottom) on [batch, channel, mel, time]: time is the last axis.
        self.pad = nn.ZeroPad2d((0, time_padding, 0, mel_padding))
        self.projection = nn.Conv2d(
            in_channels=input_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_stride,
        )
        self.num_tokens = self._measure_num_tokens(input_channels, spectrogram_shape)

    def _measure_num_tokens(
        self, input_channels: int, spectrogram_shape: Tuple[int, int]
    ) -> int:
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *spectrogram_shape)
            output_shape = self.projection(self.pad(dummy)).shape
        return int(math.prod(output_shape[-2:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pad(x)
        x = self.projection(x)
        return x.flatten(2).transpose(1, 2)  # [batch, num_tokens, d_model]


class AudioSpectrogramTransformer(nn.Module):
    """Normalized mel-dB spectrogram ``[batch, 1, n_mels, frames]`` ->
    conditioning ``[batch, num_conditioning_outputs, d_model]``.

    Patch embedding -> prepended learned embed tokens -> learned positional encoding
    over the full sequence -> pre-norm Transformer encoder (GELU, no dropout) -> the
    embed-token outputs through a final linear projection. Defaults are the paper's
    ``encoder/ast.yaml``.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 8,
        num_conditioning_outputs: int = 8,
        patch_size: int = 16,
        patch_stride: int = 10,
        input_channels: int = 1,
        spectrogram_shape: Tuple[int, int] = (128, 401),
    ) -> None:
        super().__init__()
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            patch_stride=patch_stride,
            input_channels=input_channels,
            d_model=d_model,
            spectrogram_shape=spectrogram_shape,
        )
        self.positional_encoding = PositionalEncoding(
            d_model, self.patch_embed.num_tokens + num_conditioning_outputs
        )
        self.embed_tokens = nn.Parameter(
            torch.empty(1, num_conditioning_outputs, d_model).normal_(0.0, 1e-6)
        )
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model,
                    num_heads,
                    d_model,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        embed_tokens = self.embed_tokens.expand(x.shape[0], -1, -1)
        x = torch.cat((embed_tokens, x), dim=1)
        x = self.positional_encoding(x)
        for block in self.blocks:
            x = block(x)
        x = x[:, : self.embed_tokens.shape[1]]
        return self.out_proj(x)
