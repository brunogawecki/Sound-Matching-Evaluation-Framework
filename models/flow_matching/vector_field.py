"""The learned vector fields the flow integrates.

Stage 1 ports the non-equivariant field: ``ConditionalResidualMLPField``, the paper's
``ConditionalResidualMLP`` (``components/residual_mlp.py``) behind its CNF (MLP) rows.
A residual MLP over the parameter vector whose blocks are modulated (Ada-LN-style
scale/shift/gate) by a conditioning vector built from the audio encoding and a
sinusoidal encoding of flow time. Like the reference, one conditioning token per
block: the encoder emits ``[batch, num_layers, conditioning_dim]`` and block ``i``
reads slice ``i``.

Every field exposes the shared interface consumed by
:mod:`models.flow_matching.flow_matching` and the Lightning module:
``forward(x, t, conditioning)`` where ``conditioning=None`` selects the learned
CFG-dropout token (the unconditional branch), ``apply_dropout(conditioning, rate)``
for train-time classifier-free-guidance dropout, and ``penalty()`` for any auxiliary
regularizer (zero here; the Param2Tok field's assignment-matrix L1 arrives in Stage 2).
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class SinusoidalEncoding(nn.Module):
    """Sinusoidal encoding of a scalar (flow time) into ``encoding_dimension`` features."""

    def __init__(self, encoding_dimension: int) -> None:
        super().__init__()
        half = encoding_dimension // 2
        basis = 1.0 / torch.pow(torch.tensor(10000.0), torch.arange(0, half) / half)
        self.register_buffer("basis", basis[None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, 1] -> [batch, 2 * half]
        return torch.cat([torch.cos(x * self.basis), torch.sin(x * self.basis)], dim=-1)


class ConditionalResidualMLPBlock(nn.Module):
    """One residual MLP block with scale/shift/gate conditioning (the reference's block).

    The reference normalizes twice (``norm(x)`` then ``g * norm(x) + b``); kept as-is
    for parity.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.cond = nn.Sequential(nn.GELU(), nn.Linear(d_model, d_model * 3))

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        gate, scale, shift = self.cond(conditioning).chunk(3, dim=-1)
        x = scale * self.norm(x) + shift
        x = self.net(x)
        return residual + gate * x


class ConditionalResidualMLPField(nn.Module):
    """The CNF (MLP) vector field: ``(x_t, t, conditioning) -> velocity``.

    Defaults are the paper's ``model/surge_flowmlp.yaml`` (``d_model`` 768, 9 layers,
    256-dimensional sinusoidal time encoding). ``conditioning`` is either
    ``[batch, conditioning_dim]`` (one shared vector) or
    ``[batch, num_layers, conditioning_dim]`` (per-block, the configuration the paper
    trains); ``None`` substitutes the learned CFG-dropout token.
    """

    def __init__(
        self,
        num_params: int,
        d_model: int = 768,
        time_encoding_dimension: int = 256,
        conditioning_dim: int = 512,
        num_layers: int = 9,
    ) -> None:
        super().__init__()
        self.cfg_dropout_token = nn.Parameter(torch.randn(1, 1, conditioning_dim))
        self.in_proj = nn.Linear(num_params, d_model)
        self.out_proj = nn.Linear(d_model, num_params)
        self.blocks = nn.ModuleList(
            [ConditionalResidualMLPBlock(d_model) for _ in range(num_layers)]
        )
        self.time_encoding = SinusoidalEncoding(time_encoding_dimension)
        self.conditioning_ffn = nn.Sequential(
            nn.Linear(conditioning_dim + time_encoding_dimension, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def penalty(self) -> torch.Tensor:
        return torch.zeros((), device=self.cfg_dropout_token.device)

    def apply_dropout(self, conditioning: torch.Tensor, rate: float = 0.1) -> torch.Tensor:
        """Replace each sample's conditioning with the CFG-dropout token at ``rate``."""
        if rate == 0.0:
            return conditioning
        keep_mask = torch.rand(conditioning.shape[0], device=conditioning.device) > rate
        if conditioning.ndim == 2:
            keep_mask = keep_mask[..., None]
            dropout_token = self.cfg_dropout_token[0]
        elif conditioning.ndim == 3:
            keep_mask = keep_mask[..., None, None]
            dropout_token = self.cfg_dropout_token
        else:
            raise ValueError(f"Unexpected conditioning shape {tuple(conditioning.shape)}.")
        return torch.where(keep_mask, conditioning, dropout_token)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if conditioning is None:
            conditioning = self.cfg_dropout_token[0].expand(x.shape[0], -1)

        t = self.time_encoding(t)
        if conditioning.ndim == 3:
            t = t.unsqueeze(1).repeat(1, conditioning.shape[1], 1)
        z = torch.cat([conditioning, t], dim=-1)
        z = self.conditioning_ffn(z)

        x = self.in_proj(x)
        for i, block in enumerate(self.blocks):
            block_conditioning = z if z.ndim == 2 else z[:, i]
            x = block(x, block_conditioning)
        return self.out_proj(x)
