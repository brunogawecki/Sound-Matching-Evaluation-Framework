"""The learned vector fields the flow integrates.

Two fields, one per CNF family, differing only in how they treat the parameter vector:

``ConditionalResidualMLPField`` (the paper's ``ConditionalResidualMLP``,
``components/residual_mlp.py``) is the non-equivariant one behind the CNF (MLP) rows.
A residual MLP over the parameter vector whose blocks are modulated (Ada-LN-style
scale/shift/gate) by a conditioning vector built from the audio encoding and a
sinusoidal encoding of flow time.

``EquivariantTransformerField`` (the paper's ``ApproxEquivTransformer``) is the
*approximately* permutation-equivariant one behind the CNF (Param2Tok) rows. It maps
the parameter vector to a set of tokens with :class:`Param2TokProjection`, runs a
Diffusion Transformer with **no positional encoding** over that set, and maps back.
Equivariance is what buys it the symmetry argument: with no positional encoding the
transformer is permutation-equivariant on its tokens, and the learned assignment is
pushed toward routing functionally-equivalent parameters (e.g. one operator's
parameters) to the same token slot. It is only *approximate* because the assignment is
learned rather than derived from the synth's known symmetry group.

Both fields take conditioning per block: the encoder emits
``[batch, num_layers, conditioning_dim]`` and block ``i`` reads slice ``i``.

Every field exposes the shared interface consumed by
:mod:`models.flow_matching.flow_matching` and the Lightning module:
``forward(x, t, conditioning)`` where ``conditioning=None`` selects the learned
CFG-dropout token (the unconditional branch), ``apply_dropout(conditioning, rate)``
for train-time classifier-free-guidance dropout, and ``penalty()`` for any auxiliary
regularizer (zero for the MLP field, the weighted assignment-matrix L1 for Param2Tok).
"""
from __future__ import annotations

import math
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


class ConditionedField(nn.Module):
    """Shared base: the learned CFG-dropout token and the auxiliary penalty hook."""

    def __init__(self, conditioning_dim: int) -> None:
        super().__init__()
        self.cfg_dropout_token = nn.Parameter(torch.randn(1, 1, conditioning_dim))

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

    def _conditioning_or_dropout_token(
        self, conditioning: Optional[torch.Tensor], batch_size: int
    ) -> torch.Tensor:
        if conditioning is None:
            return self.cfg_dropout_token[0].expand(batch_size, -1)
        return conditioning


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


class ConditionalResidualMLPField(ConditionedField):
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
        super().__init__(conditioning_dim)
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

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        conditioning = self._conditioning_or_dropout_token(conditioning, x.shape[0])

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


class Param2TokProjection(nn.Module):
    """Param2Tok: the learned parameters <-> tokens map (the reference's ``LearntProjection``).

    ``param_to_token`` embeds each scalar parameter along its own learned direction
    (``in_projection``, one ``d_token`` vector per parameter), lifts it with an FFN, then
    contracts the ``num_params`` embeddings down to ``num_tokens`` token slots through the
    learned assignment matrix ``A`` ``[num_tokens, num_params]``. ``token_to_param`` runs
    the same path backwards with ``A^T``.

    Whether this becomes a *symmetry-aware* map is not enforced anywhere: it is coaxed by
    the L1 :meth:`penalty` on ``A``, which pushes the assignment toward being sparse, so
    each token slot ends up carrying a small group of parameters. That is where the
    "approximately" in approximately-equivariant lives.

    Note on tying: ``out_projection`` is *initialized* as the transpose of
    ``in_projection`` but is a separate parameter, so the two drift apart during training
    (this matches the reference, which clones rather than ties).
    """

    def __init__(
        self,
        d_model: int,
        d_token: int,
        num_params: int,
        num_tokens: int,
        initial_ffn: bool = True,
        final_ffn: bool = False,
    ) -> None:
        super().__init__()
        assignment = torch.full(
            (num_tokens, num_params), 1.0 / math.sqrt(num_tokens * num_params)
        )
        self.assignment = nn.Parameter(assignment + 1e-4 * torch.randn_like(assignment))

        # One shared random direction, repeated per parameter and then jittered, so the
        # parameters start near-indistinguishable to the network (the reference's init).
        projection = (torch.randn(1, d_token) / math.sqrt(d_token)).repeat(num_params, 1)
        projection = projection + 1e-4 * torch.randn_like(projection)
        self.in_projection = nn.Parameter(projection.clone())
        self.out_projection = nn.Parameter(projection.T.clone())

        self.initial_ffn = (
            nn.Sequential(nn.Linear(d_token, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            if initial_ffn
            else None
        )
        if final_ffn:
            self.final_ffn: Optional[nn.Module] = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_token)
            )
        elif d_token == d_model:
            self.final_ffn = None
        else:
            self.final_ffn = nn.Linear(d_model, d_token)

    def param_to_token(self, x: torch.Tensor) -> torch.Tensor:
        """``[batch, num_params]`` -> ``[batch, num_tokens, d_model]``."""
        values = torch.einsum("bn,nd->bnd", x, self.in_projection)
        if self.initial_ffn is not None:
            values = self.initial_ffn(values)
        return torch.einsum("bnd,kn->bkd", values, self.assignment)

    def token_to_param(self, x: torch.Tensor) -> torch.Tensor:
        """``[batch, num_tokens, d_model]`` -> ``[batch, num_params]``."""
        deassigned = torch.einsum("bkd,kn->bnd", x, self.assignment)
        if self.final_ffn is not None:
            deassigned = self.final_ffn(deassigned)
        return torch.einsum("bnd,dn->bn", deassigned, self.out_projection)

    def penalty(self) -> torch.Tensor:
        return self.assignment.abs().mean()


class DiffusionTransformerBlock(nn.Module):
    """A DiT block: self-attention + FFN, both Ada-LN modulated by the conditioning.

    The conditioning vector produces six modulations per block (scale/shift/gate for the
    attention and the FFN branch) -- the reference's ``adaln_mode="basic"``, its Surge
    setting. No positional encoding is applied anywhere, which is what keeps the block
    permutation-equivariant over its tokens.
    """

    def __init__(
        self,
        d_model: int,
        conditioning_dim: int,
        num_heads: int,
        feedforward_dimension: int,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dimension),
            nn.GELU(),
            nn.Linear(feedforward_dimension, d_model),
        )
        self.cond = nn.Sequential(nn.GELU(), nn.Linear(conditioning_dim, d_model * 6))

        # The reference's zero_init=False branch (its Surge setting).
        nn.init.xavier_normal_(self.ff[0].weight)
        nn.init.zeros_(self.ff[0].bias)
        nn.init.xavier_normal_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale1, shift1, gate1, scale2, shift2, gate2 = (
            self.cond(conditioning)[:, None].chunk(6, dim=-1)
        )

        residual = x
        x = scale1 * self.norm1(x) + shift1
        x = gate1 * self.attn(x, x, x)[0] + residual

        residual = x
        x = scale2 * self.norm2(x) + shift2
        return gate2 * self.ff(x) + residual


class EquivariantTransformerField(ConditionedField):
    """The CNF (Param2Tok) vector field: the paper's ``ApproxEquivTransformer``.

    Param2Tok maps the parameter vector to ``num_tokens`` tokens, a stack of DiT blocks
    with no positional encoding processes them (hence permutation-equivariantly), and
    Param2Tok maps back to a velocity. Defaults are the paper's ``surge_flow.yaml``
    (``d_model`` 512, 8 layers, 8 heads, 128 tokens, 0.01 on the assignment L1).

    :meth:`penalty` returns the *weighted* assignment L1, so the Lightning module can add
    it to the flow loss unscaled, exactly like the reference does.
    """

    def __init__(
        self,
        num_params: int,
        d_model: int = 512,
        time_encoding_dimension: int = 256,
        conditioning_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        num_tokens: int = 128,
        feedforward_dimension: Optional[int] = None,
        projection_penalty: float = 0.01,
    ) -> None:
        super().__init__(conditioning_dim)
        self.projection = Param2TokProjection(
            d_model=d_model,
            d_token=d_model,
            num_params=num_params,
            num_tokens=num_tokens,
        )
        self.layers = nn.ModuleList(
            [
                DiffusionTransformerBlock(
                    d_model,
                    conditioning_dim=d_model,
                    num_heads=num_heads,
                    feedforward_dimension=feedforward_dimension or d_model,
                )
                for _ in range(num_layers)
            ]
        )
        self.time_encoding = SinusoidalEncoding(time_encoding_dimension)
        self.conditioning_ffn = nn.Sequential(
            nn.Linear(conditioning_dim + time_encoding_dimension, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self._projection_penalty = float(projection_penalty)

    def penalty(self) -> torch.Tensor:
        return self._projection_penalty * self.projection.penalty()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        conditioning: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        conditioning = self._conditioning_or_dropout_token(conditioning, x.shape[0])

        x = self.projection.param_to_token(x)

        t = self.time_encoding(t)
        if conditioning.ndim == 3:
            t = t.unsqueeze(1).repeat(1, conditioning.shape[1], 1)
        z = self.conditioning_ffn(torch.cat([conditioning, t], dim=-1))

        for i, layer in enumerate(self.layers):
            x = layer(x, z if z.ndim == 2 else z[:, i])
        return self.projection.token_to_param(x)
