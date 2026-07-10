"""RealNVP normalizing flow -- a plain-torch port of preset-gen-vae's ``CustomRealNVP``.

The paper builds its flows from the ``nflows`` package; this module ports exactly the
pieces ``CustomRealNVP`` uses -- affine coupling with alternating checkerboard masks, the
``ResidualNet`` conditioner, and flow batch-norm -- so the flow can live beside the
network as a predict-time dependency (no training-only imports, D-FRAMEWORK).

Forward direction only: the port uses the flow feed-forward (the flow regressor head,
issue #35; the latent z0 -> zK flow, issue #36). The inverse is never needed.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn


class _ResidualBlock(nn.Module):
    """One pre-activation residual block of nflows' ``ResidualNet`` (no context)."""

    def __init__(
        self, features: int, dropout: float, use_batch_norm: bool
    ) -> None:
        super().__init__()
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            self.batch_norm_layers = nn.ModuleList(
                [nn.BatchNorm1d(features, eps=1e-3) for _ in range(2)]
            )
        self.linear_layers = nn.ModuleList([nn.Linear(features, features) for _ in range(2)])
        self.dropout = nn.Dropout(dropout)
        # nflows' zero_initialization: the block starts close to the identity map.
        nn.init.uniform_(self.linear_layers[-1].weight, -1e-3, 1e-3)
        nn.init.uniform_(self.linear_layers[-1].bias, -1e-3, 1e-3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        if self.use_batch_norm:
            hidden = self.batch_norm_layers[0](hidden)
        hidden = F.relu(hidden)
        hidden = self.linear_layers[0](hidden)
        if self.use_batch_norm:
            hidden = self.batch_norm_layers[1](hidden)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.linear_layers[1](hidden)
        return inputs + hidden


class ResidualNetwork(nn.Module):
    """nflows' ``ResidualNet``: the conditioner mapping the coupling's identity half to
    the (shift, scale) parameters of its transformed half."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int,
        residual_blocks: int,
        dropout: float,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        self.initial_layer = nn.Linear(in_features, hidden_features)
        self.blocks = nn.ModuleList(
            [
                _ResidualBlock(hidden_features, dropout, use_batch_norm)
                for _ in range(residual_blocks)
            ]
        )
        self.final_layer = nn.Linear(hidden_features, out_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.initial_layer(inputs)
        for block in self.blocks:
            hidden = block(hidden)
        return self.final_layer(hidden)


class AffineCouplingLayer(nn.Module):
    """nflows' ``AffineCouplingTransform`` with a :class:`ResidualNetwork` conditioner.

    ``mask`` entries <= 0 mark the identity half; the conditioner reads it and emits shift
    and (sigmoid-constrained) scale for the other half.
    """

    def __init__(
        self,
        mask: torch.Tensor,
        hidden_features: int,
        residual_blocks: int,
        dropout: float,
        use_batch_norm: bool,
    ) -> None:
        super().__init__()
        feature_indices = torch.arange(mask.numel())
        self.register_buffer("identity_indices", feature_indices[mask <= 0], persistent=False)
        self.register_buffer("transform_indices", feature_indices[mask > 0], persistent=False)
        self._num_transform_features = int(self.transform_indices.numel())
        self.conditioner = ResidualNetwork(
            in_features=int(self.identity_indices.numel()),
            out_features=2 * self._num_transform_features,
            hidden_features=hidden_features,
            residual_blocks=residual_blocks,
            dropout=dropout,
            use_batch_norm=use_batch_norm,
        )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        identity_split = inputs[:, self.identity_indices]
        transform_split = inputs[:, self.transform_indices]
        transform_params = self.conditioner(identity_split)
        shift = transform_params[:, : self._num_transform_features]
        unconstrained_scale = transform_params[:, self._num_transform_features :]
        # nflows' default scale activation: strictly positive, near 1 at initialization.
        scale = torch.sigmoid(unconstrained_scale + 2.0) + 1e-3
        outputs = torch.empty_like(inputs)
        outputs[:, self.identity_indices] = identity_split
        outputs[:, self.transform_indices] = transform_split * scale + shift
        log_abs_determinant = torch.log(scale).sum(dim=1)
        return outputs, log_abs_determinant


class FlowBatchNorm(nn.Module):
    """nflows' ``transforms.normalization.BatchNorm``: batch-norm as an invertible
    transform, with a softplus-constrained positive weight and a log-det term."""

    def __init__(self, features: int, eps: float = 1e-5, momentum: float = 0.1) -> None:
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        initial_weight = math.log(math.exp(1.0 - eps) - 1.0)  # softplus(w) + eps == 1 at init
        self.unconstrained_weight = nn.Parameter(torch.full((features,), initial_weight))
        self.bias = nn.Parameter(torch.zeros(features))
        self.register_buffer("running_mean", torch.zeros(features))
        self.register_buffer("running_variance", torch.zeros(features))

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            mean, variance = inputs.mean(dim=0), inputs.var(dim=0)
            self.running_mean.mul_(1.0 - self.momentum).add_(mean.detach() * self.momentum)
            self.running_variance.mul_(1.0 - self.momentum).add_(variance.detach() * self.momentum)
        else:
            mean, variance = self.running_mean, self.running_variance
        weight = F.softplus(self.unconstrained_weight) + self.eps
        outputs = weight * (inputs - mean) / torch.sqrt(variance + self.eps) + self.bias
        log_abs_determinant = torch.sum(
            torch.log(weight) - 0.5 * torch.log(variance + self.eps)
        ).expand(inputs.shape[0])
        return outputs, log_abs_determinant


class RealNVP(nn.Module):
    """The paper's ``CustomRealNVP`` (e.g. ``realnvp_6l300``): ``coupling_layers`` affine
    couplings with alternating checkerboard masks, flow batch-norm between layers, and no
    batch-norm / dropout on the two last layers.

    ``forward`` returns the transformed tensor only (the feed-forward regressor use);
    :meth:`forward_with_log_determinant` also returns the per-sample log|det J| the
    latent-flow KL needs (issue #36).
    """

    def __init__(
        self,
        features: int,
        hidden_features: int,
        coupling_layers: int,
        residual_blocks_per_coupling: int = 2,
        dropout: float = 0.0,
        batch_norm_between_layers: bool = True,
        batch_norm_within_layers: bool = True,
    ) -> None:
        super().__init__()
        if features < 2:
            raise ValueError(f"RealNVP needs features >= 2 to split, got {features}.")
        if coupling_layers < 1:
            raise ValueError(f"coupling_layers must be >= 1, got {coupling_layers}.")
        mask = torch.ones(features)
        mask[::2] = -1
        layers = []
        for layer_index in range(coupling_layers):
            not_in_last_two = layer_index < coupling_layers - 2
            layers.append(
                AffineCouplingLayer(
                    mask,
                    hidden_features=hidden_features,
                    residual_blocks=residual_blocks_per_coupling,
                    dropout=dropout if not_in_last_two else 0.0,
                    use_batch_norm=batch_norm_within_layers,
                )
            )
            mask = -mask
            if batch_norm_between_layers and not_in_last_two:
                layers.append(FlowBatchNorm(features))
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.forward_with_log_determinant(inputs)
        return outputs

    def forward_with_log_determinant(
        self, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = inputs
        log_abs_determinant = torch.zeros(
            inputs.shape[0], device=inputs.device, dtype=inputs.dtype
        )
        for layer in self.layers:
            outputs, layer_log_abs_determinant = layer(outputs)
            log_abs_determinant = log_abs_determinant + layer_log_abs_determinant
        return outputs, log_abs_determinant
