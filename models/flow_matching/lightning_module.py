"""LightningModule that trains the flow-matching network.

The paper's training recipe (``surge_flow_matching_module.py``), on the framework's
harness: per batch, rescale the ML-side targets to ``[-1, 1]``, draw Gaussian noise,
minibatch-OT-pair it with the targets, drop conditioning at the CFG rate, and regress
the vector field onto the rectified path's target velocity with plain MSE (uniform
time weighting), plus the field's auxiliary ``penalty()`` term.

This is a bespoke objective, not a :class:`ParameterLoss` recipe -- the MSE is on a
*velocity* in flow space, where the routed continuous/categorical split does not
apply. The hook-injected ``ParameterLoss`` is deliberately unused. Validation follows
the paper's model-selection metric instead: actually sample (RK4 + CFG) and log the
parameter MSE in flow space as ``val_loss``, so checkpointing selects on the same
quantity as the reference (its ``val/param_mse``).

Optimizers come from the shared :func:`build_optimizers`; grad clipping and the rest
of the trainer recipe stay in :class:`TrainingConfig` as for every family.
"""
from __future__ import annotations

from typing import Any, Dict, List

import lightning.pytorch as pl
import torch
from torch import nn

from models.flow_matching.flow_matching import (
    optimal_transport_pairing,
    rectified_path_sample,
    rectified_target_velocity,
)
from models.training.config import OptimizerConfig
from models.training.lightning_module import build_optimizers


class LightningFlowMatching(pl.LightningModule):
    """Wraps a :class:`FlowMatchingNetwork` into the paper's training recipe.

    Args:
        network: the :class:`FlowMatchingNetwork` (owns featurization, encoder, field).
        optimizer_config: optimizer + scheduler settings (shared recipe).
        cfg_dropout_rate: probability of replacing a sample's conditioning with the
            learned dropout token (classifier-free guidance training).
        rectified_sigma_min: the rectified path's ``sigma_min`` (paper: 0).
        ot_pairing: minibatch optimal-transport coupling of noise to targets.
        validation_sample_steps / validation_cfg_strength: the RK4 sampling settings
            for the ``val_loss`` metric (paper: 50 steps, strength 2).
    """

    def __init__(
        self,
        network: nn.Module,
        optimizer_config: OptimizerConfig,
        cfg_dropout_rate: float = 0.1,
        rectified_sigma_min: float = 0.0,
        ot_pairing: bool = True,
        validation_sample_steps: int = 50,
        validation_cfg_strength: float = 2.0,
    ) -> None:
        super().__init__()
        self.network = network
        self._optimizer_config = optimizer_config
        self._cfg_dropout_rate = float(cfg_dropout_rate)
        self._rectified_sigma_min = float(rectified_sigma_min)
        self._ot_pairing = bool(ot_pairing)
        self._validation_sample_steps = int(validation_sample_steps)
        self._validation_cfg_strength = float(validation_cfg_strength)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.network(audio)

    def training_step(self, batch: List[torch.Tensor], batch_index: int) -> torch.Tensor:
        audio, targets = batch
        params = targets * 2.0 - 1.0  # ML-side [0, 1] -> flow space [-1, 1]

        noise = torch.randn_like(params)
        if self._ot_pairing:
            noise = optimal_transport_pairing(noise, params)

        conditioning = self.network.encode(audio)
        conditioning = self.network.vector_field.apply_dropout(
            conditioning, self._cfg_dropout_rate
        )

        with torch.no_grad():
            t = torch.rand(params.shape[0], 1, device=params.device, dtype=params.dtype)
            x_t = rectified_path_sample(noise, params, t, self._rectified_sigma_min)
            target_velocity = rectified_target_velocity(noise, params)

        prediction = self.network.velocity(x_t, t, conditioning)
        flow_loss = (prediction - target_velocity).square().mean(dim=-1).mean()
        penalty = self.network.vector_field.penalty()

        log = dict(on_step=False, on_epoch=True, batch_size=audio.shape[0])
        self.log("train_loss", flow_loss, prog_bar=True, **log)
        self.log("train_penalty", penalty, **log)
        return flow_loss + penalty

    def validation_step(self, batch: List[torch.Tensor], batch_index: int) -> torch.Tensor:
        audio, targets = batch
        params = targets * 2.0 - 1.0
        sampled = self.network.sample(
            audio,
            num_steps=self._validation_sample_steps,
            cfg_strength=self._validation_cfg_strength,
        )
        parameter_mse = (sampled - params).square().mean()
        self.log(
            "val_loss",
            parameter_mse,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=audio.shape[0],
        )
        return parameter_mse

    def configure_optimizers(self) -> Dict[str, Any]:
        return build_optimizers(self.network, self._optimizer_config)
