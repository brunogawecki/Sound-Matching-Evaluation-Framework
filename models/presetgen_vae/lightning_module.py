"""LightningModule that trains the preset-gen-vae VAE regressor.

Objective: ``reconstruction_MSE + beta * latent_loss + controls_loss``. ``beta`` is warmed up
linearly (preset-gen-vae's ``LinearDynamicParam``) and scales the latent term either way, as
the paper's ``train.py`` does; the controls term reuses :class:`ParameterLoss` unchanged.
Extends :class:`LightningRegressor` -- only the loss step differs; step routing, optimizers,
and the ``ParameterLoss`` logging are inherited.

The latent term follows the network: the Monte-Carlo estimate when it has a latent flow (the
paper's ``FlowVAE``, and both models it reports), the closed-form KL when it does not.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from models.presetgen_vae.network import VAENetworkOutput
from models.training.config import LossConfig, OptimizerConfig
from models.training.lightning_module import LightningRegressor
from models.training.loss import ParameterLoss, flow_latent_loss, gaussian_kl_divergence


def linear_warmup(
    current_epoch: int, start_value: float, end_value: float, warmup_epochs: int
) -> float:
    """Linear ramp ``start_value`` -> ``end_value`` over ``warmup_epochs`` epochs (from epoch 0).

    Clamps to ``start_value`` at epoch 0 and to ``end_value`` at/after ``warmup_epochs``; a
    non-positive ``warmup_epochs`` disables the ramp.
    """
    if warmup_epochs <= 0 or current_epoch >= warmup_epochs:
        return end_value
    if current_epoch <= 0:
        return start_value
    return start_value + (end_value - start_value) * current_epoch / warmup_epochs


class LightningVAERegressor(LightningRegressor):
    """Wraps a VAE network + :class:`ParameterLoss` into a trainable module.

    ``network`` exposes ``forward(audio) -> [batch, ml_dimension]`` and
    ``forward_training(audio) -> VAENetworkOutput``. ``loss_config`` supplies the
    reconstruction/latent knobs (``beta``, warmup, normalization).
    """

    def __init__(
        self,
        network: nn.Module,
        parameter_loss: ParameterLoss,
        optimizer_config: OptimizerConfig,
        loss_config: LossConfig,
    ) -> None:
        super().__init__(network, parameter_loss, optimizer_config)
        if loss_config.reconstruction_loss != "mse":
            raise ValueError(
                f"reconstruction_loss must be 'mse', got '{loss_config.reconstruction_loss}'."
            )
        self._beta = float(loss_config.beta)
        self._beta_start_value = float(loss_config.beta_start_value)
        self._beta_warmup_epochs = int(loss_config.beta_warmup_epochs)
        self._normalize_latent_loss = bool(loss_config.normalize_latent_loss)

    def _current_beta(self, stage: str) -> float:
        # Warm up in training; validation uses the final beta so val_loss stays a stationary monitor.
        if stage != "train":
            return self._beta
        return linear_warmup(
            self.current_epoch, self._beta_start_value, self._beta, self._beta_warmup_epochs
        )

    def _latent_loss(self, output: VAENetworkOutput) -> torch.Tensor:
        """The latent regularization term, chosen by whether the network has a latent flow."""
        if output.log_abs_determinant is None:
            return gaussian_kl_divergence(
                output.mu, output.logvar, normalize=self._normalize_latent_loss
            )
        return flow_latent_loss(
            output.mu,
            output.logvar,
            output.latent_sample,
            output.transformed_latent_sample,
            output.log_abs_determinant,
            normalize=self._normalize_latent_loss,
        )

    def _shared_step(self, batch: List[torch.Tensor], stage: str) -> torch.Tensor:
        audio, targets = batch
        output = self.network.forward_training(audio)
        reconstruction_loss = F.mse_loss(
            output.reconstruction, output.target_spectrogram, reduction="mean"
        )
        latent_loss = self._latent_loss(output)
        controls = self.parameter_loss(output.prediction, targets)
        beta = self._current_beta(stage)
        total = reconstruction_loss + beta * latent_loss + controls["loss"]

        log = dict(on_step=False, on_epoch=True, batch_size=audio.shape[0])
        self.log(f"{stage}_loss", total, prog_bar=True, **log)
        self.log(f"{stage}_reconstruction_loss", reconstruction_loss, prog_bar=True, **log)
        self.log(f"{stage}_latent_loss", latent_loss, **log)
        self.log(f"{stage}_controls_loss", controls["loss"], **log)
        self._log_parameter_losses(output.prediction, targets, controls, stage, log)
        if stage == "train":
            self.log("beta", beta, **log)
        return total
