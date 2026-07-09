"""LightningModule that trains a Gaussian-latent VAE regressor (Stage 2).

Objective: ``reconstruction_MSE + beta * KL(q(z|x) || N(0, I)) + controls_loss``. ``beta`` is
warmed up linearly (preset-gen-vae's ``LinearDynamicParam``); the controls term reuses
:class:`ParameterLoss` unchanged. Like :class:`LightningRegressor`, ``forward`` is
prediction-only and the wrapper exists during training only.
"""
from __future__ import annotations

from typing import Any, Dict, List

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn

from models.training.config import LossConfig, OptimizerConfig
from models.training.lightning_module import build_optimizers
from models.training.loss import ParameterLoss, gaussian_kl_divergence


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


class LightningVAERegressor(pl.LightningModule):
    """Wraps a VAE network + :class:`ParameterLoss` into a trainable module.

    ``network`` exposes ``forward(audio) -> [batch, ml_dimension]`` and
    ``forward_training(audio) -> VaeNetworkOutput``. ``loss_config`` supplies the
    reconstruction/latent knobs (``beta``, warmup, normalization).
    """

    def __init__(
        self,
        network: nn.Module,
        parameter_loss: ParameterLoss,
        optimizer_config: OptimizerConfig,
        loss_config: LossConfig,
    ) -> None:
        super().__init__()
        if loss_config.reconstruction_loss != "mse":
            raise ValueError(
                f"reconstruction_loss must be 'mse', got '{loss_config.reconstruction_loss}'."
            )
        self.network = network
        self.parameter_loss = parameter_loss
        self._optimizer_config = optimizer_config
        self._beta = float(loss_config.beta)
        self._beta_start_value = float(loss_config.beta_start_value)
        self._beta_warmup_epochs = int(loss_config.beta_warmup_epochs)
        self._normalize_latent_loss = bool(loss_config.normalize_latent_loss)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.network(audio)

    def training_step(self, batch: List[torch.Tensor], batch_index: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: List[torch.Tensor], batch_index: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def _current_beta(self, stage: str) -> float:
        # Warm up in training; validation uses the final beta so val_loss stays a stationary monitor.
        if stage != "train":
            return self._beta
        return linear_warmup(
            self.current_epoch, self._beta_start_value, self._beta, self._beta_warmup_epochs
        )

    def _shared_step(self, batch: List[torch.Tensor], stage: str) -> torch.Tensor:
        audio, targets = batch
        output = self.network.forward_training(audio)
        reconstruction_loss = F.mse_loss(
            output.reconstruction, output.target_spectrogram, reduction="mean"
        )
        kl_loss = gaussian_kl_divergence(
            output.mu, output.logvar, normalize=self._normalize_latent_loss
        )
        controls = self.parameter_loss(output.prediction, targets)
        beta = self._current_beta(stage)
        total = reconstruction_loss + beta * kl_loss + controls["loss"]

        log = dict(on_step=False, on_epoch=True, batch_size=audio.shape[0])
        self.log(f"{stage}_loss", total, prog_bar=True, **log)
        self.log(f"{stage}_reconstruction_loss", reconstruction_loss, prog_bar=True, **log)
        self.log(f"{stage}_kl_loss", kl_loss, **log)
        self.log(f"{stage}_controls_loss", controls["loss"], **log)
        self.log(f"{stage}_continuous_loss", controls["continuous_loss"], **log)
        self.log(f"{stage}_categorical_loss", controls["categorical_loss"], **log)
        if self.parameter_loss.has_categorical:
            accuracy = self.parameter_loss.categorical_accuracy(output.prediction, targets)
            self.log(f"{stage}_categorical_accuracy", accuracy, **log)
        if stage == "train":
            self.log("beta", beta, **log)
        return total

    def configure_optimizers(self) -> Dict[str, Any]:
        return build_optimizers(self.network, self._optimizer_config)
