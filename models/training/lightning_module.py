"""LightningModule that trains an injected prediction network.

Wraps a plain ``nn.Module`` network with the training recipe (step functions,
optimizer, logging) and a :class:`ParameterLoss`. The network owns featurization
inside its ``forward``; the wrapper exists during training only.
"""
from __future__ import annotations

from typing import Any, Dict, List

import lightning.pytorch as pl
import torch
from torch import nn

from models.training.config import OptimizerConfig
from models.training.loss import ParameterLoss


def build_optimizers(network: nn.Module, config: OptimizerConfig) -> Dict[str, Any]:
    """The optimizer (+ optional cosine scheduler) dict Lightning expects. Shared by every
    training module so the recipe lives in one place."""
    if config.name.lower() == "adamw":
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    elif config.name.lower() == "adam":
        optimizer = torch.optim.Adam(
            network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer '{config.name}' (use 'adamw' or 'adam').")

    if config.scheduler is None:
        return {"optimizer": optimizer}
    if config.scheduler.lower() == "cosine":
        if not config.scheduler_max_epochs:
            raise ValueError("scheduler='cosine' requires optimizer.scheduler_max_epochs.")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.scheduler_max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    raise ValueError(f"Unsupported scheduler '{config.scheduler}' (use 'cosine' or null).")


class LightningRegressor(pl.LightningModule):
    """Wraps a network + :class:`ParameterLoss` into a trainable module.

    Args:
        network: the plain ``nn.Module`` mapping ``audio [batch, num_samples]``
            to an ML-side prediction ``[batch, ml_dimension]`` (raw floats for
            continuous slots, logits for categorical blocks).
        parameter_loss: the routed MSE/CE loss.
        optimizer_config: optimizer + scheduler settings.
    """

    def __init__(
        self,
        network: nn.Module,
        parameter_loss: ParameterLoss,
        optimizer_config: OptimizerConfig,
    ) -> None:
        super().__init__()
        self.network = network
        self.parameter_loss = parameter_loss
        self._optimizer_config = optimizer_config

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.network(audio)

    def training_step(self, batch: List[torch.Tensor], batch_index: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: List[torch.Tensor], batch_index: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def _shared_step(self, batch: List[torch.Tensor], stage: str) -> torch.Tensor:
        audio, targets = batch
        predictions = self.network(audio)
        losses = self.parameter_loss(predictions, targets)
        # Epoch aggregates: {stage}_loss is what ModelCheckpoint/EarlyStopping monitor.
        log = dict(on_step=False, on_epoch=True, batch_size=audio.shape[0])
        self.log(f"{stage}_loss", losses["loss"], prog_bar=True, **log)
        self._log_parameter_losses(predictions, targets, losses, stage, log)
        return losses["loss"]

    def _log_parameter_losses(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        losses: Dict[str, torch.Tensor],
        stage: str,
        log_kwargs: Dict[str, Any],
    ) -> None:
        """Log the :class:`ParameterLoss` components (shared with subclasses)."""
        self.log(f"{stage}_continuous_loss", losses["continuous_loss"], **log_kwargs)
        self.log(f"{stage}_categorical_loss", losses["categorical_loss"], **log_kwargs)
        if self.parameter_loss.has_categorical:
            accuracy = self.parameter_loss.categorical_accuracy(predictions, targets)
            self.log(f"{stage}_categorical_accuracy", accuracy, **log_kwargs)

    def configure_optimizers(self) -> Dict[str, Any]:
        return build_optimizers(self.network, self._optimizer_config)
