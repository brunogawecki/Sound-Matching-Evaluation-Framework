"""The generic training wrapper around a prediction network (issue #22).

A :class:`LightningRegressor` is a thin :class:`~lightning.pytorch.LightningModule` that
wraps an injected **network** (a plain ``nn.Module``) for training only. The
network owns everything inference needs -- featurization (audio -> STFT/mel) lives
inside its ``forward``, so ``predict`` later runs end-to-end from the raw waveform
with the network alone, no Lightning. The LightningModule adds only the training
recipe: the step functions, the optimizer, and logging.

This is the decoupling that keeps Lightning off the Mac eval path (D-FRAMEWORK):
the wrapper exists during training; the checkpoint exported afterwards carries only
the network's ``state_dict``.

Imports Lightning: a training-only module.
"""
from __future__ import annotations

from typing import Any, Dict, List

import lightning.pytorch as pl
import torch
from torch import nn

from models.training.config import OptimizerConfig
from models.training.loss import ParameterLoss


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
        batch_size = audio.shape[0]
        self.log(f"{stage}_loss", losses["loss"], prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}_continuous_loss", losses["continuous_loss"], batch_size=batch_size)
        self.log(f"{stage}_categorical_loss", losses["categorical_loss"], batch_size=batch_size)
        if self.parameter_loss.has_categorical:
            accuracy = self.parameter_loss.categorical_accuracy(predictions, targets)
            self.log(f"{stage}_categorical_accuracy", accuracy, batch_size=batch_size)
        return losses["loss"]

    def configure_optimizers(self) -> Dict[str, Any]:
        config = self._optimizer_config
        if config.name.lower() == "adamw":
            optimizer: torch.optim.Optimizer = torch.optim.AdamW(
                self.network.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        elif config.name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
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
