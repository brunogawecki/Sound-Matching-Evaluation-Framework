"""LightningModule for the SynthRL parameter stage (SynthRL-p, paper stage 1).

Objective: the paper's per-parameter classification loss (§3.3) -- cross-entropy with
**Gaussian label smoothing** over the network's per-parameter class heads. The batch
carries the framework's ML-side target vector (continuous floats + one-hot categorical
blocks); this module maps it to per-head target class indices (continuous values are
binned, categorical blocks argmax-decoded), looks up the Gaussian-smoothed soft targets
from :meth:`SynthRLRepresentation.smoothing_matrices`, and averages the soft cross-entropy
across heads and batch.

Training-only (D-FRAMEWORK): imported lazily by the SynthRL families, so the eval path
needs no Lightning. This module is the RL-free stage; the RL curriculum (SynthRL-i) is a
separate module (see the package plan).
"""
from __future__ import annotations

from typing import List

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn

from models.synthrl.representation import SynthRLRepresentation
from models.training.config import OptimizerConfig
from models.training.lightning_module import build_optimizers


class SynthRLParameterRegressor(pl.LightningModule):
    """Trains a :class:`SynthRLNetwork` with the Gaussian-smoothed classification loss.

    ``network`` maps ``audio [batch, num_samples]`` to flat class logits
    ``[batch, total_class_dimension]``. ``representation`` supplies the class layout,
    the ML-target -> class-index mapping, and the per-head Gaussian smoothing.
    """

    def __init__(
        self,
        network: nn.Module,
        representation: SynthRLRepresentation,
        optimizer_config: OptimizerConfig,
    ) -> None:
        super().__init__()
        self.network = network
        self._optimizer_config = optimizer_config
        self._class_slices = representation.class_slices
        self._num_parameters = len(representation.class_counts)

        # Per-parameter ML-side layout for the target -> class-index mapping.
        self._parameter_kinds: List[str] = []
        self._ml_slices: List[slice] = []
        continuous_low = torch.zeros(self._num_parameters)
        continuous_high = torch.ones(self._num_parameters)
        specs = representation.parameter_space.parameter_specs
        for position, (ml_slice, kind, _name) in enumerate(
            representation.parameter_space.loss_slices
        ):
            self._parameter_kinds.append(kind)
            self._ml_slices.append(ml_slice)
            if kind == "continuous":
                continuous_low[position], continuous_high[position] = specs[position].bounds
        self._num_bins = representation.num_bins
        self.register_buffer("_continuous_low", continuous_low, persistent=False)
        self.register_buffer("_continuous_high", continuous_high, persistent=False)

        # Per-head Gaussian-smoothing lookup tables (derived; kept off the checkpoint).
        for position, matrix in enumerate(representation.smoothing_matrices()):
            self.register_buffer(
                f"_smoothing_{position}", torch.tensor(matrix, dtype=torch.float32), persistent=False
            )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        return self.network(audio)

    def training_step(self, batch: List[torch.Tensor], batch_index: int) -> torch.Tensor:
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch: List[torch.Tensor], batch_index: int) -> torch.Tensor:
        return self._shared_step(batch, stage="val")

    def _target_class_index(self, position: int, ml_targets: torch.Tensor) -> torch.Tensor:
        """The per-sample target class index ``[batch]`` for parameter ``position``."""
        ml_slice = self._ml_slices[position]
        if self._parameter_kinds[position] == "categorical":
            return ml_targets[:, ml_slice].argmax(dim=1)
        value = ml_targets[:, ml_slice.start]
        low = self._continuous_low[position]
        high = self._continuous_high[position]
        fraction = (value - low) / (high - low)
        return (fraction * self._num_bins).floor().long().clamp(0, self._num_bins - 1)

    def _shared_step(self, batch: List[torch.Tensor], stage: str) -> torch.Tensor:
        audio, ml_targets = batch
        logits = self.network(audio)  # [batch, total_class_dimension]

        total_loss = logits.new_zeros(())
        total_correct = logits.new_zeros(())
        for position in range(self._num_parameters):
            class_index = self._target_class_index(position, ml_targets)  # [batch]
            block_logits = logits[:, self._class_slices[position]]  # [batch, count]
            smoothing = getattr(self, f"_smoothing_{position}")
            soft_target = smoothing[class_index]  # [batch, count]
            total_loss = total_loss - (soft_target * F.log_softmax(block_logits, dim=1)).sum(dim=1).mean()
            total_correct = total_correct + (block_logits.argmax(dim=1) == class_index).float().mean()

        loss = total_loss / self._num_parameters
        accuracy = total_correct / self._num_parameters
        log = dict(on_step=False, on_epoch=True, batch_size=audio.shape[0])
        self.log(f"{stage}_loss", loss, prog_bar=True, **log)
        self.log(f"{stage}_class_accuracy", accuracy, prog_bar=True, **log)
        return loss

    def configure_optimizers(self):
        return build_optimizers(self.network, self._optimizer_config)
