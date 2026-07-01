"""Parameter-space training loss, routed by the corpus's ParameterSpace (issue #22).

Implements D2: continuous parameters are regressed (MSE, or MAE) and categorical
parameters are classified (cross-entropy over their one-hot block). The routing is
driven entirely by :attr:`ParameterSpace.loss_slices`, so the same loss works for
any subset without per-parameter special-casing.

Structure mirrors preset-gen-vae's ``SynthParamsLoss``
(``paper_repos/preset-gen-vae/model/loss.py``): a per-element-mean continuous term
plus a per-categorical-param-mean cross-entropy term, combined as
``continuous + categorical_loss_weight * categorical``. The default weight ``0.2``
matches its empirically-tuned ``categorical_loss_factor`` (cross-entropy is usually
much larger than MSE).

Pure ``torch`` -- no Lightning import, so it can be unit-tested on CPU with no
training framework.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.training.config import LossConfig
from synth.parameter_space import ParameterSpace


class ParameterLoss(nn.Module):
    """Routes predictions/targets through MSE (continuous) + CE (categorical).

    Args:
        parameter_space: defines the ML-side vector layout (``loss_slices``).
        loss_config: continuous-loss choice (``"mse"``/``"mae"``) and the
            categorical weight.

    Inputs to :meth:`forward` are ``[batch, ml_dimension]`` tensors: ``predictions``
    are raw model outputs (raw floats for continuous slots, **logits** for
    categorical blocks); ``targets`` are the corpus's ML-side vectors (continuous
    floats in place, one-hot categorical blocks).
    """

    def __init__(self, parameter_space: ParameterSpace, loss_config: LossConfig) -> None:
        super().__init__()
        if loss_config.continuous_loss not in ("mse", "mae"):
            raise ValueError(
                f"continuous_loss must be 'mse' or 'mae', got '{loss_config.continuous_loss}'."
            )
        self._continuous_loss = loss_config.continuous_loss
        self._categorical_loss_weight = float(loss_config.categorical_loss_weight)

        continuous_indices: List[int] = []
        categorical_slices: List[Tuple[int, int]] = []
        for vector_slice, kind, _name in parameter_space.loss_slices:
            if kind == "categorical":
                categorical_slices.append((vector_slice.start, vector_slice.stop))
            else:
                continuous_indices.append(vector_slice.start)

        # Registered as a buffer so it follows the module across .to(device).
        self.register_buffer(
            "_continuous_indices",
            torch.tensor(continuous_indices, dtype=torch.long),
            persistent=False,
        )
        self._categorical_slices = categorical_slices

    @property
    def has_continuous(self) -> bool:
        return self._continuous_indices.numel() > 0

    @property
    def has_categorical(self) -> bool:
        return len(self._categorical_slices) > 0

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute total and per-kind losses.

        Returns a dict with ``loss`` (the scalar to backpropagate),
        ``continuous_loss`` and ``categorical_loss`` (both detached-safe scalars
        for logging; they are real graph nodes, log with ``.detach()`` if needed).
        """
        continuous_loss = self._continuous_term(predictions, targets)
        categorical_loss = self._categorical_term(predictions, targets)
        total = continuous_loss + self._categorical_loss_weight * categorical_loss
        return {
            "loss": total,
            "continuous_loss": continuous_loss,
            "categorical_loss": categorical_loss,
        }

    def _continuous_term(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if not self.has_continuous:
            return predictions.new_zeros(())
        predicted = predictions.index_select(1, self._continuous_indices)
        target = targets.index_select(1, self._continuous_indices)
        if self._continuous_loss == "mae":
            return F.l1_loss(predicted, target, reduction="mean")
        return F.mse_loss(predicted, target, reduction="mean")

    def _categorical_term(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if not self.has_categorical:
            return predictions.new_zeros(())
        # Per-block cross-entropy, averaged over blocks -- equivalent to summing and
        # dividing by the number of categorical params (preset-gen-vae normalization).
        per_block = predictions.new_zeros(())
        for start, stop in self._categorical_slices:
            block_logits = predictions[:, start:stop]
            target_class = targets[:, start:stop].argmax(dim=1)
            per_block = per_block + F.cross_entropy(block_logits, target_class, reduction="mean")
        return per_block / len(self._categorical_slices)

    @torch.no_grad()
    def categorical_accuracy(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Mean top-1 accuracy over all categorical blocks (cheap sanity metric).

        Returns ``NaN`` when the space has no categorical parameters.
        """
        if not self.has_categorical:
            return predictions.new_full((), float("nan"))
        correct = predictions.new_zeros(())
        for start, stop in self._categorical_slices:
            predicted_class = predictions[:, start:stop].argmax(dim=1)
            target_class = targets[:, start:stop].argmax(dim=1)
            correct = correct + (predicted_class == target_class).float().mean()
        return correct / len(self._categorical_slices)
