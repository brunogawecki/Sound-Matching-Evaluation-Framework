"""Parameter-space training loss, routed by the corpus's ParameterSpace.

Continuous parameters are regressed (MSE or MAE); categorical parameters are
classified (cross-entropy over their one-hot block). Routing is driven by
:attr:`ParameterSpace.loss_slices`, so the same loss works for any subset. The terms
combine as ``continuous + categorical_loss_weight * categorical``.
"""
from __future__ import annotations

import math
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


def gaussian_kl_divergence(
    mu: torch.Tensor, logvar: torch.Tensor, normalize: bool = True
) -> torch.Tensor:
    """Closed-form KL of the diagonal-Gaussian posterior from ``N(0, I)`` (VAE latent term).

    Ports preset-gen-vae's ``GaussianDkl``: ``0.5 * sum(exp(logvar) + mu^2 - logvar - 1)``
    summed over the latent dimension and averaged over the batch. ``normalize`` also divides
    by the latent dimension so the term stays comparable to a mean-reduced reconstruction MSE.
    ``mu``/``logvar`` are ``[batch, latent_dimension]``.

    Used only when a VAE has **no** latent flow; with one, see :func:`flow_latent_loss`.
    """
    per_sample = 0.5 * torch.sum(torch.exp(logvar) + mu.square() - logvar - 1.0, dim=1)
    divergence = per_sample.mean()
    if normalize:
        divergence = divergence / mu.shape[1]
    return divergence


_LOG_2_PI = math.log(2.0 * math.pi)


def standard_gaussian_log_probability(samples: torch.Tensor) -> torch.Tensor:
    """Per-sample log-density under ``N(0, I)``. Ports preset-gen-vae's ``utils/probability.py``."""
    return -0.5 * (samples.shape[1] * _LOG_2_PI + torch.sum(samples.square(), dim=1))


def gaussian_log_probability(
    samples: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor
) -> torch.Tensor:
    """Per-sample log-density under a diagonal Gaussian. Ports ``utils/probability.py``."""
    return -0.5 * (
        samples.shape[1] * _LOG_2_PI
        + torch.sum(logvar + (samples - mu).square() / torch.exp(logvar), dim=1)
    )


def flow_latent_loss(
    mu: torch.Tensor,
    logvar: torch.Tensor,
    latent_sample: torch.Tensor,
    transformed_latent_sample: torch.Tensor,
    log_abs_determinant: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """Monte-Carlo latent term for a VAE with a latent normalizing flow (z0 -> zK).

    Ports preset-gen-vae's ``FlowVAE.latent_loss``. With a flow, ``q(zK|x)`` has no closed
    form, so the KL is replaced by a one-sample estimate of the negated ELBO latent terms:
    ``-(log p(zK) - log q(z0) + log|det J|)``, averaged over the batch. ``normalize`` also
    divides by the latent dimension, matching :func:`gaussian_kl_divergence`.
    """
    log_probability_prior = standard_gaussian_log_probability(transformed_latent_sample)
    log_probability_posterior = gaussian_log_probability(latent_sample, mu, logvar)
    per_sample = log_probability_prior - log_probability_posterior + log_abs_determinant
    latent_loss = -per_sample.mean()
    if normalize:
        latent_loss = latent_loss / latent_sample.shape[1]
    return latent_loss
