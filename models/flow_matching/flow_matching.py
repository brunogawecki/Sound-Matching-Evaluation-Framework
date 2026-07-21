"""The flow-matching core: rectified probability path, minibatch OT pairing, and the
CFG-guided RK4 sampler.

Framework-agnostic port of the training/sampling math from ``synth-permutations``
(``surge_flow_matching_module.py`` + ``data/ot.py``): a rectified-flow path between a
standard-normal source and the parameter targets, minibatch optimal-transport coupling
of noise to targets (Hungarian algorithm), and 4th-order Runge-Kutta ODE integration
with classifier-free guidance at sample time. Pure functions over tensors -- no
Lightning, no framework classes -- so both the training step and the offline
``predict`` path share the exact same math.

A vector field here is any module with the paper's interface:
``forward(x, t, conditioning) -> velocity`` (``conditioning=None`` means the
unconditional/CFG-dropout branch), plus ``apply_dropout`` and ``penalty()``.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn


def rectified_path_sample(
    x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, sigma_min: float = 0.0
) -> torch.Tensor:
    """A point on the rectified (straight-line) path: ``x0 * (1 - t) * (1 - sigma_min) + x1 * t``.

    ``x0`` is the noise sample, ``x1`` the data sample, ``t`` in ``[0, 1]`` shaped to
    broadcast (``[batch, 1]`` against ``[batch, dimension]``). ``sigma_min=0`` is the
    paper's setting (its ``rectified_sigma_min``).
    """
    return x0 * (1.0 - t) * (1.0 - sigma_min) + x1 * t


def rectified_target_velocity(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    """The rectified path's target vector field, constant in ``t``: ``x1 - x0``."""
    return x1 - x0


def optimal_transport_pairing(noise: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Reorder ``noise`` rows to the minibatch optimal-transport coupling with ``targets``.

    Hungarian assignment on the pairwise Euclidean cost (the paper's ``_hungarian_match``),
    solved with ``scipy.optimize.linear_sum_assignment``. Returns ``noise`` permuted so row
    ``i`` is the noise sample matched to ``targets[i]`` -- the targets (and everything
    aligned with them: audio, conditioning) stay in place.
    """
    from scipy.optimize import linear_sum_assignment

    with torch.no_grad():
        cost = torch.cdist(targets.detach().float().cpu(), noise.detach().float().cpu())
    _, noise_indices = linear_sum_assignment(cost.numpy())
    return noise[torch.from_numpy(noise_indices).to(noise.device)]


def _guided_velocity(
    vector_field: nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    conditioning: Optional[torch.Tensor],
    cfg_strength: float,
) -> torch.Tensor:
    """Classifier-free-guided velocity: ``(1 - w) * unconditional + w * conditional``."""
    conditional = vector_field(x, t, conditioning)
    unconditional = vector_field(x, t, None)
    return (1.0 - cfg_strength) * unconditional + cfg_strength * conditional


def rk4_sample(
    vector_field: nn.Module,
    noise: torch.Tensor,
    conditioning: Optional[torch.Tensor],
    num_steps: int,
    cfg_strength: float,
) -> torch.Tensor:
    """Integrate the learned ODE from ``t=0`` (noise) to ``t=1`` with 4th-order Runge-Kutta.

    The paper's ``rk4_with_cfg`` loop: fixed step ``dt = 1 / num_steps``, each stage
    evaluated with classifier-free guidance. Deterministic given ``noise``. Returns the
    sample in the flow's own space (the ``[-1, 1]``-rescaled ML-side vector).
    """
    t = torch.zeros(noise.shape[0], 1, device=noise.device, dtype=noise.dtype)
    dt = 1.0 / num_steps
    sample = noise
    for _ in range(num_steps):
        k1 = _guided_velocity(vector_field, sample, t, conditioning, cfg_strength)
        k2 = _guided_velocity(vector_field, sample + dt * k1 / 2, t + dt / 2, conditioning, cfg_strength)
        k3 = _guided_velocity(vector_field, sample + dt * k2 / 2, t + dt / 2, conditioning, cfg_strength)
        k4 = _guided_velocity(vector_field, sample + dt * k3, t + dt, conditioning, cfg_strength)
        sample = sample + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t = t + dt
    return sample
