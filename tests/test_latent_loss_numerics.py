"""Regression tests for the latent term's numerics (the Diva PresetGenVAEMLPRegressor NaN).

In eval the VAE skips reparameterization and z0 *is* mu, so ``samples - mu`` is exactly zero
and the quadratic term becomes ``0 / exp(logvar)``. Once logvar drifts far enough negative for
``exp`` to underflow, that is 0/0 = NaN, which poisons val_loss while every other term stays
finite. Job 1073696 died this way at epoch 6.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

from models.training.loss import (
    _LOGVAR_CLAMP,
    flow_latent_loss,
    gaussian_log_probability,
)


def test_density_is_finite_when_the_sample_is_the_mean_and_the_variance_underflows():
    """The exact eval-path failure: z0 == mu, logvar past exp's underflow point."""
    mu = torch.randn(4, 8)
    logvar = torch.full((4, 8), -800.0)
    assert torch.exp(logvar).eq(0.0).all()  # the underflow that used to make it 0/0

    log_probability = gaussian_log_probability(mu, mu, logvar)

    assert torch.isfinite(log_probability).all()


def test_latent_loss_is_finite_when_the_variance_underflows():
    """Same condition through the term the trainer actually logs."""
    mu = torch.randn(4, 8)
    logvar = torch.full((4, 8), -800.0)
    transformed = torch.randn(4, 8)
    log_determinant = torch.zeros(4)

    latent_loss = flow_latent_loss(mu, logvar, mu, transformed, log_determinant)

    assert torch.isfinite(latent_loss)


def test_the_clamp_is_a_no_op_for_healthy_logvar():
    """Well-conditioned runs must score identically to the unclamped port."""
    torch.manual_seed(0)
    samples, mu = torch.randn(4, 8), torch.randn(4, 8)
    logvar = torch.randn(4, 8)  # |logvar| << the clamp
    unclamped = -0.5 * (
        samples.shape[1] * float(torch.log(torch.tensor(2.0 * torch.pi)))
        + torch.sum(logvar + (samples - mu).square() / torch.exp(logvar), dim=1)
    )

    assert torch.allclose(gaussian_log_probability(samples, mu, logvar), unclamped, atol=1e-6)


def test_the_clamp_bounds_are_representable():
    """Both ends must stay normal floats, or the clamp reintroduces the bug it fixes."""
    assert torch.exp(torch.tensor(-_LOGVAR_CLAMP)) > 0.0
    assert torch.isfinite(torch.exp(torch.tensor(_LOGVAR_CLAMP)))
