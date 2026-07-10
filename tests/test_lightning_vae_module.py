"""Unit tests for the Stage 2 VAE trainer: beta warmup schedule + the multi-term loss step."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
pytest.importorskip("lightning")

import torch.nn.functional as F
from torch import nn

from models.presetgen_vae import VAENetworkOutput
from models.training.config import LossConfig, OptimizerConfig
from models.training.lightning_vae_module import LightningVAERegressor, linear_warmup
from models.training.loss import ParameterLoss, gaussian_kl_divergence
from synth.parameter_space import ParameterSpace, ParameterSpecification


# -- beta warmup -------------------------------------------------------------

def test_linear_warmup_clamps_and_interpolates():
    assert linear_warmup(0, 0.1, 0.2, 10) == pytest.approx(0.1)     # start
    assert linear_warmup(5, 0.1, 0.2, 10) == pytest.approx(0.15)    # midpoint
    assert linear_warmup(10, 0.1, 0.2, 10) == pytest.approx(0.2)    # end
    assert linear_warmup(99, 0.1, 0.2, 10) == pytest.approx(0.2)    # past end


def test_linear_warmup_disabled_when_no_warmup_epochs():
    assert linear_warmup(0, 0.1, 0.2, 0) == pytest.approx(0.2)


# -- the loss step -----------------------------------------------------------

class FakeVAENetwork(nn.Module):
    """Returns fixed VAE outputs regardless of input, so the loss composition is checkable."""

    def __init__(self, output: VAENetworkOutput):
        super().__init__()
        self._output = output
        self._parameter = nn.Linear(1, 1)  # gives configure_optimizers something to optimize

    def forward(self, audio):
        return self._output.prediction

    def forward_training(self, audio):
        return self._output


def continuous_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="A", kind="continuous"),
        ParameterSpecification(name="B", kind="continuous"),
    ])


def make_module(loss_config: LossConfig):
    prediction = torch.tensor([[0.6, 0.2], [0.1, 0.9]])
    reconstruction = torch.tensor([[[[0.5, -0.5]]], [[[0.0, 0.2]]]])
    target = torch.tensor([[[[0.4, -0.6]]], [[[0.1, 0.3]]]])
    mu = torch.tensor([[0.5, -0.3, 0.1], [0.2, 0.0, -0.4]])
    logvar = torch.tensor([[0.0, 0.2, -0.1], [0.1, -0.2, 0.0]])
    network = FakeVAENetwork(
        VAENetworkOutput(prediction, reconstruction, target, mu, logvar)
    )
    parameter_loss = ParameterLoss(continuous_space(), loss_config)
    module = LightningVAERegressor(network, parameter_loss, OptimizerConfig(), loss_config)
    targets = torch.tensor([[0.5, 0.5], [0.0, 1.0]])
    return module, network, targets


def test_step_total_is_reconstruction_plus_beta_kl_plus_controls():
    loss_config = LossConfig()
    module, network, targets = make_module(loss_config)
    audio = torch.zeros(2, 8)
    out = network.forward_training(audio)

    total = module._shared_step((audio, targets), stage="train")

    expected_recons = F.mse_loss(out.reconstruction, out.target_spectrogram)
    expected_kl = gaussian_kl_divergence(out.mu, out.logvar, normalize=True)
    expected_controls = F.mse_loss(out.prediction, targets)
    # current_epoch is 0 without a trainer -> beta is the warmup start value.
    beta = loss_config.beta_start_value
    assert total.item() == pytest.approx(
        (expected_recons + beta * expected_kl + expected_controls).item()
    )


def test_validation_step_uses_final_beta():
    loss_config = LossConfig()
    module, network, targets = make_module(loss_config)
    audio = torch.zeros(2, 8)
    out = network.forward_training(audio)

    total = module._shared_step((audio, targets), stage="val")

    expected_recons = F.mse_loss(out.reconstruction, out.target_spectrogram)
    expected_kl = gaussian_kl_divergence(out.mu, out.logvar, normalize=True)
    expected_controls = F.mse_loss(out.prediction, targets)
    assert total.item() == pytest.approx(
        (expected_recons + loss_config.beta * expected_kl + expected_controls).item()
    )


def test_unsupported_reconstruction_loss_is_rejected():
    with pytest.raises(ValueError, match="reconstruction_loss"):
        make_module(LossConfig(reconstruction_loss="l1"))
