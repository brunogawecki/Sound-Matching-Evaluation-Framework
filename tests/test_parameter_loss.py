import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
import torch.nn.functional as F

from models.training.config import LossConfig
from models.training.loss import ParameterLoss, gaussian_kl_divergence
from synth.parameter_space import ParameterSpace, ParameterSpecification


def continuous_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="A", kind="continuous"),
        ParameterSpecification(name="B", kind="continuous"),
    ])


def categorical_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="C", kind="categorical", options=[0.0, 0.5, 1.0]),
    ])


def mixed_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="A", kind="continuous"),
        ParameterSpecification(name="C", kind="categorical", options=[0.0, 0.5, 1.0]),
    ])


# -- continuous --------------------------------------------------------------

def test_continuous_only_matches_plain_mse():
    loss = ParameterLoss(continuous_space(), LossConfig())
    predictions = torch.tensor([[0.2, 0.8], [0.5, 0.1]])
    targets = torch.tensor([[0.0, 1.0], [0.4, 0.3]])
    out = loss(predictions, targets)
    assert out["continuous_loss"] == pytest.approx(F.mse_loss(predictions, targets).item())
    assert out["categorical_loss"].item() == pytest.approx(0.0)
    assert out["loss"] == pytest.approx(out["continuous_loss"].item())


def test_mae_option_matches_l1():
    loss = ParameterLoss(continuous_space(), LossConfig(continuous_loss="mae"))
    predictions = torch.tensor([[0.2, 0.8]])
    targets = torch.tensor([[0.0, 1.0]])
    out = loss(predictions, targets)
    assert out["continuous_loss"] == pytest.approx(F.l1_loss(predictions, targets).item())


def test_invalid_continuous_loss_is_rejected():
    with pytest.raises(ValueError, match="mse.*mae"):
        ParameterLoss(continuous_space(), LossConfig(continuous_loss="huber"))


# -- categorical -------------------------------------------------------------

def test_categorical_confident_correct_is_near_zero():
    loss = ParameterLoss(categorical_space(), LossConfig())
    predictions = torch.tensor([[10.0, -10.0, -10.0]])  # confident class 0
    targets = torch.tensor([[1.0, 0.0, 0.0]])           # one-hot class 0
    out = loss(predictions, targets)
    assert out["categorical_loss"].item() < 1e-3
    assert out["continuous_loss"].item() == pytest.approx(0.0)


def test_categorical_confident_wrong_is_large():
    loss = ParameterLoss(categorical_space(), LossConfig())
    predictions = torch.tensor([[10.0, -10.0, -10.0]])  # confident class 0
    targets = torch.tensor([[0.0, 0.0, 1.0]])           # truth is class 2
    out = loss(predictions, targets)
    assert out["categorical_loss"].item() > 10.0


def test_categorical_block_uses_cross_entropy():
    loss = ParameterLoss(categorical_space(), LossConfig())
    predictions = torch.tensor([[0.3, 1.2, -0.5]])
    targets = torch.tensor([[0.0, 1.0, 0.0]])  # class 1
    expected = F.cross_entropy(predictions, torch.tensor([1])).item()
    assert loss(predictions, targets)["categorical_loss"].item() == pytest.approx(expected)


# -- mixed routing -----------------------------------------------------------

def test_mixed_routing_partitions_the_vector():
    space = mixed_space()
    weight = 0.2
    loss = ParameterLoss(space, LossConfig(categorical_loss_weight=weight))
    predictions = torch.tensor([[0.6, 2.0, 0.0, -1.0]])
    targets = torch.tensor([[0.5, 0.0, 1.0, 0.0]])  # cont 0.5; cat class 1

    out = loss(predictions, targets)
    expected_continuous = F.mse_loss(predictions[:, 0:1], targets[:, 0:1]).item()
    expected_categorical = F.cross_entropy(predictions[:, 1:4], torch.tensor([1])).item()
    assert out["continuous_loss"].item() == pytest.approx(expected_continuous)
    assert out["categorical_loss"].item() == pytest.approx(expected_categorical)
    assert out["loss"].item() == pytest.approx(
        expected_continuous + weight * expected_categorical
    )


def test_categorical_loss_weight_scales_only_the_categorical_term():
    space = mixed_space()
    predictions = torch.tensor([[0.6, 2.0, 0.0, -1.0]])
    targets = torch.tensor([[0.5, 0.0, 1.0, 0.0]])
    low = ParameterLoss(space, LossConfig(categorical_loss_weight=0.0))(predictions, targets)
    high = ParameterLoss(space, LossConfig(categorical_loss_weight=1.0))(predictions, targets)
    # weight 0 drops the categorical term entirely -> total is the continuous loss.
    assert low["loss"].item() == pytest.approx(low["continuous_loss"].item())
    assert high["loss"].item() == pytest.approx(
        high["continuous_loss"].item() + high["categorical_loss"].item()
    )


# -- accuracy ----------------------------------------------------------------

def test_categorical_accuracy_counts_top1_matches():
    loss = ParameterLoss(categorical_space(), LossConfig())
    predictions = torch.tensor([[5.0, 0.0, 0.0], [0.0, 0.0, 5.0]])  # classes 0, 2
    targets = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])      # classes 0, 0
    assert loss.categorical_accuracy(predictions, targets).item() == pytest.approx(0.5)


def test_categorical_accuracy_is_nan_without_categoricals():
    loss = ParameterLoss(continuous_space(), LossConfig())
    predictions = torch.tensor([[0.2, 0.8]])
    targets = torch.tensor([[0.0, 1.0]])
    assert torch.isnan(loss.categorical_accuracy(predictions, targets))


# -- Gaussian KL (VAE latent term) -------------------------------------------

def test_kl_is_zero_for_standard_normal_posterior():
    mu = torch.zeros(4, 8)
    logvar = torch.zeros(4, 8)  # variance 1
    assert gaussian_kl_divergence(mu, logvar).item() == pytest.approx(0.0)


def test_kl_matches_closed_form():
    torch.manual_seed(0)
    mu = torch.randn(3, 5)
    logvar = torch.randn(3, 5)
    expected = 0.5 * torch.sum(torch.exp(logvar) + mu.square() - logvar - 1.0, dim=1).mean()
    assert gaussian_kl_divergence(mu, logvar, normalize=False).item() == pytest.approx(
        expected.item()
    )
    assert gaussian_kl_divergence(mu, logvar, normalize=True).item() == pytest.approx(
        expected.item() / mu.shape[1]
    )


def test_kl_is_non_negative_and_grows_with_departure_from_prior():
    small = gaussian_kl_divergence(torch.full((2, 4), 0.5), torch.zeros(2, 4))
    large = gaussian_kl_divergence(torch.full((2, 4), 3.0), torch.zeros(2, 4))
    assert small.item() >= 0.0
    assert large.item() > small.item()
