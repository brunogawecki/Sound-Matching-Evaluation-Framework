import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")  # checkpoint I/O is pure torch (no Lightning)
from torch import nn

from models.training.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    export_checkpoint,
    load_checkpoint,
)
from synth.parameter_space import ParameterSpace, ParameterSpecification


class SmallNetwork(nn.Module):
    def __init__(self, ml_dimension: int, hidden: int = 4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(ml_dimension, hidden), nn.ReLU(), nn.Linear(hidden, ml_dimension))

    def forward(self, x):
        return self.net(x)


def make_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="A", kind="continuous"),
        ParameterSpecification(name="C", kind="categorical", options=[0.0, 0.5, 1.0]),
    ])


def test_round_trips_weights_hparams_and_space(tmp_path):
    space = make_space()
    hparams = {"ml_dimension": space.ml_dimension, "hidden": 4}
    network = SmallNetwork(space.ml_dimension, hidden=4)
    path = tmp_path / "checkpoint.pt"

    export_checkpoint(network, hparams, space, path)
    payload = load_checkpoint(path)

    assert payload["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert payload["architecture_hparams"] == hparams
    assert ParameterSpace.from_dict(payload["parameter_space"]).names == space.names
    assert set(payload["state_dict"]) == set(network.state_dict())


def test_reloaded_network_reproduces_forward(tmp_path):
    space = make_space()
    network = SmallNetwork(space.ml_dimension)
    network.eval()
    example = torch.randn(2, space.ml_dimension)
    with torch.no_grad():
        before = network(example)

    path = tmp_path / "checkpoint.pt"
    export_checkpoint(network, {"ml_dimension": space.ml_dimension, "hidden": 4}, space, path)

    reloaded = SmallNetwork(space.ml_dimension)
    reloaded.load_state_dict(load_checkpoint(path)["state_dict"])
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(example)
    torch.testing.assert_close(before, after)


def test_export_creates_parent_directories(tmp_path):
    space = make_space()
    network = SmallNetwork(space.ml_dimension)
    nested = tmp_path / "checkpoints" / "run" / "model.pt"
    export_checkpoint(network, {"ml_dimension": space.ml_dimension, "hidden": 4}, space, nested)
    assert nested.exists()


def test_unsupported_format_version_is_rejected(tmp_path):
    space = make_space()
    network = SmallNetwork(space.ml_dimension)
    path = tmp_path / "checkpoint.pt"
    export_checkpoint(network, {"ml_dimension": space.ml_dimension, "hidden": 4}, space, path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    payload["format_version"] = 999
    torch.save(payload, path)
    with pytest.raises(ValueError, match="format_version"):
        load_checkpoint(path)
