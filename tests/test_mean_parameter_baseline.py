import os
import sys
from typing import Dict, List

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")  # Layer 3 is the framework's torch land

from synth.parameter_space import ParameterSpecification, ParameterSpace
from models import MeanParameterBaseline


# One continuous param + one 3-option categorical (ml_dimension == 1 + 3 == 4).
def make_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="AMP", kind="continuous", default=0.5),
        ParameterSpecification(name="CAT", kind="categorical", options=[0.0, 0.5, 1.0], default=0.0),
    ])


class FakeTargetDataset:
    """A stand-in exposing only what MeanParameterBaseline.fit reads.

    Holds the two attributes the baseline touches -- ``parameter_space`` and the
    ``(N, ml_dimension)`` ``targets`` matrix -- built from explicit synth-side
    rows so the expected mean/mode is known exactly.
    """

    def __init__(self, space: ParameterSpace, rows: List[Dict[str, float]]):
        self.parameter_space = space
        if rows:
            matrix = np.stack([space.synth_dict_to_ml_vector(row) for row in rows]).astype(np.float32)
        else:
            matrix = np.zeros((0, space.ml_dimension), dtype=np.float32)
        self.targets = torch.from_numpy(matrix)


# Rows: AMP mean is 0.4; CAT class 0.0 occurs twice vs 1.0 once -> mode is 0.0.
ROWS = [
    {"AMP": 0.2, "CAT": 0.0},
    {"AMP": 0.4, "CAT": 0.0},
    {"AMP": 0.6, "CAT": 1.0},
]


def fitted_baseline() -> MeanParameterBaseline:
    model = MeanParameterBaseline()
    model.fit(FakeTargetDataset(make_space(), ROWS))
    return model


# -- prediction content ------------------------------------------------------

def test_continuous_parameter_is_the_column_mean():
    prediction = fitted_baseline().predict(torch.zeros(8))
    assert prediction["AMP"] == pytest.approx(0.4)


def test_categorical_parameter_is_the_majority_class():
    prediction = fitted_baseline().predict(torch.zeros(8))
    assert prediction["CAT"] == 0.0  # averaged one-hot argmax -> most frequent class


def test_predict_ignores_the_audio():
    model = fitted_baseline()
    first = model.predict(torch.rand(8))
    second = model.predict(torch.rand(16) * 100.0)
    assert first == second


def test_prediction_is_a_valid_synth_dict():
    space = make_space()
    prediction = fitted_baseline().predict(torch.zeros(8))
    assert set(prediction) == set(space.names)  # keys match the subset exactly
    assert all(np.isfinite(value) for value in prediction.values())


# -- persistence -------------------------------------------------------------

def test_save_load_round_trip(tmp_path):
    model = fitted_baseline()
    before = model.predict(torch.zeros(8))
    path = tmp_path / "baseline.json"
    model.save(path)

    restored = MeanParameterBaseline()  # no dataset / VST needed to load
    restored.load(path)
    assert restored.predict(torch.zeros(8)) == before


# -- guards ------------------------------------------------------------------

def test_fit_on_empty_corpus_raises():
    model = MeanParameterBaseline()
    with pytest.raises(ValueError, match="empty corpus"):
        model.fit(FakeTargetDataset(make_space(), []))


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        MeanParameterBaseline().predict(torch.zeros(8))
