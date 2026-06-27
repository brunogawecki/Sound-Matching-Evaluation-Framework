import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synth.parameter_space import ParameterSpecification, ParameterSpace
from evaluation.metrics.parameter import param_mae, param_mse, param_accuracy


# ---------------------------------------------------------------------------
# Pure-Python tests (no plugin, no torch). Parameter metrics operate on
# ML-side vectors routed through ParameterSpace.loss_slices.
# ---------------------------------------------------------------------------

# Two continuous params + two categorical blocks. Vector layout (ml_dimension 7):
#   AMP  -> [0]
#   FREQ -> [1]
#   CAT3 -> [2, 3, 4]   (options [0.0, 0.5, 1.0])
#   CAT2 -> [5, 6]      (options [0.0, 1.0])
def make_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="AMP", kind="continuous", default=0.5),
        ParameterSpecification(name="FREQ", kind="continuous", default=0.5),
        ParameterSpecification(name="CAT3", kind="categorical", options=[0.0, 0.5, 1.0], default=0.0),
        ParameterSpecification(name="CAT2", kind="categorical", options=[0.0, 1.0], default=0.0),
    ])


# CAT3 wrong (class 0 vs class 2), CAT2 correct (both class 0).
TARGET = np.array([0.2, 0.6, 1.0, 0.0, 0.0, 1.0, 0.0])
PREDICTED = np.array([0.5, 0.6, 0.0, 0.0, 1.0, 1.0, 0.0])


# -- continuous metrics (param_mae / param_mse) ------------------------------

def test_param_mae_averages_absolute_error_over_continuous_only():
    # |0.2-0.5|=0.3, |0.6-0.6|=0.0 -> mean 0.15. Categorical blocks are ignored.
    assert param_mae(TARGET, PREDICTED, make_space()) == pytest.approx(0.15)


def test_param_mse_averages_squared_error_over_continuous_only():
    # 0.3^2=0.09, 0.0^2=0.0 -> mean 0.045.
    assert param_mse(TARGET, PREDICTED, make_space()) == pytest.approx(0.045)


def test_continuous_metrics_are_zero_for_identical_vectors():
    space = make_space()
    assert param_mae(TARGET, TARGET, space) == pytest.approx(0.0)
    assert param_mse(TARGET, TARGET, space) == pytest.approx(0.0)


def test_continuous_metrics_ignore_categorical_blocks():
    # Perturb only the categorical entries; continuous error must be unchanged.
    space = make_space()
    perturbed = PREDICTED.copy()
    perturbed[2:] = [0.0, 1.0, 0.0, 0.0, 1.0]
    assert param_mae(TARGET, perturbed, space) == pytest.approx(param_mae(TARGET, PREDICTED, space))
    assert param_mse(TARGET, perturbed, space) == pytest.approx(param_mse(TARGET, PREDICTED, space))


# -- categorical metric (param_accuracy) -------------------------------------

def test_param_accuracy_is_fraction_of_matching_categorical_argmax():
    # CAT3 wrong, CAT2 correct -> 1 of 2 -> 0.5.
    assert param_accuracy(TARGET, PREDICTED, make_space()) == pytest.approx(0.5)


def test_param_accuracy_is_one_for_identical_vectors():
    assert param_accuracy(TARGET, TARGET, make_space()) == pytest.approx(1.0)


def test_param_accuracy_is_zero_when_every_categorical_is_wrong():
    space = make_space()
    # Flip both categorical argmaxes away from the target.
    wrong = TARGET.copy()
    wrong[2:5] = [0.0, 0.0, 1.0]  # CAT3 class 2 vs target class 0
    wrong[5:7] = [0.0, 1.0]       # CAT2 class 1 vs target class 0
    assert param_accuracy(TARGET, wrong, space) == pytest.approx(0.0)


def test_param_accuracy_accepts_logits_via_argmax():
    space = make_space()
    # Predicted categorical blocks as raw probabilities, not one-hot.
    logits = TARGET.copy()
    logits[2:5] = [0.1, 0.7, 0.2]  # argmax -> class 1, target class 0 -> wrong
    logits[5:7] = [0.9, 0.1]       # argmax -> class 0, target class 0 -> correct
    assert param_accuracy(TARGET, logits, space) == pytest.approx(0.5)


def test_param_accuracy_ignores_continuous_entries():
    space = make_space()
    perturbed = PREDICTED.copy()
    perturbed[0] = 0.99  # change a continuous value only
    assert param_accuracy(TARGET, perturbed, space) == pytest.approx(
        param_accuracy(TARGET, PREDICTED, space)
    )


# -- degenerate spaces -------------------------------------------------------

def test_param_accuracy_is_nan_without_categorical_parameters():
    space = ParameterSpace([ParameterSpecification(name="AMP", kind="continuous", default=0.5)])
    target = np.array([0.3])
    predicted = np.array([0.7])
    assert math.isnan(param_accuracy(target, predicted, space))


def test_continuous_metrics_are_nan_without_continuous_parameters():
    space = ParameterSpace([
        ParameterSpecification(name="CAT", kind="categorical", options=[0.0, 1.0], default=0.0),
    ])
    target = np.array([1.0, 0.0])
    predicted = np.array([0.0, 1.0])
    assert math.isnan(param_mae(target, predicted, space))
    assert math.isnan(param_mse(target, predicted, space))
