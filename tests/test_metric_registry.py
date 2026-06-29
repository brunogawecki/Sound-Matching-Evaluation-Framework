import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synth.parameter_space import ParameterSpecification, ParameterSpace
from evaluation.registry import MetricSpecification, METRIC_PANEL, metric_names


def small_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="AMP", kind="continuous", default=0.5),
        ParameterSpecification(name="CAT", kind="categorical", options=[0.0, 1.0], default=0.0),
    ])


def spec_by_name(name: str) -> MetricSpecification:
    return next(spec for spec in METRIC_PANEL if spec.name == name)


# -- registry structure ------------------------------------------------------

def test_metric_names_are_unique():
    names = [spec.name for spec in METRIC_PANEL]
    assert len(names) == len(set(names))


def test_metric_names_helper_matches_panel():
    assert metric_names() == [spec.name for spec in METRIC_PANEL]


def test_parameter_axis_specs_are_parameter_input_and_unnormalized():
    for spec in METRIC_PANEL:
        if spec.axis == "parameter":
            assert spec.input_type == "parameter"
            assert spec.normalize_level is False


def test_higher_is_better_flags():
    assert spec_by_name("param_accuracy").higher_is_better is True
    assert spec_by_name("param_mae").higher_is_better is False
    assert spec_by_name("param_mse").higher_is_better is False


def test_panel_contains_the_three_parameter_metrics():
    assert {"param_mae", "param_mse", "param_accuracy"} <= set(metric_names())


# -- magnitude axis (slice #2) -----------------------------------------------

MAGNITUDE_METRICS = {"lsd", "spectral_convergence", "mel_mae", "mel_mse", "mss"}


def test_panel_contains_the_magnitude_metrics():
    assert MAGNITUDE_METRICS <= set(metric_names())


def test_magnitude_specs_are_raw_audio_and_lower_is_better():
    for spec in METRIC_PANEL:
        if spec.name in MAGNITUDE_METRICS:
            assert spec.axis == "magnitude"
            assert spec.input_type == "audio"
            assert spec.normalize_level is False
            assert spec.higher_is_better is False


# -- loudness + pitch axes (slice #4) ----------------------------------------

LOUDNESS_METRICS = {"loudness_envelope_l1", "integrated_loudness_error"}
PITCH_METRICS = {"f0_rmse"}


def test_panel_contains_the_loudness_and_pitch_metrics():
    assert (LOUDNESS_METRICS | PITCH_METRICS) <= set(metric_names())


def test_loudness_and_pitch_specs_are_raw_audio_and_lower_is_better():
    for spec in METRIC_PANEL:
        if spec.name in (LOUDNESS_METRICS | PITCH_METRICS):
            assert spec.axis == ("loudness" if spec.name in LOUDNESS_METRICS else "pitch")
            assert spec.input_type == "audio"
            assert spec.normalize_level is False
            assert spec.higher_is_better is False


# -- guard -------------------------------------------------------------------

def test_normalize_level_must_be_false_for_parameter_metrics():
    with pytest.raises(ValueError):
        MetricSpecification(
            name="bad",
            axis="parameter",
            input_type="parameter",
            normalize_level=True,
            higher_is_better=False,
            compute=lambda *args, **kwargs: 0.0,
        )


# -- specs are callable ------------------------------------------------------

def test_each_parameter_spec_compute_returns_a_float():
    space = small_space()
    target = np.array([0.3, 1.0, 0.0])
    predicted = np.array([0.6, 0.0, 1.0])
    for spec in METRIC_PANEL:
        if spec.input_type == "parameter":
            value = spec.compute(target, predicted, space)
            assert isinstance(value, float)
