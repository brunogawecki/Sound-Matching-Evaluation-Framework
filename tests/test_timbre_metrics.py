import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

librosa = pytest.importorskip("librosa")

from evaluation.metrics.audio_based import mfcc_mae, mfcc_mse

# ---------------------------------------------------------------------------
# Synthetic-audio tests for the timbre axis. The MFCC metrics are pure callables
# over two raw mono waveforms with the uniform audio signature ``fn(target,
# prediction, *, sample_rate) -> float``. Both are lower-is-better; identical
# inputs score 0.
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
DURATION_SEC = 0.5

ALL_METRICS = [mfcc_mae, mfcc_mse]


def sine(frequency: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * DURATION_SEC)) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)


@pytest.fixture
def target() -> np.ndarray:
    return sine(440.0)


@pytest.fixture
def different() -> np.ndarray:
    return sine(880.0)


# -- identical inputs --------------------------------------------------------

@pytest.mark.parametrize("metric", ALL_METRICS)
def test_metric_is_zero_for_identical_audio(metric, target):
    assert metric(target, target, sample_rate=SAMPLE_RATE) == pytest.approx(0.0, abs=1e-6)


# -- different inputs --------------------------------------------------------

@pytest.mark.parametrize("metric", ALL_METRICS)
def test_metric_is_positive_for_different_audio(metric, target, different):
    assert metric(target, different, sample_rate=SAMPLE_RATE) > 0.0


@pytest.mark.parametrize("metric", ALL_METRICS)
def test_metric_returns_python_float(metric, target, different):
    value = metric(target, different, sample_rate=SAMPLE_RATE)
    assert isinstance(value, float)


# -- symmetric metrics -------------------------------------------------------

@pytest.mark.parametrize("metric", ALL_METRICS)
def test_metrics_are_order_independent(metric, target, different):
    forward = metric(target, different, sample_rate=SAMPLE_RATE)
    backward = metric(different, target, sample_rate=SAMPLE_RATE)
    assert forward == pytest.approx(backward)


# -- invariance to a common gain ---------------------------------------------
# A gain common to both inputs shifts only the 0th cepstral coefficient, equally,
# so the per-coefficient difference -- and thus the MAE/MSE -- is unchanged.

@pytest.mark.parametrize("metric", ALL_METRICS)
def test_metric_is_invariant_to_common_gain(metric, target, different):
    base = metric(target, different, sample_rate=SAMPLE_RATE)
    scaled = metric(2.0 * target, 2.0 * different, sample_rate=SAMPLE_RATE)
    assert scaled == pytest.approx(base, rel=1e-3)


# -- finiteness on silence ---------------------------------------------------

@pytest.mark.parametrize("metric", ALL_METRICS)
def test_metric_is_finite_for_silent_inputs(metric, target):
    silence = np.zeros_like(target)
    assert math.isfinite(metric(target, silence, sample_rate=SAMPLE_RATE))
