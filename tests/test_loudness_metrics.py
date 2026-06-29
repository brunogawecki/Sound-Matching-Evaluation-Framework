import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

librosa = pytest.importorskip("librosa")
pyloudnorm = pytest.importorskip("pyloudnorm")

from evaluation.metrics.audio_based import integrated_loudness_error, loudness_envelope_l1

# ---------------------------------------------------------------------------
# Synthetic-audio tests for the loudness axis. Both metrics are pure callables
# over two raw mono waveforms with the uniform audio signature ``fn(target,
# prediction, *, sample_rate) -> float``. Both are lower-is-better; identical
# inputs score 0. Unlike the magnitude/timbre metrics these are *level* metrics:
# a common gain is expected to change them, not cancel.
# ---------------------------------------------------------------------------

# 0.5 s clears one pyloudnorm loudness block (ITU-R BS.1770 gating, 400 ms).
SAMPLE_RATE = 16000
DURATION_SEC = 0.5

ALL_METRICS = [loudness_envelope_l1, integrated_loudness_error]


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


# -- level sensitivity -------------------------------------------------------
# These are loudness metrics, so a gain applied to only one input must register;
# +6 dB doubling the amplitude shifts both an A-weighted contour and the
# integrated LUFS by ~6.02, equally across the signal.

@pytest.mark.parametrize("metric", ALL_METRICS)
def test_metric_detects_a_one_sided_gain(metric, target):
    louder = 2.0 * target
    assert metric(target, louder, sample_rate=SAMPLE_RATE) == pytest.approx(20.0 * math.log10(2.0), abs=0.2)


# -- finiteness on silence ---------------------------------------------------
# Integrated loudness of silence is -inf (gated); the floor keeps it finite. The
# envelope falls to its log floor. Either way the error stays finite.

@pytest.mark.parametrize("metric", ALL_METRICS)
def test_metric_is_finite_for_silent_inputs(metric, target):
    silence = np.zeros_like(target)
    assert math.isfinite(metric(target, silence, sample_rate=SAMPLE_RATE))
