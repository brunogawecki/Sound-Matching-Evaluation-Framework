import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

librosa = pytest.importorskip("librosa")

from evaluation.metrics.audio_based import f0_rmse

# ---------------------------------------------------------------------------
# Synthetic-audio tests for the pitch axis. ``f0_rmse`` is a pure callable over
# two raw mono waveforms with the uniform audio signature ``fn(target,
# prediction, *, sample_rate) -> float``. Lower is better; identical inputs
# score 0. It compares only frames voiced in *both* signals and returns nan when
# they never overlap (e.g. against silence).
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
DURATION_SEC = 0.5


def sine(frequency: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * DURATION_SEC)) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * np.pi * frequency * t)).astype(np.float32)


@pytest.fixture
def target() -> np.ndarray:
    return sine(440.0)


@pytest.fixture
def different() -> np.ndarray:
    return sine(880.0)


def test_f0_rmse_is_zero_for_identical_audio(target):
    assert f0_rmse(target, target, sample_rate=SAMPLE_RATE) == pytest.approx(0.0, abs=1e-6)


def test_f0_rmse_is_positive_for_different_pitches(target, different):
    assert f0_rmse(target, different, sample_rate=SAMPLE_RATE) > 0.0


def test_f0_rmse_recovers_the_octave_gap(target, different):
    # A 440 Hz vs 880 Hz pair differs by 440 Hz per voiced frame, so the RMSE
    # lands near 440 Hz (pyin tracks pure tones tightly).
    assert f0_rmse(target, different, sample_rate=SAMPLE_RATE) == pytest.approx(440.0, abs=30.0)


def test_f0_rmse_returns_python_float(target, different):
    assert isinstance(f0_rmse(target, different, sample_rate=SAMPLE_RATE), float)


def test_f0_rmse_is_order_independent(target, different):
    forward = f0_rmse(target, different, sample_rate=SAMPLE_RATE)
    backward = f0_rmse(different, target, sample_rate=SAMPLE_RATE)
    assert forward == pytest.approx(backward)


def test_f0_rmse_is_invariant_to_a_common_gain(target):
    # Pitch is independent of amplitude, so scaling both inputs leaves it at 0.
    assert f0_rmse(2.0 * target, 0.25 * target, sample_rate=SAMPLE_RATE) == pytest.approx(0.0, abs=1e-6)


def test_f0_rmse_is_nan_without_a_commonly_voiced_frame(target):
    # Silence is unvoiced everywhere, so there is no frame voiced in both.
    silence = np.zeros_like(target)
    assert math.isnan(f0_rmse(target, silence, sample_rate=SAMPLE_RATE))
