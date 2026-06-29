import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

librosa = pytest.importorskip("librosa")

from evaluation.metrics.audio_based import (
    lsd,
    spectral_convergence,
    mel_mae,
    mel_mse,
    mss,
)

# ---------------------------------------------------------------------------
# Synthetic-audio tests. Magnitude metrics are pure callables over two raw mono
# waveforms with the uniform audio signature ``fn(target, prediction, *,
# sample_rate) -> float``. All are lower-is-better; identical inputs score 0.
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
DURATION_SEC = 0.5

ALL_METRICS = [lsd, spectral_convergence, mel_mae, mel_mse, mss]


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


# -- spectral convergence: anchored closed-form properties -------------------

def test_spectral_convergence_is_one_when_prediction_is_silent(target):
    # SC = ||S_t - S_p||_F / ||S_t||_F; with S_p == 0 this is exactly 1.
    silence = np.zeros_like(target)
    assert spectral_convergence(target, silence, sample_rate=SAMPLE_RATE) == pytest.approx(1.0)


def test_spectral_convergence_is_asymmetric_via_target_normalization(target):
    # Normalizing by the *target* norm makes SC order-dependent.
    loud = sine(440.0, amplitude=1.0)
    quiet = sine(440.0, amplitude=0.25)
    forward = spectral_convergence(loud, quiet, sample_rate=SAMPLE_RATE)
    backward = spectral_convergence(quiet, loud, sample_rate=SAMPLE_RATE)
    assert forward != pytest.approx(backward)


# -- magnitude metrics are NOT gain-invariant --------------------------------
# Like ``lsd``, the mel metrics use a fixed *absolute* dB floor (``top_db=None``
# disables librosa's default per-signal-max clip; the residual ``amin`` floor is
# absolute). An absolute floor does not track a common gain, so scaling both
# inputs shifts above-floor bins but not floored bins, changing the score. This
# matches the anchors -- preset-gen-vae's ``SimilarityEvaluator`` and InverSynth2
# compare absolute log-magnitudes with a fixed floor and never normalize to each
# signal's own peak (preset-gen-vae explicitly rejected the dynamic-range variant).

@pytest.mark.parametrize("metric", [lsd, mel_mae, mel_mse])
def test_magnitude_metric_is_not_gain_invariant(metric, target, different):
    base = metric(target, different, sample_rate=SAMPLE_RATE)
    scaled = metric(2.0 * target, 2.0 * different, sample_rate=SAMPLE_RATE)
    assert scaled != pytest.approx(base, rel=1e-3)


# -- symmetric metrics -------------------------------------------------------

@pytest.mark.parametrize("metric", [lsd, mel_mae, mel_mse, mss])
def test_symmetric_metrics_are_order_independent(metric, target, different):
    forward = metric(target, different, sample_rate=SAMPLE_RATE)
    backward = metric(different, target, sample_rate=SAMPLE_RATE)
    assert forward == pytest.approx(backward)


# -- finiteness on silence ---------------------------------------------------

@pytest.mark.parametrize("metric", ALL_METRICS)
def test_metric_is_finite_for_silent_inputs(metric, target):
    silence = np.zeros_like(target)
    assert math.isfinite(metric(target, silence, sample_rate=SAMPLE_RATE))
