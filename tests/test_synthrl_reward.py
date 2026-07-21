"""Tests for the SynthRL sound-matching reward (Step 5).

The plan's definitive check: reward(target, target) > reward(target, random). Also
covers monotonicity (a closer prediction scores higher), the finite perfect-match value,
and the silent-target floor. Uses pure numpy waveforms; needs ``librosa`` (the metric
callables' dependency), so it skips cleanly when absent.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

pytest.importorskip("librosa")  # the reward reuses the librosa-backed metric callables

from models.synthrl.reward import RewardWeights, sound_matching_reward

SAMPLE_RATE = 8000
DURATION_SEC = 0.5


def _tone(frequency: float, amplitude: float = 0.8) -> np.ndarray:
    samples = int(DURATION_SEC * SAMPLE_RATE)
    time = np.arange(samples) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)


def test_perfect_match_beats_a_random_prediction():
    target = _tone(220.0)
    random_prediction = np.random.default_rng(0).standard_normal(target.shape).astype(np.float32)

    perfect = sound_matching_reward(target, target, sample_rate=SAMPLE_RATE)
    against_random = sound_matching_reward(target, random_prediction, sample_rate=SAMPLE_RATE)

    assert perfect > against_random


def test_perfect_match_is_finite_and_large():
    target = _tone(220.0)
    reward = sound_matching_reward(target, target, sample_rate=SAMPLE_RATE)
    assert np.isfinite(reward)
    assert reward > 0.0


def test_closer_prediction_scores_higher():
    target = _tone(220.0)
    close = _tone(225.0)   # a few Hz off
    far = _tone(440.0)     # an octave off
    reward_close = sound_matching_reward(target, close, sample_rate=SAMPLE_RATE)
    reward_far = sound_matching_reward(target, far, sample_rate=SAMPLE_RATE)
    assert reward_close > reward_far


def test_silent_target_returns_the_floor():
    silent_target = np.zeros(int(DURATION_SEC * SAMPLE_RATE), dtype=np.float32)
    prediction = _tone(220.0)
    # Spectral convergence is undefined against a silent target (nan) -> reward floors at 0.
    assert sound_matching_reward(silent_target, prediction, sample_rate=SAMPLE_RATE) == 0.0


def test_weights_reweight_the_terms():
    target = _tone(220.0)
    prediction = _tone(440.0)
    default_reward = sound_matching_reward(target, prediction, sample_rate=SAMPLE_RATE)
    # Down-weighting every term shrinks the denominator, so the reward rises.
    lighter = RewardWeights(spectrogram=0.027, spectral_convergence=0.07, mfcc=0.003)
    lighter_reward = sound_matching_reward(target, prediction, sample_rate=SAMPLE_RATE, weights=lighter)
    assert lighter_reward > default_reward
