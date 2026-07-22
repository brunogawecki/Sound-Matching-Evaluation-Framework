"""The SynthRL sound-matching reward (paper §3.4, Eqs. 2-5).

``R(y, y_hat) = 1 / clamp(w1*Spec + w2*SC + w3*MFCC, 0.1, 5.0)`` over a target ``y`` and
the rendered prediction ``y_hat``: higher reward means a closer audio match. The paper
clamps the weighted distance before inverting (repo ``model/loss.py``:
``1 / clamp(., 0.1, 5.0)``), bounding the reward to ``[0.2, 10.0]`` so a near-perfect
match cannot blow the reward up and a very poor one still gets the floor. The three
distance terms are the framework's existing metric callables, so the RL reward and the
evaluation panel measure sound similarity the same way (single source of truth):

  * ``Spec`` = ``lsd`` -- log-STFT L1 distance.
  * ``SC``   = ``spectral_convergence``.
  * ``MFCC`` = ``mfcc_mae`` -- 13-band MFCC MAE (``N_MFCC=13`` already matches the paper).

The paper writes ``Spec`` as an ln/sum over the STFT while ``lsd`` is a log10/mean; the
two differ only by a positive constant that folds into ``w1``, so the paper's weights are
kept as the documented default (``weights`` is the knob if they need retuning).

The reward is a black-box scalar -- no gradient flows through it, which is all REINFORCE
needs. It never touches a live synth; the caller renders ``y_hat`` first.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from evaluation.metrics.audio_based import lsd, mfcc_mae, spectral_convergence


@dataclass(frozen=True)
class RewardWeights:
    """The three reward-term weights (paper Eq. 5 defaults)."""

    spectrogram: float = 0.27
    spectral_convergence: float = 0.7
    mfcc: float = 0.03


DEFAULT_REWARD_WEIGHTS = RewardWeights()
# The paper clamps the weighted distance to [min, max] before inverting, bounding the
# reward to [1/max, 1/min] = [0.2, 10.0]. The lower bound also keeps the reward finite at
# a perfect match (distance 0 -> clamped to 0.1).
REWARD_DENOMINATOR_MIN = 0.1
REWARD_DENOMINATOR_MAX = 5.0


def sound_matching_reward(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    sample_rate: int,
    weights: RewardWeights = DEFAULT_REWARD_WEIGHTS,
) -> float:
    """Reward for one ``(target, rendered-prediction)`` pair; higher is a closer match.

    Returns the floor reward ``1 / REWARD_DENOMINATOR_MAX`` (0.2, the worst clamped value)
    when any distance term is undefined (``nan`` / non-finite) -- e.g. a silent target
    makes spectral convergence undefined.
    """
    weighted_distance = (
        weights.spectrogram * lsd(target, prediction, sample_rate=sample_rate)
        + weights.spectral_convergence
        * spectral_convergence(target, prediction, sample_rate=sample_rate)
        + weights.mfcc * mfcc_mae(target, prediction, sample_rate=sample_rate)
    )
    if not np.isfinite(weighted_distance):
        return float(1.0 / REWARD_DENOMINATOR_MAX)
    clamped_distance = min(REWARD_DENOMINATOR_MAX, max(REWARD_DENOMINATOR_MIN, weighted_distance))
    return float(1.0 / clamped_distance)
