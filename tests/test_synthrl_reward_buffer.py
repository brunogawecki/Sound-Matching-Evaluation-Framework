"""Tests for the SynthRL per-target reward-based PER buffer (Step 5).

The plan's definitive check: the buffer replaces its minimum-reward entry only when a new
reward exceeds it. Pure numpy, no audio or torch dependency.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.synthrl.reward_buffer import RewardExperience, RewardPrioritizedReplayBuffer


def _action(value: int) -> np.ndarray:
    return np.array([value, value + 1], dtype=np.int64)


def test_fills_up_to_capacity_then_holds_the_best():
    buffer = RewardPrioritizedReplayBuffer(capacity_per_target=3)
    for reward in (0.1, 0.5, 0.3):
        assert buffer.add(0, _action(int(reward * 10)), reward) is True
    assert buffer.size(0) == 3
    rewards = sorted(experience.reward for experience in buffer.entries(0))
    assert rewards == [0.1, 0.3, 0.5]


def test_replaces_minimum_only_when_reward_exceeds_it():
    buffer = RewardPrioritizedReplayBuffer(capacity_per_target=3)
    for reward in (0.1, 0.5, 0.3):
        buffer.add(0, _action(0), reward)

    # Below the current minimum (0.1): rejected, buffer unchanged.
    assert buffer.add(0, _action(9), 0.05) is False
    assert sorted(e.reward for e in buffer.entries(0)) == [0.1, 0.3, 0.5]

    # Above the minimum: accepted, and it is the 0.1 entry that is evicted.
    assert buffer.add(0, _action(9), 0.4) is True
    assert sorted(e.reward for e in buffer.entries(0)) == [0.3, 0.4, 0.5]

    # Equal to the minimum is not "exceeds": rejected.
    assert buffer.add(0, _action(9), 0.3) is False
    assert sorted(e.reward for e in buffer.entries(0)) == [0.3, 0.4, 0.5]


def test_buffers_are_isolated_per_target():
    buffer = RewardPrioritizedReplayBuffer(capacity_per_target=2)
    buffer.add(0, _action(0), 0.9)
    buffer.add(1, _action(1), 0.2)
    assert buffer.size(0) == 1 and buffer.size(1) == 1
    assert 0 in buffer and 1 in buffer and 2 not in buffer
    assert buffer.entries(0)[0].reward == 0.9
    assert buffer.entries(1)[0].reward == 0.2


def test_stored_action_is_copied_not_aliased():
    buffer = RewardPrioritizedReplayBuffer(capacity_per_target=1)
    action = _action(0)
    buffer.add(0, action, 0.5)
    action[0] = 99  # mutate the caller's array after storing
    assert buffer.entries(0)[0].action[0] == 0


def test_sample_returns_stored_experiences():
    buffer = RewardPrioritizedReplayBuffer(capacity_per_target=3)
    for reward in (0.1, 0.5, 0.3):
        buffer.add(0, _action(int(reward * 10)), reward)
    rng = np.random.default_rng(0)

    # Fewer than the buffer -> distinct (without replacement).
    drawn = buffer.sample(0, count=3, rng=rng)
    assert len(drawn) == 3
    assert {e.reward for e in drawn} == {0.1, 0.3, 0.5}

    # More than the buffer -> with replacement, still only stored experiences.
    over = buffer.sample(0, count=5, rng=rng)
    assert len(over) == 5
    assert all(isinstance(e, RewardExperience) for e in over)
    assert all(e.reward in {0.1, 0.3, 0.5} for e in over)


def test_rejects_nonpositive_capacity():
    with pytest.raises(ValueError):
        RewardPrioritizedReplayBuffer(capacity_per_target=0)
