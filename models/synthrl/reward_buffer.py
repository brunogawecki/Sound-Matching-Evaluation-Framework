"""Per-target reward-based prioritized experience replay for the SynthRL RL stage.

SynthRL Algorithm 1: each target (a corpus index) keeps a bounded buffer of its
``capacity_per_target`` highest-reward experiences, where an experience is a
``(sampled class-index action, reward)`` pair. A new experience is stored only if the
buffer has spare room, or if its reward exceeds the buffer's current minimum -- in which
case it replaces that minimum. The RL loss samples experiences from these buffers (the
paper's uniform ``1/m`` importance weight) to form the REINFORCE objective, so the policy
keeps being trained on the best actions found so far for each target.

Representation-agnostic: an action is just the per-parameter class-index vector the policy
sampled, so the buffer never needs the network, the synth, or the ParameterSpace.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class RewardExperience:
    """One stored experience: the sampled class-index action and the reward it earned."""

    action: np.ndarray  # [num_parameters] integer class indices
    reward: float


class RewardPrioritizedReplayBuffer:
    """Per-target buffers holding the top ``capacity_per_target`` experiences by reward."""

    def __init__(self, capacity_per_target: int):
        if capacity_per_target < 1:
            raise ValueError("capacity_per_target must be >= 1")
        self._capacity = capacity_per_target
        self._buffers: Dict[int, List[RewardExperience]] = {}

    def add(self, target_index: int, action: np.ndarray, reward: float) -> bool:
        """Store ``(action, reward)`` for ``target_index``; return whether it was kept.

        Kept when the buffer has spare capacity, or when ``reward`` exceeds the buffer's
        current minimum reward (then it replaces that minimum entry). Otherwise dropped.
        """
        experience = RewardExperience(np.asarray(action).copy(), float(reward))
        buffer = self._buffers.setdefault(target_index, [])
        if len(buffer) < self._capacity:
            buffer.append(experience)
            return True
        minimum_position = min(range(len(buffer)), key=lambda position: buffer[position].reward)
        if experience.reward > buffer[minimum_position].reward:
            buffer[minimum_position] = experience
            return True
        return False

    def sample(
        self, target_index: int, count: int, rng: np.random.Generator
    ) -> List[RewardExperience]:
        """Uniformly sample ``count`` experiences for ``target_index``.

        Without replacement when the buffer holds at least ``count`` experiences, with
        replacement otherwise (early in training, before a target's buffer has filled).
        """
        buffer = self._buffers[target_index]
        positions = rng.choice(len(buffer), size=count, replace=count > len(buffer))
        return [buffer[int(position)] for position in positions]

    def entries(self, target_index: int) -> List[RewardExperience]:
        """All experiences currently stored for ``target_index`` (empty if none)."""
        return list(self._buffers.get(target_index, []))

    def size(self, target_index: int) -> int:
        return len(self._buffers.get(target_index, []))

    def __contains__(self, target_index: int) -> bool:
        return self.size(target_index) > 0
