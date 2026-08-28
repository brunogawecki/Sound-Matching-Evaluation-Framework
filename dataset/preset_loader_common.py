"""Synth-agnostic half of the human-preset pipeline: what a loaded preset is, and how a
collection of them is deduplicated and split.

Every human-preset loader (`dexed_preset_loader`, `dexed_sqlite_preset_loader`,
`diva_preset_loader`) reads a different file format but then does the same three things:
wrap each preset as a :class:`LoadedPreset`, drop near-twins on their subset projection, and
make a seeded train/test split. Only the reading is synth-specific, so only the reading lives
in the per-synth modules.

Nothing here touches a synthesizer or a plugin. It works on ``Dict[str, float]`` parameter
dicts and a :class:`~synth.parameter_space.ParameterSpace`, so it is as portable as those are.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

from synth.parameter_space import ParameterSpace


@dataclass(frozen=True)
class LoadedPreset:
    """One human preset: its full unpacked parameters plus provenance.

    ``voice_index`` is the preset's position within ``source_file`` for formats that pack
    several presets per file (a DX7 ``.syx`` holds 32); it is 0 for one-preset-per-file
    formats.
    """
    params: Dict[str, float]
    source_file: str
    voice_index: int
    voice_name: str


@dataclass(frozen=True)
class PresetSplit:
    """Disjoint train / test partitions of deduplicated human presets."""
    train: List[LoadedPreset]
    test: List[LoadedPreset]


def _projected_vector(preset: LoadedPreset, parameter_space: ParameterSpace) -> np.ndarray:
    """The preset's ML vector on the estimated subset (what actually gets rendered)."""
    subset = {name: preset.params[name] for name in parameter_space.names}
    return parameter_space.synth_dict_to_ml_vector(subset)


def deduplicate_presets(
    presets: List[LoadedPreset],
    parameter_space: ParameterSpace,
    dedup_threshold: float = 1e-3,
    show_progress: bool = False,
) -> List[LoadedPreset]:
    """Drop near-twins: any preset whose subset projection is within
    ``dedup_threshold`` (max-norm) of one already kept. Presets that render
    identically under the fixed contract collapse to a single representative.

    This is O(n^2) in the number of presets; pass ``show_progress=True`` to draw a
    tqdm bar (the scan is silent and slow on the full ~30k-voice collection).
    """
    kept: List[LoadedPreset] = []
    kept_vectors: List[np.ndarray] = []
    for preset in tqdm(presets, desc="Deduplicating", unit="preset", disable=not show_progress):
        vector = _projected_vector(preset, parameter_space)
        if any(
            np.max(np.abs(vector - other)) <= dedup_threshold
            for other in kept_vectors
        ):
            continue
        kept.append(preset)
        kept_vectors.append(vector)
    return kept


def split_indices(
    count: int,
    test_fraction: float,
    split_seed: int = 0,
) -> Tuple[List[int], List[int]]:
    """Seeded train/test partition of ``range(count)`` positions; the two lists are
    disjoint by construction (a position is in exactly one) and each stays in
    ascending order. The single source of truth for how a set is split into
    train/test, shared by :func:`split_presets` and the post-render corpus split."""
    order = np.random.default_rng(split_seed).permutation(count)
    num_test = int(round(count * test_fraction))
    test_positions = set(order[:num_test].tolist())
    train = [index for index in range(count) if index not in test_positions]
    test = [index for index in range(count) if index in test_positions]
    return train, test


def split_presets(
    presets: List[LoadedPreset],
    test_fraction: float,
    split_seed: int = 0,
) -> PresetSplit:
    """Seeded voice-level train/test split; the two partitions are disjoint by
    construction (a preset is in exactly one)."""
    train_positions, test_positions = split_indices(len(presets), test_fraction, split_seed)
    train = [presets[index] for index in train_positions]
    test = [presets[index] for index in test_positions]
    return PresetSplit(train=train, test=test)
