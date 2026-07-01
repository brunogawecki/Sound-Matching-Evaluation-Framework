"""Dexed preset loader: turn ``.syx`` cartridges into a train/test preset split.

The synth-specific half of the human-preset pipeline. It loads each DX7 voice
(a ``.syx`` is a 32-voice bulk dump) as a :class:`LoadedPreset`, deduplicates
near-twins on their subset projection (so presets that render identically
collapse to one), and makes a seeded voice-level train/test split so the two
partitions are provably disjoint.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from tqdm import tqdm

from synth.parameter_space import ParameterSpace
from synth.dexed.cartridge import NUM_VOICES, voice_names, voice_parameters


@dataclass(frozen=True)
class LoadedPreset:
    """One human preset: its full unpacked parameters plus provenance."""
    params: Dict[str, float]
    source_file: str
    voice_index: int
    voice_name: str


@dataclass(frozen=True)
class PresetSplit:
    """Disjoint train / test partitions of deduplicated human presets."""
    train: List[LoadedPreset]
    test: List[LoadedPreset]


# -- shared dedup / split (synth-agnostic; reused by every human-preset loader) --

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


def split_presets(
    presets: List[LoadedPreset],
    test_fraction: float,
    split_seed: int = 0,
) -> PresetSplit:
    """Seeded voice-level train/test split; the two partitions are disjoint by
    construction (a preset is in exactly one)."""
    order = np.random.default_rng(split_seed).permutation(len(presets))
    num_test = int(round(len(presets) * test_fraction))
    test_positions = set(order[:num_test].tolist())
    train = [preset for index, preset in enumerate(presets) if index not in test_positions]
    test = [preset for index, preset in enumerate(presets) if index in test_positions]
    return PresetSplit(train=train, test=test)


class DexedPresetLoader:
    """Load, deduplicate and split DX7 ``.syx`` cartridges into human presets.

    Args:
        parameter_space: the estimated subset; presets are projected onto it for
            deduplication (so dedup sees what actually gets rendered).
        test_fraction: share of surviving voices held out for the test set
            (default 0.0 -- all presets go to train; raise to hold out a
            seeded, disjoint test set).
        split_seed: seed for the voice-level train/test shuffle.
        dedup_threshold: max-norm distance between projected ML vectors below
            which two presets are duplicates. Default catches exact twins and
            float noise; raise to cluster close presets (see docs/DECISIONS.md).
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        test_fraction: float = 0.0,
        split_seed: int = 0,
        dedup_threshold: float = 1e-3,
    ):
        if not 0.0 <= test_fraction <= 1.0:
            raise ValueError(f"test_fraction must be in [0, 1], got {test_fraction}.")
        self._parameter_space = parameter_space
        self._test_fraction = float(test_fraction)
        self._split_seed = int(split_seed)
        self._dedup_threshold = float(dedup_threshold)

    def load(self, syx_paths: Sequence[str], show_progress: bool = False) -> PresetSplit:
        """Load every voice from the given cartridges, deduplicate, and split into train/test."""
        presets = self._load_presets_from_cartridges(syx_paths)
        kept = deduplicate_presets(
            presets, self._parameter_space, self._dedup_threshold, show_progress=show_progress
        )
        return split_presets(kept, self._test_fraction, self._split_seed)

    # -- loading -------------------------------------------------------------
    def _load_presets_from_cartridges(self, syx_paths: Sequence[str]) -> List[LoadedPreset]:
        presets: List[LoadedPreset] = []
        for path in syx_paths:
            with open(path, "rb") as syx_file:
                data = syx_file.read()
            names = voice_names(data)  # validates the cartridge
            source_file = os.path.basename(path)
            for voice_index in range(NUM_VOICES):
                presets.append(
                    LoadedPreset(
                        params=voice_parameters(data, voice_index),
                        source_file=source_file,
                        voice_index=voice_index,
                        voice_name=names[voice_index].rstrip(),
                    )
                )
        return presets
