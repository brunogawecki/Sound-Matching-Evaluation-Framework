"""Dexed preset loader: turn ``.syx`` cartridges into a train/test preset split.

This is the synth-specific half of the human-preset pipeline. It hides the
cartridge-vs-single-file difference behind a flat stream of one-preset-per-item,
deduplicates near-twins, and splits voices into train / test partitions. The
synth-agnostic :class:`dataset.preset_sources.PresetSource` then projects each preset
onto the estimated subset.

Three responsibilities (all synth-specific, all here):

* **Loading voices** -- a DX7 ``.syx`` is a 32-voice bulk dump; each of its
  voices becomes one :class:`LoadedPreset` (via ``synth/dexed/cartridge.py``).
* **Deduplication** -- a load-bearing pass: the train/test split is honest only
  if no near-duplicate voice straddles it. Duplicates are compared on the
  *subset projection* (one-hot ML vector), so two presets that differ only in
  dropped parameters -- and would therefore render identically under the fixed
  render contract -- collapse to one.
* **Voice-level split** -- a seeded random 80/20 split over surviving voices
  (not whole cartridges), so train and test are provably disjoint.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

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


class DexedPresetLoader:
    """Load, deduplicate and split DX7 ``.syx`` cartridges into human presets.

    Args:
        parameter_space: the estimated subset; used to project presets for
            deduplication (so dedup sees what actually gets rendered).
        test_fraction: share of surviving voices held out for the test set.
        split_seed: seed for the voice-level train/test shuffle.
        dedup_threshold: two presets are duplicates if every component of their
            projected ML vectors differs by at most this much (max-norm). The
            default catches exact twins and float noise; raise it to cluster
            perceptually-close presets (a tuning knob -- see docs/DECISIONS.md).
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        test_fraction: float = 0.20,
        split_seed: int = 0,
        dedup_threshold: float = 1e-3,
    ):
        if not 0.0 <= test_fraction <= 1.0:
            raise ValueError(f"test_fraction must be in [0, 1], got {test_fraction}.")
        self._parameter_space = parameter_space
        self._test_fraction = float(test_fraction)
        self._split_seed = int(split_seed)
        self._dedup_threshold = float(dedup_threshold)

    def load(self, syx_paths: Sequence[str]) -> PresetSplit:
        """Load every voice from the given cartridges, deduplicate, and split into train/test."""
        presets = self._load_presets_from_cartridges(syx_paths)
        kept = self._deduplicate(presets)
        return self._split(kept)

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

    # -- deduplication -------------------------------------------------------
    def _projected_vector(self, preset: LoadedPreset) -> np.ndarray:
        subset = {name: preset.params[name] for name in self._parameter_space.names}
        return self._parameter_space.synth_dict_to_ml_vector(subset)

    def _deduplicate(self, presets: List[LoadedPreset]) -> List[LoadedPreset]:
        kept: List[LoadedPreset] = []
        kept_vectors: List[np.ndarray] = []
        for preset in presets:
            vector = self._projected_vector(preset)
            if any(
                np.max(np.abs(vector - other)) <= self._dedup_threshold
                for other in kept_vectors
            ):
                continue
            kept.append(preset)
            kept_vectors.append(vector)
        return kept

    # -- split ---------------------------------------------------------------
    def _split(self, presets: List[LoadedPreset]) -> PresetSplit:
        order = np.random.default_rng(self._split_seed).permutation(len(presets))
        num_test = int(round(len(presets) * self._test_fraction))
        test_positions = set(order[:num_test].tolist())
        train = [preset for index, preset in enumerate(presets) if index not in test_positions]
        test = [preset for index, preset in enumerate(presets) if index in test_positions]
        return PresetSplit(train=train, test=test)
