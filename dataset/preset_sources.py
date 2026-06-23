"""Preset sources: where a dataset's presets come from (Layer 2, synth-agnostic).

A PresetSource yields PresetRecords (a synth-side subset dict plus provenance)
for the builder to render; it is the only place that decides *which* presets
exist. Three strategies produce the same record stream: SyntheticPresetSource
(random draws), HumanPresetSource (real presets projected onto the subset), and
HybridPresetSource (blend or augment the two).

Draws are deterministic in a master ``seed`` via per-slot ``SeedSequence``
streams: a preset at ``slot`` is a pure function of ``(seed, slot, attempt)``,
so output is order-independent and safe to render out of order.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field, replace
from typing import Dict, Iterator, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from synth.parameter_space import ParameterSpace

if TYPE_CHECKING:
    from .dexed_preset_loader import LoadedPreset

# Record-level provenance tags (the dataset-level method lives in the run summary).
METHOD_SYNTHETIC = "synthetic"
METHOD_HUMAN = "human"
METHOD_AUGMENT = "augment"


@dataclass(frozen=True)
class PresetRecord:
    """One preset to render: a subset dict plus where it came from.

    ``params`` keys must equal the parameter space's subset names exactly. The
    provenance fields are written to the dataset's ``metadata.csv``; ``slot`` is
    internal bookkeeping (the deterministic-resample key) and is never stored.
    """
    params: Dict[str, float]
    method: str
    partition: str
    source_file: Optional[str] = None
    voice_index: Optional[int] = None
    voice_name: Optional[str] = None
    parent_id: Optional[str] = None
    slot: Optional[int] = field(default=None, compare=False)


class PresetSource(abc.ABC):
    """A finite, deterministic stream of presets to render."""

    @abc.abstractmethod
    def iter_presets(self) -> Iterator[PresetRecord]:
        """Yield the presets of this source, in a stable order."""

    def resample(self, record: PresetRecord, attempt: int) -> Optional[PresetRecord]:
        """Return a replacement for a near-silent ``record``, or ``None``.

        Generative sources (synthetic / augment) return a fresh deterministic
        draw for the same slot; sources backed by fixed human presets cannot
        resample and return ``None`` (the builder keeps the preset and flags it).
        """
        return None

    @abc.abstractmethod
    def describe(self) -> Dict[str, object]:
        """A JSON-serializable description of this source for the run summary."""


def _rng(seed: int, *tags: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), *(int(tag) for tag in tags)]))


class SyntheticPresetSource(PresetSource):
    """Random presets over the parameter space (the "synthetic" method).

    ``sampling_ranges`` optionally draws named continuous params from a narrow
    sub-range at sampling time (e.g. pinning a carrier loud, via the synth's
    ``audible_sampling_ranges``); empty/omitted is plain uniform sampling. The
    builder still redraws any residual near-silent preset via :meth:`resample`.
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        count: int,
        seed: int,
        partition: str = "train",
        sampling_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self._parameter_space = parameter_space
        self._count = int(count)
        self._seed = int(seed)
        self._partition = partition
        self._sampling_ranges = dict(sampling_ranges or {})

    def _sample(self, slot: int, attempt: int) -> Dict[str, float]:
        rng = _rng(self._seed, slot, attempt)
        return self._parameter_space.sample_constrained(rng, self._sampling_ranges)

    def iter_presets(self) -> Iterator[PresetRecord]:
        for slot in range(self._count):
            yield PresetRecord(
                params=self._sample(slot, 0),
                method=METHOD_SYNTHETIC,
                partition=self._partition,
                slot=slot,
            )

    def resample(self, record: PresetRecord, attempt: int) -> Optional[PresetRecord]:
        return replace(record, params=self._sample(record.slot, attempt))

    def describe(self) -> Dict[str, object]:
        return {
            "method": METHOD_SYNTHETIC,
            "count": self._count,
            "seed": self._seed,
            "partition": self._partition,
            "sampling_ranges": dict(self._sampling_ranges),
        }


class HumanPresetSource(PresetSource):
    """Human-made presets projected onto the estimated subset (the "human" method).

    Presets are pre-loaded, deduplicated and split by a synth-specific loader
    (e.g. :class:`dataset.dexed_preset_loader.DexedPresetLoader`); this source
    projects each onto the subset and tags the partition. Dropped parameters fall
    back to the synth defaults at render time (near-lossless; see DECISIONS.md D1).
    """

    def __init__(
        self,
        presets: List[LoadedPreset],
        parameter_space: ParameterSpace,
        partition: str,
    ):
        self._presets = list(presets)
        self._parameter_space = parameter_space
        self._partition = partition

    def _extract_subset_parameters(self, params: Dict[str, float]) -> Dict[str, float]:
        missing = [name for name in self._parameter_space.names if name not in params]
        if missing:
            raise KeyError(f"Preset is missing subset parameters: {missing}")
        return {name: float(params[name]) for name in self._parameter_space.names}

    def iter_presets(self) -> Iterator[PresetRecord]:
        for preset in self._presets:
            yield PresetRecord(
                params=self._extract_subset_parameters(preset.params),
                method=METHOD_HUMAN,
                partition=self._partition,
                source_file=preset.source_file,
                voice_index=preset.voice_index,
                voice_name=preset.voice_name,
            )

    def describe(self) -> Dict[str, object]:
        return {
            "method": METHOD_HUMAN,
            "count": len(self._presets),
            "partition": self._partition,
        }


class HybridPresetSource(PresetSource):
    """Combine human-train presets with synthetic material (the "hybrid" method).

    Two construction modes:

    * ``"blend"`` -- each slot is, with probability ``synthetic_ratio``, a fresh
      synthetic draw; otherwise a randomly chosen human-train preset.
    * ``"augment"`` -- each slot perturbs a randomly chosen human-train preset:
      ``num_perturbed_params`` parameters are jittered (continuous) or flipped
      (categorical, only if ``flip_categoricals``), tagged with its parent.

    Material derives only from the human **train** partition, so it never leaks
    the held-out human test set into training.
    """

    BLEND = "blend"
    AUGMENT = "augment"

    def __init__(
        self,
        mode: str,
        human_presets: List[PresetRecord],
        parameter_space: ParameterSpace,
        count: int,
        seed: int,
        *,
        synthetic_ratio: float = 0.5,
        num_perturbed_params: int = 2,
        jitter: float = 0.05,
        flip_categoricals: bool = False,
        partition: str = "train",
        sampling_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        if mode not in (self.BLEND, self.AUGMENT):
            raise ValueError(
                f"Unknown hybrid mode '{mode}'; expected '{self.BLEND}' or '{self.AUGMENT}'."
            )
        if not human_presets:
            raise ValueError("HybridPresetSource requires at least one human (train) preset.")
        self._mode = mode
        self._human_presets = list(human_presets)
        self._parameter_space = parameter_space
        self._count = int(count)
        self._seed = int(seed)
        self._synthetic_ratio = float(synthetic_ratio)
        self._num_perturbed_params = int(num_perturbed_params)
        self._jitter = float(jitter)
        self._flip_categoricals = bool(flip_categoricals)
        self._partition = partition
        self._sampling_ranges = dict(sampling_ranges or {})

        specs = parameter_space.parameter_specs
        self._continuous_names = [spec.name for spec in specs if spec.kind == "continuous"]
        self._categorical_names = [spec.name for spec in specs if spec.kind == "categorical"]
        self._spec_by_name = {spec.name: spec for spec in specs}

    # -- blend ---------------------------------------------------------------
    def _blend_slot(self, slot: int, attempt: int) -> PresetRecord:
        if _rng(self._seed, slot, 0).random() < self._synthetic_ratio:
            return PresetRecord(
                params=self._parameter_space.sample_constrained(
                    _rng(self._seed, slot, 1, attempt), self._sampling_ranges
                ),
                method=METHOD_SYNTHETIC,
                partition=self._partition,
                slot=slot,
            )
        parent = self._human_presets[int(_rng(self._seed, slot, 2).integers(len(self._human_presets)))]
        return replace(parent, partition=self._partition, slot=slot)

    # -- augment -------------------------------------------------------------
    def _parent_for(self, slot: int) -> PresetRecord:
        return self._human_presets[int(_rng(self._seed, slot, 0).integers(len(self._human_presets)))]

    def _perturb(self, params: Dict[str, float], rng: np.random.Generator) -> Dict[str, float]:
        pool = list(self._continuous_names)
        if self._flip_categoricals:
            pool += self._categorical_names
        count = min(self._num_perturbed_params, len(pool))
        chosen = rng.choice(pool, size=count, replace=False) if count else []

        perturbed = dict(params)
        for name in chosen:
            spec = self._spec_by_name[name]
            if spec.kind == "continuous":
                low, high = spec.bounds
                perturbed[name] = float(
                    np.clip(params[name] + rng.uniform(-self._jitter, self._jitter), low, high)
                )
            else:
                alternatives = [option for option in spec.options if option != params[name]]
                if alternatives:
                    perturbed[name] = float(rng.choice(alternatives))
        return perturbed

    def _augment_slot(self, slot: int, attempt: int) -> PresetRecord:
        parent = self._parent_for(slot)
        return PresetRecord(
            params=self._perturb(parent.params, _rng(self._seed, slot, 1, attempt)),
            method=METHOD_AUGMENT,
            partition=self._partition,
            source_file=parent.source_file,
            voice_index=parent.voice_index,
            voice_name=parent.voice_name,
            parent_id=_parent_id(parent),
            slot=slot,
        )

    def _build_slot(self, slot: int, attempt: int) -> PresetRecord:
        if self._mode == self.BLEND:
            return self._blend_slot(slot, attempt)
        return self._augment_slot(slot, attempt)

    def iter_presets(self) -> Iterator[PresetRecord]:
        for slot in range(self._count):
            yield self._build_slot(slot, 0)

    def resample(self, record: PresetRecord, attempt: int) -> Optional[PresetRecord]:
        # A blended human pick is a fixed preset and cannot be redrawn.
        if record.method == METHOD_HUMAN:
            return None
        return self._build_slot(record.slot, attempt)

    def describe(self) -> Dict[str, object]:
        summary: Dict[str, object] = {
            "method": "hybrid",
            "mode": self._mode,
            "count": self._count,
            "seed": self._seed,
            "num_human_parents": len(self._human_presets),
            "partition": self._partition,
        }
        if self._mode == self.BLEND:
            summary["synthetic_ratio"] = self._synthetic_ratio
            summary["sampling_ranges"] = dict(self._sampling_ranges)
        else:
            summary["num_perturbed_params"] = self._num_perturbed_params
            summary["jitter"] = self._jitter
            summary["flip_categoricals"] = self._flip_categoricals
        return summary


def _parent_id(parent: PresetRecord) -> str:
    """A stable, run-independent identifier for an augmented preset's parent."""
    return f"{parent.source_file}:{parent.voice_index}"
