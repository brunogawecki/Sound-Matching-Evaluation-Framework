"""DatasetBuilder: render a PresetSource into an audio corpus (synth-agnostic, Layer 2).

The builder is the bridge between a :class:`dataset.sources.PresetSource` (which
decides *which* presets exist) and a synthesizer wrapper (which renders them).
For each preset it builds the full parameter dict ``defaults <- subset`` (the
subset overrides the synth's defaults; every non-subset parameter stays locked),
renders one sound under the fixed render contract, and writes a WAV plus a row
of metadata.

Rendering goes through a pluggable :class:`RenderExecutor` so the isolation
strategy can change without touching the builder: Issue #4 ships the in-process
:class:`SequentialExecutor`; Issue #5 swaps in ``spawn`` worker pools (training)
and fresh-process-per-render (test) executors.

Output (per run, under ``output_root/<run_name>/``)::

    run_summary.json  # render settings, renderer, seeds, subset, defaults, source
    metadata.csv      # one row per sound: id, path, <subset columns>, provenance, rms
    audio/<id>.wav    # mono float32, one per row

The corpus is a deterministic function of the source's seed: re-running with the
same seed reproduces identical metadata and bit-identical WAVs.
"""
from __future__ import annotations

import abc
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.io import wavfile

import config
from synth.base_synth import BaseSynthesizer
from .sources import PresetRecord, PresetSource


@dataclass(frozen=True)
class RenderSettings:
    """The fixed render contract (note, velocity, durations) for a corpus."""
    midi_note: int
    velocity: int
    duration_sec: float
    note_duration_sec: float

    @classmethod
    def from_config(cls) -> "RenderSettings":
        return cls(
            midi_note=config.MIDI_NOTE,
            velocity=config.VELOCITY,
            duration_sec=config.DURATION_SEC,
            note_duration_sec=config.NOTE_DURATION_SEC,
        )


class RenderExecutor(abc.ABC):
    """Strategy for turning a full parameter dict into rendered mono audio."""

    @abc.abstractmethod
    def render(self, full_params: Dict[str, float]) -> np.ndarray:
        """Render one preset and return 1-D mono audio."""


class SequentialExecutor(RenderExecutor):
    """In-process, single-process executor: one persistent wrapper, rendered in order.

    Correct and deterministic, so the builder is fully testable on its own. The
    training corpus accepts Dexed's reproducible context leak as input noise;
    the clean-room isolation for the test set is Issue #5's job.
    """

    def __init__(self, synth: BaseSynthesizer, render_settings: RenderSettings):
        self._synth = synth
        self._settings = render_settings

    def render(self, full_params: Dict[str, float]) -> np.ndarray:
        self._synth.set_parameters(full_params)
        return self._synth.render_audio(
            self._settings.midi_note,
            self._settings.velocity,
            self._settings.duration_sec,
            self._settings.note_duration_sec,
        )


# Columns written to metadata.csv, in order, around the subset parameter columns.
_LEADING_COLUMNS = ["sample_id", "audio_path"]
_TRAILING_COLUMNS = [
    "method",
    "partition",
    "source_file",
    "voice_index",
    "voice_name",
    "parent_id",
    "rms",
    "near_silent",
]


class DatasetBuilder:
    """Render a PresetSource into a WAV + metadata corpus.

    Args:
        synth: the wrapper supplying defaults, the subset, sample rate and
            renderer name (also the render engine for :class:`SequentialExecutor`).
        render_settings: the render contract; defaults to :meth:`RenderSettings.from_config`.
        executor: rendering strategy; defaults to an in-process
            :class:`SequentialExecutor` over ``synth``.
        near_silence_threshold: peak-amplitude floor below which a render counts
            as near-silent.
        max_redraw_attempts: how many times to ask the source for a louder
            replacement before giving up and storing the near-silent preset.
    """

    def __init__(
        self,
        synth: BaseSynthesizer,
        render_settings: Optional[RenderSettings] = None,
        executor: Optional[RenderExecutor] = None,
        near_silence_threshold: float = 1e-3,
        max_redraw_attempts: int = 10,
    ):
        self._synth = synth
        self._settings = render_settings or RenderSettings.from_config()
        self._executor = executor or SequentialExecutor(synth, self._settings)
        self._near_silence_threshold = float(near_silence_threshold)
        self._max_redraw_attempts = int(max_redraw_attempts)

        self._defaults = synth.get_parameter_defaults()
        self._parameter_space = synth.parameter_space
        self._subset_names = self._parameter_space.names

    def build(
        self,
        source: PresetSource,
        run_name: str,
        output_root: Optional[Path] = None,
    ) -> Dict[str, object]:
        """Render every preset from ``source`` into ``output_root/<run_name>/``.

        Returns the run-summary dict (also written to ``run_summary.json``).
        """
        run_dir = Path(output_root or config.DATASET_DIR) / run_name
        audio_dir = run_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, object]] = []
        for index, preset in enumerate(source.iter_presets()):
            sample_id = f"sample_{index:06d}"
            kept_preset, audio = self._render_with_redraw(source, preset)
            relative_path = f"audio/{sample_id}.wav"
            wavfile.write(str(run_dir / relative_path), self._synth.sample_rate, audio.astype(np.float32))
            rows.append(self._build_metadata_row(sample_id, relative_path, kept_preset, audio))

        df_metadata = pd.DataFrame(rows, columns=_LEADING_COLUMNS + self._subset_names + _TRAILING_COLUMNS)
        df_metadata.to_csv(run_dir / "metadata.csv", index=False)

        run_summary = self._build_run_summary(run_name, source, rows)
        with open(run_dir / "run_summary.json", "w") as run_summary_file:
            json.dump(run_summary, run_summary_file, indent=2)
        return run_summary

    # -- rendering -----------------------------------------------------------
    def _full_params(self, preset: PresetRecord) -> Dict[str, float]:
        extra = set(preset.params) - set(self._subset_names)
        if extra:
            raise KeyError(f"Preset carries non-subset parameters: {sorted(extra)}")
        return {**self._defaults, **preset.params}

    def _render_with_redraw(self, source: PresetSource, preset: PresetRecord):
        attempt = 0
        current = preset
        while True:
            audio = self._executor.render(self._full_params(current))
            if not self._is_near_silent(audio) or attempt >= self._max_redraw_attempts:
                return current, audio
            replacement = source.resample(current, attempt + 1)
            if replacement is None:
                return current, audio
            current, attempt = replacement, attempt + 1

    def _is_near_silent(self, audio: np.ndarray) -> bool:
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        return peak < self._near_silence_threshold

    # -- metadata ------------------------------------------------------------
    def _build_metadata_row(
        self, sample_id: str, relative_path: str, preset: PresetRecord, audio: np.ndarray
    ) -> Dict[str, object]:
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        row: Dict[str, object] = {
            "sample_id": sample_id,
            "audio_path": relative_path,
            "method": preset.method,
            "partition": preset.partition,
            "source_file": preset.source_file,
            "voice_index": preset.voice_index,
            "voice_name": preset.voice_name,
            "parent_id": preset.parent_id,
            "rms": rms,
            "near_silent": self._is_near_silent(audio),
        }
        row.update({name: preset.params[name] for name in self._subset_names})
        return row

    def _build_run_summary(
        self, run_name: str, source: PresetSource, rows: List[Dict[str, object]]
    ) -> Dict[str, object]:
        near_silent_count = sum(1 for row in rows if row["near_silent"])
        method_counts: Dict[str, int] = {}
        for row in rows:
            method_counts[row["method"]] = method_counts.get(row["method"], 0) + 1
        return {
            "run_name": run_name,
            "num_samples": len(rows),
            "near_silent_count": near_silent_count,
            "method_counts": method_counts,
            "render_settings": asdict(self._settings),
            "sample_rate": self._synth.sample_rate,
            "renderer": getattr(self._synth, "renderer_name", None),
            "subset_names": list(self._subset_names),
            "default_params": {name: float(value) for name, value in self._defaults.items()},
            "near_silence_threshold": self._near_silence_threshold,
            "max_redraw_attempts": self._max_redraw_attempts,
            "source": source.describe(),
            "git_revision": _git_revision(),
        }


def _git_revision() -> Optional[str]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(config.BASE_DIR),
            stderr=subprocess.DEVNULL,
        )
        return revision.decode().strip()
    except Exception:
        return None
