"""Render a PresetSource into a WAV + metadata corpus (Layer 2, synth-agnostic).

For each preset, merges the subset over the synth defaults, renders one sound
under the fixed render contract, and writes a WAV plus a metadata row. Per run,
writes run_summary.json, metadata.csv, and audio/<id>.wav under
output_root/<run_name>/. The corpus is a deterministic function of the source's
seed (identical metadata and bit-identical WAVs on re-run).
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyloudnorm
from scipy.io import wavfile
from tqdm import tqdm

import config
from synth.base_synth import BaseSynthesizer
from .render_backends import InProcessRenderBackend, RenderSettings
from .preset_sources import PresetRecord, PresetSource


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
    "loudness_lufs",
    "near_silent",
]


class DatasetBuilder:
    """Render a PresetSource into a WAV + metadata corpus.

    Args:
        synth: wrapper supplying defaults, subset, sample rate and renderer.
        render_settings: the render contract; defaults to RenderSettings.from_config().
        min_loudness_lufs: integrated-loudness floor (LUFS) below which a render
            counts as near-silent and triggers a redraw. Default is calibrated to
            the built-in Dexed presets; recalibrate per synth (see D-AUDIBLE).
        max_redraw_attempts: redraw attempts before storing a near-silent preset.
        render_backend: where renders run. Defaults to an in-process backend reusing
            ``synth`` (fast; the training path). Pass a FreshProcessRenderBackend to
            render each preset in a clean spawned process for test/eval corpora (D-REPRO).
    """

    def __init__(
        self,
        synth: BaseSynthesizer,
        render_settings: Optional[RenderSettings] = None,
        min_loudness_lufs: float = -34.0,
        max_redraw_attempts: int = 10,
        render_backend=None,
    ):
        self._synth = synth
        self._settings = render_settings or RenderSettings.from_config()
        self._backend = render_backend or InProcessRenderBackend(synth, self._settings)
        self._min_loudness_lufs = float(min_loudness_lufs)
        self._max_redraw_attempts = int(max_redraw_attempts)
        self._loudness_meter = pyloudnorm.Meter(int(synth.sample_rate))

        self._defaults = synth.get_parameter_defaults()
        self._parameter_space = synth.parameter_space
        self._subset_names = self._parameter_space.names

    def build(
        self,
        source: PresetSource,
        run_name: str,
        output_root: Optional[Path] = None,
        show_progress: bool = False,
    ) -> Dict[str, object]:
        """Render every preset from ``source`` into ``output_root/<run_name>/``.

        Pass ``show_progress=True`` to draw a tqdm bar over the render loop.

        Returns the run-summary dict (also written to ``run_summary.json``).
        """
        run_dir = Path(output_root or config.DATASET_DIR) / run_name
        audio_dir = run_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        total = source.describe().get("count")
        rows: List[Dict[str, object]] = []
        try:
            presets = tqdm(
                source.iter_presets(), total=total, desc="Rendering",
                unit="preset", disable=not show_progress,
            )
            for index, preset in enumerate(presets):
                sample_id = f"sample_{index:06d}"
                kept_preset, audio, loudness = self._render_with_redraw(source, preset)
                relative_path = f"audio/{sample_id}.wav"
                wavfile.write(str(run_dir / relative_path), self._synth.sample_rate, audio.astype(np.float32))
                rows.append(self._build_metadata_row(sample_id, relative_path, kept_preset, audio, loudness))
        finally:
            self._backend.close()

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

    def _render_with_redraw(
        self, source: PresetSource, preset: PresetRecord
    ) -> Tuple[PresetRecord, np.ndarray, float]:
        """Render ``preset``, redrawing near-silent results until audible or capped."""
        attempt = 0
        current = preset
        while True:
            audio = self._backend.render(self._full_params(current))
            loudness = self._integrated_loudness(audio)
            if loudness >= self._min_loudness_lufs or attempt >= self._max_redraw_attempts:
                return current, audio, loudness
            replacement = source.resample(current, attempt + 1)
            if replacement is None:
                return current, audio, loudness
            current, attempt = replacement, attempt + 1

    def _integrated_loudness(self, audio: np.ndarray) -> float:
        """Integrated loudness in LUFS (-inf for silence); gates out the release tail."""
        if audio.size == 0 or not np.any(audio):
            return float("-inf")
        return float(self._loudness_meter.integrated_loudness(audio))

    # -- metadata ------------------------------------------------------------
    def _build_metadata_row(
        self, sample_id: str, relative_path: str, preset: PresetRecord,
        audio: np.ndarray, loudness_lufs: float,
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
            "loudness_lufs": loudness_lufs,
            "near_silent": loudness_lufs < self._min_loudness_lufs,
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
            "render_process": getattr(self._backend, "process_mode", "in-process"),
            "sample_rate": self._synth.sample_rate,
            "renderer": getattr(self._synth, "renderer_name", None),
            "subset_names": list(self._subset_names),
            "parameter_space": self._parameter_space.to_dict(),
            "default_params": {name: float(value) for name, value in self._defaults.items()},
            "min_loudness_lufs": self._min_loudness_lufs,
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
