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
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyloudnorm
from scipy.io import wavfile
from tqdm import tqdm

import config
from synth.base_synth import BaseSynthesizer
from synth.parameter_space import ParameterSpace
from .render_backends import DEFAULT_SYNTH, InProcessRenderBackend, RenderSettings, RenderTimeoutError
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
        default_params: values to lock the non-subset parameters at, overriding the
            synth's own init patch. Needed when a preset source froze parameters at a
            base patch of its own; see ``restrict_to_realized`` in
            :mod:`dataset.preset_loader_common`. Merged over the synth defaults, so a
            partial dict is fine, and written to the run summary either way.
        parameter_space: the space this corpus estimates, overriding the synth's own
            subset. Pass the narrowed space from ``restrict_to_realized`` alongside its
            ``default_params``; it must be the same space the ``PresetSource`` was built
            against, and it is what gets serialized into the run summary (D-SELFDESC).
    """

    def __init__(
        self,
        synth: BaseSynthesizer,
        render_settings: Optional[RenderSettings] = None,
        min_loudness_lufs: float = -34.0,
        max_redraw_attempts: int = 10,
        render_backend=None,
        default_params: Optional[Dict[str, float]] = None,
        parameter_space: Optional[ParameterSpace] = None,
    ):
        self._synth = synth
        self._settings = render_settings or RenderSettings.from_config()
        self._backend = render_backend or InProcessRenderBackend(synth, self._settings)
        self._min_loudness_lufs = float(min_loudness_lufs)
        self._max_redraw_attempts = int(max_redraw_attempts)
        self._loudness_meter = pyloudnorm.Meter(int(synth.sample_rate))

        synth_defaults = synth.get_parameter_defaults()
        unknown = set(default_params or {}) - set(synth_defaults)
        if unknown:
            raise KeyError(f"default_params names the synth does not expose: {sorted(unknown)}")
        self._defaults = {**synth_defaults, **(default_params or {})}
        self._parameter_space = parameter_space or synth.parameter_space
        self._subset_names = self._parameter_space.names
        outside = [name for name in self._subset_names if name not in synth_defaults]
        if outside:
            raise KeyError(f"parameter_space names the synth does not expose: {outside}")

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
        dropped = 0
        try:
            rendered = tqdm(
                self._iter_rendered(source), total=total, desc="Rendering",
                unit="preset", disable=not show_progress,
            )
            for result in rendered:
                if result is None:
                    # Every attempt (the original render plus redraws) either timed out or
                    # ran out of redraws while the source could no longer replace it -- see
                    # _render_with_redraw. Dropped rather than written with placeholder audio.
                    dropped += 1
                    continue
                kept_preset, audio, loudness = result
                sample_id = f"sample_{len(rows):06d}"
                relative_path = f"audio/{sample_id}.wav"
                wavfile.write(str(run_dir / relative_path), self._synth.sample_rate, audio.astype(np.float32))
                rows.append(self._build_metadata_row(sample_id, relative_path, kept_preset, audio, loudness))
        finally:
            self._backend.close()

        if dropped:
            print(
                f"Warning: {dropped} preset(s) never produced audio within the render timeout "
                f"after {self._max_redraw_attempts} redraw attempt(s) and were dropped; corpus "
                f"has {len(rows)} sample(s) instead of the requested {total}."
            )

        df_metadata = pd.DataFrame(rows, columns=_LEADING_COLUMNS + self._subset_names + _TRAILING_COLUMNS)
        df_metadata.to_csv(run_dir / "metadata.csv", index=False)

        run_summary = self._build_run_summary(run_name, source, rows, dropped)
        with open(run_dir / "run_summary.json", "w") as run_summary_file:
            json.dump(run_summary, run_summary_file, indent=2)
        return run_summary

    # -- rendering -----------------------------------------------------------
    def _full_params(self, preset: PresetRecord) -> Dict[str, float]:
        extra = set(preset.params) - set(self._subset_names)
        if extra:
            raise KeyError(f"Preset carries non-subset parameters: {sorted(extra)}")
        return {**self._defaults, **preset.params}

    def _iter_rendered(
        self, source: PresetSource
    ) -> Iterator[Optional[Tuple[PresetRecord, np.ndarray, float]]]:
        """Yield ``(preset, audio, loudness)`` in source order, or ``None`` for a preset that
        never produced audio at all (see ``_render_with_redraw``); the caller drops those.

        A backend with a ``render_batch`` fans a chunk of presets across its workers; one
        without renders them one at a time. Either way the results come back in source
        order, so sample ids and metadata rows are unaffected by the choice.
        """
        batch_size = self._render_batch_size()
        for batch in self._batched(source.iter_presets(), batch_size):
            if batch_size == 1:
                yield self._render_with_redraw(source, batch[0])
                continue
            waveforms = self._backend.render_batch([self._full_params(p) for p in batch])
            for preset, audio in zip(batch, waveforms):
                if audio is None:
                    # A timed-out slot (ParallelFreshProcessRenderBackend.render_batch's
                    # convention for "this one was killed"), not a render that ran and
                    # happened to be silent.
                    yield self._render_with_redraw(source, preset, timed_out=True)
                else:
                    # The batch render counts as the first attempt, so it is handed to the
                    # redraw path rather than thrown away and repeated.
                    yield self._render_with_redraw(
                        source, preset, audio, self._integrated_loudness(audio)
                    )

    def _render_batch_size(self) -> int:
        """How many presets to hand the backend at once: 1 unless it renders batches."""
        if not hasattr(self._backend, "render_batch"):
            return 1
        # Enough to keep every worker fed without buffering a large pile of audio.
        return max(1, 4 * int(getattr(self._backend, "num_workers", 1)))

    @staticmethod
    def _batched(
        presets: Iterator[PresetRecord], size: int
    ) -> Iterator[List[PresetRecord]]:
        batch: List[PresetRecord] = []
        for preset in presets:
            batch.append(preset)
            if len(batch) == size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _render_with_redraw(
        self,
        source: PresetSource,
        preset: PresetRecord,
        audio: Optional[np.ndarray] = None,
        loudness: Optional[float] = None,
        timed_out: bool = False,
    ) -> Optional[Tuple[PresetRecord, np.ndarray, float]]:
        """Render ``preset``, redrawing near-silent or timed-out results until one succeeds.

        ``audio`` / ``loudness`` let the batch path pass in the render it already has, so the
        first attempt is judged rather than repeated; pass ``timed_out=True`` instead when
        that attempt was killed for exceeding the render timeout (see ``RenderTimeoutError``)
        rather than merely quiet.

        Returns ``None`` if every attempt (the original draw plus up to
        ``max_redraw_attempts`` redraws) either timed out, or ran out of redraws while
        ``source`` could no longer supply a replacement (e.g. a fixed human preset) --
        the caller drops that slot rather than inventing placeholder audio for a preset
        that never actually rendered. A near-silent-but-real render is still kept as-is,
        exactly as before this method also had to handle timeouts.
        """
        attempt = 0
        current = preset
        while True:
            if audio is None and not timed_out:
                try:
                    audio = self._backend.render(self._full_params(current))
                except RenderTimeoutError:
                    timed_out = True
            if timed_out:
                give_up = attempt >= self._max_redraw_attempts
            else:
                loudness = self._integrated_loudness(audio)
                give_up = loudness >= self._min_loudness_lufs or attempt >= self._max_redraw_attempts
            if give_up:
                return None if timed_out else (current, audio, loudness)
            replacement = source.resample(current, attempt + 1)
            if replacement is None:
                return None if timed_out else (current, audio, loudness)
            current, attempt = replacement, attempt + 1
            audio, timed_out = None, False

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
        self, run_name: str, source: PresetSource, rows: List[Dict[str, object]], dropped: int = 0
    ) -> Dict[str, object]:
        near_silent_count = sum(1 for row in rows if row["near_silent"])
        method_counts: Dict[str, int] = {}
        for row in rows:
            method_counts[row["method"]] = method_counts.get(row["method"], 0) + 1
        return {
            "run_name": run_name,
            "num_samples": len(rows),
            "near_silent_count": near_silent_count,
            "render_timeout_dropped_count": dropped,
            "method_counts": method_counts,
            "render_settings": asdict(self._settings),
            "render_process": getattr(self._backend, "process_mode", "in-process"),
            "sample_rate": self._synth.sample_rate,
            "renderer": getattr(self._synth, "renderer_name", None),
            "synth": getattr(self._synth, "synth_name", DEFAULT_SYNTH),
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
