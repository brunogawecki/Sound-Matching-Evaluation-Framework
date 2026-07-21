"""The render-execution layer for the dataset builder (Layer 2): how a parameter dict
becomes audio.

The :class:`~dataset.builder.DatasetBuilder` orchestrates *what* to render (preset
iteration, redraw-on-silence, writing WAV/CSV); this module owns *how* each render runs.
It holds the render contract (:class:`RenderSettings`) and two interchangeable backends
exposing the same ``render(params) -> np.ndarray`` interface, plus the picklable worker the
fresh-process backend runs.

The two backends differ only in process isolation (D-REPRO, docs/DECISIONS.md): Dexed's
plugin binary carries hidden per-voice state (LFO / sample-&-hold / noise) that survives
re-applying parameters and in-process wrapper reloads; only a fresh OS process resets it.

  * :class:`InProcessRenderBackend` -- one reused wrapper, fast (~4 ms); fine for training
    data, where the hidden-state leak adds an *equal* noise floor to every model and so does
    not bias the between-framework ranking (D-REPRO policy).
  * :class:`FreshProcessRenderBackend` -- one spawned worker per render at position 0 of a
    clean heap (never fork), slow but leak-free; used for test/eval corpora, where generation
    and evaluation render contexts must agree.
  * :class:`ParallelFreshProcessRenderBackend` -- the same leak-free per-render isolation across
    ``num_workers`` workers, with a batch API; used by the SynthRL RL stage, which renders a
    batch of predicted patches for the reward each training step.

The Evaluator (#9) re-renders predictions through :class:`FreshProcessRenderBackend` so that
target and re-render share an identical clean context.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

import config
from synth.base_synth import BaseSynthesizer
from synth.dexed import DexedWrapper, suppressed_stderr


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


def _make_wrapper(renderer: str) -> DexedWrapper:
    """Construct a renderer-backed Dexed wrapper (caller is responsible for stderr suppression)."""
    return DexedWrapper(
        plugin_path=os.path.expanduser(config.DEXED_PATH),
        sample_rate=config.SAMPLE_RATE,
        buffer_size=config.BUFFER_SIZE,
        renderer=renderer,
    )


def render_patch_in_fresh_process(
    payload: Tuple[Dict[str, float], RenderSettings, str]
) -> np.ndarray:
    """Render one patch at position 0 of a brand-new Dexed wrapper.

    Top-level (picklable) so it can run inside a spawned worker. Each call constructs its own
    wrapper and renders a single patch, so when the worker process itself is fresh (spawn +
    ``maxtasksperchild=1``) the render happens on a clean OS heap -- the only context in which
    Dexed's hidden per-voice state is reset (D-REPRO). Returns mono float32 audio.
    """
    patch, settings, renderer = payload
    with suppressed_stderr():
        wrapper = _make_wrapper(renderer)
    wrapper.set_parameters(patch)
    audio = wrapper.render_audio(
        settings.midi_note,
        settings.velocity,
        settings.duration_sec,
        settings.note_duration_sec,
    )
    return np.asarray(audio, dtype=np.float32)


# A picklable ``(patch, settings, renderer) -> mono audio`` render worker. Top-level so
# it survives the spawn pickle; the default is the real Dexed render above.
RenderWorker = Callable[[Tuple[Dict[str, float], RenderSettings, str]], np.ndarray]


class InProcessRenderBackend:
    """Render every patch through one reused wrapper (fast; the default training path).

    The hidden voice state leaks across renders, but it adds an equal noise floor to every
    model and so does not bias the between-framework ranking (D-REPRO policy).
    """

    process_mode = "in-process"

    def __init__(self, synth: BaseSynthesizer, settings: RenderSettings):
        self._synth = synth
        self._settings = settings

    def render(self, params: Dict[str, float]) -> np.ndarray:
        self._synth.set_parameters(params)
        return self._synth.render_audio(
            self._settings.midi_note,
            self._settings.velocity,
            self._settings.duration_sec,
            self._settings.note_duration_sec,
        )

    def close(self) -> None:
        pass


class FreshProcessRenderBackend:
    """Render each patch at position 0 of its own spawned worker (leak-free; test/eval path).

    Holds a persistent single-worker pool with the **spawn** start method and
    ``maxtasksperchild=1``, so the worker is torn down and a clean interpreter spawned for
    every render -- a genuinely fresh heap per patch (never **fork**, which inherits the
    parent's dirty memory). Serial: one render at a time. Call :meth:`close` (or use as a
    context manager) to tear the pool down.
    """

    process_mode = "fresh"

    def __init__(self, settings: RenderSettings, renderer: str = "dawdreamer"):
        self._settings = settings
        self._renderer = renderer
        self._pool = mp.get_context("spawn").Pool(processes=1, maxtasksperchild=1)

    def render(self, params: Dict[str, float]) -> np.ndarray:
        return self._pool.apply(
            render_patch_in_fresh_process, ((params, self._settings, self._renderer),)
        )

    def close(self) -> None:
        self._pool.terminate()
        self._pool.join()

    def __enter__(self) -> "FreshProcessRenderBackend":
        return self

    def __exit__(self, *exception) -> None:
        self.close()


class ParallelFreshProcessRenderBackend:
    """The fresh-process backend widened to ``num_workers`` parallel workers (RL reward path).

    Identical isolation to :class:`FreshProcessRenderBackend` -- same **spawn** start method,
    same ``maxtasksperchild=1`` so every render lands on a clean OS heap (D-REPRO) -- but with
    a pool of ``num_workers`` instead of one. The SynthRL RL stage renders a whole batch of
    predicted patches per training step; :meth:`render_batch` fans that batch across the pool.
    Each patch still gets its own single-use process, so parallelism changes only throughput,
    not the per-render result.

    ``render_worker`` is the picklable render function (defaults to the real Dexed render);
    tests inject a VST-free stand-in. Serial ``render`` is kept for interface parity with
    :class:`FreshProcessRenderBackend`.
    """

    process_mode = "fresh"

    def __init__(
        self,
        settings: RenderSettings,
        renderer: str = "dawdreamer",
        num_workers: Optional[int] = None,
        render_worker: RenderWorker = render_patch_in_fresh_process,
    ):
        self._settings = settings
        self._renderer = renderer
        self._render_worker = render_worker
        self.num_workers = num_workers if num_workers is not None else (os.cpu_count() or 1)
        self._pool = mp.get_context("spawn").Pool(
            processes=self.num_workers, maxtasksperchild=1
        )

    def _payload(self, params: Dict[str, float]) -> Tuple[Dict[str, float], RenderSettings, str]:
        return (params, self._settings, self._renderer)

    def render(self, params: Dict[str, float]) -> np.ndarray:
        return self._pool.apply(self._render_worker, (self._payload(params),))

    def render_batch(self, params_batch: List[Dict[str, float]]) -> List[np.ndarray]:
        """Render a list of patches in parallel, preserving input order."""
        payloads = [self._payload(params) for params in params_batch]
        return self._pool.map(self._render_worker, payloads)

    def close(self) -> None:
        self._pool.terminate()
        self._pool.join()

    def __enter__(self) -> "ParallelFreshProcessRenderBackend":
        return self

    def __exit__(self, *exception) -> None:
        self.close()
