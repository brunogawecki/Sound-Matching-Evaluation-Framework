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
    ``num_workers`` workers, with a batch API; the faithful (default) SynthRL RL reward path.
  * :class:`ParallelInProcessRenderBackend` -- ``num_workers`` workers that each reuse one wrapper
    across every render, so the hidden state leaks (a measured, deliberate bias -- D-RL-RENDER);
    tens to hundreds of times faster, the opt-in SynthRL RL reward path for cluster runs.

The Evaluator (#9) re-renders predictions through :class:`FreshProcessRenderBackend` so that
target and re-render share an identical clean context; the in-process backend never sits on the
eval path.
"""
from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Callable, ContextManager, Dict, List, Optional, Tuple, Type

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


def _import_dexed_wrapper() -> Type[BaseSynthesizer]:
    return DexedWrapper


def _import_diva_wrapper() -> Type[BaseSynthesizer]:
    from synth.diva import DivaWrapper
    return DivaWrapper


def _suppress_diva_output() -> ContextManager[None]:
    # Diva writes its banner to stdout as well as stderr, so it needs the wider suppressor.
    from synth.plugin_output import suppressed_plugin_output
    return suppressed_plugin_output()


@dataclass(frozen=True)
class _SynthSpec:
    """Everything a render worker needs to build one synth's wrapper.

    ``import_wrapper_class`` is a deferred import rather than a direct class reference so a
    Dexed render never imports the Diva package. Spawned workers re-import this module on
    every render, and Dexed's reproducibility has already proved sensitive to what is imported
    alongside it (see the module docstring of ``synth/plugin_output.py``).
    """
    import_wrapper_class: Callable[[], Type[BaseSynthesizer]]
    plugin_path_config_attribute: str
    open_output_suppressor: Callable[[], ContextManager[None]]


_SYNTH_REGISTRY: Dict[str, _SynthSpec] = {
    "dexed": _SynthSpec(_import_dexed_wrapper, "DEXED_PATH", suppressed_stderr),
    "diva": _SynthSpec(_import_diva_wrapper, "DIVA_PATH", _suppress_diva_output),
}

DEFAULT_SYNTH = "dexed"


def _synth_spec(synth_name: str) -> _SynthSpec:
    try:
        return _SYNTH_REGISTRY[synth_name]
    except KeyError:
        raise ValueError(
            f"Unknown synth {synth_name!r}. Known: {sorted(_SYNTH_REGISTRY)}."
        ) from None


def synth_plugin_path(synth_name: str) -> str:
    """Expanded local path to a synth's plugin binary, from ``config``."""
    return os.path.expanduser(getattr(config, _synth_spec(synth_name).plugin_path_config_attribute))


def make_wrapper(renderer: str, synth_name: str = DEFAULT_SYNTH) -> BaseSynthesizer:
    """Construct a renderer-backed wrapper (caller is responsible for output suppression)."""
    spec = _synth_spec(synth_name)
    wrapper_class = spec.import_wrapper_class()
    return wrapper_class(
        plugin_path=synth_plugin_path(synth_name),
        sample_rate=config.SAMPLE_RATE,
        buffer_size=config.BUFFER_SIZE,
        renderer=renderer,
    )


def render_patch_in_fresh_process(
    payload: Tuple[Dict[str, float], RenderSettings, str, str]
) -> np.ndarray:
    """Render one patch at position 0 of a brand-new wrapper.

    Top-level (picklable) so it can run inside a spawned worker. Each call constructs its own
    wrapper and renders a single patch, so when the worker process itself is fresh (spawn +
    ``maxtasksperchild=1``) the render happens on a clean OS heap -- the only context in which
    Dexed's hidden per-voice state is reset, and the only context in which Diva reproduces at
    all (D-REPRO / D-DIVA-RENDER). Returns mono float32 audio.
    """
    patch, settings, renderer, synth_name = payload
    spec = _synth_spec(synth_name)
    with spec.open_output_suppressor():
        wrapper = make_wrapper(renderer, synth_name)
    wrapper.set_parameters(patch)
    audio = wrapper.render_audio(
        settings.midi_note,
        settings.velocity,
        settings.duration_sec,
        settings.note_duration_sec,
    )
    return np.asarray(audio, dtype=np.float32)


# A picklable ``(patch, settings, renderer, synth_name) -> mono audio`` render worker.
# Top-level so it survives the spawn pickle; the default is the real render above.
RenderWorker = Callable[[Tuple[Dict[str, float], RenderSettings, str, str]], np.ndarray]


class InProcessRenderBackend:
    """Render every patch through one reused wrapper (fast; the default training path).

    The hidden voice state leaks across renders, but it adds an equal noise floor to every
    model and so does not bias the between-framework ranking (D-REPRO policy).
    """

    process_mode = "in-process"

    def __init__(self, synth: BaseSynthesizer, settings: RenderSettings):
        # getattr, not attribute access: BaseSynthesizer defaults this to True, so every real
        # wrapper has it, but VST-free test doubles duck-type the interface without subclassing.
        if not getattr(synth, "supports_in_process_render", True):
            raise ValueError(
                f"{type(synth).__name__} does not reproduce in-process; use "
                "FreshProcessRenderBackend. See D-DIVA-RENDER in docs/DECISIONS.md."
            )
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

    def __init__(
        self,
        settings: RenderSettings,
        renderer: str = "dawdreamer",
        synth_name: str = DEFAULT_SYNTH,
    ):
        self._settings = settings
        self._renderer = renderer
        self._synth_name = synth_name
        _synth_spec(synth_name)  # fail here, not inside a spawned worker
        self._pool = mp.get_context("spawn").Pool(processes=1, maxtasksperchild=1)

    def render(self, params: Dict[str, float]) -> np.ndarray:
        return self._pool.apply(
            render_patch_in_fresh_process,
            ((params, self._settings, self._renderer, self._synth_name),),
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
    not the per-render result -- but see :meth:`render_batch`, which has to defeat ``Pool.map``'s
    default chunking to keep that true.

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
        synth_name: str = DEFAULT_SYNTH,
    ):
        self._settings = settings
        self._renderer = renderer
        self._render_worker = render_worker
        self._synth_name = synth_name
        _synth_spec(synth_name)  # fail here, not inside a spawned worker
        self.num_workers = num_workers if num_workers is not None else (os.cpu_count() or 1)
        self._pool = mp.get_context("spawn").Pool(
            processes=self.num_workers, maxtasksperchild=1
        )

    def _payload(
        self, params: Dict[str, float]
    ) -> Tuple[Dict[str, float], RenderSettings, str, str]:
        return (params, self._settings, self._renderer, self._synth_name)

    def render(self, params: Dict[str, float]) -> np.ndarray:
        return self._pool.apply(self._render_worker, (self._payload(params),))

    def render_batch(self, params_batch: List[Dict[str, float]]) -> List[np.ndarray]:
        """Render a list of patches in parallel, preserving input order.

        ``chunksize=1`` is load-bearing, not a tuning knob. ``maxtasksperchild`` retires a
        worker after one *task*, and ``Pool.map`` packs ``ceil(n / (4 * workers))`` payloads
        into a task by default, so any batch longer than ``4 * num_workers`` would put several
        renders in one process and silently lose the isolation this class exists for.
        """
        payloads = [self._payload(params) for params in params_batch]
        return self._pool.map(self._render_worker, payloads, chunksize=1)

    def close(self) -> None:
        self._pool.terminate()
        self._pool.join()

    def __enter__(self) -> "ParallelFreshProcessRenderBackend":
        return self

    def __exit__(self, *exception) -> None:
        self.close()


# One persistent wrapper per worker, built once by the pool initializer and reused for every
# render that worker handles. Module-level so a spawned worker rebuilds them on import.
_REUSE_WRAPPER: Optional[BaseSynthesizer] = None
_REUSE_SETTINGS: Optional[RenderSettings] = None


def init_reuse_worker(
    renderer: str, settings: RenderSettings, synth_name: str = DEFAULT_SYNTH
) -> None:
    """Pool initializer: build the one wrapper this worker reuses across all its renders."""
    global _REUSE_WRAPPER, _REUSE_SETTINGS
    spec = _synth_spec(synth_name)
    with spec.open_output_suppressor():
        _REUSE_WRAPPER = make_wrapper(renderer, synth_name)
    _REUSE_SETTINGS = settings


def render_patch_in_reused_wrapper(patch: Dict[str, float]) -> np.ndarray:
    """Render one patch through this worker's persistent wrapper (state leaks across renders)."""
    _REUSE_WRAPPER.set_parameters(patch)
    audio = _REUSE_WRAPPER.render_audio(
        _REUSE_SETTINGS.midi_note,
        _REUSE_SETTINGS.velocity,
        _REUSE_SETTINGS.duration_sec,
        _REUSE_SETTINGS.note_duration_sec,
    )
    return np.asarray(audio, dtype=np.float32)


# A picklable ``patch -> mono audio`` worker for the reuse pool; the settings/renderer are
# baked into each worker by the initializer, so only the patch crosses per call.
ReuseRenderWorker = Callable[[Dict[str, float]], np.ndarray]


class ParallelInProcessRenderBackend:
    """``num_workers`` workers that each reuse one wrapper across every render (fast, leaky).

    Unlike :class:`ParallelFreshProcessRenderBackend`, the pool has **no** ``maxtasksperchild``
    limit: each worker builds a :class:`DexedWrapper` once (the pool ``initializer``) and reuses
    it, so Dexed's hidden per-voice state (LFO / sample-&-hold / noise) leaks across the renders
    that worker performs. That leak is measured and deliberate (D-RL-RENDER): it biases the RL
    reward but is tens to hundreds of times faster, and eval never uses this backend. Same
    :meth:`render` / :meth:`render_batch` interface as the fresh-process backend, so the RL stage
    swaps one for the other by config.

    ``worker_initializer`` / ``render_worker`` are the picklable seam tests use to inject a
    VST-free stand-in.
    """

    process_mode = "reuse"

    def __init__(
        self,
        settings: RenderSettings,
        renderer: str = "dawdreamer",
        num_workers: Optional[int] = None,
        worker_initializer: Callable[[str, RenderSettings, str], None] = init_reuse_worker,
        render_worker: ReuseRenderWorker = render_patch_in_reused_wrapper,
        synth_name: str = DEFAULT_SYNTH,
    ):
        wrapper_class = _synth_spec(synth_name).import_wrapper_class()
        if not wrapper_class.supports_in_process_render:
            raise ValueError(
                f"{wrapper_class.__name__} does not reproduce in-process; use "
                "ParallelFreshProcessRenderBackend. See D-DIVA-RENDER in docs/DECISIONS.md."
            )
        self._settings = settings
        self._renderer = renderer
        self._render_worker = render_worker
        self._synth_name = synth_name
        self.num_workers = num_workers if num_workers is not None else (os.cpu_count() or 1)
        self._pool = mp.get_context("spawn").Pool(
            processes=self.num_workers,
            initializer=worker_initializer,
            initargs=(renderer, settings, synth_name),
        )

    def render(self, params: Dict[str, float]) -> np.ndarray:
        return self._pool.apply(self._render_worker, (params,))

    def render_batch(self, params_batch: List[Dict[str, float]]) -> List[np.ndarray]:
        """Render a list of patches in parallel, preserving input order."""
        return self._pool.map(self._render_worker, params_batch)

    def close(self) -> None:
        self._pool.terminate()
        self._pool.join()

    def __enter__(self) -> "ParallelInProcessRenderBackend":
        return self

    def __exit__(self, *exception) -> None:
        self.close()
