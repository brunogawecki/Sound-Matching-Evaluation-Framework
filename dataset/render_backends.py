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
import time
from dataclasses import dataclass
from multiprocessing import connection as mp_connection
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
    # Whether the plugin logs from its destructor, which happens at worker-process exit and
    # so escapes the suppressor around construction and rendering. See _render_worker_entry.
    logs_at_teardown: bool = False


_SYNTH_REGISTRY: Dict[str, _SynthSpec] = {
    "dexed": _SynthSpec(_import_dexed_wrapper, "DEXED_PATH", suppressed_stderr),
    "diva": _SynthSpec(
        _import_diva_wrapper, "DIVA_PATH", _suppress_diva_output, logs_at_teardown=True
    ),
}

DEFAULT_SYNTH = "dexed"


def synth_logs_at_teardown(synth_name: str) -> bool:
    """Whether this synth's plugin logs from its destructor (Diva does, Dexed does not)."""
    return _synth_spec(synth_name).logs_at_teardown


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
    wrapper and renders a single patch, so when the caller runs this inside a freshly spawned,
    single-use process (see ``_spawn_render``) the render happens on a clean OS heap -- the
    only context in which Dexed's hidden per-voice state is reset, and the only context in
    which Diva reproduces at all (D-REPRO / D-DIVA-RENDER). Returns mono float32 audio.
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

# Generous relative to a real render (measured well under 1s): a patch that takes this long
# is not slow, it is hung -- e.g. a parameter combination that drives the plugin's DSP into a
# runaway loop (observed on Diva under heavy random/augmented sampling). See RenderTimeoutError.
DEFAULT_RENDER_TIMEOUT_SEC = 30.0


class RenderTimeoutError(Exception):
    """The worker process did not return a result -- killed after hanging, or it crashed on
    its own (observed on Diva: a native crash in the plugin's DSP for a specific pathological
    parameter combination, distinct from a hang and just as unpredictable in advance).

    Raised by :meth:`FreshProcessRenderBackend.render`; reported as a ``None`` slot by
    :meth:`ParallelFreshProcessRenderBackend.render_batch` (a single bad item must not block
    the rest of the batch the way an exception propagating out of ``Pool.map`` would -- and
    must not abort the whole build either, unlike a real bug in our own code, which still
    raises normally). :class:`~dataset.builder.DatasetBuilder` treats either case as a
    redraw-worthy failure, same machinery as a near-silent render, and drops the preset if no
    redraw is possible.
    """


def _render_worker_entry(
    render_worker: RenderWorker,
    payload: Tuple[Dict[str, float], RenderSettings, str, str],
    conn: "mp_connection.Connection",
) -> None:
    """Run one render in this fresh process, reporting the outcome back over ``conn``.

    Top-level (picklable) so it survives the spawn pickle. Silences a teardown-chatty plugin's
    stdout/stderr for this process's whole life -- it is one render long by construction (a
    brand-new process per call, never reused), so there is no pool-initializer step separate
    from this to hang the silencing off of; Diva logs ~15 lines from its destructor, which
    runs after the render call returns, so the redirect has to outlive this function, not
    just wrap it (see ``synth/plugin_output.py``). Exceptions cross the process boundary by
    pickling them onto ``conn`` rather than by traceback on stderr, same as a Pool worker.
    """
    synth_name = payload[3]
    if _synth_spec(synth_name).logs_at_teardown:
        from synth.plugin_output import silence_plugin_output_from_now_on

        silence_plugin_output_from_now_on()
    try:
        audio = render_worker(payload)
        conn.send(("ok", audio))
    except BaseException as exc:  # noqa: BLE001 -- forward any failure to the parent process
        conn.send(("error", exc))
    finally:
        conn.close()


def _spawn_render(
    context: "mp.context.SpawnContext",
    render_worker: RenderWorker,
    payload: Tuple[Dict[str, float], RenderSettings, str, str],
) -> Tuple["mp.process.BaseProcess", "mp_connection.Connection"]:
    """Start one render in a brand-new process; the caller collects it via ``_collect_render``."""
    parent_conn, child_conn = context.Pipe(duplex=False)
    process = context.Process(target=_render_worker_entry, args=(render_worker, payload, child_conn))
    process.start()
    child_conn.close()  # only the child writes; drop the parent's copy of that end
    return process, parent_conn


def _reap(process: "mp.process.BaseProcess") -> None:
    """Join a process that should already be finishing, escalating to a hard kill if it
    somehow isn't -- never leaves a process behind for the caller to forget about."""
    process.join(5.0)
    if process.is_alive():
        process.kill()
        process.join()


def _collect_render(
    process: "mp.process.BaseProcess", conn: "mp_connection.Connection", timeout_sec: float
) -> np.ndarray:
    """Wait for one spawned render to finish, killing it and raising on timeout or crash.

    A hung render can sit deep enough in native plugin code that it never checks for
    signals, so ``terminate()`` (SIGTERM) is given a short grace period before escalating to
    ``kill()`` (SIGKILL) -- observed necessary in practice, not a hypothetical. A render that
    crashes outright closes its end of ``conn`` without ever sending -- ``recv()`` raises
    ``EOFError`` for that, also observed in practice, and is treated the same as a timeout
    rather than let the raw ``EOFError`` escape as if it were a bug in this code.
    """
    if conn.poll(timeout_sec):
        try:
            status, value = conn.recv()
        except EOFError:
            conn.close()
            _reap(process)
            raise RenderTimeoutError(
                "Render worker exited without producing a result -- most likely a native "
                "crash in the plugin's DSP for this specific patch, not a timeout."
            )
        conn.close()
        _reap(process)
        if status == "ok":
            return value
        raise value
    process.terminate()
    _reap(process)
    conn.close()
    raise RenderTimeoutError(
        f"Render did not finish within {timeout_sec:.0f}s and was killed; the patch likely "
        "drives the plugin's DSP into a runaway loop."
    )


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

    Every call to :meth:`render` starts a brand-new **spawn** (never **fork**) process, so
    each render lands on a genuinely clean OS heap. Serial: one render at a time, with no
    persistent pool to tear down -- :meth:`close` and the context-manager form are kept for
    interface parity, but there is nothing left running between renders to close.

    A render that exceeds ``timeout_sec`` is killed and raises :class:`RenderTimeoutError`
    rather than hanging forever -- see that class's docstring.
    """

    process_mode = "fresh"

    def __init__(
        self,
        settings: RenderSettings,
        renderer: str = "dawdreamer",
        synth_name: str = DEFAULT_SYNTH,
        timeout_sec: float = DEFAULT_RENDER_TIMEOUT_SEC,
    ):
        self._settings = settings
        self._renderer = renderer
        self._synth_name = synth_name
        self._timeout_sec = float(timeout_sec)
        _synth_spec(synth_name)  # fail here, not inside a spawned worker
        self._context = mp.get_context("spawn")

    def render(self, params: Dict[str, float]) -> np.ndarray:
        payload = (params, self._settings, self._renderer, self._synth_name)
        process, conn = _spawn_render(self._context, render_patch_in_fresh_process, payload)
        return _collect_render(process, conn, self._timeout_sec)

    def close(self) -> None:
        pass

    def __enter__(self) -> "FreshProcessRenderBackend":
        return self

    def __exit__(self, *exception) -> None:
        self.close()


class ParallelFreshProcessRenderBackend:
    """The fresh-process backend widened to ``num_workers`` parallel workers (RL reward path).

    Identical isolation to :class:`FreshProcessRenderBackend` -- same **spawn** start method,
    a genuinely new process per render -- but up to ``num_workers`` renders run concurrently.
    The SynthRL RL stage renders a whole batch of predicted patches per training step;
    :meth:`render_batch` fans that batch across up to ``num_workers`` renders at a time,
    launching a fresh replacement the moment each slot finishes (or is killed for exceeding
    ``timeout_sec``), so a hung render costs one slot's worth of time, not the whole batch's.
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
        synth_name: str = DEFAULT_SYNTH,
        timeout_sec: float = DEFAULT_RENDER_TIMEOUT_SEC,
    ):
        self._settings = settings
        self._renderer = renderer
        self._render_worker = render_worker
        self._synth_name = synth_name
        self._timeout_sec = float(timeout_sec)
        _synth_spec(synth_name)  # fail here, not inside a spawned worker
        self.num_workers = num_workers if num_workers is not None else (os.cpu_count() or 1)
        self._context = mp.get_context("spawn")

    def _payload(
        self, params: Dict[str, float]
    ) -> Tuple[Dict[str, float], RenderSettings, str, str]:
        return (params, self._settings, self._renderer, self._synth_name)

    def render(self, params: Dict[str, float]) -> np.ndarray:
        process, conn = _spawn_render(self._context, self._render_worker, self._payload(params))
        return _collect_render(process, conn, self._timeout_sec)

    def render_batch(self, params_batch: List[Dict[str, float]]) -> List[Optional[np.ndarray]]:
        """Render a list of patches with up to ``num_workers`` concurrent fresh processes.

        Preserves input order. A slot that times out **or whose worker crashes outright**
        (a native crash in the plugin's DSP, not merely a hang -- observed in practice) is
        killed/reaped and comes back as ``None`` rather than raising --
        :class:`~dataset.builder.DatasetBuilder` treats that as a redraw-worthy failure, the
        same as a near-silent render, instead of one bad patch aborting every other render
        already in flight. A genuine exception raised by our own code (not the plugin dying)
        is still re-raised, once every in-flight render has been collected or killed so
        nothing is left orphaned.
        """
        payloads = [self._payload(params) for params in params_batch]
        total = len(payloads)
        results: List[Optional[np.ndarray]] = [None] * total
        in_flight: Dict[mp_connection.Connection, Tuple[int, "mp.process.BaseProcess", float]] = {}
        next_index = 0
        pending_error: Optional[BaseException] = None

        def launch(index: int) -> None:
            process, conn = _spawn_render(self._context, self._render_worker, payloads[index])
            in_flight[conn] = (index, process, time.monotonic())

        while next_index < total and len(in_flight) < self.num_workers:
            launch(next_index)
            next_index += 1

        while in_flight:
            deadline = min(started + self._timeout_sec for _, _, started in in_flight.values())
            ready = set(mp_connection.wait(list(in_flight), timeout=max(0.0, deadline - time.monotonic())))
            now = time.monotonic()
            for conn, (index, process, started) in list(in_flight.items()):
                if conn in ready:
                    try:
                        status, value = conn.recv()
                    except EOFError:
                        # The worker died before sending anything -- a crash, not a bug in our
                        # code. results[index] stays None, same treatment as a timeout below.
                        status, value = None, None
                    conn.close()
                    _reap(process)
                    if status == "ok":
                        results[index] = value
                    elif status == "error" and pending_error is None:
                        pending_error = value
                elif now - started >= self._timeout_sec:
                    process.terminate()
                    _reap(process)
                    conn.close()
                    # results[index] stays None: a timed-out slot, not a hard failure.
                else:
                    continue
                del in_flight[conn]
                if next_index < total:
                    launch(next_index)
                    next_index += 1

        if pending_error is not None:
            raise pending_error
        return results

    def close(self) -> None:
        pass

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
