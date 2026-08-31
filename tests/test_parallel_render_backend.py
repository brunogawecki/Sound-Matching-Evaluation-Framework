"""Tests for the parallel fresh-process render backend (SynthRL RL reward path, Step 4).

Verifies the three properties the plan asks for: parallel renders equal serial renders,
renders are non-silent, and per-render process isolation is preserved (every render lands on
its own single-use process). The parallel/serial-equality and isolation tests use a picklable
VST-free stand-in worker so they run anywhere; a Dexed-gated test confirms real renders are
non-silent. Workers are module-level so they survive the spawn pickle.
"""
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from dataset.render_backends import (
    FreshProcessRenderBackend,
    ParallelFreshProcessRenderBackend,
    ParallelInProcessRenderBackend,
    RenderSettings,
    RenderTimeoutError,
)

SAMPLE_RATE = 8000
SETTINGS = RenderSettings(midi_note=60, velocity=100, duration_sec=0.25, note_duration_sec=0.25)


def _sine_for(amp: float) -> np.ndarray:
    samples = int(SETTINGS.duration_sec * SAMPLE_RATE)
    time = np.arange(samples) / SAMPLE_RATE
    return (amp * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32)


def fake_render(payload) -> np.ndarray:
    """A deterministic, VST-free render worker: a sine whose amplitude is the patch's ``AMP``."""
    patch, _settings, _renderer, _synth_name = payload
    return _sine_for(float(patch["AMP"]))


def pid_render(payload) -> np.ndarray:
    """A worker that reports the OS process it ran in (first sample = pid), to probe isolation."""
    return np.array([float(os.getpid())], dtype=np.float32)


_HANG_SENTINEL = -1.0


def hang_forever_render(payload) -> np.ndarray:
    """A worker that never returns, to exercise the render-timeout path (never a real render)."""
    time.sleep(3600)
    raise AssertionError("should have been killed by the timeout before waking up")


def hang_for_sentinel_render(payload) -> np.ndarray:
    """Hangs only for the sentinel AMP, so one batch can mix a hung slot with normal ones."""
    patch, _settings, _renderer, _synth_name = payload
    if float(patch["AMP"]) == _HANG_SENTINEL:
        time.sleep(3600)
    return _sine_for(float(patch["AMP"]))


def raising_render(payload) -> np.ndarray:
    """A worker that fails outright (not a timeout), to check that error still propagates."""
    raise ValueError("boom")


_CRASH_SENTINEL = -2.0


def crash_render(payload) -> np.ndarray:
    """Exits the process immediately with no result, simulating a native plugin crash."""
    os._exit(1)


def crash_for_sentinel_render(payload) -> np.ndarray:
    """Crashes only for the sentinel AMP, so one batch can mix a crashed slot with normal ones."""
    patch, _settings, _renderer, _synth_name = payload
    if float(patch["AMP"]) == _CRASH_SENTINEL:
        os._exit(1)
    return _sine_for(float(patch["AMP"]))


# The reuse backend's worker takes a bare patch (settings/renderer are baked in by the
# initializer), so its stand-ins have a different signature from the fresh backend's.
def noop_reuse_init(renderer, settings, synth_name="dexed") -> None:
    pass


def fake_reuse_render(patch) -> np.ndarray:
    return _sine_for(float(patch["AMP"]))


def pid_reuse_render(patch) -> np.ndarray:
    return np.array([float(os.getpid())], dtype=np.float32)


def test_parallel_batch_equals_serial_and_expected(tmp_path):
    patches = [{"AMP": value} for value in (0.1, 0.4, 0.7, 1.0)]

    with ParallelFreshProcessRenderBackend(
        SETTINGS, num_workers=2, render_worker=fake_render
    ) as backend:
        parallel = backend.render_batch(patches)
        single = backend.render(patches[2])

    # Order preserved, and each equals the deterministic ground truth.
    assert len(parallel) == len(patches)
    for rendered, patch in zip(parallel, patches):
        np.testing.assert_allclose(rendered, _sine_for(patch["AMP"]))
    # The serial single-render path agrees with the batch.
    np.testing.assert_allclose(single, parallel[2])


def test_renders_are_non_silent():
    patches = [{"AMP": 0.5}, {"AMP": 0.9}]
    with ParallelFreshProcessRenderBackend(
        SETTINGS, num_workers=2, render_worker=fake_render
    ) as backend:
        for rendered in backend.render_batch(patches):
            assert np.max(np.abs(rendered)) > 0.0


def test_every_render_runs_in_its_own_fresh_process():
    # maxtasksperchild=1 means a batch of N yields N distinct single-use processes, even
    # with fewer workers -- the isolation guarantee the RL reward relies on (D-REPRO).
    patches = [{"AMP": 0.5} for _ in range(4)]
    with ParallelFreshProcessRenderBackend(
        SETTINGS, num_workers=2, render_worker=pid_render
    ) as backend:
        pids = {int(rendered[0]) for rendered in backend.render_batch(patches)}
    assert len(pids) == len(patches)


def test_isolation_survives_a_batch_longer_than_the_chunking_threshold():
    # The case above cannot catch Pool.map's default chunking: it packs
    # ceil(n / (4 * workers)) payloads per task, which is 1 until the batch exceeds
    # 4 * num_workers. maxtasksperchild retires a worker per task, not per render, so
    # without chunksize=1 the renders past that threshold share a process. The RL stage
    # renders batches of tens of patches, so it sits well past it.
    patches = [{"AMP": 0.5} for _ in range(9)]
    with ParallelFreshProcessRenderBackend(
        SETTINGS, num_workers=2, render_worker=pid_render
    ) as backend:
        pids = {int(rendered[0]) for rendered in backend.render_batch(patches)}
    assert len(pids) == len(patches)


def test_defaults_to_cpu_count_workers():
    with ParallelFreshProcessRenderBackend(SETTINGS, render_worker=fake_render) as backend:
        assert backend.num_workers == (os.cpu_count() or 1)


# ---------------------------------------------------------------------------
# Render timeouts: a hung render must not block the caller forever, and must not
# block other renders already in flight in the same batch.
# ---------------------------------------------------------------------------
def test_serial_render_raises_timeout_error_and_kills_the_hung_worker():
    with ParallelFreshProcessRenderBackend(
        SETTINGS, num_workers=1, render_worker=hang_forever_render, timeout_sec=0.3,
    ) as backend:
        with pytest.raises(RenderTimeoutError):
            backend.render({"AMP": 0.5})
    import multiprocessing as mp

    assert not mp.active_children(), "the hung worker must be killed, not left running"


def test_render_batch_returns_none_for_a_timed_out_slot_and_keeps_the_rest():
    patches = [{"AMP": 0.2}, {"AMP": _HANG_SENTINEL}, {"AMP": 0.6}]
    with ParallelFreshProcessRenderBackend(
        SETTINGS, num_workers=2, render_worker=hang_for_sentinel_render, timeout_sec=0.3,
    ) as backend:
        results = backend.render_batch(patches)
    assert len(results) == 3
    assert results[1] is None
    np.testing.assert_allclose(results[0], _sine_for(0.2))
    np.testing.assert_allclose(results[2], _sine_for(0.6))


def test_render_batch_does_not_wait_for_the_whole_batch_when_one_slot_hangs():
    # Regression for the original bug: Pool.map only returns once every item finishes, so
    # workers 2 and 3 (fed after the hung slot 0 is retired) must still make progress rather
    # than sitting idle for the full test.
    patches = [{"AMP": _HANG_SENTINEL}] + [{"AMP": 0.5} for _ in range(5)]
    start = time.monotonic()
    with ParallelFreshProcessRenderBackend(
        SETTINGS, num_workers=2, render_worker=hang_for_sentinel_render, timeout_sec=0.3,
    ) as backend:
        results = backend.render_batch(patches)
    elapsed = time.monotonic() - start
    assert results[0] is None
    for rendered in results[1:]:
        np.testing.assert_allclose(rendered, _sine_for(0.5))
    # Generous bound: one timeout (0.3s) plus five quick renders, nowhere near serial-forever.
    assert elapsed < 5.0


def test_render_batch_still_reraises_a_genuine_worker_exception():
    with ParallelFreshProcessRenderBackend(
        SETTINGS, num_workers=2, render_worker=raising_render, timeout_sec=5.0,
    ) as backend:
        with pytest.raises(ValueError, match="boom"):
            backend.render_batch([{"AMP": 0.5}, {"AMP": 0.6}])


def test_serial_render_raises_timeout_error_when_worker_crashes():
    # A worker that exits without sending anything (os._exit) must not surface as a raw
    # EOFError -- it is exactly as retry-worthy as a hang, just a different cause.
    with ParallelFreshProcessRenderBackend(
        SETTINGS, num_workers=1, render_worker=crash_render, timeout_sec=5.0,
    ) as backend:
        with pytest.raises(RenderTimeoutError):
            backend.render({"AMP": 0.5})


def test_render_batch_returns_none_for_a_crashed_slot_and_keeps_the_rest():
    patches = [{"AMP": 0.2}, {"AMP": _CRASH_SENTINEL}, {"AMP": 0.6}]
    with ParallelFreshProcessRenderBackend(
        SETTINGS, num_workers=2, render_worker=crash_for_sentinel_render, timeout_sec=5.0,
    ) as backend:
        results = backend.render_batch(patches)
    assert len(results) == 3
    assert results[1] is None
    np.testing.assert_allclose(results[0], _sine_for(0.2))
    np.testing.assert_allclose(results[2], _sine_for(0.6))


# ---------------------------------------------------------------------------
# The reuse (in-process) backend: same batch/serial interface, but workers persist
# across renders (the opposite of the fresh backend's per-render isolation).
# ---------------------------------------------------------------------------
def test_reuse_batch_equals_serial_and_expected():
    patches = [{"AMP": value} for value in (0.1, 0.4, 0.7, 1.0)]
    with ParallelInProcessRenderBackend(
        SETTINGS, num_workers=2,
        worker_initializer=noop_reuse_init, render_worker=fake_reuse_render,
    ) as backend:
        parallel = backend.render_batch(patches)
        single = backend.render(patches[2])

    assert len(parallel) == len(patches)
    for rendered, patch in zip(parallel, patches):
        np.testing.assert_allclose(rendered, _sine_for(patch["AMP"]))
    np.testing.assert_allclose(single, parallel[2])


def test_reuse_workers_persist_across_renders():
    # The point of reuse mode: a batch of N is served by at most num_workers persistent
    # processes, not N single-use ones (contrast test_every_render_runs_in_its_own_fresh_process).
    patches = [{"AMP": 0.5} for _ in range(6)]
    with ParallelInProcessRenderBackend(
        SETTINGS, num_workers=2,
        worker_initializer=noop_reuse_init, render_worker=pid_reuse_render,
    ) as backend:
        pids = {int(rendered[0]) for rendered in backend.render_batch(patches)}
    assert len(pids) <= 2 < len(patches)


def test_reuse_defaults_to_cpu_count_workers():
    with ParallelInProcessRenderBackend(
        SETTINGS, worker_initializer=noop_reuse_init, render_worker=fake_reuse_render
    ) as backend:
        assert backend.num_workers == (os.cpu_count() or 1)


# ---------------------------------------------------------------------------
# Dexed-gated: real renders through the parallel pool match the serial backend and
# are non-silent. Skips when the plugin is absent.
# ---------------------------------------------------------------------------
PLUGIN_PATH = os.path.expanduser(config.DEXED_PATH)
needs_plugin = pytest.mark.skipif(
    not os.path.exists(PLUGIN_PATH), reason=f"Dexed plugin not found at {PLUGIN_PATH}"
)


@needs_plugin
def test_parallel_matches_serial_with_real_dexed():
    from synth.dexed import DexedWrapper

    synth = DexedWrapper(PLUGIN_PATH, sample_rate=config.SAMPLE_RATE, buffer_size=config.BUFFER_SIZE)
    space = synth.parameter_space
    settings = RenderSettings.from_config()
    patches = [space.sample_uniform(np.random.default_rng(seed)) for seed in (0, 1)]

    with FreshProcessRenderBackend(settings) as serial:
        serial_audio = [serial.render(patch) for patch in patches]
    with ParallelFreshProcessRenderBackend(settings, num_workers=2) as parallel:
        parallel_audio = parallel.render_batch(patches)

    for reference, candidate in zip(serial_audio, parallel_audio):
        assert np.max(np.abs(candidate)) > 0.0
        np.testing.assert_allclose(candidate, reference, atol=1e-6)


@needs_plugin
def test_reuse_backend_renders_non_silent_real_dexed():
    from synth.dexed import DexedWrapper

    synth = DexedWrapper(PLUGIN_PATH, sample_rate=config.SAMPLE_RATE, buffer_size=config.BUFFER_SIZE)
    space = synth.parameter_space
    settings = RenderSettings.from_config()
    patches = [space.sample_uniform(np.random.default_rng(seed)) for seed in (0, 1, 2)]

    with ParallelInProcessRenderBackend(settings, num_workers=2) as backend:
        rendered = backend.render_batch(patches)

    assert len(rendered) == len(patches)
    for audio in rendered:
        assert np.max(np.abs(audio)) > 0.0


# ---------------------------------------------------------------------------
# The synth registry: which synth a backend renders, and which backends each synth allows.
# ---------------------------------------------------------------------------
def test_unknown_synth_is_rejected_in_the_parent_process():
    # Not inside a spawned worker, where the traceback would be far less useful.
    for backend_class in (FreshProcessRenderBackend, ParallelFreshProcessRenderBackend):
        with pytest.raises(ValueError, match="Unknown synth"):
            backend_class(SETTINGS, synth_name="moog")


def test_payload_carries_the_synth_name():
    backend = ParallelFreshProcessRenderBackend(
        SETTINGS, num_workers=1, render_worker=fake_render, synth_name="diva"
    )
    try:
        assert backend._payload({"AMP": 0.5})[3] == "diva"
    finally:
        backend.close()


def test_in_process_backends_refuse_a_synth_that_cannot_reproduce_in_process():
    # D-DIVA-RENDER: Diva does not reproduce in-process at all, so the reuse backends must
    # refuse it rather than quietly produce a corpus that cannot be reproduced.
    from dataset.render_backends import InProcessRenderBackend

    class _NoInProcessSynth:
        supports_in_process_render = False

    with pytest.raises(ValueError, match="does not reproduce in-process"):
        InProcessRenderBackend(_NoInProcessSynth(), SETTINGS)

    with pytest.raises(ValueError, match="does not reproduce in-process"):
        ParallelInProcessRenderBackend(
            SETTINGS, synth_name="diva",
            worker_initializer=noop_reuse_init, render_worker=fake_reuse_render,
        )


def test_dexed_still_allows_the_in_process_backends():
    from synth.dexed import DexedWrapper
    assert DexedWrapper.supports_in_process_render is True
