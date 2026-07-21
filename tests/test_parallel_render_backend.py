"""Tests for the parallel fresh-process render backend (SynthRL RL reward path, Step 4).

Verifies the three properties the plan asks for: parallel renders equal serial renders,
renders are non-silent, and per-render process isolation is preserved (every render lands on
its own single-use process). The parallel/serial-equality and isolation tests use a picklable
VST-free stand-in worker so they run anywhere; a Dexed-gated test confirms real renders are
non-silent. Workers are module-level so they survive the spawn pickle.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from dataset.render_backends import (
    FreshProcessRenderBackend,
    ParallelFreshProcessRenderBackend,
    RenderSettings,
)

SAMPLE_RATE = 8000
SETTINGS = RenderSettings(midi_note=60, velocity=100, duration_sec=0.25, note_duration_sec=0.25)


def _sine_for(amp: float) -> np.ndarray:
    samples = int(SETTINGS.duration_sec * SAMPLE_RATE)
    time = np.arange(samples) / SAMPLE_RATE
    return (amp * np.sin(2.0 * np.pi * 220.0 * time)).astype(np.float32)


def fake_render(payload) -> np.ndarray:
    """A deterministic, VST-free render worker: a sine whose amplitude is the patch's ``AMP``."""
    patch, _settings, _renderer = payload
    return _sine_for(float(patch["AMP"]))


def pid_render(payload) -> np.ndarray:
    """A worker that reports the OS process it ran in (first sample = pid), to probe isolation."""
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


def test_defaults_to_cpu_count_workers():
    with ParallelFreshProcessRenderBackend(SETTINGS, render_worker=fake_render) as backend:
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
