"""
Benchmark DawDreamer vs Pedalboard as renderers for Dexed.

Primary question: which engine renders faster (this gates dataset generation)?
Secondary question: do the two engines produce the same audio for the same patch
(a host-robustness / threat-to-validity check)?

The same seeded patches (over the provisional ParameterSpace subset) are rendered through both
renderers with identical sample rate, MIDI note, velocity, and durations. Renderers are never
mixed within a real dataset/eval run (D-REPRO, docs/DECISIONS.md); rendering the same patches
through both here is a deliberate, separate comparison.

Run:
    pip install pedalboard
    python scripts/benchmark_renderers.py [--num-patches N] [--seed S]

The agreement metrics (log-spectral distance, spectral convergence, RMS difference) are a
preview of the future Layer-4 metric panel and are computed locally with scipy only.
"""
import argparse
import contextlib
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import stft

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from synth.dexed import DexedWrapper

# A random subset patch can be near-silent; agreement metrics on silence are meaningless, so
# such patches are excluded from the agreement table (but still timed for speed).
MIN_AMPLITUDE = 1e-3
_EPSILON = 1e-10


@contextlib.contextmanager
def suppressed_stderr():
    """Silence the benign JUCE 'invalid URI' notice the VST3 host writes to the
    OS stderr file descriptor while Dexed loads. Only fd 2 is redirected, so real
    Python exceptions and tracebacks are unaffected."""
    saved_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull_fd)
        os.close(saved_fd)


def build_patches(reference: DexedWrapper, num_patches: int, seed: int) -> List[Dict[str, float]]:
    """Sample num_patches synth-side dicts over the subset, deterministically."""
    rng = np.random.default_rng(seed)
    space = reference.parameter_space
    return [space.sample_uniform(rng) for _ in range(num_patches)]


def time_renderer(
    renderer: str, patches: List[Dict[str, float]]
) -> Tuple[Optional[DexedWrapper], List[np.ndarray], List[float], float]:
    """
    Construct a renderer-backed wrapper, render every patch, and time it.

    Returns (wrapper, mono_renders, per_render_seconds, construction_seconds). The first render
    is timed but reported separately as warm-up. Returns (None, [], [], nan) if the renderer
    cannot be constructed (e.g. pedalboard not installed).
    """
    plugin_path = os.path.expanduser(config.DEXED_PATH)
    construction_start = time.perf_counter()
    try:
        with suppressed_stderr():
            wrapper = DexedWrapper(
                plugin_path=plugin_path,
                sample_rate=config.SAMPLE_RATE,
                buffer_size=config.BUFFER_SIZE,
                renderer=renderer,
            )
    except Exception as error:  # noqa: BLE001 - report and skip, do not abort the whole run
        print(f"  [{renderer}] could not be constructed: {error}")
        return None, [], [], float("nan")
    construction_seconds = time.perf_counter() - construction_start

    renders: List[np.ndarray] = []
    per_render_seconds: List[float] = []
    for patch in patches:
        wrapper.set_parameters(patch)
        start = time.perf_counter()
        audio = wrapper.render_audio(
            midi_note=config.MIDI_NOTE,
            velocity=config.VELOCITY,
            duration_sec=config.DURATION_SEC,
            note_duration_sec=config.NOTE_DURATION_SEC,
        )
        per_render_seconds.append(time.perf_counter() - start)
        renders.append(audio)
    return wrapper, renders, per_render_seconds, construction_seconds


def report_speed(renderer: str, per_render_seconds: List[float], construction_seconds: float) -> None:
    if not per_render_seconds:
        print(f"  {renderer:<11}  (no renders)")
        return
    timings = np.asarray(per_render_seconds)
    warm = timings[1:] if len(timings) > 1 else timings  # drop warm-up render
    total = float(timings.sum())
    print(
        f"  {renderer:<11}  total {total:8.3f}s  | "
        f"median {np.median(warm) * 1e3:7.1f}ms  p90 {np.percentile(warm, 90) * 1e3:7.1f}ms  | "
        f"{len(warm) / warm.sum():6.1f} renders/s  | load {construction_seconds:5.2f}s"
    )


def _log_magnitude(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    _, _, spectrum = stft(audio, fs=sample_rate, nperseg=1024, noverlap=768)
    return np.log(np.abs(spectrum) + _EPSILON)


def agreement_metrics(audio_a: np.ndarray, audio_b: np.ndarray, sample_rate: int) -> Dict[str, float]:
    """Log-spectral distance (dB), spectral convergence, and normalized RMS difference."""
    length = min(len(audio_a), len(audio_b))
    audio_a, audio_b = audio_a[:length], audio_b[:length]

    magnitude_a = np.abs(stft(audio_a, fs=sample_rate, nperseg=1024, noverlap=768)[2])
    magnitude_b = np.abs(stft(audio_b, fs=sample_rate, nperseg=1024, noverlap=768)[2])

    log_spectral_distance = float(
        np.sqrt(np.mean((20.0 * (np.log10(magnitude_a + _EPSILON) - np.log10(magnitude_b + _EPSILON))) ** 2))
    )
    spectral_convergence = float(
        np.linalg.norm(magnitude_a - magnitude_b) / (np.linalg.norm(magnitude_a) + _EPSILON)
    )
    rms_a = np.sqrt(np.mean(audio_a ** 2))
    normalized_rms_difference = float(
        np.sqrt(np.mean((audio_a - audio_b) ** 2)) / (rms_a + _EPSILON)
    )
    return {
        "log_spectral_distance_db": log_spectral_distance,
        "spectral_convergence": spectral_convergence,
        "normalized_rms_difference": normalized_rms_difference,
    }


def report_agreement(
    renders_a: List[np.ndarray], renders_b: List[np.ndarray], sample_rate: int
) -> None:
    rows: List[Dict[str, float]] = []
    skipped = 0
    for audio_a, audio_b in zip(renders_a, renders_b):
        if max(np.max(np.abs(audio_a)), np.max(np.abs(audio_b))) < MIN_AMPLITUDE:
            skipped += 1
            continue
        rows.append(agreement_metrics(audio_a, audio_b, sample_rate))

    if not rows:
        print("  (no non-silent patches to compare)")
        return

    print(f"  compared {len(rows)} patches ({skipped} near-silent skipped)")
    for metric in ("log_spectral_distance_db", "spectral_convergence", "normalized_rms_difference"):
        values = np.asarray([row[metric] for row in rows])
        print(
            f"  {metric:<28} mean {values.mean():8.4f}  median {np.median(values):8.4f}  "
            f"p90 {np.percentile(values, 90):8.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-patches", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    plugin_path = os.path.expanduser(config.DEXED_PATH)
    if not os.path.exists(plugin_path):
        print(f"Could not find Dexed plugin at: {plugin_path}. Update DEXED_PATH in .env.")
        sys.exit(1)

    # Use a DawDreamer-backed wrapper purely to define the patch set (subset is engine-agnostic).
    with suppressed_stderr():
        reference = DexedWrapper(
            plugin_path=plugin_path,
            sample_rate=config.SAMPLE_RATE,
            buffer_size=config.BUFFER_SIZE,
        )
    patches = build_patches(reference, args.num_patches, args.seed)
    print(
        f"Benchmarking {args.num_patches} patches @ {config.SAMPLE_RATE}Hz, "
        f"{config.DURATION_SEC}s render (note {config.NOTE_DURATION_SEC}s), seed {args.seed}\n"
    )

    results: Dict[str, Tuple[Optional[DexedWrapper], List[np.ndarray], List[float], float]] = {}
    for renderer in ("dawdreamer", "pedalboard"):
        print(f"Rendering with {renderer}...")
        results[renderer] = time_renderer(renderer, patches)

    print("\n=== Render speed (warm-up render excluded from median/p90/throughput) ===")
    for renderer in ("dawdreamer", "pedalboard"):
        _, _, per_render_seconds, construction_seconds = results[renderer]
        report_speed(renderer, per_render_seconds, construction_seconds)

    totals = {
        renderer: sum(results[renderer][2]) for renderer in results if results[renderer][2]
    }
    if len(totals) == 2:
        faster = min(totals, key=totals.get)
        slower = max(totals, key=totals.get)
        ratio = totals[slower] / totals[faster]
        print(f"\n  -> {faster} is {ratio:.2f}x faster in total render time.")

    print("\n=== Audio agreement (DawDreamer vs Pedalboard, same patch) ===")
    daw_renders = results["dawdreamer"][1]
    pedalboard_renders = results["pedalboard"][1]
    if daw_renders and pedalboard_renders:
        report_agreement(daw_renders, pedalboard_renders, config.SAMPLE_RATE)
    else:
        print("  (need both renderers available to compare)")


if __name__ == "__main__":
    main()
