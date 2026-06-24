"""
Benchmark three rendering strategies for Dexed.

Primary question: how much slower is reloading the plugin on every render (the
preset-gen-vae approach) than reusing one instance, and how does either compare to
Pedalboard? Secondary question: do the strategies produce the same audio for the same
patch -- and, in particular, does reload-per-render make DawDreamer agree with another
engine better than reuse does (a *causal/interventional* version of the cross-engine
agreement question, where the earlier confirmatory test was only correlational)?

The arms (all on the same seeded patches, identical sample rate / MIDI note /
velocity / durations):

  1. dawdreamer-reuse   -- one persistent DexedWrapper, all patches rendered through it
                           (the default, the engine all D-REPRO work was done on).
  2. dawdreamer-reload  -- a fresh DexedWrapper constructed per render (engine rebuild +
                           name-map re-resolve + defaults re-read), previous wrapper
                           dropped first. In-process; faithful to preset-gen-vae's
                           reload-per-render (data/dexeddataset.py:243).
  3. pedalboard         -- one persistent Pedalboard-backed DexedWrapper.
  4. subprocess         -- (optional, --subprocess) each patch rendered at position 0 of a
                           *fresh OS process* (spawn start method, maxtasksperchild=1), so
                           every render starts from a clean heap. Run twice (subprocess-a /
                           subprocess-b) so the two independent realizations can be compared:
                           D-REPRO predicts they agree to ~0 even on the sensitive patches that
                           keep a full tail under the in-process arms. A fresh process is the
                           only context that resets the hidden voice state (in-process reload only
                           re-diverges); it is used here as the clean *reference*, not an
                           engine-level fix -- the project accepts and documents the leak rather
                           than fixing it (D-REPRO policy, docs/DECISIONS.md).
                           It is the slow arm -- every render pays a full process spawn + plugin
                           load; expect many minutes at N=3000.

Renderers are never mixed within a real dataset/eval run (D-REPRO, docs/DECISIONS.md);
rendering the same patches through every arm here is a deliberate, separate comparison.

Run:
    pip install pedalboard
    python scripts/benchmark_renderers.py [--num-patches N] [--seed S] [--subprocess] [--dump-agreement-csv PATH]

The agreement metrics (log-spectral distance, spectral convergence, RMS difference) are a
preview of the future Layer-4 metric panel and are computed locally with scipy only.
"""
import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import stft

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from synth.dexed import DexedWrapper, suppressed_stderr
from synth.dexed.cartridge import NUM_VOICES, validate_cartridge, voice_names, voice_parameters
from dataset.render_backends import (
    RenderSettings,
    _make_wrapper,
    render_patch_in_fresh_process,
)

# A patch can be near-silent (a random subset draw, or an intrinsically quiet preset);
# agreement metrics on silence are meaningless, so such patches are excluded from the
# agreement table (but still timed for speed).
MIN_AMPLITUDE = 1e-3
_EPSILON = 1e-10

# Standard Dexed cartridge install location (same default render_preset.py uses); overridable
# with --cartridges. Pointing at a directory loads every valid 32-voice .syx under it.
DEFAULT_CARTRIDGES_DIR = os.path.expanduser(
    "~/Library/Application Support/DigitalSuburban/Dexed/Cartridges"
)


def build_patches(reference: DexedWrapper, num_patches: int, seed: int) -> List[Dict[str, float]]:
    """Sample num_patches synth-side dicts over the subset, deterministically."""
    rng = np.random.default_rng(seed)
    space = reference.parameter_space
    return [space.sample_uniform(rng) for _ in range(num_patches)]


def build_cartridge_patches(cartridges_dir: str) -> Tuple[List[Dict[str, float]], List[str]]:
    """Load every voice from every valid 32-voice .syx cartridge under cartridges_dir.

    Each patch is a full synth-side voice dict (the named DX7 voice applied as Dexed
    parameters -- the only way to load a cartridge voice offline; see synth/dexed/cartridge.py).
    Unlike the random subset, these vary *all* voice parameters, not just the provisional subset.

    Returns (patches, labels), where each label is ``<cartridge stem>:<voice index> <voice name>``.
    Files that are not 32-voice bulk dumps (e.g. single-voice dumps) are skipped with a notice.
    """
    paths = sorted(Path(cartridges_dir).rglob("*.syx")) if os.path.isdir(cartridges_dir) \
        else [Path(cartridges_dir)]
    patches: List[Dict[str, float]] = []
    labels: List[str] = []
    skipped = 0
    for path in paths:
        data = path.read_bytes()
        try:
            validate_cartridge(data)
        except ValueError:
            skipped += 1
            print(f"  skipping (not a 32-voice bulk dump): {path.name}")
            continue
        names = voice_names(data)
        for voice_index in range(NUM_VOICES):
            patches.append(voice_parameters(data, voice_index))
            labels.append(f"{path.stem}:{voice_index:02d} {names[voice_index].strip()}")
    print(
        f"  loaded {len(patches)} voices from {len(paths) - skipped} cartridge(s) "
        f"under {cartridges_dir}" + (f" ({skipped} file(s) skipped)" if skipped else "")
    )
    return patches, labels


def _render(wrapper: DexedWrapper) -> np.ndarray:
    return wrapper.render_audio(
        midi_note=config.MIDI_NOTE,
        velocity=config.VELOCITY,
        duration_sec=config.DURATION_SEC,
        note_duration_sec=config.NOTE_DURATION_SEC,
    )


def time_reuse_arm(
    renderer: str, patches: List[Dict[str, float]]
) -> Tuple[List[np.ndarray], List[float], float]:
    """
    Persistent-instance arm: construct one wrapper, render every patch through it.

    Returns (renders, per_render_seconds, load_seconds). Returns ([], [], nan) if the
    renderer cannot be constructed (e.g. pedalboard not installed).
    """
    load_start = time.perf_counter()
    try:
        with suppressed_stderr():
            wrapper = _make_wrapper(renderer)
    except Exception as error:  # noqa: BLE001 - report and skip, do not abort the whole run
        print(f"  [{renderer}] could not be constructed: {error}")
        return [], [], float("nan")
    load_seconds = time.perf_counter() - load_start

    renders: List[np.ndarray] = []
    per_render_seconds: List[float] = []
    for patch in patches:
        wrapper.set_parameters(patch)
        start = time.perf_counter()
        audio = _render(wrapper)
        per_render_seconds.append(time.perf_counter() - start)
        renders.append(audio)
    return renders, per_render_seconds, load_seconds


def time_reload_arm(
    patches: List[Dict[str, float]],
) -> Tuple[List[np.ndarray], List[float], List[float], List[float]]:
    """
    Reload-per-render arm (the preset-gen-vae approach): render every patch through a
    freshly-constructed DawDreamer-backed wrapper, dropping the previous one first so each
    render starts from a wrapper teardown. The reload (construction) and the render are timed
    separately.

    Returns (renders, reload_seconds, render_seconds, total_seconds), one entry per patch.
    """
    renders: List[np.ndarray] = []
    reload_seconds: List[float] = []
    render_seconds: List[float] = []
    total_seconds: List[float] = []

    wrapper: Optional[DexedWrapper] = None
    for patch in patches:
        wrapper = None  # free the previous engine before rebuilding (faithful teardown)
        reload_start = time.perf_counter()
        with suppressed_stderr():
            wrapper = _make_wrapper("dawdreamer")
        reload_elapsed = time.perf_counter() - reload_start

        wrapper.set_parameters(patch)
        render_start = time.perf_counter()
        audio = _render(wrapper)
        render_elapsed = time.perf_counter() - render_start

        renders.append(audio)
        reload_seconds.append(reload_elapsed)
        render_seconds.append(render_elapsed)
        total_seconds.append(reload_elapsed + render_elapsed)
    return renders, reload_seconds, render_seconds, total_seconds


def time_subprocess_arm(
    patches: List[Dict[str, float]],
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Fresh-OS-process-per-render arm: render every patch in its own spawned process.

    Uses a single-worker pool with ``maxtasksperchild=1`` and the **spawn** start method, so the
    worker is torn down and a clean interpreter is spawned for each patch -- a genuinely fresh
    heap per render (never **fork**, which would inherit the parent's dirty memory). Serial, so
    the per-render timing is comparable to the other arms (each entry is one process's
    spawn + plugin load + render wall-clock; the first entry also carries pool startup and is
    treated as warm-up).

    Returns (renders, per_render_seconds).
    """
    context = mp.get_context("spawn")
    settings = RenderSettings.from_config()
    payloads = [(patch, settings, "dawdreamer") for patch in patches]
    renders: List[np.ndarray] = []
    per_render_seconds: List[float] = []
    last = time.perf_counter()
    with context.Pool(processes=1, maxtasksperchild=1) as pool:
        for audio in pool.imap(render_patch_in_fresh_process, payloads):
            now = time.perf_counter()
            per_render_seconds.append(now - last)
            last = now
            renders.append(np.asarray(audio))
    return renders, per_render_seconds


def _summarize(per_render_seconds: List[float]) -> Optional[Dict[str, float]]:
    """Total wall-clock (incl. warm-up) plus median/p90/throughput over the warm renders
    (the first render is dropped as warm-up)."""
    if not per_render_seconds:
        return None
    timings = np.asarray(per_render_seconds)
    warm = timings[1:] if len(timings) > 1 else timings
    return {
        "total": float(timings.sum()),
        "median_ms": float(np.median(warm) * 1e3),
        "p90_ms": float(np.percentile(warm, 90) * 1e3),
        "throughput": float(len(warm) / warm.sum()) if warm.sum() > 0 else float("nan"),
    }


def _median_warm_ms(seconds: List[float]) -> float:
    timings = np.asarray(seconds)
    warm = timings[1:] if len(timings) > 1 else timings
    return float(np.median(warm) * 1e3)


def report_speed_line(
    label: str, summary: Optional[Dict[str, float]],
    load_label: str = "load", load_seconds: Optional[float] = None,
) -> None:
    if summary is None:
        print(f"  {label:<18}  (no renders)")
        return
    # The subprocess arm has no single load step (spawn + load is folded into every render),
    # so it passes load_seconds=None and the load column is omitted.
    load_tail = f"  | {load_label} {load_seconds:5.2f}s" if load_seconds is not None else ""
    print(
        f"  {label:<18}  total {summary['total']:8.3f}s  | "
        f"median {summary['median_ms']:7.1f}ms  p90 {summary['p90_ms']:7.1f}ms  | "
        f"{summary['throughput']:6.1f} renders/s{load_tail}"
    )


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


_AGREEMENT_METRICS = (
    "log_spectral_distance_db",
    "spectral_convergence",
    "normalized_rms_difference",
)

# The core arm-pairs compared for agreement. The third (reload vs pedalboard) against the
# first (reuse vs pedalboard) is the interventional test: if reload escapes the hidden voice
# state, its tail collapses relative to reuse on the same patches.
_PAIRS = (
    ("reuse_vs_pedalboard", "dawdreamer-reuse", "pedalboard"),
    ("reload_vs_pedalboard", "dawdreamer-reload", "pedalboard"),
    ("reuse_vs_reload", "dawdreamer-reuse", "dawdreamer-reload"),
)

# Extra pairs enabled by --subprocess. ``subprocess_a_vs_b`` is the *positive* control: two
# independent fresh-process realizations of every patch. D-REPRO predicts this tail collapses to
# ~0 (fresh processes are deterministic) even where the in-process arms keep a full tail -- the
# demonstration that fresh-process isolation, not in-process reload, is what resets the state
# (the clean reference the project documents against, not an engine-level fix).
# ``reuse_vs_subprocess`` contrasts the default (context-accumulating) path with the clean one.
_SUBPROCESS_PAIRS = (
    ("subprocess_a_vs_b", "subprocess-a", "subprocess-b"),
    ("reuse_vs_subprocess", "dawdreamer-reuse", "subprocess-a"),
)


def compute_agreement_rows(
    renders_a: List[np.ndarray], renders_b: List[np.ndarray], sample_rate: int
) -> Tuple[List[Dict[str, float]], int]:
    """Per-patch agreement metrics for every patch with signal in at least one arm.

    Returns (rows, skipped). Each row carries the original ``patch_index``; patches
    near-silent in *both* arms (agreement on silence is meaningless) are counted in
    ``skipped`` and excluded. This per-pair mask matches the original 2-arm benchmark, so
    the reuse-vs-pedalboard table reproduces the recorded D-RENDERER numbers.
    """
    rows: List[Dict[str, float]] = []
    skipped = 0
    for patch_index, (audio_a, audio_b) in enumerate(zip(renders_a, renders_b)):
        if max(np.max(np.abs(audio_a)), np.max(np.abs(audio_b))) < MIN_AMPLITUDE:
            skipped += 1
            continue
        metrics = agreement_metrics(audio_a, audio_b, sample_rate)
        metrics["patch_index"] = patch_index
        rows.append(metrics)
    return rows, skipped


def report_agreement(title: str, rows: List[Dict[str, float]], skipped: int) -> None:
    print(f"  {title}")
    if not rows:
        print("    (no non-silent patches to compare)")
        return
    print(f"    compared {len(rows)} patches ({skipped} near-silent skipped)")
    for metric in _AGREEMENT_METRICS:
        values = np.asarray([row[metric] for row in rows])
        print(
            f"    {metric:<28} mean {values.mean():8.4f}  median {np.median(values):8.4f}  "
            f"p90 {np.percentile(values, 90):8.4f}  p95 {np.percentile(values, 95):8.4f}"
        )


def compute_three_way_rows(
    arm_renders: Dict[str, List[np.ndarray]], sample_rate: int,
    labels: Optional[List[str]] = None,
    pairs: Tuple[Tuple[str, str, str], ...] = _PAIRS,
) -> Tuple[List[Dict[str, float]], int]:
    """Per-patch agreement across the requested arm-pairs, on patches non-silent in *all*
    arms that participate in any pair (a single shared intersection mask, so every metric
    column is signal-vs-signal and the pairs line up per patch for the tail-collapse
    analysis).

    Returns (rows, skipped). Each row: ``patch_index`` + ``{pair}__{metric}`` for every pair
    in ``pairs`` and every metric in ``_AGREEMENT_METRICS``. Arms with no renders (e.g.
    pedalboard not installed, or the subprocess arm when --subprocess is off) drop the pairs
    that need them.
    """
    available_pairs = [
        (name, arm_x, arm_y)
        for (name, arm_x, arm_y) in pairs
        if arm_renders.get(arm_x) and arm_renders.get(arm_y)
    ]
    if not available_pairs:
        return [], 0
    participating_arms = sorted({arm for _, x, y in available_pairs for arm in (x, y)})
    num_patches = min(len(arm_renders[arm]) for arm in participating_arms)

    rows: List[Dict[str, float]] = []
    skipped = 0
    for patch_index in range(num_patches):
        peaks = [float(np.max(np.abs(arm_renders[arm][patch_index]))) for arm in participating_arms]
        if min(peaks) < MIN_AMPLITUDE:  # silent in at least one participating arm
            skipped += 1
            continue
        row: Dict[str, float] = {"patch_index": patch_index}
        if labels is not None:
            row["patch_label"] = labels[patch_index]
        for name, arm_x, arm_y in available_pairs:
            metrics = agreement_metrics(
                arm_renders[arm_x][patch_index], arm_renders[arm_y][patch_index], sample_rate
            )
            for metric_name, value in metrics.items():
                row[f"{name}__{metric_name}"] = value
        rows.append(row)
    return rows, skipped


def write_three_way_csv(rows: List[Dict[str, float]], path: str) -> None:
    """Write the combined per-patch agreement rows (one row per patch non-silent across all
    participating arms) to CSV. Columns: ``patch_index`` + ``{pair}__{metric}`` for the three
    pairs. This is the regenerable data source for the interventional / tail-collapse analysis,
    computed with the same functions that produce the printed tables, so it cannot drift."""
    import csv

    if not rows:
        print("  (no rows to write -- need at least two arms with non-silent renders)")
        return
    fieldnames = ["patch_index"] + [key for key in rows[0] if key != "patch_index"]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"  wrote {len(rows)} per-patch rows -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-patches", type=int, default=200,
                        help="Number of random subset patches (ignored when --cartridges is set).")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for random patches (ignored when --cartridges is set).")
    parser.add_argument(
        "--cartridges", nargs="?", const=DEFAULT_CARTRIDGES_DIR, default=None,
        help="Benchmark real DX7 cartridge voices instead of random patches. Pass a .syx file "
             "or a directory of them; bare --cartridges uses the standard Dexed install location.",
    )
    parser.add_argument(
        "--subprocess", action="store_true",
        help="Add a fresh-OS-process-per-render arm (rendered twice, subprocess-a/-b, for the "
             "tail-collapse control). This is the only arm that resets the hidden voice state -- "
             "the clean reference, not an engine-level fix -- but the slow arm: every render pays "
             "a process spawn + plugin load.",
    )
    parser.add_argument(
        "--dump-agreement-csv",
        default=None,
        help="If set, write the combined per-patch agreement metrics to this CSV path.",
    )
    args = parser.parse_args()

    plugin_path = os.path.expanduser(config.DEXED_PATH)
    if not os.path.exists(plugin_path):
        print(f"Could not find Dexed plugin at: {plugin_path}. Update DEXED_PATH in .env.")
        sys.exit(1)

    labels: Optional[List[str]] = None
    if args.cartridges is not None:
        if not os.path.exists(args.cartridges):
            print(f"Cartridge path not found: {args.cartridges}")
            sys.exit(1)
        print(f"Loading DX7 cartridge voices from {args.cartridges}")
        patches, labels = build_cartridge_patches(args.cartridges)
        if not patches:
            print("No usable cartridge voices found.")
            sys.exit(1)
        source = f"{len(patches)} cartridge voices"
    else:
        # Use a DawDreamer-backed wrapper purely to define the patch set (subset is engine-agnostic).
        with suppressed_stderr():
            reference = DexedWrapper(
                plugin_path=plugin_path,
                sample_rate=config.SAMPLE_RATE,
                buffer_size=config.BUFFER_SIZE,
            )
        patches = build_patches(reference, args.num_patches, args.seed)
        del reference
        source = f"{args.num_patches} random patches (seed {args.seed})"
    print(
        f"Benchmarking {source} @ {config.SAMPLE_RATE}Hz, "
        f"{config.DURATION_SEC}s render (note {config.NOTE_DURATION_SEC}s)\n"
    )

    print("Rendering dawdreamer-reuse...")
    reuse_renders, reuse_per_render, reuse_load = time_reuse_arm("dawdreamer", patches)
    print("Rendering dawdreamer-reload (fresh plugin per render -- this is the slow arm)...")
    reload_renders, reload_reload_s, reload_render_s, reload_total_s = time_reload_arm(patches)
    print("Rendering pedalboard...")
    pedalboard_renders, pedalboard_per_render, pedalboard_load = time_reuse_arm("pedalboard", patches)

    subprocess_a_per_render: List[float] = []
    if args.subprocess:
        print(f"Rendering subprocess-a (fresh OS process per render, {len(patches)} patches -- "
              "the slow arm)...")
        subprocess_a_renders, subprocess_a_per_render = time_subprocess_arm(patches)
        print("Rendering subprocess-b (independent second pass, for the tail-collapse control)...")
        subprocess_b_renders, _ = time_subprocess_arm(patches)
    else:
        subprocess_a_renders, subprocess_b_renders = [], []

    arm_renders = {
        "dawdreamer-reuse": reuse_renders,
        "dawdreamer-reload": reload_renders,
        "pedalboard": pedalboard_renders,
        "subprocess-a": subprocess_a_renders,
        "subprocess-b": subprocess_b_renders,
    }
    pairs = _PAIRS + (_SUBPROCESS_PAIRS if args.subprocess else ())

    # ---- Speed --------------------------------------------------------------------------
    print("\n=== Render speed ===")
    print("  (total = full wall-clock incl. warm-up; median/p90/throughput drop the first render)\n")
    reuse_summary = _summarize(reuse_per_render)
    reload_summary = _summarize(reload_total_s)
    pedalboard_summary = _summarize(pedalboard_per_render)
    subprocess_summary = _summarize(subprocess_a_per_render)

    report_speed_line("dawdreamer-reuse", reuse_summary, "load", reuse_load)
    report_speed_line("dawdreamer-reload", reload_summary, "reload med", _median_warm_ms(reload_reload_s) / 1e3)
    if reload_total_s:
        print(
            f"      decomposition: reload {_median_warm_ms(reload_reload_s):7.1f}ms + "
            f"render {_median_warm_ms(reload_render_s):7.1f}ms per render"
            + (f"  (reuse render {reuse_summary['median_ms']:.1f}ms)" if reuse_summary else "")
        )
    report_speed_line("pedalboard", pedalboard_summary, "load", pedalboard_load)
    if subprocess_summary is not None:
        # No single load step: spawn + plugin load is folded into every render (load column omitted).
        report_speed_line("subprocess-a", subprocess_summary)

    def _total(summary: Optional[Dict[str, float]]) -> str:
        return f"{summary['total']:.1f}s" if summary else "n/a"

    print("\n  Headline -- total wall-clock to render all patches:")
    print(
        f"    reuse {_total(reuse_summary)}  |  "
        f"reload {_total(reload_summary)}  |  "
        f"pedalboard {_total(pedalboard_summary)}"
        + (f"  |  subprocess {_total(subprocess_summary)}" if subprocess_summary else "")
    )
    print("\n  Headline -- median per-render (portable; total wall-clock is noisier):")
    if reuse_summary and reload_summary:
        print(f"    reload is {reload_summary['median_ms'] / reuse_summary['median_ms']:.2f}x slower than reuse")
    if reuse_summary and pedalboard_summary:
        print(f"    reuse is {pedalboard_summary['median_ms'] / reuse_summary['median_ms']:.2f}x faster than pedalboard")
    if reuse_summary and subprocess_summary:
        print(f"    subprocess (fresh process) is {subprocess_summary['median_ms'] / reuse_summary['median_ms']:.2f}x slower than reuse")

    # ---- Agreement ----------------------------------------------------------------------
    print("\n=== Audio agreement (same patch, different arm) ===")
    for pair_name, arm_x, arm_y in pairs:
        renders_x, renders_y = arm_renders[arm_x], arm_renders[arm_y]
        if renders_x and renders_y:
            rows, skipped = compute_agreement_rows(renders_x, renders_y, config.SAMPLE_RATE)
            report_agreement(f"{arm_x}  vs  {arm_y}  [{pair_name}]", rows, skipped)
        else:
            print(f"  {arm_x}  vs  {arm_y}  [{pair_name}]\n    (need both arms available to compare)")
    if args.subprocess:
        print("\n  subprocess_a_vs_b ~ 0 confirms fresh-process renders are context-independent\n"
              "  (fresh processes reset the state); reuse_vs_subprocess shows the context the default\n"
              "  path carries, while reuse_vs_reload keeps a full tail (in-process reload does not\n"
              "  escape it).")

    # ---- Combined per-patch CSV (shared intersection mask) ------------------------------
    three_way_rows, three_way_skipped = compute_three_way_rows(
        arm_renders, config.SAMPLE_RATE, labels=labels, pairs=pairs
    )
    print(
        f"\n  3-way per-patch set (non-silent across all participating arms): "
        f"{len(three_way_rows)} patches ({three_way_skipped} skipped)"
    )

    # With named cartridge voices, surface which real presets diverge most across methods.
    rank_key = "reuse_vs_pedalboard__log_spectral_distance_db"
    if labels is not None and three_way_rows and rank_key in three_way_rows[0]:
        print("\n  Most cross-method-divergent presets (reuse vs pedalboard, log-spectral dB):")
        worst = sorted(three_way_rows, key=lambda row: row[rank_key], reverse=True)[:10]
        for row in worst:
            print(f"    {row[rank_key]:6.2f} dB  [{row['patch_index']:4d}]  {row['patch_label']}")

    if args.dump_agreement_csv:
        write_three_way_csv(three_way_rows, args.dump_agreement_csv)


if __name__ == "__main__":
    main()
