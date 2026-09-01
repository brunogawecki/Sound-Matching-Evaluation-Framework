"""Probe: does a render's opening carry the *previous* patch?

Diva smooths parameter changes over roughly 126 ms of audio. ``set_parameters`` writes the values
and the note starts at sample 0, so the opening of a render is a crossfade out of whatever the
plugin held before -- in a fresh process, the init patch. Every corpus render is fresh-process
(D-DIVA-RENDER), so every render is a "render 1" and every sample carries it, on the attack.

Five probes, each runnable for either synth via ``--synth``:

A  opening contamination -- a patch that cannot sound, rendered fresh-process: peak in the first
   50 ms against peak in the body, and correlation of that opening with an init-patch render's.
B  cross-patch opening similarity -- mean pairwise correlation over the first 50 ms of unrelated
   patches, against a mid-render window. Unrelated patches should not agree anywhere.
C  warm-up sweep -- probe A's first-50 ms peak with a warm-up of 0 .. 0.3 s. Always overrides
   the wrapper's own setting; the other probes measure the shipped path unless --warmup-sec says
   otherwise.
D  in-process divergence, renders 1..5 through one wrapper. D-DIVA-RENDER's recorded table
   compares render 1 against 2-4, so it conflates this bug with Diva's analog drift; the
   3-vs-2 / 4-vs-2 / 5-vs-2 rows here are the drift on its own.
E  cross-process bit-identity with the warm-up applied -- two independent processes, sha256.

Run:
    python scripts/measure_diva_warmup.py --synth diva
    python scripts/measure_diva_warmup.py --synth dexed
"""
import argparse
import hashlib
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from benchmark_renderers import agreement_metrics
from dataset.render_backends import RenderSettings, make_wrapper, _synth_spec

# The window the smoothing lives in, and a mid-render window to contrast it against.
_OPENING_SEC = 0.05
_MID_WINDOW_START_SEC = 1.0
# Well clear of the ~126 ms smoothing, so 'body' means the patch itself.
_BODY_START_SEC = 0.5

_WARMUP_SWEEP_SEC = (0.0, 0.05, 0.10, 0.20, 0.30)

# Diva: every oscillator off and the VCA closed. Dexed: every operator silent. Neither patch
# can produce sound, so anything in its render came from somewhere else.
_DIVA_SILENCERS = {
    "OSC.Triangle1On": 0.0, "OSC.Sine2On": 0.0, "OSC.Saw1On": 0.0, "OSC.Pwm1On": 0.0,
    "OSC.Triangle2On": 0.0, "OSC.Saw2On": 0.0, "OSC.Pulse2On": 0.0, "OSC.PWM2On": 0.0,
    "OSC.Noise1On": 0.0, "OSC.Volume1": 0.0, "OSC.Volume2": 0.0, "OSC.Volume3": 0.0,
    "OSC.NoiseVol": 0.0, "VCA1.Volume": 0.0,
}
_DEXED_SILENCERS = {f"OP{op} OUTPUT LEVEL": 0.0 for op in range(1, 7)}

_SILENCERS = {"diva": _DIVA_SILENCERS, "dexed": _DEXED_SILENCERS}


def _override_warmup(synth_name: str, warmup_sec: Optional[float]) -> float:
    """Force the wrapper's warm-up to ``warmup_sec``; return what is left to do by hand.

    ``None`` leaves the shipped wrapper alone, so a default run measures the real render path.
    A value overrides it, which is how the sweep works and how the pre-fix numbers are
    reproduced. On code that predates ``_WARMUP_SEC`` there is nothing to override, so the
    caller renders the warm-up itself and the probe reads the same either way.
    """
    if warmup_sec is None:
        return 0.0
    if synth_name == "diva":
        from synth.diva import synth as diva_synth
        if hasattr(diva_synth, "_WARMUP_SEC"):
            diva_synth._WARMUP_SEC = warmup_sec
            return 0.0
    return warmup_sec


def render_in_fresh_process(
    payload: Tuple[Dict[str, float], RenderSettings, str, str, Optional[float], int]
) -> np.ndarray:
    """Render one patch at position 0 of a new wrapper, under a given warm-up.

    Top-level so it survives the spawn pickle. Mirrors
    ``dataset.render_backends.render_patch_in_fresh_process``; ``warmup_sec`` of ``None`` runs
    that path unmodified.
    """
    patch, settings, renderer, synth_name, warmup_sec, num_renders = payload
    spec = _synth_spec(synth_name)
    with spec.open_output_suppressor():
        wrapper = make_wrapper(renderer, synth_name)
    manual_warmup_sec = _override_warmup(synth_name, warmup_sec)
    wrapper.set_parameters(patch)

    with spec.open_output_suppressor():
        if manual_warmup_sec > 0.0:
            wrapper._renderer.render_note(
                settings.midi_note, settings.velocity, manual_warmup_sec, manual_warmup_sec
            )
    renders = [
        np.asarray(
            wrapper.render_audio(
                settings.midi_note, settings.velocity,
                settings.duration_sec, settings.note_duration_sec,
            ),
            dtype=np.float32,
        )
        for _ in range(num_renders)
    ]
    return np.stack(renders) if num_renders > 1 else renders[0]


def _run_fresh(
    patch: Dict[str, float], settings: RenderSettings, synth_name: str,
    warmup_sec: Optional[float] = None, num_renders: int = 1,
) -> np.ndarray:
    """One spawned single-use process per call, exactly as a corpus build renders."""
    from dataset.render_backends import _worker_silencing
    context = mp.get_context("spawn")
    pool = context.Pool(processes=1, maxtasksperchild=1, **_worker_silencing(synth_name))
    try:
        return pool.apply(
            render_in_fresh_process,
            ((patch, settings, "dawdreamer", synth_name, warmup_sec, num_renders),),
        )
    finally:
        pool.terminate()
        pool.join()


def _window(audio: np.ndarray, start_sec: float, length_sec: float) -> np.ndarray:
    start = int(start_sec * config.SAMPLE_RATE)
    return audio[start:start + int(length_sec * config.SAMPLE_RATE)]


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    length = min(len(a), len(b))
    a, b = a[:length] - a[:length].mean(), b[:length] - b[:length].mean()
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denominator) if denominator > 0 else float("nan")


def _silent_patch(synth_name: str, defaults: Dict[str, float]) -> Dict[str, float]:
    patch = dict(defaults)
    for name, value in _SILENCERS[synth_name].items():
        if name not in patch:
            raise KeyError(f"{synth_name} does not expose {name!r}")
        patch[name] = value
    return patch


def probe_a_opening_contamination(
    synth_name: str, settings: RenderSettings, defaults: Dict[str, float],
    silent: Dict[str, float], warmup_sec: Optional[float]
) -> None:
    print("\n=== A. Opening contamination (patch that cannot sound, fresh process) ===")
    silent_audio = _run_fresh(silent, settings, synth_name, warmup_sec=warmup_sec)
    init_audio = _run_fresh({}, settings, synth_name, warmup_sec=warmup_sec)

    opening = _window(silent_audio, 0.0, _OPENING_SEC)
    body = silent_audio[int(_BODY_START_SEC * config.SAMPLE_RATE):]
    init_opening = _window(init_audio, 0.0, _OPENING_SEC)

    print(f"  silent patch, first {_OPENING_SEC * 1000:.0f} ms peak : {np.abs(opening).max():.6f}")
    print(f"  silent patch, peak after {_BODY_START_SEC:.1f} s        : {np.abs(body).max():.6f}")
    print(f"  init patch,   first {_OPENING_SEC * 1000:.0f} ms peak : {np.abs(init_opening).max():.6f}")
    print(f"  correlation of the two openings       : {_correlation(opening, init_opening):+.4f}")


def probe_b_cross_patch_openings(
    synth_name: str, settings: RenderSettings, patches: List[Dict[str, float]],
    warmup_sec: Optional[float]
) -> None:
    print(f"\n=== B. Cross-patch opening similarity ({len(patches)} unrelated patches) ===")
    renders = [_run_fresh(patch, settings, synth_name, warmup_sec=warmup_sec) for patch in patches]
    for label, start in (("opening", 0.0), ("mid-render", _MID_WINDOW_START_SEC)):
        windows = [_window(audio, start, _OPENING_SEC) for audio in renders]
        pairs = [
            _correlation(windows[i], windows[j])
            for i in range(len(windows)) for j in range(i + 1, len(windows))
        ]
        pairs = [value for value in pairs if not np.isnan(value)]
        mean = np.mean(pairs) if pairs else float("nan")
        maximum = np.max(pairs) if pairs else float("nan")
        print(f"  {label:11} mean pairwise r = {mean:+.4f}   max = {maximum:+.4f}")


def probe_c_warmup_sweep(
    synth_name: str, settings: RenderSettings, silent: Dict[str, float]
) -> None:
    print("\n=== C. Warm-up sweep (silent patch, first 50 ms peak) ===")
    for warmup_sec in _WARMUP_SWEEP_SEC:
        audio = _run_fresh(silent, settings, synth_name, warmup_sec=warmup_sec)
        opening = _window(audio, 0.0, _OPENING_SEC)
        print(f"  warm-up {warmup_sec:.2f} s -> peak {np.abs(opening).max():.6f}")


def probe_d_in_process_divergence(
    synth_name: str, settings: RenderSettings, patches: List[Dict[str, float]], num_renders: int,
    warmup_sec: Optional[float]
) -> None:
    print(f"\n=== D. In-process divergence, {num_renders} renders through one wrapper ===")
    print("  Comparisons against render 1 include the smoothing bug; against render 2 do not.")
    header = f"  {'patch':>5} {'pair':>8} {'maxdiff/peak':>13} {'LSD dB':>8} {'RMS drift':>10}"
    rows: Dict[str, List[Tuple[float, float, float]]] = {}
    print(header)
    for index, patch in enumerate(patches):
        renders = _run_fresh(patch, settings, synth_name, warmup_sec=warmup_sec,
                             num_renders=num_renders)
        peak = float(np.abs(renders[0]).max())
        pairs = [(2, 1)] + [(n, 2) for n in range(3, num_renders + 1)]
        for later, earlier in pairs:
            a, b = renders[later - 1], renders[earlier - 1]
            metrics = agreement_metrics(a, b, config.SAMPLE_RATE)
            relative = float(np.abs(a - b).max() / peak) if peak > 0 else float("nan")
            label = f"{later}v{earlier}"
            rows.setdefault(label, []).append(
                (relative, metrics["log_spectral_distance_db"],
                 metrics["normalized_rms_difference"])
            )
            print(f"  {index:>5} {label:>8} {relative:>13.4f} "
                  f"{metrics['log_spectral_distance_db']:>8.2f} "
                  f"{metrics['normalized_rms_difference'] * 100:>9.2f}%")
    print("\n  medians:")
    for label, values in rows.items():
        array = np.array(values)
        print(f"  {label:>8} maxdiff/peak {np.median(array[:, 0]):>8.4f}   "
              f"LSD {np.median(array[:, 1]):>6.2f} dB   "
              f"RMS drift {np.median(array[:, 2]) * 100:>7.2f}%")
    identical = sum(
        1 for values in zip(*rows.values()) if all(value[0] == 0.0 for value in values)
    ) if rows else 0
    print(f"\n  patches whose renders past the first agree exactly: {identical}/{len(patches)}")


def probe_e_cross_process_identity(
    synth_name: str, settings: RenderSettings, patches: List[Dict[str, float]],
    warmup_sec: Optional[float]
) -> None:
    label = "shipped" if warmup_sec is None else f"{warmup_sec:.2f} s"
    print(f"\n=== E. Cross-process bit-identity, warm-up {label} ===")
    for index, patch in enumerate(patches):
        digests = [
            hashlib.sha256(
                _run_fresh(patch, settings, synth_name, warmup_sec=warmup_sec).tobytes()
            ).hexdigest()
            for _ in range(2)
        ]
        verdict = "identical" if digests[0] == digests[1] else "DIFFER"
        print(f"  patch {index}: {digests[0][:16]} / {digests[1][:16]}  {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--synth", choices=sorted(_SILENCERS), default="diva")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--patches", type=int, default=8,
                        help="uniform draws from the subset used by probes B, D and E")
    parser.add_argument("--renders", type=int, default=5,
                        help="consecutive in-process renders for probe D")
    parser.add_argument("--warmup-sec", type=float, default=None,
                        help="override the wrapper's own warm-up in probes A, B, D and E "
                             "(default: leave it alone and measure the shipped render path; "
                             "pass 0 to reproduce the pre-fix numbers)")
    parser.add_argument("--probes", default="ABCDE",
                        help="subset of the probe letters to run")
    args = parser.parse_args()

    settings = RenderSettings.from_config()
    spec = _synth_spec(args.synth)
    with spec.open_output_suppressor():
        wrapper = make_wrapper("dawdreamer", args.synth)
        defaults = wrapper.get_parameter_defaults()
        space = wrapper.parameter_space
        patches = [
            space.sample_uniform(np.random.default_rng(args.seed + index))
            for index in range(args.patches)
        ]

    print(f"synth={args.synth}  sample_rate={config.SAMPLE_RATE}  settings={settings}")
    silent = _silent_patch(args.synth, defaults)

    probes = args.probes.upper()
    if "A" in probes:
        probe_a_opening_contamination(args.synth, settings, defaults, silent, args.warmup_sec)
    if "B" in probes:
        probe_b_cross_patch_openings(args.synth, settings, patches, args.warmup_sec)
    if "C" in probes:
        probe_c_warmup_sweep(args.synth, settings, silent)
    if "D" in probes:
        probe_d_in_process_divergence(args.synth, settings, patches, args.renders, args.warmup_sec)
    if "E" in probes:
        probe_e_cross_process_identity(args.synth, settings, patches[:3], args.warmup_sec)

    if spec.logs_at_teardown:
        # Everything of ours is printed; the rest is Diva's destructor report.
        from synth.plugin_output import silence_plugin_output_from_now_on
        silence_plugin_output_from_now_on()


if __name__ == "__main__":
    main()
