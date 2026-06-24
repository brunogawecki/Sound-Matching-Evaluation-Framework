"""Confirmatory test for the D-RENDERER cross-engine interpretation (docs/DECISIONS.md).

Hypothesis under test: the cross-engine (DawDreamer vs Pedalboard) disagreement tail is the
same D-REPRO hidden-voice-state mechanism showing up *between* engines, not the two hosts
rendering patches differently. If so, the patches that diverge most across engines should be
the same patches that are most *context-dependent within a single engine*.

This script measures, per patch and within one DawDreamer process, a **context-leakage**
score: how much the patch's own render shifts when it is rendered after one primer patch vs
after a different primer patch (the A/C-primer method of the D-REPRO xfail test
`test_render_unaffected_by_previous_render_content`). It then correlates that score against
the per-patch cross-engine divergence already saved by
`benchmark_renderers.py --dump-agreement-csv`.

Both axes use the *same* log-spectral-distance definition (reused from benchmark_renderers),
over the *same* seeded patches, so the comparison is apples-to-apples.

The same probe runs on either renderer via ``--renderer`` (default ``dawdreamer``). Running it with
``--renderer pedalboard`` answers whether Pedalboard exhibits the *same* within-engine leakage as
DawDreamer (expected, since the hidden state lives in the shared Dexed plugin binary, not the host)
or is a clean anchor. Patch indices are renderer-independent (patches are sampled from the
engine-agnostic ``parameter_space``), so both runs align with the same ``--agreement-csv``.

Run (after the agreement CSV exists):
    python scripts/measure_context_leakage.py \
        --agreement-csv figures/data/host_agreement_seed0.csv --seed 0 --num-patches 3000
    python scripts/measure_context_leakage.py --renderer pedalboard \
        --agreement-csv figures/data/host_agreement_seed0.csv --seed 0 --num-patches 3000

``--cartridges`` switches to a different study: a 3-arm decomposed leak re-test over the real DX7
cartridge voices (not the cross-engine correlation above). Each arm applies a parameter constraint to
every rendered patch -- baseline (none), S&H->square (preset-gen-vae's ``prevent_SH_LFO``), and LFO
disabled (both global LFO depths zeroed) -- and the same A/C-primer probe measures each voice's
within-engine context leakage under it. Comparing arms attributes the leak tail to sample&hold vs.
general LFO vs. deeper non-LFO per-voice state, and shows how much preset-gen-vae's mitigation buys:
    python scripts/measure_context_leakage.py --cartridges
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import benchmark_renderers as bench  # reuse build_patches, agreement_metrics
from synth.dexed import DexedWrapper, suppressed_stderr

# Two fixed primer patches define the two contexts each probe patch is rendered in. Mirrors the
# surrounding patches in the D-REPRO xfail test; fixed seeds keep the whole run reproducible.
_PRIMER_SEED_A = 11
_PRIMER_SEED_C = 13

# LFO WAVE normalized option values, verified via the plugin's displayed parameter text
# (get_parameter_text) on the installed Dexed VST3 build:
#   0.0 TRIANGE | 0.2 SAW DOWN | 0.4 SAW UP | 0.6 SQUARE | 0.8 SINE | 1.0 S&HOLD
# S&H is the only non-deterministic LFO wave (a random value sampled and held each cycle); its held
# value is the prime suspect for the hidden per-voice leak. SQUARE is the deterministic replacement,
# matching the intent of preset-gen-vae's prevent_SH_LFO -- whose own "4/5" constant was for a
# different build's wave order and must NOT be reused here (this build puts SINE, not SQUARE, at 0.8).
_LFO_SH_VALUE = 1.0
_LFO_SQUARE_VALUE = 0.6


def _render(wrapper: DexedWrapper, patch: dict) -> np.ndarray:
    wrapper.set_parameters(patch)
    return wrapper.render_audio(
        midi_note=config.MIDI_NOTE,
        velocity=config.VELOCITY,
        duration_sec=config.DURATION_SEC,
        note_duration_sec=config.NOTE_DURATION_SEC,
    )


def _lsd(audio_a: np.ndarray, audio_b: np.ndarray) -> float:
    return bench.agreement_metrics(audio_a, audio_b, config.SAMPLE_RATE)["log_spectral_distance_db"]


def _assert_lfo_wave_semantics(wrapper: DexedWrapper) -> None:
    """Re-verify at runtime that LFO WAVE 1.0 == S&H and 0.6 == SQUARE, so a future plugin build
    that reorders the waves fails loudly instead of silently mislabeling. Reaches the DawDreamer
    processor's get_parameter_text; skipped if the active renderer cannot report value text."""
    synth = getattr(wrapper._renderer, "_synth", None)
    if synth is None or not hasattr(synth, "get_parameter_text"):
        return
    index = next(d["index"] for d in wrapper._renderer.parameter_descriptions()
                 if d["name"] == "LFO WAVE")
    synth.set_parameter(index, _LFO_SH_VALUE)
    sh_text = synth.get_parameter_text(index).upper()
    synth.set_parameter(index, _LFO_SQUARE_VALUE)
    square_text = synth.get_parameter_text(index).upper()
    if "S&H" not in sh_text or "SQUARE" not in square_text:
        raise RuntimeError(
            f"LFO WAVE semantics changed on this build: {_LFO_SH_VALUE} -> {sh_text!r}, "
            f"{_LFO_SQUARE_VALUE} -> {square_text!r}. Update _LFO_SH_VALUE / _LFO_SQUARE_VALUE."
        )


def _arm_baseline(patch: dict) -> dict:
    """No constraint: the patch as authored. The total-leak reference."""
    return patch


def _arm_sh_to_square(patch: dict) -> dict:
    """preset-gen-vae's prevent_SH_LFO: if the LFO wave is sample&hold, make it square. Removes only
    the random S&H state while leaving a (deterministic) LFO running."""
    if np.isclose(patch.get("LFO WAVE", 0.0), _LFO_SH_VALUE):
        patch = dict(patch)
        patch["LFO WAVE"] = _LFO_SQUARE_VALUE
    return patch


def _arm_lfo_off(patch: dict) -> dict:
    """Remove all LFO influence by zeroing both global modulation depths (pitch and amplitude),
    regardless of wave or phase."""
    patch = dict(patch)
    patch["LFO PM DEPTH"] = 0.0
    patch["LFO AM DEPTH"] = 0.0
    return patch


# (csv column, label, constraint). Order matters for the attribution printout below.
_ARMS = (
    ("leak_baseline_db", "baseline", _arm_baseline),
    ("leak_sh_square_db", "S&H->square", _arm_sh_to_square),
    ("leak_lfo_off_db", "LFO disabled", _arm_lfo_off),
)


def run_cartridge_arms(cartridges_dir: str, out_csv: str) -> None:
    """3-arm decomposed within-engine context-leak measurement over real DX7 cartridge voices.

    Each arm is a parameter constraint applied to *every* rendered patch -- both primers and the
    probe voice -- mirroring a dataset where the constraint is always on. For each voice the leak is
    the A/C-primer probe: render the voice right after primer A and right after primer C in one
    persistent process and take the LSD between the two (0 if the engine were stateless). Comparing
    arms attributes the leak tail to sample&hold vs. general LFO vs. deeper non-LFO per-voice state:
      S&H share        = baseline - (S&H->square)
      non-S&H LFO      = (S&H->square) - (LFO disabled)
      residual         = LFO disabled   (leak with no LFO influence at all)
    """
    patches, labels = bench.build_cartridge_patches(cartridges_dir)
    if not patches:
        print("No usable cartridge voices found.")
        sys.exit(1)

    sh_count = sum(1 for patch in patches if np.isclose(patch.get("LFO WAVE", 0.0), _LFO_SH_VALUE))
    lfo_active = sum(1 for patch in patches
                     if patch.get("LFO PM DEPTH", 0.0) > 0.0 or patch.get("LFO AM DEPTH", 0.0) > 0.0)
    print(f"  {sh_count}/{len(patches)} voices use sample&hold LFO; "
          f"{lfo_active}/{len(patches)} have non-zero LFO depth.")

    plugin_path = os.path.expanduser(config.DEXED_PATH)
    per_arm_leak = {}
    for column, arm_label, transform in _ARMS:
        print(f"Measuring arm: {arm_label} ...")
        with suppressed_stderr():
            wrapper = DexedWrapper(plugin_path, sample_rate=config.SAMPLE_RATE,
                                   buffer_size=config.BUFFER_SIZE)
        _assert_lfo_wave_semantics(wrapper)
        primer_a = transform(wrapper.parameter_space.sample_uniform(np.random.default_rng(_PRIMER_SEED_A)))
        primer_c = transform(wrapper.parameter_space.sample_uniform(np.random.default_rng(_PRIMER_SEED_C)))
        leaks = np.empty(len(patches))
        for patch_index, patch in enumerate(patches):
            probe = transform(patch)
            _render(wrapper, primer_a)
            after_a = _render(wrapper, probe)
            _render(wrapper, primer_c)
            after_c = _render(wrapper, probe)
            leaks[patch_index] = _lsd(after_a, after_c)
        per_arm_leak[column] = leaks
        del wrapper

    result = pd.DataFrame({"patch_label": labels})
    for column, _, _ in _ARMS:
        result[column] = per_arm_leak[column]
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    result.to_csv(out_csv, index=False)
    print(f"\n  wrote per-voice leak CSV -> {out_csv}")

    print(f"\nWithin-engine context leakage by arm (n={len(patches)} cartridge voices), LSD dB:")
    print(f"  {'arm':<14}{'median':>10}{'p90':>10}{'p95':>10}")
    for column, arm_label, _ in _ARMS:
        leaks = per_arm_leak[column]
        print(f"  {arm_label:<14}{np.median(leaks):10.4f}{np.percentile(leaks, 90):10.4f}"
              f"{np.percentile(leaks, 95):10.4f}")

    base, shsq, off = (per_arm_leak[column] for column, _, _ in _ARMS)
    p95 = lambda values: float(np.percentile(values, 95))
    print("\n  p95-tail attribution (dB):")
    print(f"    total (baseline)               {p95(base):8.4f}")
    print(f"    S&H share (baseline - S&H->sq) {p95(base) - p95(shsq):8.4f}")
    print(f"    non-S&H LFO (S&H->sq - LFOoff) {p95(shsq) - p95(off):8.4f}")
    print(f"    residual (LFO disabled)        {p95(off):8.4f}")

    print("\n  Top-10 leaking voices at baseline, and their leak under each arm (LSD dB):")
    print(f"    {'baseline':>9}{'S&H->sq':>9}{'LFO off':>9}   voice")
    for patch_index in np.argsort(base)[::-1][:10]:
        print(f"    {base[patch_index]:9.2f}{shsq[patch_index]:9.2f}{off[patch_index]:9.2f}   "
              f"{labels[patch_index]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", choices=("dawdreamer", "pedalboard"), default="dawdreamer",
                        help="Engine whose within-engine context leakage is measured (default dawdreamer).")
    parser.add_argument("--agreement-csv", default="figures/data/host_agreement_seed0.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-patches", type=int, default=3000)
    parser.add_argument(
        "--cartridges", nargs="?", const=bench.DEFAULT_CARTRIDGES_DIR, default=None,
        help="Run the 3-arm decomposed leak re-test (baseline / S&H->square / LFO disabled) over "
             "real DX7 cartridge voices instead of the cross-engine correlation analysis. Pass a "
             ".syx file or directory; bare --cartridges uses the standard Dexed install location.",
    )
    parser.add_argument("--out-csv", default=None,
                        help="Per-patch cross-engine vs context-leakage LSD (regenerable). Defaults to "
                             "figures/data/context_leakage[_<renderer>]_seed<seed>.csv, or "
                             "figures/data/context_leakage_arms_cartridges.csv with --cartridges.")
    args = parser.parse_args()

    if args.cartridges is not None:
        if not os.path.exists(args.cartridges):
            print(f"Cartridge path not found: {args.cartridges}")
            sys.exit(1)
        out_csv = args.out_csv or "figures/data/context_leakage_arms_cartridges.csv"
        print(f"Loading DX7 cartridge voices from {args.cartridges}")
        run_cartridge_arms(args.cartridges, out_csv)
        return

    if args.out_csv is None:
        # dawdreamer keeps the original unsuffixed name; other renderers get their own file.
        renderer_suffix = "" if args.renderer == "dawdreamer" else f"_{args.renderer}"
        args.out_csv = f"figures/data/context_leakage{renderer_suffix}_seed{args.seed}.csv"

    agreement = pd.read_csv(args.agreement_csv).set_index("patch_index")
    cross_engine_lsd = agreement["log_spectral_distance_db"]

    plugin_path = os.path.expanduser(config.DEXED_PATH)
    with suppressed_stderr():
        wrapper = DexedWrapper(plugin_path, sample_rate=config.SAMPLE_RATE,
                               buffer_size=config.BUFFER_SIZE, renderer=args.renderer)

    # Same seeded patch set the agreement CSV was built from, so patch_index aligns.
    patches = bench.build_patches(wrapper, args.num_patches, args.seed)
    primer_a = wrapper.parameter_space.sample_uniform(np.random.default_rng(_PRIMER_SEED_A))
    primer_c = wrapper.parameter_space.sample_uniform(np.random.default_rng(_PRIMER_SEED_C))

    # The canonical run uses the same --num-patches the agreement CSV was built from, so every
    # CSV index is available. A smaller --num-patches (e.g. a smoke test) only probes the indices
    # actually built.
    records = []
    for patch_index in cross_engine_lsd.index:  # non-silent patches only
        if patch_index >= len(patches):
            continue
        patch = patches[patch_index]
        _render(wrapper, primer_a)
        audio_after_a = _render(wrapper, patch)
        _render(wrapper, primer_c)
        audio_after_c = _render(wrapper, patch)
        records.append({
            "patch_index": int(patch_index),
            "cross_engine_lsd_db": float(cross_engine_lsd.loc[patch_index]),
            "context_leakage_lsd_db": float(_lsd(audio_after_a, audio_after_c)),
        })

    result = pd.DataFrame.from_records(records)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    result.to_csv(args.out_csv, index=False)

    cross = result["cross_engine_lsd_db"].to_numpy()
    leak = result["context_leakage_lsd_db"].to_numpy()
    rho, p_value = spearmanr(cross, leak)

    n = len(result)
    k = max(1, round(0.10 * n))
    top_cross = set(np.argsort(cross)[-k:])
    top_leak = set(np.argsort(leak)[-k:])
    overlap = len(top_cross & top_leak) / k
    chance = k / n

    print(f"\nContext-leakage vs cross-engine divergence "
          f"({args.renderer}, seed {args.seed}, n={n} non-silent patches)")
    print(f"  wrote per-patch CSV -> {args.out_csv}")
    print(f"  context-leakage LSD (dB):  median {np.median(leak):.4f}  p90 {np.percentile(leak,90):.4f}  p95 {np.percentile(leak,95):.4f}")
    print(f"  Spearman rho = {rho:.3f}  (p = {p_value:.2e})")
    print(f"  top-decile overlap = {overlap:.2%}  (chance {chance:.2%}, fold {overlap/chance:.1f}x)")


if __name__ == "__main__":
    main()
