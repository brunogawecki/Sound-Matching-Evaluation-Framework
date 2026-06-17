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

Run (after the agreement CSV exists):
    python scripts/measure_context_leakage.py \
        --agreement-csv figures/data/host_agreement_seed0.csv --seed 0 --num-patches 3000
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
import benchmark_renderers as bench  # reuse suppressed_stderr, build_patches, agreement_metrics
from synth.dexed import DexedWrapper

# Two fixed primer patches define the two contexts each probe patch is rendered in. Mirrors the
# surrounding patches in the D-REPRO xfail test; fixed seeds keep the whole run reproducible.
_PRIMER_SEED_A = 11
_PRIMER_SEED_C = 13


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agreement-csv", default="figures/data/host_agreement_seed0.csv")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-patches", type=int, default=3000)
    parser.add_argument("--out-csv", default="figures/data/context_leakage_seed0.csv",
                        help="Per-patch cross-engine vs context-leakage LSD (regenerable).")
    args = parser.parse_args()

    agreement = pd.read_csv(args.agreement_csv).set_index("patch_index")
    cross_engine_lsd = agreement["log_spectral_distance_db"]

    plugin_path = os.path.expanduser(config.DEXED_PATH)
    with bench.suppressed_stderr():
        wrapper = DexedWrapper(plugin_path, sample_rate=config.SAMPLE_RATE,
                               buffer_size=config.BUFFER_SIZE)

    # Same seeded patch set the agreement CSV was built from, so patch_index aligns.
    patches = bench.build_patches(wrapper, args.num_patches, args.seed)
    primer_a = wrapper.parameter_space.sample_uniform(np.random.default_rng(_PRIMER_SEED_A))
    primer_c = wrapper.parameter_space.sample_uniform(np.random.default_rng(_PRIMER_SEED_C))

    records = []
    for patch_index in cross_engine_lsd.index:  # non-silent patches only
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

    print(f"\nContext-leakage vs cross-engine divergence (seed {args.seed}, n={n} non-silent patches)")
    print(f"  wrote per-patch CSV -> {args.out_csv}")
    print(f"  context-leakage LSD (dB):  median {np.median(leak):.4f}  p90 {np.percentile(leak,90):.4f}  p95 {np.percentile(leak,95):.4f}")
    print(f"  Spearman rho = {rho:.3f}  (p = {p_value:.2e})")
    print(f"  top-decile overlap = {overlap:.2%}  (chance {chance:.2%}, fold {overlap/chance:.1f}x)")


if __name__ == "__main__":
    main()
