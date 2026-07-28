"""Check that the live Dexed reports the parameter names a corpus was built with.

A SynthRLi training run renders with the compute node's Dexed to score its RL reward,
setting patches by name (D-NAMING). The corpus's ParameterSpace was built by a possibly
different Dexed build (e.g. the Mac's), so if this node's plugin spells, drops, or adds a
name, the reward would score the wrong sound. Run this once on the node before submitting.

    python scripts/verify_parameter_parity.py --corpus /path/to/corpus

Exits non-zero on any mismatch. Reads DEXED_PATH from config.py (cluster.env / .env).
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from synth.dexed import DexedWrapper
from synth.parameter_space import ParameterSpace


def load_corpus_space(corpus_dir: Path) -> ParameterSpace:
    summary_path = corpus_dir / "run_summary.json"
    if not summary_path.exists():
        print(f"No run_summary.json at {summary_path}.")
        sys.exit(1)
    with open(summary_path) as summary_file:
        summary = json.load(summary_file)
    if "parameter_space" not in summary:
        print(f"{summary_path} has no 'parameter_space'. Rebuild the corpus with the current "
              "DatasetBuilder so it carries its parameter map.")
        sys.exit(1)
    return ParameterSpace.from_dict(summary["parameter_space"])


def verify_parameter_parity(corpus_dir: Path) -> None:
    space = load_corpus_space(corpus_dir)
    print(f"--- Parameter-name parity: {corpus_dir} ---")
    print(f"Corpus parameter space: {space.synth_dimension} parameters")

    plugin_path = os.path.expanduser(config.DEXED_PATH)
    if not os.path.exists(plugin_path):
        print(f"Could not find Dexed plugin at: {plugin_path}")
        print("Set DEXED_PATH in cluster/cluster.env (cluster) or .env (laptop).")
        sys.exit(1)

    synth = DexedWrapper(
        plugin_path=plugin_path,
        sample_rate=config.SAMPLE_RATE,
        buffer_size=config.BUFFER_SIZE,
    )
    live_names = set(synth.parameter_names)
    print(f"Live Dexed ({plugin_path}): {len(live_names)} exposed parameters")

    # A corpus name the live plugin does not expose: set_parameters would KeyError mid-render.
    missing = [name for name in space.names if name not in live_names]

    # Same name, different categorical width: sets a valid float that decodes to a different
    # option, silently. Categorical cardinality is the ML-side block width (ml_dimension).
    live_categoricals = synth.get_categorical_mappings()
    cardinality_mismatches = []
    for spec in space.parameter_specs:
        if spec.name in missing or spec.kind != "categorical":
            continue
        live_cardinality = live_categoricals.get(spec.name, {}).get("cardinality")
        if live_cardinality != spec.ml_dimension:
            cardinality_mismatches.append((spec.name, spec.ml_dimension, live_cardinality))

    if missing:
        print(f"\nFAIL: {len(missing)} corpus parameter(s) not exposed by this Dexed:")
        for name in missing:
            print(f"  - {name!r}")
    if cardinality_mismatches:
        print(f"\nFAIL: {len(cardinality_mismatches)} categorical(s) with a different option count:")
        for name, expected, live in cardinality_mismatches:
            print(f"  - {name!r}: corpus expects {expected} options, live Dexed reports {live}")

    if missing or cardinality_mismatches:
        print("\nThis Dexed build does not match the corpus. The RL reward would score the "
              "wrong sound. Do not submit a SynthRLi run against it.")
        sys.exit(1)

    print(f"\nSuccess! All {space.synth_dimension} corpus parameters match this Dexed by name "
          "and categorical width. Safe to run SynthRLi.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        required=True,
        help="Corpus directory (contains run_summary.json) the SynthRLi run will train on.",
    )
    verify_parameter_parity(parser.parse_args().corpus)
