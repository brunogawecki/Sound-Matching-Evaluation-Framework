"""Build a small synthetic dataset through the real Dexed VST.

A convenience runner around :class:`dataset.builder.DatasetBuilder` for smoke-
testing the pipeline by hand: it renders a uniform-random synthetic corpus and
writes ``run_summary.json`` + ``metadata.csv`` + ``audio/*.wav`` under a per-run
subdirectory of ``config.DATASET_DIR``.

This is the in-process :class:`SequentialExecutor` path (Issue #4): correct and
deterministic within one OS process. Bit-identical reproducibility across runs
requires a fresh OS process per run (the Dexed context leak; see D-REPRO and
Issue #5).

Usage::

    python scripts/build_dataset.py                       # 16 presets, seed 0
    python scripts/build_dataset.py --count 64 --seed 7
    python scripts/build_dataset.py --run-name my_corpus
"""
import argparse
import os
import sys
from pathlib import Path

# This script lives in scripts/; put the project root on the path so the
# top-level packages (config, synth, dataset) import when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synth.dexed import DexedWrapper
from dataset.sources import SyntheticSampler
from dataset.builder import DatasetBuilder
import config


def build_dataset(count: int, seed: int, run_name: str) -> None:
    plugin_path = os.path.expanduser(config.DEXED_PATH)

    if not os.path.exists(plugin_path):
        print(f"Could not find Dexed plugin at: {plugin_path}")
        print("Please update DEXED_PATH in your .env file.")
        sys.exit(1)

    print(f"--- Building '{run_name}': {count} synthetic presets (seed {seed}) ---")

    synth = DexedWrapper(
        plugin_path=plugin_path,
        sample_rate=config.SAMPLE_RATE,
        buffer_size=config.BUFFER_SIZE,
    )
    print(f"Initialized Dexed at {synth.sample_rate}Hz; subset = {len(synth.parameter_space.names)} params")

    source = SyntheticSampler(
        synth.parameter_space, count=count, seed=seed, sampling_ranges=synth.audible_sampling_ranges
    )
    summary = DatasetBuilder(synth).build(source, run_name=run_name)

    run_dir = Path(config.DATASET_DIR) / run_name
    print("\nSuccess!")
    print(f"Samples: {summary['num_samples']} | near-silent: {summary['near_silent_count']}")
    print(f"Renderer: {summary['renderer']} | git revision: {summary['git_revision']}")
    print(f"Written to: {run_dir}")
    print(f"  {run_dir / 'run_summary.json'}")
    print(f"  {run_dir / 'metadata.csv'}")
    print(f"  {run_dir / 'audio'}/*.wav")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a small synthetic Dexed dataset.")
    parser.add_argument("--count", type=int, default=16, help="number of presets to render")
    parser.add_argument("--seed", type=int, default=0, help="master seed for the synthetic sampler")
    parser.add_argument("--run-name", default="synthetic_smoke", help="output subdirectory name")
    args = parser.parse_args()
    build_dataset(count=args.count, seed=args.seed, run_name=args.run_name)


if __name__ == "__main__":
    main()
