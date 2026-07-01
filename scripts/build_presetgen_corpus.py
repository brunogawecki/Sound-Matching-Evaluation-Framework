"""Build a Dexed training corpus from the preset-gen-vae human DX7 collection.

Loads real DX7 voices from the preset-gen-vae SQLite database
(``paper_repos/preset-gen-vae/synth/dexed_presets.sqlite``), projects each onto
the estimated subset, deduplicates and splits them, then renders the **train**
partition in-process (fast; D-REPRO reserves fresh-process rendering for the
test/eval corpora). The held-out test partition is produced by the split but not
rendered here -- its composition is settled by D4 in Phase 6.

The database is Git LFS-backed: run ``git lfs pull`` inside
``paper_repos/preset-gen-vae/`` first if the file is a small pointer.

Usage
-----
Render a capped pilot (validate the whole path end-to-end before a full run)::

    python scripts/build_presetgen_corpus.py --limit 500 --run-name presetgen_pilot

    --db-path          path to dexed_presets.sqlite
                       [default: config.PRESETGEN_DB_PATH, overridable via
                       PRESETGEN_DB_PATH in .env]
    --limit            cap on raw voices read from the DB before dedup/split;
                       omit to use the whole collection (~30k)   [default: none]
    --run-name         output subdirectory name                  [default: presetgen_train]
    --test-fraction    share held out for the (unrendered) test set  [default: 0.20]
    --split-seed       seed for the train/test shuffle            [default: 0]
    --dedup-threshold  distance below which two voices collapse to one  [default: 0.001]
"""
import argparse
import os
import sys
from pathlib import Path

# This script lives in scripts/; put the project root on the path so the
# top-level packages (config, synth, dataset) import when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synth.dexed import DexedWrapper
from dataset.builder import DatasetBuilder
from dataset.preset_sources import HumanPresetSource
from dataset.dexed_sqlite_preset_loader import DexedSqlitePresetLoader
import config


def _make_synth() -> DexedWrapper:
    plugin_path = os.path.expanduser(config.DEXED_PATH)
    if not os.path.exists(plugin_path):
        print(f"Could not find Dexed plugin at: {plugin_path}")
        print("Please update DEXED_PATH in your .env file.")
        sys.exit(1)
    synth = DexedWrapper(
        plugin_path=plugin_path,
        sample_rate=config.SAMPLE_RATE,
        buffer_size=config.BUFFER_SIZE,
    )
    print(f"Initialized Dexed at {synth.sample_rate}Hz; subset = {len(synth.parameter_space.names)} params")
    return synth


def _report(summary: dict, run_dir: Path) -> None:
    print("\nSuccess!")
    print(f"Samples: {summary['num_samples']} | near-silent: {summary['near_silent_count']}")
    print(f"Methods: {summary['method_counts']}")
    print(f"Renderer: {summary['renderer']} | git revision: {summary['git_revision']}")
    print(f"Written to: {run_dir}")
    print(f"  {run_dir / 'run_summary.json'}")
    print(f"  {run_dir / 'metadata.csv'}")
    print(f"  {run_dir / 'audio'}/*.wav")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Dexed training corpus from the preset-gen-vae SQLite collection."
    )
    parser.add_argument("--db-path", default=config.PRESETGEN_DB_PATH, help="path to dexed_presets.sqlite")
    parser.add_argument("--limit", type=int, default=None, help="cap on raw voices read before dedup/split")
    parser.add_argument("--run-name", default="presetgen_train", help="output subdirectory name")
    parser.add_argument("--test-fraction", type=float, default=0.20, help="share held out for test")
    parser.add_argument("--split-seed", type=int, default=0, help="seed for the train/test split")
    parser.add_argument("--dedup-threshold", type=float, default=1e-3, help="duplicate distance")
    args = parser.parse_args()

    db_path = os.path.expanduser(args.db_path)
    if not os.path.exists(db_path):
        print(f"Could not find the preset database at: {db_path}")
        print("Run `git lfs pull` inside paper_repos/preset-gen-vae/ to materialize it.")
        sys.exit(1)
    if os.path.getsize(db_path) < 1_000_000:
        print(f"The preset database at {db_path} is only {os.path.getsize(db_path)} bytes -- "
              "likely an unresolved Git LFS pointer. Run `git lfs pull` to materialize it.")
        sys.exit(1)

    synth = _make_synth()
    split = DexedSqlitePresetLoader(
        synth.parameter_space,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        dedup_threshold=args.dedup_threshold,
    ).load(db_path, limit=args.limit)
    print(f"--- preset-gen-vae split: {len(split.train)} train / {len(split.test)} test after dedup ---")

    # Train renders in-process (fast; D-REPRO reserves fresh-process for the test/eval
    # corpora). The test partition is produced by the split but not rendered here (D4).
    print(f"--- Building '{args.run_name}': {len(split.train)} train presets (in-process) ---")
    source = HumanPresetSource(split.train, synth.parameter_space, partition="train")
    summary = DatasetBuilder(synth).build(source, run_name=args.run_name)
    _report(summary, Path(config.DATASET_DIR) / args.run_name)


if __name__ == "__main__":
    main()
