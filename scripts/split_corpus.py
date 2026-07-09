"""Split an already-rendered corpus into a train corpus and a test corpus.

Carves a held-out test set out of an existing corpus without re-rendering all of
it. The **train** partition is copied verbatim (its render context is irrelevant
to training); the **test** partition is re-rendered in fresh processes at position
0 (D-REPRO), so it satisfies the eval render contract even when the source was
built in-process. See D-SPLIT in docs/DECISIONS.md.

Hybrid corpora are refused (train/test leakage): split their human source
cartridges at build time instead.

    python scripts/split_corpus.py --corpus full_preset-gen-vae --test-fraction 0.2

    --corpus         source corpus: a run name under DATASET_DIR, or a path  [REQUIRED]
    --test-fraction  share of samples held out as the test set              [default: 0.20]
    --split-seed     seed for the train/test row shuffle                    [default: 0]
    --run-name       output base name; the two corpora get _train/_test
                     suffixes                                               [default: source name]

The train corpus keeps the source's render process; the test corpus renders fresh
(so it shows up as eval-ready). Neither modifies the source corpus.
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
from dataset.render_backends import FreshProcessRenderBackend, RenderSettings
from dataset.preset_sources import CorpusPresetSource
from dataset.dexed_preset_loader import split_indices
from dataset.corpus_splitter import (
    assert_splittable,
    clean_records,
    load_corpus,
    split_source_description,
    write_copied_partition,
)
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


def _resolve_corpus(corpus_arg: str) -> Path:
    """A run name under DATASET_DIR or an explicit path -> the corpus directory."""
    candidate = Path(os.path.expanduser(corpus_arg))
    corpus_dir = candidate if candidate.is_dir() else Path(config.DATASET_DIR) / corpus_arg
    if not (corpus_dir / "run_summary.json").is_file():
        print(f"No corpus with a run_summary.json at: {corpus_dir}")
        sys.exit(1)
    return corpus_dir


def _report(label: str, summary: dict, run_dir: Path) -> None:
    print(f"\n{label}: '{summary['run_name']}'")
    print(f"  Samples: {summary['num_samples']} | near-silent: {summary['near_silent_count']}")
    print(f"  Methods: {summary['method_counts']} | render: {summary['render_process']}")
    print(f"  Written to: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a rendered corpus into train/test corpora.")
    parser.add_argument("--corpus", required=True,
                        help="source corpus: a run name under DATASET_DIR, or a path")
    parser.add_argument("--test-fraction", type=float, default=0.2,
                        help="share of samples held out as the test set")
    parser.add_argument("--split-seed", type=int, default=0, help="seed for the train/test row shuffle")
    parser.add_argument("--run-name", default=None,
                        help="output base name; _train/_test are appended. Default: source name")
    args = parser.parse_args()

    if not 0.0 < args.test_fraction < 1.0:
        print(f"--test-fraction must be strictly between 0 and 1, got {args.test_fraction}.")
        sys.exit(1)

    corpus_dir = _resolve_corpus(args.corpus)
    summary, df_metadata = load_corpus(corpus_dir)
    try:
        assert_splittable(summary)
    except ValueError as error:
        print(error)
        sys.exit(1)

    base = args.run_name or corpus_dir.name
    dataset_dir = Path(config.DATASET_DIR)
    train_dir = dataset_dir / f"{base}_train"
    test_dir = dataset_dir / f"{base}_test"
    for out_dir in (train_dir, test_dir):
        if out_dir.exists():
            print(f"Output already exists, refusing to overwrite: {out_dir}")
            sys.exit(1)

    train_positions, test_positions = split_indices(len(df_metadata), args.test_fraction, args.split_seed)
    if not train_positions or not test_positions:
        print(
            f"Split leaves an empty partition ({len(train_positions)} train / {len(test_positions)} "
            f"test from {len(df_metadata)} samples). Pick a different --test-fraction."
        )
        sys.exit(1)
    df_train = df_metadata.iloc[train_positions]
    df_test = df_metadata.iloc[test_positions]

    if summary.get("sample_rate") not in (None, config.SAMPLE_RATE):
        print(
            f"Warning: corpus sample_rate {summary.get('sample_rate')} != config.SAMPLE_RATE "
            f"{config.SAMPLE_RATE}; the test partition re-renders at {config.SAMPLE_RATE}Hz."
        )

    print(
        f"--- Splitting '{corpus_dir.name}' ({len(df_metadata)} samples) -> "
        f"{len(df_train)} train / {len(df_test)} test (seed {args.split_seed}) ---"
    )

    # Train: copy audio verbatim (no VST needed; render context is irrelevant to training).
    train_desc = split_source_description(
        summary, "train", corpus_dir.name, args.test_fraction, args.split_seed, len(df_train)
    )
    train_summary = write_copied_partition(corpus_dir, df_train, train_dir, summary, train_desc, "train")
    _report("Train (copied)", train_summary, train_dir)

    # Test: re-render fresh-process at pos 0 (D-REPRO) so it satisfies the eval contract.
    print("--- Re-rendering the test partition in fresh spawned processes (D-REPRO) ---")
    synth = _make_synth()
    settings = RenderSettings(**summary["render_settings"])
    renderer = summary.get("renderer") or "dawdreamer"
    test_desc = split_source_description(
        summary, "test", corpus_dir.name, args.test_fraction, args.split_seed, len(df_test)
    )
    source = CorpusPresetSource(clean_records(df_test), synth.parameter_space, test_desc, partition="test")
    backend = FreshProcessRenderBackend(settings, renderer=renderer)
    test_summary = DatasetBuilder(synth, render_settings=settings, render_backend=backend).build(
        source, run_name=test_dir.name, show_progress=True
    )
    _report("Test (re-rendered fresh)", test_summary, test_dir)

    print("\nSuccess! Evaluate models on the fresh-process test corpus.")


if __name__ == "__main__":
    main()
