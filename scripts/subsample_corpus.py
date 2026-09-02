"""Carve a smaller corpus out of an already-rendered one by copying a seeded subset.

Evaluation cost is linear in test-set size, and the Evaluator re-renders every
prediction in a fresh process (D-REPRO), so a full-size test corpus dominates the
benchmark's wall clock. This copies a seeded subset of an existing corpus verbatim --
no re-rendering, because the source's WAVs already satisfy whatever render contract
it was built under, and the copy preserves ``render_process``.

The output is a corpus in its own right (self-describing, D-SELFDESC) with its own
name, so results land under ``results/<subsampled name>/`` and never mix with the
full-size run.

    python scripts/subsample_corpus.py --corpus full_preset-gen-vae_test --size 1500

    --corpus         source corpus: a run name under DATASET_DIR, or a path  [REQUIRED]
    --size           how many samples to keep                                [REQUIRED]
    --subsample-seed seed for the row selection                              [default: 0]
    --run-name       output corpus name              [default: <source>_<size>]

Use one subsampled corpus for every model in a comparison: the paired significance
tests join per ``sample_id``, so families scored on different subsets cannot be compared.
"""
import argparse
import sys
from pathlib import Path

# This script lives in scripts/; put the project root on the path so the
# top-level packages (config, dataset) import when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from dataset.corpus_splitter import (
    load_corpus,
    subsample_indices,
    subsample_source_description,
    write_copied_partition,
)


def _resolve_corpus_dir(corpus: str) -> Path:
    candidate = Path(corpus)
    return candidate if candidate.exists() else Path(config.DATASET_DIR) / corpus


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy a seeded subset of an already-rendered corpus into a smaller one."
    )
    parser.add_argument("--corpus", required=True, help="source run name under DATASET_DIR, or a path")
    parser.add_argument("--size", type=int, required=True, help="how many samples to keep")
    parser.add_argument("--subsample-seed", type=int, default=0, help="seed for the row selection")
    parser.add_argument("--run-name", default=None, help="output corpus name (default: <source>_<size>)")
    args = parser.parse_args()

    source_dir = _resolve_corpus_dir(args.corpus)
    if not (source_dir / "run_summary.json").exists():
        print(f"Not a corpus (no run_summary.json): {source_dir}")
        sys.exit(1)

    summary, df_metadata = load_corpus(source_dir)
    source_count = len(df_metadata)
    if args.size > source_count:
        print(f"Cannot keep {args.size} samples: '{source_dir.name}' has {source_count}.")
        sys.exit(1)

    render_process = str(summary.get("render_process", "in-process"))
    if not render_process.startswith("fresh"):
        print(f"Warning: source renders '{render_process}', not fresh-process.")
        print("The copy preserves that, so it is not a valid eval corpus (D-REPRO/D-EVAL).")

    run_name = args.run_name or f"{source_dir.name}_{args.size}"
    out_dir = Path(config.DATASET_DIR) / run_name
    if out_dir.exists():
        print(f"Output corpus already exists: {out_dir}")
        sys.exit(1)

    positions = subsample_indices(source_count, args.size, args.subsample_seed)
    df_partition = df_metadata.iloc[positions]
    partition = str(df_partition["partition"].iloc[0]) if "partition" in df_partition else "test"
    description = subsample_source_description(
        summary, source_dir.name, args.subsample_seed, args.size, source_count
    )

    print(f"--- Subsampling '{source_dir.name}' ({source_count}) -> '{run_name}' ({args.size}) ---")
    written = write_copied_partition(
        source_dir, df_partition, out_dir, summary, description, partition=partition
    )
    print(f"Wrote {written['num_samples']} samples ({written['render_process']}) to: {out_dir}")


if __name__ == "__main__":
    main()
