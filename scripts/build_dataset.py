"""Build a Dexed dataset corpus: synthetic, human, or hybrid.

A hand runner around :class:`dataset.builder.DatasetBuilder` and the three
preset sources. Each subcommand renders a corpus and writes run_summary.json +
metadata.csv + audio/*.wav under <config.DATASET_DIR>/<run-name>/.

Tutorial
--------
Pick one of three subcommands by where the presets come from. Each parameter
below is tagged REQUIRED or shows its default; everything else is optional.
Every run is reproducible: the same subcommand and arguments produce the same
corpus.

``synthetic`` -- random draws over the parameter space (no presets needed)::

    python scripts/build_dataset.py synthetic --count 64 --seed 7

    --count     how many presets to render                  [default: 16]
    --seed      master seed for the random sampler           [default: 0]
    --run-name  output subdirectory name                     [default: synthetic_smoke]

``human`` -- real DX7 presets read from ``.syx`` cartridges::

    python scripts/build_dataset.py human --cartridges presets/ --partition test

    --cartridges       where the .syx files are: a folder (recurses for *.syx),
                       a glob like "presets/*.syx", or explicit file paths;
                       accepts several at once               [REQUIRED]
    --partition        which half of the split to render, "train" or "test"
                                                             [default: train]
    --test-fraction    share of presets held out as the test set
                                                             [default: 0.20]
    --split-seed       seed for the train/test shuffle       [default: 0]
    --dedup-threshold  distance below which two presets count as duplicates
                       and collapse to one                   [default: 0.001]
    --run-name         output subdirectory name              [default: human_<partition>]

``hybrid`` -- human train presets combined with synthetic material::

    python scripts/build_dataset.py hybrid --cartridges presets/ --mode blend --count 128

    --cartridges       same as for ``human``                 [REQUIRED]
    --mode             "blend" mixes in whole synthetic draws; "augment"
                       perturbs human presets                [default: blend]
    --count            how many presets to render            [default: 64]
    --seed             master seed for the sampler           [default: 0]
    --synthetic-ratio  blend only: probability each slot is synthetic
                                                             [default: 0.5]
    --num-perturbed-params  augment only: how many parameters to change
                                                             [default: 2]
    --jitter           augment only: size of the continuous nudge
                                                             [default: 0.05]
    --flip-categoricals  augment only: also allow categorical params to flip
                                                             [default: off]
    --test-fraction    share held out as the test set        [default: 0.20]
    --split-seed       seed for the train/test shuffle       [default: 0]
    --dedup-threshold  duplicate-collapse distance           [default: 0.001]
    --run-name         output subdirectory name              [default: hybrid_<mode>]

All three subcommands also accept ``--fresh-process``: render each preset in its
own clean spawned process (slow, leak-free). Use it for **test / evaluation**
corpora, where the generation and evaluation render contexts must agree (D-REPRO);
leave it off for training data, which renders fast in-process.
"""
import argparse
import glob
import os
import sys
from pathlib import Path

# This script lives in scripts/; put the project root on the path so the
# top-level packages (config, synth, dataset) import when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synth.dexed import DexedWrapper
from dataset.builder import DatasetBuilder
from dataset.render_backends import FreshProcessRenderBackend, RenderSettings
from dataset.preset_sources import (
    HumanPresetSource,
    HybridPresetSource,
    SyntheticPresetSource,
)
from dataset.dexed_preset_loader import DexedPresetLoader
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


def _resolve_cartridges(patterns: list) -> list:
    """Expand .syx arguments to files: a directory recurses for *.syx, a glob
    expands, a plain path is taken as-is."""
    paths: list = []
    for pattern in patterns:
        expanded = os.path.expanduser(pattern)
        if os.path.isdir(expanded):
            matches = sorted(glob.glob(os.path.join(expanded, "**", "*.syx"), recursive=True))
        else:
            matches = sorted(glob.glob(expanded))
        paths.extend(matches or [expanded])
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        print(f"Cartridge(s) not found: {missing}")
        sys.exit(1)
    if not paths:
        print("No .syx cartridges matched the given paths.")
        sys.exit(1)
    return paths


def _report(summary: dict, run_dir: Path) -> None:
    print("\nSuccess!")
    print(f"Samples: {summary['num_samples']} | near-silent: {summary['near_silent_count']}")
    print(f"Methods: {summary['method_counts']}")
    print(f"Renderer: {summary['renderer']} | git revision: {summary['git_revision']}")
    print(f"Written to: {run_dir}")
    print(f"  {run_dir / 'run_summary.json'}")
    print(f"  {run_dir / 'metadata.csv'}")
    print(f"  {run_dir / 'audio'}/*.wav")


def _build(synth: DexedWrapper, source, run_name: str, fresh_process: bool = False) -> None:
    # Fresh-process rendering (one clean spawned worker per preset) is for test/eval
    # corpora, where the generation and evaluation render contexts must agree (D-REPRO);
    # training data stays on the fast in-process path. The builder closes the backend.
    backend = (
        FreshProcessRenderBackend(RenderSettings.from_config(), renderer="dawdreamer")
        if fresh_process
        else None
    )
    if fresh_process:
        print("--- Rendering in fresh spawned processes (one per preset; D-REPRO) ---")
    summary = DatasetBuilder(synth, render_backend=backend).build(source, run_name=run_name)
    _report(summary, Path(config.DATASET_DIR) / run_name)


def build_synthetic(args: argparse.Namespace) -> None:
    synth = _make_synth()
    print(f"--- Building '{args.run_name}': {args.count} synthetic presets (seed {args.seed}) ---")
    source = SyntheticPresetSource(
        synth.parameter_space,
        count=args.count,
        seed=args.seed,
        sampling_ranges=synth.audible_sampling_ranges,
    )
    _build(synth, source, args.run_name, fresh_process=args.fresh_process)


def build_human(args: argparse.Namespace) -> None:
    synth = _make_synth()
    cartridges = _resolve_cartridges(args.cartridges)
    split = DexedPresetLoader(
        synth.parameter_space,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        dedup_threshold=args.dedup_threshold,
    ).load(cartridges)
    presets = split.test if args.partition == "test" else split.train
    print(
        f"--- Building '{args.run_name}': {len(presets)} human presets "
        f"({args.partition}; {len(split.train)} train / {len(split.test)} test after dedup) ---"
    )
    source = HumanPresetSource(presets, synth.parameter_space, partition=args.partition)
    _build(synth, source, args.run_name, fresh_process=args.fresh_process)


def build_hybrid(args: argparse.Namespace) -> None:
    synth = _make_synth()
    cartridges = _resolve_cartridges(args.cartridges)
    split = DexedPresetLoader(
        synth.parameter_space,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        dedup_threshold=args.dedup_threshold,
    ).load(cartridges)
    # Parents come only from the human train partition (never the held-out test set).
    parents = list(HumanPresetSource(split.train, synth.parameter_space, "train").iter_presets())
    print(
        f"--- Building '{args.run_name}': {args.count} hybrid presets "
        f"(mode={args.mode}, seed {args.seed}, {len(parents)} human-train parents) ---"
    )
    source = HybridPresetSource(
        mode=args.mode,
        human_presets=parents,
        parameter_space=synth.parameter_space,
        count=args.count,
        seed=args.seed,
        synthetic_ratio=args.synthetic_ratio,
        num_perturbed_params=args.num_perturbed_params,
        jitter=args.jitter,
        flip_categoricals=args.flip_categoricals,
        sampling_ranges=synth.audible_sampling_ranges,
    )
    _build(synth, source, args.run_name, fresh_process=args.fresh_process)


def _add_fresh_process_flag(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--fresh-process",
        action="store_true",
        help="render each preset in its own clean spawned process (slow, leak-free); "
        "use for test/eval corpora, leave off for training data (D-REPRO)",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Dexed dataset corpus.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("synthetic", help="random draws over the parameter space")
    synthetic.add_argument("--count", type=int, default=16, help="number of presets to render")
    synthetic.add_argument("--seed", type=int, default=0, help="master seed for the sampler")
    synthetic.add_argument("--run-name", default="synthetic_smoke", help="output subdirectory name")
    _add_fresh_process_flag(synthetic)
    synthetic.set_defaults(func=build_synthetic)

    human = subparsers.add_parser("human", help="real .syx presets projected onto the subset")
    human.add_argument("--cartridges", nargs="+", required=True, help=".syx paths or globs")
    human.add_argument("--partition", choices=["train", "test"], default="train")
    human.add_argument("--test-fraction", type=float, default=0.20, help="share held out for test")
    human.add_argument("--split-seed", type=int, default=0, help="seed for the train/test split")
    human.add_argument("--dedup-threshold", type=float, default=1e-3, help="duplicate distance")
    human.add_argument("--run-name", default=None, help="output subdirectory name")
    _add_fresh_process_flag(human)
    human.set_defaults(func=build_human)

    hybrid = subparsers.add_parser("hybrid", help="human-train presets blended/augmented with synthetic")
    hybrid.add_argument("--cartridges", nargs="+", required=True, help=".syx paths or globs")
    hybrid.add_argument("--mode", choices=[HybridPresetSource.BLEND, HybridPresetSource.AUGMENT], default=HybridPresetSource.BLEND)
    hybrid.add_argument("--count", type=int, default=64, help="number of presets to render")
    hybrid.add_argument("--seed", type=int, default=0, help="master seed for the sampler")
    hybrid.add_argument("--synthetic-ratio", type=float, default=0.5, help="blend: P(synthetic per slot)")
    hybrid.add_argument("--num-perturbed-params", type=int, default=2, help="augment: params jittered/flipped")
    hybrid.add_argument("--jitter", type=float, default=0.05, help="augment: continuous jitter magnitude")
    hybrid.add_argument("--flip-categoricals", action="store_true", help="augment: allow categorical flips")
    hybrid.add_argument("--test-fraction", type=float, default=0.20, help="share held out for test")
    hybrid.add_argument("--split-seed", type=int, default=0, help="seed for the train/test split")
    hybrid.add_argument("--dedup-threshold", type=float, default=1e-3, help="duplicate distance")
    hybrid.add_argument("--run-name", default=None, help="output subdirectory name")
    _add_fresh_process_flag(hybrid)
    hybrid.set_defaults(func=build_hybrid)

    args = parser.parse_args()
    if args.command == "human" and args.run_name is None:
        args.run_name = f"human_{args.partition}"
    if args.command == "hybrid" and args.run_name is None:
        args.run_name = f"hybrid_{args.mode}"
    args.func(args)


if __name__ == "__main__":
    main()
