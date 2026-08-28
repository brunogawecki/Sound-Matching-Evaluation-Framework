"""Build a dataset corpus: synthetic, human, or hybrid, for Dexed or Diva.

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

``human`` -- real presets projected onto the subset. By default every preset is
train (``--test-fraction 0.0``); raise it to also render a held-out test
partition in the same run, disjoint by construction (no seed to re-match across
separate runs). Empty partitions are skipped::

    python scripts/build_dataset.py human --cartridges presets/
    python scripts/build_dataset.py human --synth diva --limit 500

    --presets          where the presets are. dexed: .syx files, a folder
                       (recurses for *.syx), a glob, or explicit paths, several
                       at once [REQUIRED]. diva: the one collection, a
                       diva_raw.zip or an extracted directory
                       [default: config.DIVA_RAW_PATH].
                       Spelled --cartridges too, for the Dexed commands that
                       predate the flag being synth-neutral
    --limit            diva only: cap on raw presets read      [default: all]
    --partition        render only this half, "train" or "test"
                                                             [default: both]
    --test-fraction    share of presets held out as the test set; 0.0 renders
                       every preset as train                 [default: 0.00]
    --split-seed       seed for the train/test shuffle       [default: 0]
    --dedup-threshold  distance below which two presets count as duplicates
                       and collapse to one                   [default: 0.001]
    --keep-constant-params  keep parameters the presets never vary
                                                             [default: off]
    --run-name         output subdirectory name; rendering both partitions gives
                       each a _train/_test suffix            [default: human_<partition>]

``hybrid`` -- human train presets combined with synthetic material::

    python scripts/build_dataset.py hybrid --cartridges presets/ --mode blend --count 128

    --presets          same as for ``human``                 [REQUIRED for dexed]
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
    --test-fraction    share held out as the test set        [default: 0.00]
    --split-seed       seed for the train/test shuffle       [default: 0]
    --dedup-threshold  duplicate-collapse distance           [default: 0.001]
    --run-name         output subdirectory name              [default: hybrid_<mode>]

Every subcommand takes ``--synth {dexed,diva}`` (default ``dexed``) and ``--workers N``
(default 1). ``--workers`` only affects fresh-process partitions: each patch still gets its
own single-use process, so the audio is byte-identical and only throughput changes. On a
10-core Mac, 60 Diva presets took 49 s serial and 12 s at 8 workers.

Render backend (D-REPRO): test/eval corpora must render in clean spawned processes
(slow, leak-free) so the generation and evaluation render contexts agree; training
data renders fast in-process. ``human`` applies this automatically -- its test
partition always renders fresh, its train partition in-process. All three
subcommands also accept ``--fresh-process`` to force fresh rendering for every
partition they build (on ``synthetic`` / ``hybrid`` it is the only way to opt in).
**Diva always renders fresh-process**, because it does not reproduce in-process at
all (D-DIVA-RENDER).

Corpus-variance rule (D-DIVA-SUBSET): a subset parameter that every loaded preset
holds at the same value leaves the estimated set and is locked at the presets' own
value instead of the plugin's init patch, so the corpus serializes a narrower
parameter space than the synth's. This is a no-op on preset collections that vary
everything (the Dexed case) and drops 179 of 237 parameters on the Flow Synthesizer
Diva collection. ``--keep-constant-params`` opts out.
"""
import argparse
import glob
import os
import sys
from pathlib import Path
from typing import Optional

# This script lives in scripts/; put the project root on the path so the
# top-level packages (config, synth, dataset) import when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from synth.base_synth import BaseSynthesizer
from synth.parameter_space import ParameterSpace
from dataset.builder import DatasetBuilder
from dataset.render_backends import (
    FreshProcessRenderBackend,
    ParallelFreshProcessRenderBackend,
    RenderSettings,
    make_wrapper,
    synth_plugin_path,
)
from dataset.preset_sources import (
    HumanPresetSource,
    HybridPresetSource,
    SyntheticPresetSource,
)
from dataset.dexed_preset_loader import DexedPresetLoader
from dataset.diva_preset_loader import DivaPresetLoader
from dataset.preset_loader_common import PresetSplit, restrict_to_realized
import config

SYNTH_CHOICES = ["dexed", "diva"]

# Diva does not reproduce in-process at all, so every Diva partition renders in a fresh
# process regardless of --fresh-process (D-DIVA-RENDER, docs/DECISIONS.md).
_ALWAYS_FRESH_PROCESS = {"diva"}


def _make_synth(synth_name: str) -> BaseSynthesizer:
    plugin_path = synth_plugin_path(synth_name)
    if not os.path.exists(plugin_path):
        print(f"Could not find the {synth_name} plugin at: {plugin_path}")
        print(f"Please update {synth_name.upper()}_PATH in your .env file.")
        sys.exit(1)
    synth = make_wrapper(renderer="dawdreamer", synth_name=synth_name)
    print(
        f"Initialized {synth_name} at {synth.sample_rate}Hz; "
        f"subset = {len(synth.parameter_space.names)} params"
    )
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


def _load_split(synth, args: argparse.Namespace) -> PresetSplit:
    """Load, deduplicate and split the human presets for whichever synth is being built.

    Only the reading differs per synth: Dexed takes any number of ``.syx`` cartridges,
    Diva takes the one Flow Synthesizer collection (``diva_raw.zip`` or an extracted
    directory). One split feeds both partitions, so train and test are disjoint by
    construction (no seed to re-match across separate runs).
    """
    space = synth.parameter_space
    if args.synth == "diva":
        source_path = os.path.expanduser(args.presets[0] if args.presets else config.DIVA_RAW_PATH)
        if len(args.presets or []) > 1:
            print("--presets takes a single path for diva (the corpus zip or directory).")
            sys.exit(1)
        if not os.path.exists(source_path):
            print(f"Diva preset collection not found at: {source_path}")
            print("Download diva_raw.zip (see dataset/diva_preset_loader.py) or set DIVA_RAW_PATH.")
            sys.exit(1)
        print(f"--- Reading Diva presets from {source_path} ---")
        return DivaPresetLoader(
            space,
            test_fraction=args.test_fraction,
            split_seed=args.split_seed,
            dedup_threshold=args.dedup_threshold,
        ).load(source_path, limit=args.limit, show_progress=True)

    if not args.presets:
        print("--cartridges is required for dexed.")
        sys.exit(1)
    cartridges = _resolve_cartridges(args.presets)
    split = DexedPresetLoader(
        space,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        dedup_threshold=args.dedup_threshold,
    ).load(cartridges)
    if args.limit is not None:
        print("--limit is ignored for dexed (.syx cartridges are read whole).")
    return split


def _corpus_space(
    split: PresetSplit, space: ParameterSpace, keep_constant: bool
) -> tuple:
    """Narrow the space to what these presets actually vary, and report what was frozen.

    The corpus-variance rule (D-DIVA-SUBSET, docs/DECISIONS.md): a parameter constant
    across every preset is free for a model to "predict", so it leaves the estimated set
    and is locked at the source's own value instead of the plugin's init patch. A no-op
    on preset collections that vary everything, which is the Dexed case.
    """
    narrowed, frozen = restrict_to_realized(split.train + split.test, space)
    if not frozen:
        return space, {}
    if keep_constant:
        print(
            f"--- {len(frozen)} of {len(space.names)} subset params are constant across these "
            "presets; kept anyway (--keep-constant-params) ---"
        )
        return space, {}
    print(
        f"--- {len(frozen)} of {len(space.names)} subset params are constant across these "
        f"presets: dropped, and locked at the presets' own values ---"
    )
    print(
        f"    corpus estimates {len(narrowed.names)} params, "
        f"ML dimension {narrowed.ml_dimension} (was {space.ml_dimension})"
    )
    return narrowed, frozen


def _report(summary: dict, run_dir: Path) -> None:
    print("\nSuccess!")
    print(f"Samples: {summary['num_samples']} | near-silent: {summary['near_silent_count']}")
    print(f"Methods: {summary['method_counts']}")
    print(f"Renderer: {summary['renderer']} | git revision: {summary['git_revision']}")
    print(f"Written to: {run_dir}")
    print(f"  {run_dir / 'run_summary.json'}")
    print(f"  {run_dir / 'metadata.csv'}")
    print(f"  {run_dir / 'audio'}/*.wav")


def _build(
    synth: BaseSynthesizer,
    source,
    run_name: str,
    synth_name: str,
    fresh_process: bool = False,
    workers: int = 1,
    parameter_space: Optional[ParameterSpace] = None,
    default_params: Optional[dict] = None,
) -> None:
    # Fresh-process rendering (one clean spawned worker per preset) is for test/eval
    # corpora, where the generation and evaluation render contexts must agree (D-REPRO);
    # training data stays on the fast in-process path. Diva has no in-process path at all
    # (D-DIVA-RENDER). The builder closes the backend.
    fresh_process = fresh_process or synth_name in _ALWAYS_FRESH_PROCESS
    settings = RenderSettings.from_config()
    backend = None
    if fresh_process and workers > 1:
        # Same per-render isolation, spread over a pool: every patch still lands on its own
        # single-use process, so only throughput changes.
        backend = ParallelFreshProcessRenderBackend(
            settings, renderer="dawdreamer", synth_name=synth_name, num_workers=workers
        )
        print(f"--- Rendering in fresh spawned processes, {workers} at a time (D-REPRO) ---")
    elif fresh_process:
        backend = FreshProcessRenderBackend(
            settings, renderer="dawdreamer", synth_name=synth_name
        )
        print("--- Rendering in fresh spawned processes (one per preset; D-REPRO) ---")
    summary = DatasetBuilder(
        synth,
        render_backend=backend,
        parameter_space=parameter_space,
        default_params=default_params,
    ).build(source, run_name=run_name, show_progress=True)
    _report(summary, Path(config.DATASET_DIR) / run_name)


def build_synthetic(args: argparse.Namespace) -> None:
    synth = _make_synth(args.synth)
    print(f"--- Building '{args.run_name}': {args.count} synthetic presets (seed {args.seed}) ---")
    source = SyntheticPresetSource(
        synth.parameter_space,
        count=args.count,
        seed=args.seed,
        sampling_ranges=synth.audible_sampling_ranges,
    )
    _build(synth, source, args.run_name, args.synth,
           fresh_process=args.fresh_process, workers=args.workers)


def _human_run_name(custom: Optional[str], partition: str, render_both: bool) -> str:
    """Output folder for one human partition. Default ``human_<partition>``; a custom
    name is suffixed per partition only when both are rendered, so they never collide."""
    if custom is None:
        return f"human_{partition}"
    return f"{custom}_{partition}" if render_both else custom


def build_human(args: argparse.Namespace) -> None:
    synth = _make_synth(args.synth)
    split = _load_split(synth, args)
    requested = [args.partition] if args.partition else ["train", "test"]
    # Skip any partition the split left empty (e.g. test when --test-fraction is 0).
    partitions = [p for p in requested if (split.test if p == "test" else split.train)]
    render_both = len(partitions) == 2
    print(f"--- Human split: {len(split.train)} train / {len(split.test)} test after dedup ---")
    if not partitions:
        print(f"No presets to render for partition(s) {requested}. Nothing written.")
        return
    # Both partitions share one narrowed space, or the two corpora would not be comparable.
    space, frozen = _corpus_space(split, synth.parameter_space, args.keep_constant_params)
    for partition in partitions:
        presets = split.test if partition == "test" else split.train
        # The test set renders in fresh processes (D-REPRO, generation/eval contexts must
        # agree); train renders fast in-process. --fresh-process forces fresh on either.
        fresh_process = args.fresh_process or partition == "test"
        run_name = _human_run_name(args.run_name, partition, render_both)
        print(f"--- Building '{run_name}': {len(presets)} {partition} presets ---")
        source = HumanPresetSource(presets, space, partition=partition)
        _build(
            synth, source, run_name, args.synth, fresh_process=fresh_process,
            workers=args.workers, parameter_space=space, default_params=frozen,
        )


def build_hybrid(args: argparse.Namespace) -> None:
    synth = _make_synth(args.synth)
    split = _load_split(synth, args)
    space, frozen = _corpus_space(split, synth.parameter_space, args.keep_constant_params)
    # Parents come only from the human train partition (never the held-out test set).
    parents = list(HumanPresetSource(split.train, space, "train").iter_presets())
    print(
        f"--- Building '{args.run_name}': {args.count} hybrid presets "
        f"(mode={args.mode}, seed {args.seed}, {len(parents)} human-train parents) ---"
    )
    source = HybridPresetSource(
        mode=args.mode,
        human_presets=parents,
        parameter_space=space,
        count=args.count,
        seed=args.seed,
        synthetic_ratio=args.synthetic_ratio,
        num_perturbed_params=args.num_perturbed_params,
        jitter=args.jitter,
        flip_categoricals=args.flip_categoricals,
        sampling_ranges=synth.audible_sampling_ranges,
    )
    _build(
        synth, source, args.run_name, args.synth, fresh_process=args.fresh_process,
        workers=args.workers, parameter_space=space, default_params=frozen,
    )


def _add_synth_flag(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--synth", choices=SYNTH_CHOICES, default="dexed",
        help="which synthesizer to render (default: dexed)",
    )


def _add_preset_source_flags(subparser: argparse.ArgumentParser) -> None:
    """The human-preset source and how it is deduplicated and split."""
    subparser.add_argument(
        "--presets", "--cartridges", dest="presets", nargs="+", default=None,
        help="dexed: .syx paths, folders or globs (required). "
        "diva: the one preset collection, a diva_raw.zip or an extracted directory "
        "(defaults to config.DIVA_RAW_PATH)",
    )
    subparser.add_argument("--limit", type=int, default=None,
                           help="diva: cap on raw presets read before dedup")
    subparser.add_argument("--test-fraction", type=float, default=0.0,
                           help="share held out for test; 0.0 (default) renders every preset as train")
    subparser.add_argument("--split-seed", type=int, default=0, help="seed for the train/test split")
    subparser.add_argument("--dedup-threshold", type=float, default=1e-3, help="duplicate distance")
    subparser.add_argument(
        "--keep-constant-params", action="store_true",
        help="keep subset parameters the presets never vary, instead of dropping them and "
        "locking them at the presets' own values (D-DIVA-SUBSET's corpus-variance rule)",
    )


def _add_fresh_process_flag(subparser: argparse.ArgumentParser) -> None:
    subparser.add_argument(
        "--fresh-process",
        action="store_true",
        help="force every rendered partition into its own clean spawned process "
        "(slow, leak-free; D-REPRO). human renders its test partition this way "
        "regardless; pass this to force it for train too",
    )
    subparser.add_argument(
        "--workers", type=int, default=1,
        help="render this many presets in parallel. Only affects fresh-process "
        "partitions, where each patch still gets its own single-use process, so the "
        "audio is unchanged and only throughput differs. Default 1 (serial)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a dataset corpus.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("synthetic", help="random draws over the parameter space")
    synthetic.add_argument("--count", type=int, default=16, help="number of presets to render")
    synthetic.add_argument("--seed", type=int, default=0, help="master seed for the sampler")
    synthetic.add_argument("--run-name", default="synthetic_smoke", help="output subdirectory name")
    _add_synth_flag(synthetic)
    _add_fresh_process_flag(synthetic)
    synthetic.set_defaults(func=build_synthetic)

    human = subparsers.add_parser("human", help="real presets projected onto the subset")
    human.add_argument("--partition", choices=["train", "test"], default=None,
                       help="render only this partition; default renders both")
    human.add_argument("--run-name", default=None, help="output subdirectory name")
    _add_synth_flag(human)
    _add_preset_source_flags(human)
    _add_fresh_process_flag(human)
    human.set_defaults(func=build_human)

    hybrid = subparsers.add_parser("hybrid", help="human-train presets blended/augmented with synthetic")
    hybrid.add_argument("--mode", choices=[HybridPresetSource.BLEND, HybridPresetSource.AUGMENT], default=HybridPresetSource.BLEND)
    hybrid.add_argument("--count", type=int, default=64, help="number of presets to render")
    hybrid.add_argument("--seed", type=int, default=0, help="master seed for the sampler")
    hybrid.add_argument("--synthetic-ratio", type=float, default=0.5, help="blend: P(synthetic per slot)")
    hybrid.add_argument("--num-perturbed-params", type=int, default=2, help="augment: params jittered/flipped")
    hybrid.add_argument("--jitter", type=float, default=0.05, help="augment: continuous jitter magnitude")
    hybrid.add_argument("--flip-categoricals", action="store_true", help="augment: allow categorical flips")
    hybrid.add_argument("--run-name", default=None, help="output subdirectory name")
    _add_synth_flag(hybrid)
    _add_preset_source_flags(hybrid)
    _add_fresh_process_flag(hybrid)
    hybrid.set_defaults(func=build_hybrid)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    # human resolves its own per-partition run names (it may render both at once).
    if args.command == "hybrid" and args.run_name is None:
        args.run_name = f"hybrid_{args.mode}"
    args.func(args)


if __name__ == "__main__":
    main()
