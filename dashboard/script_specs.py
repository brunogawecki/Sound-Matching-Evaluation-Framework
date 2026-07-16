"""Declarative flag-specs mirroring each pipeline script's argparse signature.

One :class:`ScriptSpec` per script (or subcommand). These are transcribed by hand
from the ``scripts/*.py`` argparse setup and are the single place a new script is
registered -- adding a future ``train_deep.py`` page is one more ScriptSpec here,
no page rewrite (the "generic seam").

Kinds:
    int / float / str    scalar; emitted as ``--flag value`` when non-empty
    choice               one of ``choices``; emitted when non-empty
    bool                 store_true flag; emitted (bare) only when True
    path                 like str, but the UI renders a path field
    paths                nargs="+"; one entry per line -> ``--flag a b c`` (a path may
                         itself contain spaces, e.g. macOS "Application Support")

An empty/None value for a non-required arg is omitted, so the script's own
default applies (e.g. ``--run-name`` left blank -> the script picks the name).
"""
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple


@dataclass(frozen=True)
class ArgSpec:
    name: str  # dict key / widget identity
    flag: str  # "--count"
    kind: str  # int | float | str | choice | bool | path | paths
    default: Any = ""
    choices: Tuple[str, ...] = ()
    required: bool = False
    help: str = ""
    label: str = ""  # UI label; falls back to flag


@dataclass(frozen=True)
class ScriptSpec:
    key: str  # UI label for this command
    script: str  # "scripts/build_dataset.py" (relative to project root)
    subcommand: str = ""  # "synthetic" | "" if none
    args: Tuple[ArgSpec, ...] = field(default_factory=tuple)
    description: str = ""


# --- shared arg groups ------------------------------------------------------

_CARTRIDGES = ArgSpec(
    "cartridges", "--cartridges", "paths", required=True,
    help="One .syx path, glob, or folder per line (a folder recurses for *.syx).",
    label="Cartridges (.syx paths / globs / folders)",
)
_SPLIT_ARGS = (
    ArgSpec("test_fraction", "--test-fraction", "float", 0.0,
            help="Share held out as the test set; 0.0 renders every voice as train."),
    ArgSpec("split_seed", "--split-seed", "int", 0, help="Seed for the train/test shuffle."),
    ArgSpec("dedup_threshold", "--dedup-threshold", "float", 0.001,
            help="Distance below which two presets collapse to one."),
)
_FRESH = ArgSpec("fresh_process", "--fresh-process", "bool", False,
                 help="Force every partition into a clean spawned process (slow, leak-free; D-REPRO).")
_RUN_NAME_OPT = ArgSpec("run_name", "--run-name", "str", "",
                        help="Output subdirectory name. Leave blank for the script default.")


# --- the specs --------------------------------------------------------------

BUILD_SYNTHETIC = ScriptSpec(
    key="synthetic",
    script="scripts/build_dataset.py",
    subcommand="synthetic",
    description="Random draws over the (locked D1) parameter space; no presets needed.",
    args=(
        ArgSpec("count", "--count", "int", 16, help="How many presets to render."),
        ArgSpec("seed", "--seed", "int", 0, help="Master seed for the sampler."),
        ArgSpec("run_name", "--run-name", "str", "synthetic_smoke", help="Output subdirectory name."),
        _FRESH,
    ),
)

BUILD_HUMAN = ScriptSpec(
    key="human",
    script="scripts/build_dataset.py",
    subcommand="human",
    description="Real DX7 voices from .syx cartridges, projected onto the D1 subset.",
    args=(
        _CARTRIDGES,
        ArgSpec("partition", "--partition", "choice", "", choices=("train", "test"),
                help="Render only this partition. Blank renders both."),
        *_SPLIT_ARGS,
        _RUN_NAME_OPT,
        _FRESH,
    ),
)

BUILD_HYBRID = ScriptSpec(
    key="hybrid",
    script="scripts/build_dataset.py",
    subcommand="hybrid",
    description="Human-train presets blended with, or augmented by, synthetic material.",
    args=(
        _CARTRIDGES,
        ArgSpec("mode", "--mode", "choice", "blend", choices=("blend", "augment"),
                help="blend mixes in whole synthetic draws; augment perturbs human presets."),
        ArgSpec("count", "--count", "int", 64, help="How many presets to render."),
        ArgSpec("seed", "--seed", "int", 0, help="Master seed for the sampler."),
        ArgSpec("synthetic_ratio", "--synthetic-ratio", "float", 0.5,
                help="blend only: probability each slot is synthetic."),
        ArgSpec("num_perturbed_params", "--num-perturbed-params", "int", 2,
                help="augment only: how many parameters to change."),
        ArgSpec("jitter", "--jitter", "float", 0.05, help="augment only: continuous nudge size."),
        ArgSpec("flip_categoricals", "--flip-categoricals", "bool", False,
                help="augment only: also allow categorical params to flip."),
        *_SPLIT_ARGS,
        _RUN_NAME_OPT,
        _FRESH,
    ),
)

BUILD_PRESETGEN = ScriptSpec(
    key="presetgen",
    script="scripts/build_presetgen_corpus.py",
    description="Train corpus from the preset-gen-vae DX7 SQLite collection (in-process render).",
    args=(
        ArgSpec("db_path", "--db-path", "path", "",
                help="Path to dexed_presets.sqlite. Blank uses config.PRESETGEN_DB_PATH."),
        ArgSpec("limit", "--limit", "str", "",
                help="Cap on raw voices read before dedup/split. Blank uses the whole ~30k collection."),
        ArgSpec("run_name", "--run-name", "str", "presetgen_train", help="Output subdirectory name."),
        *_SPLIT_ARGS,
    ),
)

SPLIT_CORPUS = ScriptSpec(
    key="split",
    script="scripts/split_corpus.py",
    description="Split an already-built corpus into a train corpus (audio copied) and a "
                "test corpus (re-rendered fresh-process; D-REPRO). Not for hybrid corpora.",
    args=(
        ArgSpec("corpus", "--corpus", "path", "", required=True,
                help="Source corpus directory (picked from the dropdown below)."),
        ArgSpec("test_fraction", "--test-fraction", "float", 0.2,
                help="Share of samples held out as the test set."),
        ArgSpec("split_seed", "--split-seed", "int", 0, help="Seed for the train/test row shuffle."),
        ArgSpec("run_name", "--run-name", "str", "",
                help="Output base name; _train/_test are appended. Blank uses the source name."),
    ),
)

# Mirror of models.registry.MODEL_REGISTRY keys. Kept as plain strings (not
# imported) so the dashboard never pulls in the torch-heavy pipeline library.
MODEL_CHOICES = (
    "MeanParameterBaseline",
    "Sound2SynthSpectrogramRegressor",
    "PresetGenVAEMLPRegressor",
    "PresetGenVAEFlowRegressor",
    "IS",
    "IS2xITF",
    "IS2",
)

EVALUATE = ScriptSpec(
    key="evaluate",
    script="scripts/evaluate.py",
    description="Score a fitted checkpoint on a fresh-process corpus through the metric panel.",
    args=(
        ArgSpec("checkpoint", "--checkpoint", "path", "", required=True,
                help="Saved model file to load and fingerprint."),
        ArgSpec("corpus", "--corpus", "path", "", required=True,
                help="Eval corpus directory (must be fresh-process)."),
        ArgSpec("model", "--model", "choice", "MeanParameterBaseline",
                choices=MODEL_CHOICES, required=True,
                help="Model class to load the checkpoint into."),
        ArgSpec("out", "--out", "path", "", help="Results root. Blank uses <project>/results."),
        ArgSpec("save_audio", "--save-audio", "bool", False,
                help="Persist prediction WAVs for a seeded random sample subset."),
        ArgSpec("save_audio_n", "--save-audio-n", "int", 20,
                help="Cap on how many samples get their prediction audio saved."),
    ),
)

# Registered by the "Build dataset" page radio (preset source -> spec).
BUILD_SOURCES = {
    "synthetic": BUILD_SYNTHETIC,
    "human": BUILD_HUMAN,
    "hybrid": BUILD_HYBRID,
    "presetgen": BUILD_PRESETGEN,
}

