"""The build script's synth-neutral CLI surface and its corpus-narrowing step.

Pure Python: parses argv and exercises the narrowing helper against hand-built presets.
Nothing here loads a plugin or renders.
"""
import importlib.util
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synth.parameter_space import ParameterSpecification, ParameterSpace
from dataset.preset_loader_common import LoadedPreset, PresetSplit


def _load_build_dataset():
    """Import scripts/build_dataset.py by path (scripts/ is not a package)."""
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "build_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("build_dataset_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_dataset = _load_build_dataset()


# --- the --synth flag and the preset-source flag ----------------------------

def test_synth_defaults_to_dexed_on_every_subcommand():
    parser = build_dataset.build_parser()
    for argv in (
        ["synthetic"],
        ["human", "--presets", "a.syx"],
        ["hybrid", "--presets", "a.syx"],
    ):
        assert parser.parse_args(argv).synth == "dexed"


def test_synth_accepts_diva_and_rejects_anything_else():
    parser = build_dataset.build_parser()
    assert parser.parse_args(["human", "--synth", "diva"]).synth == "diva"
    with pytest.raises(SystemExit):
        parser.parse_args(["human", "--synth", "moog"])


def test_cartridges_still_spells_the_preset_source():
    # The Dexed invocations that predate the flag being synth-neutral must keep working.
    parser = build_dataset.build_parser()
    legacy = parser.parse_args(["human", "--cartridges", "a.syx", "b.syx"])
    current = parser.parse_args(["human", "--presets", "a.syx", "b.syx"])
    assert legacy.presets == current.presets == ["a.syx", "b.syx"]


def test_preset_source_is_optional_so_diva_falls_back_to_its_default_collection():
    assert build_dataset.build_parser().parse_args(["human", "--synth", "diva"]).presets is None


def test_diva_always_renders_fresh_process():
    # Diva does not reproduce in-process at all (D-DIVA-RENDER), so the train partition
    # cannot take the fast path the way Dexed's does.
    assert "diva" in build_dataset._ALWAYS_FRESH_PROCESS
    assert "dexed" not in build_dataset._ALWAYS_FRESH_PROCESS


# --- the corpus-variance rule as the script applies it ----------------------

def make_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="VARIES", kind="continuous", default=0.0),
        ParameterSpecification(name="FROZEN", kind="continuous", default=0.0),
    ])


def make_split(frozen_value: float = 0.7) -> PresetSplit:
    presets = [
        LoadedPreset(
            params={"VARIES": index / 10, "FROZEN": frozen_value},
            source_file="corpus", voice_index=index, voice_name=f"v{index}",
        )
        for index in range(1, 5)
    ]
    return PresetSplit(train=presets[:3], test=presets[3:])


def test_corpus_space_drops_constants_and_reports_their_values(capsys):
    space, frozen = build_dataset._corpus_space(make_split(), make_space(), keep_constant=False)
    assert space.names == ["VARIES"]
    assert frozen == {"FROZEN": pytest.approx(0.7)}
    assert "1 of 2 subset params are constant" in capsys.readouterr().out


def test_corpus_space_narrowing_spans_train_and_test():
    # Both partitions share one space, or the two corpora would not be comparable. A
    # parameter that varies only inside the test half must therefore survive.
    split = make_split()
    split.test[0].params["FROZEN"] = 0.2
    space, frozen = build_dataset._corpus_space(split, make_space(), keep_constant=False)
    assert space.names == ["VARIES", "FROZEN"]
    assert frozen == {}


def test_corpus_space_is_a_no_op_when_the_presets_vary_everything():
    split = make_split()
    for index, preset in enumerate(split.train + split.test):
        preset.params["FROZEN"] = index / 8
    space, frozen = build_dataset._corpus_space(split, make_space(), keep_constant=False)
    assert space.names == make_space().names
    assert frozen == {}


def test_keep_constant_params_opts_out(capsys):
    space, frozen = build_dataset._corpus_space(make_split(), make_space(), keep_constant=True)
    assert space.names == make_space().names   # nothing dropped
    assert frozen == {}                        # so nothing to lock
    assert "kept anyway" in capsys.readouterr().out
