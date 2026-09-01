"""DivaH2pPresetLoader: load, deduplicate and split Diva's installed .h2p preset library.

Deliberately parallel to test_diva_preset_loader.py (the npz-source sibling): the same
dedup/split/coverage behaviour, inherited from preset_loader_common.py, over a different file
format. Parsing and decoding are pinned separately in test_diva_h2p_patch.py; here the map is
monkeypatched to a small fixture so these tests exercise the loader's own logic, not today's
specific key assignments.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synth.diva import patch as diva_patch
from synth.diva.h2p_param_map import Decoding
from synth.parameter_space import ParameterSpecification, ParameterSpace
from dataset.diva_h2p_preset_loader import DivaH2pPresetLoader

_FIXTURE_MAP = {
    ("ENV1", "Atk"): Decoding("ENV1.Attack", "linear", minimum=0.0, maximum=100.0),
    ("VCC", "Trsp"): Decoding("VCC.Transpose", "grid", grid=("-24.00", "0.00", "24.00")),
}


@pytest.fixture(autouse=True)
def fixture_map(monkeypatch):
    monkeypatch.setattr(diva_patch, "H2P_PARAMETER_MAP", _FIXTURE_MAP)


def make_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="ENV1.Attack", kind="continuous", default=0.0),
        ParameterSpecification(
            name="VCC.Transpose", kind="categorical",
            options=[0.0, 0.5, 1.0], default=0.5,
        ),
    ])


def write_h2p(directory, filename, attack: float, transpose: float) -> None:
    lines = [
        "#AM=Diva",
        "#cm=ENV1", f"Atk={attack:.2f}",
        "#cm=VCC", f"Trsp={transpose:.2f}",
        "// Section for ugly compressed binary Data",
        "garbage",
    ]
    (directory / filename).write_text("\n".join(lines), encoding="latin-1")


def write_library(tmp_path, presets, name="Diva"):
    """presets: list of (subdirectory, filename, attack, transpose)."""
    root = tmp_path / name
    for subdirectory, filename, attack, transpose in presets:
        target = root / subdirectory
        target.mkdir(parents=True, exist_ok=True)
        write_h2p(target, filename, attack, transpose)
    return str(root)


def test_loads_and_maps_h2p_presets(tmp_path):
    space = make_space()
    path = write_library(tmp_path, [("1 BASS", "Warm Bass.h2p", 25.0, 0.0)])
    split = DivaH2pPresetLoader(space, test_fraction=0.0).load(path)
    assert len(split.train) == 1
    preset = split.train[0]
    assert preset.params["ENV1.Attack"] == pytest.approx(0.25)
    assert preset.params["VCC.Transpose"] == pytest.approx(0.5)
    assert preset.voice_name == "Warm Bass"


def test_recurses_into_category_subdirectories(tmp_path):
    space = make_space()
    path = write_library(tmp_path, [
        ("1 BASS", "A.h2p", 0.0, 0.0),
        ("2 LEAD", "B.h2p", 50.0, 0.0),
        ("THIRD PARTY", "C.h2p", 100.0, 24.0),
    ])
    split = DivaH2pPresetLoader(space, test_fraction=0.0).load(path)
    assert len(split.train) == 3


def test_provenance_keeps_third_party_membership_recoverable(tmp_path):
    space = make_space()
    path = write_library(tmp_path, [("THIRD PARTY", "Foo.h2p", 0.0, 0.0)])
    preset = DivaH2pPresetLoader(space).load(path).train[0]
    assert preset.source_file == os.path.join("THIRD PARTY", "Foo.h2p")


def test_raises_when_a_subset_name_is_absent(tmp_path):
    space = ParameterSpace([
        ParameterSpecification(name="ENV1.Attack", kind="continuous", default=0.0),
        ParameterSpecification(name="HPF.Frequency", kind="continuous", default=0.0),
    ])
    path = write_library(tmp_path, [("1 BASS", "A.h2p", 0.0, 0.0)])
    with pytest.raises(RuntimeError, match="Subset parameter names not present"):
        DivaH2pPresetLoader(space).load(path)


def test_raises_on_an_empty_directory(tmp_path):
    space = make_space()
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="No .h2p presets found"):
        DivaH2pPresetLoader(space).load(str(empty))


def test_raises_on_a_path_that_is_not_a_directory(tmp_path):
    space = make_space()
    stray = tmp_path / "notes.txt"
    stray.write_text("not a library")
    with pytest.raises(FileNotFoundError, match="is not a directory"):
        DivaH2pPresetLoader(space).load(str(stray))


def test_dedup_collapses_identical_presets(tmp_path):
    space = make_space()
    path = write_library(tmp_path, [
        ("1 BASS", f"{i}.h2p", 40.0, 0.0) for i in range(5)
    ])
    split = DivaH2pPresetLoader(space).load(path)
    assert len(split.train) == 1


def test_distinct_presets_are_not_deduplicated(tmp_path):
    space = make_space()
    path = write_library(tmp_path, [
        ("1 BASS", f"{i}.h2p", i * 10.0, 0.0) for i in range(5)
    ])
    split = DivaH2pPresetLoader(space).load(path)
    assert len(split.train) == 5


def test_limit_caps_raw_presets(tmp_path):
    space = make_space()
    path = write_library(tmp_path, [
        ("1 BASS", f"{i}.h2p", i * 5.0, 0.0) for i in range(20)
    ])
    split = DivaH2pPresetLoader(space).load(path, limit=5)
    assert len(split.train) == 5


def test_split_is_disjoint_deterministic_and_correctly_sized(tmp_path):
    space = make_space()
    path = write_library(tmp_path, [
        ("1 BASS", f"{i}.h2p", i * 12.0, 0.0) for i in range(8)
    ])

    split = DivaH2pPresetLoader(space, test_fraction=0.25, split_seed=123).load(path)
    assert len(split.test) == 2  # round(8 * 0.25)
    assert len(split.train) == 6
    assert {p.voice_index for p in split.train}.isdisjoint({p.voice_index for p in split.test})

    again = DivaH2pPresetLoader(space, test_fraction=0.25, split_seed=123).load(path)
    assert [p.voice_index for p in again.test] == [p.voice_index for p in split.test]
