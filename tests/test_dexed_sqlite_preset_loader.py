import io
import os
import sqlite3
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synth.parameter_space import ParameterSpecification, ParameterSpace
from dataset.dexed_sqlite_preset_loader import DexedSqlitePresetLoader


# ---------------------------------------------------------------------------
# Pure-Python: a tiny SQLite database built row-by-row, no VST and no 25 MB DB.
# The param table names are real Dexed names (a subset of the plugin's), which
# is all the name-based adapter needs; vectors are hand-crafted.
# ---------------------------------------------------------------------------

# Real Dexed parameter names, matching the fixture vector order below.
FIXTURE_PARAM_NAMES = ["OP1 OUTPUT LEVEL", "OP1 F FINE", "ALGORITHM"]


def make_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="OP1 OUTPUT LEVEL", kind="continuous", default=0.0),
        ParameterSpecification(name="OP1 F FINE", kind="continuous", default=0.0),
        ParameterSpecification(
            name="ALGORITHM", kind="categorical",
            options=[n / 31 for n in range(32)], default=0.0,
        ),
    ])


def _pickle_vector(vector: np.ndarray) -> bytes:
    """Encode a vector the way preset-gen-vae stores it (np.save into a BLOB)."""
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(vector, dtype=np.float32))
    buffer.seek(0)
    return buffer.read()


def write_db(tmp_path, presets, param_names=FIXTURE_PARAM_NAMES, name="dexed_presets.sqlite") -> str:
    """Create a fixture DB. ``presets`` is a list of (name, vector) pairs."""
    db_path = str(tmp_path / name)
    connection = sqlite3.connect(db_path)
    connection.execute("CREATE TABLE param (index_param INTEGER, name TEXT)")
    connection.execute(
        "CREATE TABLE preset (index_preset INTEGER, name TEXT, pickled_params_np_array BLOB)"
    )
    connection.executemany(
        "INSERT INTO param (index_param, name) VALUES (?, ?)",
        list(enumerate(param_names)),
    )
    connection.executemany(
        "INSERT INTO preset (index_preset, name, pickled_params_np_array) VALUES (?, ?, ?)",
        [
            (index, preset_name, _pickle_vector(vector))
            for index, (preset_name, vector) in enumerate(presets)
        ],
    )
    connection.commit()
    connection.close()
    return db_path


def voice(output_level: float = 0.0, f_fine: float = 0.0, algorithm: float = 0.0):
    return np.array([output_level, f_fine, algorithm], dtype=np.float32)


def test_unpickles_and_maps_params_by_name(tmp_path):
    space = make_space()
    db_path = write_db(tmp_path, [("BRITE RDS", voice(output_level=0.5, f_fine=0.25))])
    split = DexedSqlitePresetLoader(space, test_fraction=0.0).load(db_path)
    assert len(split.train) == 1
    params = split.train[0].params
    assert params["OP1 OUTPUT LEVEL"] == pytest.approx(0.5)
    assert params["OP1 F FINE"] == pytest.approx(0.25)
    assert set(params) == set(FIXTURE_PARAM_NAMES)  # mapped by name, covers the subset


def test_extra_db_params_are_carried_but_projected_out_later(tmp_path):
    # A real DB names 155 params; only the subset matters. Extra names must load
    # fine (HumanPresetSource projects them out downstream).
    space = make_space()
    param_names = ["Cutoff"] + FIXTURE_PARAM_NAMES  # a non-subset param up front
    db_path = write_db(
        tmp_path,
        [("PAD", np.array([0.9, 0.5, 0.25, 0.0], dtype=np.float32))],
        param_names=param_names,
    )
    split = DexedSqlitePresetLoader(space, test_fraction=0.0).load(db_path)
    params = split.train[0].params
    assert params["Cutoff"] == pytest.approx(0.9)         # carried
    assert params["OP1 OUTPUT LEVEL"] == pytest.approx(0.5)  # still mapped by name


def test_raises_when_a_subset_name_is_absent(tmp_path):
    space = make_space()
    # param table omits ALGORITHM, which the subset requires.
    db_path = write_db(
        tmp_path,
        [("X", np.array([0.5, 0.25], dtype=np.float32))],
        param_names=["OP1 OUTPUT LEVEL", "OP1 F FINE"],
    )
    with pytest.raises(RuntimeError, match="not present in the preset database"):
        DexedSqlitePresetLoader(space, test_fraction=0.0).load(db_path)


def test_raises_on_vector_length_mismatch(tmp_path):
    space = make_space()
    # 3 param names but a 2-long vector.
    db_path = write_db(tmp_path, [("X", np.array([0.5, 0.25], dtype=np.float32))])
    with pytest.raises(ValueError, match="expected"):
        DexedSqlitePresetLoader(space, test_fraction=0.0).load(db_path)


def test_dedup_collapses_identical_voices(tmp_path):
    space = make_space()
    presets = [(f"dup{i}", voice(output_level=0.4)) for i in range(10)]  # all identical
    db_path = write_db(tmp_path, presets)
    split = DexedSqlitePresetLoader(space, test_fraction=0.0).load(db_path)
    assert len(split.train) == 1


def test_distinct_algorithms_are_not_deduplicated(tmp_path):
    space = make_space()
    presets = [(f"alg{i}", voice(output_level=0.4, algorithm=i / 31)) for i in range(32)]
    db_path = write_db(tmp_path, presets)
    split = DexedSqlitePresetLoader(space, test_fraction=0.0).load(db_path)
    assert len(split.train) == 32  # one-hot ALGORITHM blocks differ


def test_limit_caps_raw_voices(tmp_path):
    space = make_space()
    presets = [(f"v{i}", voice(output_level=i / 20)) for i in range(20)]
    db_path = write_db(tmp_path, presets)
    split = DexedSqlitePresetLoader(space, test_fraction=0.0).load(db_path, limit=5)
    assert len(split.train) == 5


def test_split_is_disjoint_deterministic_and_correctly_sized(tmp_path):
    space = make_space()
    presets = [(f"v{i}", voice(output_level=(i + 1) / 20)) for i in range(8)]  # all distinct
    db_path = write_db(tmp_path, presets)

    split = DexedSqlitePresetLoader(space, test_fraction=0.25, split_seed=123).load(db_path)
    assert len(split.test) == 2  # round(8 * 0.25)
    assert len(split.train) == 6

    train_ids = {(p.source_file, p.voice_index) for p in split.train}
    test_ids = {(p.source_file, p.voice_index) for p in split.test}
    assert train_ids.isdisjoint(test_ids)

    again = DexedSqlitePresetLoader(space, test_fraction=0.25, split_seed=123).load(db_path)
    assert [p.voice_index for p in again.test] == [p.voice_index for p in split.test]


def test_provenance_records_source_and_voice_name(tmp_path):
    space = make_space()
    db_path = write_db(tmp_path, [("PIANO 1  ", voice(output_level=0.5))], name="dexed_presets.sqlite")
    split = DexedSqlitePresetLoader(space, test_fraction=0.0).load(db_path)
    preset = split.train[0]
    assert preset.source_file == "dexed_presets.sqlite"
    assert preset.voice_index == 0
    assert preset.voice_name == "PIANO 1"  # trailing spaces stripped
