import io
import os
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from synth.parameter_space import ParameterSpecification, ParameterSpace
from dataset.diva_preset_loader import DivaPresetLoader
from dataset.preset_loader_common import constant_parameters, restrict_to_realized


# ---------------------------------------------------------------------------
# Pure-Python: hand-built .npz presets in the corpus' own shape, no VST and no
# 1.4 GB archive. Names are real Diva names written the corpus' way ('Module: Name').
# ---------------------------------------------------------------------------
FIXTURE_PARAM_NAMES = ["VCF1: Frequency", "OSC: Volume1", "VCF1: Model"]


def make_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="VCF1.Frequency", kind="continuous", default=0.0),
        ParameterSpecification(name="OSC.Volume1", kind="continuous", default=0.0),
        ParameterSpecification(
            name="VCF1.Model", kind="categorical",
            options=[n / 4 for n in range(5)], default=0.0,
        ),
    ])


def preset_bytes(values, param_names=FIXTURE_PARAM_NAMES) -> bytes:
    """One preset in the corpus' layout: 'param' plus the 'audio'/'chars' the loader ignores."""
    buffer = io.BytesIO()
    np.savez(
        buffer,
        param=np.array(dict(zip(param_names, values)), dtype=object),
        audio=np.zeros(8, dtype=np.float16),
        chars=np.zeros((10, 3)),
    )
    return buffer.getvalue()


def write_directory(tmp_path, presets, name="raw") -> str:
    """Write presets as ``<name>/raw/<hash>_60_100.npz``, mirroring the archive layout."""
    directory = tmp_path / name / "raw"
    directory.mkdir(parents=True)
    for index, values in enumerate(presets):
        (directory / f"{index:032x}_60_100.npz").write_bytes(preset_bytes(values))
    return str(tmp_path / name)


def write_zip(tmp_path, presets, name="diva_raw.zip") -> str:
    zip_path = str(tmp_path / name)
    with zipfile.ZipFile(zip_path, "w") as archive:
        for index, values in enumerate(presets):
            archive.writestr(f"raw/{index:032x}_60_100.npz", preset_bytes(values))
    return zip_path


def write_zip_with_names(tmp_path, presets, param_names, name="diva_raw.zip") -> str:
    zip_path = str(tmp_path / name)
    with zipfile.ZipFile(zip_path, "w") as archive:
        for index, values in enumerate(presets):
            archive.writestr(
                f"raw/{index:032x}_60_100.npz", preset_bytes(values, param_names)
            )
    return zip_path


def voice(frequency: float = 0.0, volume: float = 0.0, model: float = 0.0):
    return [frequency, volume, model]


def test_maps_corpus_names_onto_framework_names(tmp_path):
    space = make_space()
    path = write_zip(tmp_path, [voice(frequency=0.5, volume=0.25)])
    split = DivaPresetLoader(space, test_fraction=0.0).load(path)
    assert len(split.train) == 1
    params = split.train[0].params
    # 'VCF1: Frequency' in the file, 'VCF1.Frequency' everywhere above the wrapper (D-NAMING).
    assert params["VCF1.Frequency"] == pytest.approx(0.5)
    assert params["OSC.Volume1"] == pytest.approx(0.25)
    assert set(params) == {"VCF1.Frequency", "OSC.Volume1", "VCF1.Model"}


def test_reads_an_extracted_directory_and_a_zip_identically(tmp_path):
    space = make_space()
    presets = [voice(frequency=index / 10) for index in range(1, 5)]
    from_zip = DivaPresetLoader(space).load(write_zip(tmp_path, presets))
    from_directory = DivaPresetLoader(space).load(write_directory(tmp_path, presets))
    assert [p.params for p in from_zip.train] == [p.params for p in from_directory.train]
    assert [p.voice_name for p in from_zip.train] == [p.voice_name for p in from_directory.train]


def test_extra_corpus_params_are_carried_but_projected_out_later(tmp_path):
    # The real corpus names all 281 parameters; only the subset is estimated. Extra names
    # must load fine (HumanPresetSource projects them out downstream).
    space = make_space()
    path = write_zip_with_names(
        tmp_path, [[0.9, 0.5, 0.25, 0.0]], ["ARP: Sync"] + FIXTURE_PARAM_NAMES
    )
    split = DivaPresetLoader(space).load(path)
    params = split.train[0].params
    assert params["ARP.Sync"] == pytest.approx(0.9)          # carried
    assert params["VCF1.Frequency"] == pytest.approx(0.5)    # still mapped by name


def test_raises_when_a_subset_name_is_absent(tmp_path):
    space = make_space()
    path = write_zip_with_names(
        tmp_path, [[0.5, 0.25]], ["VCF1: Frequency", "OSC: Volume1"]  # no VCF1: Model
    )
    with pytest.raises(RuntimeError, match="Subset parameter names not present"):
        DivaPresetLoader(space).load(path)


def test_raises_on_an_empty_collection(tmp_path):
    space = make_space()
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RuntimeError, match="No .npz presets found"):
        DivaPresetLoader(space).load(str(empty))


def test_raises_on_a_path_that_is_neither_zip_nor_directory(tmp_path):
    space = make_space()
    stray = tmp_path / "notes.txt"
    stray.write_text("not a corpus")
    with pytest.raises(FileNotFoundError, match="neither a zip archive nor a directory"):
        DivaPresetLoader(space).load(str(stray))


def test_dedup_collapses_identical_presets(tmp_path):
    space = make_space()
    path = write_zip(tmp_path, [voice(frequency=0.4) for _ in range(10)])
    split = DivaPresetLoader(space).load(path)
    assert len(split.train) == 1


def test_distinct_filter_models_are_not_deduplicated(tmp_path):
    space = make_space()
    path = write_zip(tmp_path, [voice(frequency=0.4, model=n / 4) for n in range(5)])
    split = DivaPresetLoader(space).load(path)
    assert len(split.train) == 5  # one-hot VCF1.Model blocks differ


def test_limit_caps_raw_presets(tmp_path):
    space = make_space()
    path = write_zip(tmp_path, [voice(frequency=index / 20) for index in range(1, 21)])
    split = DivaPresetLoader(space).load(path, limit=5)
    assert len(split.train) == 5


def test_split_is_disjoint_deterministic_and_correctly_sized(tmp_path):
    space = make_space()
    path = write_zip(tmp_path, [voice(frequency=(index + 1) / 20) for index in range(8)])

    split = DivaPresetLoader(space, test_fraction=0.25, split_seed=123).load(path)
    assert len(split.test) == 2  # round(8 * 0.25)
    assert len(split.train) == 6
    assert {p.voice_index for p in split.train}.isdisjoint({p.voice_index for p in split.test})

    again = DivaPresetLoader(space, test_fraction=0.25, split_seed=123).load(path)
    assert [p.voice_index for p in again.test] == [p.voice_index for p in split.test]


def test_provenance_records_the_collection_and_the_preset_hash(tmp_path):
    space = make_space()
    path = write_zip(tmp_path, [voice(frequency=0.5)])
    preset = DivaPresetLoader(space).load(path).train[0]
    assert preset.source_file == "diva_raw.zip"   # the collection, not the member
    assert preset.voice_index == 0
    # One preset per file, so the file's own stem is the only name the corpus carries.
    assert preset.voice_name == f"{0:032x}_60_100"


def test_presets_are_read_in_a_stable_sorted_order(tmp_path):
    space = make_space()
    presets = [voice(frequency=(index + 1) / 20) for index in range(6)]
    first = DivaPresetLoader(space).load(write_zip(tmp_path, presets))
    second = DivaPresetLoader(space).load(write_directory(tmp_path, presets))
    assert [p.voice_name for p in first.train] == sorted(p.voice_name for p in first.train)
    assert [p.voice_name for p in second.train] == [p.voice_name for p in first.train]


# ---------------------------------------------------------------------------
# The corpus-variance rule (D-DIVA-SUBSET): a parameter frozen across every preset
# is free to "predict" and tells the benchmark nothing.
# ---------------------------------------------------------------------------
def test_constant_parameters_reports_frozen_subset_params(tmp_path):
    space = make_space()
    # VCF1.Frequency varies; OSC.Volume1 and VCF1.Model are frozen.
    path = write_zip(tmp_path, [voice(frequency=index / 10) for index in range(1, 5)])
    split = DivaPresetLoader(space).load(path)
    assert constant_parameters(split.train, space) == ["OSC.Volume1", "VCF1.Model"]


def test_constant_parameters_is_empty_when_everything_varies(tmp_path):
    space = make_space()
    path = write_zip(tmp_path, [
        voice(frequency=index / 10, volume=index / 8, model=index / 4) for index in range(1, 5)
    ])
    split = DivaPresetLoader(space).load(path)
    assert constant_parameters(split.train, space) == []


def test_restrict_to_realized_drops_frozen_params_and_reports_their_values(tmp_path):
    space = make_space()
    # VCF1.Frequency varies; OSC.Volume1 frozen at 0.3 and VCF1.Model frozen at option 2.
    path = write_zip(tmp_path, [
        voice(frequency=index / 10, volume=0.3, model=0.5) for index in range(1, 5)
    ])
    split = DivaPresetLoader(space).load(path)

    narrowed, frozen = restrict_to_realized(split.train, space)
    assert narrowed.names == ["VCF1.Frequency"]
    assert frozen == {"OSC.Volume1": pytest.approx(0.3), "VCF1.Model": pytest.approx(0.5)}


def test_restrict_to_realized_narrows_a_surviving_categorical_to_its_used_options(tmp_path):
    space = make_space()
    # VCF1.Model only ever takes options 0 and 2 of its 5.
    path = write_zip(tmp_path, [
        voice(frequency=index / 10, model=0.5 * (index % 2)) for index in range(1, 5)
    ])
    split = DivaPresetLoader(space).load(path)

    narrowed, frozen = restrict_to_realized(split.train, space)
    assert "OSC.Volume1" in frozen                      # frozen, so dropped
    model = next(s for s in narrowed.parameter_specs if s.name == "VCF1.Model")
    assert model.options == [0.0, 0.5]                  # 5 options -> the 2 realized
    assert model.default in model.options
    assert narrowed.ml_dimension == 1 + 2               # frequency + narrowed one-hot


def test_restrict_to_realized_is_a_no_op_when_everything_varies(tmp_path):
    space = make_space()
    path = write_zip(tmp_path, [
        voice(frequency=index / 10, volume=index / 8, model=index / 4) for index in range(1, 5)
    ])
    split = DivaPresetLoader(space).load(path)
    narrowed, frozen = restrict_to_realized(split.train, space)
    assert frozen == {}
    assert narrowed.names == space.names


# ---------------------------------------------------------------------------
# Corpus-gated: the real diva_raw archive against the real 237-parameter subset.
# Skips when the 1.4 GB download is absent.
# ---------------------------------------------------------------------------
needs_corpus = pytest.mark.skipif(
    not os.path.exists(config.DIVA_RAW_PATH),
    reason=f"Diva preset corpus not found at {config.DIVA_RAW_PATH}",
)


@needs_corpus
def test_real_corpus_covers_the_diva_subset_by_name():
    from synth.diva.subset import SUBSET_PARAM_NAMES

    # A ParameterSpace over the real subset names, built without a live plugin: only the
    # names matter to the loader's name-based adapter.
    space = ParameterSpace([
        ParameterSpecification(name=name, kind="continuous", default=0.0)
        for name in SUBSET_PARAM_NAMES
    ])
    split = DivaPresetLoader(space).load(config.DIVA_RAW_PATH, limit=8)
    assert len(split.train) == 8
    for preset in split.train:
        assert len(preset.params) == 281               # all of Diva's synthesis parameters
        assert set(SUBSET_PARAM_NAMES) <= set(preset.params)
        assert all(0.0 <= value <= 1.0 for value in preset.params.values())
