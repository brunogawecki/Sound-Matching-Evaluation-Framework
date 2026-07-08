"""Tests for the post-render corpus split (D-SPLIT).

The copy/split/summary logic is VST-free and covered directly. A plugin-gated
integration test at the bottom builds a tiny corpus and splits it end to end.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from dataset.corpus_splitter import (
    assert_splittable,
    clean_records,
    derive_summary,
    load_corpus,
    source_method,
    split_source_description,
    write_copied_partition,
)
from dataset.dexed_preset_loader import split_indices
from dataset.preset_sources import CorpusPresetSource
from synth.parameter_space import ParameterSpace, ParameterSpecification

SUBSET_NAMES = ["P1", "P2"]


def _parameter_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="P1", kind="continuous"),
        ParameterSpecification(name="P2", kind="continuous"),
    ])


def _make_fake_corpus(corpus_dir, count: int, method: str = "human") -> None:
    """Write a minimal but valid corpus: audio/*.wav + metadata.csv + run_summary.json."""
    audio_dir = corpus_dir / "audio"
    audio_dir.mkdir(parents=True)
    rows = []
    for index in range(count):
        sample_id = f"sample_{index:06d}"
        (audio_dir / f"{sample_id}.wav").write_bytes(f"wav-{index}".encode())
        rows.append({
            "sample_id": sample_id,
            "audio_path": f"audio/{sample_id}.wav",
            "P1": 0.1 * index,
            "P2": 0.9 - 0.1 * index,
            "method": method,
            "partition": "train",
            "source_file": f"cart_{index}.syx",
            "voice_index": index,
            "voice_name": f"voice {index}",
            "parent_id": None,
            "rms": 0.05,
            "loudness_lufs": -20.0,
            "near_silent": index == 0,  # one near-silent to exercise the count
        })
    columns = ["sample_id", "audio_path", *SUBSET_NAMES, "method", "partition",
               "source_file", "voice_index", "voice_name", "parent_id",
               "rms", "loudness_lufs", "near_silent"]
    pd.DataFrame(rows, columns=columns).to_csv(corpus_dir / "metadata.csv", index=False)
    summary = {
        "run_name": corpus_dir.name,
        "num_samples": count,
        "near_silent_count": 1,
        "method_counts": {method: count},
        "render_settings": {"midi_note": 60, "velocity": 100,
                            "duration_sec": 3.0, "note_duration_sec": 2.0},
        "render_process": "in-process",
        "sample_rate": 22050,
        "renderer": "dawdreamer",
        "subset_names": list(SUBSET_NAMES),
        "parameter_space": _parameter_space().to_dict(),
        "default_params": {"P1": 0.0, "P2": 0.0, "DROPPED": 0.5},
        "min_loudness_lufs": -34.0,
        "max_redraw_attempts": 10,
        "source": {"method": method, "count": count, "partition": "train"},
        "git_revision": "deadbeef",
    }
    (corpus_dir / "run_summary.json").write_text(json.dumps(summary))


# --- split_indices ----------------------------------------------------------

def test_split_indices_disjoint_and_complete():
    train, test = split_indices(10, 0.3, split_seed=0)
    assert len(test) == 3 and len(train) == 7
    assert set(train).isdisjoint(test)
    assert sorted(train + test) == list(range(10))
    assert train == sorted(train) and test == sorted(test)


def test_split_indices_is_deterministic():
    assert split_indices(50, 0.2, 7) == split_indices(50, 0.2, 7)
    assert split_indices(50, 0.2, 7) != split_indices(50, 0.2, 8)


# --- method guard -----------------------------------------------------------

def test_source_method_reads_summary():
    assert source_method({"source": {"method": "hybrid"}}) == "hybrid"
    assert source_method({}) is None


def test_assert_splittable_rejects_hybrid():
    with pytest.raises(ValueError, match="leakage"):
        assert_splittable({"source": {"method": "hybrid"}})


@pytest.mark.parametrize("method", ["synthetic", "human", None])
def test_assert_splittable_allows_non_hybrid(method):
    assert_splittable({"source": {"method": method}})  # does not raise


# --- provenance + record cleaning -------------------------------------------

def test_split_source_description_shape():
    summary = {"source": {"method": "human"}}
    desc = split_source_description(summary, "test", "corpus_a", 0.25, 3, count=5)
    assert desc == {
        "method": "human", "partition": "test", "split_from": "corpus_a",
        "split_test_fraction": 0.25, "split_seed": 3, "count": 5,
    }


def test_clean_records_coerces_nan_to_none():
    df = pd.DataFrame([{"P1": 0.5, "parent_id": np.nan, "voice_index": 4}])
    record = clean_records(df)[0]
    assert record["parent_id"] is None
    assert record["P1"] == 0.5


# --- copied (train) partition -----------------------------------------------

def test_write_copied_partition_copies_reindexes_and_summarizes(tmp_path):
    source_dir = tmp_path / "src"
    _make_fake_corpus(source_dir, count=4, method="human")
    summary, df = load_corpus(source_dir)

    train_positions, _ = split_indices(len(df), 0.5, 0)
    df_train = df.iloc[train_positions]
    out_dir = tmp_path / "src_train"
    desc = split_source_description(summary, "train", source_dir.name, 0.5, 0, len(df_train))

    written = write_copied_partition(source_dir, df_train, out_dir, summary, desc, "train")

    # Audio copied and re-indexed contiguously.
    copied = sorted(p.name for p in (out_dir / "audio").glob("*.wav"))
    assert copied == [f"sample_{i:06d}.wav" for i in range(len(df_train))]

    df_out = pd.read_csv(out_dir / "metadata.csv")
    assert list(df_out["sample_id"]) == [f"sample_{i:06d}" for i in range(len(df_train))]
    assert set(df_out["partition"]) == {"train"}
    # Subset params preserved (order-invariant): the copied P1 set matches the source slice.
    assert sorted(df_out["P1"]) == sorted(df_train["P1"])
    # Original columns preserved.
    assert "voice_name" in df_out.columns

    # Summary is derived and self-describing.
    assert written["num_samples"] == len(df_train)
    assert written["source"] == desc
    assert written["render_process"] == "in-process"  # copied audio keeps source process
    assert written["parameter_space"] == summary["parameter_space"]
    assert written["default_params"] == summary["default_params"]
    on_disk = json.loads((out_dir / "run_summary.json").read_text())
    assert on_disk["source"]["split_from"] == source_dir.name


def test_write_copied_partition_refuses_existing_dir(tmp_path):
    source_dir = tmp_path / "src"
    _make_fake_corpus(source_dir, count=2)
    summary, df = load_corpus(source_dir)
    out_dir = tmp_path / "out"
    (out_dir / "audio").mkdir(parents=True)  # an existing corpus dir must not be clobbered
    desc = split_source_description(summary, "train", source_dir.name, 0.5, 0, 1)
    with pytest.raises(FileExistsError):
        write_copied_partition(source_dir, df.iloc[[0]], out_dir, summary, desc, "train")


def test_derive_summary_recomputes_counts(tmp_path):
    source_dir = tmp_path / "src"
    _make_fake_corpus(source_dir, count=4, method="human")
    summary, df = load_corpus(source_dir)
    desc = split_source_description(summary, "test", source_dir.name, 0.5, 0, 2)
    derived = derive_summary(summary, "src_test", df.iloc[[0, 1]], "fresh", desc)
    assert derived["num_samples"] == 2
    assert derived["method_counts"] == {"human": 2}
    assert derived["near_silent_count"] == 1  # sample_000000 is near-silent
    assert derived["render_process"] == "fresh"


# --- CorpusPresetSource (replay from metadata; no VST) ----------------------

def test_corpus_preset_source_reconstructs_presets(tmp_path):
    source_dir = tmp_path / "src"
    _make_fake_corpus(source_dir, count=4, method="human")
    _, df = load_corpus(source_dir)
    df_test = df.iloc[[2, 3]]
    desc = {"method": "human", "partition": "test", "count": 2}

    source = CorpusPresetSource(clean_records(df_test), _parameter_space(), desc, partition="test")
    presets = list(source.iter_presets())

    assert len(presets) == 2
    assert source.describe() == desc
    first = presets[0]
    assert set(first.params) == set(SUBSET_NAMES)
    assert first.params["P1"] == pytest.approx(df_test.iloc[0]["P1"])
    assert first.partition == "test"
    assert first.method == "human"
    assert first.voice_index == int(df_test.iloc[0]["voice_index"])
    assert first.parent_id is None


# --- plugin-gated end-to-end split ------------------------------------------

PLUGIN_PATH = os.path.expanduser(config.DEXED_PATH)


@pytest.mark.skipif(not os.path.exists(PLUGIN_PATH), reason=f"Dexed plugin not found at {PLUGIN_PATH}")
def test_split_rendered_corpus_end_to_end(tmp_path):
    from synth.dexed import DexedWrapper
    from dataset.builder import DatasetBuilder
    from dataset.preset_sources import SyntheticPresetSource

    synth = DexedWrapper(plugin_path=PLUGIN_PATH, sample_rate=config.SAMPLE_RATE,
                         buffer_size=config.BUFFER_SIZE)
    # Build a tiny real corpus, then split it (in-process render keeps the test fast).
    source = SyntheticPresetSource(synth.parameter_space, count=4, seed=0,
                                   sampling_ranges=synth.audible_sampling_ranges)
    DatasetBuilder(synth).build(source, run_name="src", output_root=tmp_path)
    source_dir = tmp_path / "src"

    summary, df = load_corpus(source_dir)
    train_positions, test_positions = split_indices(len(df), 0.5, 0)
    df_train, df_test = df.iloc[train_positions], df.iloc[test_positions]

    train_desc = split_source_description(summary, "train", "src", 0.5, 0, len(df_train))
    write_copied_partition(source_dir, df_train, tmp_path / "src_train", summary, train_desc, "train")

    test_desc = split_source_description(summary, "test", "src", 0.5, 0, len(df_test))
    replay = CorpusPresetSource(clean_records(df_test), synth.parameter_space, test_desc)
    test_summary = DatasetBuilder(synth).build(replay, run_name="src_test", output_root=tmp_path)

    train_df = pd.read_csv(tmp_path / "src_train" / "metadata.csv")
    assert len(train_df) == len(df_train)
    assert test_summary["num_samples"] == len(df_test)
    assert test_summary["source"]["split_from"] == "src"
    assert len(list((tmp_path / "src_test" / "audio").glob("*.wav"))) == len(df_test)
