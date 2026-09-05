"""Tests for out-of-domain (audio-only) corpora and the evaluation-corpus loader.

Covers the two halves of the OOD path that do not need a VST: the dataset class /
dispatcher (``dataset/ood_corpus.py``) and the corpus builder's conversion +
summary-copying logic (``scripts/build_ood_corpus.py``). The builder's NSynth-reading
half is exercised against a synthetic stand-in for the dataset, so the suite stays
runnable without the 4.6 GB download.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.io import wavfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.ood_corpus import NO_TARGETS, AudioOnlyCorpusDataset, load_eval_corpus
from synth.parameter_space import ParameterSpace, ParameterSpecification

SAMPLE_RATE = 22050
DURATION_SEC = 4.0
NUM_SAMPLES = int(DURATION_SEC * SAMPLE_RATE)  # 88200 at the D3 contract


def _load_builder_module():
    """Import scripts/build_ood_corpus.py, which is a script rather than a package."""
    script_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "build_ood_corpus.py"
    )
    spec = importlib.util.spec_from_file_location("build_ood_corpus", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _small_space() -> ParameterSpace:
    return ParameterSpace([
        ParameterSpecification(name="AMP", kind="continuous", default=0.5),
        ParameterSpecification(name="CAT", kind="categorical", options=[0.0, 1.0], default=0.0),
    ])


def _noise(num_samples: int = NUM_SAMPLES, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (0.2 * rng.standard_normal(num_samples)).astype(np.float32)


def _write_reference_corpus(corpus_dir) -> dict:
    """A minimal in-domain corpus: the thing an OOD corpus copies its contract from."""
    space = _small_space()
    (corpus_dir / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for index in range(2):
        sample_id = f"sample_{index:06d}"
        relative_path = f"audio/{sample_id}.wav"
        wavfile.write(str(corpus_dir / relative_path), SAMPLE_RATE, _noise(seed=index))
        rows.append({
            "sample_id": sample_id, "audio_path": relative_path,
            "AMP": 0.4, "CAT": 0.0, "loudness_lufs": -20.0,
        })
    pd.DataFrame(rows).to_csv(corpus_dir / "metadata.csv", index=False)

    summary = {
        "run_name": corpus_dir.name,
        "num_samples": len(rows),
        "render_settings": {
            "midi_note": 60, "velocity": 100,
            "duration_sec": DURATION_SEC, "note_duration_sec": 3.0,
        },
        "renderer": "dawdreamer",
        "sample_rate": SAMPLE_RATE,
        "synth": "dexed",
        "parameter_space": space.to_dict(),
        "default_params": {"AMP": 0.5, "CAT": 0.0},
        "subset_names": space.names,
        "git_revision": "cafebabe",
    }
    with open(corpus_dir / "run_summary.json", "w") as summary_file:
        json.dump(summary, summary_file)
    return summary


def _write_fake_nsynth(nsynth_dir, notes) -> None:
    """A stand-in NSynth tree: examples.json + 16 kHz / 64000-sample int16 WAVs."""
    builder = _load_builder_module()
    by_split = {}
    for note in notes:
        by_split.setdefault(note["nsynth_split"], []).append(note)
    for split, split_notes in by_split.items():
        split_dir = nsynth_dir / f"nsynth-{split}"
        (split_dir / "audio").mkdir(parents=True, exist_ok=True)
        examples = {}
        for index, note in enumerate(split_notes):
            record = {key: value for key, value in note.items() if key != "nsynth_split"}
            examples[record["note_str"]] = record
            rng = np.random.default_rng(index + 1)
            waveform = rng.standard_normal(builder.NSYNTH_NUM_SAMPLES) * 0.1
            wavfile.write(
                str(split_dir / "audio" / f"{record['note_str']}.wav"),
                builder.NSYNTH_SAMPLE_RATE,
                (waveform * 32767).astype(np.int16),
            )
        with open(split_dir / "examples.json", "w") as examples_file:
            json.dump(examples, examples_file)


def _note(note_str, pitch, velocity, split, family="flute", source="acoustic") -> dict:
    return {
        "note_str": note_str, "pitch": pitch, "velocity": velocity,
        "instrument_family_str": family, "instrument_source_str": source,
        "nsynth_split": split,
    }


@pytest.fixture
def built_ood_corpus(tmp_path):
    """A reference corpus plus an OOD corpus built from a fake NSynth tree."""
    builder = _load_builder_module()
    reference_dir = tmp_path / "reference_test"
    reference_summary = _write_reference_corpus(reference_dir)

    nsynth_dir = tmp_path / "nsynth"
    _write_fake_nsynth(nsynth_dir, [
        _note("flute_acoustic_001-060-100", 60, 100, "test"),
        _note("guitar_acoustic_002-060-100", 60, 100, "valid", family="guitar"),
        _note("brass_acoustic_003-060-025", 60, 25, "valid", family="brass"),   # wrong velocity
        _note("organ_electronic_004-072-100", 72, 100, "test", family="organ"), # wrong pitch
    ])

    run_dir = tmp_path / "nsynth_c4_dexed"
    summary = builder.build(nsynth_dir, reference_dir, run_dir, ["test", "valid"], 60, [100])
    return run_dir, summary, reference_summary


# -- the pitch/velocity filter ------------------------------------------------

def test_filter_keeps_only_the_requested_pitch_and_velocity(built_ood_corpus):
    run_dir, summary, _ = built_ood_corpus
    metadata = pd.read_csv(run_dir / "metadata.csv")

    assert summary["num_samples"] == 2
    assert set(metadata["nsynth_note_str"]) == {
        "flute_acoustic_001-060-100", "guitar_acoustic_002-060-100",
    }
    assert set(metadata["pitch"]) == {60}
    assert set(metadata["velocity"]) == {100}
    assert set(metadata["nsynth_split"]) == {"test", "valid"}


def test_metadata_carries_no_parameter_columns(built_ood_corpus):
    run_dir, _, reference_summary = built_ood_corpus
    metadata = pd.read_csv(run_dir / "metadata.csv")
    space = ParameterSpace.from_dict(reference_summary["parameter_space"])
    assert not set(space.names) & set(metadata.columns)


def test_selection_is_deterministic(tmp_path):
    """Same arguments, same corpus: notes are sorted, so row order never drifts."""
    builder = _load_builder_module()
    nsynth_dir = tmp_path / "nsynth"
    _write_fake_nsynth(nsynth_dir, [
        _note("zither_acoustic_009-060-100", 60, 100, "valid"),
        _note("bass_acoustic_001-060-100", 60, 100, "test"),
        _note("mallet_acoustic_005-060-100", 60, 100, "valid"),
    ])
    selected = builder.select_nsynth_notes(nsynth_dir, ["test", "valid"], 60, [100])
    assert [note["note_str"] for note in selected] == [
        "bass_acoustic_001-060-100",
        "mallet_acoustic_005-060-100",
        "zither_acoustic_009-060-100",
    ]


# -- resampling ---------------------------------------------------------------

def test_resample_lands_on_the_render_contract_length(tmp_path):
    """16 kHz / 64000 samples maps to exactly 88200 at 22.05 kHz -- no padding."""
    builder = _load_builder_module()
    audio_path = tmp_path / "note.wav"
    rng = np.random.default_rng(0)
    wavfile.write(
        str(audio_path), builder.NSYNTH_SAMPLE_RATE,
        (rng.standard_normal(builder.NSYNTH_NUM_SAMPLES) * 0.1 * 32767).astype(np.int16),
    )
    resampled = builder.load_and_resample(audio_path, SAMPLE_RATE)
    assert resampled.shape == (NUM_SAMPLES,)
    assert resampled.dtype == np.float32
    assert np.isfinite(resampled).all()


def test_resample_rejects_audio_that_is_not_nsynth_shaped(tmp_path):
    builder = _load_builder_module()
    audio_path = tmp_path / "short.wav"
    wavfile.write(str(audio_path), builder.NSYNTH_SAMPLE_RATE, np.zeros(1000, dtype=np.int16))
    with pytest.raises(ValueError, match="expected"):
        builder.load_and_resample(audio_path, SAMPLE_RATE)


# -- the copied render contract ----------------------------------------------

def test_contract_is_copied_verbatim_from_the_reference_corpus(built_ood_corpus):
    _, summary, reference_summary = built_ood_corpus
    for field in ("render_settings", "renderer", "sample_rate", "default_params",
                  "synth", "parameter_space", "subset_names"):
        assert summary[field] == reference_summary[field], field


def test_summary_marks_the_corpus_as_out_of_domain(built_ood_corpus):
    _, summary, _ = built_ood_corpus
    assert summary["targets"] == NO_TARGETS
    assert summary["domain"] == "out_of_domain"
    # Provenance chain: reproducible from its own summary (D-SELFDESC).
    assert summary["source"]["dataset"] == "nsynth"
    assert summary["source"]["reference_corpus"] == "reference_test"
    assert summary["source"]["reference_corpus_git_revision"] == "cafebabe"
    assert summary["source"]["pitch"] == 60
    assert summary["source"]["velocities"] == [100]


def test_build_fails_when_the_filter_matches_nothing(tmp_path):
    builder = _load_builder_module()
    reference_dir = tmp_path / "reference_test"
    _write_reference_corpus(reference_dir)
    nsynth_dir = tmp_path / "nsynth"
    _write_fake_nsynth(nsynth_dir, [_note("flute_acoustic_001-072-100", 72, 100, "test")])
    with pytest.raises(SystemExit):
        builder.build(nsynth_dir, reference_dir, tmp_path / "empty", ["test"], 60, [100])


# -- the dataset class + dispatcher ------------------------------------------

def test_load_eval_corpus_returns_an_audio_only_dataset(built_ood_corpus):
    run_dir, _, _ = built_ood_corpus
    corpus = load_eval_corpus(run_dir)
    assert isinstance(corpus, AudioOnlyCorpusDataset)
    assert corpus.targets is None
    assert len(corpus) == 2


def test_load_eval_corpus_still_returns_a_rendered_dataset_in_domain(tmp_path):
    from dataset.torch_dataset import RenderedCorpusDataset

    reference_dir = tmp_path / "reference_test"
    _write_reference_corpus(reference_dir)
    assert isinstance(load_eval_corpus(reference_dir), RenderedCorpusDataset)


def test_getitem_returns_audio_and_no_target(built_ood_corpus):
    run_dir, _, _ = built_ood_corpus
    audio, target = load_eval_corpus(run_dir)[0]
    assert target is None
    assert isinstance(audio, torch.Tensor)
    assert audio.shape == (NUM_SAMPLES,)
    assert audio.dtype == torch.float32


def test_parameter_space_comes_from_the_reference_corpus(built_ood_corpus):
    run_dir, _, reference_summary = built_ood_corpus
    corpus = load_eval_corpus(run_dir)
    assert corpus.parameter_space.names == ParameterSpace.from_dict(
        reference_summary["parameter_space"]
    ).names


def test_truncated_audio_is_still_an_error(built_ood_corpus):
    """The length check survives: an OOD corpus is built to the contract's exact length."""
    run_dir, _, _ = built_ood_corpus
    corpus = load_eval_corpus(run_dir)
    wavfile.write(str(run_dir / "audio" / "sample_000000.wav"), SAMPLE_RATE, _noise(100))
    with pytest.raises(ValueError, match="corrupt"):
        corpus[0]


# -- filename fallback (stream-extracted splits carry no examples.json) -------

def test_record_from_note_str_recovers_every_recorded_field():
    builder = _load_builder_module()
    record = builder.record_from_note_str("flute_acoustic_002-060-100")
    assert record == {
        "note_str": "flute_acoustic_002-060-100",
        "instrument_str": "flute_acoustic_002",
        "instrument_family_str": "flute",
        "instrument_source_str": "acoustic",
        "pitch": 60,
        "velocity": 100,
    }
    # Families whose name contains no underscore trap a naive split; source is second.
    assert builder.record_from_note_str("keyboard_electronic_001-060-100")["instrument_source_str"] == "electronic"


def test_split_without_examples_json_falls_back_to_filenames(tmp_path):
    """A split holding only matching WAVs still yields the right notes and metadata."""
    builder = _load_builder_module()
    audio_dir = tmp_path / "nsynth-train" / "audio"
    audio_dir.mkdir(parents=True)
    for note_str in ("guitar_acoustic_010-060-100", "organ_electronic_020-060-100"):
        rng = np.random.default_rng(0)
        wavfile.write(
            str(audio_dir / f"{note_str}.wav"), builder.NSYNTH_SAMPLE_RATE,
            (rng.standard_normal(builder.NSYNTH_NUM_SAMPLES) * 0.1 * 32767).astype(np.int16),
        )
    selected = builder.select_nsynth_notes(tmp_path, ["train"], 60, [100])
    assert [note["note_str"] for note in selected] == [
        "guitar_acoustic_010-060-100", "organ_electronic_020-060-100",
    ]
    assert selected[0]["instrument_family_str"] == "guitar"
    assert all(note["nsynth_split"] == "train" for note in selected)


def test_missing_split_is_a_clear_error(tmp_path):
    builder = _load_builder_module()
    with pytest.raises(SystemExit, match="No NSynth train split"):
        builder.select_nsynth_notes(tmp_path, ["train"], 60, [100])


def test_summary_records_the_measured_loudness_offset(built_ood_corpus):
    """The D-METRIC-NORM caveat is stored, not just printed, so it survives the build."""
    _, summary, _ = built_ood_corpus
    source = summary["source"]
    assert source["median_loudness_lufs"] is not None
    assert source["reference_median_loudness_lufs"] == pytest.approx(-20.0)
    assert np.isfinite(source["median_loudness_lufs"])
