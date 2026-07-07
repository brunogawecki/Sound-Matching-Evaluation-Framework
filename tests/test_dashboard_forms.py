"""Tests for the dashboard's pure helpers (no Streamlit runtime needed).

Covers command building from the flag-specs, the subprocess runner, and artefact
discovery. The Streamlit pages themselves are exercised manually (they need a
running server); everything testable is factored out of them.
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "dashboard"))

import discovery  # noqa: E402
from command_runner import run_capture  # noqa: E402
from forms import build_command  # noqa: E402
from script_specs import (  # noqa: E402
    BUILD_HUMAN,
    BUILD_SYNTHETIC,
    EVALUATE,
    FIT_MODEL,
    MODEL_CHOICES,
)

from models.registry import MODEL_REGISTRY  # noqa: E402


def test_model_choices_match_registry():
    # script_specs.MODEL_CHOICES is a hand-kept mirror of MODEL_REGISTRY's keys (the
    # dashboard never imports the torch-heavy pipeline library -- see
    # dashboard/__init__.py). This test is the tripwire that catches drift between
    # the two: it's a plain test file, so it can import both without breaking that
    # boundary.
    assert set(MODEL_CHOICES) == set(MODEL_REGISTRY)


# --- build_command: argv is exactly what the CLI would take -----------------

def test_synthetic_command_is_exact():
    argv = build_command(
        BUILD_SYNTHETIC,
        {"count": 64, "seed": 7, "run_name": "run_A_train", "fresh_process": False},
    )
    assert argv == [
        sys.executable, "scripts/build_dataset.py", "synthetic",
        "--count", "64", "--seed", "7", "--run-name", "run_A_train",
    ]


def test_bool_flag_is_bare_when_true_absent_when_false():
    on = build_command(BUILD_SYNTHETIC, {"count": 8, "seed": 0,
                                         "run_name": "x", "fresh_process": True})
    assert "--fresh-process" in on
    assert on[on.index("--fresh-process") - 1] != "--fresh-process"  # no value follows
    off = build_command(BUILD_SYNTHETIC, {"count": 8, "seed": 0,
                                          "run_name": "x", "fresh_process": False})
    assert "--fresh-process" not in off


def test_paths_split_and_blank_choice_omitted():
    argv = build_command(BUILD_HUMAN, {"cartridges": "a.syx b.syx", "partition": ""})
    assert argv[:6] == [
        sys.executable, "scripts/build_dataset.py", "human",
        "--cartridges", "a.syx", "b.syx",
    ]
    assert "--partition" not in argv  # blank choice -> script default (both)
    # unspecified args fall back to their declared defaults
    assert "--test-fraction" in argv and "--dedup-threshold" in argv


def test_required_missing_raises():
    with pytest.raises(ValueError):
        build_command(BUILD_HUMAN, {"cartridges": ""})
    with pytest.raises(ValueError):
        build_command(EVALUATE, {"checkpoint": "", "corpus": "c", "model": "MeanParameterBaseline"})


def test_model_choice_required_raises_when_blank():
    with pytest.raises(ValueError):
        build_command(FIT_MODEL, {"model": "", "corpus": "dataset/run_A_train", "out": ""})
    with pytest.raises(ValueError):
        build_command(
            EVALUATE, {"checkpoint": "checkpoints/m.json", "corpus": "c", "model": ""}
        )


def test_optional_blank_out_is_omitted():
    argv = build_command(
        FIT_MODEL,
        {"model": "MeanParameterBaseline", "corpus": "dataset/run_A_train", "out": ""},
    )
    assert argv == [
        sys.executable, "scripts/fit_model.py",
        "--model", "MeanParameterBaseline",
        "--corpus", "dataset/run_A_train",
    ]


def test_fit_model_full_command():
    argv = build_command(
        FIT_MODEL,
        {"model": "Sound2SynthSpectrogramRegressor", "corpus": "dataset/run_A_train",
         "out": "checkpoints/spectrogram_cnn.pt", "config": "cluster/training_configs/smoke_config.yaml"},
    )
    assert argv == [
        sys.executable, "scripts/fit_model.py",
        "--model", "Sound2SynthSpectrogramRegressor",
        "--corpus", "dataset/run_A_train",
        "--out", "checkpoints/spectrogram_cnn.pt",
        "--config", "cluster/training_configs/smoke_config.yaml",
    ]


def test_evaluate_full_command():
    argv = build_command(
        EVALUATE,
        {"checkpoint": "checkpoints/m.json", "corpus": "dataset/run_A_test",
         "model": "MeanParameterBaseline", "out": ""},
    )
    assert argv == [
        sys.executable, "scripts/evaluate.py",
        "--checkpoint", "checkpoints/m.json",
        "--corpus", "dataset/run_A_test",
        "--model", "MeanParameterBaseline",
        "--save-audio-n", "20",
    ]


def test_evaluate_save_audio_flag_emitted_when_true():
    argv = build_command(
        EVALUATE,
        {"checkpoint": "checkpoints/m.json", "corpus": "dataset/run_A_test",
         "model": "MeanParameterBaseline", "out": "", "save_audio": True, "save_audio_n": 5},
    )
    assert argv == [
        sys.executable, "scripts/evaluate.py",
        "--checkpoint", "checkpoints/m.json",
        "--corpus", "dataset/run_A_test",
        "--model", "MeanParameterBaseline",
        "--save-audio",
        "--save-audio-n", "5",
    ]


# --- run_capture: returncode + combined output ------------------------------

def test_run_capture_success():
    code, output = run_capture([sys.executable, "-c", "print('hello-dash')"])
    assert code == 0
    assert "hello-dash" in output


def test_run_capture_nonzero():
    code, _ = run_capture([sys.executable, "-c", "import sys; sys.exit(3)"])
    assert code == 3


# --- discovery: reads the on-disk tree --------------------------------------

def test_discovery_functions_return_lists():
    assert isinstance(discovery.list_corpora(), list)
    assert isinstance(discovery.list_checkpoints(), list)
    assert isinstance(discovery.list_result_runs(), list)


def test_list_corpora_reads_run_summary(tmp_path, monkeypatch):
    corpus = tmp_path / "demo_train"
    (corpus / "audio").mkdir(parents=True)
    (corpus / "run_summary.json").write_text(
        '{"run_name": "demo_train", "num_samples": 5, "render_process": "fresh"}'
    )
    monkeypatch.setattr(discovery, "DATASET_DIR", tmp_path)
    corpora = discovery.list_corpora()
    assert len(corpora) == 1
    assert corpora[0].name == "demo_train"
    assert corpora[0].num_samples == 5
    assert corpora[0].fresh_process is True
