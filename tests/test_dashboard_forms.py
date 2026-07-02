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
    FIT_BASELINE,
)


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


def test_optional_blank_out_is_omitted():
    argv = build_command(FIT_BASELINE, {"corpus": "dataset/run_A_train", "out": ""})
    assert argv == [
        sys.executable, "scripts/fit_baseline.py",
        "--corpus", "dataset/run_A_train",
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
