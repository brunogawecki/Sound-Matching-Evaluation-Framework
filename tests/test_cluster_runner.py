"""Tests for dashboard/cluster_runner.py's pure helpers.

Nothing here touches a real SSH target: ``command_runner.run_capture`` /
``run_streaming`` are monkeypatched with fakes, so these exercise the
plumbing (env parsing, job registry, git-guard messaging, submit_job's
orchestration and error paths) without a network call.
"""
import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "dashboard"))

import cluster_runner  # noqa: E402
import command_runner  # noqa: E402


class _FakePlaceholder:
    def __init__(self):
        self.calls = []

    def code(self, text):
        self.calls.append(text)


# --- load_cluster_env --------------------------------------------------------

def test_load_cluster_env_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(cluster_runner, "CLUSTER_ENV_PATH", tmp_path / "cluster.env")
    with pytest.raises(FileNotFoundError):
        cluster_runner.load_cluster_env()


def test_load_cluster_env_reads_values(tmp_path, monkeypatch):
    env_path = tmp_path / "cluster.env"
    env_path.write_text("CLUSTER_SSH=me@login.example\nSLURM_ACCOUNT=acct123\n")
    monkeypatch.setattr(cluster_runner, "CLUSTER_ENV_PATH", env_path)
    values = cluster_runner.load_cluster_env()
    assert values["CLUSTER_SSH"] == "me@login.example"
    assert values["SLURM_ACCOUNT"] == "acct123"


# --- job registry -------------------------------------------------------------

def test_load_jobs_empty_when_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(cluster_runner, "JOBS_REGISTRY_PATH", tmp_path / "jobs.json")
    assert cluster_runner.load_jobs() == []


def test_append_job_then_load_jobs_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(cluster_runner, "JOBS_REGISTRY_PATH", tmp_path / "jobs.json")
    first = cluster_runner.Job("111", "corpus_a", "MeanParameterBaseline", "smoke", "2026-01-01T00:00:00+00:00")
    second = cluster_runner.Job("222", "corpus_b", "Sound2SynthSpectrogramRegressor", "full", "2026-01-02T00:00:00+00:00")
    cluster_runner.append_job(first)
    cluster_runner.append_job(second)
    jobs = cluster_runner.load_jobs()
    assert jobs == [first, second]


# --- git_guard_status ----------------------------------------------------------

def test_git_guard_status_clean(monkeypatch):
    def fake_run_capture(argv, cwd=None):
        if argv[:2] == ["git", "status"]:
            return 0, ""
        if argv[:2] == ["git", "log"]:
            return 0, ""
        raise AssertionError(f"unexpected argv: {argv}")

    monkeypatch.setattr(command_runner, "run_capture", fake_run_capture)
    clean, messages = cluster_runner.git_guard_status()
    assert clean is True
    assert messages == []


def test_git_guard_status_dirty_and_unpushed(monkeypatch):
    def fake_run_capture(argv, cwd=None):
        if argv[:2] == ["git", "status"]:
            return 0, " M evaluator.py\n"
        if argv[:2] == ["git", "log"]:
            return 0, "abc1234 wip\n"
        raise AssertionError(f"unexpected argv: {argv}")

    monkeypatch.setattr(command_runner, "run_capture", fake_run_capture)
    clean, messages = cluster_runner.git_guard_status()
    assert clean is False
    assert any("Uncommitted" in message for message in messages)
    assert any("not pushed" in message for message in messages)


def test_git_guard_status_no_upstream_warns_not_blocks(monkeypatch):
    def fake_run_capture(argv, cwd=None):
        if argv[:2] == ["git", "status"]:
            return 0, ""
        if argv[:2] == ["git", "log"]:
            return 128, "fatal: no upstream configured\n"
        raise AssertionError(f"unexpected argv: {argv}")

    monkeypatch.setattr(command_runner, "run_capture", fake_run_capture)
    clean, messages = cluster_runner.git_guard_status()
    assert clean is False
    assert any("Could not check" in message for message in messages)


# --- submit_job ------------------------------------------------------------

def _stub_cluster_env(monkeypatch):
    monkeypatch.setattr(
        cluster_runner, "load_cluster_env",
        lambda: {"CLUSTER_SSH": "me@login.example", "SLURM_ACCOUNT": "acct123",
                 "REMOTE_REPO_DIR": "/home/me/repo"},
    )


def test_submit_job_success_appends_and_returns_job(tmp_path, monkeypatch):
    _stub_cluster_env(monkeypatch)
    monkeypatch.setattr(cluster_runner, "JOBS_REGISTRY_PATH", tmp_path / "jobs.json")
    monkeypatch.setattr(command_runner, "run_streaming", lambda argv, placeholder: 0)

    capture_calls = []

    def fake_run_capture(argv):
        capture_calls.append(argv)
        if "git pull" in argv[-1]:
            return 0, "Already up to date.\n"
        if "sbatch" in argv[-1]:
            return 0, "Submitted batch job 98765\n"
        raise AssertionError(f"unexpected argv: {argv}")

    monkeypatch.setattr(command_runner, "run_capture", fake_run_capture)

    placeholder = _FakePlaceholder()
    job = cluster_runner.submit_job("corpus_a", "MeanParameterBaseline", "smoke", placeholder)

    assert job.job_id == "98765"
    assert job.corpus == "corpus_a"
    assert job.model == "MeanParameterBaseline"
    assert job.config == "smoke"
    assert cluster_runner.load_jobs() == [job]
    assert len(capture_calls) == 2


def test_submit_job_push_failure_raises_and_does_not_register(tmp_path, monkeypatch):
    _stub_cluster_env(monkeypatch)
    jobs_path = tmp_path / "jobs.json"
    monkeypatch.setattr(cluster_runner, "JOBS_REGISTRY_PATH", jobs_path)
    monkeypatch.setattr(command_runner, "run_streaming", lambda argv, placeholder: 1)
    monkeypatch.setattr(
        command_runner, "run_capture",
        lambda argv: (_ for _ in ()).throw(AssertionError("should not reach ssh"))
    )

    with pytest.raises(RuntimeError):
        cluster_runner.submit_job("corpus_a", "MeanParameterBaseline", "smoke", _FakePlaceholder())
    assert not jobs_path.exists()


def test_submit_job_unparseable_sbatch_output_raises(tmp_path, monkeypatch):
    _stub_cluster_env(monkeypatch)
    monkeypatch.setattr(cluster_runner, "JOBS_REGISTRY_PATH", tmp_path / "jobs.json")
    monkeypatch.setattr(command_runner, "run_streaming", lambda argv, placeholder: 0)

    def fake_run_capture(argv):
        if "git pull" in argv[-1]:
            return 0, "Already up to date.\n"
        return 0, "no job id in here\n"

    monkeypatch.setattr(command_runner, "run_capture", fake_run_capture)

    with pytest.raises(RuntimeError):
        cluster_runner.submit_job("corpus_a", "MeanParameterBaseline", "smoke", _FakePlaceholder())
