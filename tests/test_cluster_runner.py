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


# --- get_local_branch / get_remote_branch ------------------------------------

def test_get_local_branch_strips_output(monkeypatch):
    monkeypatch.setattr(command_runner, "run_capture", lambda argv, cwd=None: (0, "handle-pipeline\n"))
    assert cluster_runner.get_local_branch() == "handle-pipeline"


def test_get_remote_branch_returns_branch(monkeypatch):
    _stub_cluster_env(monkeypatch)
    monkeypatch.setattr(command_runner, "run_capture", lambda argv: (0, "main\n"))
    assert cluster_runner.get_remote_branch() == "main"


def test_get_remote_branch_raises_when_unreachable(monkeypatch):
    _stub_cluster_env(monkeypatch)
    monkeypatch.setattr(command_runner, "run_capture", lambda argv: (255, "ssh: connect timeout"))
    with pytest.raises(RuntimeError):
        cluster_runner.get_remote_branch()


# --- command builders / preview ----------------------------------------------

def test_build_sync_command_defaults_to_git_pull():
    command = cluster_runner.build_sync_command("/home/me/repo")
    assert command == "cd /home/me/repo && git pull"


def test_build_sync_command_switch_branch_hard_checks_out():
    command = cluster_runner.build_sync_command("/home/me/repo", checkout_branch="feature-x")
    assert "git fetch origin" in command
    assert "git checkout -B feature-x origin/feature-x" in command
    assert "git pull" not in command


def test_build_sbatch_command_shape():
    command = cluster_runner.build_sbatch_command(
        "/home/me/repo", "acct123", "corpus_a", "MeanParameterBaseline", "smoke"
    )
    assert command == (
        "cd /home/me/repo && sbatch -A acct123 cluster/train.sbatch "
        "corpus_a MeanParameterBaseline smoke"
    )


def test_preview_submit_commands_match_what_submit_runs(monkeypatch):
    _stub_cluster_env(monkeypatch)
    sync, sbatch = cluster_runner.preview_submit_commands(
        "corpus_a", "MeanParameterBaseline", "smoke"
    )
    assert sync == "ssh me@login.example 'cd /home/me/repo && git pull'"
    assert sbatch.startswith("ssh me@login.example 'cd /home/me/repo && sbatch -A acct123")
    assert "cluster/train.sbatch corpus_a MeanParameterBaseline smoke" in sbatch


def test_preview_submit_commands_reflects_branch_switch(monkeypatch):
    _stub_cluster_env(monkeypatch)
    sync, _ = cluster_runner.preview_submit_commands(
        "corpus_a", "MeanParameterBaseline", "smoke", checkout_branch="feature-x"
    )
    assert "git checkout -B feature-x origin/feature-x" in sync


# --- submit_job ------------------------------------------------------------

def _stub_cluster_env(monkeypatch):
    monkeypatch.setattr(
        cluster_runner, "load_cluster_env",
        lambda: {"CLUSTER_SSH": "me@login.example", "SLURM_ACCOUNT": "acct123",
                 "REMOTE_REPO_DIR": "/home/me/repo", "REMOTE_CORPORA_DIR": "/home/me/corpora"},
    )


# --- push_corpus / remote_corpus_exists --------------------------------------

def test_push_corpus_streams_script_and_succeeds(monkeypatch):
    calls = []
    monkeypatch.setattr(
        command_runner, "run_streaming",
        lambda argv, placeholder: calls.append(argv) or 0,
    )
    cluster_runner.push_corpus("corpus_a", _FakePlaceholder())
    assert calls == [["cluster/push_corpus.sh", "corpus_a"]]


def test_push_corpus_failure_raises(monkeypatch):
    monkeypatch.setattr(command_runner, "run_streaming", lambda argv, placeholder: 1)
    with pytest.raises(RuntimeError):
        cluster_runner.push_corpus("corpus_a", _FakePlaceholder())


def test_remote_corpus_exists_true_when_test_d_succeeds(monkeypatch):
    _stub_cluster_env(monkeypatch)
    calls = []
    monkeypatch.setattr(command_runner, "run_capture", lambda argv: calls.append(argv) or (0, ""))
    assert cluster_runner.remote_corpus_exists("corpus_a") is True
    assert "test -d /home/me/corpora/corpus_a" in calls[0][-1]


def test_remote_corpus_exists_false_when_test_d_fails(monkeypatch):
    _stub_cluster_env(monkeypatch)
    monkeypatch.setattr(command_runner, "run_capture", lambda argv: (1, ""))
    assert cluster_runner.remote_corpus_exists("corpus_a") is False


# --- submit_job ------------------------------------------------------------

def test_submit_job_success_appends_and_returns_job(tmp_path, monkeypatch):
    _stub_cluster_env(monkeypatch)
    monkeypatch.setattr(cluster_runner, "JOBS_REGISTRY_PATH", tmp_path / "jobs.json")

    capture_calls = []

    def fake_run_capture(argv):
        capture_calls.append(argv)
        if "test -d" in argv[-1]:
            return 0, ""
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
    assert len(capture_calls) == 3  # guard (test -d), git sync, sbatch


def test_submit_job_missing_corpus_raises_and_does_not_register(tmp_path, monkeypatch):
    _stub_cluster_env(monkeypatch)
    jobs_path = tmp_path / "jobs.json"
    monkeypatch.setattr(cluster_runner, "JOBS_REGISTRY_PATH", jobs_path)

    def fake_run_capture(argv):
        if "test -d" in argv[-1]:
            return 1, ""  # corpus not on the cluster
        raise AssertionError("should not sync or sbatch when the corpus is missing")

    monkeypatch.setattr(command_runner, "run_capture", fake_run_capture)

    with pytest.raises(RuntimeError):
        cluster_runner.submit_job("corpus_a", "MeanParameterBaseline", "smoke", _FakePlaceholder())
    assert not jobs_path.exists()


def test_submit_job_unparseable_sbatch_output_raises(tmp_path, monkeypatch):
    _stub_cluster_env(monkeypatch)
    monkeypatch.setattr(cluster_runner, "JOBS_REGISTRY_PATH", tmp_path / "jobs.json")

    def fake_run_capture(argv):
        if "test -d" in argv[-1]:
            return 0, ""
        if "git pull" in argv[-1]:
            return 0, "Already up to date.\n"
        return 0, "no job id in here\n"

    monkeypatch.setattr(command_runner, "run_capture", fake_run_capture)

    with pytest.raises(RuntimeError):
        cluster_runner.submit_job("corpus_a", "MeanParameterBaseline", "smoke", _FakePlaceholder())


def test_submit_job_switch_branch_hard_checks_out_on_cluster(tmp_path, monkeypatch):
    _stub_cluster_env(monkeypatch)
    monkeypatch.setattr(cluster_runner, "JOBS_REGISTRY_PATH", tmp_path / "jobs.json")

    capture_calls = []

    def fake_run_capture(argv):
        capture_calls.append(argv)
        remote_command = argv[-1]
        if "test -d" in remote_command:
            return 0, ""
        if "git checkout -B" in remote_command:
            return 0, "Reset branch 'feature-x'\n"
        if "sbatch" in remote_command:
            return 0, "Submitted batch job 55555\n"
        raise AssertionError(f"unexpected argv: {argv}")

    monkeypatch.setattr(command_runner, "run_capture", fake_run_capture)

    job = cluster_runner.submit_job(
        "corpus_a", "MeanParameterBaseline", "smoke", _FakePlaceholder(), checkout_branch="feature-x"
    )

    assert job.job_id == "55555"
    sync_command = capture_calls[1][-1]  # capture_calls[0] is the guard (test -d)
    assert "git fetch origin" in sync_command
    assert "git checkout -B feature-x origin/feature-x" in sync_command
    assert "git pull" not in sync_command


# --- get_slurm_job_state / get_remote_log_tail / cancel_job / pull_checkpoint --------------

def test_get_slurm_job_state_returns_stripped_state(monkeypatch):
    _stub_cluster_env(monkeypatch)
    monkeypatch.setattr(command_runner, "run_capture", lambda argv: (0, "RUNNING     \n"))
    assert cluster_runner.get_slurm_job_state("98765") == "RUNNING"


def test_get_slurm_job_state_unknown_on_failure(monkeypatch):
    _stub_cluster_env(monkeypatch)
    monkeypatch.setattr(command_runner, "run_capture", lambda argv: (1, ""))
    assert cluster_runner.get_slurm_job_state("98765") == "UNKNOWN"


def test_get_slurm_job_state_unknown_on_blank_output(monkeypatch):
    _stub_cluster_env(monkeypatch)
    monkeypatch.setattr(command_runner, "run_capture", lambda argv: (0, "   \n"))
    assert cluster_runner.get_slurm_job_state("98765") == "UNKNOWN"


def test_get_remote_log_tail_returns_output(monkeypatch):
    _stub_cluster_env(monkeypatch)
    monkeypatch.setattr(command_runner, "run_capture", lambda argv: (0, "epoch 1/10\nepoch 2/10\n"))
    assert cluster_runner.get_remote_log_tail("98765") == "epoch 1/10\nepoch 2/10\n"


def test_get_remote_log_tail_none_when_unreadable(monkeypatch):
    _stub_cluster_env(monkeypatch)
    monkeypatch.setattr(command_runner, "run_capture", lambda argv: (1, "No such file"))
    assert cluster_runner.get_remote_log_tail("98765") is None


def test_cancel_job_success(monkeypatch):
    _stub_cluster_env(monkeypatch)
    calls = []
    monkeypatch.setattr(command_runner, "run_capture", lambda argv: calls.append(argv) or (0, ""))
    cluster_runner.cancel_job("98765")
    assert any("scancel 98765" in argv[-1] for argv in calls)


def test_cancel_job_failure_raises(monkeypatch):
    _stub_cluster_env(monkeypatch)
    monkeypatch.setattr(command_runner, "run_capture", lambda argv: (1, "Invalid job id"))
    with pytest.raises(RuntimeError):
        cluster_runner.cancel_job("98765")


def test_build_pull_command_is_job_scoped():
    assert cluster_runner.build_pull_command("982169", "PresetGenVAEFlowRegressor") == [
        "cluster/pull_checkpoint.sh", "982169", "PresetGenVAEFlowRegressor"
    ]


def test_build_pull_command_with_ckpt_appends_the_flag():
    assert cluster_runner.build_pull_command(
        "982169", "PresetGenVAEFlowRegressor", with_ckpt=True
    ) == ["cluster/pull_checkpoint.sh", "982169", "PresetGenVAEFlowRegressor", "--with-ckpt"]


def test_pull_checkpoint_delegates_to_run_streaming(monkeypatch):
    calls = []
    monkeypatch.setattr(
        command_runner, "run_streaming",
        lambda argv, placeholder: calls.append(argv) or 0,
    )
    placeholder = _FakePlaceholder()
    code = cluster_runner.pull_checkpoint("111", "MeanParameterBaseline", placeholder)
    assert code == 0
    assert calls == [["cluster/pull_checkpoint.sh", "111", "MeanParameterBaseline"]]
