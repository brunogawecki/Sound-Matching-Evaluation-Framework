"""Drive the PUT SLURM cluster from the dashboard (D-DASHBOARD-CLUSTER).

Shells out to ``ssh`` and the existing ``cluster/*.sh`` scripts, mirroring
``command_runner.py``'s subprocess pattern -- no new SSH dependency. Named
``cluster_runner`` (not ``cluster``) so it doesn't collide with the top-level
``cluster/`` directory, which sits on ``sys.path`` via ``env.bootstrap()``.
"""
import json
import re
import shlex
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import dotenv

import command_runner
from env import PROJECT_ROOT

CLUSTER_ENV_PATH = PROJECT_ROOT / "cluster" / "cluster.env"
JOBS_REGISTRY_PATH = PROJECT_ROOT / "cluster" / "jobs.json"


def load_cluster_env() -> Dict[str, str]:
    """Read ``cluster/cluster.env`` (the laptop+cluster shared settings file)."""
    if not CLUSTER_ENV_PATH.exists():
        raise FileNotFoundError(
            f"{CLUSTER_ENV_PATH} not found. Copy cluster/cluster.env.example to "
            "cluster/cluster.env and fill in your SSH target, SLURM account, and remote paths."
        )
    return dotenv.dotenv_values(CLUSTER_ENV_PATH)


def git_guard_status() -> Tuple[bool, List[str]]:
    """Warn (never block) if local git has uncommitted or unpushed work.

    The cluster only ever sees what's pushed to GitHub via ``git pull``, so
    anything sitting uncommitted or unpushed locally won't be picked up by a
    submitted job.
    """
    messages: List[str] = []

    _, status_output = command_runner.run_capture(
        ["git", "status", "--porcelain"], cwd=str(PROJECT_ROOT)
    )
    if status_output.strip():
        messages.append("Uncommitted local changes:\n" + status_output.strip())

    ahead_code, ahead_output = command_runner.run_capture(
        ["git", "log", "@{u}..", "--oneline"], cwd=str(PROJECT_ROOT)
    )
    if ahead_code != 0:
        messages.append("Could not check for unpushed commits (no upstream configured?).")
    elif ahead_output.strip():
        messages.append("Commits not pushed to the remote:\n" + ahead_output.strip())

    return not messages, messages


def get_local_branch() -> str:
    """The branch the laptop's checkout is on."""
    _, output = command_runner.run_capture(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(PROJECT_ROOT)
    )
    return output.strip()


def get_remote_branch() -> str:
    """The cluster checkout's current branch. Fails fast and raises when unreachable."""
    cluster_env = load_cluster_env()
    ssh_target = cluster_env["CLUSTER_SSH"]
    remote_repo_dir = cluster_env["REMOTE_REPO_DIR"]
    code, output = command_runner.run_capture(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", ssh_target,
         f"cd {shlex.quote(remote_repo_dir)} && git rev-parse --abbrev-ref HEAD"]
    )
    if code != 0:
        raise RuntimeError(f"Could not read the cluster's branch:\n{output.strip()}")
    return output.strip()


@dataclass(frozen=True)
class Job:
    job_id: str
    corpus: str
    model: str
    config: str
    submitted_at: str  # ISO 8601, UTC


def load_jobs() -> List[Job]:
    """Every job the dashboard has submitted, oldest first (``[]`` if none yet)."""
    if not JOBS_REGISTRY_PATH.exists():
        return []
    try:
        raw = json.loads(JOBS_REGISTRY_PATH.read_text())
    except (OSError, ValueError):
        return []
    return [Job(**entry) for entry in raw]


def append_job(job: Job) -> None:
    jobs = load_jobs()
    jobs.append(job)
    JOBS_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    JOBS_REGISTRY_PATH.write_text(json.dumps([asdict(j) for j in jobs], indent=2))


def submit_job(
    corpus_name: str,
    model_name: str,
    config_arg: str,
    placeholder,
    checkout_branch: Optional[str] = None,
) -> Job:
    """Push the corpus, sync the remote checkout, and ``sbatch`` a training job.

    ``placeholder`` (an ``st.empty()``) is streamed the corpus push live. With
    ``checkout_branch`` set, hard-syncs the cluster to that pushed branch
    (``git fetch`` + ``checkout -B``); otherwise leaves it and ``git pull``s.
    Raises ``RuntimeError`` on any failing step.
    """
    cluster_env = load_cluster_env()
    ssh_target = cluster_env["CLUSTER_SSH"]
    slurm_account = cluster_env["SLURM_ACCOUNT"]
    remote_repo_dir = cluster_env["REMOTE_REPO_DIR"]

    push_code = command_runner.run_streaming(
        ["cluster/push_corpus.sh", corpus_name], placeholder
    )
    if push_code != 0:
        raise RuntimeError(f"cluster/push_corpus.sh exited {push_code}; see log above.")

    if checkout_branch:
        remote_ref = shlex.quote(f"origin/{checkout_branch}")
        sync_command = (
            f"cd {shlex.quote(remote_repo_dir)} && git fetch origin && "
            f"git checkout -B {shlex.quote(checkout_branch)} {remote_ref}"
        )
    else:
        sync_command = f"cd {shlex.quote(remote_repo_dir)} && git pull"
    sync_code, sync_output = command_runner.run_capture(["ssh", ssh_target, sync_command])
    placeholder.code(sync_output or "(no output)")
    if sync_code != 0:
        raise RuntimeError(f"remote git sync exited {sync_code}:\n{sync_output}")

    sbatch_command = (
        f"cd {shlex.quote(remote_repo_dir)} && "
        f"sbatch -A {shlex.quote(slurm_account)} cluster/train.sbatch "
        f"{shlex.quote(corpus_name)} {shlex.quote(model_name)} {shlex.quote(config_arg)}"
    )
    sbatch_code, sbatch_output = command_runner.run_capture(["ssh", ssh_target, sbatch_command])
    placeholder.code(sbatch_output or "(no output)")
    if sbatch_code != 0:
        raise RuntimeError(f"sbatch exited {sbatch_code}:\n{sbatch_output}")

    match = re.search(r"Submitted batch job (\d+)", sbatch_output)
    if not match:
        raise RuntimeError(f"Could not parse a job id from sbatch output:\n{sbatch_output}")

    job = Job(
        job_id=match.group(1),
        corpus=corpus_name,
        model=model_name,
        config=config_arg,
        submitted_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    append_job(job)
    return job


def get_slurm_job_state(job_id: str) -> str:
    """The job's current SLURM state (``PENDING``/``RUNNING``/``COMPLETED``/...).

    Returns ``"UNKNOWN"`` if ``sacct`` fails or hasn't indexed the job yet.
    """
    ssh_target = load_cluster_env()["CLUSTER_SSH"]
    code, output = command_runner.run_capture(
        ["ssh", ssh_target, f"sacct -j {shlex.quote(job_id)} --format=State --noheader -X"]
    )
    if code != 0 or not output.strip():
        return "UNKNOWN"
    return output.strip().splitlines()[0].strip()


def get_remote_log_tail(job_id: str, lines: int = 40) -> str:
    """The last ``lines`` of ``slurm-<job_id>.out`` on the cluster, raw (uncollapsed)."""
    cluster_env = load_cluster_env()
    ssh_target = cluster_env["CLUSTER_SSH"]
    remote_repo_dir = cluster_env["REMOTE_REPO_DIR"]
    log_path = f"{remote_repo_dir}/slurm-{job_id}.out"
    code, output = command_runner.run_capture(
        ["ssh", ssh_target, f"tail -n {int(lines)} {shlex.quote(log_path)}"]
    )
    if code != 0:
        return f"(could not read {log_path}: {output.strip()})"
    return output


def cancel_job(job_id: str) -> None:
    """``scancel`` a job. Raises ``RuntimeError`` on failure."""
    ssh_target = load_cluster_env()["CLUSTER_SSH"]
    code, output = command_runner.run_capture(["ssh", ssh_target, f"scancel {shlex.quote(job_id)}"])
    if code != 0:
        raise RuntimeError(f"scancel exited {code}:\n{output}")


def pull_checkpoint(model_name: str, placeholder) -> int:
    """Run ``cluster/pull_checkpoint.sh <model_name>``, streaming into ``placeholder``."""
    return command_runner.run_streaming(["cluster/pull_checkpoint.sh", model_name], placeholder)
