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
from typing import Dict, List, Tuple

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


def submit_job(corpus_name: str, model_name: str, config_arg: str, placeholder) -> Job:
    """Push the corpus, sync the remote checkout, and ``sbatch`` a training job.

    ``placeholder`` is an ``st.empty()`` (or anything with ``.code(str)``) that
    the corpus push streams into live; the quick git-pull/sbatch steps just
    write their final output there. Raises ``RuntimeError`` on any failing step.
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

    pull_code, pull_output = command_runner.run_capture(
        ["ssh", ssh_target, f"cd {shlex.quote(remote_repo_dir)} && git pull"]
    )
    placeholder.code(pull_output or "(no output)")
    if pull_code != 0:
        raise RuntimeError(f"remote git pull exited {pull_code}:\n{pull_output}")

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
