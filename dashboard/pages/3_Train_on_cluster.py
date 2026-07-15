"""Push a corpus and submit a training job on the PUT SLURM cluster."""
from env import bootstrap

bootstrap()

import streamlit as st

import cluster_runner
import command_runner
import discovery
from script_specs import MODEL_CHOICES

_TERMINAL_STATES = ("COMPLETED",)
_RUNNING_STATES = ("PENDING", "RUNNING", "UNKNOWN")


def _cached_state_key(job_id: str) -> str:
    return f"job_state_{job_id}"


def _render_terminal_job(job: cluster_runner.Job, state: str) -> None:
    """Static (non-polling) view for a job that has left the running states."""
    if state in _TERMINAL_STATES:
        st.success(f"State: **{state}**")
        with_ckpt = st.checkbox(
            "Also pull the raw Lightning .ckpt files (~900 MB)",
            key=f"with_ckpt_{job.job_id}",
            help="Only needed to resume training, which happens on the cluster. The "
                 "exported .pt already holds the best epoch's weights.",
        )
        if st.button("Pull checkpoint", key=f"pull_{job.job_id}"):
            placeholder = st.empty()
            with st.spinner(f"Pulling job {job.job_id}…"):
                code = cluster_runner.pull_checkpoint(
                    job.job_id, job.model, placeholder, with_ckpt=with_ckpt
                )
            if code == 0:
                st.success(
                    f"Pulled job {job.job_id} ({job.model}). See the Evaluate page."
                )
            else:
                st.error(f"cluster/pull_checkpoint.sh exited {code}; see log above.")
    else:
        st.error(f"State: **{state}**")
        _render_log_tail(job)


def _render_log_tail(job: cluster_runner.Job) -> None:
    """Show the job's remote log tail, or a note when there's no log to read."""
    log_text = cluster_runner.get_remote_log_tail(job.job_id)
    if log_text is None:
        st.caption("No log — the job produced no output (cancelled before it ran?).")
    else:
        st.code(command_runner.collapse_carriage_returns(log_text) or "(no output)")


@st.fragment(run_every="5s")
def _live_job_fragment(job: cluster_runner.Job) -> None:
    """Poll state + tail the log every 5s; offer Cancel while running.

    Sole owner of the state query, so ``_render_job_status`` reads the cached
    terminal state instead of re-querying. On leaving a running state, caches it
    and reruns so the static terminal view takes over and polling stops.
    """
    try:
        state = cluster_runner.get_slurm_job_state(job.job_id)
    except (RuntimeError, KeyError, FileNotFoundError) as exc:
        st.error(f"Could not check status: {exc}")
        return
    if state not in _RUNNING_STATES:
        st.session_state[_cached_state_key(job.job_id)] = state
        st.rerun()
    st.caption(f"State: **{state}**")
    # No log exists until the job leaves the queue, so don't tail while pending.
    if state == "RUNNING":
        _render_log_tail(job)
    else:
        st.caption("Queued — the log tail appears once the job starts running.")
    if st.button("Cancel job", key=f"cancel_{job.job_id}"):
        try:
            cluster_runner.cancel_job(job.job_id)
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            st.success(f"Cancel requested for job {job.job_id}.")


def _render_job_status(job: cluster_runner.Job) -> None:
    # Serve a cached terminal state statically (no SSH); else poll via the fragment.
    cached_state = st.session_state.get(_cached_state_key(job.job_id))
    if cached_state is not None and cached_state not in _RUNNING_STATES:
        _render_terminal_job(job, cached_state)
    else:
        _live_job_fragment(job)


st.set_page_config(page_title="Train on cluster", layout="wide")
st.title("Train on cluster")
st.caption(
    "Push a corpus to the cluster once, then submit training jobs against it. Submitting syncs "
    "the remote checkout (pull, or switch to your branch) and runs `cluster/train.sbatch` over "
    "SSH. Model/corpus choice happens here; training itself runs on the cluster."
)

try:
    cluster_env = cluster_runner.load_cluster_env()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

missing = [
    key
    for key in ("CLUSTER_SSH", "SLURM_ACCOUNT", "REMOTE_REPO_DIR", "REMOTE_CORPORA_DIR")
    if not cluster_env.get(key)
]
if missing:
    st.error(f"cluster/cluster.env is missing a value for: {', '.join(missing)}.")
    st.stop()

clean, guard_messages = cluster_runner.git_guard_status()
if not clean:
    st.warning(
        "The cluster only ever trains on what's pushed to GitHub (`git pull` runs before "
        "submit). Local state isn't fully synced:\n\n" + "\n\n".join(guard_messages)
    )

corpora = discovery.list_corpora()
if not corpora:
    st.info("No corpora yet. Build one on the **Build dataset** page first.")
    st.stop()

corpus = st.selectbox(
    "Training corpus",
    corpora,
    format_func=lambda c: f"{c.name}  ({c.num_samples} samples)",
)

if st.button("Push corpus"):
    placeholder = st.empty()
    with st.spinner(f"Pushing {corpus.name}…"):
        try:
            cluster_runner.push_corpus(corpus.name, placeholder)
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            st.success(f"Pushed {corpus.name} to the cluster.")

model_name = st.selectbox("Model", list(MODEL_CHOICES))

config_choice = st.selectbox(
    "Training config", (*discovery.list_training_config_names(), "custom")
)
if config_choice == "custom":
    config_arg = st.text_input(
        "Config path (relative to repo root)",
        value="",
        help="A literal cluster/training_configs/<name>.yaml path.",
    )
else:
    config_arg = config_choice

st.caption(
    f"Target: `{cluster_env['CLUSTER_SSH']}:{cluster_env['REMOTE_REPO_DIR']}` "
    f"· account `{cluster_env['SLURM_ACCOUNT']}`"
)

# Show the cluster's branch vs local; let the user hard-sync to theirs or leave it.
local_branch = cluster_runner.get_local_branch()
try:
    remote_branch = cluster_runner.get_remote_branch()
except (RuntimeError, KeyError) as exc:
    remote_branch = None
    st.caption(f"Could not read the cluster's branch (leaving it as-is): {exc}")

checkout_branch = None  # None => leave the cluster on its current branch
if remote_branch is not None:
    if remote_branch == local_branch:
        st.caption(f"Cluster is on `{remote_branch}` (matches local); it will `git pull` before submit.")
    else:
        st.caption(f"Cluster is on `{remote_branch}`, you're on `{local_branch}`.")
        switch_label = f"Switch cluster to `{local_branch}`"
        stay_label = f"Stay on `{remote_branch}`"
        choice = st.radio(
            "Sync the cluster before submitting:",
            (switch_label, stay_label),
            index=0,
            help="Switching hard-syncs the cluster to your pushed branch (unpushed work won't be included).",
        )
        if choice == switch_label:
            checkout_branch = local_branch

# Show the exact commands Submit will send over SSH (visual only — nothing runs here).
if config_arg:
    st.caption("Commands sent to the cluster on Train (visual only, not run here):")
    for command in cluster_runner.preview_submit_commands(
        corpus.name, model_name, config_arg, checkout_branch=checkout_branch
    ):
        st.code(command, language="bash")

if st.button("Train", type="primary", disabled=(config_choice == "custom" and not config_arg)):
    placeholder = st.empty()
    with st.spinner("Submitting…"):
        try:
            job = cluster_runner.submit_job(
                corpus.name, model_name, config_arg, placeholder, checkout_branch=checkout_branch
            )
        except (RuntimeError, KeyError) as exc:
            st.error(str(exc))
        else:
            st.success(f"Submitted job {job.job_id} ({job.corpus} / {job.model} / {job.config}).")

st.subheader("Submitted jobs")
jobs = cluster_runner.load_jobs()
if jobs:
    for job in reversed(jobs):
        with st.expander(
            f"Job {job.job_id} — {job.corpus} / {job.model} / {job.config} "
            f"(submitted {job.submitted_at})"
        ):
            _render_job_status(job)
else:
    st.caption("No jobs submitted yet from this dashboard.")
