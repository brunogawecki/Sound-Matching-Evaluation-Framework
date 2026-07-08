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


@st.fragment(run_every="5s")
def _live_job_fragment(job: cluster_runner.Job) -> None:
    """Polls state + tails the remote log every 5s; offers Cancel while running."""
    state = cluster_runner.get_slurm_job_state(job.job_id)
    st.caption(f"State: **{state}**")
    log_text = cluster_runner.get_remote_log_tail(job.job_id)
    st.code(command_runner.collapse_carriage_returns(log_text) or "(no output)")
    if state in _RUNNING_STATES:
        if st.button("Cancel job", key=f"cancel_{job.job_id}"):
            try:
                cluster_runner.cancel_job(job.job_id)
            except RuntimeError as exc:
                st.error(str(exc))
            else:
                st.success(f"Cancel requested for job {job.job_id}.")


def _render_job_status(job: cluster_runner.Job) -> None:
    try:
        state = cluster_runner.get_slurm_job_state(job.job_id)
    except (RuntimeError, KeyError) as exc:
        st.error(f"Could not check status: {exc}")
        return

    if state in _RUNNING_STATES:
        _live_job_fragment(job)
    elif state in _TERMINAL_STATES:
        st.success(f"State: **{state}**")
        if st.button("Pull checkpoint", key=f"pull_{job.job_id}"):
            placeholder = st.empty()
            with st.spinner("Pulling checkpoint…"):
                code = cluster_runner.pull_checkpoint(job.model, placeholder)
            if code == 0:
                st.success(f"Pulled checkpoint for {job.model}. See the Evaluate page.")
            else:
                st.error(f"cluster/pull_checkpoint.sh exited {code}; see log above.")
    else:
        st.error(f"State: **{state}**")
        log_text = cluster_runner.get_remote_log_tail(job.job_id)
        st.code(command_runner.collapse_carriage_returns(log_text) or "(no output)")


st.set_page_config(page_title="Train on cluster", layout="wide")
st.title("Train on cluster")
st.caption(
    "Pushes the selected corpus, syncs the remote checkout (`git pull`), and submits "
    "`cluster/train.sbatch` over SSH. Model/corpus choice happens here; training itself "
    "runs on the cluster."
)

try:
    cluster_env = cluster_runner.load_cluster_env()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

missing = [key for key in ("CLUSTER_SSH", "SLURM_ACCOUNT", "REMOTE_REPO_DIR") if not cluster_env.get(key)]
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
model_name = st.selectbox("Model", list(MODEL_CHOICES))

config_choice = st.selectbox("Training config", ("full", "smoke", "custom"))
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

if st.button("Push corpus & submit job", type="primary", disabled=(config_choice == "custom" and not config_arg)):
    placeholder = st.empty()
    with st.spinner("Pushing corpus and submitting…"):
        try:
            job = cluster_runner.submit_job(corpus.name, model_name, config_arg, placeholder)
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
