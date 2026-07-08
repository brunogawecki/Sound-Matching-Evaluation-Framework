"""Push a corpus and submit a training job on the PUT SLURM cluster."""
from env import bootstrap

bootstrap()

import streamlit as st

import cluster_runner
import discovery
from script_specs import MODEL_CHOICES

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
    st.dataframe(
        [
            {"job_id": j.job_id, "corpus": j.corpus, "model": j.model,
             "config": j.config, "submitted_at": j.submitted_at}
            for j in reversed(jobs)
        ],
        width="stretch",
        hide_index=True,
    )
    st.caption("Live status, log tailing, checkpoint pull, and cancel land in the next step.")
else:
    st.caption("No jobs submitted yet from this dashboard.")
