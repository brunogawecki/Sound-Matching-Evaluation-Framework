"""Sound-matching framework -- local control panel (Streamlit entrypoint).

Run with::

    streamlit run dashboard/app.py

A private accelerator over the terminal scripts: build datasets, fit the baseline,
evaluate, and browse the benchmark table -- each page just builds and runs the
matching ``scripts/*.py`` command and reads the output files back. Nothing here
imports the pipeline library, so it can never drift from the CLI.
"""
from env import bootstrap

bootstrap()

import os

import streamlit as st

import config
import discovery

st.set_page_config(page_title="Sound-matching control panel", page_icon="🎛️", layout="wide")

st.title("🎛️ Sound-matching control panel")
st.caption(
    "A local front-end over the pipeline scripts. Pick a page in the sidebar: "
    "**Build dataset → Fit model → Evaluate → Results**."
)

dexed_ok = os.path.exists(os.path.expanduser(config.DEXED_PATH))
corpora = discovery.list_corpora()
checkpoints = discovery.list_checkpoints()
results = discovery.list_result_runs()

st.subheader("Environment")
left, right = st.columns(2)
with left:
    if dexed_ok:
        st.success(f"Dexed VST found — `{config.DEXED_PATH}`")
    else:
        st.warning(
            f"Dexed VST **not** found at `{config.DEXED_PATH}`. "
            "Dataset builds and evaluation need it (set `DEXED_PATH` in `.env`)."
        )
    st.write(f"Sample rate: **{config.SAMPLE_RATE} Hz** · render **{config.DURATION_SEC}s** "
             f"(note {config.NOTE_DURATION_SEC}s) · MIDI note **{config.MIDI_NOTE}**, vel **{config.VELOCITY}**")
with right:
    st.metric("Corpora on disk", len(corpora))
    st.metric("Checkpoints", len(checkpoints))
    st.metric("Result runs", len(results))

st.subheader("Corpora")
if corpora:
    st.dataframe(
        [
            {"name": c.name, "samples": c.num_samples, "eval-ready (fresh-process)": c.fresh_process}
            for c in corpora
        ],
        width="stretch",
        hide_index=True,
    )
else:
    st.info("No corpora yet. Head to **Build dataset** to create one.")

st.caption(
    "The D1 parameter subset (103 params) is locked and defines the benchmark axis; "
    "the pages expose the per-run knobs, not the subset itself."
)
