"""Evaluate a checkpoint on a fresh-process corpus through the metric panel."""
from env import bootstrap

bootstrap()

import os
from pathlib import Path

import streamlit as st

import config
import discovery
import ui
from forms import build_command
from script_specs import EVALUATE

st.set_page_config(page_title="Evaluate", layout="wide")
st.title("Evaluate")

if not os.path.exists(os.path.expanduser(config.DEXED_PATH)):
    st.warning(
        f"Dexed VST not found at `{config.DEXED_PATH}`. Evaluation re-renders each "
        "prediction (D-REPRO) and needs the VST. Set `DEXED_PATH` in `.env`."
    )

checkpoints = discovery.list_checkpoints()
corpora = discovery.list_corpora()
if not checkpoints:
    st.info("No checkpoints yet. Fit a model on the **Fit model** page first.")
    st.stop()
if not corpora:
    st.info("No corpora yet. Build a (fresh-process) eval corpus on **Build dataset** first.")
    st.stop()

checkpoint = st.selectbox("Checkpoint", checkpoints, format_func=lambda p: p.name)

# Fresh-process corpora are the correct eval targets (D-REPRO); surface them first
# but don't hard-block the others (matches the CLI, which only warns).
fresh = [c for c in corpora if c.fresh_process]
others = [c for c in corpora if not c.fresh_process]
ordered = fresh + others
corpus = st.selectbox(
    "Eval corpus",
    ordered,
    format_func=lambda c: f"{c.name}  ({c.num_samples} samples)"
    + ("" if c.fresh_process else "  ⚠ in-process — not D-REPRO"),
)
if not corpus.fresh_process:
    st.warning("This corpus rendered in-process. Eval expects a fresh-process corpus (D-REPRO).")

model_choice = next(arg for arg in EVALUATE.args if arg.name == "model")
model = st.selectbox("Model class", list(model_choice.choices))
out = st.text_input("Results root (optional)", value="", help="Blank uses <project>/results.")

try:
    argv = build_command(
        EVALUATE,
        {"checkpoint": str(checkpoint), "corpus": str(corpus.path), "model": model, "out": out},
    )
except ValueError as exc:
    argv = None
    st.info(f"Fill required field(s): {exc}")

if argv:
    ui.command_preview(argv)
    code = ui.run_button(argv, key="run_eval")
    if code == 0:
        st.caption("See the **Results** page for the benchmark table and per-sample drill-down.")
