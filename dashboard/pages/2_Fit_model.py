"""Fit a model on a training corpus (baseline today; seam ready for deep models)."""
from env import bootstrap

bootstrap()

import streamlit as st

import discovery
import ui
from forms import build_command
from script_specs import FIT_MODEL, MODEL_CHOICES

st.set_page_config(page_title="Fit model", layout="wide")
st.title("Fit model")

corpora = discovery.list_corpora()
if not corpora:
    st.info("No corpora yet. Build one on the **Build dataset** page first.")
    st.stop()

st.caption(FIT_MODEL.description)

model_name = st.selectbox("Model", list(MODEL_CHOICES))
corpus = st.selectbox(
    "Training corpus",
    corpora,
    format_func=lambda c: f"{c.name}  ({c.num_samples} samples)",
)
out = st.text_input(
    "Checkpoint output path (optional)",
    value="",
    help="Blank uses the script default (checkpoints/<model default filename>).",
)

try:
    argv = build_command(
        FIT_MODEL, {"model": model_name, "corpus": str(corpus.path), "out": out}
    )
except ValueError as exc:
    argv = None
    st.info(f"Fill required field(s): {exc}")

if argv:
    ui.command_preview(argv)
    code = ui.run_button(argv, key="run_fit")
    if code == 0:
        st.subheader("Checkpoints on disk")
        st.write([str(path.name) for path in discovery.list_checkpoints()] or "—")
        st.caption("Head to **Evaluate** to score this checkpoint on a fresh-process corpus.")
