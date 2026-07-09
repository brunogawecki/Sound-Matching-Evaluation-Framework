"""Split an already-built corpus into a train corpus and a test corpus.

Train audio is copied; the test partition is re-rendered fresh-process (D-REPRO),
so it needs the VST. Hybrid corpora are excluded (train/test leakage).
"""
from env import bootstrap

bootstrap()

import os

import streamlit as st

import config
import discovery
import ui
from forms import build_command
from script_specs import SPLIT_CORPUS

st.set_page_config(page_title="Split corpus", layout="wide")
st.title("Split corpus")
st.caption(SPLIT_CORPUS.description)

if not os.path.exists(os.path.expanduser(config.DEXED_PATH)):
    st.warning(
        f"Dexed VST not found at `{config.DEXED_PATH}`. The split re-renders the test "
        "partition fresh-process (D-REPRO) and needs the VST. Set `DEXED_PATH` in `.env`."
    )

corpora = discovery.list_corpora()
if not corpora:
    st.info("No corpora yet. Build one on the **Build dataset** page first.")
    st.stop()


def _label(corpus: discovery.Corpus) -> str:
    tag = "  ⚠ hybrid — not splittable (leakage)" if corpus.method == "hybrid" else ""
    return f"{corpus.name}  ({corpus.num_samples} samples){tag}"


corpus = st.selectbox("Source corpus", corpora, format_func=_label)

if corpus.method == "hybrid":
    st.warning(
        "Hybrid corpora can't be split here: their augmented children (and repeated blend "
        "parents) would straddle train and test — train/test leakage. Rebuild a held-out "
        "test set from the human source cartridges on the **Build dataset** page instead."
    )
    st.stop()

test_fraction = st.number_input(
    "Test fraction", min_value=0.0, max_value=1.0, value=0.2, step=0.05, format="%.2f",
    help="Share of samples held out as the test set.",
)
split_seed = int(st.number_input("Split seed", min_value=0, value=0, step=1,
                                  help="Seed for the train/test row shuffle."))
run_name = st.text_input(
    "Output base name (optional)", value="",
    help="_train / _test are appended. Blank uses the source corpus name.",
)

try:
    argv = build_command(
        SPLIT_CORPUS,
        {
            "corpus": str(corpus.path),
            "test_fraction": float(test_fraction),
            "split_seed": split_seed,
            "run_name": run_name,
        },
    )
except ValueError as exc:
    argv = None
    st.info(f"Fill required field(s): {exc}")

if argv:
    base = run_name or corpus.name
    st.caption(f"Produces `{base}_train` (audio copied) and `{base}_test` (re-rendered fresh).")
    ui.command_preview(argv)
    code = ui.run_button(argv, key="run_split")
    if code == 0:
        st.success(
            f"Split done. `{base}_test` is fresh-process — pick it as the eval corpus on the "
            "**Evaluate** page."
        )
