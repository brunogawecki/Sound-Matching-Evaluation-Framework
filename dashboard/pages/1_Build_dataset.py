"""Build a corpus: pick a preset source, fill the per-run knobs, run the script."""
from env import bootstrap

bootstrap()

import json
from pathlib import Path

import streamlit as st

import config
import discovery
import ui
from forms import render_form
from script_specs import BUILD_SOURCES

st.set_page_config(page_title="Build dataset", layout="wide")
st.title("Build dataset")

# The D1 subset is locked; the forms below only expose the per-run knobs.
try:
    from synth.dexed.subset import SUBSET_PARAM_NAMES

    with st.expander(f"D1 parameter subset — {len(SUBSET_PARAM_NAMES)} params (LOCKED, read-only)"):
        st.write("These are the parameters every model estimates. Fixed by decision D1.")
        st.write(sorted(SUBSET_PARAM_NAMES))
except Exception:  # pragma: no cover - display-only, never block the page
    st.caption("D1 parameter subset: 103 params (locked).")

source = st.radio(
    "Preset source",
    list(BUILD_SOURCES.keys()),
    horizontal=True,
    format_func=lambda key: {
        "synthetic": "Synthetic (random draws)",
        "human": "Human (.syx cartridges)",
        "hybrid": "Hybrid (blend / augment)",
        "presetgen": "preset-gen-vae SQLite",
    }[key],
)
spec = BUILD_SOURCES[source]
st.caption(spec.description)

if not st.session_state.get("_dexed_ok_checked"):
    st.session_state["_dexed_ok_checked"] = True
if not Path(config.DEXED_PATH).expanduser().exists():
    st.warning("Dexed VST not found — rendering will fail. Set `DEXED_PATH` in `.env`.")

try:
    argv = render_form(spec)
except ValueError as exc:
    argv = None
    st.info(f"Fill required field(s): {exc}")

if argv:
    ui.command_preview(argv)
    code = ui.run_button(argv, key=f"run_{source}")
    if code == 0:
        # Show the freshest corpus written (run-name may be script-defaulted).
        corpora = discovery.list_corpora()
        if corpora:
            newest = max(corpora, key=lambda c: c.path.stat().st_mtime)
            st.subheader(f"Result: `{newest.name}`")
            summary = discovery.load_summary(newest.path / "run_summary.json")
            st.json(
                {
                    key: summary.get(key)
                    for key in ("run_name", "num_samples", "near_silent_count",
                                "method_counts", "render_process", "renderer")
                }
            )
            audio_files = sorted((newest.path / "audio").glob("*.wav"))
            if audio_files:
                st.caption(f"Preview: {audio_files[0].name}")
                st.audio(audio_files[0].read_bytes(), format="audio/wav")
