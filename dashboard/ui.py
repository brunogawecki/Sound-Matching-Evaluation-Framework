"""Small shared Streamlit widgets used by the pages."""
import shlex
from typing import List, Optional

import streamlit as st

import command_runner


def command_preview(argv: List[str]) -> None:
    """Show the exact shell command (copyable) that a Run will execute."""
    st.caption("Command (copyable — reproducible from a terminal or the cluster):")
    st.code(shlex.join(argv), language="bash")


def run_button(argv: List[str], label: str = "▶ Run", key: Optional[str] = None) -> Optional[int]:
    """Render a Run button; on click, stream the subprocess and report exit.

    Returns the process exit code on the run, else ``None`` (not clicked this rerun).
    """
    if not st.button(label, key=key, type="primary"):
        return None
    placeholder = st.empty()
    with st.spinner("Running… (this tab is busy until it finishes)"):
        code = command_runner.run_streaming(argv, placeholder)
    if code == 0:
        st.success("Finished (exit 0).")
    else:
        st.error(f"Exited with code {code}. See the log above.")
    return code
