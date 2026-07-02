"""Run a built command as a subprocess.

Two entry points share one behaviour:

* :func:`run_capture` -- blocking, returns ``(returncode, combined_output)``.
  Pure of Streamlit, so the tests use it.
* :func:`run_streaming` -- same, but tails the merged stdout/stderr into a live
  Streamlit placeholder as it arrives (block-and-stream). Text mode enables
  universal-newline translation, so ``tqdm``'s ``\\r`` progress updates arrive as
  separate lines and animate the log tail.

Commands run with ``cwd=PROJECT_ROOT`` so the scripts resolve their imports and
relative paths exactly as they do from a terminal.
"""
import subprocess
from typing import List, Optional, Tuple

from env import PROJECT_ROOT

_TAIL_LINES = 40


def run_capture(argv: List[str], cwd: Optional[str] = None) -> Tuple[int, str]:
    """Run to completion; return (returncode, combined stdout+stderr)."""
    completed = subprocess.run(
        argv,
        cwd=cwd or str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.returncode, completed.stdout


def run_streaming(argv: List[str], placeholder) -> int:
    """Run ``argv``, tailing output into a Streamlit ``placeholder``; return code.

    ``placeholder`` is an ``st.empty()`` (or any object with ``.code(str)``).
    """
    lines: List[str] = []
    process = subprocess.Popen(
        argv,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        lines.append(line.rstrip("\n"))
        placeholder.code("\n".join(lines[-_TAIL_LINES:]) or "…")
    return_code = process.wait()
    if not lines:
        placeholder.code("(no output)")
    return return_code
