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
import codecs
import os
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

    The raw byte stream is consumed so carriage returns are honoured the way a
    terminal does: a ``\\r`` rewinds the current line, so a ``tqdm`` progress bar
    stays a single line that updates in place instead of one line per tick. A
    ``\\n`` commits the current line to the scrollback. ``placeholder`` is an
    ``st.empty()`` (or any object with ``.code(str)``).
    """
    committed: List[str] = []  # finished lines (ended with \n)
    current = ""  # the line currently being (over)written
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    process = subprocess.Popen(
        argv,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
    )
    assert process.stdout is not None
    file_descriptor = process.stdout.fileno()

    def render() -> None:
        tail = committed[-_TAIL_LINES:]
        text = "\n".join(tail + ([current] if current else []))
        placeholder.code(text or "…")

    while True:
        chunk = os.read(file_descriptor, 4096)
        if not chunk:
            break
        for char in decoder.decode(chunk):
            if char == "\n":
                committed.append(current)
                current = ""
            elif char == "\r":
                current = ""  # carriage return: overwrite the current line
            else:
                current += char
        render()

    if current:
        committed.append(current)
    render()
    return_code = process.wait()
    if not committed:
        placeholder.code("(no output)")
    return return_code
