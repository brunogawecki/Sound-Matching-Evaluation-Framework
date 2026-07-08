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


def collapse_carriage_returns(text: str, tail_lines: int = _TAIL_LINES) -> str:
    """Render ``text`` the way a terminal would, keeping only the last ``tail_lines`` lines.

    A ``\\n`` commits the line being built to scrollback; a ``\\r`` rewinds to
    the start of that line instead of starting a new one, so a ``tqdm``-style
    progress bar collapses to one animating line rather than one line per
    tick. Idempotent on its own output (re-collapsing a collapsed string is a
    no-op beyond the ``tail_lines`` cut), so :func:`run_streaming` can feed its
    result back in as the running buffer; callers rendering a one-shot snapshot
    of a polled remote log tail use it with no state carried between polls.
    """
    committed: List[str] = []
    current = ""
    for char in text:
        if char == "\n":
            committed.append(current)
            current = ""
        elif char == "\r":
            current = ""
        else:
            current += char
    lines = committed + ([current] if current else [])
    return "\n".join(lines[-tail_lines:])


def run_streaming(argv: List[str], placeholder) -> int:
    """Run ``argv``, tailing output into a Streamlit ``placeholder``; return code.

    The raw byte stream is collapsed (:func:`collapse_carriage_returns`) so
    carriage returns are honoured the way a terminal does. The collapsed tail is
    fed back in as the running buffer after each chunk, so memory stays bounded
    to ``tail_lines`` and the per-chunk work stays proportional to that tail
    rather than the whole (possibly huge, e.g. a per-file ``rsync -avP``) stream.
    ``placeholder`` is an ``st.empty()`` (or any object with ``.code(str)``).
    """
    buffer = ""
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

    while True:
        chunk = os.read(file_descriptor, 4096)
        if not chunk:
            break
        buffer = collapse_carriage_returns(buffer + decoder.decode(chunk))
        placeholder.code(buffer or "…")

    placeholder.code(buffer or "(no output)")
    return process.wait()
