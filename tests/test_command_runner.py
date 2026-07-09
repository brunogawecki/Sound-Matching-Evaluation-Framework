"""Tests for dashboard/command_runner.py's pure helpers."""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "dashboard"))

import command_runner  # noqa: E402


def test_collapse_carriage_returns_empty():
    assert command_runner.collapse_carriage_returns("") == ""


def test_collapse_carriage_returns_plain_lines():
    assert command_runner.collapse_carriage_returns("a\nb\nc") == "a\nb\nc"


def test_collapse_carriage_returns_overwrites_current_line():
    # tqdm-style: repeated \r rewrites, only the final segment survives.
    text = "epoch 1/10\repoch 2/10\repoch 3/10"
    assert command_runner.collapse_carriage_returns(text) == "epoch 3/10"


def test_collapse_carriage_returns_mixes_committed_and_current():
    text = "starting\nepoch 1/10\repoch 2/10"
    assert command_runner.collapse_carriage_returns(text) == "starting\nepoch 2/10"


def test_collapse_carriage_returns_respects_tail_lines():
    text = "\n".join(f"line{i}" for i in range(5))
    assert command_runner.collapse_carriage_returns(text, tail_lines=2) == "line3\nline4"


def test_collapse_carriage_returns_idempotent_on_own_output():
    # run_streaming feeds the collapsed tail back in as its buffer, so
    # re-collapsing a collapsed string must be a no-op (below the tail cut).
    once = command_runner.collapse_carriage_returns("starting\nepoch 1/10\repoch 2/10")
    assert command_runner.collapse_carriage_returns(once) == once


def test_collapse_carriage_returns_preserves_committed_line_boundary():
    # A buffer ending in a committed (newline-terminated) line must keep that
    # boundary, so feeding the next chunk back in starts a fresh line rather
    # than gluing it onto the previous one (the run_streaming feed-back loop).
    once = command_runner.collapse_carriage_returns("row0\n")
    twice = command_runner.collapse_carriage_returns(once + "row1\n")
    assert twice.splitlines() == ["row0", "row1"]


class _FakePlaceholder:
    def __init__(self):
        self.calls = []

    def code(self, text):
        self.calls.append(text)


def test_run_streaming_collapses_carriage_returns_end_to_end():
    # A real subprocess whose stdout mixes \r overwrites and \n commits; the
    # final rendered frame must match a terminal's view, exercising the
    # feed-back buffer loop (not just the pure helper).
    placeholder = _FakePlaceholder()
    code = command_runner.run_streaming(
        ["python", "-c", r"import sys; sys.stdout.write('a\rb\nc\rd')"],
        placeholder,
    )
    assert code == 0
    assert placeholder.calls[-1] == "b\nd"


def test_run_streaming_preserves_newlines_across_chunks():
    # Lines emitted one flush at a time arrive as separate os.read() chunks, so
    # each chunk ends on a \n. The feed-back buffer must not glue consecutive
    # committed lines together (regression: 'row0row1row2' instead of 3 lines).
    placeholder = _FakePlaceholder()
    code = command_runner.run_streaming(
        ["python", "-c",
         "import sys, time\n"
         "for i in range(3):\n"
         "    sys.stdout.write('row%d\\n' % i); sys.stdout.flush(); time.sleep(0.05)"],
        placeholder,
    )
    assert code == 0
    assert placeholder.calls[-1].splitlines() == ["row0", "row1", "row2"]


def test_run_streaming_reports_no_output():
    placeholder = _FakePlaceholder()
    code = command_runner.run_streaming(["python", "-c", "pass"], placeholder)
    assert code == 0
    assert placeholder.calls[-1] == "(no output)"
