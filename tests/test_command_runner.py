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
