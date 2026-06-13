import os
import sys

import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "scripts"))

from render_preset import resolve_voice

NAMES = [
    "Say Again.", "LAURIE", "Beatmehrdr", "PHAROH", "Chroma 5",
    "Lavitar", "SloSwl", "OB Genviv", "SAW EM UP", "LFORez",
]


def test_resolve_by_index():
    assert resolve_voice("8", NAMES) == 8


def test_resolve_by_exact_name_is_case_insensitive():
    assert resolve_voice("laurie", NAMES) == 1


def test_resolve_by_unique_substring():
    assert resolve_voice("phar", NAMES) == 3


def test_resolve_ambiguous_substring_raises():
    with pytest.raises(ValueError):
        resolve_voice("la", NAMES)  # matches LAURIE and Lavitar


def test_resolve_unknown_name_raises():
    with pytest.raises(ValueError):
        resolve_voice("does-not-exist", NAMES)


def test_resolve_out_of_range_index_raises():
    with pytest.raises(ValueError):
        resolve_voice("99", NAMES)
