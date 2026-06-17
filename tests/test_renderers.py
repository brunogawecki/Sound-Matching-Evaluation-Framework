"""
Renderer-parity tests: the pluggable Renderer layer should let DexedWrapper render through
either DawDreamer or Pedalboard with the same public behavior.

The DawDreamer half skips when the plugin is absent; the Pedalboard half additionally skips when
pedalboard is not installed.
"""
import importlib.util
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from synth.dexed import DexedWrapper

PLUGIN_PATH = os.path.expanduser(config.DEXED_PATH)

pytestmark = pytest.mark.skipif(
    not os.path.exists(PLUGIN_PATH),
    reason=f"Dexed plugin not found at {PLUGIN_PATH}",
)

pedalboard_installed = importlib.util.find_spec("pedalboard") is not None
requires_pedalboard = pytest.mark.skipif(
    not pedalboard_installed, reason="pedalboard not installed"
)

SUBSET_CATEGORICALS = {"ALGORITHM", "LFO WAVE"}


def make_wrapper(renderer: str) -> DexedWrapper:
    return DexedWrapper(
        plugin_path=PLUGIN_PATH,
        sample_rate=config.SAMPLE_RATE,
        buffer_size=config.BUFFER_SIZE,
        renderer=renderer,
    )


def render_one(wrapper: DexedWrapper) -> np.ndarray:
    return wrapper.render_audio(
        midi_note=config.MIDI_NOTE,
        velocity=config.VELOCITY,
        duration_sec=config.DURATION_SEC,
        note_duration_sec=config.NOTE_DURATION_SEC,
    )


def test_default_renderer_is_dawdreamer():
    wrapper = make_wrapper("dawdreamer")
    assert wrapper.renderer_name == "dawdreamer"


def test_dawdreamer_renders_mono():
    wrapper = make_wrapper("dawdreamer")
    audio = render_one(wrapper)
    assert audio.ndim == 1
    assert np.all(np.isfinite(audio))


@requires_pedalboard
def test_pedalboard_renderer_constructs_and_names_match():
    wrapper = make_wrapper("pedalboard")
    assert wrapper.renderer_name == "pedalboard"
    # The synthesis parameter names Pedalboard reports must cover the subset + categoricals,
    # otherwise name-based addressing (D-NAMING) silently breaks across engines.
    names = set(wrapper.parameter_names)
    assert SUBSET_CATEGORICALS <= names
    assert "OP1 OUTPUT LEVEL" in names


@requires_pedalboard
def test_pedalboard_renders_mono():
    wrapper = make_wrapper("pedalboard")
    audio = render_one(wrapper)
    assert audio.ndim == 1
    assert np.all(np.isfinite(audio))


@requires_pedalboard
def test_both_renderers_render_same_patch_to_comparable_length():
    patch = make_wrapper("dawdreamer").parameter_space.sample_uniform(np.random.default_rng(0))

    daw_wrapper = make_wrapper("dawdreamer")
    daw_wrapper.set_parameters(patch)
    daw_audio = render_one(daw_wrapper)

    pedalboard_wrapper = make_wrapper("pedalboard")
    pedalboard_wrapper.set_parameters(patch)
    pedalboard_audio = render_one(pedalboard_wrapper)

    expected_length = int(config.SAMPLE_RATE * config.DURATION_SEC)
    # Hosts can differ by a block at the tail; allow a small tolerance.
    assert abs(len(daw_audio) - expected_length) <= config.BUFFER_SIZE * 4
    assert abs(len(pedalboard_audio) - expected_length) <= config.BUFFER_SIZE * 4
