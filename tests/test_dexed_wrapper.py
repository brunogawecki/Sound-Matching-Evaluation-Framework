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

EXCLUDED_NAMES = {"Cutoff", "Resonance", "Output", "MonoMode", "Bypass", "Program"}


def make_wrapper() -> DexedWrapper:
    return DexedWrapper(
        plugin_path=PLUGIN_PATH,
        sample_rate=config.SAMPLE_RATE,
        buffer_size=config.BUFFER_SIZE,
    )


@pytest.fixture(scope="module")
def synth() -> DexedWrapper:
    return make_wrapper()


# ---------------------------------------------------------------------------
# Parameter universe (D-NAMING, D-EXCLUDED)
# ---------------------------------------------------------------------------

def test_exposes_exactly_the_synthesis_parameters(synth):
    names = synth.parameter_names
    assert len(names) == 152
    assert "ALGORITHM" in names
    assert "MASTER TUNE ADJ" in names
    assert "OP1 OUTPUT LEVEL" in names
    assert "OP6 SWITCH" in names


def test_excluded_parameters_are_invisible(synth):
    names = set(synth.parameter_names)
    assert not names & EXCLUDED_NAMES
    assert not any(n.startswith("MIDI CC") for n in names)


def test_get_parameters_covers_exactly_the_exposed_universe(synth):
    params = synth.get_parameters()
    assert set(params.keys()) == set(synth.parameter_names)


def test_set_unknown_parameter_name_raises(synth):
    with pytest.raises(KeyError):
        synth.set_parameters({"NOT A REAL PARAM": 0.5})


def test_set_excluded_parameter_name_raises(synth):
    with pytest.raises(KeyError):
        synth.set_parameters({"Bypass": 1.0})


# ---------------------------------------------------------------------------
# Categorical mappings (F2 regression: names, not assumed indices)
# ---------------------------------------------------------------------------

def test_categorical_mappings_are_name_keyed_and_complete(synth):
    mappings = synth.get_categorical_mappings()
    expected = {"ALGORITHM": 32, "OSC KEY SYNC": 2, "LFO KEY SYNC": 2, "LFO WAVE": 6}
    for op in range(1, 7):
        expected[f"OP{op} MODE"] = 2
        expected[f"OP{op} L KEY SCALE"] = 4
        expected[f"OP{op} R KEY SCALE"] = 4
        expected[f"OP{op} SWITCH"] = 2
        expected[f"OP{op} F COARSE"] = 32  # D-KIND: perceptually discontinuous grid
    assert set(mappings.keys()) == set(expected.keys())
    for name, cardinality in expected.items():
        assert mappings[name]["cardinality"] == cardinality
        options = mappings[name]["options"]
        assert len(options) == cardinality
        assert options[0] == 0.0 and options[-1] == 1.0


def test_bounds_defaults_come_from_the_init_patch(synth):
    """The JUCE defaultValue field is 0.0 for everything; the freshly-loaded
    plugin state is the real default and must be what bounds report."""
    fresh = make_wrapper()
    initial = fresh.get_parameters()
    bounds = fresh.get_parameter_bounds()
    defaults = fresh.get_parameter_defaults()
    assert set(defaults.keys()) == set(fresh.parameter_names)
    for name, bound in bounds.items():
        assert bound["default"] == pytest.approx(initial[name]), name


def test_bounds_and_categoricals_partition_the_universe(synth):
    bounds = synth.get_parameter_bounds()
    categoricals = synth.get_categorical_mappings()
    names = set(synth.parameter_names)
    assert set(bounds.keys()) | set(categoricals.keys()) == names
    assert not set(bounds.keys()) & set(categoricals.keys())


# ---------------------------------------------------------------------------
# Set / get round-trip
# ---------------------------------------------------------------------------

def test_set_get_roundtrip(synth):
    rng = np.random.default_rng(42)
    params = synth.randomize_parameters(rng)
    synth.set_parameters(params)
    readback = synth.get_parameters()
    for name, value in params.items():
        assert readback[name] == pytest.approx(value, abs=1e-6), name


# ---------------------------------------------------------------------------
# Randomization (seedable, valid, exposed-only)
# ---------------------------------------------------------------------------

def test_randomize_is_deterministic_with_seed(synth):
    a = synth.randomize_parameters(np.random.default_rng(123))
    b = synth.randomize_parameters(np.random.default_rng(123))
    assert a == b


def test_randomize_only_touches_exposed_params_and_valid_values(synth):
    params = synth.randomize_parameters(np.random.default_rng(7))
    names = set(synth.parameter_names)
    assert set(params.keys()) == names
    categoricals = synth.get_categorical_mappings()
    for name, value in params.items():
        assert 0.0 <= value <= 1.0, name
        if name in categoricals:
            options = categoricals[name]["options"]
            assert any(abs(value - option) < 1e-9 for option in options), name


# ---------------------------------------------------------------------------
# Render contract (D-REPRO) and audio format
# ---------------------------------------------------------------------------

def test_render_is_mono_with_correct_length(synth):
    synth.set_parameters(synth.randomize_parameters(np.random.default_rng(1)))
    audio = synth.render_audio(midi_note=60, velocity=100, duration_sec=2.0)
    assert audio.ndim == 1
    assert len(audio) == int(2.0 * synth.sample_rate)


def test_consecutive_renders_are_bit_identical(synth):
    synth.set_parameters(synth.randomize_parameters(np.random.default_rng(2)))
    a = synth.render_audio(midi_note=60, velocity=100, duration_sec=2.0)
    b = synth.render_audio(midi_note=60, velocity=100, duration_sec=2.0)
    assert np.array_equal(a, b)


def test_renders_match_across_fresh_engines(synth):
    params = synth.randomize_parameters(np.random.default_rng(3))
    synth.set_parameters(params)
    a = synth.render_audio(midi_note=60, velocity=100, duration_sec=2.0)

    other = make_wrapper()
    other.set_parameters(params)
    b = other.render_audio(midi_note=60, velocity=100, duration_sec=2.0)
    assert np.array_equal(a, b)


@pytest.mark.xfail(
    strict=False,
    reason="Dexed keeps hidden engine state that is not reset by parameter "
    "re-application, prepareToPlay, state reload, or processor rebuild; the "
    "same patch can render audibly differently depending on what was rendered "
    "before it, and the outcome is context-dependent (some patches/sequences "
    "match by luck). Investigated 2026-06-11; see D-REPRO in docs/DECISIONS.md.",
)
def test_render_unaffected_by_previous_render_content():
    """The desirable (currently unachievable) contract: the same patch renders
    bit-identically regardless of which patch was rendered before it."""
    w = make_wrapper()
    params_a = w.randomize_parameters(np.random.default_rng(11))
    params_b = w.randomize_parameters(np.random.default_rng(12))
    params_c = w.randomize_parameters(np.random.default_rng(13))

    fresh = make_wrapper()
    fresh.set_parameters(params_b)
    reference = fresh.render_audio(60, 100, 4.0, note_duration_sec=3.0)

    w.set_parameters(params_a)
    w.render_audio(60, 100, 4.0, note_duration_sec=3.0)
    w.set_parameters(params_b)
    after_a = w.render_audio(60, 100, 4.0, note_duration_sec=3.0)

    w.set_parameters(params_c)
    w.render_audio(60, 100, 4.0, note_duration_sec=3.0)
    w.set_parameters(params_b)
    after_c = w.render_audio(60, 100, 4.0, note_duration_sec=3.0)

    assert np.array_equal(reference, after_a)
    assert np.array_equal(reference, after_c)


def test_renders_reproduce_across_identical_fresh_processes():
    """The achievable render contract (D-REPRO): the same synth-side dict
    rendered at the same position of an identical fresh process is
    bit-identical. Phase 2 dataset generation and evaluation re-rendering
    rely on this."""
    script = (
        "import hashlib, numpy as np, config\n"
        "from synth.dexed import DexedWrapper\n"
        "w = DexedWrapper(config.DEXED_PATH, config.SAMPLE_RATE, config.BUFFER_SIZE)\n"
        "p = w.parameter_space.sample_uniform(np.random.default_rng(7))\n"
        "w.set_parameters(p)\n"
        "a = w.render_audio(60, 100, 4.0, 3.0)\n"
        "b = w.render_audio(60, 100, 4.0, 3.0)\n"
        "print(hashlib.sha256(a.tobytes()).hexdigest(),"
        " hashlib.sha256(b.tobytes()).hexdigest())\n"
    )
    import subprocess

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "PYTHONPATH": root}
    outs = [
        subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, env=env, cwd=root, check=True,
        ).stdout.strip().splitlines()[-1]
        for _ in range(2)
    ]
    assert outs[0] == outs[1]


def test_note_duration_releases_before_render_end(synth):
    """With note-off mid-render the second half must differ from a held note (D3)."""
    fresh = make_wrapper()  # default patch: predictable sustained tone
    held = fresh.render_audio(midi_note=60, velocity=100, duration_sec=2.0)
    released = fresh.render_audio(
        midi_note=60, velocity=100, duration_sec=2.0, note_duration_sec=1.0
    )
    half = len(held) // 2
    assert np.array_equal(held[: half // 2], released[: half // 2])
    assert not np.array_equal(held[half:], released[half:])
