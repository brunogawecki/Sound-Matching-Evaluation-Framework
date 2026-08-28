"""DivaWrapper: parameter universe, addressing, and the render contract.

The sibling of ``test_dexed_wrapper.py``, and deliberately parallel to it, but the render
half reaches the opposite conclusion. Dexed leaks a little hidden voice state and so renders
*almost* reproducibly in-process; Diva does not reproduce in-process at all, and only a fresh
process gives back the same audio (D-DIVA-RENDER, docs/DECISIONS.md). Those tests re-verify
that rather than assuming it transfers.

Everything here needs the plugin and skips without it. The parameter *table* is checked
against the live plugin in ``test_diva_parameters.py``, and the estimated subset in
``test_diva_subset.py``; neither is repeated here.
"""
import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from synth.diva import DivaWrapper
from synth.diva.parameters import DIVA_PARAMETER_NAMES

PLUGIN_PATH = os.path.expanduser(config.DIVA_PATH)

pytestmark = pytest.mark.skipif(
    not os.path.exists(PLUGIN_PATH),
    reason=f"Diva plugin not found at {PLUGIN_PATH}",
)

# Not synthesis parameters, so the wrapper hides them (D-EXCLUDED): a master gain that moves
# every audio metric without changing the patch, and the GUI's LED tint.
EXCLUDED_NAMES = {"main.Output", "PCore.LED Colour"}


def make_wrapper() -> DivaWrapper:
    return DivaWrapper(
        plugin_path=PLUGIN_PATH,
        sample_rate=config.SAMPLE_RATE,
        buffer_size=config.BUFFER_SIZE,
    )


@pytest.fixture(scope="module")
def synth() -> DivaWrapper:
    return make_wrapper()


# ---------------------------------------------------------------------------
# Parameter universe (D-NAMING as amended, D-EXCLUDED)
# ---------------------------------------------------------------------------

def test_exposes_exactly_the_synthesis_parameters(synth):
    names = synth.parameter_names
    assert len(names) == len(DIVA_PARAMETER_NAMES) - len(EXCLUDED_NAMES) == 279
    assert "VCF1.Model" in names
    assert "OSC.Volume1" in names
    assert "ENV1.Attack" in names
    assert "ARP.Restart" in names


def test_excluded_parameters_are_invisible(synth):
    names = set(synth.parameter_names)
    assert not names & EXCLUDED_NAMES
    # The 2080 MIDI CC passthroughs and the trailing 'Program' sit outside the name table
    # entirely, so they never reach the wrapper's universe either.
    assert not any(name.startswith("MIDI CC") for name in names)
    assert "Program" not in names


def test_every_exposed_name_is_module_qualified(synth):
    # Diva's plugin-reported names are not unique -- 56 are shared by 147 parameters -- so a
    # bare name does not identify a parameter and the wrapper never speaks one.
    assert all("." in name for name in synth.parameter_names)


def test_parameter_names_are_in_plugin_index_order(synth):
    expected = [name for name in DIVA_PARAMETER_NAMES if name not in EXCLUDED_NAMES]
    assert synth.parameter_names == expected


def test_get_parameters_covers_exactly_the_exposed_universe(synth):
    assert set(synth.get_parameters()) == set(synth.parameter_names)


def test_set_unknown_parameter_name_raises(synth):
    with pytest.raises(KeyError):
        synth.set_parameters({"VCF1.NoSuchKnob": 0.5})


def test_set_excluded_parameter_name_raises(synth):
    for name in EXCLUDED_NAMES:
        with pytest.raises(KeyError):
            synth.set_parameters({name: 0.5})


def test_set_bare_unqualified_name_raises(synth):
    # 'Model' alone is ambiguous across ENV1/ENV2/OSC/HPF/VCF1; it must not resolve.
    with pytest.raises(KeyError):
        synth.set_parameters({"Model": 0.5})


# ---------------------------------------------------------------------------
# Bounds, categoricals and defaults
# ---------------------------------------------------------------------------

def test_categorical_mappings_are_name_keyed_and_complete(synth):
    mappings = synth.get_categorical_mappings()
    assert set(mappings) <= set(synth.parameter_names)
    assert "VCF1.Model" in mappings
    for name, mapping in mappings.items():
        options = mapping["options"]
        assert len(options) == mapping["cardinality"]
        assert options[0] == pytest.approx(0.0)
        assert options[-1] == pytest.approx(1.0 if len(options) > 1 else 0.0)
        assert options == sorted(options)


def test_bounds_defaults_come_from_the_init_patch(synth):
    # The freshly-loaded patch, not JUCE's defaultValue field -- the two disagree on 72
    # parameters, and a render actually starts from the loaded patch.
    bounds = synth.get_parameter_bounds()
    defaults = synth.get_parameter_defaults()
    live = synth.get_parameters()
    for name, bound in bounds.items():
        assert bound["min"] == 0.0 and bound["max"] == 1.0
        assert bound["default"] == pytest.approx(defaults[name])
        assert defaults[name] == pytest.approx(live[name])


def test_bounds_and_categoricals_partition_the_universe(synth):
    continuous = set(synth.get_parameter_bounds())
    categorical = set(synth.get_categorical_mappings())
    assert continuous.isdisjoint(categorical)
    assert continuous | categorical == set(synth.parameter_names)


# ---------------------------------------------------------------------------
# Addressing round trips
# ---------------------------------------------------------------------------

def test_set_get_roundtrip(synth):
    patch = {"VCF1.Frequency": 0.25, "OSC.Volume1": 0.75, "ENV1.Attack": 0.5}
    synth.set_parameters(patch)
    live = synth.get_parameters()
    for name, value in patch.items():
        assert live[name] == pytest.approx(value, abs=1e-4)
    synth.set_parameters(synth.get_parameter_defaults())


def test_randomize_is_deterministic_with_seed(synth):
    first = synth.randomize_parameters(np.random.default_rng(5))
    second = synth.randomize_parameters(np.random.default_rng(5))
    assert first == second


def test_randomize_only_touches_exposed_params_and_valid_values(synth):
    params = synth.randomize_parameters(np.random.default_rng(6))
    assert set(params) == set(synth.parameter_names)
    categoricals = synth.get_categorical_mappings()
    for name, value in params.items():
        assert 0.0 <= value <= 1.0
        if name in categoricals:
            assert any(
                value == pytest.approx(option) for option in categoricals[name]["options"]
            )


def test_render_preset_ignores_names_outside_the_exposed_set(synth):
    # A diva_raw preset carries all 281 names, two of which the wrapper hides; passing it
    # through unfiltered must work rather than raise (dataset/diva_preset_loader.py).
    preset = {**synth.get_parameter_defaults(), "main.Output": 0.5, "PCore.LED Colour": 1.0}
    audio = synth.render_preset(preset, 60, 100, 1.0)
    assert audio.ndim == 1
    assert synth.get_parameters()["VCF1.Frequency"] == pytest.approx(
        synth.get_parameter_defaults()["VCF1.Frequency"], abs=1e-4
    )


# ---------------------------------------------------------------------------
# Renderer restriction (D-RENDERER)
# ---------------------------------------------------------------------------

def test_only_dawdreamer_is_accepted():
    # Pedalboard reports 2271 parameters against DawDreamer's 2362 because it silently drops
    # the 91 colliding names, which shifts every index. The name table is written against
    # DawDreamer's indices, so any other renderer would repoint the whole space without
    # erroring. Refuse rather than mis-map.
    with pytest.raises(ValueError, match="does not support the 'pedalboard' renderer"):
        DivaWrapper(plugin_path=PLUGIN_PATH, renderer="pedalboard")


def test_reports_its_renderer_and_synth_name(synth):
    assert synth.renderer_name == "dawdreamer"
    # Written into every corpus's run_summary.json and read back as a render-registry key.
    assert synth.synth_name == "diva"


# ---------------------------------------------------------------------------
# Render contract and audio format
# ---------------------------------------------------------------------------

def test_render_is_mono_with_correct_length(synth):
    synth.set_parameters(synth.get_parameter_defaults())
    audio = synth.render_audio(midi_note=60, velocity=100, duration_sec=2.0)
    assert audio.ndim == 1
    assert len(audio) == int(2.0 * synth.sample_rate)


def test_init_patch_renders_non_silent():
    assert np.max(np.abs(make_wrapper().render_audio(60, 100, 2.0))) > 0.0


def test_note_duration_releases_before_render_end():
    """With note-off mid-render the tail must decay (D3).

    Compared on energy, not sample equality: two Diva renders of one patch never match
    unless they are in separate processes, so each side gets its own wrapper and only the
    envelope's effect is asserted.
    """
    held = make_wrapper().render_audio(midi_note=60, velocity=100, duration_sec=2.0)
    released = make_wrapper().render_audio(
        midi_note=60, velocity=100, duration_sec=2.0, note_duration_sec=1.0
    )
    quarter = len(held) // 4

    def rms(block):
        return float(np.sqrt(np.mean(block ** 2)))

    assert rms(released[-quarter:]) < 0.25 * rms(held[-quarter:])   # tail decayed away
    assert rms(released[:quarter]) == pytest.approx(rms(held[:quarter]), rel=0.05)


@pytest.mark.xfail(
    strict=False,
    reason="Diva does not reproduce in-process: consecutive renders of one identical patch "
    "through a single wrapper differ, and re-applying the parameter state, zeroing the OPT "
    "slop knobs and zeroing OSC.Drift all fail to fix it. Unlike Dexed's mild voice-state "
    "leak the divergence can be near-total (85% RMS apart on a uniform-sampled patch). "
    "Measured 2026-08-25; see D-DIVA-RENDER in docs/DECISIONS.md.",
)
def test_consecutive_renders_are_bit_identical(synth):
    """The desirable (for Diva, unachievable) contract, kept as a tripwire: if a future
    Diva build starts reproducing in-process, this passes and D-DIVA-RENDER can be revisited."""
    synth.set_parameters(synth.parameter_space.sample_uniform(np.random.default_rng(2)))
    first = synth.render_audio(midi_note=60, velocity=100, duration_sec=2.0)
    second = synth.render_audio(midi_note=60, velocity=100, duration_sec=2.0)
    assert np.array_equal(first, second)


def test_wrapper_declares_that_it_cannot_render_in_process():
    # The flag the in-process render backends refuse on, rather than silently building a
    # corpus that cannot be reproduced.
    assert DivaWrapper.supports_in_process_render is False


def test_renders_reproduce_across_identical_fresh_processes():
    """The achievable render contract (D-DIVA-RENDER): one patch rendered at position 0 of
    an identical fresh process is bit-identical. Corpus generation and eval re-rendering
    both rely on this, which is why the Diva path is always fresh-process."""
    script = (
        "import hashlib, numpy as np, config\n"
        "from synth.diva import DivaWrapper\n"
        "w = DivaWrapper(config.DIVA_PATH, config.SAMPLE_RATE, config.BUFFER_SIZE)\n"
        "p = w.parameter_space.sample_uniform(np.random.default_rng(7))\n"
        "w.set_parameters(p)\n"
        "a = w.render_audio(60, 100, 4.0, 3.0)\n"
        "print('HASH', hashlib.sha256(a.tobytes()).hexdigest())\n"
    )
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = {**os.environ, "PYTHONPATH": root}
    hashes = []
    for _ in range(2):
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, env=env, cwd=root, check=True,
        )
        # Diva prints a long plugin banner to stdout, so pick the tagged line out of it.
        hashes.append(
            next(line for line in completed.stdout.splitlines() if line.startswith("HASH"))
        )
    assert hashes[0] == hashes[1]
