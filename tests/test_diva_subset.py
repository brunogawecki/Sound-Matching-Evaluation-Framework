import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from synth.diva.parameters import DIVA_DISCRETE_STEPS, DIVA_PARAMETER_NAMES, module_name
from synth.diva.subset import (
    SUBSET_PARAM_NAMES,
    _DROPPED_MODULES,
    _DROPPED_PARAM_NAMES,
    build_parameter_space,
)

PLUGIN_PATH = os.path.expanduser(config.DIVA_PATH)
needs_plugin = pytest.mark.skipif(
    not os.path.exists(PLUGIN_PATH), reason=f"Diva plugin not found at {PLUGIN_PATH}"
)


@pytest.fixture(scope="module")
def diva():
    from synth.diva import DivaWrapper
    return DivaWrapper(
        plugin_path=PLUGIN_PATH,
        sample_rate=config.SAMPLE_RATE,
        buffer_size=config.BUFFER_SIZE,
    )


def test_subset_is_a_deduplicated_subset_of_the_parameter_table():
    assert len(set(SUBSET_PARAM_NAMES)) == len(SUBSET_PARAM_NAMES)
    assert set(SUBSET_PARAM_NAMES) <= set(DIVA_PARAMETER_NAMES)


def test_subset_keeps_the_table_order():
    # Order is the ML-side vector layout, so it must stay pinned to the plugin's index order.
    positions = [DIVA_PARAMETER_NAMES.index(name) for name in SUBSET_PARAM_NAMES]
    assert positions == sorted(positions)


def test_every_dropped_name_exists():
    # A typo in a drop list would silently keep the parameter it meant to drop.
    for name in _DROPPED_PARAM_NAMES:
        assert name in DIVA_PARAMETER_NAMES, f"{name!r} is not a Diva parameter"


def test_dropped_names_and_modules_are_actually_absent():
    for name in _DROPPED_PARAM_NAMES:
        assert name not in SUBSET_PARAM_NAMES
    for name in SUBSET_PARAM_NAMES:
        assert module_name(name) not in _DROPPED_MODULES


def test_subset_size_is_pinned():
    # 281 table entries, 44 dropped (2 hidden by the wrapper + 42 by D-DIVA-SUBSET).
    assert len(SUBSET_PARAM_NAMES) == 237


def test_key_follow_and_velocity_depth_are_dropped():
    # D1's rule under Diva's names: non-identifiable at one fixed note and velocity.
    for name in SUBSET_PARAM_NAMES:
        assert not name.endswith(".KeyFollow")
    assert "ENV1.Velocity" not in SUBSET_PARAM_NAMES
    assert "ENV2.Velocity" not in SUBSET_PARAM_NAMES


@needs_plugin
def test_parameter_space_matches_the_subset(diva):
    space = build_parameter_space(diva)
    assert space.names == SUBSET_PARAM_NAMES
    assert space.synth_dimension == 237
    # 135 continuous + 102 one-hot blocks summing to 965.
    assert space.ml_dimension == 1100
    assert diva.parameter_space is diva.parameter_space  # cached


@needs_plugin
def test_kind_follows_the_plugins_own_discreteness(diva):
    space = build_parameter_space(diva)
    for parameter_spec in space.parameter_specs:
        expected = "categorical" if parameter_spec.name in DIVA_DISCRETE_STEPS else "continuous"
        assert parameter_spec.kind == expected, parameter_spec.name
        if parameter_spec.kind == "categorical":
            assert len(parameter_spec.options) == DIVA_DISCRETE_STEPS[parameter_spec.name]


@needs_plugin
def test_categorical_defaults_sit_on_the_grid(diva):
    space = build_parameter_space(diva)
    for parameter_spec in space.parameter_specs:
        if parameter_spec.kind == "categorical":
            assert parameter_spec.default in parameter_spec.options


@needs_plugin
def test_sampled_patch_roundtrips_through_the_ml_vector(diva):
    space = build_parameter_space(diva)
    patch = space.sample_uniform(np.random.default_rng(0))
    roundtrip = space.ml_vector_to_synth_dict(space.synth_dict_to_ml_vector(patch))
    assert roundtrip.keys() == patch.keys()
    for name, value in patch.items():
        assert roundtrip[name] == pytest.approx(value), name


@needs_plugin
def test_sampled_patch_renders_non_silent(diva):
    space = build_parameter_space(diva)
    patch = space.sample_uniform(np.random.default_rng(0))
    audio = diva.render_preset(patch, midi_note=60, velocity=100,
                               duration_sec=4.0, note_duration_sec=3.0)
    assert audio.ndim == 1
    assert np.max(np.abs(audio)) > 0.0


@needs_plugin
def test_dropped_parameters_stay_at_their_defaults(diva):
    # The contract for everything outside the subset: locked at the loaded-patch state.
    space = build_parameter_space(diva)
    patch = space.sample_uniform(np.random.default_rng(1))
    diva.set_parameters(patch)
    defaults = diva.get_parameter_defaults()
    current = diva.get_parameters()
    for name in diva.parameter_names:
        if name not in patch:
            assert current[name] == pytest.approx(defaults[name]), name
