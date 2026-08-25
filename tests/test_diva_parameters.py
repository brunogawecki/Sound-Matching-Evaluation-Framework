import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from synth.diva.parameters import (
    DIVA_DISCRETE_STEPS,
    DIVA_PARAMETER_NAMES,
    build_name_to_index,
    module_name,
    plugin_name,
)

PLUGIN_PATH = os.path.expanduser(config.DIVA_PATH)

# The 2080 MIDI CC passthroughs Diva exposes alongside its real parameters.
MIDI_CC = re.compile(r"^CC \d+, CH \d+$")


def test_names_are_unique():
    # The whole point of qualifying by module: Diva reports six parameters named 'Rate'.
    assert len(set(DIVA_PARAMETER_NAMES)) == len(DIVA_PARAMETER_NAMES)


def test_every_name_is_module_qualified():
    for name in DIVA_PARAMETER_NAMES:
        assert "." in name, f"{name!r} is not module-qualified"
        assert module_name(name) and plugin_name(name)


def test_bare_names_really_do_collide():
    # Guards the reason this table exists: drop the module and names stop being unique.
    bare = [plugin_name(name) for name in DIVA_PARAMETER_NAMES]
    assert len(set(bare)) < len(bare)


def test_discrete_steps_are_known_parameters_with_real_grids():
    assert set(DIVA_DISCRETE_STEPS) <= set(DIVA_PARAMETER_NAMES)
    assert all(steps >= 2 for steps in DIVA_DISCRETE_STEPS.values())


def test_name_to_index_round_trips():
    mapping = build_name_to_index()
    assert len(mapping) == len(DIVA_PARAMETER_NAMES)
    assert mapping[DIVA_PARAMETER_NAMES[155]] == 155


@pytest.mark.skipif(
    not os.path.exists(PLUGIN_PATH), reason=f"Diva plugin not found at {PLUGIN_PATH}"
)
def test_table_still_matches_the_live_plugin():
    """The table is static, so a Diva update that shifts indices must fail loudly here.

    Diva never reports the owning module, so only the bare name can be checked -- but that is
    enough: an inserted parameter shifts everything after it and the names stop lining up.
    """
    from synth.renderers.dawdreamer_renderer import DawDreamerRenderer

    renderer = DawDreamerRenderer(PLUGIN_PATH, config.SAMPLE_RATE, config.BUFFER_SIZE)
    reported = [
        str(entry["name"])
        for entry in renderer.parameter_descriptions()
        if not MIDI_CC.match(str(entry["name"])) and str(entry["name"]) != "Program"
    ]

    assert len(reported) == len(DIVA_PARAMETER_NAMES), (
        f"Diva reports {len(reported)} real parameters, table has "
        f"{len(DIVA_PARAMETER_NAMES)}. The plugin version changed."
    )
    mismatches = [
        (index, expected, actual)
        for index, (expected, actual) in enumerate(zip(DIVA_PARAMETER_NAMES, reported))
        if plugin_name(expected) != actual
    ]
    assert not mismatches, f"table drifted from the plugin at {mismatches[:5]}"


@pytest.mark.skipif(
    not os.path.exists(PLUGIN_PATH), reason=f"Diva plugin not found at {PLUGIN_PATH}"
)
def test_discrete_steps_still_match_the_live_plugin():
    """DIVA_DISCRETE_STEPS is frozen so it does not depend on a renderer that reports step
    counts (Pedalboard does not). That only holds while the plugin still agrees."""
    from synth.renderers.dawdreamer_renderer import DawDreamerRenderer

    renderer = DawDreamerRenderer(PLUGIN_PATH, config.SAMPLE_RATE, config.BUFFER_SIZE)
    descriptions = renderer.parameter_descriptions()

    reported = {
        DIVA_PARAMETER_NAMES[index]: int(descriptions[index]["numSteps"])
        for index in range(len(DIVA_PARAMETER_NAMES))
        if descriptions[index]["isDiscrete"]
    }
    assert reported == DIVA_DISCRETE_STEPS
