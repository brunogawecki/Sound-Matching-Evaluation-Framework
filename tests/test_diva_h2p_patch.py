"""synth/diva/patch.py: parsing the .h2p text format and decoding it through the map.

Pure-Python fixtures for parsing and decoding (no VST, no preset library) plus a plugin-gated
round-trip check over real presets, mirroring the split test_diva_preset_loader.py uses for the
npz corpus: fabricate the input by hand for the logic tests, reach for the real thing only to
confirm the derived map still agrees with the live plugin.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from synth.diva import patch as diva_patch
from synth.diva.h2p_param_map import Decoding
from synth.diva.patch import parse_h2p, patch_parameters

PLUGIN_PATH = os.path.expanduser(config.DIVA_PATH)
PRESETS_PATH = os.path.expanduser(config.DIVA_PRESETS_PATH)

# A small, hand-built map covering all four decode kinds, isolated from the real (derived)
# H2P_PARAMETER_MAP so these tests pin the *logic*, not today's specific key assignments.
_FIXTURE_MAP = {
    ("ENV1", "Atk"): Decoding("ENV1.Attack", "linear", minimum=0.0, maximum=100.0),
    ("VCC", "Trsp"): Decoding("VCC.Transpose", "grid", grid=("-24.00", "0.00", "24.00")),
    ("VCC", "Mode"): Decoding("VCC.Mode", "label", grid=("poly", "mono", "legato")),
    ("LFO1", "Sync"): Decoding("LFO1.Sync", "index", offset=-1),
}


def write_h2p(tmp_path, modules, filename="Test Preset.h2p") -> str:
    """Write a minimal but format-correct .h2p file: header, #cm= sections, binary tail."""
    lines = ["/*@Meta", "", "Bank:", "'Test Bank'", "", "*/", "#AM=Diva", "#Vers=10001"]
    for module, keys in modules.items():
        lines.append(f"#cm={module}")
        for key, value in keys.items():
            lines.append(f"{key}={value}")
    lines.append("// Section for ugly compressed binary Data")
    lines.append("garbage-that-must-never-be-parsed=999")
    path = tmp_path / filename
    path.write_text("\n".join(lines), encoding="latin-1")
    return str(path)


@pytest.fixture(autouse=True)
def fixture_map(monkeypatch):
    monkeypatch.setattr(diva_patch, "H2P_PARAMETER_MAP", _FIXTURE_MAP)


# -- parse_h2p ----------------------------------------------------------------------------
def test_parse_reads_sections_by_module(tmp_path):
    path = write_h2p(tmp_path, {"ENV1": {"Atk": "9.00", "Dec": "10.00"}, "VCC": {"Mode": "1"}})
    parsed = parse_h2p(path)
    assert parsed.modules == {"ENV1": {"Atk": "9.00", "Dec": "10.00"}, "VCC": {"Mode": "1"}}


def test_parse_stops_at_the_binary_marker(tmp_path):
    path = write_h2p(tmp_path, {"ENV1": {"Atk": "9.00"}})
    parsed = parse_h2p(path)
    assert "garbage-that-must-never-be-parsed" not in parsed.modules.get("ENV1", {})
    assert all("garbage" not in keys for keys in parsed.modules.values())


def test_parse_strips_wrapping_quotes(tmp_path):
    path = write_h2p(tmp_path, {"FX1": {"Module": "'Chorus1'"}})
    parsed = parse_h2p(path)
    assert parsed.modules["FX1"]["Module"] == "Chorus1"


def test_parse_ignores_comment_lines_within_a_module(tmp_path):
    path = tmp_path / "commented.h2p"
    path.write_text(
        "\n".join([
            "#AM=Diva",
            "#cm=ENV1",
            "// a comment some future Diva version might add here",
            "Atk=9.00",
            "// Section for ugly compressed binary Data",
            "binary-garbage",
        ]),
        encoding="latin-1",
    )
    parsed = parse_h2p(str(path))
    assert parsed.modules == {"ENV1": {"Atk": "9.00"}}


# -- patch_parameters -----------------------------------------------------------------------
# Every test below fills in all four fixture-mapped keys (only one is the point of the test)
# so it clears the "every mapped parameter must be present" check, which has its own test.
_COMPLETE_MODULES = {
    "ENV1": {"Atk": "0.00"}, "VCC": {"Trsp": "0.00", "Mode": "poly"}, "LFO1": {"Sync": "-1"},
}


def _with(module, key, value):
    modules = {m: dict(keys) for m, keys in _COMPLETE_MODULES.items()}
    modules[module][key] = value
    return diva_patch.ParsedPatch(modules=modules)


def test_decodes_linear():
    values = patch_parameters(_with("ENV1", "Atk", "25.00"))
    assert values["ENV1.Attack"] == pytest.approx(0.25)


def test_decodes_grid_by_numeric_match():
    values = patch_parameters(_with("VCC", "Trsp", "0.00"))
    assert values["VCC.Transpose"] == pytest.approx(0.5)  # middle of a 3-step grid


def test_decodes_label_by_string_match():
    values = patch_parameters(_with("VCC", "Mode", "legato"))
    assert values["VCC.Mode"] == pytest.approx(1.0)  # last of 3 options


def test_decodes_index_with_nonzero_offset():
    # offset=-1: h2p value v means plugin step v - (-1) = v + 1.
    values = patch_parameters(_with("LFO1", "Sync", "-1"))
    assert values["LFO1.Sync"] == pytest.approx(0.0)  # step 0 of Sync's grid


def test_unknown_keys_are_ignored():
    parsed = _with("ENV1", "PSong", "1")  # PSong has no map entry
    values = patch_parameters(parsed)
    assert "PSong" not in values
    assert values["ENV1.Attack"] == pytest.approx(0.0)


def test_missing_mapped_parameter_raises():
    # The fixture map has 4 entries; a patch naming only one of them is missing the rest.
    parsed = diva_patch.ParsedPatch(modules={"ENV1": {"Atk": "0.00"}})
    with pytest.raises(KeyError, match="VCC.Transpose"):
        patch_parameters(parsed)


# -- plugin-gated round-trip over the real, derived map --------------------------------------
pytestmark_plugin = pytest.mark.skipif(
    not (os.path.exists(PLUGIN_PATH) and os.path.isdir(PRESETS_PATH)),
    reason=f"Diva plugin ({PLUGIN_PATH}) or preset library ({PRESETS_PATH}) not found",
)


@pytestmark_plugin
def test_real_map_round_trips_a_sample_of_real_presets(monkeypatch):
    """Undoes the fixture-map monkeypatch for this one test: it checks the actual, derived
    ``synth/diva/h2p_param_map.py`` against the live plugin, not the isolated fixture above.

    For every 'linear'/'grid'/'label' decoding (the ones with a display-text signal to check
    against -- 'index' has none by construction, see that kind's docstring), applying the
    decoded value must reproduce the .h2p file's own text. This is the automated half of the
    map's verification; a swap between two same-range parameters would still pass it, which is
    why the map is not trusted from this test alone (see docs/DECISIONS.md D-DIVA-START).
    """
    import dawdreamer as daw
    from pathlib import Path

    from synth.diva.h2p_param_map import H2P_PARAMETER_MAP as real_map
    from synth.diva.parameters import build_name_to_index
    from synth.plugin_output import suppressed_plugin_output

    monkeypatch.setattr(diva_patch, "H2P_PARAMETER_MAP", real_map)
    name_to_index = build_name_to_index()

    with suppressed_plugin_output():
        engine = daw.RenderEngine(config.SAMPLE_RATE, config.BUFFER_SIZE)
        plugin = engine.make_plugin_processor("diva", PLUGIN_PATH)

    paths = sorted(Path(PRESETS_PATH).rglob("*.h2p"))[:25]
    assert paths, "expected at least one .h2p preset under DIVA_PRESETS_PATH"

    with suppressed_plugin_output():
        for path in paths:
            parsed = parse_h2p(str(path))
            values = patch_parameters(parsed)
            for module, keys in parsed.modules.items():
                for key, raw in keys.items():
                    decoding = real_map.get((module, key))
                    if decoding is None or decoding.kind == "index":
                        continue
                    index = name_to_index[decoding.parameter_name]
                    plugin.set_parameter(index, values[decoding.parameter_name])
                    text = plugin.get_parameter_text(index)
                    if decoding.kind == "label":
                        assert text == raw, (path, module, key)
                    else:
                        assert abs(float(text) - float(raw)) < 0.06, (path, module, key)
