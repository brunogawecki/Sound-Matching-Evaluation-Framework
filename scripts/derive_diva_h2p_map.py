"""Derive synth/diva/h2p_param_map.py: which .h2p key means which Diva parameter.

Diva's .h2p preset format writes each parameter under a short, module-scoped key
(``#cm=ENV1`` / ``Atk=9.00``), not the module-qualified name synth/diva/parameters.py
addresses it by (``ENV1.Attack``). u-he publishes no correspondence between the two, so this
script derives one, per module, from two signals over the live plugin and the whole local
.h2p library:

  * VALUE COMPATIBILITY (hard filter) -- a key's observed values across every preset must be
    representable under the candidate parameter's own display grid (swept once via
    ``get_parameter_text``). A key holding "poly"/"mono"/"legato" cannot be 'VCF1.Model'.
  * NAME SIMILARITY (ranker among survivors) -- word-prefix segmentation of the key against
    the parameter's long name, with a digit-agreement term: without it, 'TM1On' outscores its
    real match 'TuneModOsc1' by landing on the *shorter* 'Tune1ModSrc' (matching digit '1'
    would fix it, but the ungated scorer let 'TM1On' -> 'TuneModOsc2' through instead --
    checked and confirmed on this codebase's OSC module before the digit term was added).

Per module, the two signals feed a bipartite assignment (Hungarian algorithm) rather than a
greedy one, because several modules have more keys than parameters (VCC: 27 keys / 14 params)
and a greedy match can grab a plausible-but-wrong key before a module runs out of parameters.

Three modules are skipped entirely: ``OPT``, ``Scope1``, ``PCore``. None contributes a single
parameter to D-DIVA-SUBSET (OPT and Scope1 are dropped whole; PCore's one exposed parameter,
``LED Colour``, is wrapper-excluded), and OPT's 43 h2p keys against 15 real parameters are
mostly internal housekeeping with no synthesis meaning at all -- guessing at them is pure risk
for zero benefit, since nothing downstream ever reads their decoded value.

Output is the discriminated decode this module's presets actually need, not one rule for every
parameter:

  * ``linear``    -- continuous parameters. All 164 of Diva's display exactly linearly (swept
                     and confirmed), so decode is (value - min) / (max - min), no lookup.
  * ``grid``      -- discrete parameters whose plugin display text is itself numeric (e.g.
                     'VCC.Transpose': -24.00 .. 24.00). The swept per-step text, parsed as
                     float, IS the lookup table; decode matches the .h2p value against it.
  * ``label``     -- discrete parameters whose display text is a non-numeric word ('poly',
                     'sine', 'Ladder'). The .h2p value is one of those words verbatim (after
                     the parser's quote-stripping); decode is a lookup in the swept text list.
  * ``index``     -- discrete parameters that are neither: the plugin displays a non-numeric
                     label (e.g. LFO1.Sync's '0.1s' / '1/64' / ...) but .h2p stores a bare
                     integer, so the step index has an unknown offset from 0. Solvable only
                     because the *range* of observed .h2p integers pins the offset uniquely
                     for all but a few parameters, checked against the plugin's step count.

This script requires the live Diva plugin (to sweep display grids) and the local .h2p preset
library (``config.DIVA_PRESETS_PATH``, factory + THIRD PARTY). Its output is data, reviewed
once and committed as ``synth/diva/h2p_param_map.py`` -- like ``synth/diva/parameters.py``,
never recomputed at import time, so a Diva update cannot silently repoint it.

Run:
    python scripts/derive_diva_h2p_map.py                  # print the report
    python scripts/derive_diva_h2p_map.py --write-map       # also (re)write the map module
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from collections import Counter as collections_counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from synth.diva.parameters import DIVA_PARAMETER_NAMES, module_name, plugin_name
from synth.diva.patch import parse_h2p

_MAP_MODULE_PATH = Path(__file__).resolve().parent.parent / "synth" / "diva" / "h2p_param_map.py"

# A handful of assignments the automated scorer gets wrong or leaves ambiguous, resolved by
# hand after reviewing the report this script prints (name score, runner-up, margin). Keying
# by (module, h2p_key) so a correction survives even if the parameter list is re-swept.
MANUAL_OVERRIDES: Dict[Tuple[str, str], str] = {
    # The automated scorer's word-prefix segmentation only credits a key for matching a
    # target's words from position 0, so it cannot see 'Range' sitting inside 'PRange' -- it
    # matched 'PRange' to 'Glide' by character overlap instead ('prange' / 'glide' share
    # letters, not meaning), which then starved 'VCC.GlideRange' of any candidate at all
    # (verified: every other VCC key's value range is a poor value-fit for GlideRange, so a
    # plain re-run with 'PRange' freed up would not have self-corrected). 'Porta'/'Porta2' are
    # the real portamento-time keys u-he's own naming convention (Porta = "Portamento") points
    # at; confirmed by value range -- both fit Glide/Glide2 tightly, PRange fits GlideRange.
    ("VCC", "Porta"): "VCC.Glide",
    ("VCC", "Porta2"): "VCC.Glide2",
    ("VCC", "PRange"): "VCC.GlideRange",
}

# Contribute zero parameters to D-DIVA-SUBSET (see module docstring); not mapped at all.
_SKIPPED_MODULES = frozenset({"OPT", "Scope1", "PCore"})

# Keys with no real target in DIVA_PARAMETER_NAMES at all, so any admissible candidate they
# get is coincidental (a small numeric range overlapping some unrelated parameter's grid).
# Vc1..Vc8 are VCC's per-voice-count trims, addressable nowhere in this project's parameter
# table; rMW/rPW appear in only 198/1432 presets (the minority key layout), are constant 0.00
# throughout, and their only "fit" was stealing 'VCC.GlideRange' from the real 'PRange' key
# above -- excluding them here is what makes that override land cleanly rather than leaving a
# duplicate target.
EXCLUDED_KEYS = frozenset({
    ("VCC", "Vc1"), ("VCC", "Vc2"), ("VCC", "Vc3"), ("VCC", "Vc4"),
    ("VCC", "Vc5"), ("VCC", "Vc6"), ("VCC", "Vc7"), ("VCC", "Vc8"),
    ("VCC", "rMW"), ("VCC", "rPW"),
})


# --------------------------------------------------------------------------- plugin sweep ---
@dataclass(frozen=True)
class ParamInfo:
    """What one plugin parameter can display, swept once from the live plugin."""
    is_continuous: bool
    minimum: float = 0.0
    maximum: float = 0.0
    texts: Tuple[str, ...] = ()  # discrete only: per-step get_parameter_text(), step order


def sweep_plugin(plugin_path: str) -> Dict[str, ParamInfo]:
    """Every parameter's display grid, keyed by module-qualified name."""
    import dawdreamer as daw

    from synth.plugin_output import suppressed_plugin_output

    info: Dict[str, ParamInfo] = {}
    with suppressed_plugin_output():
        engine = daw.RenderEngine(config.SAMPLE_RATE, config.BUFFER_SIZE)
        plugin = engine.make_plugin_processor("diva", plugin_path)
        descriptions = plugin.get_parameters_description()
        for index, name in enumerate(DIVA_PARAMETER_NAMES):
            description = descriptions[index]
            if description["isDiscrete"] and description["numSteps"] <= 256:
                num_steps = description["numSteps"]
                texts = []
                for step in range(num_steps):
                    value = 0.0 if num_steps == 1 else step / (num_steps - 1)
                    plugin.set_parameter(index, value)
                    texts.append(plugin.get_parameter_text(index))
                info[name] = ParamInfo(is_continuous=False, texts=tuple(texts))
            else:
                info[name] = ParamInfo(
                    is_continuous=True,
                    minimum=float(description["min"]),
                    maximum=float(description["max"]),
                )
    return info


# ------------------------------------------------------------------------------ corpus scan ---
def scan_h2p_library(root: str) -> Tuple[Dict[Tuple[str, str], List[str]], Dict[str, List[str]]]:
    """Every (module, key)'s observed raw values, and each module's key order, across every
    ``.h2p`` file under ``root`` (factory and THIRD PARTY alike -- the map must cover both)."""
    observed: Dict[Tuple[str, str], List[str]] = {}
    order: Dict[str, List[str]] = {}
    for path in sorted(Path(root).rglob("*.h2p")):
        parsed = parse_h2p(str(path))
        for module, keys in parsed.modules.items():
            order.setdefault(module, [])
            for key, value in keys.items():
                if key not in order[module]:
                    order[module].append(key)
                observed.setdefault((module, key), []).append(value)
    return observed, order


# --------------------------------------------------------------------------- name similarity ---
def _words(long_name: str) -> List[str]:
    return re.findall(r"[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+", long_name.replace("-", " "))


def _digits(text: str) -> Tuple[str, ...]:
    return tuple(re.findall(r"\d", text))


def name_score(key: str, long_name: str) -> float:
    """How well a short .h2p key could abbreviate a parameter's long name.

    Word-prefix segmentation: walk the long name's words in order, consuming a prefix of each
    from the key ('KeyFlw' against 'Key','Follow' consumes 'Key'+'Flw'). A full consumption
    scores 1.0 for that component; a difflib ratio is blended in as a smoothing term. Two
    names that reference different digits ('Tune1ModSrc' vs 'TuneModOsc2') are penalized hard,
    and two that reference the same digit are rewarded -- the term that fixes 'TM1On' landing
    on the wrong same-family parameter, see the module docstring.
    """
    key_lower, long_lower = key.lower(), long_name.lower().replace(" ", "")
    if key_lower == long_lower:
        base = 1.0
    else:
        words = [w.lower() for w in _words(long_name)]
        position = matched = 0
        for word in words:
            if position >= len(key_lower):
                break
            span = 0
            while (
                span < len(word)
                and position + span < len(key_lower)
                and key_lower[position + span] == word[span]
            ):
                span += 1
            if span > 0:
                position += span
                matched += 1
        segmentation = 1.0 if (position == len(key_lower) and matched > 0) else position / max(len(key_lower), 1)
        base = 0.65 * segmentation + 0.35 * difflib.SequenceMatcher(None, key_lower, long_lower).ratio()
    key_digits, long_digits = _digits(key), _digits(long_name)
    if key_digits and long_digits:
        base += 0.30 if key_digits == long_digits else -0.45
    elif key_digits != long_digits:
        base -= 0.10
    return base


# ------------------------------------------------------------------------- value compatibility ---
def _as_float(text: str) -> Optional[float]:
    try:
        return float(text)
    except ValueError:
        return None


def compatible_decodings(info: ParamInfo, values: Sequence[str]) -> List[str]:
    """Which decode kinds could represent every one of a key's observed raw values."""
    kinds: List[str] = []
    floats = [_as_float(value) for value in values]
    if info.is_continuous:
        if all(f is not None and info.minimum - 1e-6 <= f <= info.maximum + 1e-6 for f in floats):
            kinds.append("linear")
        return kinds
    text_floats = [_as_float(text) for text in info.texts]
    if all(t is not None for t in text_floats):
        grid = {round(t, 4) for t in text_floats}
        if all(f is not None and round(f, 4) in grid for f in floats):
            kinds.append("grid")
    if all(value in info.texts for value in values):
        kinds.append("label")
    if floats and all(f is not None and f == int(f) for f in floats):
        span = int(max(floats)) - (len(info.texts) - 1)
        if span <= int(min(floats)):
            kinds.append("index")
    return kinds


# ---------------------------------------------------------------------------- assignment ---
@dataclass(frozen=True)
class Assignment:
    module: str
    key: str
    parameter_name: str
    decoding: str
    name_score: float
    runner_up: Optional[str]
    runner_up_margin: Optional[float]


def assign_module(
    module: str,
    keys: Sequence[str],
    parameters: Sequence[str],
    info: Dict[str, ParamInfo],
    observed: Dict[Tuple[str, str], List[str]],
) -> List[Assignment]:
    keys = [key for key in keys if (module, key) not in EXCLUDED_KEYS]
    assignments: List[Assignment] = []

    # Forced pairs are pulled out and removed from both pools *before* Hungarian runs, not
    # redirected after -- redirecting a key's target post-hoc only works if Hungarian picked
    # that key for something in the first place, and the whole reason an override is needed is
    # usually that a better-scoring impostor key won the slot instead (see MANUAL_OVERRIDES'
    # 'PRange' comment). Pulling the pair out first is the only way to actually force it.
    forced = {
        key: MANUAL_OVERRIDES[(module, key)]
        for key in keys
        if (module, key) in MANUAL_OVERRIDES
    }
    for key, parameter in forced.items():
        decodings = compatible_decodings(info[parameter], observed[(module, key)])
        if not decodings:
            raise ValueError(
                f"MANUAL_OVERRIDES[{(module, key)!r}] = {parameter!r} is not value-compatible "
                f"with the observed data -- override is wrong or stale."
            )
        assignments.append(Assignment(
            module=module, key=key, parameter_name=parameter,
            decoding=_choose_decoding(decodings), name_score=name_score(key, plugin_name(parameter)),
            runner_up=None, runner_up_margin=None,
        ))
    keys = [key for key in keys if key not in forced]
    parameters = [p for p in parameters if p not in forced.values()]

    admissible = {
        (key, parameter): compatible_decodings(info[parameter], observed[(module, key)])
        for key in keys
        for parameter in parameters
    }
    usable_keys = [key for key in keys if any(admissible[(key, p)] for p in parameters)]
    if not usable_keys or not parameters:
        return assignments
    cost = np.full((len(usable_keys), len(parameters)), 1e3)
    for i, key in enumerate(usable_keys):
        for j, parameter in enumerate(parameters):
            if admissible[(key, parameter)]:
                position_penalty = 0.15 * abs(
                    i / max(len(usable_keys) - 1, 1) - j / max(len(parameters) - 1, 1)
                )
                cost[i, j] = 1.0 - name_score(key, plugin_name(parameter)) + position_penalty
    rows, cols = linear_sum_assignment(cost)
    for i, j in zip(rows, cols):
        if cost[i, j] >= 1e3:
            continue
        key, parameter = usable_keys[i], parameters[j]
        decodings = admissible[(key, parameter)]
        decoding = _choose_decoding(decodings)
        row_costs = sorted(cost[i, jj] for jj in range(len(parameters)) if cost[i, jj] < 1e3)
        runner_up = runner_up_margin = None
        if len(row_costs) > 1:
            runner_up_margin = round(row_costs[1] - row_costs[0], 3)
            runner_up_index = int(np.argsort(cost[i])[1])
            runner_up = parameters[runner_up_index]
        assignments.append(Assignment(
            module=module, key=key, parameter_name=parameter, decoding=decoding,
            name_score=round(name_score(key, plugin_name(parameter)), 3),
            runner_up=runner_up, runner_up_margin=runner_up_margin,
        ))
    return assignments


def _choose_decoding(decodings: Sequence[str]) -> str:
    """Priority when more than one decoding fits: grid needs no guessing (the .h2p value
    matches a swept display string exactly), label is the non-numeric analogue, index is the
    fallback that requires inferring an offset from the corpus and is used only when the other
    two cannot apply (a non-numeric display with an integer .h2p value, e.g. 'LFO1.Sync')."""
    for preferred in ("linear", "grid", "label", "index"):
        if preferred in decodings:
            return preferred
    raise ValueError(f"No decoding in {decodings!r}")


# --------------------------------------------------------------------------------- map data ---
def build_index_offset(
    module: str, key: str, info: ParamInfo, observed: Dict[Tuple[str, str], List[str]]
) -> int:
    """The offset an 'index' decoding needs: ``plugin_step = h2p_int - offset``.

    The observed corpus range pins a *window* of valid offsets, not always a single value.
    When 0 falls in that window, prefer it: every "grid"/"label" decoding elsewhere in this
    module family stores the plugin's own 0-based step index directly (confirmed for 99 of
    115 discrete parameters, where the window collapses to exactly one point and that point is
    always 0 except the tempo-sync parameters, which use negative offsets for their fixed-time
    options ahead of the synced divisions -- e.g. 'LFO1.Sync' at -2). A window that excludes 0
    still returns its one pinned value, or its lower bound if the corpus never narrows it
    (logged so a human can double check that specific case).
    """
    values = [int(float(v)) for v in observed[(module, key)]]
    cardinality = len(info.texts)
    lower_bound = max(values) - (cardinality - 1)
    upper_bound = min(values)
    if lower_bound > upper_bound:
        raise ValueError(f"{module}.{key}: no offset fits observed range {values}.")
    if lower_bound <= 0 <= upper_bound:
        return 0
    if lower_bound != upper_bound:
        print(f"    AMBIGUOUS OFFSET {module}.{key}: window [{lower_bound}, {upper_bound}] "
              f"does not include 0, defaulting to {lower_bound} -- verify by ear.")
    return lower_bound


def render_map_module(assignments: List[Assignment], info: Dict[str, ParamInfo],
                       observed: Dict[Tuple[str, str], List[str]]) -> str:
    lines = [
        '"""The (module, .h2p key) -> module-qualified parameter name + decoding table.',
        "",
        "Derived by scripts/derive_diva_h2p_map.py from the live plugin's display grids and",
        "every .h2p preset in config.DIVA_PRESETS_PATH (factory + THIRD PARTY, 1432 files).",
        "Committed static data, like synth/diva/parameters.py -- never recomputed at import",
        "time, so a Diva update cannot silently repoint it. Re-run the derivation script and",
        "review its report (name-score margins, resolved offsets) before regenerating.",
        '"""',
        "from typing import Dict, NamedTuple, Optional, Tuple",
        "",
        "",
        "class Decoding(NamedTuple):",
        '    """One (module, key)\'s target parameter and how to read its .h2p value.',
        "",
        "    kind is one of:",
        '      "linear" -- continuous; (minimum, maximum) as Diva displays them.',
        '      "grid"   -- discrete, numeric display; grid holds each step\'s displayed',
        "                  float, in plugin step order; decode matches by value.",
        '      "label"  -- discrete, non-numeric display; grid holds each step\'s exact',
        "                  display text, in plugin step order; decode matches by string.",
        '      "index"  -- discrete, non-numeric display, but .h2p stores a bare integer;',
        "                  offset is the plugin step at .h2p value 0.",
        '    """',
        "    parameter_name: str",
        "    kind: str",
        "    minimum: Optional[float] = None",
        "    maximum: Optional[float] = None",
        "    offset: Optional[int] = None",
        "    grid: Optional[Tuple[str, ...]] = None",
        "",
        "",
        "H2P_PARAMETER_MAP: Dict[Tuple[str, str], Decoding] = {",
    ]
    for a in sorted(assignments, key=lambda a: (a.module, a.key)):
        param_info = info[a.parameter_name]
        if a.decoding == "linear":
            lines.append(
                f"    ({a.module!r}, {a.key!r}): Decoding({a.parameter_name!r}, \"linear\", "
                f"minimum={param_info.minimum!r}, maximum={param_info.maximum!r}),"
            )
        elif a.decoding == "grid":
            grid = tuple(param_info.texts)
            lines.append(
                f"    ({a.module!r}, {a.key!r}): Decoding({a.parameter_name!r}, \"grid\", "
                f"grid={grid!r}),"
            )
        elif a.decoding == "label":
            grid = tuple(param_info.texts)
            lines.append(
                f"    ({a.module!r}, {a.key!r}): Decoding({a.parameter_name!r}, \"label\", "
                f"grid={grid!r}),"
            )
        else:
            offset = build_index_offset(a.module, a.key, param_info, observed)
            lines.append(
                f"    ({a.module!r}, {a.key!r}): Decoding({a.parameter_name!r}, \"index\", "
                f"offset={offset!r}),"
            )
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------- main ---
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-map", action="store_true",
                         help="also (re)write synth/diva/h2p_param_map.py")
    parser.add_argument("--presets", default=config.DIVA_PRESETS_PATH,
                         help="Diva's installed preset library root")
    parser.add_argument("--review-threshold", type=float, default=0.15,
                         help="flag rows whose runner-up is within this cost margin")
    args = parser.parse_args()

    plugin_path = __import__("os").path.expanduser(config.DIVA_PATH)
    print(f"--- Sweeping {plugin_path} ---")
    info = sweep_plugin(plugin_path)

    print(f"--- Scanning {args.presets} ---")
    observed, order = scan_h2p_library(args.presets)
    print(f"    {sum(len(v) for v in observed.values())} (module,key) observations")

    modules: Dict[str, List[str]] = {}
    for name in DIVA_PARAMETER_NAMES:
        module = module_name(name)
        if module in _SKIPPED_MODULES:
            continue
        modules.setdefault(module, []).append(name)

    all_assignments: List[Assignment] = []
    for module, parameters in modules.items():
        all_assignments.extend(
            assign_module(module, order.get(module, []), parameters, info, observed)
        )

    parameter_counts = collections_counter(a.parameter_name for a in all_assignments)
    duplicates = {name: count for name, count in parameter_counts.items() if count > 1}
    if duplicates:
        raise RuntimeError(
            f"Non-bijective map: {duplicates} each claimed by more than one .h2p key. "
            "A MANUAL_OVERRIDE almost certainly needs a matching EXCLUDED_KEYS entry."
        )

    covered = {a.parameter_name for a in all_assignments}
    print(f"\nAssigned {len(all_assignments)} / {len(DIVA_PARAMETER_NAMES)} parameters")
    from synth.diva.subset import SUBSET_PARAM_NAMES
    subset_covered = sum(1 for name in SUBSET_PARAM_NAMES if name in covered)
    print(f"D-DIVA-SUBSET coverage: {subset_covered} / {len(SUBSET_PARAM_NAMES)}")
    unmapped = [n for n in DIVA_PARAMETER_NAMES if n not in covered]
    if unmapped:
        print(f"UNMAPPED ({len(unmapped)}): {unmapped}")

    contested = [a for a in all_assignments if a.runner_up_margin is not None
                 and a.runner_up_margin < args.review_threshold]
    print(f"\nRows needing review (runner-up within {args.review_threshold}): {len(contested)}")
    for a in sorted(contested, key=lambda a: (a.runner_up_margin or 0)):
        print(f"    {a.module}.{a.key:10s} -> {plugin_name(a.parameter_name):20s} "
              f"(score {a.name_score}, runner-up {plugin_name(a.runner_up) if a.runner_up else None}, "
              f"margin {a.runner_up_margin})")

    if args.write_map:
        text = render_map_module(all_assignments, info, observed)
        _MAP_MODULE_PATH.write_text(text)
        print(f"\nWrote {_MAP_MODULE_PATH}")


if __name__ == "__main__":
    main()
