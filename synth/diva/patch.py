"""Diva ``.h2p`` preset format: parse and decode.

A ``.h2p`` file is plain text: a ``/*@Meta ... */`` header (bank, author, categories -- no
patch name; that comes from the filename), then ``#cm=<Module>`` sections of ``Key=Value``
lines, then a ``// Section for ugly compressed binary Data`` tail that is *not* the patch --
the plain text above it is authoritative (verified: two files differing only in that tail
section still parse to the same parameters). Keys are Diva's own short codes, scoped to their
module (``ENV1.Atk``, not ``ENV1.Attack``); :mod:`synth.diva.h2p_param_map` resolves them onto
the module-qualified names the rest of the framework addresses parameters by.

Two format facts drove the map's decode kinds (see that module's docstring): every continuous
parameter displays exactly linearly, but discrete ones split three ways depending on whether
the plugin's own display text is numeric, a label, or (rarely) a label the ``.h2p`` value
can't reach at all -- it stores a bare integer offset instead (``LFO1.Sync``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .h2p_param_map import H2P_PARAMETER_MAP
from .parameters import DIVA_DISCRETE_STEPS

_BINARY_MARKER = "// Section for ugly compressed binary Data"


@dataclass(frozen=True)
class ParsedPatch:
    """One ``.h2p`` file's plain-text content, before parameter decoding."""
    modules: Dict[str, Dict[str, str]]


def parse_h2p(path: str) -> ParsedPatch:
    """Read one ``.h2p`` file's ``#cm=`` sections, stopping at the binary tail.

    Opened as ``latin-1``: ``.h2p`` carries no encoding declaration and THIRD PARTY author
    fields include non-ASCII bytes; ``latin-1`` never raises, and every byte a parameter value
    actually uses is plain ASCII, so decoding is unaffected either way.
    """
    modules: Dict[str, Dict[str, str]] = {}
    current_module = None
    with open(path, encoding="latin-1") as handle:
        for line in handle:
            line = line.rstrip("\r\n")
            if line.startswith(_BINARY_MARKER):
                break
            if line.startswith("#cm="):
                current_module = line[len("#cm="):]
                modules.setdefault(current_module, {})
                continue
            if current_module is None or "=" not in line or line.startswith("//"):
                continue
            key, value = line.split("=", 1)
            # A handful of discrete values are written quoted ('Chorus1'); every other value
            # is bare, so stripping is a no-op for them.
            modules[current_module][key] = value.strip().strip("'")
    return ParsedPatch(modules=modules)


def _decode(decoding, raw_value: str) -> float:
    if decoding.kind == "linear":
        return (float(raw_value) - decoding.minimum) / (decoding.maximum - decoding.minimum)
    if decoding.kind == "grid":
        target = round(float(raw_value), 4)
        for step, text in enumerate(decoding.grid):
            if round(float(text), 4) == target:
                return step / (len(decoding.grid) - 1)
        raise ValueError(
            f"{decoding.parameter_name}: {raw_value!r} matches no step in {decoding.grid!r}."
        )
    if decoding.kind == "label":
        if raw_value not in decoding.grid:
            raise ValueError(
                f"{decoding.parameter_name}: {raw_value!r} is not one of {decoding.grid!r}."
            )
        return decoding.grid.index(raw_value) / (len(decoding.grid) - 1)
    if decoding.kind == "index":
        cardinality = DIVA_DISCRETE_STEPS[decoding.parameter_name]
        step = int(round(float(raw_value))) - decoding.offset
        return step / (cardinality - 1) if cardinality > 1 else 0.0
    raise ValueError(f"Unknown decoding kind {decoding.kind!r} for {decoding.parameter_name}.")


def patch_parameters(parsed: ParsedPatch) -> Dict[str, float]:
    """Decode a parsed ``.h2p`` patch into module-qualified names -> normalized ``[0, 1]``.

    A key with no entry in :data:`H2P_PARAMETER_MAP` is ignored -- ``.h2p`` carries a few
    per-preset extras with no synthesis meaning (``PSong``, ``rMW``, ``rPW``). A parameter the
    map expects but this preset does not have raises, since every ``.h2p`` file this map was
    derived from carries the same 28 modules and (module, key) universe; a preset missing one
    means the format has drifted from what was derived, not that the parameter is optional.
    """
    values: Dict[str, float] = {}
    for module, keys in parsed.modules.items():
        for key, raw_value in keys.items():
            decoding = H2P_PARAMETER_MAP.get((module, key))
            if decoding is None:
                continue
            values[decoding.parameter_name] = _decode(decoding, raw_value)
    missing = sorted(
        decoding.parameter_name
        for decoding in H2P_PARAMETER_MAP.values()
        if decoding.parameter_name not in values
    )
    if missing:
        raise KeyError(f"Preset is missing mapped parameters: {missing}")
    return values
