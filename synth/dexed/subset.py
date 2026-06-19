"""
Final Dexed parameter subset (D1 -- LOCKED 2026-06-19, see docs/DECISIONS.md).

The 103 DX7 synthesis parameters the models estimate. The rule: take the
preset-gen-vae / Le Vaillant learnable voice (the full DX7 voice, all six
operators on, all 32 algorithms, master tune and the per-op OP switches fixed)
and drop the parameters that are non-identifiable under the D3 render contract
(a single fixed note, C4, at fixed velocity 100): keyboard scaling (only
revealed across notes) and velocity sensitivity (only revealed across
velocities). See docs/DECISIONS.md (D1) for the rationale.

Every parameter not in this subset is locked at its default (the plugin's
init-patch state) and never estimated: the 6 OP switches (all on), MASTER TUNE
ADJ, and the 42 dropped keyboard-scaling / velocity params.

This module builds the ParameterSpace from a live DexedWrapper; it is swappable
without touching ParameterSpace, DatasetBuilder, or model code.
"""
from typing import List, TYPE_CHECKING

from ..parameter_space import ParameterSpecification, ParameterSpace

if TYPE_CHECKING:
    from .synth import DexedWrapper

# Global timbre / modulation parameters (19): pitch envelope, algorithm,
# feedback, oscillator key sync, the full LFO, pitch-mod sensitivity, transpose.
_GLOBAL_PARAM_NAMES: List[str] = [
    *(f"PITCH EG RATE {i}" for i in range(1, 5)),
    *(f"PITCH EG LEVEL {i}" for i in range(1, 5)),
    "ALGORITHM",
    "FEEDBACK",
    "OSC KEY SYNC",
    "LFO SPEED",
    "LFO DELAY",
    "LFO PM DEPTH",
    "LFO AM DEPTH",
    "LFO KEY SYNC",
    "LFO WAVE",
    "P MODE SENS.",
    "TRANSPOSE",
]

# Per-operator parameters kept for each of the six operators (14 each): the full
# amplitude envelope, oscillator detune, LFO amplitude-mod sensitivity, output
# level, ratio/fixed mode, and coarse/fine frequency. Dropped per operator and
# left at defaults: BREAK POINT, L/R SCALE DEPTH, L/R KEY SCALE, RATE SCALING
# (keyboard scaling, only revealed across notes) and KEY VELOCITY (only revealed
# across velocities) -- non-identifiable at the fixed C4 / velocity-100 render.
_OPERATOR_PARAM_SUFFIXES: List[str] = [
    *(f"EG RATE {i}" for i in range(1, 5)),
    *(f"EG LEVEL {i}" for i in range(1, 5)),
    "OSC DETUNE",
    "A MOD SENS.",
    "OUTPUT LEVEL",
    "MODE",
    "F COARSE",
    "F FINE",
]

# 103 parameters: 19 globals + 14 per operator x 6 operators.
SUBSET_PARAM_NAMES: List[str] = list(_GLOBAL_PARAM_NAMES)
for _op in range(1, 7):
    SUBSET_PARAM_NAMES += [f"OP{_op} {suffix}" for suffix in _OPERATOR_PARAM_SUFFIXES]


def build_parameter_space(synth: "DexedWrapper") -> ParameterSpace:
    """
    Build the D1 ParameterSpace from a live DexedWrapper.

    Options/cardinalities and defaults come from the wrapper (never hard-coded
    indices, per D-NAMING); categorical defaults are snapped to the nearest
    grid option.

    Raises:
        RuntimeError: If a subset name is not exposed by the wrapper.
    """
    available = set(synth.parameter_names)
    missing = [name for name in SUBSET_PARAM_NAMES if name not in available]
    if missing:
        raise RuntimeError(
            f"Subset parameter names not exposed by the wrapper: {missing}. "
            "The plugin build may have changed its parameter naming."
        )

    categoricals = synth.get_categorical_mappings()
    bounds = synth.get_parameter_bounds()
    defaults = synth.get_parameter_defaults()

    parameter_specs: List[ParameterSpecification] = []
    for name in SUBSET_PARAM_NAMES:
        if name in categoricals:
            options = categoricals[name]["options"]
            default = min(options, key=lambda option: abs(option - defaults[name]))
            parameter_specs.append(
                ParameterSpecification(name=name, kind="categorical", options=options, default=default)
            )
        else:
            bound = bounds[name]
            parameter_specs.append(
                ParameterSpecification(
                    name=name,
                    kind="continuous",
                    bounds=(float(bound["min"]), float(bound["max"])),
                    default=float(defaults[name]),
                )
            )
    return ParameterSpace(parameter_specs)
