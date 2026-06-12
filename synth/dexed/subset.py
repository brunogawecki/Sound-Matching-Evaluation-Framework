"""
PROVISIONAL Dexed parameter subset.

D1 (the final estimated subset) is deferred -- see docs/DECISIONS.md. This module
defines the provisional development subset used to build ParameterSpace, DatasetBuilder,
and model code; it is swappable without touching any of those layers. Do NOT
generate the real training dataset from this subset until D1 is locked.

All parameters not in the subset are locked at their defaults (the plugin's
init-patch state) and never estimated.
"""
from typing import List, TYPE_CHECKING

from ..parameter_space import ParameterSpecification, ParameterSpace

if TYPE_CHECKING:
    from .synth import DexedWrapper

# 29 parameters: global timbre/modulation levers plus, per operator, the loudest
# audible controls (level, coarse ratio, attack and release rates).
SUBSET_PARAM_NAMES: List[str] = [
    "ALGORITHM",
    "FEEDBACK",
    "LFO SPEED",
    "LFO PM DEPTH",
    "LFO WAVE",
]
for _i in range(1, 7):
    SUBSET_PARAM_NAMES += [
        f"OP{_i} OUTPUT LEVEL",
        f"OP{_i} F COARSE",
        f"OP{_i} EG RATE 1",
        f"OP{_i} EG RATE 4",
    ]


def build_parameter_space(synth: "DexedWrapper") -> ParameterSpace:
    """
    Build the provisional ParameterSpace from a live DexedWrapper.

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
