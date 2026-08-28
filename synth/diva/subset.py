"""
Diva parameter subset (D-DIVA-SUBSET, see docs/DECISIONS.md).

The Diva synthesis parameters the models estimate. The rule is D1's, unchanged: keep the
learnable voice, drop what is non-identifiable under the D3 render contract (one fixed note,
C4, at fixed velocity 100, rendered to mono). Diva's non-identifiable set is larger and less
obvious than Dexed's, so every drop below was measured on the live plugin rather than assumed;
the measurements are recorded under D-DIVA-SUBSET.

Every parameter not in this subset is locked at its default (the plugin's freshly-loaded patch
state) and never estimated.

This module builds the ParameterSpace from a live DivaWrapper; it is swappable without touching
ParameterSpace, DatasetBuilder, or model code.
"""
from typing import Dict, List, TYPE_CHECKING

from ..parameter_space import ParameterSpecification, ParameterSpace
from .parameters import DIVA_DISCRETE_STEPS, DIVA_PARAMETER_NAMES, module_name

if TYPE_CHECKING:
    from .synth import DivaWrapper

# The two parameters the wrapper already hides (main.Output is a master gain outside the patch,
# PCore.LED Colour is a GUI tint). Listed so this module's arithmetic is checkable on its own.
_WRAPPER_EXCLUDED: List[str] = ["main.Output", "PCore.LED Colour"]

# Modules dropped whole.
#
# OPT is Diva's "quality and analog slop" panel: Accuracy / OfflineAcc are CPU-versus-fidelity
# render settings that must stay fixed for a comparable corpus (D-DIVA-RENDER's bit-identity
# rests on them), the five *Slop knobs scale pseudo-random per-voice drift whose audible
# realization is a property of our render process rather than of the patch, and V1Mod..V8Mod
# are per-voice modulation offsets. Scope1 drives the on-screen oscilloscope and produces
# bit-identical audio at every setting (measured).
_DROPPED_MODULES: List[str] = ["OPT", "Scope1"]

# Voice allocation and inter-note behaviour. A single held note cannot reveal any of it:
# polyphony count, keyboard mode, note priority, glide (needs a second note to glide from),
# or pitch-bend range (no bend is sent). TuningMode selects a microtuning table, none loaded.
# MultiCore is a threading switch that must stay off for reproducible rendering. FineTuneCents
# is Diva's master-tune knob, the analogue of the MASTER TUNE ADJ that D1 drops for Dexed.
# Voice Stack and Transpose are the two VCC parameters that survive and are kept below.
_DROPPED_VOICE_PARAMS: List[str] = [
    "VCC.Voices",
    "VCC.Mode",
    "VCC.GlideMode",
    "VCC.Glide",
    "VCC.Glide2",
    "VCC.GlideRange",
    "VCC.PitchBend Up",
    "VCC.PitchBend Down",
    "VCC.TuningMode",
    "VCC.FineTuneCents",
    "VCC.Note Priority",
    "VCC.MultiCore",
]

# Key-follow and velocity depth: D1's dropped class, under Diva's names. At one fixed note and
# velocity each of these collapses to a constant offset on the thing it scales, so it is
# confounded with that parameter (measured: VCF1.KeyFollow's whole range is worth 2.8 dB LSD,
# most of which a ~0.02 shift in VCF1.Frequency reproduces).
_DROPPED_KEY_VELOCITY_PARAMS: List[str] = [
    "ENV1.Velocity",
    "ENV1.KeyFollow",
    "ENV2.Velocity",
    "ENV2.KeyFollow",
    "HPF.KeyFollow",
    "VCF1.KeyFollow",
]

# Stereo placement. The render contract is mono (BaseSynthesizer.render_audio returns a 1D
# array), so pan survives only as the pan law's level trim, and it is symmetric about centre:
# pan 0.25 and pan 0.75 render identically (measured, 0.3338 max-diff / 3.1 dB LSD against
# centre for both). That is a two-to-one map onto a level change VCA1.Volume already covers.
_DROPPED_PAN_PARAMS: List[str] = [
    "VCA1.Pan",
    "VCA1.PanModulation",
    "VCA1.PanModDepth",
]

# Parameters that select a MIDI controller or reorder held notes. Neither exists under D3: no
# CC, wheel or aftertouch is sent, and the arpeggiator has exactly one note to arpeggiate. All
# options of each render bit-identically (measured).
_DROPPED_CONTROLLER_PARAMS: List[str] = [
    "Rtary1.Controller",
    "Rtary2.Controller",
    "ARP.Direction",
    "ARP.Order",
]

_DROPPED_PARAM_NAMES: List[str] = (
    _WRAPPER_EXCLUDED
    + _DROPPED_VOICE_PARAMS
    + _DROPPED_KEY_VELOCITY_PARAMS
    + _DROPPED_PAN_PARAMS
    + _DROPPED_CONTROLLER_PARAMS
)

SUBSET_PARAM_NAMES: List[str] = [
    name
    for name in DIVA_PARAMETER_NAMES
    if module_name(name) not in _DROPPED_MODULES and name not in _DROPPED_PARAM_NAMES
]


def build_parameter_space(synth: "DivaWrapper") -> ParameterSpace:
    """
    Build the D-DIVA-SUBSET ParameterSpace from a live DivaWrapper.

    Kind follows the plugin's own discreteness: Diva reports 117 parameters as stepped, and
    every one of them is a mode, model, waveform, switch or source selector whose adjacent
    steps are perceptually discontinuous, so D-KIND makes them categorical. Diva has no
    stepped-but-smooth parameter of the 0-99 level kind, so no exception list is needed.
    Options, bounds and defaults come from the wrapper, never from hard-coded indices
    (D-NAMING); categorical defaults are snapped to the nearest grid option.

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
