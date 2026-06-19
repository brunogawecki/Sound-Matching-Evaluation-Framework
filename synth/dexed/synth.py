import numpy as np
from typing import Dict, Any, List, Optional, Union
from ..base_synth import BaseSynthesizer
from ..parameter_space import ParameterSpace
from ..renderers import make_renderer

# VST-level parameters that are not DX7 synthesis parameters. They are locked at
# plugin defaults and never exposed: randomizing 'Bypass' mutes the output and
# 'Program' loads a different patch entirely.
_EXCLUDED_PARAMS = {"Cutoff", "Resonance", "Output", "MonoMode", "Bypass", "Program"}

# JUCE exposes 2080 MIDI CC passthrough parameters (16 channels x 130) after the
# real plugin parameters; they are excluded by this prefix.
_MIDI_CC_PREFIX = "MIDI CC"

# Categorical synthesis parameters, keyed by plugin-reported name (verified against
# the live plugin -- the VST3 build's indices differ from the classic Dexed layout,
# so indices must never be hard-coded).
_CATEGORICAL_CARDINALITIES: Dict[str, int] = {
    "ALGORITHM": 32,
    "OSC KEY SYNC": 2,
    "LFO KEY SYNC": 2,
    "LFO WAVE": 6,
}
for i in range(1, 7):
    _CATEGORICAL_CARDINALITIES[f"OP{i} MODE"] = 2
    _CATEGORICAL_CARDINALITIES[f"OP{i} L KEY SCALE"] = 4
    _CATEGORICAL_CARDINALITIES[f"OP{i} R KEY SCALE"] = 4
    _CATEGORICAL_CARDINALITIES[f"OP{i} SWITCH"] = 2
    # F COARSE is ordered but perceptually discontinuous (one step can double the
    # operator frequency), and Dexed quantizes it internally to 32 values while
    # reading back the raw float -- grid points are the only honest representation
    # (D-KIND in docs/DECISIONS.md).
    _CATEGORICAL_CARDINALITIES[f"OP{i} F COARSE"] = 32


class DexedWrapper(BaseSynthesizer):
    """
    Concrete wrapper for the Dexed (FM) synthesizer using DawDreamer.
    Expects the Dexed.so or Dexed.vst3 plugin path.

    Parameters are addressed by their plugin-reported name (e.g. 'ALGORITHM',
    'OP1 OUTPUT LEVEL'); the name->index map is resolved from the live plugin at
    construction time. Only the 152 DX7 synthesis parameters are exposed.

    The VST-hosting engine is pluggable via `renderer` ('dawdreamer' default, or 'pedalboard');
    all logic here is engine-agnostic and delegates the engine-specific work to a Renderer.
    Renderers must not be mixed within one dataset/eval run (D-REPRO, docs/DECISIONS.md).
    """

    def __init__(
        self,
        plugin_path: str,
        sample_rate: int = 22050,
        buffer_size: int = 128,
        renderer: str = "dawdreamer",
    ):
        self._sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.plugin_path = plugin_path

        self._renderer = make_renderer(renderer, plugin_path, sample_rate, buffer_size)

        # Resolve the exposed parameter universe by name from the live plugin.
        self._name_to_index: Dict[str, int] = {}
        for desc in self._renderer.parameter_descriptions():
            name = desc["name"]
            if name in _EXCLUDED_PARAMS or name.startswith(_MIDI_CC_PREFIX):
                continue
            self._name_to_index[name] = desc["index"]
        self._param_names: List[str] = sorted(self._name_to_index, key=self._name_to_index.get)

        unknown_categoricals = set(_CATEGORICAL_CARDINALITIES) - set(self._name_to_index)
        if unknown_categoricals:
            raise RuntimeError(
                f"Categorical parameter names not found in plugin: {unknown_categoricals}. "
                "The plugin build may have changed its parameter naming."
            )

        # Last-applied parameter state, re-applied before every render so that
        # rendering is bit-reproducible (engine state leaks between renders otherwise).
        self._current_params: Dict[str, float] = {
            name: self._renderer.get_parameter(idx)
            for name, idx in self._name_to_index.items()
        }
        # The plugin's JUCE defaultValue field is 0.0 for every parameter in this
        # build; the freshly-loaded init-patch state is the real default.
        self._default_params: Dict[str, float] = dict(self._current_params)
        self._parameter_space: Optional[ParameterSpace] = None

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def renderer_name(self) -> str:
        """Identifier of the active renderer ('dawdreamer' / 'pedalboard')."""
        return self._renderer.name

    @property
    def parameter_names(self) -> List[str]:
        """Names of the exposed synthesis parameters, in plugin index order."""
        return list(self._param_names)

    @property
    def parameter_space(self) -> ParameterSpace:
        """The D1 Dexed subset ParameterSpace (LOCKED, see docs/DECISIONS.md), built lazily."""
        if self._parameter_space is None:
            from .subset import build_parameter_space
            self._parameter_space = build_parameter_space(self)
        return self._parameter_space

    def get_parameter_defaults(self) -> Dict[str, float]:
        """Default (init-patch) normalized values of the exposed parameters."""
        return dict(self._default_params)

    def set_parameters(self, params: Dict[str, Union[float, int]]) -> None:
        """
        Sets parameters in the DawDreamer synth.
        DawDreamer expects all values to be normalized [0, 1].

        Raises:
            KeyError: If a parameter name is unknown or not exposed by this wrapper.
        """
        unknown = set(params) - set(self._name_to_index)
        if unknown:
            raise KeyError(f"Unknown or excluded parameter names: {sorted(unknown)}")
        for name, value in params.items():
            self._renderer.set_parameter(self._name_to_index[name], float(value))
            self._current_params[name] = float(value)

    def get_parameters(self) -> Dict[str, Union[float, int]]:
        """Reads current normalized values of the exposed parameters from the engine."""
        return {
            name: self._renderer.get_parameter(idx)
            for name, idx in self._name_to_index.items()
        }

    def render_audio(
        self,
        midi_note: int,
        velocity: int,
        duration_sec: float,
        note_duration_sec: Optional[float] = None,
    ) -> np.ndarray:
        """
        Renders the audio via DawDreamer and returns mono audio.

        Args:
            midi_note: The MIDI note number to play (e.g., 60 for Middle C).
            velocity: The MIDI velocity (0-127).
            duration_sec: Total duration of the rendered audio in seconds.
            note_duration_sec: Time from note-on to note-off. Defaults to
                duration_sec (note held for the full render). Use a smaller value
                to capture the release tail.
        """
        # Re-apply the current parameter state: without this, engine state (LFO
        # phase etc.) leaks between renders and re-renders are not bit-identical.
        for name, value in self._current_params.items():
            self._renderer.set_parameter(self._name_to_index[name], value)

        if note_duration_sec is None:
            note_duration_sec = duration_sec
        note_duration_sec = min(note_duration_sec, duration_sec)

        audio = self._renderer.render_note(midi_note, velocity, note_duration_sec, duration_sec)
        return self._to_mono(audio)

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        """Average a (channels, samples) buffer down to 1D mono."""
        if audio.shape[0] >= 2:
            return (audio[0] + audio[1]) / 2.0
        return audio[0]

    def render_cartridge_voice(
        self,
        syx_path: str,
        voice_index: int,
        midi_note: int,
        velocity: int,
        duration_sec: float,
        note_duration_sec: Optional[float] = None,
    ) -> np.ndarray:
        """
        Render one voice of a DX7 32-voice .syx cartridge and return mono audio.

        The voice is parsed out of the cartridge and applied as Dexed parameters
        (Dexed ignores SysEx and MIDI Program Change in offline rendering, so the
        cartridge cannot be loaded any other way). This sets the wrapper's
        parameter state, so a following get_parameters() reflects the voice and a
        plain render_audio() re-renders it.

        Args:
            syx_path: Path to a 4104-byte DX7 32-voice bulk dump (.syx).
            voice_index: Which voice of the cartridge to play, in [0, 31].
            midi_note: MIDI note number to play (e.g. 60 for Middle C).
            velocity: MIDI velocity (0-127).
            duration_sec: Total duration of the rendered audio in seconds.
            note_duration_sec: Time from note-on to note-off. Defaults to
                duration_sec (note held for the full render).

        Raises:
            ValueError: If the file is not a 32-voice bulk dump or voice_index
                is outside [0, 31].
        """
        from .cartridge import voice_parameters

        with open(syx_path, "rb") as syx_file:
            cartridge_bytes = syx_file.read()

        self.set_parameters(voice_parameters(cartridge_bytes, voice_index))
        return self.render_audio(midi_note, velocity, duration_sec, note_duration_sec)

    def get_parameter_bounds(self) -> Dict[str, Dict[str, Union[float, int]]]:
        """
        Bounds for the continuous exposed parameters.
        DawDreamer normalizes all continuous VST parameters to [0.0, 1.0].
        """
        return {
            name: {"min": 0.0, "max": 1.0, "default": self._default_params[name]}
            for name in self._param_names
            if name not in _CATEGORICAL_CARDINALITIES
        }

    def get_categorical_mappings(self) -> Dict[str, Dict[str, Any]]:
        """
        Categorical parameter definitions, keyed by parameter name.
        Options are the evenly spaced normalized floats [0, 1] DawDreamer expects.
        """
        mappings: Dict[str, Dict[str, Any]] = {}
        for name, cardinality in _CATEGORICAL_CARDINALITIES.items():
            if cardinality > 1:
                options = [float(n) / (cardinality - 1) for n in range(cardinality)]
            else:
                options = [0.0]
            mappings[name] = {"options": options, "cardinality": cardinality}
        return mappings
