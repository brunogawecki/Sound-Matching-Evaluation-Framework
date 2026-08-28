import numpy as np
from typing import Any, Dict, List, Optional, Union

from ..base_synth import BaseSynthesizer
from ..parameter_space import ParameterSpace
from ..plugin_output import suppressed_plugin_output
from ..renderers import make_renderer
from .parameters import (
    DIVA_DISCRETE_STEPS,
    DIVA_PARAMETER_NAMES,
    build_name_to_index,
    plugin_name,
)

# Diva's index space is renderer-specific. Pedalboard reports 2271 parameters instead of
# DawDreamer's 2362 because it silently drops the 91 parameters whose names collide, which
# shifts every index -- index 280 is already a MIDI CC there, not the last synthesis parameter.
# The name table is written against DawDreamer's indices, so any other renderer would repoint
# the whole parameter space without erroring. Refuse rather than mis-map (D-RENDERER).
_SUPPORTED_RENDERERS = frozenset({"dawdreamer"})

# Not synthesis parameters, so locked at their loaded values and never exposed (D-EXCLUDED).
# 'main.Output' is a master gain: it moves every audio metric without changing the patch.
# 'PCore.LED Colour' is the GUI's LED tint and is silent.
_EXCLUDED_PARAMS = frozenset({"main.Output", "PCore.LED Colour"})


class DivaWrapper(BaseSynthesizer):
    """
    Concrete wrapper for the u-he Diva (subtractive / analog-modelling) synthesizer.
    Expects the Diva.vst3 plugin path.

    Parameters are addressed by **module-qualified** name ('VCF1.Model', 'LFO1.Rate'), because
    Diva's plugin-reported names are not unique -- 56 names are shared by 147 parameters, so a
    bare name does not identify a parameter (D-NAMING as amended, docs/DECISIONS.md). VST3 never
    reports the owning module, so the name->index map is the committed table in parameters.py
    rather than something read off the plugin; it is checked against the live plugin here, and
    construction fails if the plugin has stopped agreeing with it.

    Of Diva's 2362 VST3 parameters only the 281 real ones at indices 0..280 are considered; the
    2080 MIDI CC passthroughs and the trailing 'Program' are outside the table and so are
    invisible above the wrapper, as are the two entries in _EXCLUDED_PARAMS.

    DawDreamer is the only supported renderer (see _SUPPORTED_RENDERERS).

    **Diva must be rendered fresh-process.** Consecutive renders of one identical patch through
    a single wrapper are not reproducible: they agree on energy (~0.2% RMS) but not on waveform
    or per-frame spectrum (~7.9 dB log-spectral distance, as large as Dexed's worst cross-engine
    disagreement). Re-applying the parameter state first does not help, and neither does zeroing
    the OPT slop parameters or OSC.Drift, so this is not the mild state leak Dexed shows. Two
    renders from *separate processes* are bit-identical, so the framework's existing
    fresh-process-at-position-0 discipline satisfies D-REPRO for Diva -- but the in-process
    reuse path does not. See D-DIVA-RENDER in docs/DECISIONS.md.
    """

    synth_name = "diva"

    # D-DIVA-RENDER: the in-process backends must refuse Diva rather than degrade silently.
    supports_in_process_render = False

    def __init__(
        self,
        plugin_path: str,
        sample_rate: int = 22050,
        buffer_size: int = 128,
        renderer: str = "dawdreamer",
    ):
        if renderer not in _SUPPORTED_RENDERERS:
            raise ValueError(
                f"DivaWrapper does not support the '{renderer}' renderer, only "
                f"{sorted(_SUPPORTED_RENDERERS)}. Diva's parameter indices are "
                "renderer-specific and the name table is written against DawDreamer's."
            )

        self._sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.plugin_path = plugin_path

        with suppressed_plugin_output():
            self._renderer = make_renderer(renderer, plugin_path, sample_rate, buffer_size)
            descriptions = self._renderer.parameter_descriptions()

        self._verify_plugin_matches_table(descriptions)

        self._name_to_index: Dict[str, int] = {
            name: index
            for name, index in build_name_to_index().items()
            if name not in _EXCLUDED_PARAMS
        }
        self._param_names: List[str] = sorted(self._name_to_index, key=self._name_to_index.get)

        # Last-applied parameter state, re-applied before every render. This does not make
        # in-process re-renders reproducible for Diva (D-DIVA-RENDER) -- only a fresh process
        # does -- but it keeps the parameter state authoritative and matches DexedWrapper.
        self._current_params: Dict[str, float] = {
            name: self._renderer.get_parameter(index)
            for name, index in self._name_to_index.items()
        }
        # The patch Diva instantiates with, not its JUCE defaultValue field -- the two disagree
        # on 72 parameters, and the loaded patch is what a render actually starts from.
        self._default_params: Dict[str, float] = dict(self._current_params)
        self._parameter_space: Optional[ParameterSpace] = None

    @staticmethod
    def _verify_plugin_matches_table(descriptions: List[Dict[str, Any]]) -> None:
        """Fail loudly if the plugin's parameters have moved out from under the name table."""
        if len(descriptions) < len(DIVA_PARAMETER_NAMES):
            raise RuntimeError(
                f"Diva reported {len(descriptions)} parameters, fewer than the "
                f"{len(DIVA_PARAMETER_NAMES)} the name table describes."
            )
        mismatched = [
            (index, qualified, descriptions[index]["name"])
            for index, qualified in enumerate(DIVA_PARAMETER_NAMES)
            if descriptions[index]["name"] != plugin_name(qualified)
        ]
        if mismatched:
            index, qualified, reported = mismatched[0]
            raise RuntimeError(
                f"Diva's parameters no longer match synth/diva/parameters.py: index {index} is "
                f"'{reported}' but the table expects '{qualified}' "
                f"({len(mismatched)} mismatches). A Diva update most likely inserted or renamed "
                "a parameter; the table must be rebuilt before this plugin can be used."
            )

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def renderer_name(self) -> str:
        """Identifier of the active renderer (always 'dawdreamer' for Diva)."""
        return self._renderer.name

    @property
    def parameter_names(self) -> List[str]:
        """Module-qualified names of the exposed parameters, in plugin index order."""
        return list(self._param_names)

    @property
    def parameter_space(self) -> ParameterSpace:
        """The Diva subset ParameterSpace (D-DIVA-SUBSET, docs/DECISIONS.md), built lazily."""
        if self._parameter_space is None:
            from .subset import build_parameter_space
            self._parameter_space = build_parameter_space(self)
        return self._parameter_space

    def get_parameter_defaults(self) -> Dict[str, float]:
        """Loaded-patch normalized values of the exposed parameters."""
        return dict(self._default_params)

    def set_parameters(self, params: Dict[str, Union[float, int]]) -> None:
        """
        Set parameters by module-qualified name. Values are the raw normalized [0, 1] floats
        the plugin reports.

        Raises:
            KeyError: If a name is unknown, excluded, or not module-qualified.
        """
        unknown = set(params) - set(self._name_to_index)
        if unknown:
            raise KeyError(f"Unknown or excluded parameter names: {sorted(unknown)}")
        for name, value in params.items():
            self._renderer.set_parameter(self._name_to_index[name], float(value))
            self._current_params[name] = float(value)

    def get_parameters(self) -> Dict[str, Union[float, int]]:
        """Read current normalized values of the exposed parameters from the engine."""
        return {
            name: self._renderer.get_parameter(index)
            for name, index in self._name_to_index.items()
        }

    def render_audio(
        self,
        midi_note: int,
        velocity: int,
        duration_sec: float,
        note_duration_sec: Optional[float] = None,
    ) -> np.ndarray:
        """
        Render mono audio with the current parameter state.

        Args:
            midi_note: MIDI note number to play (e.g. 60 for Middle C).
            velocity: MIDI velocity (0-127).
            duration_sec: Total duration of the rendered audio in seconds.
            note_duration_sec: Time from note-on to note-off. Defaults to duration_sec
                (note held for the full render). Use a smaller value to capture the tail.
        """
        for name, value in self._current_params.items():
            self._renderer.set_parameter(self._name_to_index[name], value)

        if note_duration_sec is None:
            note_duration_sec = duration_sec
        note_duration_sec = min(note_duration_sec, duration_sec)

        with suppressed_plugin_output():
            audio = self._renderer.render_note(
                midi_note, velocity, note_duration_sec, duration_sec
            )
        return self._to_mono(audio)

    def render_preset(
        self,
        params: Dict[str, Union[float, int]],
        midi_note: int,
        velocity: int,
        duration_sec: float,
        note_duration_sec: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply one preset's parameters and render it, returning mono audio.

        The analogue of DexedWrapper.render_cartridge_voice without a voice index, since a Diva
        preset is one patch rather than one of 32 in a cartridge. Parameters outside this
        wrapper's exposed set are ignored rather than raising, so a preset carrying all 281
        names (as the Flow Synthesizer corpus does) can be passed through unfiltered.

        This sets the wrapper's parameter state, so a following get_parameters() reflects the
        preset and a plain render_audio() re-renders it.
        """
        self.set_parameters({
            name: value for name, value in params.items() if name in self._name_to_index
        })
        return self.render_audio(midi_note, velocity, duration_sec, note_duration_sec)

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        """Average a (channels, samples) buffer down to 1D mono."""
        if audio.shape[0] >= 2:
            return (audio[0] + audio[1]) / 2.0
        return audio[0]

    def get_parameter_bounds(self) -> Dict[str, Dict[str, Union[float, int]]]:
        """
        Bounds for the continuous exposed parameters. DawDreamer normalizes every continuous
        VST parameter to [0.0, 1.0].
        """
        return {
            name: {"min": 0.0, "max": 1.0, "default": self._default_params[name]}
            for name in self._param_names
            if name not in DIVA_DISCRETE_STEPS
        }

    def get_categorical_mappings(self) -> Dict[str, Dict[str, Any]]:
        """
        Definitions for the exposed parameters the plugin reports as discrete, keyed by
        module-qualified name. Options are the evenly spaced normalized floats in [0, 1].

        This is the plugin's own discreteness, not the D-KIND verdict; the Diva subset
        (D-DIVA-SUBSET) decides which of these are treated as categorical ML-side.
        """
        mappings: Dict[str, Dict[str, Any]] = {}
        for name in self._param_names:
            cardinality = DIVA_DISCRETE_STEPS.get(name)
            if cardinality is None:
                continue
            if cardinality > 1:
                options = [float(step) / (cardinality - 1) for step in range(cardinality)]
            else:
                options = [0.0]
            mappings[name] = {"options": options, "cardinality": cardinality}
        return mappings
