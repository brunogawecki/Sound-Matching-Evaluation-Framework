import abc
from typing import Any, Dict, List

import numpy as np


class Renderer(abc.ABC):
    """
    Engine-specific rendering layer beneath the synthesizer wrappers.

    A Renderer hides one VST-hosting engine (DawDreamer, Pedalboard, ...) behind a
    minimal surface: enumerate the plugin's parameters, get/set a single parameter by index
    (raw normalized [0, 1], as the plugin reports it), and render one held MIDI note to a raw
    multichannel buffer. All engine-agnostic logic -- name<->index resolution, parameter
    exclusions, categorical handling, the ParameterSpace, and mono conversion -- lives in the
    wrapper above (e.g. DexedWrapper), so a wrapper works with any renderer unchanged.

    The choice of renderer is recorded per dataset/evaluation run: renderers must never be mixed
    within a single run, because the render-reproducibility contract (D-REPRO in
    docs/DECISIONS.md) holds per engine, not across engines.
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short renderer identifier (e.g. 'dawdreamer', 'pedalboard') for run metadata."""
        ...

    @property
    @abc.abstractmethod
    def sample_rate(self) -> int:
        """The sample rate at which this renderer renders."""
        ...

    @abc.abstractmethod
    def parameter_descriptions(self) -> List[Dict[str, Any]]:
        """
        Describe every parameter the plugin exposes, in plugin index order.

        Returns:
            A list of dicts, each with at least 'index' (int) and 'name' (str, the
            plugin-reported parameter name). Mirrors DawDreamer's
            get_parameters_description() so the wrapper's name resolution is engine-agnostic.
        """
        ...

    @abc.abstractmethod
    def get_parameter(self, index: int) -> float:
        """Read the current raw normalized [0, 1] value of the parameter at this index."""
        ...

    @abc.abstractmethod
    def set_parameter(self, index: int, value: float) -> None:
        """Set the raw normalized [0, 1] value of the parameter at this index."""
        ...

    @abc.abstractmethod
    def render_note(
        self,
        midi_note: int,
        velocity: int,
        note_duration_sec: float,
        total_duration_sec: float,
    ) -> np.ndarray:
        """
        Render a single held MIDI note with the current parameter state.

        The note sounds at time 0 and is released at note_duration_sec; total_duration_sec
        of audio is rendered so a release tail can be captured.

        Args:
            midi_note: MIDI note number to play (e.g. 60 for Middle C).
            velocity: MIDI velocity (0-127).
            note_duration_sec: Time from note-on to note-off.
            total_duration_sec: Total length of rendered audio.

        Returns:
            A raw 2D float array shaped (channels, samples). The wrapper converts to mono.
        """
        ...
