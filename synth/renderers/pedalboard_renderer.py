import numpy as np
from typing import Any, Dict, List

from .base import Renderer

# MIDI status bytes (channel 0). Pedalboard accepts raw byte tuples, so no `mido` dependency
# is required to drive a note.
_NOTE_ON = 0x90
_NOTE_OFF = 0x80


class PedalboardRenderer(Renderer):
    """
    Renderer backed by Pedalboard (https://github.com/spotify/pedalboard).

    Loads the plugin as a software instrument and renders MIDI to audio in a single
    `process()` call. Parameters are driven by their raw normalized [0, 1] value, matching the
    synth-side representation the wrapper uses. Provided as a secondary renderer for the
    host-robustness / render-speed comparison; DawDreamer remains the default.

    `pedalboard` is imported lazily so the default DawDreamer path never requires it installed.
    """

    def __init__(self, plugin_path: str, sample_rate: int):
        import pedalboard

        self._sample_rate = sample_rate
        self._plugin_path = plugin_path
        self._plugin = pedalboard.load_plugin(plugin_path)
        if not getattr(self._plugin, "is_instrument", False):
            raise ValueError(
                f"Plugin at {plugin_path} is not an instrument; Pedalboard cannot render MIDI through it."
            )

        # Parameter order defines the index space used by get/set_parameter. Each value is an
        # AudioProcessorParameter whose `.name` is the original plugin-reported name (the dict
        # key is a python-safe alias) and whose `.raw_value` is the [0, 1] normalized value.
        self._parameters = list(self._plugin.parameters.values())

    @property
    def name(self) -> str:
        return "pedalboard"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def parameter_descriptions(self) -> List[Dict[str, Any]]:
        return [
            {"index": index, "name": parameter.name}
            for index, parameter in enumerate(self._parameters)
        ]

    def get_parameter(self, index: int) -> float:
        return float(self._parameters[index].raw_value)

    def set_parameter(self, index: int, value: float) -> None:
        self._parameters[index].raw_value = float(value)

    def render_note(
        self,
        midi_note: int,
        velocity: int,
        note_duration_sec: float,
        total_duration_sec: float,
    ) -> np.ndarray:
        midi_messages = [
            ([_NOTE_ON, int(midi_note), int(velocity)], 0.0),
            ([_NOTE_OFF, int(midi_note), 0], float(note_duration_sec)),
        ]
        audio = self._plugin.process(
            midi_messages,
            float(total_duration_sec),
            float(self._sample_rate),
            num_channels=2,
            reset=True,
        )
        return self._to_channels_first(np.asarray(audio, dtype=np.float64))

    @staticmethod
    def _to_channels_first(audio: np.ndarray) -> np.ndarray:
        """Normalize a Pedalboard buffer to (channels, samples) -- channels is the smaller axis."""
        if audio.ndim == 1:
            return audio[np.newaxis, :]
        return audio if audio.shape[0] <= audio.shape[1] else audio.T
