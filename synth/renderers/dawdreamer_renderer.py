import dawdreamer as daw
import numpy as np
from typing import Any, Dict, List

from .base import Renderer


class DawDreamerRenderer(Renderer):
    """
    Renderer backed by DawDreamer (https://github.com/DBraun/DawDreamer).

    Hosts the plugin in a single-processor DawDreamer RenderEngine graph. This is the
    framework's default renderer and the one all existing reproducibility characterization
    (D-REPRO) was performed on.
    """

    def __init__(self, plugin_path: str, sample_rate: int, buffer_size: int):
        self._sample_rate = sample_rate
        self._buffer_size = buffer_size
        self._plugin_path = plugin_path

        self._engine = daw.RenderEngine(sample_rate, buffer_size)
        self._synth = self._engine.make_plugin_processor("dexed", plugin_path)
        self._engine.load_graph([(self._synth, [])])

    @property
    def name(self) -> str:
        return "dawdreamer"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def parameter_descriptions(self) -> List[Dict[str, Any]]:
        return self._synth.get_parameters_description()

    def get_parameter(self, index: int) -> float:
        return self._synth.get_parameter(index)

    def set_parameter(self, index: int, value: float) -> None:
        self._synth.set_parameter(index, float(value))

    def render_note(
        self,
        midi_note: int,
        velocity: int,
        note_duration_sec: float,
        total_duration_sec: float,
    ) -> np.ndarray:
        self._synth.clear_midi()
        self._synth.add_midi_note(midi_note, velocity, 0.0, note_duration_sec)
        self._engine.render(total_duration_sec)
        return self._engine.get_audio()
