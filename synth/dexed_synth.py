import dawdreamer as daw
import numpy as np
from typing import Dict, Any, Union
from .base_synth import BaseSynthesizer

class DexedWrapper(BaseSynthesizer):
    """
    Concrete wrapper for the Dexed (FM) synthesizer using DawDreamer.
    Expects the Dexed.so or Dexed.vst3 plugin path.
    """

    def __init__(self, plugin_path: str, sample_rate: int = 22050, buffer_size: int = 128):
        self._sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.plugin_path = plugin_path
        
        self.engine = daw.RenderEngine(self._sample_rate, self.buffer_size)
        self.synth = self.engine.make_plugin_processor("dexed", self.plugin_path)
        self.engine.load_graph([(self.synth, [])])
        
        # Dexed has 155 parameters
        self._num_params = self.synth.get_plugin_parameter_size()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def set_parameters(self, params: Dict[str, Union[float, int]]) -> None:
        """
        Sets parameters in the DawDreamer synth. 
        DawDreamer expects all values to be normalized [0, 1].
        """
        for param_name, value in params.items():
            param_idx = int(param_name)
            self.synth.set_parameter(param_idx, float(value))

    def get_parameters(self) -> Dict[str, Union[float, int]]:
        """Reads current normalized parameters from the engine."""
        params = {}
        for i in range(self._num_params):
            params[str(i)] = self.synth.get_parameter(i)
        return params

    def render_audio(self, midi_note: int, velocity: int, duration_sec: float) -> np.ndarray:
        """
        Renders the audio via DawDreamer and returns mono audio.
        """
        # Clear any previous midi
        self.synth.clear_midi()
        
        # Add note at start (0 sec) for the given duration
        # We render slightly longer (e.g. +0.5s) to capture release tail, 
        # but let's strictly follow the requested duration_sec for the note and render.
        self.synth.add_midi_note(midi_note, velocity, 0.0, duration_sec)
        
        # Render
        self.engine.render(duration_sec)
        audio = self.engine.get_audio()
        
        # Convert stereo to mono
        if audio.shape[0] >= 2:
            audio_mono = (audio[0] + audio[1]) / 2.0
        else:
            audio_mono = audio[0]
            
        return audio_mono

    def get_parameter_bounds(self) -> Dict[str, Dict[str, Union[float, int]]]:
        """
        DawDreamer normalizes all continuous VST parameters to [0.0, 1.0].
        """
        bounds = {}
        categorical_indices = [int(k) for k in self.get_categorical_mappings().keys()]
        
        for i in range(self._num_params):
            # We skip adding bounds for categoricals if we want randomization to 
            # only use options, but for completeness, everything is 0.0 to 1.0
            if i not in categorical_indices:
                bounds[str(i)] = {'min': 0.0, 'max': 1.0, 'default': 0.5}
                
        return bounds

    def get_categorical_mappings(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns categorical mappings based on Dexed's specific architecture.
        Values are generated as normalized float options [0, 1] for DawDreamer.
        """
        mappings = {}
        
        def add_mapping(index: int, cardinality: int):
            # Generate evenly spaced options from 0.0 to 1.0
            if cardinality > 1:
                options = [float(n) / (cardinality - 1) for n in range(cardinality)]
            else:
                options = [0.0]
            mappings[str(index)] = {'options': options, 'cardinality': cardinality}

        # Algorithm
        add_mapping(4, 32)
        # OSC Key Sync
        add_mapping(6, 2)
        # LFO Key Sync
        add_mapping(11, 2)
        # LFO Wave
        add_mapping(12, 6)

        # Operator specific categoricals (6 operators, 22 params each, starting at index 23)
        for i in range(6):
            op_offset = 22 * i
            # OP Mode (ratio or fixed)
            add_mapping(32 + op_offset, 2)
            # L Scale
            add_mapping(39 + op_offset, 4)
            # R Scale
            add_mapping(40 + op_offset, 4)
            # OP On/Off Switch
            add_mapping(44 + op_offset, 2)

        return mappings
