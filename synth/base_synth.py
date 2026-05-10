import abc
from typing import Dict, Any, Union
import numpy as np

class BaseSynthesizer(abc.ABC):
    """
    Abstract base class for all synthesizer wrappers.
    Provides a unified interface for rendering audio and managing parameters,
    regardless of the underlying VST hosting engine (DawDreamer, Pedalboard, etc.).
    """

    @property
    @abc.abstractmethod
    def sample_rate(self) -> int:
        """The sample rate at which the audio is rendered."""
        pass

    @abc.abstractmethod
    def set_parameters(self, params: Dict[str, Union[float, int]]) -> None:
        """
        Set the synthesizer's parameters.
        Subclasses are responsible for mapping these 'real-world' values
        to the internal normalized formats expected by their engines.
        
        Args:
            params: A dictionary mapping parameter names to their values.
        """
        pass

    @abc.abstractmethod
    def get_parameters(self) -> Dict[str, Union[float, int]]:
        """
        Get the current state of the synthesizer's parameters.
        
        Returns:
            A dictionary of current parameter values.
        """
        pass

    @abc.abstractmethod
    def render_audio(self, midi_note: int, velocity: int, duration_sec: float) -> np.ndarray:
        """
        Render mono audio using the current parameter state.
        
        Args:
            midi_note: The MIDI note number to play (e.g., 60 for Middle C).
            velocity: The MIDI velocity (0-127).
            duration_sec: Duration of the rendered audio in seconds.
            
        Returns:
            A 1D numpy array containing the rendered mono audio waveform (shape: [samples,]).
        """
        pass

    @abc.abstractmethod
    def get_parameter_bounds(self) -> Dict[str, Dict[str, Union[float, int]]]:
        """
        Get the valid bounds and default values for all parameters.
        
        Returns:
            A dictionary where keys are parameter names and values are dicts
            containing 'min', 'max', and 'default'.
        """
        pass

    @abc.abstractmethod
    def get_categorical_mappings(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the definitions for categorical parameters.
        
        Returns:
            A dictionary where keys are categorical parameter names, and values
            are dictionaries containing their valid options or cardinality.
        """
        pass

    def randomize_parameters(self) -> Dict[str, Union[float, int]]:
        """
        Generate a completely random, valid parameter configuration based on 
        the bounds and categorical mappings.
        
        Returns:
            A dictionary of randomized parameters.
        """
        params = {}
        bounds = self.get_parameter_bounds()
        categories = self.get_categorical_mappings()
        
        # Randomize continuous parameters
        for name, bound in bounds.items():
            # If it's a categorical parameter, we handle it separately
            if name in categories:
                continue
            params[name] = np.random.uniform(bound['min'], bound['max'])
            
        # Randomize categorical parameters
        for name, category_data in categories.items():
            options = category_data.get('options', [])
            if options:
                params[name] = np.random.choice(options)
            else:
                # Fallback if categories are defined purely by bounds
                if name in bounds:
                    params[name] = np.random.randint(bounds[name]['min'], bounds[name]['max'] + 1)
                
        return params
