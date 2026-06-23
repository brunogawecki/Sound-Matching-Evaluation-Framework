import abc
from typing import Dict, Any, Optional, Tuple, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .parameter_space import ParameterSpace

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

    @property
    def parameter_space(self) -> "ParameterSpace":
        """
        The canonical ParameterSpace over this synth's estimated parameter subset.
        Subclasses wire this to their subset definition (e.g. synth/dexed_subset.py).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not define a parameter subset / ParameterSpace."
        )

    @property
    def audible_sampling_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Per-parameter ``(low, high)`` sub-ranges that keep synthetic draws audible.

        Consumed by ParameterSpace.sample_constrained. Default is empty (no
        constraint); synths whose uniform draws are mostly silent override this
        to pin a carrier loud (see docs/DECISIONS.md D-AUDIBLE).
        """
        return {}

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
    def get_parameter_defaults(self) -> Dict[str, float]:
        """
        Get the default (init-patch) normalized value of every exposed parameter.

        Returns:
            A dictionary mapping parameter names to their default values. The
            DatasetBuilder locks non-subset parameters to these defaults.
        """
        pass

    @abc.abstractmethod
    def render_audio(
        self,
        midi_note: int,
        velocity: int,
        duration_sec: float,
        note_duration_sec: Optional[float] = None,
    ) -> np.ndarray:
        """
        Render mono audio using the current parameter state.

        Args:
            midi_note: The MIDI note number to play (e.g., 60 for Middle C).
            velocity: The MIDI velocity (0-127).
            duration_sec: Total duration of the rendered audio in seconds.
            note_duration_sec: Time from note-on to note-off; defaults to
                duration_sec (note held for the full render).

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

    def randomize_parameters(
        self, rng: Optional[np.random.Generator] = None
    ) -> Dict[str, Union[float, int]]:
        """
        Generate a completely random, valid parameter configuration based on
        the bounds and categorical mappings.

        Args:
            rng: Seedable random generator for reproducible sampling.
                Defaults to a fresh unseeded generator.

        Returns:
            A dictionary of randomized parameters.
        """
        if rng is None:
            rng = np.random.default_rng()

        params = {}
        bounds = self.get_parameter_bounds()
        categories = self.get_categorical_mappings()

        # Randomize continuous parameters
        for name, bound in bounds.items():
            # If it's a categorical parameter, we handle it separately
            if name in categories:
                continue
            params[name] = float(rng.uniform(bound['min'], bound['max']))

        # Randomize categorical parameters
        for name, category_data in categories.items():
            options = category_data.get('options', [])
            if options:
                params[name] = float(rng.choice(options))
            else:
                # Fallback if categories are defined purely by bounds
                if name in bounds:
                    params[name] = int(rng.integers(bounds[name]['min'], bounds[name]['max'] + 1))

        return params
