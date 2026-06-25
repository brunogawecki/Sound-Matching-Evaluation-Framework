from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np


@dataclass(frozen=True)
class ParameterSpecification:
    """
    Specification of a single estimated parameter.

    Parameters are addressed by their plugin-reported name only (D-NAMING);
    index resolution is the synthesizer wrapper's private concern.
    """
    name: str
    kind: Literal["continuous", "categorical"]
    bounds: Tuple[float, float] = (0.0, 1.0)
    options: Optional[List[float]] = None
    default: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        if self.kind == "categorical":
            if not self.options:
                raise ValueError(f"Categorical parameter '{self.name}' requires options.")
        elif self.kind == "continuous":
            if self.options is not None:
                raise ValueError(f"Continuous parameter '{self.name}' must not define options.")
            if not self.bounds[0] < self.bounds[1]:
                raise ValueError(f"Parameter '{self.name}' has invalid bounds {self.bounds}.")
        else:
            raise ValueError(f"Parameter '{self.name}' has unknown kind '{self.kind}'.")

    @property
    def ml_dimension(self) -> int:
        """Width of this parameter in the ML-side vector (1 or one-hot cardinality)."""
        if self.kind == "categorical":
            return len(self.options)
        return 1

    def to_dict(self) -> Dict[str, object]:
        """A JSON-safe dict round-tripping through :meth:`from_dict`."""
        return {
            "name": self.name,
            "kind": self.kind,
            "bounds": [self.bounds[0], self.bounds[1]],
            "options": list(self.options) if self.options is not None else None,
            "default": self.default,
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ParameterSpecification":
        """Inverse of :meth:`to_dict`."""
        options = data.get("options")
        return cls(
            name=data["name"],
            kind=data["kind"],
            bounds=(float(data["bounds"][0]), float(data["bounds"][1])),
            options=[float(option) for option in options] if options is not None else None,
            default=float(data["default"]),
            label=data.get("label", ""),
        )


class ParameterSpace:
    """
    Canonical, ordered parameter space for a chosen synth subset.

    Owns the two-way conversion between the synth-side dict (parameter names ->
    normalized floats in [0, 1], what the synthesizer wrapper accepts) and the
    ML-side vector (continuous params as floats, categorical params as one-hot
    blocks, what models train on).
    """

    def __init__(self, parameter_specs: List[ParameterSpecification]):
        self._parameter_specs: List[ParameterSpecification] = list(parameter_specs)
        self._names: List[str] = [parameter_spec.name for parameter_spec in self._parameter_specs]
        duplicates = {name for name in self._names if self._names.count(name) > 1}
        if duplicates:
            raise ValueError(f"Duplicate parameter names: {sorted(duplicates)}")

        self._slices: List[slice] = []
        offset = 0
        for parameter_spec in self._parameter_specs:
            self._slices.append(slice(offset, offset + parameter_spec.ml_dimension))
            offset += parameter_spec.ml_dimension
        self._ml_dimension = offset

    def to_dict(self) -> Dict[str, object]:
        """Serialize the ordered spec list (JSON-safe). See :meth:`from_dict`.

        Lets a built corpus carry its own parameter map so the ML-side vector can
        be reconstructed offline, with no live synthesizer/VST (the training and
        evaluation path runs where the plugin is unavailable).
        """
        return {"parameter_specs": [parameter_spec.to_dict() for parameter_spec in self._parameter_specs]}

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ParameterSpace":
        """Rebuild a ParameterSpace from :meth:`to_dict` output; order is preserved."""
        return cls([
            ParameterSpecification.from_dict(serialized_spec)
            for serialized_spec in data["parameter_specs"]
        ])

    @property
    def parameter_specs(self) -> List[ParameterSpecification]:
        return list(self._parameter_specs)

    @property
    def names(self) -> List[str]:
        return list(self._names)

    @property
    def synth_dimension(self) -> int:
        """Number of parameters in the synth-side dict."""
        return len(self._parameter_specs)

    @property
    def ml_dimension(self) -> int:
        """Length of the ML-side vector (one-hot expanded)."""
        return self._ml_dimension

    @property
    def loss_slices(self) -> List[Tuple[slice, str, str]]:
        """
        (vector vector_slice, kind, parameter name) per parameter, in order.
        The slices partition [0, ml_dimension) exactly; used to route MSE/MAE
        (continuous) vs cross-entropy (categorical) losses and metrics.
        """
        return [
            (vector_slice, parameter_spec.kind, parameter_spec.name)
            for vector_slice, parameter_spec in zip(self._slices, self._parameter_specs)
        ]

    def synth_dict_to_ml_vector(self, params: Dict[str, Union[float, int]]) -> np.ndarray:
        """
        Convert a synth-side dict to an ML-side vector.

        The dict keys must equal the subset names exactly. Categorical values
        are snapped to the nearest grid option before one-hot encoding.

        Raises:
            KeyError: If keys are missing from or extra to the subset.
        """
        missing = set(self.names) - set(params)
        extra = set(params) - set(self.names)
        if missing or extra:
            raise KeyError(
                f"Synth dict keys must match the subset exactly; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )

        vector = np.zeros(self._ml_dimension, dtype=np.float64)
        for vector_slice, parameter_spec in zip(self._slices, self._parameter_specs):
            value = float(params[parameter_spec.name])
            if parameter_spec.kind == "categorical":
                index = int(np.argmin([abs(value - option) for option in parameter_spec.options]))
                vector[vector_slice.start + index] = 1.0
            else:
                vector[vector_slice.start] = value
        return vector

    def ml_vector_to_synth_dict(self, vector: np.ndarray) -> Dict[str, float]:
        """
        Convert an ML-side vector to a synth-side dict.

        Categorical blocks are decoded via argmax (accepts raw logits/probabilities,
        not just one-hot). Continuous values are clipped to their bounds so the
        result is always a valid input to the synthesizer wrapper.
        """
        if vector.shape != (self._ml_dimension,):
            raise ValueError(
                f"Expected vector of shape ({self._ml_dimension},), got {vector.shape}"
            )
        params: Dict[str, float] = {}
        for vector_slice, parameter_spec in zip(self._slices, self._parameter_specs):
            block = vector[vector_slice]
            if parameter_spec.kind == "categorical":
                params[parameter_spec.name] = float(parameter_spec.options[int(np.argmax(block))])
            else:
                params[parameter_spec.name] = float(np.clip(block[0], parameter_spec.bounds[0], parameter_spec.bounds[1]))
        return params

    def _draw(
        self, rng: np.random.Generator, sampling_ranges: Dict[str, Tuple[float, float]]
    ) -> Dict[str, float]:
        """Draw one synth-side dict; continuous params use the overridden range if any."""
        params: Dict[str, float] = {}
        for parameter_spec in self._parameter_specs:
            if parameter_spec.kind == "categorical":
                params[parameter_spec.name] = float(rng.choice(parameter_spec.options))
            else:
                min_val, max_val = sampling_ranges.get(parameter_spec.name, parameter_spec.bounds)
                params[parameter_spec.name] = float(rng.uniform(min_val, max_val))
        return params

    def sample_uniform(self, rng: np.random.Generator) -> Dict[str, float]:
        """
        Sample a random synth-side dict: continuous params uniform over their
        full bounds, categorical params uniform over their grid options.
        Deterministic for a given seeded generator.
        """
        return self._draw(rng, {})

    def sample_constrained(
        self, rng: np.random.Generator, sampling_ranges: Dict[str, Tuple[float, float]]
    ) -> Dict[str, float]:
        """
        Like :meth:`sample_uniform`, but continuous parameters named in
        ``sampling_ranges`` are drawn uniformly over the given ``(min, max)``
        sub-range instead of their full bounds. Everything else is unchanged; an
        empty map is identical to ``sample_uniform``.

        Raises:
            KeyError: an entry names a parameter not in this space.
            ValueError: an entry names a categorical parameter, or its range
                is not ``bounds[0] <= min < max <= bounds[1]``.
        """
        spec_by_name = {spec.name: spec for spec in self._parameter_specs}
        for parameter_name, (min_val, max_val) in sampling_ranges.items():
            if parameter_name not in spec_by_name:
                raise KeyError(f"Sampling range names unknown parameter '{parameter_name}'.")
            spec = spec_by_name[parameter_name]
            if spec.kind != "continuous":
                raise ValueError(
                    f"Sampling range on categorical parameter '{parameter_name}' is not allowed."
                )
            if not (spec.bounds[0] <= min_val < max_val <= spec.bounds[1]):
                raise ValueError(
                    f"Sampling range ({min_val}, {max_val}) for '{parameter_name}' must satisfy "
                    f"{spec.bounds[0]} <= min < max <= {spec.bounds[1]}."
                )
        return self._draw(rng, sampling_ranges)
