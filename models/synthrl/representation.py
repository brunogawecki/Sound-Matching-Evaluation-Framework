"""SynthRL-local class-index representation (SynthRL port, Layer 3).

SynthRL casts the whole parameter-estimation problem as **per-parameter
classification** (Shin & Lee, IJCAI-25, §3.3): every numerical parameter is
binned into ``num_bins`` equal-width ordinal classes; native-categorical
parameters keep their own cardinality. This module owns that scheme and is the
only place the binning lives. It **wraps** the shared :class:`ParameterSpace`
(D2/D-KIND LOCKED -- continuous=float, categorical=one-hot) and never modifies
it: the framework's synth-side dict / ML-side vector contract is untouched, and
this class-index view is private to the SynthRL family.

Layout mirrors :class:`ParameterSpace`: a flat class-logit vector of width
:attr:`total_class_dimension` is partitioned by :attr:`class_slices`, one block
per parameter, in subset order. The network (``network.py``) emits that flat
vector; decoding and the Gaussian-smoothed targets both route through the slices.

Deviations from the paper, deliberate and documented:

- 103-param Dexed subset (D1), not the paper's 144. The subset comes from the
  wrapped :class:`ParameterSpace`; this class only re-buckets it.
- The framework's own continuous/categorical split (16 categorical, 87
  continuous) decides which parameters are binned. Categorical parameters are
  treated as **unordered**: their targets are hard one-hot, with no Gaussian
  smoothing (smoothing over neighbouring class indices only makes sense for the
  ordinal bins of a numerical parameter).
"""
from __future__ import annotations

from typing import Dict, List, Union

import numpy as np

from synth.parameter_space import ParameterSpace

# Paper default: numerical parameters are binned into 25 equal-width classes.
DEFAULT_NUM_BINS = 25
# Gaussian label-smoothing width, in bin units, for the ordinal (binned) heads
# (Chen 2022 / Sound2Synth). A knob; 1.0 spreads mass onto the immediate
# neighbours of the target bin.
DEFAULT_LABEL_SMOOTHING_SIGMA = 1.0


class SynthRLRepresentation:
    """Maps a :class:`ParameterSpace` onto SynthRL's per-parameter class view.

    Continuous parameters become ``num_bins`` equal-width bins over their
    ``[low, high]`` bounds; categorical parameters keep their option grid. The
    class provides the per-parameter class counts (the classification heads), the
    synth-dict <-> class-index conversions, and the Gaussian-smoothed target
    distributions used by the cross-entropy parameter loss.
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        num_bins: int = DEFAULT_NUM_BINS,
        label_smoothing_sigma: float = DEFAULT_LABEL_SMOOTHING_SIGMA,
    ) -> None:
        if num_bins < 1:
            raise ValueError(f"num_bins must be >= 1, got {num_bins}.")
        if label_smoothing_sigma < 0.0:
            raise ValueError(f"label_smoothing_sigma must be >= 0, got {label_smoothing_sigma}.")
        self._parameter_space = parameter_space
        self._num_bins = num_bins
        self._label_smoothing_sigma = label_smoothing_sigma

        self._class_counts: List[int] = []
        for spec in parameter_space.parameter_specs:
            if spec.kind == "categorical":
                self._class_counts.append(len(spec.options))
            else:
                self._class_counts.append(num_bins)

        self._class_slices: List[slice] = []
        offset = 0
        for count in self._class_counts:
            self._class_slices.append(slice(offset, offset + count))
            offset += count
        self._total_class_dimension = offset

    @property
    def parameter_space(self) -> ParameterSpace:
        """The wrapped (unmodified) parameter space."""
        return self._parameter_space

    @property
    def num_bins(self) -> int:
        return self._num_bins

    @property
    def names(self) -> List[str]:
        """Parameter names in subset order."""
        return self._parameter_space.names

    @property
    def class_counts(self) -> List[int]:
        """Number of classes per parameter, in subset order (the head widths)."""
        return list(self._class_counts)

    @property
    def class_slices(self) -> List[slice]:
        """Per-parameter slices partitioning a flat class vector, in subset order."""
        return list(self._class_slices)

    @property
    def total_class_dimension(self) -> int:
        """Total width of the flat class-logit vector (sum of the head widths)."""
        return self._total_class_dimension

    def _bin_index(self, spec, value: float) -> int:
        """Bin a continuous value into ``[0, num_bins)`` over the spec's bounds."""
        low, high = spec.bounds
        fraction = (value - low) / (high - low)
        return int(np.clip(int(fraction * self._num_bins), 0, self._num_bins - 1))

    def _bin_center(self, spec, index: int) -> float:
        """The normalized value at the center of bin ``index``."""
        low, high = spec.bounds
        return low + (index + 0.5) * (high - low) / self._num_bins

    def synth_dict_to_class_indices(self, params: Dict[str, Union[float, int]]) -> np.ndarray:
        """Convert a synth-side dict to per-parameter target class indices.

        Continuous values are binned; categorical values are snapped to the
        nearest option (matching :meth:`ParameterSpace.synth_dict_to_ml_vector`).
        Returns an ``int64`` array of shape ``(num_parameters,)`` in subset order.
        """
        missing = set(self.names) - set(params)
        extra = set(params) - set(self.names)
        if missing or extra:
            raise KeyError(
                f"Synth dict keys must match the subset exactly; "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        indices = np.zeros(len(self._class_counts), dtype=np.int64)
        for position, spec in enumerate(self._parameter_space.parameter_specs):
            value = float(params[spec.name])
            if spec.kind == "categorical":
                indices[position] = int(np.argmin([abs(value - option) for option in spec.options]))
            else:
                indices[position] = self._bin_index(spec, value)
        return indices

    def class_indices_to_synth_dict(self, class_indices: np.ndarray) -> Dict[str, float]:
        """Convert per-parameter class indices to a synth-side dict.

        Continuous classes decode to their bin center; categorical classes decode
        to their option value. Round-trips a synth-dict to within one bin width.
        """
        class_indices = np.asarray(class_indices)
        if class_indices.shape != (len(self._class_counts),):
            raise ValueError(
                f"Expected class indices of shape ({len(self._class_counts)},), got {class_indices.shape}"
            )
        params: Dict[str, float] = {}
        for position, spec in enumerate(self._parameter_space.parameter_specs):
            index = int(class_indices[position])
            if spec.kind == "categorical":
                params[spec.name] = float(spec.options[index])
            else:
                params[spec.name] = float(self._bin_center(spec, index))
        return params

    def class_logits_to_class_indices(self, class_vector: np.ndarray) -> np.ndarray:
        """Argmax each per-parameter block of a flat class vector to class indices.

        Accepts raw logits or probabilities of shape ``(total_class_dimension,)``.
        """
        class_vector = np.asarray(class_vector)
        if class_vector.shape != (self._total_class_dimension,):
            raise ValueError(
                f"Expected class vector of shape ({self._total_class_dimension},), got {class_vector.shape}"
            )
        indices = np.zeros(len(self._class_counts), dtype=np.int64)
        for position, block_slice in enumerate(self._class_slices):
            indices[position] = int(np.argmax(class_vector[block_slice]))
        return indices

    def class_logits_to_synth_dict(self, class_vector: np.ndarray) -> Dict[str, float]:
        """Decode a flat class-logit vector straight to a synth-side dict."""
        return self.class_indices_to_synth_dict(self.class_logits_to_class_indices(class_vector))

    def smoothed_target_vector(self, class_indices: np.ndarray) -> np.ndarray:
        """Build the flat soft-target vector for the cross-entropy parameter loss.

        Each ordinal (binned) head gets a Gaussian over class indices centered on
        the target bin (width :attr:`label_smoothing_sigma`); each categorical
        head gets a hard one-hot. Every block sums to 1. Shape
        ``(total_class_dimension,)``.
        """
        class_indices = np.asarray(class_indices)
        if class_indices.shape != (len(self._class_counts),):
            raise ValueError(
                f"Expected class indices of shape ({len(self._class_counts)},), got {class_indices.shape}"
            )
        target = np.zeros(self._total_class_dimension, dtype=np.float64)
        for position, (spec, block_slice) in enumerate(
            zip(self._parameter_space.parameter_specs, self._class_slices)
        ):
            index = int(class_indices[position])
            count = self._class_counts[position]
            if spec.kind == "categorical" or self._label_smoothing_sigma == 0.0:
                block = np.zeros(count, dtype=np.float64)
                block[index] = 1.0
            else:
                positions = np.arange(count, dtype=np.float64)
                block = np.exp(-((positions - index) ** 2) / (2.0 * self._label_smoothing_sigma ** 2))
                block /= block.sum()
            target[block_slice] = block
        return target
