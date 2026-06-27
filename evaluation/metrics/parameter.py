"""Parameter-axis metrics (the diagnostic, secondary metrics).

Pure functions over **ML-side vectors** (continuous params in place, categoricals
as one-hot blocks per D2). Routing between continuous and categorical parameters
is delegated entirely to :attr:`ParameterSpace.loss_slices`, which already
partitions ``[0, ml_dimension)`` exactly -- these metrics never duplicate slice
logic and never touch a live synthesizer.

Continuous values are stored normalized in ``[0, 1]``, so the per-parameter errors
are in comparable units and a plain mean across parameters is meaningful. Per D2,
categoricals are one-hot + cross-entropy, so a magnitude error on a one-hot block
is meaningless; ``param_mae`` / ``param_mse`` therefore cover continuous parameters
only and ``param_accuracy`` covers categorical parameters only.
"""
from __future__ import annotations

import numpy as np

from synth.parameter_space import ParameterSpace


def param_mae(
    target_vector: np.ndarray,
    predicted_vector: np.ndarray,
    parameter_space: ParameterSpace,
) -> float:
    """Mean absolute error over continuous parameters (lower is better).

    Returns ``float('nan')`` if the space has no continuous parameter.
    """
    errors = [
        abs(float(target_vector[vector_slice.start]) - float(predicted_vector[vector_slice.start]))
        for vector_slice, kind, _ in parameter_space.loss_slices
        if kind == "continuous"
    ]
    if not errors:
        return float("nan")
    return float(np.mean(errors))


def param_mse(
    target_vector: np.ndarray,
    predicted_vector: np.ndarray,
    parameter_space: ParameterSpace,
) -> float:
    """Mean squared error over continuous parameters (lower is better).

    Returns ``float('nan')`` if the space has no continuous parameter.
    """
    errors = [
        (float(target_vector[vector_slice.start]) - float(predicted_vector[vector_slice.start])) ** 2
        for vector_slice, kind, _ in parameter_space.loss_slices
        if kind == "continuous"
    ]
    if not errors:
        return float("nan")
    return float(np.mean(errors))


def param_accuracy(
    target_vector: np.ndarray,
    predicted_vector: np.ndarray,
    parameter_space: ParameterSpace,
) -> float:
    """Fraction of categorical parameters whose argmax class matches (higher is better).

    Each categorical one-hot block is decoded by argmax (accepts raw
    logits/probabilities, matching ``ParameterSpace.ml_vector_to_synth_dict``).
    Returns ``float('nan')`` if the space has no categorical parameter.
    """
    matches = [
        int(np.argmax(target_vector[vector_slice]) == np.argmax(predicted_vector[vector_slice]))
        for vector_slice, kind, _ in parameter_space.loss_slices
        if kind == "categorical"
    ]
    if not matches:
        return float("nan")
    return float(np.mean(matches))
