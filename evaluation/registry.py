"""The metric panel registry (Layer 4).

The panel is a set of pure, stateless, per-sample callables described by
:class:`MetricSpecification` and collected in :data:`METRIC_PANEL`. The registry only
*holds* specs; it never calls ``compute``. The Evaluator (issue #9) routes audio vs
parameter metrics by ``input_type``, supplies the right arguments, and aggregates.

Adding a metric is one function plus one ``MetricSpecification`` line here; deleting one is
removing the line. Metrics are grouped by **metric axis** so near-collinear metrics
within an axis can later be pruned to a representative few.

``compute`` call conventions::

    audio      compute(target, prediction, *, sample_rate) -> float
    parameter  compute(target_vector, predicted_vector, parameter_space) -> float
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal

from evaluation.metrics.audio_based import (
    f0_rmse,
    integrated_loudness_error,
    loudness_envelope_l1,
    lsd,
    mel_mae,
    mel_mse,
    mfcc_mae,
    mfcc_mse,
    mss,
    spectral_convergence,
)
from evaluation.metrics.parameter import param_accuracy, param_mae, param_mse

# "perceptual" (embedding metrics) is reserved but unimplemented -- deferred to
# potential future work (see D-METRIC-PERCEPTUAL in docs/DECISIONS.md).
MetricAxis = Literal["magnitude", "timbre", "perceptual", "loudness", "pitch", "parameter"]
MetricInput = Literal["audio", "parameter"]


@dataclass(frozen=True)
class MetricSpecification:
    """One metric: its axis, input type, orientation, and the callable computing it.

    Audio metrics compare the target and re-rendered prediction **as rendered**,
    with no loudness matching (D-METRIC-NORM). ``higher_is_better`` records the
    metric's orientation for ranking.
    """

    name: str
    axis: MetricAxis
    input_type: MetricInput
    higher_is_better: bool
    compute: Callable[..., float]


# The panel -- one line per metric. Parameter, magnitude, timbre, loudness, and
# pitch axes are implemented here. The perceptual (embedding) axis is reserved but
# unimplemented -- deferred to potential future work (D-METRIC-PERCEPTUAL).
METRIC_PANEL: List[MetricSpecification] = [
    MetricSpecification("param_mae", "parameter", "parameter", False, param_mae),
    MetricSpecification("param_mse", "parameter", "parameter", False, param_mse),
    MetricSpecification("param_accuracy", "parameter", "parameter", True, param_accuracy),
    MetricSpecification("lsd", "magnitude", "audio", False, lsd),
    MetricSpecification("spectral_convergence", "magnitude", "audio", False, spectral_convergence),
    MetricSpecification("mel_mae", "magnitude", "audio", False, mel_mae),
    MetricSpecification("mel_mse", "magnitude", "audio", False, mel_mse),
    MetricSpecification("mss", "magnitude", "audio", False, mss),
    MetricSpecification("mfcc_mae", "timbre", "audio", False, mfcc_mae),
    MetricSpecification("mfcc_mse", "timbre", "audio", False, mfcc_mse),
    MetricSpecification("loudness_envelope_l1", "loudness", "audio", False, loudness_envelope_l1),
    MetricSpecification("integrated_loudness_error", "loudness", "audio", False, integrated_loudness_error),
    MetricSpecification("f0_rmse", "pitch", "audio", False, f0_rmse),
]


def metric_names() -> List[str]:
    """The names of every metric in the panel, in registry order."""
    return [spec.name for spec in METRIC_PANEL]
