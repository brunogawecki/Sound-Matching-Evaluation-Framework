"""The metric panel registry (Layer 4).

The panel is a set of pure, stateless, per-sample callables described by
:class:`MetricSpecification` and collected in :data:`METRIC_PANEL`. The registry only
*holds* specs; it never calls ``compute``. The Evaluator (issue #9) routes audio vs
parameter metrics by ``input_type``, supplies the right arguments, applies level
normalization centrally, and aggregates.

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
    lsd,
    mel_mae,
    mel_mse,
    mfcc_mae,
    mfcc_mse,
    mss,
    spectral_convergence,
)
from evaluation.metrics.parameter import param_accuracy, param_mae, param_mse

MetricAxis = Literal["magnitude", "timbre", "perceptual", "loudness", "pitch", "parameter"]
MetricInput = Literal["audio", "parameter"]


@dataclass(frozen=True)
class MetricSpecification:
    """One metric: its axis, input type, orientation, and the callable computing it.

    ``normalize_level`` (loudness-match the prediction to the target before the
    metric) is an **audio-only** concern and must stay ``False`` for parameter
    metrics. ``higher_is_better`` records the metric's orientation for ranking.
    """

    name: str
    axis: MetricAxis
    input_type: MetricInput
    normalize_level: bool
    higher_is_better: bool
    compute: Callable[..., float]

    def __post_init__(self) -> None:
        if self.input_type == "parameter" and self.normalize_level:
            raise ValueError(
                f"Metric '{self.name}': normalize_level is audio-only and must be False "
                "for parameter metrics."
            )


# The panel -- one line per metric. Parameter, magnitude, and timbre axes land
# here; the remaining audio axes (loudness, pitch, perceptual) follow in later
# build-order slices.
METRIC_PANEL: List[MetricSpecification] = [
    MetricSpecification("param_mae", "parameter", "parameter", False, False, param_mae),
    MetricSpecification("param_mse", "parameter", "parameter", False, False, param_mse),
    MetricSpecification("param_accuracy", "parameter", "parameter", False, True, param_accuracy),
    MetricSpecification("lsd", "magnitude", "audio", False, False, lsd),
    MetricSpecification("spectral_convergence", "magnitude", "audio", False, False, spectral_convergence),
    MetricSpecification("mel_mae", "magnitude", "audio", False, False, mel_mae),
    MetricSpecification("mel_mse", "magnitude", "audio", False, False, mel_mse),
    MetricSpecification("mss", "magnitude", "audio", False, False, mss),
    MetricSpecification("mfcc_mae", "timbre", "audio", False, False, mfcc_mae),
    MetricSpecification("mfcc_mse", "timbre", "audio", False, False, mfcc_mse),
]


def metric_names() -> List[str]:
    """The names of every metric in the panel, in registry order."""
    return [spec.name for spec in METRIC_PANEL]
