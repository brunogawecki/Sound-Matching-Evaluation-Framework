"""Layer 4 -- the metric panel and (future) Evaluator.

``MetricSpecification`` / ``METRIC_PANEL`` define the fixed, per-sample metric panel; the
Evaluator (issue #9) consumes the panel to score model predictions.
"""
from evaluation.registry import METRIC_PANEL, MetricSpecification, metric_names

__all__ = ["METRIC_PANEL", "MetricSpecification", "metric_names"]
