"""Layer 4 -- the metric panel and the Evaluator.

``MetricSpecification`` / ``METRIC_PANEL`` define the fixed, per-sample metric panel; the
:class:`evaluation.evaluator.Evaluator` consumes the panel to score model predictions.
The Evaluator is **not** re-exported here: it pulls in the render/VST stack, and panel-only
consumers (e.g. the training cluster) should be able to ``import evaluation`` without it.
"""
from evaluation.registry import METRIC_PANEL, MetricSpecification, metric_names

__all__ = ["METRIC_PANEL", "MetricSpecification", "metric_names"]
