"""Aggregate scored evaluation runs into benchmark tables (Layer 4).

The Evaluator writes one ``results/<corpus>/<model>/`` folder per scored cell, holding
``per_sample.csv`` (the source of truth) and ``eval_summary.json`` (mean / std /
valid_count only). This module is their consumer: it loads many such folders, adds the
statistics the summaries do not carry -- bootstrap confidence intervals and paired
significance -- and prunes the metric panel to a non-redundant subset (D-METRIC-PRUNE).

Everything here is pure apart from :func:`load_result_runs`. The statistics take arrays
and return numbers, so they are testable without touching disk.

This module deliberately does **not** import ``dashboard/discovery.py``, which walks the
same tree: the dashboard never imports the pipeline, and importing it *from* the pipeline
would invert that rule. The directory walk is small enough to keep here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from evaluation.registry import METRIC_PANEL, MetricSpecification

# Bootstrap settings. Seeded so a regenerated table is byte-identical.
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
CONFIDENCE_LEVEL = 0.95

# Two metrics count as redundant above this |Spearman rho| (D-METRIC-PRUNE).
#
# The panel's within-axis correlations are bimodal: the three MAE/MSE duplicate pairs sit at
# 0.93-0.99 on both synths, and the next-highest pair (mel_mae/mss) at 0.85. Any threshold in
# 0.86-0.93 therefore selects exactly the same subset, so the cut is set in the middle of that
# gap rather than at a round number that happens to land on an edge.
REDUNDANCY_THRESHOLD = 0.90

# One metric per axis for the headline tables, drawn from the pruned panel. The parameter axis
# contributes two because its metrics are not interchangeable: param_mae scores the continuous
# parameters and param_accuracy the categorical ones, so dropping either hides half the vector.
# Where an axis offers several non-redundant options the established literature definition wins
# (05-implementation.tex commits the panel to published definitions): mss is DDSP's multi-scale
# spectral loss, mfcc_mae follows the preset-gen-vae similarity evaluator.
HEADLINE_METRICS: Tuple[str, ...] = (
    "param_mae",
    "param_accuracy",
    "mss",
    "mfcc_mae",
    "loudness_envelope_l1",
    "f0_rmse",
)


@dataclass(frozen=True)
class ResultRun:
    """One scored (corpus, model) cell, loaded from disk."""

    corpus: str
    model: str
    per_sample: pd.DataFrame
    summary: dict

    @property
    def num_samples(self) -> int:
        return int(self.summary["num_samples"])


def load_result_runs(results_root: Path, corpus: str) -> List[ResultRun]:
    """Load every scored model for one corpus, ordered by model name.

    ``eval_summary.json`` may contain bare ``NaN`` literals on an out-of-domain corpus
    (Python's json writes them; strict JSON has no such token), so it must be read with
    Python's own decoder.
    """
    corpus_dir = Path(results_root) / corpus
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"No results for corpus '{corpus}' under {results_root}")

    runs: List[ResultRun] = []
    for model_dir in sorted(corpus_dir.iterdir()):
        summary_path = model_dir / "eval_summary.json"
        per_sample_path = model_dir / "per_sample.csv"
        if not (summary_path.is_file() and per_sample_path.is_file()):
            continue
        with open(summary_path) as summary_file:
            summary = json.load(summary_file)
        runs.append(ResultRun(
            corpus=corpus,
            model=model_dir.name,
            per_sample=pd.read_csv(per_sample_path),
            summary=summary,
        ))
    if not runs:
        raise FileNotFoundError(f"No scored models under {corpus_dir}")
    return runs


def assert_paired(runs: Sequence[ResultRun]) -> None:
    """Fail loudly unless every run scored the identical sample set.

    The paired tests below are only valid because each model on a corpus sees the same
    targets. A mismatch would silently produce a meaningless p-value, so it is checked
    rather than assumed.
    """
    reference = list(runs[0].per_sample["sample_id"])
    for run in runs[1:]:
        if list(run.per_sample["sample_id"]) != reference:
            raise ValueError(
                f"'{run.model}' and '{runs[0].model}' on '{run.corpus}' scored different "
                "samples, so they cannot be compared pairwise"
            )


def bootstrap_confidence_interval(
    values: np.ndarray,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = CONFIDENCE_LEVEL,
) -> Tuple[float, float]:
    """Percentile bootstrap CI for the mean, over the non-NaN values.

    A percentile bootstrap rather than a normal approximation because the panel's
    distributions are heavily skewed -- ``f0_rmse`` runs a mean of 75 against a std of
    171, i.e. a handful of samples dominate -- so a symmetric interval would misstate it.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[~np.isnan(finite)]
    if finite.size == 0:
        return (float("nan"), float("nan"))
    if finite.size == 1:
        return (float(finite[0]), float(finite[0]))

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, finite.size, size=(resamples, finite.size))
    means = finite[draws].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.percentile(means, [100 * tail, 100 * (1 - tail)])
    return (float(low), float(high))


def paired_wilcoxon(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value over samples valid in both runs.

    Returns 1.0 when the comparison is undefined (no usable pairs, or every difference is
    exactly zero), so an undefined test can never be read as a significant win.
    """
    from scipy.stats import wilcoxon

    first = np.asarray(values_a, dtype=float)
    second = np.asarray(values_b, dtype=float)
    both_valid = ~np.isnan(first) & ~np.isnan(second)
    first, second = first[both_valid], second[both_valid]
    if first.size == 0 or np.allclose(first, second):
        return 1.0
    try:
        return float(wilcoxon(first, second).pvalue)
    except ValueError:
        return 1.0


def best_with_significance(
    runs: Sequence[ResultRun],
    metric: str,
    higher_is_better: bool,
    alpha: float = 0.05,
) -> Tuple[Optional[str], bool]:
    """The winning model on one metric, and whether it beats the runner-up.

    Ranks models by mean, then tests the leader against every other model with a paired
    Wilcoxon, Holm-corrected across those comparisons. The leader is reported as a
    significant winner only if it beats *all* of them -- so a metric where the top two are
    indistinguishable bolds nothing, which is the intended behaviour.
    """
    means: List[Tuple[str, float]] = []
    for run in runs:
        column = run.per_sample[metric].to_numpy(dtype=float)
        valid = column[~np.isnan(column)]
        if valid.size:
            means.append((run.model, float(valid.mean())))
    if not means:
        return (None, False)

    means.sort(key=lambda item: item[1], reverse=higher_is_better)
    leader = means[0][0]
    if len(means) == 1:
        return (leader, False)

    by_model = {run.model: run.per_sample[metric].to_numpy(dtype=float) for run in runs}
    challengers = [model for model, _ in means[1:]]
    p_values = [paired_wilcoxon(by_model[leader], by_model[other]) for other in challengers]
    return (leader, _holm_all_reject(p_values, alpha))


def _holm_all_reject(p_values: Sequence[float], alpha: float) -> bool:
    """True when every p-value survives Holm correction at ``alpha``.

    Holm rather than raw thresholds because the leader is tested against every other model
    in the column, and an uncorrected family of ~10 tests would manufacture a winner.
    """
    if not p_values:
        return False
    ordered = sorted(p_values)
    total = len(ordered)
    return all(p <= alpha / (total - index) for index, p in enumerate(ordered))


def spearman_matrix(runs: Sequence[ResultRun], metric_names: Sequence[str]) -> pd.DataFrame:
    """Spearman rank correlation between metrics, pooled over every model on a corpus.

    Pooled across models on purpose: redundancy is a property of the panel over the whole
    range of match quality it has to describe, not of one model's slice of it.
    """
    pooled = pd.concat([run.per_sample[list(metric_names)] for run in runs], ignore_index=True)
    return pooled.corr(method="spearman")


def prune_metric_panel(
    correlations: Sequence[pd.DataFrame],
    panel: Sequence[MetricSpecification] = tuple(METRIC_PANEL),
    threshold: float = REDUNDANCY_THRESHOLD,
) -> List[str]:
    """Reduce the panel to a non-redundant subset (D-METRIC-PRUNE).

    Takes one correlation matrix per corpus and drops a metric only when it is redundant on
    **every** one of them. Being conservative in that direction matters: a metric that
    duplicates another on Dexed but not on Diva is carrying real information about at least
    one synth, and the benchmark reports both.

    The rule, applied in panel order so the outcome is reproducible:

    1. every axis keeps at least one metric, so no aspect of similarity is lost;
    2. within an axis, a metric is dropped when its lowest |rho| across corpora still exceeds
       ``threshold`` against a metric already retained.

    Correlation is compared on absolute value, so a retained higher-is-better metric can still
    absorb a lower-is-better one that ranks samples identically in reverse.
    """
    if not correlations:
        raise ValueError("prune_metric_panel needs at least one correlation matrix")

    def weakest(first: str, second: str) -> float:
        """The most conservative evidence of redundancy across corpora."""
        return min(abs(matrix.loc[first, second]) for matrix in correlations)

    by_axis: Dict[str, List[str]] = {}
    for spec in panel:
        by_axis.setdefault(spec.axis, []).append(spec.name)

    retained: List[str] = []
    for spec in panel:
        if spec.name in retained:
            continue
        if any(spec.name not in matrix.columns for matrix in correlations):
            continue
        kept_on_axis = [held for held in retained if held in by_axis[spec.axis]]
        if not any(weakest(spec.name, held) > threshold for held in kept_on_axis):
            retained.append(spec.name)
    return [spec.name for spec in panel if spec.name in retained]


def metric_table(runs: Sequence[ResultRun], metric_names: Sequence[str]) -> pd.DataFrame:
    """Long-form table: one row per (model, metric) with mean, CI, and validity.

    ``mean`` is read from ``eval_summary.json`` rather than recomputed, so a generated
    table can never disagree with the published summary it came from.
    """
    rows = []
    for run in runs:
        for name in metric_names:
            stats = run.summary["per_metric"][name]
            column = run.per_sample[name].to_numpy(dtype=float)
            low, high = bootstrap_confidence_interval(column)
            rows.append({
                "corpus": run.corpus,
                "model": run.model,
                "metric": name,
                "mean": float(stats["mean"]),
                "ci_low": low,
                "ci_high": high,
                "valid_count": int(stats["valid_count"]),
                "num_samples": run.num_samples,
                "higher_is_better": bool(stats["higher_is_better"]),
            })
    return pd.DataFrame(rows)
