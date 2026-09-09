"""Tests for the results aggregator (evaluation/aggregate.py)."""
import json

import numpy as np
import pandas as pd
import pytest

from evaluation.aggregate import (
    HEADLINE_METRICS,
    REDUNDANCY_THRESHOLD,
    ResultRun,
    assert_paired,
    best_with_significance,
    bootstrap_confidence_interval,
    load_result_runs,
    metric_table,
    paired_wilcoxon,
    prune_metric_panel,
    spearman_matrix,
)
from evaluation.registry import METRIC_PANEL

METRIC_NAMES = [spec.name for spec in METRIC_PANEL]


def _run(model, values, corpus="corpus"):
    """A ResultRun whose per-sample frame holds `values` (name -> sequence)."""
    sample_count = len(next(iter(values.values())))
    frame = pd.DataFrame({"sample_id": [f"sample_{i:06d}" for i in range(sample_count)]})
    for name in METRIC_NAMES:
        frame[name] = list(values.get(name, np.zeros(sample_count)))
    per_metric = {}
    for spec in METRIC_PANEL:
        column = frame[spec.name].to_numpy(dtype=float)
        valid = column[~np.isnan(column)]
        per_metric[spec.name] = {
            "mean": float(valid.mean()) if valid.size else float("nan"),
            "std": float(valid.std()) if valid.size else float("nan"),
            "valid_count": int(valid.size),
            "higher_is_better": spec.higher_is_better,
        }
    summary = {"model_class": model, "num_samples": sample_count, "per_metric": per_metric}
    return ResultRun(corpus=corpus, model=model, per_sample=frame, summary=summary)


class TestBootstrapConfidenceInterval:
    def test_brackets_the_mean_of_a_known_sample(self):
        rng = np.random.default_rng(0)
        values = rng.normal(loc=5.0, scale=1.0, size=2000)
        low, high = bootstrap_confidence_interval(values)
        assert low < values.mean() < high

    def test_interval_is_narrow_for_a_large_low_variance_sample(self):
        low, high = bootstrap_confidence_interval(np.full(1000, 3.0))
        assert low == pytest.approx(3.0) and high == pytest.approx(3.0)

    def test_ignores_nan_values(self):
        with_nan = np.array([1.0, 2.0, 3.0, np.nan, np.nan])
        assert bootstrap_confidence_interval(with_nan) == bootstrap_confidence_interval(
            np.array([1.0, 2.0, 3.0])
        )

    def test_all_nan_yields_nan_not_a_fabricated_number(self):
        low, high = bootstrap_confidence_interval(np.array([np.nan, np.nan]))
        assert np.isnan(low) and np.isnan(high)

    def test_is_seeded_and_therefore_reproducible(self):
        rng = np.random.default_rng(1)
        values = rng.gamma(2.0, 2.0, size=500)
        assert bootstrap_confidence_interval(values) == bootstrap_confidence_interval(values)


class TestPairedWilcoxon:
    def test_agrees_with_scipy_on_a_plain_case(self):
        from scipy.stats import wilcoxon

        rng = np.random.default_rng(2)
        first = rng.normal(size=60)
        second = first + rng.normal(loc=0.5, scale=0.2, size=60)
        assert paired_wilcoxon(first, second) == pytest.approx(wilcoxon(first, second).pvalue)

    def test_identical_inputs_are_not_significant(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        assert paired_wilcoxon(values, values) == 1.0

    def test_uses_only_pairs_valid_in_both_runs(self):
        from scipy.stats import wilcoxon

        first = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 6.0])
        second = np.array([2.0, 3.0, 4.0, np.nan, 5.0, 7.0])
        expected = wilcoxon([1.0, 2.0, 3.0, 6.0], [2.0, 3.0, 4.0, 7.0]).pvalue
        assert paired_wilcoxon(first, second) == pytest.approx(expected)

    def test_no_overlapping_valid_pairs_is_reported_as_not_significant(self):
        first = np.array([1.0, np.nan])
        second = np.array([np.nan, 2.0])
        assert paired_wilcoxon(first, second) == 1.0


class TestBestWithSignificance:
    def test_picks_the_lowest_mean_when_lower_is_better(self):
        rng = np.random.default_rng(3)
        good = rng.normal(loc=1.0, scale=0.1, size=200)
        runs = [_run("good", {"mss": good}), _run("bad", {"mss": good + 1.0})]
        winner, significant = best_with_significance(runs, "mss", higher_is_better=False)
        assert winner == "good" and significant

    def test_picks_the_highest_mean_when_higher_is_better(self):
        rng = np.random.default_rng(4)
        low = rng.normal(loc=0.5, scale=0.05, size=200)
        runs = [_run("low", {"param_accuracy": low}), _run("high", {"param_accuracy": low + 0.3})]
        winner, significant = best_with_significance(runs, "param_accuracy", higher_is_better=True)
        assert winner == "high" and significant

    def test_indistinguishable_models_are_not_declared_significant(self):
        rng = np.random.default_rng(5)
        values = rng.normal(loc=1.0, scale=1.0, size=300)
        runs = [_run("a", {"mss": values}), _run("b", {"mss": values + 1e-9})]
        _, significant = best_with_significance(runs, "mss", higher_is_better=False)
        assert not significant

    def test_leader_beating_only_some_challengers_is_not_significant(self):
        rng = np.random.default_rng(6)
        base = rng.normal(loc=1.0, scale=0.1, size=300)
        runs = [
            _run("leader", {"mss": base}),
            _run("tie", {"mss": base + 1e-9}),
            _run("far", {"mss": base + 5.0}),
        ]
        winner, significant = best_with_significance(runs, "mss", higher_is_better=False)
        assert winner == "leader" and not significant

    def test_all_nan_metric_has_no_winner(self):
        nans = np.full(10, np.nan)
        runs = [_run("a", {"param_mae": nans}), _run("b", {"param_mae": nans})]
        assert best_with_significance(runs, "param_mae", higher_is_better=False) == (None, False)


class TestPruneMetricPanel:
    def _matrix(self, correlations):
        frame = pd.DataFrame(
            np.eye(len(METRIC_NAMES)), index=METRIC_NAMES, columns=METRIC_NAMES
        )
        for (first, second), value in correlations.items():
            frame.loc[first, second] = value
            frame.loc[second, first] = value
        return frame

    def test_drops_a_metric_redundant_on_every_corpus(self):
        high = self._matrix({("mel_mae", "mel_mse"): 0.99})
        kept = prune_metric_panel([high, high])
        assert "mel_mae" in kept and "mel_mse" not in kept

    def test_keeps_a_metric_redundant_on_only_one_corpus(self):
        redundant = self._matrix({("mel_mae", "mel_mse"): 0.99})
        independent = self._matrix({("mel_mae", "mel_mse"): 0.10})
        assert "mel_mse" in prune_metric_panel([redundant, independent])

    def test_never_empties_an_axis(self):
        everything_correlated = self._matrix(
            {(a, b): 0.999 for a in METRIC_NAMES for b in METRIC_NAMES if a != b}
        )
        kept = prune_metric_panel([everything_correlated])
        axes = {spec.axis for spec in METRIC_PANEL}
        kept_axes = {spec.axis for spec in METRIC_PANEL if spec.name in kept}
        assert kept_axes == axes

    def test_treats_negative_correlation_as_redundant(self):
        inverted = self._matrix({("param_mae", "param_mse"): -0.99})
        assert "param_mse" not in prune_metric_panel([inverted, inverted])

    def test_is_deterministic(self):
        matrix = self._matrix({("mel_mae", "mel_mse"): 0.99, ("mfcc_mae", "mfcc_mse"): 0.97})
        assert prune_metric_panel([matrix]) == prune_metric_panel([matrix])

    def test_headline_metrics_are_six(self):
        # Six is a hard limit, not a preference: seven metrics overflow the thesis text
        # width by 12.7pt and eight by 70pt, measured by compiling the generated table.
        assert len(HEADLINE_METRICS) == 6

    def test_headline_metrics_are_all_real_panel_metrics(self):
        assert set(HEADLINE_METRICS) <= {spec.name for spec in METRIC_PANEL}

    def test_headline_metrics_keep_both_parameter_kinds(self):
        # The parameter metrics are not interchangeable -- one scores the continuous
        # parameters and one the categorical ones -- so a headline table carrying only one
        # of them would report half the predicted vector while appearing to report all of it.
        assert {"param_mae", "param_accuracy"} <= set(HEADLINE_METRICS)


class TestAssertPaired:
    def test_accepts_runs_over_the_same_samples(self):
        assert_paired([_run("a", {"mss": [1.0, 2.0]}), _run("b", {"mss": [3.0, 4.0]})])

    def test_rejects_runs_over_different_samples(self):
        first = _run("a", {"mss": [1.0, 2.0]})
        second = _run("b", {"mss": [1.0, 2.0]})
        second.per_sample.loc[0, "sample_id"] = "sample_999999"
        with pytest.raises(ValueError, match="different"):
            assert_paired([first, second])


class TestMetricTable:
    def test_mean_is_read_from_the_summary_not_recomputed(self):
        run = _run("a", {"mss": [1.0, 2.0, 3.0]})
        run.summary["per_metric"]["mss"]["mean"] = 99.0
        table = metric_table([run], ["mss"])
        assert table.loc[0, "mean"] == 99.0

    def test_undefined_metric_reports_zero_valid_and_nan_bounds(self):
        run = _run("a", {"param_mae": [np.nan, np.nan]})
        row = metric_table([run], ["param_mae"]).iloc[0]
        assert row["valid_count"] == 0
        assert np.isnan(row["ci_low"]) and np.isnan(row["ci_high"])

    def test_confidence_interval_brackets_the_mean(self):
        rng = np.random.default_rng(7)
        run = _run("a", {"mss": rng.normal(loc=4.0, scale=1.0, size=500)})
        row = metric_table([run], ["mss"]).iloc[0]
        assert row["ci_low"] < row["mean"] < row["ci_high"]


class TestLoadResultRuns:
    def test_reads_nan_literals_that_strict_json_would_reject(self, tmp_path):
        model_dir = tmp_path / "corpus" / "Model"
        model_dir.mkdir(parents=True)
        (model_dir / "eval_summary.json").write_text(
            '{"model_class": "Model", "num_samples": 1, '
            '"per_metric": {"param_mae": {"mean": NaN, "std": NaN, '
            '"valid_count": 0, "higher_is_better": false}}}'
        )
        (model_dir / "per_sample.csv").write_text("sample_id,param_mae\nsample_000000,\n")
        runs = load_result_runs(tmp_path, "corpus")
        assert np.isnan(runs[0].summary["per_metric"]["param_mae"]["mean"])

    def test_skips_directories_without_both_files(self, tmp_path):
        corpus = tmp_path / "corpus"
        (corpus / "Complete").mkdir(parents=True)
        (corpus / "Partial").mkdir(parents=True)
        (corpus / "Complete" / "eval_summary.json").write_text(
            '{"model_class": "Complete", "num_samples": 1, "per_metric": {}}'
        )
        (corpus / "Complete" / "per_sample.csv").write_text("sample_id\nsample_000000\n")
        (corpus / "Partial" / "per_sample.csv").write_text("sample_id\nsample_000000\n")
        assert [run.model for run in load_result_runs(tmp_path, "corpus")] == ["Complete"]

    def test_missing_corpus_is_an_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_result_runs(tmp_path, "absent")
