"""Build the thesis benchmark tables from the scored results on disk.

Reads ``results/<corpus>/<model>/`` for each benchmark corpus, adds the statistics the
Evaluator's summaries do not carry (bootstrap confidence intervals, paired significance),
and writes ``\\input``-able LaTeX fragments plus one machine-readable CSV.

    python scripts/build_results_tables.py

    --results-root   where the scored runs live           [default: <project>/results]
    --out-dir        where the .tex fragments go          [default: ../thesis_latex/tables]
    --csv-out        the long-form aggregate CSV     [default: <project>/results/aggregate]
    --skip-parameter-counts   do not load checkpoints to count model weights

Model weight counts are recorded nowhere in the pipeline, so they are derived here by
loading each checkpoint through ``MODEL_REGISTRY``. That needs torch but never a VST or a
corpus. Pass ``--skip-parameter-counts`` to build the tables without it.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.aggregate import (  # noqa: E402
    HEADLINE_METRICS,
    ResultRun,
    assert_paired,
    best_with_significance,
    load_result_runs,
    metric_table,
    prune_metric_panel,
    spearman_matrix,
)
from evaluation.presentation import (  # noqa: E402
    MODEL_PRESENTATION,
    SYNTH_DISPLAY,
    decimals_for,
    format_cell,
    format_parameter_count,
    metric_header,
    sort_models,
    tabular,
)
from evaluation.registry import METRIC_PANEL  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The two benchmark corpora. dexed_builtin_test is a retired pilot (and one of its cells
# points at a checkpoint that no longer exists), full_preset-gen-vae_test is superseded by
# its 1500-sample subsample (D4), and the *_smoke_test corpora are shakedowns -- none of
# them are benchmark results, so none are read here.
BENCHMARK_CORPORA = {"dexed": "full_preset-gen-vae_test_1500", "diva": "diva_h2p_test"}

# The out-of-domain corpora (D-OOD). Their targets are NSynth recordings the synth never made,
# so they carry no ground-truth parameters: the three parameter metrics report NaN with a
# valid_count of 0 and are omitted from these tables entirely rather than printed as a column
# of em dashes. Per D-OOD the resulting numbers rank models against each other and are NOT
# absolute fidelity figures, because a perfect prediction no longer floors the audio metrics at
# zero -- an NSynth flute is generally unreachable by Dexed. They are therefore emitted as their
# own tables and must never be tabled beside the in-domain values as if the scales matched.
OOD_CORPORA = {"dexed": "nsynth_c4_dexed", "diva": "nsynth_c4_diva"}

# Appendix tables are split by axis so neither exceeds the thesis text width.
APPENDIX_GROUPS = {
    "parameter": ("parameter",),
    "magnitude": ("magnitude",),
    "timbre-loudness-pitch": ("timbre", "loudness", "pitch"),
}

PANEL_BY_NAME = {spec.name: spec for spec in METRIC_PANEL}


def parameter_counts(runs: Sequence[ResultRun]) -> Dict[str, Optional[int]]:
    """Weight count per model, loaded from the checkpoint each result was scored with.

    Checkpoint paths are absolute in older summaries and relative in newer ones, so they
    are normalised against the project root before opening.
    """
    warnings.filterwarnings("ignore")
    from models.registry import MODEL_REGISTRY

    counts: Dict[str, Optional[int]] = {}
    for run in runs:
        recorded = Path(run.summary["checkpoint"]["path"])
        checkpoint = recorded if recorded.is_absolute() else PROJECT_ROOT / recorded
        if not checkpoint.exists():
            print(f"  ! {run.model}: checkpoint missing ({checkpoint}), no weight count")
            counts[run.model] = None
            continue
        model = MODEL_REGISTRY[run.model].model_class()
        model.load(checkpoint)
        network = getattr(model, "_network", None)
        counts[run.model] = (
            None if network is None else int(sum(p.numel() for p in network.parameters()))
        )
    return counts


def _winners(runs: Sequence[ResultRun], metrics: Sequence[str]) -> Dict[str, Optional[str]]:
    """Model to bold per metric: the leader, but only when it is significantly ahead."""
    winners: Dict[str, Optional[str]] = {}
    for metric in metrics:
        leader, significant = best_with_significance(
            runs, metric, PANEL_BY_NAME[metric].higher_is_better
        )
        winners[metric] = leader if significant else None
    return winners


def build_metric_table(
    runs: Sequence[ResultRun],
    metrics: Sequence[str],
    note: str,
) -> str:
    """A models x metrics LaTeX table, grouped by family, best value bolded when significant.

    The family is a spanning header row rather than a column: as a column it was the widest
    thing in the table ("Reinforcement learning") and pushed every one of these past the
    thesis text width.
    """
    table = metric_table(runs, metrics).set_index(["model", "metric"])
    winners = _winners(runs, metrics)
    models = sort_models([run.model for run in runs])

    decimals = {
        metric: decimals_for(
            [(table.loc[(m, metric), "ci_high"] - table.loc[(m, metric), "ci_low"]) / 2 for m in models],
            [table.loc[(m, metric), "mean"] for m in models],
        )
        for metric in metrics
    }

    column_count = 1 + len(metrics)
    header = ["Model"] + [metric_header(m, PANEL_BY_NAME[m].higher_is_better) for m in metrics]
    column_spec = "l" + "r" * len(metrics)

    body: List[Optional[List[str]]] = []
    previous_family = None
    for model in models:
        presentation = MODEL_PRESENTATION[model]
        if presentation.family != previous_family:
            if previous_family is not None:
                body.append(None)
            body.append([
                f"\\multicolumn{{{column_count}}}{{l}}{{\\textit{{{presentation.family}}}}}"
            ])
            previous_family = presentation.family
        row = [f"\\quad {presentation.display}"]
        for metric in metrics:
            cell = table.loc[(model, metric)]
            half_width = (cell["ci_high"] - cell["ci_low"]) / 2
            row.append(
                format_cell(cell["mean"], half_width, decimals[metric], bold=winners[metric] == model)
            )
        body.append(row)

    return tabular(
        column_spec, [header], body, note=note,
        font_size="footnotesize", column_separation="3pt",
    )


def build_model_size_table(counts_by_synth: Dict[str, Dict[str, Optional[int]]]) -> str:
    """Trainable weight count per model on each synthesizer.

    Its own table rather than a column in the results: the count is a property of the model,
    it differs between synths only because the output dimension does (Dexed 333, Diva 892),
    and carrying it inside the results tables pushed them past the text width.
    """
    models = sort_models(sorted(set().union(*(set(c) for c in counts_by_synth.values()))))
    header = ["Model"] + [SYNTH_DISPLAY[synth] for synth in counts_by_synth]
    body: List[Optional[List[str]]] = []
    previous_family = None
    for model in models:
        presentation = MODEL_PRESENTATION[model]
        if presentation.family != previous_family:
            if previous_family is not None:
                body.append(None)
            body.append([
                f"\\multicolumn{{{1 + len(counts_by_synth)}}}{{l}}"
                f"{{\\textit{{{presentation.family}}}}}"
            ])
            previous_family = presentation.family
        body.append(
            [f"\\quad {presentation.display}"]
            + [
                format_parameter_count(counts_by_synth[synth].get(model))
                for synth in counts_by_synth
            ]
        )
    note = (
        "Trainable weights per model. The two columns differ because the estimated parameter\n"
        "vector does: 333 dimensions on Dexed against 892 on Diva. An em dash marks a model\n"
        "with no network, or one whose checkpoint is not available locally."
    )
    return tabular("l" + "r" * len(counts_by_synth), [header], body, note=note)


def _oriented_means(runs: Sequence[ResultRun], models: Sequence[str], metric: str) -> List[float]:
    """Per-model means, sign-flipped so that a smaller value is always the better model."""
    sign = -1.0 if PANEL_BY_NAME[metric].higher_is_better else 1.0
    by_model = {run.model: run.summary["per_metric"][metric]["mean"] for run in runs}
    return [sign * by_model[model] for model in models]


def _rank_agreement(
    dexed_values: Sequence[float],
    diva_values: Sequence[float],
    resamples: int = 2000,
    seed: int = 0,
):
    """Spearman rho between two rankings, with a bootstrap CI resampled over *models*.

    The CI is over models rather than samples because the population being generalized to is
    "model families one might benchmark", and there are only ten of them. It comes out wide,
    which is the honest reading: ten models cannot resolve a moderate rank correlation.
    """
    from scipy.stats import spearmanr

    first = np.asarray(dexed_values, dtype=float)
    second = np.asarray(diva_values, dtype=float)
    observed = spearmanr(first, second)

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        index = rng.integers(0, first.size, size=first.size)
        if np.unique(first[index]).size < 3:
            continue
        draws.append(spearmanr(first[index], second[index]).statistic)
    finite = [value for value in draws if np.isfinite(value)]
    low, high = (np.percentile(finite, [2.5, 97.5]) if finite else (float("nan"),) * 2)
    return observed.statistic, observed.pvalue, float(low), float(high)


def build_cross_synth_table(runs_by_synth: Dict[str, List[ResultRun]], metrics: Sequence[str]) -> str:
    """Does model ranking transfer between synthesizers?

    Spearman correlation between each synth's ranking of the models they share, per metric.
    Only models scored on both synths take part, so the comparison is like for like.
    """
    shared = sorted(
        set(run.model for run in runs_by_synth["dexed"])
        & set(run.model for run in runs_by_synth["diva"])
    )

    rows: List[Optional[List[str]]] = []
    for metric in metrics:
        spec = PANEL_BY_NAME[metric]
        rho, p_value, low, high = _rank_agreement(
            _oriented_means(runs_by_synth["dexed"], shared, metric),
            _oriented_means(runs_by_synth["diva"], shared, metric),
        )
        rows.append([
            metric_header(metric, spec.higher_is_better, short=False),
            f"{rho:.2f}",
            f"[{low:.2f}, {high:.2f}]",
            f"{p_value:.3f}",
        ])

    note = (
        f"Rank agreement across the {len(shared)} models scored on both synthesizers.\n"
        "Ranks are oriented so that rank 1 is the best model on that metric.\n"
        "The interval is a 95% bootstrap CI resampled over models. With only "
        f"{len(shared)} models a\nSpearman rho must exceed about 0.65 to reach p<0.05, so this "
        "test can fail to detect a\nmoderate correlation. Read a near-zero rho as absence of "
        "evidence for rank transfer,\nnot as evidence of its absence."
    )
    return tabular(
        "lrrr",
        [["Metric", "Spearman $\\rho$", "95\\% CI", "$p$"]],
        rows,
        note=note,
    )


def matched_sample_check(
    runs_by_synth: Dict[str, List[ResultRun]],
    metrics: Sequence[str],
    seed: int = 0,
) -> pd.DataFrame:
    """Repeat the rank-agreement analysis with Dexed cut to Diva's sample count.

    Free, because it re-samples rows of a per_sample.csv that is already on disk rather than
    re-evaluating anything: the render is deterministic, so the rows a 271-sample corpus would
    have produced are the rows already there. It answers the one objection the cross-synth
    finding invites -- that the two synths' rankings differ only because 1500 samples resolve
    finer differences than 271 do.
    """
    shared = sorted(
        set(run.model for run in runs_by_synth["dexed"])
        & set(run.model for run in runs_by_synth["diva"])
    )
    target_size = runs_by_synth["diva"][0].num_samples
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(runs_by_synth["dexed"][0].num_samples, target_size, replace=False))

    matched = []
    for run in runs_by_synth["dexed"]:
        frame = run.per_sample.iloc[keep]
        per_metric = {}
        for name in metrics:
            column = frame[name].to_numpy(dtype=float)
            valid = column[~np.isnan(column)]
            per_metric[name] = {
                "mean": float(valid.mean()) if valid.size else float("nan"),
                "higher_is_better": run.summary["per_metric"][name]["higher_is_better"],
            }
        matched.append(ResultRun(run.corpus, run.model, frame, {**run.summary,
                                                               "num_samples": target_size,
                                                               "per_metric": per_metric}))

    rows = []
    for metric in metrics:
        full_rho, _, _, _ = _rank_agreement(
            _oriented_means(runs_by_synth["dexed"], shared, metric),
            _oriented_means(runs_by_synth["diva"], shared, metric),
        )
        matched_rho, _, _, _ = _rank_agreement(
            _oriented_means(matched, shared, metric),
            _oriented_means(runs_by_synth["diva"], shared, metric),
        )
        rows.append({"metric": metric, "rho_full_n": full_rho, "rho_matched_n": matched_rho})
    return pd.DataFrame(rows)


def build_domain_transfer_table(
    in_domain: Dict[str, List[ResultRun]],
    out_of_domain: Dict[str, List[ResultRun]],
    metrics: Sequence[str],
) -> str:
    """Does a model's in-domain rank predict its out-of-domain rank, on the same synth?

    This is the question the out-of-domain axis exists to answer. Both rankings come from the
    same models on the same synthesizer, so unlike the cross-synth comparison there is no
    confound from a different parameter space -- only the targets change, from presets the
    synth made to recordings it never could.
    """
    rows: List[Optional[List[str]]] = []
    for synth, runs in in_domain.items():
        if synth not in out_of_domain:
            continue
        shared = sorted(
            set(r.model for r in runs) & set(r.model for r in out_of_domain[synth])
        )
        rows.append([
            f"\\multicolumn{{4}}{{l}}{{\\textit{{{SYNTH_DISPLAY[synth]}}} "
            f"({len(shared)} models)}}"
        ])
        for metric in metrics:
            rho, p_value, low, high = _rank_agreement(
                _oriented_means(runs, shared, metric),
                _oriented_means(out_of_domain[synth], shared, metric),
            )
            rows.append([
                "\\quad " + metric_header(metric, PANEL_BY_NAME[metric].higher_is_better, short=False),
                f"{rho:.2f}",
                f"[{low:.2f}, {high:.2f}]",
                f"{p_value:.3f}",
            ])
    note = (
        "Agreement between each model's in-domain rank and its out-of-domain rank on the same\n"
        "synthesizer. Ranks are oriented so that rank 1 is the best model. Audio metrics only,\n"
        "since the parameter axis is undefined out of domain. The interval is a 95% bootstrap CI\n"
        "resampled over models; with ten models it is wide, so read a near-zero value as absence\n"
        "of evidence for transfer rather than evidence of its absence."
    )
    return tabular(
        "lrrr",
        [["Metric", "Spearman $\\rho$", "95\\% CI", "$p$"]],
        rows,
        note=note,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the thesis benchmark tables.")
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT.parent / "thesis_latex" / "tables"))
    parser.add_argument("--csv-out", default=str(PROJECT_ROOT / "results" / "aggregate"))
    parser.add_argument("--skip-parameter-counts", action="store_true")
    args = parser.parse_args()

    results_root = Path(args.results_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = Path(args.csv_out)
    csv_dir.mkdir(parents=True, exist_ok=True)

    panel_names = [spec.name for spec in METRIC_PANEL]
    runs_by_synth: Dict[str, List[ResultRun]] = {}
    for synth, corpus in BENCHMARK_CORPORA.items():
        runs = load_result_runs(results_root, corpus)
        assert_paired(runs)
        runs_by_synth[synth] = runs
        print(f"{synth}: {len(runs)} models on '{corpus}' ({runs[0].num_samples} samples)")

    correlations = {
        synth: spearman_matrix(runs, panel_names) for synth, runs in runs_by_synth.items()
    }
    retained = prune_metric_panel(list(correlations.values()))
    dropped = [name for name in panel_names if name not in retained]
    print(f"\npruned panel: kept {len(retained)}, dropped {dropped}")
    for synth, matrix in correlations.items():
        matrix.to_csv(csv_dir / f"metric_correlation_{synth}.csv")

    counts_by_synth: Dict[str, Optional[Dict[str, Optional[int]]]] = {}
    for synth, runs in runs_by_synth.items():
        if args.skip_parameter_counts:
            counts_by_synth[synth] = None
            continue
        print(f"\ncounting {synth} model weights")
        counts_by_synth[synth] = parameter_counts(runs)

    long_form = []
    for synth, runs in runs_by_synth.items():
        display = SYNTH_DISPLAY[synth]
        sample_count = runs[0].num_samples

        headline_note = (
            f"{display} benchmark, n={sample_count}. Mean with a 95% percentile-bootstrap\n"
            "confidence interval (10,000 resamples, seed 0), shown as a half-width.\n"
            "Bold marks a model that beats every other by a paired Wilcoxon signed-rank test\n"
            "at alpha=0.05 with Holm correction; a metric with no bold has no separable winner."
        )
        (out_dir / f"results-main-{synth}.tex").write_text(
            build_metric_table(runs, list(HEADLINE_METRICS), headline_note)
        )

        for suffix, axes in APPENDIX_GROUPS.items():
            metrics = [spec.name for spec in METRIC_PANEL if spec.axis in axes]
            note = (
                f"{display} benchmark, n={sample_count}. Full metric panel, "
                f"{'/'.join(axes)} axes.\n"
                "Metrics dropped by the redundancy analysis are retained here for completeness."
            )
            (out_dir / f"results-appendix-{synth}-{suffix}.tex").write_text(
                build_metric_table(runs, metrics, note)
            )

        table = metric_table(runs, panel_names)
        table.insert(0, "synth", synth)
        table["retained_after_pruning"] = table["metric"].isin(retained)
        long_form.append(table)

    if not args.skip_parameter_counts:
        (out_dir / "results-model-size.tex").write_text(
            build_model_size_table({s: c for s, c in counts_by_synth.items() if c is not None})
        )

    ood_by_synth: Dict[str, List[ResultRun]] = {}
    for synth, corpus in OOD_CORPORA.items():
        try:
            runs = load_result_runs(results_root, corpus)
        except FileNotFoundError:
            print(f"\nout-of-domain: '{corpus}' not scored yet, skipping")
            continue
        assert_paired(runs)
        ood_by_synth[synth] = runs

    if ood_by_synth:
        audio_headline = [m for m in HEADLINE_METRICS if PANEL_BY_NAME[m].input_type == "audio"]
        audio_retained = [m for m in retained if PANEL_BY_NAME[m].input_type == "audio"]
        for synth, runs in ood_by_synth.items():
            display = SYNTH_DISPLAY[synth]
            note = (
                f"{display}, out-of-domain (NSynth), n={runs[0].num_samples}. Audio metrics only:\n"
                "the targets carry no ground-truth parameters, so the parameter axis is undefined\n"
                "(D-OOD) and is omitted rather than printed as empty cells.\n"
                "These numbers rank models against each other. They are NOT absolute fidelity\n"
                "figures and are not comparable with the in-domain tables: the error floor is not\n"
                "zero, because the synthesizer generally cannot reach an NSynth target at all."
            )
            (out_dir / f"results-ood-{synth}.tex").write_text(
                build_metric_table(runs, audio_headline, note)
            )
            table = metric_table(runs, panel_names)
            table.insert(0, "synth", synth)
            table["retained_after_pruning"] = table["metric"].isin(retained)
            long_form.append(table)

        (out_dir / "results-domain-transfer.tex").write_text(
            build_domain_transfer_table(runs_by_synth, ood_by_synth, audio_retained)
        )

    (out_dir / "results-cross-synth.tex").write_text(
        build_cross_synth_table(runs_by_synth, retained)
    )

    matched = matched_sample_check(runs_by_synth, retained)
    matched.to_csv(csv_dir / "cross_synth_matched_n.csv", index=False)
    drift = (matched["rho_full_n"] - matched["rho_matched_n"]).abs().max()
    print(
        f"\nmatched-n check: Dexed cut to {runs_by_synth['diva'][0].num_samples} samples, "
        f"max |rho| change = {drift:.3f}"
    )

    aggregate = pd.concat(long_form, ignore_index=True)
    aggregate.to_csv(csv_dir / "summary.csv", index=False)
    pd.Series(retained).to_csv(csv_dir / "pruned_panel.csv", index=False, header=["metric"])

    print(f"\nwrote {len(list(out_dir.glob('results-*.tex')))} tables to {out_dir}")
    print(f"wrote {csv_dir / 'summary.csv'} ({len(aggregate)} rows)")


if __name__ == "__main__":
    main()
