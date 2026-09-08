"""Cross-synthesizer rank-transfer figure for the thesis Results chapter.

Each model's mean rank across the pruned metric panel on Dexed, joined to its mean rank on
Diva. Parallel lines would mean a model's standing carries from one synthesizer to the
other; crossing lines mean it does not.

Mean rank across the whole pruned panel rather than one chosen metric, so the picture
cannot be read as a metric cherry-picked to cross the most. Only models scored on both
synthesizers appear.

Run:
    python figures/plot_cross_synth_ranks.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evaluation.aggregate import (  # noqa: E402
    load_result_runs,
    prune_metric_panel,
    spearman_matrix,
)
from evaluation.presentation import MODEL_PRESENTATION, SYNTH_DISPLAY  # noqa: E402
from evaluation.registry import METRIC_PANEL  # noqa: E402
from figures.style import apply_paper_style, HALF_WIDTH_IN, figure_size  # noqa: E402

_BENCHMARK_CORPORA = {"dexed": "full_preset-gen-vae_test_1500", "diva": "diva_h2p_test"}
_DEFAULT_PDF = _REPO_ROOT.parent / "thesis_latex" / "figures" / "results-cross-synth-ranks.pdf"

# One colour per family, so crossings read as families changing places, not just models.
_FAMILY_COLOURS = {
    "Baseline": "#666666",
    "Discriminative": "#0072B2",
    "Generative (VAE)": "#009E73",
    "Neural proxy": "#D55E00",
    "Flow matching": "#CC79A7",
    "Reinforcement learning": "#E69F00",
}


def mean_ranks(runs, models, metrics) -> pd.Series:
    """Mean rank (1 = best) of each model across the metrics, honouring each one's direction."""
    panel = {spec.name: spec for spec in METRIC_PANEL}
    frame = pd.DataFrame(
        {
            metric: [
                next(r for r in runs if r.model == m).summary["per_metric"][metric]["mean"]
                for m in models
            ]
            for metric in metrics
        },
        index=models,
    )
    ranks = pd.DataFrame(
        {
            metric: frame[metric].rank(ascending=not panel[metric].higher_is_better)
            for metric in metrics
        },
        index=models,
    )
    return ranks.mean(axis=1)


def spread_labels(positions, minimum_gap: float):
    """Nudge label positions apart just enough to stop them overlapping.

    Models with near-identical mean ranks would otherwise print their names on top of each
    other. Only the *label* moves; the plotted point stays on its true rank.
    """
    order = sorted(range(len(positions)), key=lambda index: positions[index])
    adjusted = list(positions)
    for earlier, later in zip(order, order[1:]):
        gap = adjusted[later] - adjusted[earlier]
        if gap < minimum_gap:
            adjusted[later] = adjusted[earlier] + minimum_gap
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(_REPO_ROOT / "results"))
    parser.add_argument("--out", default=str(_DEFAULT_PDF))
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    using_tex = apply_paper_style()
    names = [spec.name for spec in METRIC_PANEL]

    runs_by_synth = {
        synth: load_result_runs(Path(args.results_root), corpus)
        for synth, corpus in _BENCHMARK_CORPORA.items()
    }
    retained = prune_metric_panel(
        [spearman_matrix(runs, names) for runs in runs_by_synth.values()]
    )
    shared = sorted(
        set(r.model for r in runs_by_synth["dexed"]) & set(r.model for r in runs_by_synth["diva"]),
        key=lambda name: MODEL_PRESENTATION[name].order,
    )
    ranks = {
        synth: mean_ranks(runs, shared, retained) for synth, runs in runs_by_synth.items()
    }

    fig, axis = plt.subplots(figsize=figure_size(HALF_WIDTH_IN * 1.6, aspect=0.85))

    span = max(max(ranks["dexed"]), max(ranks["diva"])) - min(
        min(ranks["dexed"]), min(ranks["diva"])
    )
    label_gap = span * 0.055
    left_labels = spread_labels([ranks["dexed"][model] for model in shared], label_gap)
    right_labels = spread_labels([ranks["diva"][model] for model in shared], label_gap)

    for index, model in enumerate(shared):
        presentation = MODEL_PRESENTATION[model]
        left, right = ranks["dexed"][model], ranks["diva"][model]
        colour = _FAMILY_COLOURS[presentation.family]
        axis.plot([0, 1], [left, right], color=colour, marker="o", markersize=3.5, linewidth=1.2)
        axis.annotate(
            presentation.display, xy=(-0.03, left_labels[index]),
            ha="right", va="center", color=colour,
        )
        axis.annotate(
            presentation.display, xy=(1.03, right_labels[index]),
            ha="left", va="center", color=colour,
        )

    axis.set_xticks([0, 1])
    axis.set_xticklabels([SYNTH_DISPLAY["dexed"], SYNTH_DISPLAY["diva"]])
    axis.set_xlim(-0.62, 1.62)
    axis.set_ylabel(f"Mean rank over {len(retained)} metrics (1 = best)")
    axis.invert_yaxis()
    axis.spines[["top", "right", "bottom"]].set_visible(False)
    axis.tick_params(axis="x", length=0)

    fig.savefig(args.out)
    print(f"wrote {args.out}  ({len(shared)} models, {len(retained)} metrics, usetex={using_tex})")


if __name__ == "__main__":
    main()
