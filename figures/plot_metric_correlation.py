"""Metric-panel redundancy heatmap for the thesis Results chapter (D-METRIC-PRUNE).

Spearman rank correlation between every pair of the 13 panel metrics, one panel per
synthesizer, pooled over every model scored on that synth's benchmark corpus. This is the
evidence behind the panel reduction that Chapter 5 promises is "reported there": metrics
whose correlation exceeds the redundancy threshold on *both* synths are dropped.

The figure carries no interpretive annotation beyond marking the dropped metrics; the
reading is left to the prose.

Run:
    python figures/plot_metric_correlation.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evaluation.aggregate import (  # noqa: E402
    REDUNDANCY_THRESHOLD,
    load_result_runs,
    prune_metric_panel,
    spearman_matrix,
)
from evaluation.presentation import METRIC_DISPLAY, SYNTH_DISPLAY  # noqa: E402
from evaluation.registry import METRIC_PANEL  # noqa: E402
from figures.style import apply_paper_style, FULL_WIDTH_IN, figure_size  # noqa: E402

_BENCHMARK_CORPORA = {"dexed": "full_preset-gen-vae_test_1500", "diva": "diva_h2p_test"}
_DEFAULT_PDF = _REPO_ROOT.parent / "thesis_latex" / "figures" / "results-metric-correlation.pdf"


def _plain(label: str) -> str:
    """Strip the LaTeX escaping the tables need; matplotlib mathtext wants the bare text."""
    return label.replace("\\ ", " ").replace("$F_0$", "F0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(_REPO_ROOT / "results"))
    parser.add_argument("--out", default=str(_DEFAULT_PDF))
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    using_tex = apply_paper_style()
    names = [spec.name for spec in METRIC_PANEL]
    labels = [_plain(METRIC_DISPLAY[name]) for name in names]

    matrices = {}
    for synth, corpus in _BENCHMARK_CORPORA.items():
        runs = load_result_runs(Path(args.results_root), corpus)
        matrices[synth] = spearman_matrix(runs, names)

    retained = prune_metric_panel(list(matrices.values()))
    dropped = [name for name in names if name not in retained]

    fig, axes = plt.subplots(1, 2, figsize=figure_size(FULL_WIDTH_IN, aspect=0.52))
    image = None
    for axis, (synth, matrix) in zip(axes, matrices.items()):
        image = axis.imshow(matrix.to_numpy(), vmin=-1.0, vmax=1.0, cmap="RdBu_r")
        axis.set_xticks(range(len(names)))
        axis.set_yticks(range(len(names)))
        axis.set_xticklabels(labels, rotation=90)
        axis.set_yticklabels(labels if synth == "dexed" else [])
        axis.set_title(SYNTH_DISPLAY[synth])
        # Mark the metrics the redundancy rule removes, so the figure shows the outcome
        # alongside the evidence for it.
        for index, name in enumerate(names):
            if name in dropped:
                axis.get_xticklabels()[index].set_color("#B00020")
                if synth == "dexed":
                    axis.get_yticklabels()[index].set_color("#B00020")

    colorbar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Spearman $\\rho$")

    fig.savefig(args.out)
    print(
        f"wrote {args.out}  (threshold {REDUNDANCY_THRESHOLD}, "
        f"dropped {dropped}, usetex={using_tex})"
    )


if __name__ == "__main__":
    main()
