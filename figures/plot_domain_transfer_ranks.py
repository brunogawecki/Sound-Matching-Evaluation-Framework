"""Domain-transfer rank figure for the thesis Results chapter.

One panel per synthesizer. Each model's mean rank over the pruned *audio* panel on the
in-domain benchmark corpus, joined to its mean rank on the out-of-domain NSynth corpus.
Parallel lines would mean a model's standing carries from presets the synth made to
recordings it never could; crossing lines mean it does not.

The companion to ``plot_cross_synth_ranks.py``, and the two answer different questions.
There the parameter space changes with the synthesizer, which confounds the comparison.
Here only the targets change, so a crossing is about the evaluation setting alone.

Audio metrics only: out of domain there is no ground-truth parameter vector, so the three
parameter metrics are undefined (D-OOD). Ranks are taken over the whole retained audio
panel rather than one chosen metric, so the picture cannot be read as cherry-picked.

Run:
    python figures/plot_domain_transfer_ranks.py
"""
import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from evaluation.aggregate import (  # noqa: E402
    load_result_runs,
    prune_metric_panel,
    spearman_matrix,
)
from evaluation.presentation import MODEL_PRESENTATION, SYNTH_DISPLAY  # noqa: E402
from evaluation.registry import METRIC_PANEL  # noqa: E402
from figures.plot_cross_synth_ranks import mean_ranks, spread_labels, _FAMILY_COLOURS  # noqa: E402
from figures.style import apply_paper_style, FULL_WIDTH_IN, figure_size  # noqa: E402

_BENCHMARK_CORPORA = {"dexed": "full_preset-gen-vae_test_1500", "diva": "diva_h2p_test"}
_OOD_CORPORA = {"dexed": "nsynth_c4_dexed", "diva": "nsynth_c4_diva"}
_DEFAULT_PDF = _REPO_ROOT.parent / "thesis_latex" / "figures" / "results-domain-transfer-ranks.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default=str(_REPO_ROOT / "results"))
    parser.add_argument("--out", default=str(_DEFAULT_PDF))
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    using_tex = apply_paper_style()
    results_root = Path(args.results_root)
    panel = {spec.name: spec for spec in METRIC_PANEL}
    names = [spec.name for spec in METRIC_PANEL]

    in_domain = {
        synth: load_result_runs(results_root, corpus)
        for synth, corpus in _BENCHMARK_CORPORA.items()
    }
    out_of_domain = {
        synth: load_result_runs(results_root, corpus) for synth, corpus in _OOD_CORPORA.items()
    }
    retained = prune_metric_panel([spearman_matrix(runs, names) for runs in in_domain.values()])
    audio_metrics = [m for m in retained if panel[m].input_type == "audio"]

    figure, axes = plt.subplots(
        1, len(in_domain), figsize=figure_size(FULL_WIDTH_IN, aspect=0.52), sharey=True
    )

    for axis, synth in zip(axes, in_domain):
        shared = sorted(
            set(r.model for r in in_domain[synth]) & set(r.model for r in out_of_domain[synth]),
            key=lambda name: MODEL_PRESENTATION[name].order,
        )
        left_ranks = mean_ranks(in_domain[synth], shared, audio_metrics)
        right_ranks = mean_ranks(out_of_domain[synth], shared, audio_metrics)

        label_gap = (max(len(shared) - 1, 1)) * 0.062
        left_labels = spread_labels([left_ranks[model] for model in shared], label_gap)
        right_labels = spread_labels([right_ranks[model] for model in shared], label_gap)

        for index, model in enumerate(shared):
            presentation = MODEL_PRESENTATION[model]
            colour = _FAMILY_COLOURS[presentation.family]
            axis.plot(
                [0, 1], [left_ranks[model], right_ranks[model]],
                color=colour, marker="o", markersize=3.0, linewidth=1.1,
            )
            axis.annotate(
                presentation.display, xy=(-0.09, left_labels[index]),
                ha="right", va="center", color=colour, fontsize=6.5,
            )
            axis.annotate(
                presentation.display, xy=(1.09, right_labels[index]),
                ha="left", va="center", color=colour, fontsize=6.5,
            )

        axis.set_title(f"{SYNTH_DISPLAY[synth]} ({len(shared)} models)", fontsize=9)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["In-domain", "Out-of-domain"])
        axis.set_xlim(-1.35, 2.35)
        axis.spines[["top", "right", "bottom"]].set_visible(False)
        axis.tick_params(axis="x", length=0)

    axes[0].set_ylabel(f"Mean rank over {len(audio_metrics)} audio metrics (1 = best)")
    axes[0].invert_yaxis()

    figure.savefig(args.out)
    print(
        f"wrote {args.out}  ({len(audio_metrics)} audio metrics, usetex={using_tex})"
    )


if __name__ == "__main__":
    main()
