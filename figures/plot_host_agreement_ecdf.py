"""Cross-engine agreement ECDF figure for the thesis Discussion chapter.

Plots the empirical cumulative distribution (ECDF) of DawDreamer-vs-Pedalboard
agreement over the non-silent canonical (seed-0) patches from the D-RENDERER
benchmark. Only log-spectral distance (dB) is plotted.

Run:
    python figures/plot_host_agreement_ecdf.py
"""
import argparse
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from figures.style import apply_paper_style, FULL_WIDTH_IN, figure_size  # noqa: E402

_DEFAULT_CSV = _REPO_ROOT / "figures" / "data" / "host_agreement_seed0.csv"
_DEFAULT_PDF = _REPO_ROOT.parent / "thesis_latex" / "figures" / "discussion-host-agreement-ecdf.pdf"

# Percentiles marked on every curve, with a distinct marker each.
_PERCENTILES = ((50, "median", "o"), (90, "p90", "s"), (95, "p95", "^"))


def ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sorted values and their empirical cumulative fraction (1/n .. 1)."""
    ordered = np.sort(values)
    fraction = np.arange(1, len(ordered) + 1) / len(ordered)
    return ordered, fraction


def _draw_curve(axis, values: np.ndarray, color: str, label: str) -> None:
    ordered, fraction = ecdf(values)
    axis.plot(ordered, fraction, color=color, label=label)
    for percentile, _, marker in _PERCENTILES:
        x = np.percentile(values, percentile)
        axis.plot(x, percentile / 100.0, marker=marker, color=color,
                  markeredgecolor="white", markeredgewidth=0.4, linestyle="none", zorder=5)


def _percentile_legend(axis) -> None:
    """A neutral legend mapping marker shape -> percentile (shape, not colour, encodes it)."""
    handles = [
        Line2D([], [], marker=marker, color="0.35", linestyle="none", label=name)
        for _, name, marker in _PERCENTILES
    ]
    axis.legend(handles=handles, loc="lower right", title=None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(_DEFAULT_CSV), help="Per-patch agreement CSV.")
    parser.add_argument("--out", default=str(_DEFAULT_PDF), help="Output PDF path.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Agreement CSV not found: {csv_path}\n"
              f"Generate it with:\n"
              f"  python scripts/benchmark_renderers.py --num-patches 3000 --seed 0 "
              f"--dump-agreement-csv {csv_path}")
        sys.exit(1)

    data = pd.read_csv(csv_path)
    lsd = data["log_spectral_distance_db"].to_numpy()

    import matplotlib.pyplot as plt

    using_tex = apply_paper_style()
    # Changed to 1 subplot instead of 2, slightly smaller width aspect ratio so it doesn't look awkwardly wide
    fig, axis_lsd = plt.subplots(
        1, 1, figsize=figure_size(FULL_WIDTH_IN * 0.7, aspect=0.6)
    )

    _draw_curve(axis_lsd, lsd, color="#0072B2", label="Log-spectral distance")
    axis_lsd.set_xlabel("Log-spectral distance (dB)")
    axis_lsd.set_ylabel("Cumulative fraction of patches")
    axis_lsd.set_xlim(left=0.0)
    axis_lsd.set_ylim(0.0, 1.02)
    _percentile_legend(axis_lsd)

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote {args.out}  (n={len(data)} patches, usetex={using_tex})")


if __name__ == "__main__":
    main()
