"""Presentation layer for benchmark tables: names, families, and LaTeX formatting.

Separated from ``evaluation/aggregate.py`` so the statistics stay free of display concerns
and stay testable without any LaTeX in the picture. Nothing here computes a number.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# Model families, in the order the tables present them: the naive floor first, then each
# paper lineage. The grouping is what makes the tables answer the thesis's own question
# (which *family* of approach wins) rather than only which checkpoint won.
FAMILY_ORDER: List[str] = [
    "Baseline",
    "Discriminative (Chen et al.)",
    "Generative (Le Vaillant et al.)",
    "Neural proxy (Barkan et al.)",
    "Flow matching (Hayes et al.)",
    r"Reinforcement learning (Shin \& Lee)",
]

@dataclass(frozen=True)
class ModelPresentation:
    family: str
    display: str
    order: int


# Display names follow each paper's own naming, so a reader can match a row to its source.
# InverSynth II's "xITF" reads *excluding* inference-time finetuning, not a variant of it.
MODEL_PRESENTATION: Dict[str, ModelPresentation] = {
    "MeanParameterBaseline": ModelPresentation("Baseline", "Mean parameter", 0),
    "Sound2SynthSpectrogramRegressor": ModelPresentation("Discriminative (Chen et al.)", "Sound2Synth", 1),
    "PresetGenVAEMLPRegressor": ModelPresentation("Generative (Le Vaillant et al.)", "VAE + MLP", 2),
    "PresetGenVAEFlowRegressor": ModelPresentation("Generative (Le Vaillant et al.)", "VAE + RealNVP", 3),
    "IS": ModelPresentation("Neural proxy (Barkan et al.)", "IS", 4),
    "IS2xITF": ModelPresentation("Neural proxy (Barkan et al.)", "IS2xITF", 5),
    "IS2": ModelPresentation("Neural proxy (Barkan et al.)", "IS2", 6),
    "FlowMatchingMLP": ModelPresentation("Flow matching (Hayes et al.)", "CNF (MLP)", 7),
    "FlowMatchingParam2Tok": ModelPresentation("Flow matching (Hayes et al.)", "CNF (Param2Tok)", 8),
    "SynthRLp": ModelPresentation(r"Reinforcement learning (Shin \& Lee)", "SynthRL-p", 9),
    "SynthRLi": ModelPresentation(r"Reinforcement learning (Shin \& Lee)", "SynthRL-i", 10),
}

METRIC_DISPLAY: Dict[str, str] = {
    "param_mae": "Param MAE",
    "param_mse": "Param MSE",
    "param_accuracy": "Param acc.",
    "lsd": "LSD",
    "spectral_convergence": "Spec.\\ conv.",
    "mel_mae": "Mel MAE",
    "mel_mse": "Mel MSE",
    "mss": "MSS",
    "mfcc_mae": "MFCC MAE",
    "mfcc_mse": "MFCC MSE",
    "loudness_envelope_l1": "Loud.\\ env.",
    "integrated_loudness_error": "LUFS err.",
    "f0_rmse": "$F_0$ RMSE",
}

# Column headers are abbreviated so a six-metric table fits the thesis text width, which is
# also the convention in this literature's results tables. METRIC_DISPLAY carries the long
# form for prose and for the caption that expands these.
METRIC_SHORT: Dict[str, str] = {
    "param_mae": "P-MAE",
    "param_mse": "P-MSE",
    "param_accuracy": "P-Acc",
    "lsd": "LSD",
    "spectral_convergence": "SC",
    "mel_mae": "Mel-MAE",
    "mel_mse": "Mel-MSE",
    "mss": "MSS",
    "mfcc_mae": "MFCC",
    "mfcc_mse": "MFCC$^2$",
    "loudness_envelope_l1": "Loud",
    "integrated_loudness_error": "LUFS",
    "f0_rmse": "$F_0$",
}

SYNTH_DISPLAY: Dict[str, str] = {"dexed": "Dexed", "diva": "Diva"}


def sort_models(model_names: Sequence[str]) -> List[str]:
    """Order models by family, then by the presentation order within a family."""
    return sorted(model_names, key=lambda name: MODEL_PRESENTATION[name].order)


def decimals_for(half_widths: Sequence[float], values: Sequence[float]) -> int:
    """Decimal places sized to the confidence interval, not to the value.

    Follows the usual convention for reporting an uncertainty: one significant figure,
    widened to two only when the interval's leading digit is 1, where a single figure would
    round away up to half its size. Printing more digits than the interval justifies invents
    precision; printing fewer hides real differences between models.

    Falls back to the values themselves when every interval is degenerate.
    """
    usable = [abs(width) for width in half_widths if width and math.isfinite(width) and width > 0]
    if not usable:
        usable = [abs(value) for value in values if value and math.isfinite(value)]
    if not usable:
        return 2
    smallest = min(usable)
    exponent = math.floor(math.log10(smallest))
    leading_digit = smallest / (10 ** exponent)
    significant_figures = 2 if leading_digit < 2 else 1
    return int(min(4, max(0, significant_figures - 1 - exponent)))


def format_cell(
    mean: float,
    half_width: float,
    decimals: int,
    bold: bool = False,
) -> str:
    """One table cell: ``mean {\\scriptsize$\\pm$ half-width}``, bolded when it is the winner.

    An undefined metric prints an em dash rather than a number, which is how the parameter
    axis appears on an out-of-domain corpus (D-OOD: ``valid_count`` 0).
    """
    if mean is None or not math.isfinite(mean):
        return "---"
    body = f"{mean:.{decimals}f}"
    if bold:
        body = f"\\textbf{{{body}}}"
    if half_width is None or not math.isfinite(half_width):
        return body
    return f"{body}\\,{{\\scriptsize$\\pm${half_width:.{decimals}f}}}"


def format_parameter_count(count: Optional[int]) -> str:
    """Weight count as a compact ``12.0M``; an em dash for a model that has no network."""
    if count is None:
        return "---"
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    if count >= 1_000:
        return f"{count / 1_000:.0f}K"
    return str(count)


def metric_header(metric: str, higher_is_better: bool, short: bool = True) -> str:
    """Column header carrying the metric's direction, read from the panel, never typed."""
    arrow = "\\uparrow" if higher_is_better else "\\downarrow"
    name = METRIC_SHORT[metric] if short else METRIC_DISPLAY[metric]
    return f"{name} ${arrow}$"


def tabular(
    column_spec: str,
    header_rows: Sequence[Sequence[str]],
    body_rows: Sequence[Optional[Sequence[str]]],
    required_packages: Sequence[str] = ("booktabs",),
    note: str = "",
    font_size: str = "",
    column_separation: str = "",
) -> str:
    """Assemble a standalone ``tabular`` fragment for ``\\input``.

    Emits no ``table`` wrapper, caption or label: the writing session owns placement and
    wording, and a generated file that carried them would fight every edit made there. A
    ``None`` body row becomes a ``\\midrule``, which is how family groups are separated.
    """
    lines = [
        "% Generated by scripts/build_results_tables.py -- do not edit by hand.",
        "% Requires: " + ", ".join("\\usepackage{" + name + "}" for name in required_packages),
    ]
    if note:
        lines += [f"% {line}" for line in note.splitlines()]
    if font_size:
        lines.append("{\\" + font_size)
    if column_separation:
        lines.append(f"\\setlength{{\\tabcolsep}}{{{column_separation}}}")
    lines.append(f"\\begin{{tabular}}{{{column_spec}}}")
    lines.append("\\toprule")
    for header in header_rows:
        lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for row in body_rows:
        lines.append("\\midrule" if row is None else " & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if font_size:
        lines.append("}")
    return "\n".join(lines) + "\n"
