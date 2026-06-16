"""Reusable plotting setup for thesis figures.

Every figure script applies the paper style through :func:`apply_paper_style` and sizes
its figure with :data:`FULL_WIDTH_IN` / :data:`HALF_WIDTH_IN`, so the whole figure set
stays visually consistent. The authoritative style values live in ``paper.mplstyle`` next
to this file; the human-readable contract is ``docs/figure-style.md``.

Authoring rule: size figures at their *physical* width here (centimetres → inches) and
include them in LaTeX at that natural size, so LaTeX never rescales — and therefore never
rescales — the fonts.
"""
import shutil
from pathlib import Path
from typing import Tuple

import matplotlib

_STYLE_PATH = Path(__file__).resolve().parent / "paper.mplstyle"

# Thesis text width is 15 cm. Author full-width figures at exactly that, half-width at ~7.3 cm.
_CM_PER_INCH = 2.54
FULL_WIDTH_IN: float = 15.0 / _CM_PER_INCH   # 5.91 in
HALF_WIDTH_IN: float = 7.3 / _CM_PER_INCH    # 2.87 in


def _latex_stack_available() -> bool:
    """True only if the full matplotlib usetex pipeline (latex + dvipng) is present."""
    return shutil.which("latex") is not None and shutil.which("dvipng") is not None


def apply_paper_style(prefer_usetex: bool = True) -> bool:
    """Apply ``paper.mplstyle`` and return whether real LaTeX (usetex) is in effect.

    ``paper.mplstyle`` pins ``text.usetex: False`` so a figure regenerates on any machine.
    When ``prefer_usetex`` is set and a latex+dvipng stack is found, usetex is switched on
    for true Computer Modern; otherwise the cm mathtext fallback in the style is used. The
    visual intent is identical either way — only the text rendering backend differs.
    """
    import matplotlib.pyplot as plt

    plt.style.use(str(_STYLE_PATH))
    use_tex = bool(prefer_usetex) and _latex_stack_available()
    matplotlib.rcParams["text.usetex"] = use_tex
    if use_tex:
        # Match the cm look the fallback gives, under a real TeX engine.
        matplotlib.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}"
    return use_tex


def figure_size(width_in: float, aspect: float = 0.6) -> Tuple[float, float]:
    """(width, height) in inches for a given physical width and height:width ratio."""
    return (width_in, width_in * aspect)
