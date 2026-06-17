# Figure style contract

The authoritative look for every figure that goes into the thesis. This document is the
human-readable spec; **`figures/paper.mplstyle` is its machine enforcement** and
`figures/style.py` applies it. The thesis (`../thesis_latex/`) consumes only the PDFs
produced here, so this repo owns the style. **Keep this file and `paper.mplstyle` in
sync** — if you change one, change the other.

## How to use it

```python
from figures.style import apply_paper_style, FULL_WIDTH_IN, figure_size
import matplotlib.pyplot as plt

apply_paper_style()                       # loads paper.mplstyle (+ usetex if available)
fig, ax = plt.subplots(figsize=figure_size(FULL_WIDTH_IN, aspect=0.5))
# ... draw ...
fig.savefig("../thesis_latex/figures/<name>.pdf")   # vector PDF, no in-figure title
```

## The choices (and why)

### Output
- **Vector PDF** (`savefig.format: pdf`, fonts embedded, `pdf.fonttype: 42`). Scales
  cleanly in print and keeps text selectable/searchable.
- **Constrained ("tight") layout** (`figure.constrained_layout.use: True`,
  `savefig.bbox: tight`) so panels are evenly spaced and nothing is clipped.
- **No in-figure title.** The LaTeX `\caption` owns the title; scripts must not call
  `suptitle`/`set_title` for the figure title. Axis labels and legends are fine.

### Sizing — author at physical size
- **Full width = 15 cm = 5.9 in** (`FULL_WIDTH_IN`), matching the thesis `\textwidth`.
- **Half width ≈ 7.3 cm ≈ 2.87 in** (`HALF_WIDTH_IN`).
- Figures are authored at their final physical width and included in LaTeX at that natural
  size (no `width=` rescale). Because LaTeX never rescales the figure, it never rescales
  the fonts — so 9 pt in the figure is 9 pt on the page.

### Fonts — Computer Modern, ~9 pt base
- `font.family: serif` with `font.serif: Computer Modern Roman, CMU Serif, DejaVu Serif`
  and `mathtext.fontset: cm`, so text and math read as Computer Modern to match the body.
- **Base size 9 pt**; ticks/legend 8 pt.
- **`text.usetex` is `False` in `paper.mplstyle`** so a figure regenerates on any machine
  without a LaTeX install. `apply_paper_style()` **auto-upgrades to `usetex` when a
  `latex` + `dvipng` stack is detected**, giving true Computer Modern; the `cm` mathtext
  fontset is the visual fallback when it is not. Same visual intent either way — only the
  text-rendering backend differs. Pass `apply_paper_style(prefer_usetex=False)` to force
  the fallback.

### Visuals
- **Colourblind-safe palette:** Okabe–Ito, set as the colour cycle
  (`0072B2` blue, `D55E00` vermillion, `009E73` green, `CC79A7` purple, `E69F00` orange,
  `56B4E9` sky, `000000` black). Yellow is dropped — too low-contrast on white.
- **Thin, consistent lines:** `lines.linewidth: 1.0`, `axes.linewidth: 0.6`.
- **Light grid behind the data:** grey `B0B0B0`, `linewidth 0.4`, `alpha 0.5`,
  `axes.axisbelow: True`.
- **Unobtrusive frame:** top and right spines off; ticks point outward.
- **Frameless legends** (`legend.frameon: False`).

## Regenerating figures
Each figure script reads a committed data file (CSV) in this repo and writes its PDF to
`../thesis_latex/figures/`. The data is generated separately (e.g. the cross-engine CSV
comes from `scripts/benchmark_renderers.py --dump-agreement-csv`, which needs the VST),
so the figure itself regenerates from the CSV alone — no plugin required.
