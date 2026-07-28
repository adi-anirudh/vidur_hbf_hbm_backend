"""Shared style for SPLASH Background/Motivation figures. Import and call apply().
ONE font (DejaVu Sans), ONE standard font-size scheme, ONE aesthetic, ONE wide-SHORT
shape -- so every motivation figure renders at the SAME small height and the SAME font
size when included at \\columnwidth.

Rendered height at \\columnwidth = columnwidth / (WIDTH/HEIGHT). WIDTH is fixed for every
figure, so all figures scale identically -> identical rendered font size. HEIGHT is small
and the canvas is wide, so the figures are short in the paper.

Use ms.FS_* for EVERY explicit fontsize (annotations, colorbar labels, ...) so nothing
drifts from the standard."""
import matplotlib.pyplot as plt

WIDTH = 5.6           # native width (in) — IDENTICAL for every figure => uniform font size
HEIGHT = 1.8          # native height (in) — short (wide:tall ~3.1 -> ~1.4in at columnwidth)
W_COL = WIDTH         # back-compat aliases
W_WIDE = WIDTH
LW = 2.2              # data line width
MS = 6.0              # marker size
MEW = 1.0             # marker white-edge width
GRID = dict(ls=(0, (4, 4)), lw=0.7, color="#dadada")
INK = "#111111"
PAL = {"dark": "#111111", "primary": "#c15a1b", "gold": "#e6a92c",
       "mid": "#8c4a1e", "rust": "#a34c14", "fill": "#f0d9c4"}

# ONE standard size scheme — used everywhere, including explicit fontsizes below.
FS_LABEL = 12.0       # axis labels
FS_TICK = 11.0        # tick labels
FS_LEGEND = 11.0      # legends
FS_ANNOT = 11.0       # on-plot annotations, colorbar labels — SAME across all figures


def apply():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
        "mathtext.fontset": "dejavusans",
        "font.size": FS_TICK, "axes.titlesize": FS_LABEL, "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEGEND,
        "axes.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 400,
        "axes.edgecolor": "#222222", "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": "#222222", "ytick.color": "#222222"})


def grid(ax):
    ax.grid(True, which="major", zorder=0, **GRID)
    ax.set_axisbelow(True)


def despine(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
