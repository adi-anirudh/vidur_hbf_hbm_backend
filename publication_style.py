"""Single publication style for every paper-facing SPLASH evaluation plot."""
from __future__ import annotations

import matplotlib.pyplot as plt


# Colorblind-safe, print-distinguishable semantic palette.
COLORS = {
    "hbm": "#6b7280",
    "dense": "#9b3a32",
    "token": "#d08b2e",
    "full_score": "#b66d2c",
    "global": "#6b7f3b",
    "splash": "#176b87",
    "accent": "#3f6f9f",
    "ink": "#202124",
    "grid": "#d8d8d8",
}

LINE_WIDTH = 1.7
MARKER_SIZE = 4.0
BAR_EDGE_WIDTH = 0.4
ANNOTATION_SIZE = 6.5


def apply() -> None:
    """Apply consistent full-width-paper typography and vector export."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "mathtext.fontset": "dejavusans",
        "font.size": 7.5,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "legend.fontsize": 7.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "axes.linewidth": 0.7,
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "text.color": COLORS["ink"],
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": MARKER_SIZE,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def finish_axis(ax, *, grid_axis: str = "both") -> None:
    """Apply shared grid/spine treatment after plotting."""
    ax.grid(
        True, axis=grid_axis, linestyle=(0, (2.2, 2.2)),
        linewidth=0.55, color=COLORS["grid"], alpha=0.85,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def finish_twin_axis(ax) -> None:
    """Twin axes retain the right spine but match the rest of the style."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_linewidth(0.7)
    ax.spines["right"].set_color(COLORS["ink"])
