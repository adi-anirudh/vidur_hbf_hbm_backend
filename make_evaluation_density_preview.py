#!/usr/bin/env python3
"""Raster review sheet for checking six single-column figures at paper scale.

The individual PDFs remain the publication artifacts; this sheet is only a
layout-density preview.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
PLOTS = ROOT / "results" / "plots" / "eval_result_style_review"
PAGE_WIDTH = 7.2
PAGE_HEIGHT = 8.35


def place(fig, name: str, x: float, top: float, width: float) -> float:
    image = plt.imread(PLOTS / f"{name}.png")
    height = (
        width * PAGE_WIDTH * image.shape[0] / image.shape[1] / PAGE_HEIGHT
    )
    axis = fig.add_axes([x, top - height, width, height])
    axis.imshow(image)
    axis.axis("off")
    return top - height


def main() -> None:
    fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT), facecolor="white")
    column_width = 0.475
    left_x, right_x = 0.015, 0.510
    top = 0.985
    gap = 0.010

    left_top = top
    for name in (
        "attention_mechanism",
        "granularity_quality_performance",
        "ablation_validated_128k",
    ):
        left_top = place(fig, name, left_x, left_top, column_width) - gap

    right_top = top
    for name in (
        "sensitivity_design_knobs",
        "plane_balance_crossbenchmark",
        "energy_projection",
    ):
        right_top = place(fig, name, right_x, right_top, column_width) - gap

    fig.text(
        0.5, 0.006,
        "Density preview only — use the six vector PDFs for publication.",
        ha="center", va="bottom", fontsize=6, color="#666666",
    )
    for ext in ("png", "pdf"):
        fig.savefig(
            PLOTS / f"publication_density_preview.{ext}",
            dpi=300, facecolor="white",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
