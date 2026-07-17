#!/usr/bin/env python3
"""Plot achieved throughput under common, preselected GPU caps and a TBT SLO.

For every context, session count, pruning setting, and method, this selects
the fastest configuration that both fits the common GPU cap for that session
count and meets the requested TBT SLO. Aggregate decode throughput is
sessions * 1000 / TBT_ms. The caps come from the same a-priori rule used by
the corrected fixed-budget figures: the unpruned balanced-static
heterogeneous baseline at 32K/100 ms, without observing dynamic results.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import corrected_microbatch_sweep as sweep
from make_corrected_microbatch_plots import (
    COLORS,
    FIXED_MICROBATCHES,
    best_at_budget,
    index_points,
    load_points,
    save,
)


def plot_throughput_at_slo(
    index,
    budgets: dict[int, int],
    outdir: Path,
    slo_ms: float,
) -> None:
    contexts = list(sweep.base.CONTEXTS)
    sessions_to_plot = (32, 64, 128, 256)
    pruning_settings = ((1.0, "No pruning (c=1.00)"), (0.98, "Pruning (c=.98)"))

    fig, axes = plt.subplots(
        len(sessions_to_plot),
        len(pruning_settings),
        figsize=(13.0, 13.5),
        sharex=True,
        sharey="row",
    )
    handles = []
    labels = []

    for row, sessions in enumerate(sessions_to_plot):
        for col, (retained, pruning_label) in enumerate(pruning_settings):
            ax = axes[row, col]
            budget = budgets[sessions]
            variants = [
                (
                    sweep.base.SYSTEM_HBM,
                    1.0,
                    None,
                    "All-HBM colocated",
                    sweep.base.SYSTEM_HBM,
                    "x:",
                ),
                (
                    sweep.base.SYSTEM_HBF,
                    1.0,
                    None,
                    "All-HBF colocated",
                    sweep.base.SYSTEM_HBF,
                    "o-",
                ),
                (
                    sweep.base.SYSTEM_STATIC,
                    retained,
                    FIXED_MICROBATCHES,
                    "Static heterogeneous EP",
                    "static",
                    "s--",
                ),
                (
                    sweep.base.SYSTEM_DYNAMIC,
                    retained,
                    FIXED_MICROBATCHES,
                    "Proposed dynamic EP",
                    "dynamic",
                    "D-",
                ),
            ]

            for system, method_retained, microbatches, label, color_key, style in variants:
                throughputs = []
                for context in contexts:
                    point = best_at_budget(
                        index[(context, sessions, method_retained, system)],
                        budget,
                        microbatches,
                    )
                    if point is not None and point.tbt_ms > slo_ms:
                        point = None
                    throughput = (
                        np.nan
                        if point is None
                        else sessions * 1000.0 / point.tbt_ms
                    )
                    throughputs.append(throughput)

                line = ax.plot(
                    [context // 1024 for context in contexts],
                    throughputs,
                    style,
                    color=COLORS[color_key],
                    linewidth=2.1,
                    markersize=7,
                    label=label,
                )[0]
                if row == 0 and col == 0:
                    handles.append(line)
                    labels.append(label)

            ax.set_title(f"S={sessions} · cap={budget} GPUs · {pruning_label}")
            ax.set_xscale("log", base=2)
            ax.set_xticks([8, 16, 32, 64, 128], ["8", "16", "32", "64", "128"])
            ax.grid(alpha=0.25)

        axes[row, 0].set_ylabel("aggregate throughput (tokens/s) ↑")

    for ax in axes[-1, :]:
        ax.set_xlabel("context length (K tokens)")

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.suptitle(
        f"Achieved throughput under common GPU caps and {slo_ms:g} ms TBT",
        fontsize=15,
    )
    fig.text(
        0.5,
        0.035,
        "Throughput = S × 1000 / achieved TBT; missing points are capacity- or SLO-infeasible. Caps are fixed from the unpruned static baseline at 32K/100 ms. Heterogeneous EP uses matched c and two microbatches.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.98))
    save(fig, outdir, f"throughput_at_{slo_ms:g}ms")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(
            Path(__file__).resolve().parent.parent
            / "results"
            / "corrected_microbatch_sweep.csv"
        ),
    )
    parser.add_argument(
        "--outdir",
        default=str(Path(__file__).resolve().parent.parent / "figures" / "current"),
    )
    parser.add_argument("--slo-ms", type=float, default=100.0)
    args = parser.parse_args()

    points = load_points(Path(args.csv))
    index = index_points(points)
    budgets = sweep.choose_budgets(sweep.group_points(points))
    plot_throughput_at_slo(index, budgets, Path(args.outdir), args.slo_ms)
    print(f"wrote throughput_at_{args.slo_ms:g}ms as PNG+PDF under {args.outdir}")


if __name__ == "__main__":
    main()
