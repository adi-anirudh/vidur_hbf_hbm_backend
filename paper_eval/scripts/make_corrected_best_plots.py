#!/usr/bin/env python3
"""Paper-oriented plots for corrected_ctx_workload_sweep.csv.

Selections are deterministic:
* fixed budgets are recomputed using the corrected sweep's documented rule;
* resource-reallocation panels include every c=.98 cell where dynamic uses fewer
  MoE GPUs than balanced static at that fixed budget;
* Pareto panels use those same cells, ordered by dynamic TBT improvement.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import corrected_ctx_workload_sweep as sweep  # noqa: E402


COLORS = {
    sweep.SYSTEM_HBM: "#8c564b",
    sweep.SYSTEM_HBF: "#4c78a8",
    "static-100": "#9c9c9c",
    "dynamic-100": "#f28e2b",
    "static-098": "#7b6fd0",
    "dynamic-098": "#d62728",
}


def load_points(path: Path) -> list[sweep.Point]:
    points = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            points.append(
                sweep.Point(
                    context=int(row["context_tokens"]),
                    sessions=int(row["sessions"]),
                    retained_mass=float(row["retained_mass"]),
                    system=row["system"],
                    gpus=int(row["gpus"]),
                    attention_gpus=int(row["attention_gpus"]),
                    moe_gpus=int(row["moe_gpus"]),
                    attention_ms=float(row["attention_ms"]),
                    moe_ms=float(row["moe_ms"]),
                    tbt_ms=float(row["tbt_ms"]),
                    mean_retained_experts=float(row["mean_retained_experts"]),
                    mean_routes_per_token=float(row["mean_routes_per_token"]),
                )
            )
    return points


def best(points: list[sweep.Point], budget: int) -> sweep.Point | None:
    eligible = [point for point in points if point.gpus <= budget]
    return min(eligible, key=lambda p: (p.tbt_ms, p.gpus)) if eligible else None


def index_points(points: list[sweep.Point]):
    index = defaultdict(list)
    for point in points:
        index[(point.context, point.sessions, point.retained_mass, point.system)].append(point)
    return index


def save(fig, outdir: Path, filename: str):
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_regime_lines(index, budgets, outdir):
    """TBT vs context at fixed, baseline-selected budgets."""
    sessions_to_plot = (128, 256, 512, 1024)
    contexts = list(sweep.CONTEXTS)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True)
    variants = [
        (sweep.SYSTEM_HBF, 1.0, "All-HBF colocated", sweep.SYSTEM_HBF, "o-"),
        (sweep.SYSTEM_STATIC, 1.0, "Hetero static", "static-100", "s--"),
        (sweep.SYSTEM_DYNAMIC, 1.0, "Hetero dynamic", "dynamic-100", "D-"),
        (sweep.SYSTEM_STATIC, 0.98, "Hetero static + prune", "static-098", "^--"),
        (sweep.SYSTEM_DYNAMIC, 0.98, "Hetero dynamic + prune", "dynamic-098", "*-"),
    ]
    handles = []
    labels = []
    for ax, sessions in zip(axes.flat, sessions_to_plot):
        budget = budgets[sessions]
        for system, retained, label, color_key, style in variants:
            values = []
            for context in contexts:
                point = best(index[(context, sessions, retained, system)], budget)
                values.append(np.nan if point is None else point.tbt_ms)
            line = ax.plot(
                [context // 1024 for context in contexts],
                values,
                style,
                color=COLORS[color_key],
                linewidth=2.2,
                markersize=7,
                label=label,
            )[0]
            if not handles:
                pass
            if ax is axes.flat[0]:
                handles.append(line)
                labels.append(label)
        ax.axhline(100, color="#444444", linestyle=":", linewidth=1.3)
        ax.set_title(f"S={sessions}, budget={budget} GPUs")
        ax.set_ylabel("TBT (ms) ↓")
        ax.set_xticks([8, 16, 32, 64, 128])
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.25)
        ax.text(126, 102, "100 ms SLO", ha="right", va="bottom", fontsize=8, color="#444444")
    for ax in axes[-1, :]:
        ax.set_xlabel("context length (K tokens)")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Where heterogeneous HBM-attention/HBF-MoE helps", fontsize=15, y=1.01)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, outdir, "fig1_regime_tbt_vs_context.png")


def plot_fair_gain_heatmap(index, budgets, outdir):
    """Fair comparison: dynamic vs balanced static at identical c=.98."""
    contexts = list(sweep.CONTEXTS)
    sessions = list(sweep.CONCURRENCIES)
    gain = np.full((len(sessions), len(contexts)), np.nan)
    released = np.zeros_like(gain)
    for i, s in enumerate(sessions):
        for j, context in enumerate(contexts):
            budget = budgets[s]
            static = best(index[(context, s, 0.98, sweep.SYSTEM_STATIC)], budget)
            dynamic = best(index[(context, s, 0.98, sweep.SYSTEM_DYNAMIC)], budget)
            if static and dynamic:
                gain[i, j] = 100 * (1 - dynamic.tbt_ms / static.tbt_ms)
                released[i, j] = static.moe_gpus - dynamic.moe_gpus

    fig, ax = plt.subplots(figsize=(9.2, 5.7))
    image = ax.imshow(gain, cmap="YlOrRd", aspect="auto", vmin=0, vmax=np.nanmax(gain))
    ax.set_xticks(range(len(contexts)), [f"{context // 1024}K" for context in contexts])
    ax.set_yticks(range(len(sessions)), sessions)
    ax.set_xlabel("context length")
    ax.set_ylabel("concurrent decode sessions")
    ax.set_title("Pruning-aware dynamic EP vs balanced static EP\nSame c=.98 and same fixed GPU budget")
    for i in range(len(sessions)):
        for j in range(len(contexts)):
            if np.isfinite(gain[i, j]):
                marker = "  ★" if released[i, j] >= 1 else ""
                ax.text(j, i, f"{gain[i, j]:.1f}%{marker}", ha="center", va="center", fontsize=9)
            else:
                ax.text(j, i, "capacity\ninfeasible", ha="center", va="center", fontsize=7, color="#555555")
    fig.colorbar(image, ax=ax, label="TBT reduction (%)")
    fig.text(0.5, 0.01, "★ dynamic EP releases at least one MoE GPU to the attention pool", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, outdir, "fig2_fair_dynamic_gain_c098.png")


def release_cases(index, budgets):
    cases = []
    for context in sweep.CONTEXTS:
        for sessions in sweep.CONCURRENCIES:
            budget = budgets[sessions]
            static = best(index[(context, sessions, 0.98, sweep.SYSTEM_STATIC)], budget)
            dynamic = best(index[(context, sessions, 0.98, sweep.SYSTEM_DYNAMIC)], budget)
            if static and dynamic and dynamic.moe_gpus < static.moe_gpus:
                gain = 100 * (1 - dynamic.tbt_ms / static.tbt_ms)
                cases.append((gain, context, sessions, budget, static, dynamic))
    return sorted(cases, reverse=True)


def plot_resource_reallocation(index, budgets, outdir):
    cases = release_cases(index, budgets)
    if not cases:
        return
    labels = [f"{context // 1024}K / S={sessions}" for _, context, sessions, _, _, _ in cases]
    x = np.arange(len(cases))
    width = 0.35
    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(10, 7.8), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.4]}
    )

    static_tbt = [case[4].tbt_ms for case in cases]
    dynamic_tbt = [case[5].tbt_ms for case in cases]
    ax_top.bar(x - width / 2, static_tbt, width, color=COLORS["static-098"], label="Balanced static + prune")
    ax_top.bar(x + width / 2, dynamic_tbt, width, color=COLORS["dynamic-098"], label="Dynamic + prune")
    for i, case in enumerate(cases):
        gain = case[0]
        ax_top.text(i + width / 2, dynamic_tbt[i] + 0.8, f"−{gain:.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax_top.set_ylabel("TBT (ms) ↓")
    ax_top.set_title("Dynamic EP turns lower MoE time into more attention GPUs (c=.98)")
    ax_top.legend(frameon=False, ncol=2)
    ax_top.grid(axis="y", alpha=0.25)

    static_a = np.array([case[4].attention_gpus for case in cases])
    static_m = np.array([case[4].moe_gpus for case in cases])
    dynamic_a = np.array([case[5].attention_gpus for case in cases])
    dynamic_m = np.array([case[5].moe_gpus for case in cases])
    ax_bottom.bar(x - width / 2, static_a, width, color="#59a14f", label="Attention GPUs")
    ax_bottom.bar(x - width / 2, static_m, width, bottom=static_a, color="#bab0ac", label="MoE GPUs")
    ax_bottom.bar(x + width / 2, dynamic_a, width, color="#59a14f")
    ax_bottom.bar(x + width / 2, dynamic_m, width, bottom=dynamic_a, color="#f28e2b")
    for i, case in enumerate(cases):
        ax_bottom.text(i - width / 2, static_a[i] + static_m[i] + 0.4, f"{static_a[i]}A:{static_m[i]}M", ha="center", fontsize=9)
        ax_bottom.text(i + width / 2, dynamic_a[i] + dynamic_m[i] + 0.4, f"{dynamic_a[i]}A:{dynamic_m[i]}M", ha="center", fontsize=9, fontweight="bold")
    ax_bottom.set_ylabel("GPU allocation")
    ax_bottom.set_xticks(x, labels)
    ax_bottom.set_xlabel("context / workload")
    ax_bottom.grid(axis="y", alpha=0.25)
    handles, legend_labels = ax_bottom.get_legend_handles_labels()
    ax_bottom.legend(handles[:2], legend_labels[:2], frameon=False, ncol=2)
    fig.tight_layout()
    save(fig, outdir, "fig3_resource_reallocation_c098.png")


def pareto(points):
    ordered = sorted(points, key=lambda p: (p.gpus, p.tbt_ms))
    result = []
    best_tbt = math.inf
    for point in ordered:
        if point.tbt_ms < best_tbt - 1e-9:
            result.append(point)
            best_tbt = point.tbt_ms
    return result


def plot_release_frontiers(index, budgets, outdir):
    cases = release_cases(index, budgets)[:3]
    if not cases:
        return
    fig, axes = plt.subplots(1, len(cases), figsize=(5.2 * len(cases), 4.8), squeeze=False)
    variants = [
        (sweep.SYSTEM_HBF, 1.0, "All-HBF colocated", sweep.SYSTEM_HBF, "o-"),
        (sweep.SYSTEM_STATIC, 1.0, "Static", "static-100", "s--"),
        (sweep.SYSTEM_DYNAMIC, 1.0, "Dynamic", "dynamic-100", "D-"),
        (sweep.SYSTEM_STATIC, 0.98, "Static + prune", "static-098", "^--"),
        (sweep.SYSTEM_DYNAMIC, 0.98, "Dynamic + prune", "dynamic-098", "*-"),
    ]
    for ax, (_, context, sessions, budget, _, _) in zip(axes.flat, cases):
        lower = max(2, budget // 2)
        upper = min(sweep.MAX_TOTAL_GPUS, int(math.ceil(budget * 1.5)))
        for system, retained, label, color_key, style in variants:
            candidates = [
                p for p in index[(context, sessions, retained, system)]
                if lower <= p.gpus <= upper
            ]
            front = pareto(candidates)
            ax.plot(
                [p.gpus for p in front], [p.tbt_ms for p in front], style,
                color=COLORS[color_key], linewidth=2, markersize=6, label=label,
            )
        ax.axvline(budget, color="#333333", linestyle=":", linewidth=1)
        ax.axhline(100, color="#777777", linestyle=":", linewidth=1)
        ax.set_title(f"{context // 1024}K, S={sessions}")
        ax.set_xlabel("total GPUs")
        ax.set_ylabel("TBT (ms) ↓")
        ax.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Cost–latency frontiers for every c=.98 GPU-reallocation case", y=1.02)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    save(fig, outdir, "fig4_cost_latency_release_cases.png")


def plot_pruning_sensitivity(index, budgets, outdir):
    masses = list(sweep.RETAINED_MASS)
    gains_by_mass = []
    releases = []
    for retained in masses:
        gains = []
        released = 0
        feasible = 0
        for context in sweep.CONTEXTS:
            for sessions in sweep.CONCURRENCIES:
                budget = budgets[sessions]
                static = best(index[(context, sessions, retained, sweep.SYSTEM_STATIC)], budget)
                dynamic = best(index[(context, sessions, retained, sweep.SYSTEM_DYNAMIC)], budget)
                if static and dynamic:
                    feasible += 1
                    gains.append(100 * (1 - dynamic.tbt_ms / static.tbt_ms))
                    released += dynamic.moe_gpus < static.moe_gpus
        gains_by_mass.append(gains)
        releases.append((released, feasible))

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    positions = np.arange(len(masses))
    box = ax.boxplot(gains_by_mass, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    for patch in box["boxes"]:
        patch.set_facecolor("#f6bd60")
        patch.set_alpha(0.8)
    ax.set_xticks(positions, [f"{mass:.2f}" for mass in masses])
    ax.set_xlabel("retained gate-mass target c")
    ax.set_ylabel("Dynamic TBT improvement over balanced static (%)")
    ax.set_title("Dynamic scheduling becomes more valuable as pruning creates placement holes")
    ax.grid(axis="y", alpha=0.25)
    for i, (released, feasible) in enumerate(releases):
        ax.text(i, ax.get_ylim()[1] * 0.94, f"GPU moved\n{released}/{feasible}", ha="center", va="top", fontsize=8)
    fig.tight_layout()
    save(fig, outdir, "fig5_pruning_sensitivity.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(SCRIPT_DIR.parent / "results" / "corrected_ctx_workload_sweep.csv"),
    )
    parser.add_argument(
        "--outdir",
        default=str(SCRIPT_DIR.parent / "figures" / "corrected" / "best_results"),
    )
    args = parser.parse_args()
    points = load_points(Path(args.csv))
    index = index_points(points)
    budgets = sweep.choose_fixed_budgets(sweep.group_points(points))
    outdir = Path(args.outdir)
    plot_regime_lines(index, budgets, outdir)
    plot_fair_gain_heatmap(index, budgets, outdir)
    plot_resource_reallocation(index, budgets, outdir)
    plot_release_frontiers(index, budgets, outdir)
    plot_pruning_sensitivity(index, budgets, outdir)
    print(f"wrote 5 figures under {outdir}")


if __name__ == "__main__":
    main()
