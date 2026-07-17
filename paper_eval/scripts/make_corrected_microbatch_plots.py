#!/usr/bin/env python3
"""Paper-oriented plots from the corrected per-microbatch sweep.

Heterogeneous HBM-attention/HBF-MoE systems are evaluated at two
microbatches. Colocated baselines are allowed to choose their best
microbatch count; forcing them to use two would unfairly add repeated weight
reads without enabling cross-pool overlap.
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
from matplotlib.ticker import MaxNLocator, ScalarFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import corrected_microbatch_sweep as sweep  # noqa: E402


FIXED_MICROBATCHES = 2
COLORS = {
    sweep.base.SYSTEM_HBM: "#8c564b",
    sweep.base.SYSTEM_HBF: "#4c78a8",
    "static-100": "#9c9c9c",
    "dynamic-100": "#f28e2b",
    "static-098": "#7b6fd0",
    "dynamic-098": "#d62728",
    "static": "#7b6fd0",
    "dynamic": "#d62728",
    "attention": "#59a14f",
    "moe": "#f28e2b",
    "tbt": "#222222",
}


def load_points(path: Path) -> list[sweep.MBPoint]:
    points = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            points.append(
                sweep.MBPoint(
                    context=int(row["context_tokens"]),
                    sessions=int(row["sessions"]),
                    retained_mass=float(row["retained_mass"]),
                    system=row["system"],
                    gpus=int(row["gpus"]),
                    attention_gpus=int(row["attention_gpus"]),
                    moe_gpus=int(row["moe_gpus"]),
                    microbatches=int(row["microbatches"]),
                    attention_ms=float(row["attention_ms"]),
                    moe_ms=float(row["moe_ms"]),
                    tbt_ms=float(row["tbt_ms"]),
                    mean_retained_experts=float(row["mean_retained_experts"]),
                    mean_routes_per_token=float(row["mean_routes_per_token"]),
                )
            )
    return points


def index_points(points: list[sweep.MBPoint]):
    index = defaultdict(list)
    for point in points:
        index[(point.context, point.sessions, point.retained_mass, point.system)].append(point)
    return index


def best_at_budget(
    points: list[sweep.MBPoint],
    budget: int,
    microbatches: int | None = None,
) -> sweep.MBPoint | None:
    eligible = [
        point
        for point in points
        if point.gpus <= budget
        and (microbatches is None or point.microbatches == microbatches)
    ]
    return min(
        eligible,
        key=lambda point: (point.tbt_ms, point.gpus, point.microbatches, point.moe_gpus),
    ) if eligible else None


def min_gpus_at_slo(
    points: list[sweep.MBPoint],
    slo_ms: float,
    microbatches: int | None = None,
) -> sweep.MBPoint | None:
    eligible = [
        point
        for point in points
        if point.tbt_ms <= slo_ms
        and (microbatches is None or point.microbatches == microbatches)
    ]
    return min(
        eligible,
        key=lambda point: (point.gpus, point.tbt_ms, point.microbatches),
    ) if eligible else None


def save(fig, outdir: Path, filename: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{filename}.png", dpi=200, bbox_inches="tight")
    fig.savefig(outdir / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_regime_lines(index, budgets, outdir: Path) -> None:
    sessions_to_plot = (32, 64)
    contexts = list(sweep.base.CONTEXTS[:-1])
    pruning_settings = ((1.0, "No pruning (c=1.00)"), (0.98, "Pruning (c=.98)"))
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.0), sharex=True, sharey="row")
    handles = []
    labels = []
    for row, sessions in enumerate(sessions_to_plot):
        budget = budgets[sessions]
        for col, (retained, pruning_label) in enumerate(pruning_settings):
            ax = axes[row, col]
            variants = [
                (sweep.base.SYSTEM_HBM, 1.0, None, "All-HBM colocated", sweep.base.SYSTEM_HBM, "x:"),
                (sweep.base.SYSTEM_HBF, 1.0, None, "All-HBF colocated", sweep.base.SYSTEM_HBF, "o-"),
                (sweep.base.SYSTEM_STATIC, retained, 2, "Static heterogeneous EP", "static", "^--"),
                (sweep.base.SYSTEM_DYNAMIC, retained, 2, "Proposed dynamic EP", "dynamic", "*-"),
            ]
            for system, method_retained, microbatches, label, color_key, style in variants:
                values = []
                for context in contexts:
                    point = best_at_budget(
                        index[(context, sessions, method_retained, system)],
                        budget,
                        microbatches,
                    )
                    values.append(np.nan if point is None else point.tbt_ms)
                line = ax.plot(
                    [context // 1024 for context in contexts],
                    values,
                    style,
                    color=COLORS[color_key],
                    linewidth=2.1,
                    markersize=7,
                    label=label,
                )[0]
                if row == 0 and col == 0:
                    handles.append(line)
                    labels.append(label)
            ax.axhline(100, color="#444444", linestyle=":", linewidth=1.2)
            ax.set_title(f"{pruning_label}\nS={sessions}, budget={budget} GPUs")
            ax.set_xscale("log", base=2)
            ax.set_xticks([8, 16, 32, 64], ["8", "16", "32", "64"])
            ax.grid(alpha=0.25)
            ax.text(0.03, 0.94, "All-HBM: capacity infeasible", transform=ax.transAxes, va="top", fontsize=8, color=COLORS[sweep.base.SYSTEM_HBM])
        axes[row, 0].set_ylabel("TBT (ms) ↓")
    for ax in axes[-1, :]:
        ax.set_xlabel("context length (K tokens)")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("End-to-end TBT with and without expert pruning", fontsize=15)
    fig.text(
        0.5,
        0.025,
        "Static and dynamic heterogeneous EP use the same c within each panel; colocated baselines are unpruned.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.10, 1, 0.97))
    save(fig, outdir, "fig6_system_advantage")


def plot_dynamic_gain_heatmap(index, budgets, outdir: Path) -> None:
    contexts = list(sweep.base.CONTEXTS)
    sessions = list(sweep.base.CONCURRENCIES)
    pruning_settings = ((1.0, "No pruning (c=1.00)"), (0.98, "Pruning (c=.98)"))
    matrices = []
    for retained, _ in pruning_settings:
        gain = np.full((len(sessions), len(contexts)), np.nan)
        released = np.zeros_like(gain)
        for i, active_sessions in enumerate(sessions):
            for j, context in enumerate(contexts):
                budget = budgets[active_sessions]
                static = best_at_budget(
                    index[(context, active_sessions, retained, sweep.base.SYSTEM_STATIC)], budget, 2
                )
                dynamic = best_at_budget(
                    index[(context, active_sessions, retained, sweep.base.SYSTEM_DYNAMIC)], budget, 2
                )
                if static and dynamic:
                    gain[i, j] = 100 * (1 - dynamic.tbt_ms / static.tbt_ms)
                    released[i, j] = static.moe_gpus - dynamic.moe_gpus
        matrices.append((gain, released))

    finite = np.concatenate([gain[np.isfinite(gain)] for gain, _ in matrices])
    vmax = max(1.0, math.ceil(float(np.max(finite))))
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.8), sharex=True, sharey=True)
    image = None
    for panel, (ax, (_, pruning_label), (gain, released)) in enumerate(
        zip(axes, pruning_settings, matrices)
    ):
        image = ax.imshow(gain, cmap="YlOrRd", aspect="auto", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(contexts)), [f"{context // 1024}K" for context in contexts])
        ax.set_yticks(range(len(sessions)), sessions)
        ax.set_xlabel("context length")
        if panel == 0:
            ax.set_ylabel("concurrent decode sessions")
        ax.set_title(pruning_label)
        for i in range(len(sessions)):
            for j in range(len(contexts)):
                if np.isfinite(gain[i, j]):
                    marker = "  ★" if released[i, j] >= 1 else ""
                    ax.text(j, i, f"{gain[i, j]:.1f}%{marker}", ha="center", va="center", fontsize=8.5)
                else:
                    ax.text(j, i, "capacity\ninfeasible", ha="center", va="center", fontsize=7, color="#555555")
    fig.suptitle("Dynamic EP versus balanced static EP at matched pruning", fontsize=15)
    colorbar_ax = fig.add_axes([0.91, 0.14, 0.018, 0.68])
    fig.colorbar(image, cax=colorbar_ax, label="TBT reduction (%)")
    fig.text(0.5, 0.01, "★ dynamic EP moves at least one GPU from MoE to attention", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.12, top=0.84, wspace=0.12)
    save(fig, outdir, "fig4_dynamic_vs_static")


def plot_microbatch_sensitivity(index, budgets, outdir: Path) -> None:
    medians = []
    retained_medians = []
    feasible_counts = []
    for microbatches in sweep.MICROBATCH_COUNTS:
        tbts = []
        retained = []
        for context in sweep.base.CONTEXTS:
            for sessions in sweep.base.CONCURRENCIES:
                point = best_at_budget(
                    index[(context, sessions, 0.98, sweep.base.SYSTEM_DYNAMIC)],
                    budgets[sessions],
                    microbatches,
                )
                if point:
                    tbts.append(point.tbt_ms)
                    retained.append(point.mean_retained_experts)
        medians.append(float(np.median(tbts)))
        retained_medians.append(float(np.median(retained)))
        feasible_counts.append(len(tbts))

    x = np.arange(len(sweep.MICROBATCH_COUNTS))
    fig, ax_tbt = plt.subplots(figsize=(8.7, 5.3))
    bars = ax_tbt.bar(x, medians, width=0.62, color="#4c78a8", alpha=0.88, label="Median TBT")
    bars[1].set_color("#d62728")
    ax_tbt.set_xticks(x, [str(value) for value in sweep.MICROBATCH_COUNTS])
    ax_tbt.set_xlabel("number of microbatches μ")
    ax_tbt.set_ylabel("median TBT (ms) ↓")
    ax_tbt.grid(axis="y", alpha=0.25)
    for i, (bar, value) in enumerate(zip(bars, medians)):
        ax_tbt.text(bar.get_x() + bar.get_width() / 2, value + 4, f"{value:.1f}", ha="center", fontsize=9)
        ax_tbt.text(bar.get_x() + bar.get_width() / 2, 3, f"n={feasible_counts[i]}", ha="center", fontsize=8, color="white")

    ax_experts = ax_tbt.twinx()
    line = ax_experts.plot(
        x,
        retained_medians,
        "o-",
        color="#f28e2b",
        linewidth=2.2,
        label="Active experts / microbatch",
    )[0]
    ax_experts.set_ylabel("median active experts per microbatch")
    ax_experts.set_ylim(bottom=0)
    ax_tbt.set_title("Two microbatches best balance overlap and repeated weight reads\nDynamic EP, c=.98, fixed baseline-selected budgets")
    fig.legend([bars, line], ["Median TBT", "Active experts / microbatch"], frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save(fig, outdir, "fig3_microbatch_sensitivity")


def plot_selected_breakdown(index, budgets, outdir: Path) -> None:
    context = 65536
    sessions = 128
    budget = budgets[sessions]
    points = [
        best_at_budget(
            index[(context, sessions, 0.98, sweep.base.SYSTEM_DYNAMIC)],
            budget,
            microbatches,
        )
        for microbatches in sweep.MICROBATCH_COUNTS
    ]
    if any(point is None for point in points):
        raise RuntimeError("selected microbatch-breakdown case is unexpectedly infeasible")

    x = np.arange(len(points))
    width = 0.34
    attention = [point.attention_ms for point in points]
    moe = [point.moe_ms for point in points]
    tbt = [point.tbt_ms for point in points]
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax.bar(x - width / 2, attention, width, color=COLORS["attention"], label="Total attention work")
    ax.bar(x + width / 2, moe, width, color=COLORS["moe"], label="Total MoE work")
    ax.plot(x, tbt, "o-", color=COLORS["tbt"], linewidth=2.2, label="Pipeline TBT")
    for i, point in enumerate(points):
        ax.text(i, max(attention[i], moe[i]) + 12, f"{point.attention_gpus}A:{point.moe_gpus}M", ha="center", fontsize=9)
        ax.text(i, tbt[i] - 12 if tbt[i] > 35 else tbt[i] + 8, f"{tbt[i]:.1f}", ha="center", fontsize=9, color=COLORS["tbt"])
    ax.set_xticks(x, [f"μ{point.microbatches}\n{sessions // point.microbatches} sessions/MB" for point in points])
    ax.set_ylabel("time/work (ms)")
    ax.set_xlabel("microbatch count and size")
    ax.set_title("Why μ=2 wins at 64K context, S=128, budget=16\nDynamic EP with c=.98; A:M optimized independently")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=4, loc="upper left")
    fig.text(0.5, 0.01, "μ1 is serial (TBT=A+M); μ≥2 uses the steady-state overlap bound TBT=max(A,M).", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, outdir, "diagnostic_selected_workload_breakdown")


def plot_gpu_cost_at_slo(index, outdir: Path) -> None:
    contexts = list(sweep.base.CONTEXTS)
    sessions_to_plot = (32, 64, 128, 256)
    pruning_settings = ((1.0, "No pruning (c=1.00)"), (0.98, "Pruning (c=.98)"))
    fig, axes = plt.subplots(4, 2, figsize=(13.0, 13.5), sharex=True, sharey="row")
    handles = []
    labels = []
    for row, sessions in enumerate(sessions_to_plot):
        for col, (retained, pruning_label) in enumerate(pruning_settings):
            ax = axes[row, col]
            variants = [
                (sweep.base.SYSTEM_HBM, 1.0, None, "All-HBM colocated", sweep.base.SYSTEM_HBM, "x:"),
                (sweep.base.SYSTEM_HBF, 1.0, None, "All-HBF colocated", sweep.base.SYSTEM_HBF, "o-"),
                (sweep.base.SYSTEM_STATIC, retained, 2, "Static heterogeneous EP", "static", "s--"),
                (sweep.base.SYSTEM_DYNAMIC, retained, 2, "Proposed dynamic EP", "dynamic", "D-"),
            ]
            for system, method_retained, microbatches, label, color_key, style in variants:
                values = []
                for context in contexts:
                    point = min_gpus_at_slo(
                        index[(context, sessions, method_retained, system)],
                        100.0,
                        microbatches,
                    )
                    values.append(np.nan if point is None else point.gpus)
                line = ax.plot(
                    [context // 1024 for context in contexts],
                    values,
                    style,
                    color=COLORS[color_key],
                    linewidth=2.1,
                    markersize=7,
                    label=label,
                )[0]
                if row == 0 and col == 0:
                    handles.append(line)
                    labels.append(label)
            ax.set_title(f"S={sessions} · {pruning_label}")
            ax.set_xscale("log", base=2)
            ax.set_xticks([8, 16, 32, 64, 128], ["8", "16", "32", "64", "128"])
            ax.grid(alpha=0.25)
        axes[row, 0].set_ylabel("minimum GPUs ↓")
    for ax in axes[-1, :]:
        ax.set_xlabel("context length (K tokens)")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Minimum GPUs for 100 ms TBT with and without expert pruning", fontsize=15)
    fig.text(0.5, 0.035, "Static and dynamic heterogeneous EP use matched c and two microbatches; colocated baselines are unpruned and choose their best microbatch count.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 0.98))
    save(fig, outdir, "fig7_gpu_cost_100ms")


def plot_tbt_at_slo_gpu_counts(index, outdir: Path) -> None:
    """Diagnostic: achieved TBT at each method-specific minimum-GPU point."""
    contexts = list(sweep.base.CONTEXTS)
    sessions_to_plot = (32, 64, 128, 256)
    pruning_settings = ((1.0, "No pruning (c=1.00)"), (0.98, "Pruning (c=.98)"))
    fig, axes = plt.subplots(4, 2, figsize=(13.0, 13.5), sharex=True, sharey=True)
    handles = []
    labels = []
    for row, sessions in enumerate(sessions_to_plot):
        for col, (retained, pruning_label) in enumerate(pruning_settings):
            ax = axes[row, col]
            variants = [
                (sweep.base.SYSTEM_HBM, 1.0, None, "All-HBM colocated", sweep.base.SYSTEM_HBM, "x:"),
                (sweep.base.SYSTEM_HBF, 1.0, None, "All-HBF colocated", sweep.base.SYSTEM_HBF, "o-"),
                (sweep.base.SYSTEM_STATIC, retained, 2, "Static heterogeneous EP", "static", "s--"),
                (sweep.base.SYSTEM_DYNAMIC, retained, 2, "Proposed dynamic EP", "dynamic", "D-"),
            ]
            x_values = [context // 1024 for context in contexts]
            for system, method_retained, microbatches, label, color_key, style in variants:
                selected = [
                    min_gpus_at_slo(
                        index[(context, sessions, method_retained, system)],
                        100.0,
                        microbatches,
                    )
                    for context in contexts
                ]
                values = [np.nan if point is None else point.tbt_ms for point in selected]
                line = ax.plot(
                    x_values,
                    values,
                    style,
                    color=COLORS[color_key],
                    linewidth=2.1,
                    markersize=7,
                    label=label,
                )[0]
                if row == 0 and col == 0:
                    handles.append(line)
                    labels.append(label)
            ax.axhline(100, color="#444444", linestyle=":", linewidth=1.2)
            ax.set_title(f"S={sessions} · {pruning_label}")
            ax.set_xscale("log", base=2)
            ax.set_xticks([8, 16, 32, 64, 128], ["8", "16", "32", "64", "128"])
            ax.grid(alpha=0.25)
        axes[row, 0].set_ylabel("achieved TBT (ms) ↓")
    for ax in axes[-1, :]:
        ax.set_xlabel("context length (K tokens)")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("TBT achieved by method-specific minimum-GPU configurations", fontsize=15)
    fig.text(
        0.5,
        0.035,
        "Diagnostic only: curves use different method-specific GPU counts and are not fixed-budget comparisons.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.98))
    save(fig, outdir, "diagnostic_tbt_at_min_gpu")


def plot_gpu_tbt_pareto(index, outdir: Path) -> None:
    """Plot common-budget TBT curves and mark each method's 100 ms crossing."""
    context = 131072
    sessions_to_plot = (32, 64, 128, 256)
    pruning_settings = ((1.0, "No pruning (c=1.00)"), (0.98, "Pruning (c=.98)"))
    fig, axes = plt.subplots(4, 2, figsize=(13.0, 13.5), sharey=True)
    handles = []
    labels = []
    for row, sessions in enumerate(sessions_to_plot):
        for col, (retained, pruning_label) in enumerate(pruning_settings):
            ax = axes[row, col]
            variants = [
                (sweep.base.SYSTEM_HBM, 1.0, None, "All-HBM colocated", sweep.base.SYSTEM_HBM, "x:"),
                (sweep.base.SYSTEM_HBF, 1.0, None, "All-HBF colocated", sweep.base.SYSTEM_HBF, "o-"),
                (sweep.base.SYSTEM_STATIC, retained, 2, "Static heterogeneous EP", "static", "s--"),
                (sweep.base.SYSTEM_DYNAMIC, retained, 2, "Proposed dynamic EP", "dynamic", "D-"),
            ]
            slo_points = [
                min_gpus_at_slo(index[(context, sessions, method_retained, system)], 100.0, microbatches)
                for system, method_retained, microbatches, _, _, _ in variants
            ]
            feasible_slo_points = [point for point in slo_points if point is not None]
            if not feasible_slo_points:
                raise RuntimeError(f"no 100 ms Pareto crossing for S={sessions}, c={retained}")
            min_budget = max(2, min(point.gpus for point in feasible_slo_points) - 4)
            max_budget = max(point.gpus for point in feasible_slo_points) + 4
            budgets = list(range(min_budget, max_budget + 1))

            for variant_index, (system, method_retained, microbatches, label, color_key, style) in enumerate(variants):
                values = []
                for budget in budgets:
                    point = best_at_budget(
                        index[(context, sessions, method_retained, system)],
                        budget,
                        microbatches,
                    )
                    values.append(np.nan if point is None else point.tbt_ms)
                line = ax.plot(
                    budgets,
                    values,
                    style,
                    color=COLORS[color_key],
                    linewidth=2.1,
                    markersize=5,
                    markevery=2,
                    label=label,
                )[0]
                if row == 0 and col == 0:
                    handles.append(line)
                    labels.append(label)
                crossing = slo_points[variant_index]
                if crossing is not None:
                    ax.scatter(
                        crossing.gpus,
                        crossing.tbt_ms,
                        s=95,
                        facecolors="none",
                        edgecolors=COLORS[color_key],
                        linewidths=2.0,
                        zorder=6,
                    )
            ax.axhline(100, color="#222222", linestyle=":", linewidth=1.3)
            ax.set_title(f"S={sessions} · {pruning_label}")
            ax.set_xlim(min_budget, max_budget)
            ax.set_yscale("log")
            ax.set_ylim(40, 1300)
            ax.set_yticks([50, 100, 200, 500, 1000])
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
            ax.grid(alpha=0.25)
        axes[row, 0].set_ylabel("best modeled TBT (ms, log scale) ↓")
    for ax in axes[-1, :]:
        ax.set_xlabel("total GPU budget G")
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("GPU–TBT Pareto curves at 128K context", fontsize=15)
    fig.text(
        0.5,
        0.035,
        "Every vertical slice compares a common GPU budget; open circles mark each method's minimum-G crossing of the 100 ms target.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.98))
    save(fig, outdir, "fig5_gpu_tbt_pareto_128k")


def plot_pruned_expert_percentage(index, outdir: Path) -> None:
    """Show the reduction from active-before-pruning to retained-after-pruning."""
    sessions = list(sweep.base.CONCURRENCIES)
    active_before = []
    retained = []
    for active_sessions in sessions:
        before_values = {
            round(point.mean_retained_experts, 6)
            for context in sweep.base.CONTEXTS
            for point in index[(context, active_sessions, 1.00, sweep.base.SYSTEM_DYNAMIC)]
            if point.microbatches == FIXED_MICROBATCHES
        }
        after_values = {
            round(point.mean_retained_experts, 6)
            for context in sweep.base.CONTEXTS
            for point in index[(context, active_sessions, 0.98, sweep.base.SYSTEM_DYNAMIC)]
            if point.microbatches == FIXED_MICROBATCHES
        }
        if len(before_values) != 1:
            raise AssertionError(
                f"unexpected pre-pruning active-expert values for S={active_sessions}: {before_values}"
            )
        if len(after_values) != 1:
            raise AssertionError(
                f"unexpected post-pruning retained-expert values for S={active_sessions}: {after_values}"
            )
        active_before.append(before_values.pop())
        retained.append(after_values.pop())
    active_before = np.asarray(active_before)
    retained = np.asarray(retained)
    pruned = 100 * (active_before - retained) / active_before
    x = np.arange(len(sessions))
    fig, ax_pruned = plt.subplots(figsize=(9.2, 5.6))
    bars = ax_pruned.bar(x, pruned, width=0.62, color="#4c78a8", alpha=0.9, label="Experts pruned")
    ax_pruned.set_xticks(x, sessions)
    ax_pruned.set_xlabel("concurrent decode sessions S")
    ax_pruned.set_ylabel("experts pruned (%)")
    ax_pruned.set_ylim(0, 32)
    ax_pruned.grid(axis="y", alpha=0.25)
    for bar, percentage, before, after in zip(bars, pruned, active_before, retained):
        ax_pruned.text(
            bar.get_x() + bar.get_width() / 2,
            percentage + 1.3,
            f"{percentage:.1f}%\n{before:.1f}→{after:.1f}",
            ha="center",
            fontsize=9,
        )
    ax_retained = ax_pruned.twinx()
    line = ax_retained.plot(x, retained, "o-", color="#d62728", linewidth=2.2, label="Experts retained")[0]
    ax_retained.set_ylabel("mean experts retained per microbatch")
    ax_retained.set_ylim(0, 256)
    ax_pruned.set_title("Synthetic expert-pruning sensitivity to decode workload\nreduction from active experts at c=1.00 to retained experts at c=.98")
    fig.legend([bars, line], ["Experts pruned", "Experts retained"], frameon=False, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=2)
    fig.text(0.5, 0.06, "Two microbatches; post-pruning set includes top-1 protection; synthetic router has no context-length dependence.", ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save(fig, outdir, "fig2_expert_pruning_percentage")




def write_readme(outdir: Path, budgets: dict[int, int]) -> None:
    lines = [
        "# Current corrected figures",
        "",
        "These figures use `results/corrected_microbatch_sweep.csv`.",
        "",
        "- Heterogeneous static and dynamic systems are fixed at two microbatches.",
        "- Colocated baselines select their best swept microbatch count for fairness.",
        "- HBM and HBF bandwidth are both 1024 GB/s.",
        "- Routing/pruning are synthetic and pipeline overlap is an analytical steady-state bound.",
        f"- Fixed budgets by workload: `{budgets}`.",
        "",
        "## Figure numbering",
        "",
        "1. Synthetic router-mass concentration.",
        "2. Synthetic reduction from active-before-pruning to retained-after-pruning experts versus workload.",
        "3. Microbatch-count sensitivity.",
        "4. Dynamic EP versus balanced static EP at matched c=1.00 and c=.98.",
        "5. Common-budget GPU–TBT Pareto curves at 128K context, with 100 ms crossings.",
        "6. End-to-end TBT advantage-regime zoom, without pruning and with c=.98 pruning.",
        "7. Minimum GPUs at 100 ms for S=32, 64, 128, and 256, without and with pruning.",
        "",
        "Unnumbered diagnostic: achieved TBT at method-specific minimum-GPU points; not a fixed-budget comparison.",
        "",
        "Both PNG and vector PDF versions are generated.",
        "",
    ]
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "README.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(SCRIPT_DIR.parent / "results" / "corrected_microbatch_sweep.csv"),
    )
    parser.add_argument(
        "--outdir",
        default=str(SCRIPT_DIR.parent / "figures" / "current"),
    )
    args = parser.parse_args()

    points = load_points(Path(args.csv))
    index = index_points(points)
    budgets = sweep.choose_budgets(sweep.group_points(points))
    outdir = Path(args.outdir)
    plot_regime_lines(index, budgets, outdir)
    plot_dynamic_gain_heatmap(index, budgets, outdir)
    plot_microbatch_sensitivity(index, budgets, outdir)
    plot_gpu_cost_at_slo(index, outdir)
    plot_tbt_at_slo_gpu_counts(index, outdir)
    plot_gpu_tbt_pareto(index, outdir)
    plot_pruned_expert_percentage(index, outdir)
    write_readme(outdir, budgets)
    print(f"wrote 7 core figures as PNG+PDF under {outdir}")


if __name__ == "__main__":
    main()
