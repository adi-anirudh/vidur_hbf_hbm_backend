#!/usr/bin/env python3
"""Space-efficient evaluation figures matching the established result_* style."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "plots" / "eval_revised_review"

# Existing result-plot palette and treatment.
BROWN = "#9e5e34"
ORANGE = "#e6913a"
TAN = "#eed6a0"
PURPLE = "#8279bd"
PURPLE_EDGE = "#3d3570"
INK = "#2b2b2b"
GRID = "#cfcfcf"
RED = "#d21f1f"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "hatch.linewidth": 0.45,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
})


def finish(ax, grid_axis: str = "y") -> None:
    ax.grid(
        True, axis=grid_axis, ls=(0, (4, 3)), lw=0.5,
        color=GRID, zorder=0,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel(ax, label: str) -> None:
    ax.text(
        -0.13, 1.02, label, transform=ax.transAxes,
        ha="left", va="bottom", fontsize=8.5, fontweight="bold",
    )


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_sparse_rows() -> list[dict]:
    return json.loads(
        (ROOT / "results" / "sparse_operating_point.json").read_text()
    )["rows"]


def sparse_axis(ax, rows: list[dict], *, show_legend: bool = True) -> None:
    x = np.arange(len(rows))
    labels = [
        f"{r['context_length'] // 1024}K"
        if r["context_length"] < 1_048_576
        else f"{r['context_length'] // 1_048_576}M"
        for r in rows
    ]
    width = 0.34
    token = [r["token_throughput_per_gpu"] for r in rows]
    page = [r["page_throughput_per_gpu"] for r in rows]
    ax.bar(
        x - width / 2, token, width, color=TAN, edgecolor=INK,
        linewidth=0.5, label="Exact token top-$k$", zorder=3,
    )
    ax.bar(
        x + width / 2, page, width, color=PURPLE, edgecolor=PURPLE_EDGE,
        linewidth=0.5, hatch="///", label="SPLASH (4-KB page top-$k$)",
        zorder=3,
    )
    for i, r in enumerate(rows):
        ax.text(
            i + width / 2, page[i] + max(page) * 0.025,
            f"{r['page_over_token_throughput']:.2f}$\\times$",
            ha="center", va="bottom", fontsize=6.2, color=PURPLE_EDGE,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Context length")
    ax.set_ylabel("Throughput (token/s/GPU)")
    ax.set_ylim(0, max(page) * 1.18)
    finish(ax, "y")
    if show_legend:
        ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1,
            frameon=False, handlelength=1.3, handletextpad=0.5,
        )


def sparse_operating_point() -> None:
    fig, ax = plt.subplots(figsize=(3.45, 1.85))
    sparse_axis(ax, load_sparse_rows())
    fig.subplots_adjust(left=0.17, right=0.99, top=0.76, bottom=0.25)
    save(fig, "sparse_operating_point_revised")


def granularity() -> None:
    data = json.loads(
        (ROOT / "results" / "granularity_comparison.json").read_text()
    )
    suites = data["suite_quality"]
    sparse = load_sparse_rows()
    fig, axes = plt.subplots(
        2, 1, figsize=(3.45, 3.15),
        gridspec_kw={"height_ratios": [0.9, 1.1]},
    )
    ax = axes[0]
    x = np.arange(len(suites))
    width = 0.34
    token_delta = np.array([
        r["token_mean_retention_percent"] - 100 for r in suites
    ])
    page_delta = np.array([
        r["page_mean_retention_percent"] - 100 for r in suites
    ])
    ax.bar(
        x - width / 2, token_delta, width, color=TAN, edgecolor=INK,
        linewidth=0.5, label="Exact token top-$k$", zorder=3,
    )
    page_bars = ax.bar(
        x + width / 2, page_delta, width, color=PURPLE,
        edgecolor=PURPLE_EDGE, linewidth=0.5, hatch="///",
        label="SPLASH (4-KB page top-$k$)", zorder=3,
    )
    del page_bars
    for xi, value in zip(x - width / 2, token_delta):
        ax.text(
            xi, value + (0.22 if value >= 0 else -0.22),
            f"{value:+.1f}", ha="center",
            va="bottom" if value >= 0 else "top", fontsize=5.5,
        )
    for xi, value in zip(x + width / 2, page_delta):
        ax.text(
            xi, value + (0.22 if value >= 0 else -0.22),
            f"{value:+.1f}", ha="center",
            va="bottom" if value >= 0 else "top", fontsize=5.5,
            color=PURPLE_EDGE,
        )
    ax.axhline(0, color="#666666", ls=(0, (3, 2)), lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([
        f"{r['suite']}\n({r['conditions']})" for r in suites
    ], fontsize=6.2)
    ax.set_ylabel("$\\Delta$ quality vs Full\n(percentage points)")
    ax.set_ylim(-4.0, 5.5)
    finish(ax, "y")
    panel(ax, "(a)")
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1,
        frameon=False, handlelength=1.2, handletextpad=0.45,
        borderaxespad=0,
    )

    sparse_axis(axes[1], sparse, show_legend=False)
    panel(axes[1], "(b)")
    fig.subplots_adjust(
        left=0.20, right=0.99, top=0.88, bottom=0.11, hspace=0.52,
    )
    save(fig, "granularity_comparison_revised")


def attention_mechanism() -> None:
    data = json.loads(
        (ROOT / "results" / "attention_mechanism_breakdown.json").read_text()
    )["rows"]
    rows = [r for r in data if int(r["context_length"]) == 1_048_576]
    order = [
        "Dense HBF", "Token-granular", "Page + full-K score",
        "Global page top-k", "SPLASH",
    ]
    by_name = {r["system"]: r for r in rows}
    totals = np.array([by_name[s]["attention_critical_ms"] for s in order])
    savings = totals[:-1] - totals[1:]
    x = np.arange(len(order))
    xlabels = [
        "H3\n(dense)",
        "Token\n top-$k$",
        "Page\n top-$k$",
        "Centroid\n top-$k$",
        "SPLASH\n(quota)",
    ]

    fig, axes = plt.subplots(
        2, 1, figsize=(3.45, 3.35),
        gridspec_kw={"height_ratios": [1.18, 1.0]},
    )

    # Causal waterfall: each floating bar is the latency eliminated by the
    # mechanism introduced at that stage; points show the resulting total.
    ax = axes[0]
    ax.bar(
        0, totals[0], 0.58, color=BROWN, edgecolor=INK,
        linewidth=0.5, zorder=3,
    )
    step_colors = [ORANGE, TAN, "#b9adcf", PURPLE]
    step_hatches = [None, None, None, "///"]
    for i in range(1, len(order)):
        bar = ax.bar(
            i, savings[i - 1], 0.58, bottom=totals[i],
            color=step_colors[i - 1], edgecolor=(
                PURPLE_EDGE if i == len(order) - 1 else INK
            ),
            linewidth=0.5, hatch=step_hatches[i - 1], zorder=3,
        )
        del bar
        ax.text(
            i, totals[i - 1] + 3.0, f"$-$ {savings[i - 1]:.1f}",
            ha="center", va="bottom", fontsize=6.0,
        )
    ax.plot(
        x, totals, color=INK, marker="o", ms=3.0, lw=0.75,
        drawstyle="steps-post", zorder=4,
    )
    for i, value in enumerate(totals):
        ax.text(
            i, value + 3.1 if i == 0 else value - 3.2,
            f"{value:.1f}", ha="center",
            va="bottom" if i == 0 else "top",
            fontsize=6.2, fontweight="bold",
            bbox=None if i == 0 else {
                "facecolor": "white", "edgecolor": "none",
                "alpha": 0.75, "pad": 0.2,
            },
        )
    hot_floor = by_name["SPLASH"]["hot_hbm_ms"]
    ax.axhline(hot_floor, color="#777777", ls=(0, (3, 2)), lw=0.7)
    ax.text(
        3.95, hot_floor + 2.0, f"hot-HBM floor: {hot_floor:.1f} ms",
        fontsize=5.8, color="#555555", ha="right",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=6.2)
    ax.set_ylabel("Attention path\n(ms/token)")
    ax.set_ylim(0, 154)
    finish(ax, "y")
    panel(ax, "(a)")
    ax.text(
        0.99, 0.97, "1M context, batch 64, TP=8",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=5.8, style="italic", color="#555555",
    )

    # Physical explanation for the same five stages. Stacked bars are HBF
    # internal traffic; diamonds show what crosses HBF->GPU; annotations expose
    # the critical busiest-plane multiplier.
    ax = axes[1]
    score_scan = np.array([0.0, 50.0, 50.0, 3.125, 3.125])
    selected_read = np.array([100.0, 17.8175, 5.0, 10.0, 10.0])
    link = np.array([100.0, 17.8, 5.0, 10.0, 10.0])
    max_plane = np.array([1.0, 1.341, 1.0, 2.175, 1.0])
    ax.bar(
        x, score_scan, 0.60, color=ORANGE, edgecolor=INK,
        linewidth=0.45, label="Score scan", zorder=3,
    )
    bars = ax.bar(
        x, selected_read, 0.60, bottom=score_scan, color=TAN,
        edgecolor=INK, linewidth=0.45, label="Selected-data read", zorder=3,
    )
    bars[-1].set_edgecolor(PURPLE_EDGE)
    bars[-1].set_hatch("///")
    ax.scatter(
        x, link, marker="D", s=18, color=PURPLE_EDGE,
        edgecolor="white", linewidth=0.35, label="HBF$\\rightarrow$GPU",
        zorder=5,
    )
    internal = score_scan + selected_read
    for i, (value, plane_mult) in enumerate(zip(internal, max_plane)):
        ax.text(
            i, value + 4.0, f"{value:.1f}%", ha="center",
            va="bottom", fontsize=5.8,
        )
        color = PURPLE_EDGE if plane_mult > 2 else "#555555"
        ax.text(
            i, 112, f"plane {plane_mult:.2f}$\\times$", ha="center",
            va="bottom", fontsize=5.4, color=color,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=6.2)
    ax.set_ylabel("Traffic (% of\ncold K+V)")
    ax.set_ylim(0, 126)
    finish(ax, "y")
    panel(ax, "(b)")
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3,
        frameon=False, fontsize=5.8, handlelength=1.0,
        handletextpad=0.3, columnspacing=0.65,
    )

    fig.subplots_adjust(left=0.19, right=0.99, top=0.97, bottom=0.10, hspace=0.48)
    save(fig, "attention_mechanism_revised")


def ablation() -> None:
    data = json.loads(
        (ROOT / "results" / "ablation_matrix_summary.json").read_text()
    )
    systems = next(
        c["systems"] for c in data["contexts"]
        if int(c["context_length"]) == 1_048_576
    )
    order = [
        "Dense", "TokenGranular", "FullPageScore",
        "PlaneImbalance", "SPLASH",
    ]
    labels = [
        "H3", "Token\ntop-$k$", "Page top-$k$\n(full-K)",
        "Centroid top-$k$\n(global)", "SPLASH\n(plane quota)",
    ]
    colors = [BROWN, ORANGE, TAN, "#b9adcf", PURPLE]
    vals = [systems[s]["normalized_to_dense"] for s in order]
    fig, ax = plt.subplots(figsize=(3.45, 1.72))
    x = np.arange(len(order))
    bars = ax.bar(
        x, vals, color=colors, edgecolor=INK, linewidth=0.5, zorder=3,
    )
    bars[-1].set_hatch("///")
    bars[-1].set_edgecolor(PURPLE_EDGE)
    for i, value in enumerate(vals):
        ax.text(
            i, value + 0.10, f"{value:.2f}$\\times$",
            ha="center", va="bottom", fontsize=6.2,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("Throughput / H3")
    ax.set_ylim(0, max(vals) * 1.20)
    finish(ax, "y")
    fig.subplots_adjust(left=0.17, right=0.99, top=0.96, bottom=0.26)
    save(fig, "ablation_revised")


def plane_tail() -> None:
    raw = json.loads(
        Path("/home/adityaan/splash_accuracy_tests/planeload_1024.json").read_text()
    )
    stripe_load = np.asarray(raw["stripe"]["load"], dtype=float)
    splash_load = np.asarray(raw["splash"]["load"], dtype=float)
    stripe_rounds = np.asarray(raw["stripe"]["rounds"], dtype=float)
    splash_rounds = np.asarray(raw["splash"]["rounds"], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.45))

    rank = 100 * (np.arange(1024) + 1) / 1024
    axes[0].plot(
        rank, np.sort(stripe_load / stripe_load.mean()),
        color=ORANGE, label="Global page top-$k$",
    )
    axes[0].plot(
        rank, np.sort(splash_load / splash_load.mean()),
        color=PURPLE_EDGE, label="SPLASH plane quota",
    )
    axes[0].axhspan(0.75, 1.25, color="#777777", alpha=0.08)
    axes[0].axhline(1, color="#666666", ls=(0, (3, 2)), lw=0.8)
    axes[0].set_ylabel("Plane load / mean")
    axes[0].set_xlabel("Plane-load percentile (%)")
    axes[0].set_ylim(0.2, 2.45)
    finish(axes[0], "y")
    panel(axes[0], "(a)")
    axes[0].legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
        frameon=False, fontsize=6.2, handlelength=1.25,
        columnspacing=0.9,
    )
    axes[0].text(
        0.98, 0.10, "96.29% within $\\pm$25%\nsole $>1.5\\times$: sink plane",
        transform=axes[0].transAxes, ha="right", va="bottom", fontsize=6.1,
    )

    for values, color, label in (
        (stripe_rounds, ORANGE, "Global page top-$k$"),
        (splash_rounds, PURPLE_EDGE, "SPLASH plane quota"),
    ):
        sorted_values = np.sort(values)
        cdf = 100 * (np.arange(len(values)) + 1) / len(values)
        axes[1].plot(sorted_values, cdf, color=color, label=label)
    axes[1].axvline(98, color=ORANGE, ls=(0, (3, 2)), lw=0.8)
    axes[1].axvline(64, color=PURPLE_EDGE, ls=(0, (3, 2)), lw=0.8)
    axes[1].set_xlabel("Serialized HBF read rounds")
    axes[1].set_ylabel("Events completed (%)")
    finish(axes[1], "both")
    panel(axes[1], "(b)")
    axes[1].text(
        0.04, 0.70, "p99: 98 $\\rightarrow$ 64\n99.93% improve",
        transform=axes[1].transAxes, fontsize=6.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )
    fig.subplots_adjust(left=0.18, right=0.98, top=0.90, bottom=0.12, hspace=0.62)
    save(fig, "plane_tail_revised")


def sensitivity() -> None:
    data = json.loads(
        (ROOT / "results" / "design_sensitivities_summary.json").read_text()
    )
    bp = data["budget_performance"]
    bq = data["budget_quality"]
    pq = data["page_quality"]
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.45))

    xb = np.array([r["selection_fraction"] * 100 for r in bp])
    throughput = np.array([r["throughput_per_gpu"] for r in bp])
    ppl = np.array([r["ppl_overhead_percent"] for r in bq])
    axes[0].plot(
        xb, throughput, "-o", color=PURPLE_EDGE, ms=3.6,
        label="Throughput",
    )
    axes[0].set_ylabel("token/s/GPU", color=PURPLE_EDGE)
    axes[0].tick_params(axis="y", labelcolor=PURPLE_EDGE)
    axes[0].set_xticks(xb)
    axes[0].set_xlabel("Selected historical KV (%)")
    axes[0].axvline(10, color="#666666", ls=(0, (3, 2)), lw=0.8)
    twin = axes[0].twinx()
    twin.plot(
        xb, ppl, "--s", color=ORANGE, ms=3.4,
        label="PG-19 PPL overhead",
    )
    twin.set_ylabel("PPL overhead (%)", color=ORANGE)
    twin.tick_params(axis="y", labelcolor=ORANGE)
    twin.spines["top"].set_visible(False)
    finish(axes[0], "x")
    panel(axes[0], "(a)")
    axes[0].text(
        10, np.interp(10, xb, throughput), "  chosen",
        fontsize=6.1, va="bottom", color="#555555",
    )

    xp = np.array([r["page_kb"] for r in pq])
    coverage = np.array([100 * r["top_budget_token_recall"] for r in pq])
    metadata = np.array([100 / (2 * r["page_tokens"]) for r in pq])
    axes[1].plot(
        xp, coverage, "-o", color=PURPLE_EDGE, ms=3.6,
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(xp)
    axes[1].set_xticklabels([str(v) for v in xp])
    axes[1].set_xlabel("Physical HBF page size (KB)")
    axes[1].set_ylabel("Useful-token coverage (%)", color=PURPLE_EDGE)
    axes[1].tick_params(axis="y", labelcolor=PURPLE_EDGE)
    twin2 = axes[1].twinx()
    twin2.plot(xp, metadata, "--s", color=ORANGE, ms=3.4)
    twin2.set_ylabel("Centroids (% K+V)", color=ORANGE)
    twin2.tick_params(axis="y", labelcolor=ORANGE)
    twin2.spines["top"].set_visible(False)
    finish(axes[1], "x")
    panel(axes[1], "(b)")
    fig.subplots_adjust(left=0.18, right=0.82, top=0.97, bottom=0.13, hspace=0.68)
    save(fig, "sensitivity_revised")


def energy() -> None:
    summary = json.loads(
        (ROOT / "results" / "energy_b200" /
         "provisional_hybrid_summary.json").read_text()
    )
    cases = summary["sensitivity_cases"]
    central = next(
        c for c in cases
        if c["active_power_w_per_gpu"] == 700.0
        and abs(c["hbm_pj_per_byte"] - 11.926605504587155) < 1e-9
        and c["hbf_array_pj_per_byte"] == 64.0
    )
    dense = [c["dense_over_splash_energy_geomean"] for c in cases]
    naive = [c["naive_over_splash_energy_geomean"] for c in cases]
    vals = [
        central["dense_over_splash_energy_geomean"],
        central["naive_over_splash_energy_geomean"],
        1.0,
    ]
    lower = [vals[0] - min(dense), vals[1] - min(naive), 0]
    upper = [max(dense) - vals[0], max(naive) - vals[1], 0]
    fig, ax = plt.subplots(figsize=(3.45, 1.62))
    x = np.arange(3)
    bars = ax.bar(
        x, vals, color=[ORANGE, TAN, PURPLE], edgecolor=INK,
        linewidth=0.5, yerr=np.asarray([lower, upper]),
        error_kw={"elinewidth": 0.8, "capsize": 2.5, "capthick": 0.8},
        zorder=3,
    )
    bars[-1].set_hatch("///")
    bars[-1].set_edgecolor(PURPLE_EDGE)
    ax.set_xticks(x)
    ax.set_xticklabels(["H3", "Naive Sparse", "SPLASH (Ours)"])
    ax.set_ylabel("Energy / SPLASH")
    ax.set_ylim(0, max(vals) * 1.18)
    for i, value in enumerate(vals):
        ax.text(
            i, value + 0.13, f"{value:.2f}$\\times$",
            ha="center", va="bottom", fontsize=6.5,
        )
    finish(ax, "y")
    ax.text(
        0.99, 0.95, "projection; error bars span all coefficients",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=6.1, style="italic", color="#555555",
    )
    fig.subplots_adjust(left=0.17, right=0.99, top=0.96, bottom=0.24)
    save(fig, "energy_revised")


def main() -> None:
    sparse_operating_point()
    granularity()
    attention_mechanism()
    ablation()
    plane_tail()
    sensitivity()
    energy()
    print(OUT)


if __name__ == "__main__":
    main()
