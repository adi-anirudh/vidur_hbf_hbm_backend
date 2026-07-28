#!/usr/bin/env python3
"""Single-column publication figures for the condensed evaluation section."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import publication_style as ps
from plot_main_evaluation import CONTEXT_LABEL, select, short


ROOT = Path(__file__).resolve().parent
PAPER_IMAGES = Path("/home/adityaan/splash_paper/Images")
OUT = ROOT / "results" / "plots"


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER_IMAGES.mkdir(parents=True, exist_ok=True)
    for base in (OUT / name, PAPER_IMAGES / name):
        fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def compact_style() -> None:
    ps.apply()
    plt.rcParams.update({
        "font.size": 7.0,
        "axes.labelsize": 7.1,
        "axes.titlesize": 7.5,
        "legend.fontsize": 6.4,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
    })


def performance() -> None:
    rows = [
        r for r in csv.DictReader(
            (ROOT / "results" / "sweep_b200_weight_valid.csv").open()
        )
        if r["status"] == "OK"
    ]
    models = sorted({r["model"] for r in rows}, key=short)
    contexts = sorted({int(r["context_length"]) for r in rows})
    chosen = {slo: select(rows, slo) for slo in (50.0, 100.0)}
    summary = json.loads(
        (ROOT / "results" / "main_evaluation_summary.json").read_text()
    )

    fig, axes = plt.subplots(
        2, 1, figsize=(3.45, 4.05),
        gridspec_kw={"height_ratios": [2.25, 1.0]},
    )
    matrix = np.full((len(models), len(contexts)), np.nan)
    for i, model in enumerate(models):
        for j, context in enumerate(contexts):
            dense = chosen[100.0].get((model, context, "Dense"))
            splash = chosen[100.0].get((model, context, "Sparse"))
            if dense and splash:
                matrix[i, j] = (
                    splash["throughput_per_gpu"]
                    / dense["throughput_per_gpu"]
                )
    im = axes[0].imshow(
        matrix, aspect="auto", cmap="YlGnBu", vmin=3.0, vmax=8.0,
    )
    axes[0].set_xticks(range(len(contexts)))
    axes[0].set_xticklabels([CONTEXT_LABEL[c] for c in contexts])
    axes[0].set_yticks(range(len(models)))
    labels = [
        short(m)
        .replace("DeepSeek-", "DS-")
        .replace("Mixtral-", "Mix-")
        .replace("Mistral-", "Mis-")
        for m in models
    ]
    axes[0].set_yticklabels(labels, fontsize=5.7)
    for i in range(len(models)):
        for j in range(len(contexts)):
            value = matrix[i, j]
            if np.isfinite(value):
                axes[0].text(
                    j, i, f"{value:.1f}", ha="center", va="center",
                    fontsize=4.8, color="white" if value >= 5.7 else "black",
                )
    cb = fig.colorbar(im, ax=axes[0], fraction=0.04, pad=0.025)
    cb.set_label(r"SPLASH / H3", fontsize=6.5)
    cb.ax.tick_params(labelsize=5.8)
    axes[0].set_title("(a) Throughput/GPU speedup, 100-ms SLO")

    x = np.arange(len(contexts))
    for slo, marker, color in (
        (50, "s", ps.COLORS["token"]),
        (100, "o", ps.COLORS["splash"]),
    ):
        by_context = summary["slo"][str(slo)]["splash_vs_dense"]["by_context"]
        y = [by_context[str(c)]["geomean_speedup"] for c in contexts]
        axes[1].plot(
            x, y, marker=marker, color=color, label=f"{slo} ms",
        )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([CONTEXT_LABEL[c] for c in contexts])
    axes[1].set_ylabel("Geomean speedup")
    axes[1].set_title("(b) Robust across service targets")
    axes[1].legend(ncol=2, loc="upper left")
    ps.finish_axis(axes[1], grid_axis="y")
    fig.tight_layout(h_pad=0.65)
    save(fig, "compact_performance")


def accuracy() -> None:
    data = json.loads(
        (ROOT / "results" / "granularity_comparison.json").read_text()
    )
    suites = data["suite_quality"]
    perf = data["performance"]
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.25))

    x = np.arange(len(suites))
    width = 0.34
    axes[0].bar(
        x - width / 2,
        [r["token_mean_retention_percent"] for r in suites],
        width, label="Exact token", color=ps.COLORS["token"],
        edgecolor="black", linewidth=0.35,
    )
    axes[0].bar(
        x + width / 2,
        [r["page_mean_retention_percent"] for r in suites],
        width, label="SPLASH page", color=ps.COLORS["splash"],
        edgecolor="black", linewidth=0.35,
    )
    axes[0].axhline(100, color="#777", lw=0.8, ls=":")
    axes[0].set_ylim(90, 107)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([r["suite"] for r in suites])
    axes[0].set_ylabel("Quality retained (%)")
    axes[0].set_title("(a) Same 10% logical-token budget")
    axes[0].legend(ncol=2, loc="lower left")
    ps.finish_axis(axes[0], grid_axis="y")

    contexts = [r["context_length"] for r in perf]
    ratio = [r["page_over_token"] for r in perf]
    axes[1].plot(
        np.arange(len(perf)), ratio, "-o", color=ps.COLORS["splash"],
    )
    axes[1].axhline(1, color="#777", lw=0.8, ls=":")
    axes[1].set_xticks(np.arange(len(perf)))
    axes[1].set_xticklabels([
        f"{c // 1024}K" if c < 1048576 else f"{c // 1048576}M"
        for c in contexts
    ])
    axes[1].set_ylabel("Page / token throughput")
    axes[1].set_title("(b) Physical gain at matched accuracy budget")
    for i, value in enumerate(ratio):
        axes[1].annotate(
            f"{value:.2f}×", (i, value), xytext=(0, 4),
            textcoords="offset points", ha="center", fontsize=5.8,
        )
    axes[1].set_ylim(0.8, max(ratio) * 1.22)
    ps.finish_axis(axes[1], grid_axis="y")
    fig.tight_layout(h_pad=0.75)
    save(fig, "compact_accuracy")


def sensitivity() -> None:
    data = json.loads(
        (ROOT / "results" / "design_sensitivities_summary.json").read_text()
    )
    bp = data["budget_performance"]
    bq = data["budget_quality"]
    pq = data["page_quality"]
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.35))

    xb = np.array([r["selection_fraction"] * 100 for r in bp])
    throughput = [r["throughput_per_gpu"] for r in bp]
    ppl = [r["ppl_overhead_percent"] for r in bq]
    axes[0].plot(
        xb, throughput, "-o", color=ps.COLORS["splash"],
        label="Throughput",
    )
    axes[0].set_ylabel("token/s/GPU", color=ps.COLORS["splash"])
    axes[0].tick_params(axis="y", labelcolor=ps.COLORS["splash"])
    twin = axes[0].twinx()
    twin.plot(
        xb, ppl, "--s", color=ps.COLORS["dense"],
        label="PPL overhead",
    )
    twin.set_ylabel("PPL overhead (%)", color=ps.COLORS["dense"])
    twin.tick_params(axis="y", labelcolor=ps.COLORS["dense"])
    axes[0].axvline(10, color="#777", lw=0.8, ls=":")
    axes[0].set_xticks(xb)
    axes[0].set_xlabel("Selected historical KV (%)")
    axes[0].set_title("(a) Selection budget")
    ps.finish_axis(axes[0], grid_axis="x")
    ps.finish_twin_axis(twin)

    xp = np.array([r["page_kb"] for r in pq])
    coverage = [100 * r["top_budget_token_recall"] for r in pq]
    metadata = [100 / (2 * r["page_tokens"]) for r in pq]
    axes[1].plot(
        xp, coverage, "-o", color=ps.COLORS["splash"],
    )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(xp)
    axes[1].set_xticklabels([str(v) for v in xp])
    axes[1].set_xlabel("Physical page size (KB)")
    axes[1].set_ylabel("Useful-token coverage (%)", color=ps.COLORS["splash"])
    axes[1].tick_params(axis="y", labelcolor=ps.COLORS["splash"])
    twin2 = axes[1].twinx()
    twin2.plot(
        xp, metadata, "--s", color=ps.COLORS["token"],
    )
    twin2.set_ylabel("Centroids (% K+V)", color=ps.COLORS["token"])
    twin2.tick_params(axis="y", labelcolor=ps.COLORS["token"])
    axes[1].set_title("(b) Page-size precision/metadata trade-off")
    ps.finish_axis(axes[1], grid_axis="x")
    ps.finish_twin_axis(twin2)
    fig.tight_layout(h_pad=0.65)
    save(fig, "compact_sensitivity")


def ablation() -> None:
    abl = json.loads(
        (ROOT / "results" / "ablation_matrix_summary.json").read_text()
    )
    attn = json.loads(
        (ROOT / "results" / "attention_mechanism_breakdown.json").read_text()
    )["rows"]
    plane = json.loads(
        (ROOT / "results" / "plane_tail_statistics.json").read_text()
    )
    target = next(
        c for c in abl["contexts"] if c["context_length"] == 1048576
    )["systems"]
    order = [
        "Dense", "TokenGranular", "FullPageScore",
        "PlaneImbalance", "SPLASH",
    ]
    labels = ["H3", "Token", "Page", "Centroid", "SPLASH"]
    colors = [
        ps.COLORS["dense"], ps.COLORS["token"], ps.COLORS["full_score"],
        ps.COLORS["global"], ps.COLORS["splash"],
    ]
    fig, axes = plt.subplots(
        3, 1, figsize=(3.45, 4.45),
        gridspec_kw={"height_ratios": [1.0, 1.45, 0.72]},
    )

    vals = [target[s]["normalized_to_dense"] for s in order]
    axes[0].bar(
        np.arange(len(order)), vals, color=colors,
        edgecolor="black", linewidth=0.35,
    )
    axes[0].set_xticks(np.arange(len(order)))
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Throughput / H3")
    axes[0].set_title("(a) End-to-end factor isolation, 1M")
    for i, value in enumerate(vals):
        axes[0].text(
            i, value + 0.12, f"{value:.2f}×",
            ha="center", fontsize=5.7,
        )
    axes[0].set_ylim(0, max(vals) * 1.2)
    ps.finish_axis(axes[0], grid_axis="y")

    rows = [
        r for r in attn
        if r["context_length"] == 1048576
    ]
    sys_order = [
        "Dense HBF", "Token-granular", "Page + full-K score",
        "Global page top-k", "SPLASH",
    ]
    row_by_sys = {r["system"]: r for r in rows}
    y = np.arange(len(sys_order))
    hot = [row_by_sys[s]["hot_hbm_ms"] for s in sys_order]
    score = [row_by_sys[s]["exposed_score_ms"] for s in sys_order]
    selected = [row_by_sys[s]["selected_hbf_ms"] for s in sys_order]
    axes[1].barh(y, hot, color=ps.COLORS["hbm"], label="Hot HBM")
    axes[1].barh(
        y, score, left=hot, color=ps.COLORS["full_score"],
        label="Exposed score",
    )
    axes[1].barh(
        y, selected, left=np.array(hot) + np.array(score),
        color=ps.COLORS["splash"], label="HBF reads",
    )
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(
        ["H3", "Token", "Page", "Centroid", "SPLASH"],
    )
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Attention critical path (ms/token)")
    axes[1].set_title("(b) Low-level latency decomposition, 1M")
    axes[1].legend(ncol=3, loc="lower right")
    ps.finish_axis(axes[1], grid_axis="x")

    rounds = [
        plane["global_striped"]["serial_rounds"]["p99"],
        plane["splash"]["serial_rounds"]["p99"],
    ]
    axes[2].barh(
        [0, 1], rounds,
        color=[ps.COLORS["global"], ps.COLORS["splash"]],
        edgecolor="black", linewidth=0.35,
    )
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["Global top-$k$", "Plane quota"])
    axes[2].invert_yaxis()
    axes[2].set_xlabel("p99 serialized read rounds")
    axes[2].set_title("(c) Measured plane tail, 4,096 events")
    for i, value in enumerate(rounds):
        axes[2].text(value + 1, i, f"{value:.0f}", va="center", fontsize=5.8)
    axes[2].set_xlim(0, max(rounds) * 1.18)
    ps.finish_axis(axes[2], grid_axis="x")

    fig.tight_layout(h_pad=0.7)
    save(fig, "compact_ablation")


def main() -> None:
    compact_style()
    performance()
    accuracy()
    sensitivity()
    ablation()


if __name__ == "__main__":
    main()
