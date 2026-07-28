#!/usr/bin/env python3
"""Standalone evaluation plots in the exact established result_* visual style."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "plots" / "eval_result_style_review"

HBM = "#9e5e34"
H3 = "#e6913a"
NAIVE = "#eed6a0"
SPLASH = "#8279bd"
SPLASH_EDGE = "#3d3570"
EDGE = "#2b2b2b"
GRID = "#cfcfcf"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "legend.fontsize": 6.5,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 0,
    "ytick.major.size": 2.5,
    "hatch.linewidth": 0.45,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
})


def clean(ax, grid_axis: str = "y") -> None:
    ax.grid(
        True, axis=grid_axis, ls=(0, (4, 3)), lw=0.5,
        color=GRID, zorder=0,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def granularity_ablation() -> None:
    """Link end-task quality and serving gain at one matched sparse point."""
    quality_rows = json.loads(
        (ROOT / "results" / "granularity_comparison.json").read_text()
    )["quality"]
    perf_rows = json.loads(
        (ROOT / "results" / "sparse_operating_point.json").read_text()
    )["rows"]

    # A direct page-minus-token comparison retains every quality condition
    # without spending fourteen rows on paired bars.
    page_minus_token = np.array([
        r["page_minus_token_retention_pp"] for r in quality_rows
    ])
    task_abbr = [
        "NQA", "SQA", "MH", "SUM", "RET", "CNT", "CODE",
        "MV", "MQ", "VT", "AGG", "32K", "128K", "PK",
    ]
    suite_groups = [
        ("LongBench", 0, 6),
        ("RULER", 7, 10),
        ("PG-19", 11, 12),
        ("Passkey", 13, 13),
    ]

    fig, (axq, axp) = plt.subplots(
        1, 2, figsize=(3.45, 1.28),
        gridspec_kw={"width_ratios": [1.55, 1.0]},
    )
    fig.subplots_adjust(
        left=0.12, right=0.995, top=0.82, bottom=0.23, wspace=0.39,
    )

    # (a) Quality parity at the matched 10% logical-token budget.
    xq = np.arange(len(quality_rows))
    axq.vlines(
        xq, np.minimum(0, page_minus_token), np.maximum(0, page_minus_token),
        color="#9b96b9", lw=0.65, zorder=2,
    )
    axq.scatter(
        xq, page_minus_token, s=11, marker="o", color=SPLASH,
        edgecolor=SPLASH_EDGE, linewidth=0.45, zorder=3,
    )
    axq.axhline(0, color="#555555", lw=0.7, zorder=1)
    for _, start, end in suite_groups[:-1]:
        axq.axvline(end + 0.5, color="#c8c8c8", lw=0.45, zorder=0)
    for name, start, end in suite_groups:
        axq.text(
            (start + end) / 2, 1.035, name,
            transform=axq.get_xaxis_transform(), ha="center", va="bottom",
            fontsize=4.8,
        )
    axq.set_xlim(-0.6, len(quality_rows) - 0.4)
    axq.set_ylim(-9, 21.5)
    axq.set_yticks([-5, 0, 10, 20])
    axq.set_ylabel("Page $-$ token\nquality (pp)", fontsize=6.2, labelpad=1)
    axq.set_xticks(xq)
    axq.set_xticklabels(task_abbr, rotation=90, fontsize=4.45)
    axq.tick_params(axis="x", pad=1)
    axq.text(
        0.0, 1.19, "(a) Quality parity: all 14 conditions",
        transform=axq.transAxes, ha="left", va="bottom", fontsize=5.7,
    )
    clean(axq)

    # (b) SLO-constrained serving gain at the identical sparse point.
    xp = np.arange(len(perf_rows))
    gain = np.array([
        r["page_over_token_throughput"] for r in perf_rows
    ])
    axp.plot(
        xp, gain, color=SPLASH_EDGE, lw=1.05, marker="o", ms=3.3,
        markerfacecolor=SPLASH, markeredgecolor=SPLASH_EDGE, zorder=3,
    )
    for i, value in enumerate(gain):
        axp.text(
            i, value + 0.12, f"{value:.1f}$\\times$",
            ha="center", va="bottom", fontsize=4.65, color=SPLASH_EDGE,
        )
    context_labels = [
        f"{r['context_length'] // 1024}K"
        if r["context_length"] < 1_048_576
        else f"{r['context_length'] // 1_048_576}M"
        for r in perf_rows
    ]
    axp.axhline(1, color="#555555", lw=0.7, ls=(0, (3, 2)), zorder=1)
    axp.set_xlim(-0.25, len(perf_rows) - 0.75)
    axp.set_ylim(0.82, 3.25)
    axp.set_yticks([1, 2, 3])
    axp.set_ylabel("Throughput\nPage / token", fontsize=6.2, labelpad=1)
    axp.set_xticks(xp)
    axp.set_xticklabels(context_labels, fontsize=5.1)
    axp.text(
        0.0, 1.19, "(b) SLO-constrained serving",
        transform=axp.transAxes, ha="left", va="bottom", fontsize=5.7,
    )
    clean(axp)

    save(fig, "granularity_quality_performance")


def attention() -> None:
    rows = json.loads(
        (ROOT / "results" / "attention_mechanism_breakdown.json").read_text()
    )["rows"]
    order = [
        "Dense HBF", "Token-granular", "Page + full-K score",
        "Global page top-k", "SPLASH",
    ]
    by_name = {
        r["system"]: r for r in rows
        if int(r["context_length"]) == 1_048_576
    }
    hot = np.array([by_name[s]["hot_hbm_ms"] for s in order])
    score = np.array([by_name[s]["exposed_score_ms"] for s in order])
    selected = np.array([by_name[s]["selected_hbf_ms"] for s in order])
    tail = np.array([by_name[s]["plane_tail_ms"] for s in order])
    y = np.arange(5)[::-1]
    fig, ax = plt.subplots(figsize=(3.45, 1.10))
    ax.barh(
        y, hot, 0.58, color=HBM, edgecolor=EDGE, linewidth=0.45,
        label="HBM-resident KV", zorder=3,
    )
    ax.barh(
        y, score, 0.58, left=hot, color=H3, edgecolor=EDGE,
        linewidth=0.45, label="Exposed selection scan", zorder=3,
    )
    ax.barh(
        y, selected, 0.58, left=hot + score, color=NAIVE,
        edgecolor=EDGE, linewidth=0.45, label="Returned HBF data", zorder=3,
    )
    ax.barh(
        y, tail, 0.58, left=hot + score + selected, color=SPLASH,
        edgecolor=SPLASH_EDGE, linewidth=0.45, hatch="///",
        label="Busiest-plane tail", zorder=3,
    )
    total = hot + score + selected + tail
    for i, value in enumerate(total):
        ax.text(value + 2.2, y[i], f"{value:.1f}", va="center", fontsize=5.7)
    ax.set_xlim(0, 151)
    ax.set_xticks([])
    ax.set_yticks(y)
    ax.set_yticklabels([
        "Dense HBF",
        "Token top-$k$",
        "Page + full-K score",
        "Global centroid top-$k$",
        "SPLASH",
    ], fontsize=5.7)
    clean(ax, "x")
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    ax.legend(
        loc="center right", bbox_to_anchor=(0.995, 0.40), ncol=1,
        frameon=False, fontsize=5.4, handlelength=1.15,
        handleheight=0.85, handletextpad=0.35, labelspacing=0.28,
        borderaxespad=0,
    )
    fig.subplots_adjust(left=0.31, right=0.99, top=0.98, bottom=0.04)
    save(fig, "attention_mechanism")


def ablation() -> None:
    data = json.loads(
        (ROOT / "results" / "ablation_matrix_summary.json").read_text()
    )["contexts"]
    x = np.arange(len(data))
    labels = [
        f"{c['context_length'] // 1024}K"
        if c["context_length"] < 1_048_576
        else f"{c['context_length'] // 1_048_576}M"
        for c in data
    ]
    systems = [
        ("TokenGranular", "w/o page-native access", HBM, None),
        ("FullPageScore", "w/o centroid scoring", H3, None),
        ("PlaneImbalance", "w/o plane quotas", NAIVE, None),
        ("SPLASH", "Full SPLASH", SPLASH, "///"),
    ]
    width = 0.19
    fig, ax = plt.subplots(figsize=(3.45, 1.55))
    for i, (key, label, color, hatch) in enumerate(systems):
        vals = np.array([
            c["systems"][key]["throughput_per_gpu"]
            / c["systems"]["SPLASH"]["throughput_per_gpu"]
            for c in data
        ])
        ax.bar(
            x + (i - 1.5) * width, vals, width, color=color,
            edgecolor=SPLASH_EDGE if key == "SPLASH" else EDGE,
            linewidth=0.45, hatch=hatch, label=label, zorder=3,
        )
    ax.set_ylim(0, 1.14)
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_ylabel("Throughput /\nFull SPLASH")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Context length")
    clean(ax)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
        frameon=False, fontsize=5.8, handlelength=1.15,
        handletextpad=0.35, columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.18, right=0.99, top=0.72, bottom=0.24)
    save(fig, "ablation_leave_one_out")


def validated_system_ablation() -> None:
    """System ablation from the exact validated ten-model headline sweep."""
    source = ROOT / "results" / "sweep_b200_weight_valid.csv"
    rows = [
        r for r in csv.DictReader(source.open())
        if r["status"] == "OK" and int(r["context_length"]) == 131072
    ]
    models = sorted({r["model"] for r in rows})
    systems = ("Dense", "Naive", "Sparse")
    values = {}
    for slo in (50.0, 100.0):
        best = {}
        for r in rows:
            if r["baseline"] not in systems:
                continue
            tpot = float(r["tpot_p50_ms"])
            if tpot > slo:
                continue
            throughput = (
                int(r["batch"]) * 1000.0 / (int(r["tp"]) * tpot)
            )
            key = (r["model"], r["baseline"])
            if key not in best or throughput > best[key]:
                best[key] = throughput
        ratios = {system: [] for system in systems}
        for model in models:
            dense = best.get((model, "Dense"))
            if not dense:
                continue
            for system in systems:
                point = best.get((model, system))
                if point:
                    ratios[system].append(point / dense)
        values[slo] = [
            math.exp(sum(math.log(v) for v in ratios[s]) / len(ratios[s]))
            for s in systems
        ]

    x = np.arange(3)
    width = 0.34
    fig, ax = plt.subplots(figsize=(3.45, 1.42))
    colors = [HBM, NAIVE, SPLASH]
    for offset, slo, alpha, hatch in (
        (-width / 2, 50.0, 0.68, None),
        (width / 2, 100.0, 1.0, "///"),
    ):
        bars = ax.bar(
            x + offset, values[slo], width, color=colors,
            edgecolor=[EDGE, EDGE, SPLASH_EDGE], linewidth=0.48,
            alpha=alpha, hatch=hatch, label=f"{int(slo)}-ms SLO", zorder=3,
        )
        for bar, value in zip(bars, values[slo]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, value + 0.12,
                f"{value:.2f}$\\times$", ha="center", va="bottom",
                fontsize=5.35,
            )
    ax.axhline(1, color="#666666", ls=(0, (4, 2)), lw=0.75)
    ax.set_ylim(0, 6.05)
    ax.set_yticks([0, 2, 4, 6])
    ax.set_ylabel("Throughput/GPU / H3\n(geomean)")
    ax.set_xticks(x)
    ax.set_xticklabels(["H3", "Token sparse", "SPLASH"], fontsize=6.2)
    ax.text(
        0.5, 1.30,
        "128K context  |  10 models  |  exhaustive valid batch $\\times$ TP points",
        transform=ax.transAxes, ha="center", va="bottom", fontsize=5.9,
    )
    clean(ax)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
        frameon=False, fontsize=5.8, handlelength=1.1,
        handletextpad=0.35, columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.20, right=0.99, top=0.68, bottom=0.19)
    save(fig, "ablation_validated_128k")


def plane_load() -> None:
    raw = json.loads(
        Path("/home/adityaan/splash_accuracy_tests/planeload_1024.json").read_text()
    )
    rank = 100 * (np.arange(1024) + 1) / 1024
    fig, ax = plt.subplots(figsize=(3.45, 1.48))
    for key, color, label in (
        ("stripe", H3, "Global page top-$k$"),
        ("splash", SPLASH_EDGE, "SPLASH plane quota"),
    ):
        values = np.asarray(raw[key]["load"], dtype=float)
        ax.plot(rank, np.sort(values / values.mean()), color=color, label=label)
    ax.axhline(1, color="#666666", ls=(0, (4, 2)), lw=0.8)
    ax.axhspan(0.75, 1.25, color="#777777", alpha=0.08)
    ax.set_ylim(0.2, 2.45)
    ax.set_ylabel("Plane load /\nmean")
    ax.set_xlabel("Plane-load percentile (%)")
    clean(ax)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
        frameon=False, fontsize=6.1, handlelength=1.2,
        columnspacing=0.9,
    )
    fig.subplots_adjust(left=0.18, right=0.99, top=0.78, bottom=0.23)
    save(fig, "plane_load_distribution")


def plane_rounds() -> None:
    raw = json.loads(
        Path("/home/adityaan/splash_accuracy_tests/planeload_1024.json").read_text()
    )
    fig, ax = plt.subplots(figsize=(3.45, 1.48))
    for key, color, label in (
        ("stripe", H3, "Global page top-$k$"),
        ("splash", SPLASH_EDGE, "SPLASH plane quota"),
    ):
        values = np.sort(np.asarray(raw[key]["rounds"], dtype=float))
        cdf = 100 * (np.arange(len(values)) + 1) / len(values)
        ax.plot(values, cdf, color=color, label=label)
    ax.axvline(98, color=H3, ls=(0, (4, 2)), lw=0.8)
    ax.axvline(64, color=SPLASH_EDGE, ls=(0, (4, 2)), lw=0.8)
    ax.set_ylabel("Events completed (%)")
    ax.set_xlabel("Serialized HBF read rounds")
    clean(ax, "both")
    ax.legend(
        loc="lower right", frameon=False, fontsize=6.1,
        handlelength=1.2,
    )
    fig.subplots_adjust(left=0.18, right=0.99, top=0.96, bottom=0.23)
    save(fig, "plane_round_cdf")


def plane_crossbenchmark() -> None:
    """Per-event plane balance and its serial-read consequence across suites."""
    accuracy_root = Path("/home/adityaan/splash_accuracy_tests")
    stats = []
    for name in (
        "plane_crossbench_lb_corrected.jsonl",
        "plane_crossbench_other_corrected.jsonl",
    ):
        stats_path = accuracy_root / name
        if stats_path.exists():
            stats.extend(
                json.loads(line) for line in stats_path.read_text().splitlines()
                if line.strip()
            )
    pg_path = (
        accuracy_root / "planeload_crossbench_pg19_corrected_gpu0.json"
    )
    if pg_path.exists():
        pg = json.loads(pg_path.read_text())
        for mode, key in (("global", "stripe"), ("plane", "splash")):
            stats.append({
                "suite": "PG-19", "task": "PG-19 128K", "ctx": 131072,
                "mode": mode, "samples": 2, **pg[key]["stats"],
            })

    label_map = {
        ("LongBench", "narrativeqa"): "LB: Narrative QA",
        ("LongBench", "qasper"): "LB: Scientific QA",
        ("LongBench", "hotpotqa"): "LB: Multi-hop QA",
        ("LongBench", "gov_report"): "LB: Summarization",
        ("LongBench", "passage_retrieval_en"): "LB: Retrieval",
        ("LongBench", "passage_count"): "LB: Counting",
        ("LongBench", "lcc"): "LB: Code",
        ("RULER", "niah_multivalue"): "RULER: Multi-value",
        ("RULER", "niah_multiquery"): "RULER: Multi-query",
        ("RULER", "vt"): "RULER: Var. tracing",
        ("RULER", "cwe"): "RULER: Aggregation",
        ("PG-19", "PG-19 128K"): "PG-19: 128K",
        ("Passkey", "Passkey 128K"): "Passkey: 128K",
    }
    order = list(label_map)
    grouped = {}
    for row in stats:
        suite = row["suite"]
        if suite == "Passkey":
            task = "Passkey 128K"
        elif suite == "PG-19":
            task = "PG-19 128K"
        else:
            task = row["task"]
        key = (suite, task)
        if key in label_map:
            grouped.setdefault(key, {})[row["mode"]] = row
    complete = [
        key for key in order
        if {"global", "plane"} <= set(grouped.get(key, {}))
    ]
    if not complete:
        raise RuntimeError("No complete global/plane cross-benchmark statistics")

    labels = [label_map[k] for k in complete]
    global_p99 = np.array([
        grouped[k]["global"]["event_p99_over_mean_mean"] for k in complete
    ])
    splash_p99 = np.array([
        grouped[k]["plane"]["event_p99_over_mean_mean"] for k in complete
    ])
    splash_max = np.array([
        grouped[k]["plane"]["event_max_over_mean_p50"] for k in complete
    ])
    global_rounds = np.array([
        grouped[k]["global"]["rounds_p99"] for k in complete
    ])
    splash_rounds = np.array([
        grouped[k]["plane"]["rounds_p99"] for k in complete
    ])
    events = int(sum(grouped[k]["plane"]["attention_invocations"] for k in complete))

    fig = plt.figure(figsize=(3.45, 2.22))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.02, 1.0], wspace=0.25,
        left=0.31, right=0.985, top=0.76, bottom=0.18,
    )
    axl = fig.add_subplot(gs[0])
    axr = fig.add_subplot(gs[1], sharey=axl)
    y = np.arange(len(complete))[::-1]

    # The p99 body and per-event maximum are deliberately shown together:
    # a long connector with p99 near one is the isolated-impulse signature.
    for yi, p99, mx in zip(y, splash_p99, splash_max):
        axl.plot([p99, mx], [yi, yi], color="#aaa5c8", lw=0.8, zorder=1)
    axl.scatter(global_p99, y, marker="x", s=13, color=H3, linewidths=0.9,
                label="Global p99", zorder=3)
    axl.scatter(splash_p99, y, marker="o", s=13, color=SPLASH_EDGE,
                label="SPLASH p99", zorder=3)
    axl.scatter(splash_max, y, marker="^", s=16, color=HBM,
                label="SPLASH max", zorder=3)
    axl.axvline(1, color="#666666", lw=0.7, ls=(0, (3, 2)))
    axl.set_xscale("log", base=2)
    max_load = max(4.0, float(splash_max.max()) * 1.10)
    ticks = [1, 2, 4, 8, 16, 32]
    axl.set_xlim(0.82, max_load)
    axl.set_xticks([v for v in ticks if v <= max_load * 1.02])
    axl.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    axl.set_yticks(y)
    axl.set_yticklabels(labels, fontsize=5.15)
    axl.tick_params(axis="y", pad=2)
    axl.set_xlabel("Plane load / mean", labelpad=2)
    axl.text(0, 1.17, "(a) p99 and maximum plane load",
             transform=axl.transAxes, ha="left", va="bottom", fontsize=5.55)
    axl.legend(
        loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3,
        frameon=False, fontsize=4.55, handlelength=0.8,
        handletextpad=0.2, columnspacing=0.4, borderaxespad=0,
    )
    clean(axl, "x")

    for yi, before, after in zip(y, global_rounds, splash_rounds):
        axr.plot([after, before], [yi, yi], color="#b8b8b8", lw=0.8, zorder=1)
    axr.scatter(global_rounds, y, marker="s", s=14, color=H3,
                label="Global top-$k$", zorder=3)
    axr.scatter(splash_rounds, y, marker="o", s=14, color=SPLASH_EDGE,
                label="SPLASH", zorder=3)
    axr.set_xlim(0, max(global_rounds.max(), splash_rounds.max()) * 1.12)
    axr.set_xlabel("p99 read rounds", labelpad=2)
    axr.tick_params(axis="y", left=False, labelleft=False)
    axr.text(0, 1.17, "(b) Tail serialized reads",
             transform=axr.transAxes, ha="left", va="bottom", fontsize=5.55)
    axr.legend(
        loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2,
        frameon=False, fontsize=4.7, handlelength=0.8,
        handletextpad=0.2, columnspacing=0.5, borderaxespad=0,
    )
    clean(axr, "x")

    for after in (7, 11, 12):
        if after < len(complete):
            line_y = len(complete) - after - 0.5
            axl.axhline(line_y, color="#bdbdbd", lw=0.45)
            axr.axhline(line_y, color="#bdbdbd", lw=0.45)
    # The bottom two rows are the HBF-saturating long-context cases; short
    # benchmark prompts are intentionally retained to show the underfilled
    # regime where plane balancing is not on the critical path.
    if len(complete) >= 2:
        for axis in (axl, axr):
            axis.axhspan(-0.5, 1.5, color=SPLASH, alpha=0.07, zorder=0)
    fig.text(
        0.64, 0.985,
        f"10% budget | 4 KB | $P$=1024 | {len(complete)} conditions | "
        f"{events:,} events | shaded: HBF-saturating 128K",
        ha="center", va="top", fontsize=5.65,
    )
    save(fig, "plane_balance_crossbenchmark")


def budget_sensitivity() -> None:
    data = json.loads(
        (ROOT / "results" / "design_sensitivities_summary.json").read_text()
    )
    perf = data["budget_performance"]
    quality_rows = data["budget_quality"]
    x = np.array([r["selection_fraction"] * 100 for r in perf])
    fig, ax = plt.subplots(figsize=(3.45, 1.48))
    ax.plot(
        x, [r["throughput_per_gpu"] for r in perf],
        "-o", color=SPLASH_EDGE, ms=3.2,
    )
    ax.set_ylabel("Throughput\n(token/s/GPU)", color=SPLASH_EDGE)
    ax.tick_params(axis="y", labelcolor=SPLASH_EDGE)
    ax.set_xlabel("Selected historical KV (%)")
    ax.set_xticks(x)
    ax.axvline(10, color="#666666", ls=(0, (4, 2)), lw=0.8)
    twin = ax.twinx()
    twin.plot(
        x, [r["ppl_overhead_percent"] for r in quality_rows],
        "--s", color=H3, ms=3.0,
    )
    twin.set_ylabel("PG-19 PPL\noverhead (%)", color=H3)
    twin.tick_params(axis="y", labelcolor=H3)
    twin.spines["top"].set_visible(False)
    clean(ax, "x")
    fig.subplots_adjust(left=0.18, right=0.82, top=0.96, bottom=0.23)
    save(fig, "sensitivity_budget")


def page_sensitivity() -> None:
    data = json.loads(
        (ROOT / "results" / "design_sensitivities_summary.json").read_text()
    )["page_quality"]
    x = np.array([r["page_kb"] for r in data])
    fig, ax = plt.subplots(figsize=(3.45, 1.48))
    ax.plot(
        x, [100 * r["top_budget_token_recall"] for r in data],
        "-o", color=SPLASH_EDGE, ms=3.2,
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in x])
    ax.set_xlabel("Physical HBF page size (KB)")
    ax.set_ylabel("Useful-token\ncoverage (%)", color=SPLASH_EDGE)
    ax.tick_params(axis="y", labelcolor=SPLASH_EDGE)
    twin = ax.twinx()
    twin.plot(
        x, [100 / (2 * r["page_tokens"]) for r in data],
        "--s", color=H3, ms=3.0,
    )
    twin.set_ylabel("Centroids\n(% K+V)", color=H3)
    twin.tick_params(axis="y", labelcolor=H3)
    twin.spines["top"].set_visible(False)
    clean(ax, "x")
    fig.subplots_adjust(left=0.18, right=0.82, top=0.96, bottom=0.23)
    save(fig, "sensitivity_page_size")


def design_sensitivity() -> None:
    """Two actual design knobs, with quality and serving consequences linked."""
    accuracy_root = Path("/home/adityaan/splash_accuracy_tests")
    fracs = np.array([0.02, 0.05, 0.10, 0.20, 0.30, 0.50])

    def load_jsonl(name: str) -> list[dict]:
        return [
            json.loads(line) for line in
            (accuracy_root / name).read_text().splitlines() if line.strip()
        ]

    # Normalize every task independently before aggregating, so task metric
    # scales cannot dominate the suite curve.
    lb = load_jsonl("lb_centroid_frac.jsonl")
    ruler = load_jsonl("ruler_centroid.jsonl")
    ppl = load_jsonl("ppl5.jsonl")

    def suite_retention(rows: list[dict], value_key: str,
                        dense_mode: str = "dense") -> tuple[np.ndarray, np.ndarray]:
        task_key = "task"
        dense = {
            r[task_key]: float(r[value_key]) for r in rows
            if r.get("mode") == dense_mode
        }
        dense_suite_mean = float(np.mean(list(dense.values())))
        med, spread = [], []
        for frac in fracs:
            raw = [
                float(r[value_key])
                for r in rows
                if r.get("mode") == "plane"
                and abs(float(r.get("frac", -1)) - frac) < 1e-9
                and r.get(task_key) in dense
            ]
            med.append(100 * float(np.mean(raw)) / dense_suite_mean)
            spread.append(100 * float(np.std(raw)) / dense_suite_mean)
        return np.asarray(med), np.asarray(spread)

    lb_ret, _ = suite_retention(lb, "score")
    ruler_ret, _ = suite_retention(ruler, "recall")
    dense_ppl = next(
        float(r["ppl"]) for r in ppl
        if int(r["ctx"]) == 32768 and r["mode"] == "dense"
    )
    pg_ret = np.array([
        100 * dense_ppl / next(
            float(r["ppl"]) for r in ppl
            if int(r["ctx"]) == 32768 and r["mode"] == "plane"
            and r["score_mode"] == "centroid"
            and abs(float(r["frac"]) - frac) < 1e-9
        )
        for frac in fracs
    ])

    perf_rows = list(csv.DictReader(
        (ROOT / "results" / "sensitivity_budget_multictx.csv").open()
    ))
    perf_by_ctx = {}
    for ctx in (131072, 524288, 1048576):
        values = []
        for frac in fracs:
            candidates = [
                int(r["batch"]) * 1000
                / (int(r["tp"]) * float(r["tpot_p50_ms"]))
                for r in perf_rows
                if r["status"] == "OK"
                and int(r["context_length"]) == ctx
                and abs(float(r["selection_fraction"]) - frac) < 1e-9
                and float(r["tpot_p50_ms"]) <= 100
            ]
            values.append(max(candidates))
        perf_by_ctx[ctx] = np.asarray(values)

    design = json.loads(
        (ROOT / "results" / "design_sensitivities_summary.json").read_text()
    )
    page = design["page_quality"]
    page_x = np.asarray([r["page_kb"] for r in page])
    coverage = np.asarray([100 * r["top_budget_token_recall"] for r in page])
    amp = np.asarray([r["read_amplification_token_baseline"] for r in page])
    centroid = np.asarray([100 / (2 * r["page_tokens"]) for r in page])

    fig = plt.figure(figsize=(3.45, 2.82))
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1, 1.04], hspace=0.68, wspace=0.74,
        left=0.16, right=0.98, top=0.89, bottom=0.13,
    )
    axq = fig.add_subplot(gs[0, 0])
    axt = fig.add_subplot(gs[0, 1])
    axp = fig.add_subplot(gs[1, :])

    x = fracs * 100
    quality_lines = (
        ("LongBench (6)", lb_ret, SPLASH_EDGE, "o"),
        ("RULER (6)", ruler_ret, H3, "s"),
        ("PG-19 PPL", pg_ret, HBM, "^"),
    )
    for label, values, color, marker in quality_lines:
        axq.plot(x, values, color=color, marker=marker, ms=2.7, lw=1.0,
                 label=label)
    axq.axvline(10, color="#777777", lw=0.65, ls=(0, (3, 2)))
    axq.axhline(100, color="#777777", lw=0.65, ls=(0, (3, 2)))
    axq.set_ylim(92, 105)
    axq.set_yticks([92, 96, 100, 104])
    axq.set_xticks([2, 10, 30, 50])
    axq.set_ylabel("Quality / Full (%)", fontsize=7.5)
    axq.set_xlabel("Selected cold KV (%)", labelpad=2)
    axq.text(
        0, 1.05,
        "(a) End-task quality\nLB $\\leq$31.5K; RULER 16K; PG-19 32K",
        transform=axq.transAxes, ha="left", va="bottom", fontsize=5.65,
    )
    axq.legend(loc="lower right", frameon=False, fontsize=5.0,
               handlelength=1.15, handletextpad=0.25, borderpad=0.1)
    clean(axq)

    for ctx, color, marker in (
        (131072, NAIVE, "o"), (524288, H3, "s"), (1048576, SPLASH_EDGE, "^")
    ):
        label = f"{ctx // 1024}K" if ctx < 1048576 else "1M"
        axt.plot(x, perf_by_ctx[ctx], color=color, marker=marker, ms=2.7,
                 lw=1.0, label=label)
    axt.axvline(10, color="#777777", lw=0.65, ls=(0, (3, 2)))
    axt.set_xticks([2, 10, 30, 50])
    axt.set_ylim(70, 470)
    axt.set_yticks([100, 250, 400])
    axt.set_ylabel("Throughput/GPU\n(tokens/s)", fontsize=7.5)
    axt.set_xlabel("Selected cold KV (%)", labelpad=2)
    axt.text(
        0, 1.05,
        "(b) SLO-constrained serving\nLlama-3-8B, TP=8, TPOT $\\leq$100 ms",
        transform=axt.transAxes, ha="left", va="bottom", fontsize=5.65,
    )
    axt.legend(loc="upper right", frameon=False, fontsize=5.2,
               ncol=1, handlelength=1.1, handletextpad=0.25)
    clean(axt)

    axp.plot(page_x, coverage, color=SPLASH_EDGE, marker="o", ms=3.0,
             lw=1.1, label="True top-budget tokens covered")
    axp.plot(page_x, 100 * centroid / centroid[0], color=H3, marker="s",
             ms=2.8, lw=1.0, ls="--", label="Centroid bytes (4-KB=100%)")
    axp.set_xscale("log", base=2)
    axp.set_xticks(page_x)
    axp.set_xticklabels([str(int(v)) for v in page_x])
    axp.set_ylim(0, 110)
    axp.set_yticks([0, 50, 100])
    axp.set_ylabel("Coverage / metadata (%)", fontsize=7.5)
    axp.set_xlabel("Physical HBF page size (KB; minimum = 4 KB)", labelpad=2)
    twin = axp.twinx()
    twin.plot(page_x, amp, color=HBM, marker="^", ms=3.0, lw=1.0,
              label="Token-selection read amplification")
    twin.set_ylim(3, 7.3)
    twin.set_yticks([4, 5, 6, 7])
    twin.set_ylabel("Read amplification ($\\times$)", color=HBM,
                    fontsize=7.5)
    twin.tick_params(axis="y", labelcolor=HBM)
    twin.spines["top"].set_visible(False)
    handles1, labels1 = axp.get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    axp.legend(handles1 + handles2, labels1 + labels2, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False,
               fontsize=5.1, handlelength=1.2, handletextpad=0.3,
               columnspacing=0.6)
    axp.text(0, 1.16,
             "(c) Page size: real 128K attention, 10% budget, $P$=1024",
             transform=axp.transAxes, ha="left", va="bottom", fontsize=6.0)
    clean(axp)
    save(fig, "sensitivity_design_knobs")


def energy() -> None:
    summary = json.loads(
        (ROOT / "results" / "energy_b200" /
         "provisional_hybrid_summary.json").read_text()
    )
    data = summary["sensitivity_cases"]
    central = next(
        r for r in data
        if r["active_power_w_per_gpu"] == 700
        and abs(r["hbm_pj_per_byte"] - 11.926605504587155) < 1e-9
        and r["hbf_array_pj_per_byte"] == 64
    )
    dense = [r["dense_over_splash_energy_geomean"] for r in data]
    naive = [r["naive_over_splash_energy_geomean"] for r in data]
    values = np.array([
        central["dense_over_splash_energy_geomean"],
        central["naive_over_splash_energy_geomean"], 1.0,
    ])
    low = np.array([values[0] - min(dense), values[1] - min(naive), 0])
    high = np.array([max(dense) - values[0], max(naive) - values[1], 0])
    fig = plt.figure(figsize=(3.45, 2.55))
    gs = fig.add_gridspec(
        2, 1, height_ratios=[1.0, 1.08], hspace=0.58,
        left=0.20, right=0.96, top=0.84, bottom=0.13,
    )
    ax = fig.add_subplot(gs[0])
    axh = fig.add_subplot(gs[1])
    x = np.arange(3)
    bars = ax.bar(
        x, values, 0.60, color=[H3, NAIVE, SPLASH], edgecolor=EDGE,
        linewidth=0.5, yerr=np.vstack([low, high]),
        error_kw={"elinewidth": 0.8, "capsize": 2.5}, zorder=3,
    )
    bars[-1].set_hatch("///")
    bars[-1].set_edgecolor(SPLASH_EDGE)
    ax.set_xticks(x)
    ax.set_xticklabels(["H3", "Naive Sparse", "SPLASH (Ours)"])
    ax.set_ylabel("Energy /\nSPLASH")
    ax.set_ylim(0, 6.8)
    clean(ax)
    ax.text(
        0.5, 1.34,
        "Hybrid projection (not measured B200 power)  |  "
        f"{summary['selected_operating_points']} matched operating points",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=5.9, color="#7a2f2f",
    )
    ax.text(
        0.0, 1.04,
        "(a) Central estimate; whiskers span all power/energy assumptions",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=5.8,
    )

    powers = [450.0, 700.0, 1000.0]
    hbf_costs = [32.0, 64.0, 96.0]
    matrix = np.empty((3, 3))
    for i, power in enumerate(powers):
        for j, hbf_cost in enumerate(hbf_costs):
            cases = [
                r for r in data
                if r["active_power_w_per_gpu"] == power
                and r["hbf_array_pj_per_byte"] == hbf_cost
                and abs(
                    r["hbm_pj_per_byte"] - 11.926605504587155
                ) < 1e-9
            ]
            matrix[i, j] = cases[0]["dense_over_splash_energy_geomean"]
    image_ = axh.imshow(
        matrix, aspect="auto", cmap="Purples", vmin=5.9, vmax=6.4,
    )
    for i in range(3):
        for j in range(3):
            axh.text(
                j, i, f"{matrix[i, j]:.2f}$\\times$",
                ha="center", va="center", fontsize=5.7,
                color="white" if matrix[i, j] > 6.2 else "black",
            )
    axh.set_xticks(range(3))
    axh.set_xticklabels(["32", "64", "96"])
    axh.set_yticks(range(3))
    axh.set_yticklabels(["450", "700", "1000"])
    axh.set_xlabel("HBF array energy (pJ/byte)", labelpad=2)
    axh.set_ylabel("GPU active\npower (W)")
    axh.text(
        0.0, 1.06,
        "(b) H3/SPLASH energy across assumptions (HBM=11.93 pJ/byte)",
        transform=axh.transAxes, ha="left", va="bottom", fontsize=5.8,
    )
    axh.tick_params(axis="both", length=0)
    for spine in axh.spines.values():
        spine.set_visible(False)
    save(fig, "energy_projection")


def main() -> None:
    granularity_ablation()
    attention()
    validated_system_ablation()
    if (
        Path("/home/adityaan/splash_accuracy_tests") /
        "planeload_crossbench_pg19_corrected_gpu0.json"
    ).exists():
        plane_crossbenchmark()
    design_sensitivity()
    energy()
    print(OUT)


if __name__ == "__main__":
    main()
