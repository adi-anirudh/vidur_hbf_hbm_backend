#!/usr/bin/env python3
"""Plot the end-to-end ablation at 1M and its context-length robustness."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation_common import (
    PLATFORM,
    best_slo_point,
    capacity_feasible,
    throughput_per_gpu,
)
import publication_style as ps

CSV = Path("results/ablation_matrix.csv")
OUT = Path("results/plots")
SUMMARY = Path("results/ablation_matrix_summary.json")
TARGET_CONTEXT = 1048576

rows = [
    r for r in csv.DictReader(CSV.open())
    if r["status"] == "OK"
    and capacity_feasible(
        r["model"], int(r["batch"]), int(r["context_length"]),
        int(r["tp"]), "hbf",
    )
]
groups = defaultdict(list)
for row in rows:
    groups[(int(row["context_length"]), row["system"])].append(row)

selected = {}
for key, candidates in groups.items():
    point = best_slo_point(candidates)
    if point:
        selected[key] = {
            **point,
            "throughput_per_gpu": throughput_per_gpu(
                int(point["batch"]),
                int(point["tp"]),
                float(point["tpot_p50_ms"]),
            ),
        }

contexts = sorted({key[0] for key in selected})
systems = ("Dense", "TokenGranular", "FullPageScore", "PlaneImbalance", "SPLASH")
summary = {
    "model": "meta-llama/Meta-Llama-3-8B",
    "tp": 8,
    "slo_ms": PLATFORM.tpot_slo_ms,
    "selection_policy": "maximum throughput/GPU among SLO-feasible batches",
    "contexts": [],
}
for context in contexts:
    dense = selected.get((context, "Dense"))
    if not dense:
        continue
    item = {"context_length": context, "systems": {}}
    for system in systems:
        point = selected.get((context, system))
        if not point:
            continue
        item["systems"][system] = {
            "batch": int(point["batch"]),
            "tpot_p50_ms": float(point["tpot_p50_ms"]),
            "throughput_per_gpu": point["throughput_per_gpu"],
            "normalized_to_dense": (
                point["throughput_per_gpu"] / dense["throughput_per_gpu"]
            ),
        }
    summary["contexts"].append(item)
SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")

ps.apply()

style = {
    "Dense": ("H3 (dense)", ps.COLORS["dense"], "s"),
    "TokenGranular": ("Token-granular", ps.COLORS["token"], "v"),
    "FullPageScore": ("Full-key scoring", ps.COLORS["full_score"], "^"),
    "PlaneImbalance": ("Global top-$k$", ps.COLORS["global"], "D"),
    "SPLASH": ("SPLASH", ps.COLORS["splash"], "o"),
}

plt.rcParams.update({"axes.titlesize": 6.8, "axes.labelsize": 6.6,
                     "xtick.labelsize": 6.3, "ytick.labelsize": 6.3,
                     "legend.fontsize": 5.6})
fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.7))
bar_order = ("SPLASH", "PlaneImbalance", "FullPageScore", "TokenGranular", "Dense")
dense_target = selected[(TARGET_CONTEXT, "Dense")]["throughput_per_gpu"]
bar_values = [
    selected[(TARGET_CONTEXT, s)]["throughput_per_gpu"] / dense_target
    for s in bar_order
]
axes[0].bar(
    np.arange(len(bar_order)), bar_values, width=0.68,
    color=[style[s][1] for s in bar_order], edgecolor="black", linewidth=0.45,
    zorder=3,
)
axes[0].set_xticks(np.arange(len(bar_order)))
axes[0].set_xticklabels(
    ["SPLASH", "Global\n$k$", "Full-key\nscore", "Token-\ngran.", "H3"]
)
axes[0].set_ylabel("Thr./GPU vs H3\n(100 ms SLO)")
axes[0].set_title("(a) Factor isolation at 1M")
ps.finish_axis(axes[0], grid_axis="y")
for x, value in enumerate(bar_values):
    axes[0].text(x, value + 0.08, f"{value:.2f}$\\times$",
                 ha="center", va="bottom", fontsize=ps.ANNOTATION_SIZE)
axes[0].set_ylim(0, max(bar_values) * 1.18)

x = np.arange(len(contexts))
for system in systems:
    y = []
    for context in contexts:
        dense = selected[(context, "Dense")]["throughput_per_gpu"]
        point = selected.get((context, system))
        y.append(point["throughput_per_gpu"] / dense if point else np.nan)
    label, color, marker = style[system]
    axes[1].plot(x, y, label=label, color=color, marker=marker)
axes[1].set_xticks(x)
axes[1].set_xticklabels(
    [f"{c // 1024}K" if c < 1048576 else f"{c // 1048576}M" for c in contexts]
)
axes[1].set_xlabel("Context length")
axes[1].set_ylabel("Thr./GPU vs H3")
axes[1].set_title("(b) Ablation across context")
ps.finish_axis(axes[1], grid_axis="y")
axes[1].legend(frameon=False, ncol=2, loc="upper left")

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"ablation_matrix.{ext}", bbox_inches="tight")
print(json.dumps(summary, indent=2))
