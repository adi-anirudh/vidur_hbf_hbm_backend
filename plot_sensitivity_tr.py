#!/usr/bin/env python3
"""Plot and summarize the physical tR sensitivity experiment."""
from __future__ import annotations

import csv
import json
import math
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

CSV = Path("results/sensitivity_tr.csv")
OUT = Path("results/plots")
SUMMARY = Path("results/sensitivity_tr_summary.json")

rows = [
    r for r in csv.DictReader(CSV.open())
    if r["status"] == "OK"
    and capacity_feasible(
        r["model"], int(r["batch"]), int(r["context_length"]),
        int(r["tp"]), "hbf",
    )
]
groups = defaultdict(list)
for r in rows:
    groups[(
        r["model"], int(r["context_length"]), r["system"], float(r["tr_ns"])
    )].append(r)

selected = {}
for key, rs in groups.items():
    point = best_slo_point(rs)
    if point:
        selected[key] = {
            **point,
            "throughput_per_gpu": throughput_per_gpu(
                int(point["batch"]), int(point["tp"]),
                float(point["tpot_p50_ms"]),
            ),
        }

trs = sorted({k[3] for k in selected})
workloads = sorted({(k[0], k[1]) for k in selected})
nominal = PLATFORM.hbf_nominal_tr_ns


def geomean(values):
    vals = [v for v in values if v > 0]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else math.nan


summary = {
    "slo_ms": PLATFORM.tpot_slo_ms,
    "models": sorted({model for model, _ in workloads}),
    "contexts": sorted({context for _, context in workloads}),
    "fixed_tp_by_model": {
        model: int(next(
            point["tp"] for key, point in selected.items() if key[0] == model
        ))
        for model in sorted({model for model, _ in workloads})
    },
    "points": [],
}
for tr in trs:
    speedups = []
    splash_rel_nominal = []
    h3_rel_nominal_splash = []
    for model, context in workloads:
        s = selected.get((model, context, "SPLASH", tr))
        h = selected.get((model, context, "H3", tr))
        sn = selected.get((model, context, "SPLASH", nominal))
        if s and h:
            speedups.append(s["throughput_per_gpu"] / h["throughput_per_gpu"])
        if s and sn:
            splash_rel_nominal.append(
                s["throughput_per_gpu"] / sn["throughput_per_gpu"]
            )
        if h and sn:
            h3_rel_nominal_splash.append(
                h["throughput_per_gpu"] / sn["throughput_per_gpu"]
            )
    summary["points"].append({
        "tr_ns": tr,
        "effective_hbf_bw_gbps": PLATFORM.effective_hbf_bw_gbps(tr),
        "num_speedup_pairs": len(speedups),
        "splash_vs_h3_geomean": geomean(speedups),
        "splash_vs_h3_minimum": min(speedups) if speedups else math.nan,
        "splash_vs_h3_maximum": max(speedups) if speedups else math.nan,
        "splash_throughput_vs_nominal_geomean": geomean(splash_rel_nominal),
        "h3_throughput_vs_nominal_splash_geomean": geomean(
            h3_rel_nominal_splash
        ),
    })

summary["selected_operating_points"] = [
    {
        "model": model,
        "context_length": context,
        "system": system,
        "tr_ns": tr,
        "effective_hbf_bw_gbps": PLATFORM.effective_hbf_bw_gbps(tr),
        "tp": int(point["tp"]),
        "batch": int(point["batch"]),
        "hbm_kv_fraction": float(point["hbm_kv_fraction"]),
        "tpot_p50_ms": float(point["tpot_p50_ms"]),
        "tpot_p99_ms": float(point["tpot_p99_ms"]),
        "throughput_per_gpu": point["throughput_per_gpu"],
    }
    for (model, context, system, tr), point in sorted(selected.items())
]

SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.7,
    "grid.alpha": 0.3,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
})

x = np.array([p["tr_ns"] / 1000 for p in summary["points"]])
bw = np.array([p["effective_hbf_bw_gbps"] / 1000 for p in summary["points"]])
sp = np.array([p["splash_throughput_vs_nominal_geomean"] for p in summary["points"]])
h3 = np.array([p["h3_throughput_vs_nominal_splash_geomean"] for p in summary["points"]])
gain = np.array([p["splash_vs_h3_geomean"] for p in summary["points"]])

fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.45))
a, b = axes
a.plot(x, sp, "-o", color="#4c72b0", label="SPLASH", lw=1.5, ms=4)
a.plot(x, h3, "-s", color="#c44e52", label="H3", lw=1.5, ms=4)
a.axvline(nominal / 1000, color="0.4", ls=":", lw=0.9)
a.set_xlabel(r"NAND read latency $t_R$ ($\mu$s)")
a.set_ylabel("Throughput / GPU\n(norm. to nominal SPLASH)")
a.set_xscale("log", base=2)
a.set_xticks(x); a.set_xticklabels([f"{v:g}" for v in x])
a.grid(True, axis="y")
a.legend(frameon=False)

b.plot(x, gain, "-o", color="#6b5b95", lw=1.6, ms=4)
b.axvline(nominal / 1000, color="0.4", ls=":", lw=0.9)
b.set_xlabel(r"NAND read latency $t_R$ ($\mu$s)")
b.set_ylabel("SPLASH / H3 throughput")
b.set_xscale("log", base=2)
b.set_xticks(x); b.set_xticklabels([f"{v:g}" for v in x])
b.grid(True, axis="y")
for xi, yi, bi in zip(x, gain, bw):
    b.annotate(f"{yi:.1f}$\\times$\n{bi:.1f} TB/s", (xi, yi),
               xytext=(0, 5), textcoords="offset points",
               ha="center", fontsize=6.3)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"sensitivity_tr.{ext}", bbox_inches="tight")
print(json.dumps(summary, indent=2))
