#!/usr/bin/env python3
"""Plot the two paper-facing SPLASH design knobs: budget and page size."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation_common import PLATFORM, best_slo_point, throughput_per_gpu
import publication_style as ps


ROOT = Path(__file__).resolve().parent
ACC = Path("/home/adityaan/splash_accuracy_tests")
OUT = ROOT / "results" / "plots" / "design_sensitivities"
SUMMARY = ROOT / "results" / "design_sensitivities_summary.json"


def load_budget_performance() -> list[dict]:
    rows = list(csv.DictReader(
        (ROOT / "results" / "sensitivity_budget.csv").open()
    ))
    out = []
    for frac in sorted({float(r["selection_fraction"]) for r in rows}):
        point = best_slo_point([
            r for r in rows if float(r["selection_fraction"]) == frac
        ])
        if point:
            out.append({
                "selection_fraction": frac,
                "batch": int(point["batch"]),
                "tpot_p50_ms": float(point["tpot_p50_ms"]),
                "tpot_p99_ms": float(point["tpot_p99_ms"]),
                "throughput_per_gpu": throughput_per_gpu(
                    int(point["batch"]), int(point["tp"]),
                    float(point["tpot_p50_ms"]),
                ),
            })
    return out


def load_budget_quality() -> list[dict]:
    rows = [json.loads(line) for line in
            (ACC / "centroid_ppl.jsonl").open() if line.strip()]
    dense = next(
        r for r in rows
        if r["ctx"] == 32768 and r["mode"] == "dense"
    )["ppl"]
    out = []
    for r in rows:
        if (
            r["ctx"] == 32768
            and r.get("score_mode") == "centroid"
            and r["mode"] == "global"
        ):
            out.append({
                "selection_fraction": float(r["frac"]),
                "ppl": float(r["ppl"]),
                "ppl_overhead_percent": (float(r["ppl"]) / dense - 1.0) * 100,
            })
    return sorted(out, key=lambda x: x["selection_fraction"])


def load_page_performance() -> list[dict]:
    rows = list(csv.DictReader(
        (ROOT / "results" / "sensitivity_page_size.csv").open()
    ))
    out = []
    for page_kb in sorted({int(r["page_kb"]) for r in rows}):
        point = best_slo_point([
            r for r in rows if int(r["page_kb"]) == page_kb
        ])
        if point:
            out.append({
                "page_kb": page_kb,
                "page_tokens": int(point["page_tokens"]),
                "batch": int(point["batch"]),
                "tpot_p50_ms": float(point["tpot_p50_ms"]),
                "tpot_p99_ms": float(point["tpot_p99_ms"]),
                "throughput_per_gpu": throughput_per_gpu(
                    int(point["batch"]), int(point["tp"]),
                    float(point["tpot_p50_ms"]),
                ),
                "centroid_fraction_of_kv": float(
                    point["centroid_fraction_of_kv"]
                ),
            })
    return out


def load_page_quality() -> list[dict]:
    token_to_kb = {16: 4, 32: 8, 64: 16, 128: 32}
    rows = [json.loads(line) for line in
            (ACC / "useful_hist_ps.jsonl").open() if line.strip()]
    out = []
    for r in rows:
        tokens = int(r["page_size"])
        if tokens not in token_to_kb:
            continue
        out.append({
            "page_kb": token_to_kb[tokens],
            "page_tokens": tokens,
            "top_budget_token_recall": float(r["recall_budget"]),
            "read_amplification_token_baseline": float(r["amp"]),
            "useful_tokens_per_page": float(r["useful_per_page_page"]),
        })
    return sorted(out, key=lambda x: x["page_kb"])


def main() -> None:
    bp = load_budget_performance()
    bq = load_budget_quality()
    pp = load_page_performance()
    pq = load_page_quality()
    if not (bp and bq and pp and pq):
        raise SystemExit("missing sensitivity inputs")

    ps.apply()
    blue = ps.COLORS["splash"]
    rust = ps.COLORS["dense"]
    gold = ps.COLORS["token"]
    green = ps.COLORS["global"]
    plt.rcParams.update({"axes.titlesize":7.0,"axes.labelsize":6.4,"xtick.labelsize":5.8,"ytick.labelsize":5.8,"legend.fontsize":5.5})
    fig, axes = plt.subplots(4, 1, figsize=(3.4, 6.6))

    x = [r["selection_fraction"] * 100 for r in bp]
    y = [r["throughput_per_gpu"] for r in bp]
    axes[0].plot(x, y, "-o", color=blue, lw=1.8, ms=4)
    last_batch = None
    for r, xx, yy in zip(bp, x, y):
        if r["batch"] != last_batch:
            axes[0].annotate(
                f"B={r['batch']}", (xx, yy), xytext=(4, 5),
                textcoords="offset points", ha="left",
                fontsize=ps.ANNOTATION_SIZE,
            )
            last_batch = r["batch"]
    axes[0].set_ylabel("SLO throughput (token/s/GPU)")
    axes[0].set_title("(a) Selection budget: performance")

    xq = [r["selection_fraction"] * 100 for r in bq]
    yq = [r["ppl_overhead_percent"] for r in bq]
    axes[1].plot(xq, yq, "-o", color=rust, lw=1.8, ms=4)
    axes[1].axvline(10, color="#777", ls=":", lw=1)
    axes[1].annotate(
        "default", (10, np.interp(10, xq, yq)),
        xytext=(5, 10), textcoords="offset points",
        fontsize=ps.ANNOTATION_SIZE,
    )
    axes[1].set_ylabel("PG-19 perplexity overhead (%)")
    axes[1].set_title("(b) Selection budget: quality")

    xp = [r["page_kb"] for r in pp]
    yp = [r["throughput_per_gpu"] for r in pp]
    axes[2].plot(xp, yp, "-o", color=green, lw=1.8, ms=4)
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks([4, 8, 16, 32])
    axes[2].set_xticklabels(["4", "8", "16", "32"])
    axes[2].set_xlabel("Physical HBF page size (KB)")
    axes[2].set_ylabel("SLO throughput (token/s/GPU)")
    axes[2].set_ylim(0, max(yp) * 1.12)
    axes[2].text(
        5.0, max(yp) * 0.82,
        "centroid scan hidden\nby hot-HBM attention",
        color=green, fontsize=ps.ANNOTATION_SIZE,
    )
    axes[2].set_title("(c) Page size: end-to-end performance")

    xr = [r["page_kb"] for r in pq]
    recall = [r["top_budget_token_recall"] * 100 for r in pq]
    metadata = [100 / (2 * r["page_tokens"]) for r in pq]
    axes[3].plot(
        xr, recall, "-o", color=blue, lw=1.8, ms=4,
        label="Top-budget token recall",
    )
    axes[3].set_xscale("log", base=2)
    axes[3].set_xticks([4, 8, 16, 32])
    axes[3].set_xticklabels(["4", "8", "16", "32"])
    axes[3].set_xlabel("Physical HBF page size (KB)")
    axes[3].set_ylabel("Top-10% token coverage (%)", color=blue)
    axes[3].tick_params(axis="y", labelcolor=blue)
    ax2 = axes[3].twinx()
    ax2.plot(
        xr, metadata, "--s", color=gold, lw=1.5, ms=3.5,
        label="Centroid footprint",
    )
    ax2.set_ylabel("Centroids (% of K+V)", color=gold)
    ax2.tick_params(axis="y", labelcolor=gold)
    axes[3].set_title("(d) Page size: quality vs metadata")

    for ax in axes.flat:
        ps.finish_axis(ax)
    ps.finish_twin_axis(ax2)
    fig.tight_layout(w_pad=1.5, h_pad=1.6)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)

    SUMMARY.write_text(json.dumps({
        "contract": {
            "performance_model": (
                "Llama-3-8B, 1M context, TP=8; maximize throughput/GPU "
                "under 100-ms TPOT-p50 over batch 1..128"
            ),
            "budget_quality": (
                "Llama-3.1-8B PG-19 at 32K, centroid global page selection"
            ),
            "page_quality": (
                "real Llama-3.1-8B/PG-19 attention at 128K, P=1024, "
                "10% selected-token budget"
            ),
            "page_mapping": (
                "separate FP16 K and V head streams, dimension 128: "
                "256 B/token in either stream; 4/8/16/32 KB = "
                "16/32/64/128 tokens"
            ),
            "minimum_page_kb": 4,
        },
        "budget_performance": bp,
        "budget_quality": bq,
        "page_performance": pp,
        "page_quality": pq,
    }, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
