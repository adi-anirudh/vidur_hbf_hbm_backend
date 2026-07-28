#!/usr/bin/env python3
"""Byte-faithful attention data-path breakdown for the SPLASH ablation.

This is not a second performance model.  It expands the exact flat-bandwidth
critical-path equations used by HBFLinearRegressionExecutionTimePredictor into
named components so the paper can attribute each latency reduction.

For sparse methods:
    T_attention = max(T_hot-HBM, T_score) + T_selected-HBF

Dense HBF has no selection stage:
    T_attention = T_hot-HBM + T_cold-HBF

All byte counts are per GPU after tensor-parallel sharding and include every
decoder layer.  Output latencies are therefore complete decode-step attention
data-path latencies, not per-layer values.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import publication_style as ps
from evaluation_common import (
    PLATFORM,
    capacity_feasible,
    hbm_kv_fraction,
)
from sweep_capacity import kv_bytes


ROOT = Path(__file__).resolve().parent
OUT_CSV = ROOT / "results" / "attention_mechanism_breakdown.csv"
OUT_JSON = ROOT / "results" / "attention_mechanism_breakdown.json"
OUT_FIG = ROOT / "results" / "plots" / "attention_mechanism_breakdown"

MODEL = "meta-llama/Meta-Llama-3-8B"
TP = 8
BATCH = 64
CONTEXTS = (131072, 262144, 524288, 1048576, 2097152)
SELECTION = 0.10
PAGE_TOKENS = 16             # 4-KB minimum physical page
READ_AMP = 3.5635            # measured P=1024, 128K, 10% budget
PLANE_IMBALANCE = 2.175      # measured P=1024, 128K, 10% budget

SYSTEMS = (
    "Dense HBF",
    "Token-granular",
    "Page + full-K score",
    "Global page top-k",
    "SPLASH",
)


def time_ms(num_bytes: float) -> float:
    return num_bytes / (PLATFORM.effective_hbf_bw_gbps() * 1e6)


def row_for(context: int, system: str) -> dict:
    hot_fraction = hbm_kv_fraction(MODEL, BATCH, context, TP, "hbf")
    total = kv_bytes(MODEL, BATCH, context) / TP
    hot = total * hot_fraction
    cold = total - hot

    hot_ms = time_ms(hot)
    score_bytes = 0.0
    selected_bytes = 0.0
    useful_bytes = cold
    plane_tail_factor = 1.0

    if system == "Dense HBF":
        selected_bytes = cold
    elif system == "Token-granular":
        score_bytes = cold / 2.0
        useful_bytes = cold * SELECTION / 2.0
        selected_bytes = cold / 2.0 * min(1.0, SELECTION * READ_AMP)
    elif system == "Page + full-K score":
        score_bytes = cold / 2.0
        useful_bytes = cold * SELECTION / 2.0
        selected_bytes = useful_bytes
    elif system == "Global page top-k":
        score_bytes = cold / (2.0 * PAGE_TOKENS)
        useful_bytes = cold * SELECTION
        # Global and plane-balanced page top-k return the same aggregate K+V
        # bytes. Imbalance stretches the critical path through the busiest
        # plane; it must not be reported as additional logical HBF traffic.
        selected_bytes = useful_bytes
        plane_tail_factor = PLANE_IMBALANCE
    elif system == "SPLASH":
        score_bytes = cold / (2.0 * PAGE_TOKENS)
        useful_bytes = cold * SELECTION
        selected_bytes = useful_bytes
    else:
        raise ValueError(system)

    score_ms = time_ms(score_bytes)
    selected_ms = time_ms(selected_bytes)
    plane_tail_ms = selected_ms * (plane_tail_factor - 1.0)
    exposed_score_ms = max(0.0, score_ms - hot_ms)
    critical_ms = hot_ms + exposed_score_ms + selected_ms + plane_tail_ms
    return {
        "model": MODEL,
        "tp": TP,
        "batch": BATCH,
        "context_length": context,
        "system": system,
        "selection_fraction": SELECTION,
        "page_tokens": PAGE_TOKENS,
        "page_kb": 4,
        "hbm_kv_fraction": hot_fraction,
        "hot_hbm_gb": hot / 1e9,
        "score_scan_gb": score_bytes / 1e9,
        "selected_hbf_gb": selected_bytes / 1e9,
        "useful_selected_gb": useful_bytes / 1e9,
        "hot_hbm_ms": hot_ms,
        "score_scan_ms": score_ms,
        "exposed_score_ms": exposed_score_ms,
        "selected_hbf_ms": selected_ms,
        "plane_tail_ms": plane_tail_ms,
        "busiest_plane_factor": plane_tail_factor,
        "attention_critical_ms": critical_ms,
        "score_hidden": score_ms <= hot_ms,
        "capacity_feasible": capacity_feasible(
            MODEL, BATCH, context, TP, "hbf"
        ),
    }


def plot(rows: list[dict]) -> None:
    ps.apply()
    colors = {
        "hot": ps.COLORS["hbm"],
        "score": ps.COLORS["token"],
        "read": ps.COLORS["dense"],
    }
    line_colors = {
        "Dense HBF": ps.COLORS["dense"],
        "Token-granular": ps.COLORS["token"],
        "Page + full-K score": ps.COLORS["full_score"],
        "Global page top-k": ps.COLORS["global"],
        "SPLASH": ps.COLORS["splash"],
    }

    plt.rcParams.update({"axes.titlesize": 7.0, "axes.labelsize": 6.4,
                         "xtick.labelsize": 5.8, "ytick.labelsize": 5.8,
                         "legend.fontsize": 5.5})
    fig, axes = plt.subplots(2, 1, figsize=(3.4, 4.35))
    representative = 1_048_576
    rr = [r for r in rows if r["context_length"] == representative]
    x = np.arange(len(rr))
    hot = np.array([r["hot_hbm_ms"] for r in rr])
    exposed = np.array([r["exposed_score_ms"] for r in rr])
    selected = np.array([r["selected_hbf_ms"] for r in rr])
    axes[0].bar(x, hot, color=colors["hot"], label="Hot HBM")
    axes[0].bar(
        x, exposed, bottom=hot, color=colors["score"],
        label="Exposed scoring",
    )
    axes[0].bar(
        x, selected, bottom=hot + exposed, color=colors["read"],
        label="HBF page read",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(
        ["Dense", "Token", "Page", "Global", "SPLASH"],
        rotation=22, ha="right",
    )
    axes[0].set_ylabel("Crit. path\n(ms/token)")
    axes[0].set_title("(a) Where attention latency goes at 1M")
    ps.finish_axis(axes[0], grid_axis="y")
    axes[0].legend(
        ncol=3, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, 1.00),
    )

    for system in SYSTEMS:
        sr = [r for r in rows if r["system"] == system]
        axes[1].plot(
            [r["context_length"] / 1024 for r in sr],
            [r["attention_critical_ms"] for r in sr],
            marker="o",
            color=line_colors[system], label=system,
        )
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks([128, 256, 512, 1024, 2048])
    axes[1].set_xticklabels(["128K", "256K", "512K", "1M", "2M"])
    axes[1].set_xlabel("Context length")
    axes[1].set_ylabel("Crit. path\n(ms/token)")
    axes[1].set_title("(b) Mechanism scaling with context")
    ps.finish_axis(axes[1])
    axes[1].legend(frameon=False, ncol=1, loc="upper left")

    fig.tight_layout(w_pad=1.6)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_FIG}.{ext}", bbox_inches="tight", dpi=220)
    plt.close(fig)


def main() -> None:
    rows = [
        row_for(context, system)
        for context in CONTEXTS
        for system in SYSTEMS
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    OUT_JSON.write_text(json.dumps({
        "contract": {
            "model": MODEL,
            "tp": TP,
            "batch": BATCH,
            "selection_fraction": SELECTION,
            "page_tokens": PAGE_TOKENS,
            "page_kb": 4,
            "read_amplification": READ_AMP,
            "plane_imbalance": PLANE_IMBALANCE,
            "equation": (
                "sparse: max(hot_hbm, score_scan) + selected_hbf "
                "+ plane_serialization_tail; "
                "dense: hot_hbm + cold_hbf"
            ),
            "bandwidth_gbps": PLATFORM.effective_hbf_bw_gbps(),
            "provenance": (
                "byte expansion of the HBF predictor; amplification and "
                "imbalance measured on real Llama-3.1-8B/PG-19 attention "
                "with P=1024 at 128K and 10% selection"
            ),
        },
        "rows": rows,
    }, indent=2))
    plot(rows)
    print(OUT_CSV)
    for r in rows:
        if r["context_length"] == 1_048_576:
            print(
                f"{r['system']:22s} {r['attention_critical_ms']:8.3f} ms "
                f"(hot {r['hot_hbm_ms']:.3f}, exposed-score "
                f"{r['exposed_score_ms']:.3f}, read "
                f"{r['selected_hbf_ms']:.3f}, tail "
                f"{r['plane_tail_ms']:.3f})"
            )


if __name__ == "__main__":
    main()
