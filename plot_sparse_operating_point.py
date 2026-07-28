#!/usr/bin/env python3
"""Matched 10% logical-token-budget performance: token vs page sparsity."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import publication_style as ps

ROOT = Path(__file__).resolve().parent
ABLATION = ROOT / "results" / "ablation_matrix_summary.json"
ATTENTION = ROOT / "results" / "attention_mechanism_breakdown.json"
OUT_CSV = ROOT / "results" / "sparse_operating_point.csv"
OUT_JSON = ROOT / "results" / "sparse_operating_point.json"
OUT_FIG = ROOT / "results" / "plots" / "sparse_operating_point"


def main() -> None:
    ablation = json.loads(ABLATION.read_text())
    attention = json.loads(ATTENTION.read_text())["rows"]
    rows = []
    for context in ablation["contexts"]:
        ctx = int(context["context_length"])
        token = context["systems"]["TokenGranular"]
        page = context["systems"]["SPLASH"]
        attn_rows = [
            row for row in attention if int(row["context_length"]) == ctx
        ]
        token_attn = next(
            float(row["attention_critical_ms"]) for row in attn_rows
            if row["system"] == "Token-granular"
        )
        page_attn = next(
            float(row["attention_critical_ms"]) for row in attn_rows
            if row["system"] == "SPLASH"
        )
        rows.append({
            "context_length": ctx,
            "token_batch": int(token["batch"]),
            "token_tpot_p50_ms": float(token["tpot_p50_ms"]),
            "token_throughput_per_gpu": float(token["throughput_per_gpu"]),
            "page_batch": int(page["batch"]),
            "page_tpot_p50_ms": float(page["tpot_p50_ms"]),
            "page_throughput_per_gpu": float(page["throughput_per_gpu"]),
            "page_over_token_throughput": (
                float(page["throughput_per_gpu"])
                / float(token["throughput_per_gpu"])
            ),
            "token_attention_ms": token_attn,
            "page_attention_ms": page_attn,
            "token_over_page_attention_speedup": token_attn / page_attn,
        })

    with OUT_CSV.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    OUT_JSON.write_text(json.dumps({
        "contract": {
            "logical_selected_token_fraction": 0.10,
            "model": "Llama-3-8B",
            "tp": 8,
            "slo_ms": 100,
            "serving_selection": (
                "each method independently selects its highest-throughput "
                "batch satisfying TPOT-p50 <= 100 ms"
            ),
            "token": (
                "exact per-head q·k top-k; full cold-K score scan; selected "
                "V reads pay measured 3.5635x physical-page amplification"
            ),
            "page": (
                "P=1024 plane-balanced centroid top-k; 16 tokens per 4-KB "
                "K or V page; selected exact K+V returned"
            ),
            "fixed_batch_attention": "batch=64, TP=8",
            "internal_hbf_traffic_over_cold_kv": {
                "token_percent": 67.8175,
                "page_percent": 13.125,
            },
        },
        "rows": rows,
    }, indent=2) + "\n")

    ps.apply()
    token_c = ps.COLORS["token"]
    page_c = ps.COLORS["splash"]
    x = np.arange(len(rows))
    labels = [
        f"{row['context_length'] // 1024}K"
        if row["context_length"] < 1_048_576
        else f"{row['context_length'] // 1_048_576}M"
        for row in rows
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))

    width = 0.36
    axes[0].bar(
        x - width / 2,
        [row["token_throughput_per_gpu"] for row in rows],
        width, color=token_c, label="Exact token top-k",
    )
    axes[0].bar(
        x + width / 2,
        [row["page_throughput_per_gpu"] for row in rows],
        width, color=page_c, label="4-KB page top-k",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("Context length")
    axes[0].set_ylabel("SLO throughput (token/s/GPU)")
    axes[0].set_title("(a) Same 10% logical token budget")
    ps.finish_axis(axes[0], grid_axis="y")
    axes[0].legend(frameon=False)
    for i, row in enumerate(rows):
        axes[0].text(
            i + width / 2,
            row["page_throughput_per_gpu"] + 9,
            f"{row['page_over_token_throughput']:.2f}×",
            ha="center", va="bottom", fontsize=ps.ANNOTATION_SIZE,
            color=page_c,
        )
    axes[0].set_ylim(
        0, max(row["page_throughput_per_gpu"] for row in rows) * 1.18
    )

    axes[1].plot(
        x, [row["token_tpot_p50_ms"] for row in rows], "-o",
        color=token_c, label="Exact token top-k",
    )
    axes[1].plot(
        x, [row["page_tpot_p50_ms"] for row in rows], "-o",
        color=page_c, label="4-KB page top-k",
    )
    axes[1].axhline(100, color="#555", ls=":", lw=1, label="SLO")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("Context length")
    axes[1].set_ylabel("TPOT p50 (ms)")
    axes[1].set_title("(b) Independently selected batch")
    ps.finish_axis(axes[1])
    for i, row in enumerate(rows):
        if row["token_batch"] == row["page_batch"]:
            continue
        axes[1].annotate(
            f"B{row['token_batch']}/{row['page_batch']}",
            (i, max(row["token_tpot_p50_ms"], row["page_tpot_p50_ms"])),
            xytext=(0, 5), textcoords="offset points",
            ha="center", fontsize=ps.ANNOTATION_SIZE,
        )

    axes[2].plot(
        x, [row["token_attention_ms"] for row in rows], "-o",
        color=token_c, label="Exact token top-k",
    )
    axes[2].plot(
        x, [row["page_attention_ms"] for row in rows], "-o",
        color=page_c, label="4-KB page top-k",
    )
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels)
    axes[2].set_xlabel("Context length")
    axes[2].set_ylabel("Attention path (ms/token)")
    axes[2].set_title("(c) Fixed batch=64")
    ps.finish_axis(axes[2])
    axes[2].text(
        0.04, 0.94,
        "Internal HBF traffic:\n67.8% token vs 13.1% page",
        transform=axes[2].transAxes, va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )

    fig.tight_layout(w_pad=1.3)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_FIG}.{ext}", bbox_inches="tight", dpi=240)
    plt.close(fig)

    print(OUT_FIG)
    for row in rows:
        print(
            f"{row['context_length']:7d}: "
            f"token={row['token_throughput_per_gpu']:.2f}, "
            f"page={row['page_throughput_per_gpu']:.2f} "
            f"({row['page_over_token_throughput']:.2f}x)"
        )


if __name__ == "__main__":
    main()
