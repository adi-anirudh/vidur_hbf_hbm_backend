#!/usr/bin/env python3
"""Comprehensive end-to-end summary of the exhaustive Blackwell sweep."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation_common import throughput_per_gpu
import publication_style as ps


BASELINES = ("HBM-only", "Dense", "Naive", "Sparse")
LABEL = {
    "HBM-only": "HBM-only",
    "Dense": "H3",
    "Naive": "Token sparse",
    "Sparse": "SPLASH",
}
COLOR = {
    "HBM-only": ps.COLORS["hbm"],
    "Dense": ps.COLORS["dense"],
    "Naive": ps.COLORS["token"],
    "Sparse": ps.COLORS["splash"],
}
CONTEXT_LABEL = {
    131072: "128K", 196608: "192K", 262144: "256K", 393216: "384K",
    524288: "512K", 786432: "768K", 1048576: "1M", 2097152: "2M",
}


def short(model: str) -> str:
    name = model.split("/")[-1]
    replacements = (
        ("Meta-Llama-", "Llama-"),
        ("-Instruct", ""),
        ("-v0.1", ""),
        ("-chat", ""),
        ("deepseek-llm-", "DeepSeek-"),
        ("Qwen3-235B-A22B", "Qwen3-235B"),
        ("phi-2", "Phi-2"),
    )
    for old, new in replacements:
        name = name.replace(old, new)
    return name


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(v) for v in values) / len(values))


def select(rows: list[dict], slo: float) -> dict[tuple, dict]:
    out = {}
    for r in rows:
        tpot = float(r["tpot_p50_ms"])
        if tpot > slo:
            continue
        key = (r["model"], int(r["context_length"]), r["baseline"])
        throughput = throughput_per_gpu(
            int(r["batch"]), int(r["tp"]), tpot
        )
        if key not in out or throughput > out[key]["throughput_per_gpu"]:
            out[key] = {
                **r,
                "throughput_per_gpu": throughput,
                "tpot_p50_ms": tpot,
                "tpot_p99_ms": float(r["tpot_p99_ms"]),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv", type=Path,
        default=Path("results/sweep_b200_weight_valid.csv"),
    )
    parser.add_argument(
        "--out", type=Path,
        default=Path("results/plots/main_evaluation"),
    )
    parser.add_argument(
        "--summary", type=Path,
        default=Path("results/main_evaluation_summary.json"),
    )
    args = parser.parse_args()
    rows = [
        r for r in csv.DictReader(args.csv.open())
        if r["status"] == "OK"
    ]
    models = sorted({r["model"] for r in rows}, key=short)
    contexts = sorted({int(r["context_length"]) for r in rows})
    total_workloads = len(models) * len(contexts)
    chosen = {slo: select(rows, slo) for slo in (50.0, 100.0)}

    summary = {
        "source": str(args.csv),
        "models": models,
        "contexts": contexts,
        "total_model_context_workloads": total_workloads,
        "slo": {},
    }
    for slo, selected in chosen.items():
        item = {"feasible_workloads": {}, "splash_vs_dense": {}}
        for baseline in BASELINES:
            item["feasible_workloads"][baseline] = sum(
                1 for key in selected if key[2] == baseline
            )
        ratios = []
        splash_only = 0
        by_context = {}
        for context in contexts:
            cratios = []
            concurrency_ratios = []
            tpot_ratios = []
            conly = 0
            for model in models:
                dense = selected.get((model, context, "Dense"))
                splash = selected.get((model, context, "Sparse"))
                if dense and splash:
                    ratio = (
                        splash["throughput_per_gpu"]
                        / dense["throughput_per_gpu"]
                    )
                    ratios.append(ratio)
                    cratios.append(ratio)
                    concurrency_ratios.append(
                        (int(splash["batch"]) / int(splash["tp"]))
                        / (int(dense["batch"]) / int(dense["tp"]))
                    )
                    tpot_ratios.append(
                        dense["tpot_p50_ms"] / splash["tpot_p50_ms"]
                    )
                elif splash:
                    splash_only += 1
                    conly += 1
            by_context[str(context)] = {
                "paired": len(cratios),
                "geomean_speedup": geomean(cratios) if cratios else None,
                "geomean_concurrency_per_gpu_ratio": (
                    geomean(concurrency_ratios)
                    if concurrency_ratios else None
                ),
                "geomean_tpot_ratio": (
                    geomean(tpot_ratios) if tpot_ratios else None
                ),
                "splash_only": conly,
            }
        item["splash_vs_dense"] = {
            "paired": len(ratios),
            "geomean_speedup": geomean(ratios) if ratios else None,
            "median_speedup": median(ratios) if ratios else None,
            "maximum_speedup": max(ratios) if ratios else None,
            "splash_only_workloads": splash_only,
            "by_context": by_context,
        }
        summary["slo"][str(int(slo))] = item
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")

    ps.apply()
    plt.rcParams.update({
        "axes.titlesize": 7.0, "axes.labelsize": 6.4,
        "xtick.labelsize": 5.7, "ytick.labelsize": 5.7, "legend.fontsize": 5.5,
    })
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(3.4, 6.7))
    gs = gridspec.GridSpec(
        4, 1, height_ratios=[2.75, 1.05, 1.05, 1.05], hspace=0.66,
        left=0.205, right=0.895, top=0.97, bottom=0.055,
    )
    axA = fig.add_subplot(gs[0]); axB = fig.add_subplot(gs[1])
    axC = fig.add_subplot(gs[2]); axD = fig.add_subplot(gs[3])

    # (a) Complete model x context throughput speedup matrix at the 100-ms SLO.
    selected = chosen[100.0]
    matrix = np.full((len(models), len(contexts)), np.nan)
    labels = [["" for _ in contexts] for _ in models]
    finite = []
    for i, model in enumerate(models):
        for j, context in enumerate(contexts):
            d = selected.get((model, context, "Dense"))
            s = selected.get((model, context, "Sparse"))
            if d and s:
                value = s["throughput_per_gpu"] / d["throughput_per_gpu"]
                matrix[i, j] = value
                finite.append(value)
                labels[i][j] = f"{value:.1f}"
            elif s:
                labels[i][j] = "S"
            else:
                labels[i][j] = "--"
    vmax = max(2.0, min(9.0, max(finite)))
    im = axA.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=1.0, vmax=vmax)
    axA.set_xticks(range(len(contexts)))
    axA.set_xticklabels([CONTEXT_LABEL[c] for c in contexts], rotation=45)
    axA.set_yticks(range(len(models)))
    axA.set_yticklabels([short(m) for m in models])
    for i in range(len(models)):
        for j in range(len(contexts)):
            axA.text(
                j, i, labels[i][j], ha="center", va="center", fontsize=4.6,
                color="white" if np.isfinite(matrix[i, j])
                and matrix[i, j] > (1 + vmax) / 2 else "black",
            )
    axA.set_title("(a) Throughput/GPU speedup, SPLASH / H3 (100-ms SLO)")
    cb = fig.colorbar(im, ax=axA, fraction=0.045, pad=0.02)
    cb.set_label("SPLASH / H3", fontsize=6.0)
    cb.ax.tick_params(labelsize=5.2)

    # (b) Speedup increases with context at both service targets.
    for slo, marker in ((50.0, "s"), (100.0, "o")):
        items = summary["slo"][str(int(slo))]["splash_vs_dense"]["by_context"]
        y = [items[str(c)]["geomean_speedup"] for c in contexts]
        axB.plot(range(len(contexts)), y, marker=marker, lw=1.4, ms=3.0,
                 label=f"{int(slo)} ms")
    axB.set_xticks(range(len(contexts)))
    axB.set_xticklabels([CONTEXT_LABEL[c] for c in contexts], rotation=45)
    axB.set_ylabel("Geomean\nspeedup")
    axB.set_title("(b) Speedup scales with context")
    axB.legend(frameon=False, ncol=2, handlelength=1.2, columnspacing=0.9,
               loc="upper left")
    ps.finish_axis(axB)

    # (c) Feasible model-context coverage.
    x = np.arange(len(BASELINES))
    width = 0.38
    for offset, slo in ((-width / 2, 50.0), (width / 2, 100.0)):
        vals = [summary["slo"][str(int(slo))]["feasible_workloads"][b]
                for b in BASELINES]
        axC.bar(x + offset, vals, width, color=[COLOR[b] for b in BASELINES],
                alpha=0.72 if slo == 50 else 1.0, edgecolor="black",
                linewidth=0.35)
    axC.axhline(total_workloads, color="#777", ls=":", lw=1)
    axC.set_xticks(x)
    axC.set_xticklabels([LABEL[b] for b in BASELINES], rotation=12)
    axC.set_ylabel("Feasible\nworkloads")
    axC.set_ylim(0, total_workloads * 1.24)
    axC.set_title(f"(c) SLO-feasible workloads (of {total_workloads})")
    ps.finish_axis(axC, grid_axis="y")
    axC.legend(
        handles=[Patch(fc="#9aa0a6", alpha=0.72, ec="black", lw=0.35, label="50 ms"),
                 Patch(fc="#9aa0a6", alpha=1.0, ec="black", lw=0.35, label="100 ms")],
        frameon=False, ncol=2, handlelength=1.0, columnspacing=0.9,
        loc="upper center")

    # (d) Decompose independently optimized throughput into its two exact
    # multiplicative factors: concurrency/GPU and inverse TPOT.
    items = summary["slo"]["100"]["splash_vs_dense"]["by_context"]
    concurrency = [items[str(c)]["geomean_concurrency_per_gpu_ratio"] for c in contexts]
    latency = [items[str(c)]["geomean_tpot_ratio"] for c in contexts]
    throughput = [items[str(c)]["geomean_speedup"] for c in contexts]
    for values, label, color, marker in (
        (concurrency, "Concurrency/GPU", ps.COLORS["global"], "^"),
        (latency, "TPOT reduction", ps.COLORS["token"], "s"),
        (throughput, "Throughput/GPU", ps.COLORS["splash"], "o"),
    ):
        axD.plot(range(len(contexts)), values, marker=marker, lw=1.4, ms=3.0,
                 color=color, label=label)
    axD.axhline(1.0, color="#777", ls=":", lw=1)
    axD.set_xticks(range(len(contexts)))
    axD.set_xticklabels([CONTEXT_LABEL[c] for c in contexts], rotation=45)
    axD.set_ylabel("Factor\nvs H3")
    axD.set_title("(d) Source of 100-ms throughput gain")
    axD.legend(frameon=False, ncol=1, handlelength=1.2, loc="upper left")
    ps.finish_axis(axD, grid_axis="y")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(args.out)
    print(json.dumps(summary["slo"], indent=2))


if __name__ == "__main__":
    main()
