#!/usr/bin/env python3
"""Matched-token-budget quality and performance: token vs page sparsity."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import publication_style as ps

ROOT = Path(__file__).resolve().parent
ACC = Path("/home/adityaan/splash_accuracy_tests")
OUT = ROOT / "results" / "plots" / "granularity_comparison"
SUMMARY = ROOT / "results" / "granularity_comparison.json"
OUT_TEX = ROOT / "results" / "granularity_comparison_table.tex"
OUT_MD = ROOT / "results" / "granularity_comparison.md"

LONG_PAGE = ACC / "longbench_stress_p1024.jsonl"
LONG_TOKEN = ACC / "longbench_token_p1024.jsonl"
RULER = ACC / "ruler_granularity_p1024.jsonl"
PPL = ACC / "granularity_ppl_p1024.jsonl"
PASSKEY = ACC / "granularity_passkey_p1024.jsonl"

LONG_GROUPS = (
    ("Narrative QA", ("narrativeqa",)),
    ("Scientific QA", ("qasper",)),
    ("Multi-hop QA", ("hotpotqa",)),
    ("Summarization", ("gov_report",)),
    ("Exact retrieval", ("passage_retrieval_en",)),
    ("Counting", ("passage_count",)),
    ("Code", ("lcc",)),
)
RULER_GROUPS = (
    ("Multi-value", ("niah_multivalue",)),
    ("Multi-query", ("niah_multiquery",)),
    ("Var. tracing", ("vt",)),
    ("Aggregation", ("cwe",)),
)


def read(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return [json.loads(line) for line in path.open() if line.strip()]


def mode_mean(rows: list[dict], tasks: tuple[str, ...], mode: str, key: str) -> float:
    values = [
        float(r[key]) for r in rows
        if r.get("task") in tasks and r.get("mode") == mode
    ]
    if not values:
        raise ValueError(f"no {mode} {tasks} in {key}")
    return mean(values)


def quality_rows() -> list[dict]:
    long_page = read(LONG_PAGE)
    long_token = read(LONG_TOKEN)
    ruler = read(RULER)
    ppl = read(PPL)
    passkey = read(PASSKEY)
    out = []

    for label, tasks in LONG_GROUPS:
        dense = mode_mean(long_page, tasks, "dense", "score")
        token = mode_mean(long_token, tasks, "token", "score")
        page = mode_mean(long_page, tasks, "plane", "score")
        out.append({
            "category": label,
            "suite": "LongBench",
            "metric": "score $\\uparrow$",
            "dense": dense,
            "token": token,
            "page": page,
            "token_retention_percent": 100 * token / dense if dense else 0,
            "page_retention_percent": 100 * page / dense if dense else 0,
        })

    for label, tasks in RULER_GROUPS:
        dense = mode_mean(ruler, tasks, "dense", "recall")
        token = mode_mean(ruler, tasks, "token", "recall")
        page = mode_mean(ruler, tasks, "plane", "recall")
        out.append({
            "category": label,
            "suite": "RULER",
            "metric": "recall (\\%) $\\uparrow$",
            "dense": 100 * dense,
            "token": 100 * token,
            "page": 100 * page,
            "token_retention_percent": 100 * token / dense if dense else 0,
            "page_retention_percent": 100 * page / dense if dense else 0,
        })

    for context, label in ((32768, "PG-19 32K"), (131072, "PG-19 128K")):
        rows = [r for r in ppl if int(r["ctx"]) == context]
        dense = next(float(r["ppl"]) for r in rows if r["mode"] == "dense")
        token = next(float(r["ppl"]) for r in rows if r["mode"] == "token")
        page = next(float(r["ppl"]) for r in rows if r["mode"] == "plane")
        out.append({
            "category": label,
            "suite": "PG-19",
            "metric": "PPL $\\downarrow$",
            "dense": dense,
            "token": token,
            "page": page,
            # Perplexity is lower-better; invert for a comparable retention axis.
            "token_retention_percent": 100 * dense / token,
            "page_retention_percent": 100 * dense / page,
        })

    dense = next(float(r["acc"]) for r in passkey if r["mode"] == "dense")
    token = next(float(r["acc"]) for r in passkey if r["mode"] == "token")
    page = next(float(r["acc"]) for r in passkey if r["mode"] == "plane")
    out.append({
        "category": "Passkey 128K",
        "suite": "Passkey",
        "metric": "accuracy (\\%) $\\uparrow$",
        "dense": 100 * dense,
        "token": 100 * token,
        "page": 100 * page,
        "token_retention_percent": 100 * token / dense if dense else 0,
        "page_retention_percent": 100 * page / dense if dense else 0,
    })
    return out


def main() -> None:
    quality = quality_rows()
    ablation = json.loads(
        (ROOT / "results" / "ablation_matrix_summary.json").read_text()
    )
    attention = json.loads(
        (ROOT / "results" / "attention_mechanism_breakdown.json").read_text()
    )["rows"]

    perf = []
    for context in ablation["contexts"]:
        systems = context["systems"]
        perf.append({
            "context_length": int(context["context_length"]),
            "token": systems["TokenGranular"]["normalized_to_dense"],
            "page": systems["SPLASH"]["normalized_to_dense"],
            "page_over_token": (
                systems["SPLASH"]["throughput_per_gpu"]
                / systems["TokenGranular"]["throughput_per_gpu"]
            ),
            "token_throughput_per_gpu": (
                systems["TokenGranular"]["throughput_per_gpu"]
            ),
            "page_throughput_per_gpu": systems["SPLASH"]["throughput_per_gpu"],
        })
    attn = []
    for context in sorted({int(r["context_length"]) for r in attention}):
        rows = [r for r in attention if int(r["context_length"]) == context]
        attn.append({
            "context_length": context,
            "token_ms": next(
                float(r["attention_critical_ms"]) for r in rows
                if r["system"] == "Token-granular"
            ),
            "page_ms": next(
                float(r["attention_critical_ms"]) for r in rows
                if r["system"] == "SPLASH"
            ),
        })
        attn[-1]["token_over_page"] = (
            attn[-1]["token_ms"] / attn[-1]["page_ms"]
        )

    for row in quality:
        row["page_minus_token_retention_pp"] = (
            row["page_retention_percent"] - row["token_retention_percent"]
        )
    suites = ["LongBench", "RULER", "PG-19", "Passkey"]
    suite_quality = []
    for suite in suites:
        suite_rows = [row for row in quality if row["suite"] == suite]
        token_retention = mean(
            row["token_retention_percent"] for row in suite_rows
        )
        page_retention = mean(
            row["page_retention_percent"] for row in suite_rows
        )
        suite_quality.append({
            "suite": suite,
            "conditions": len(suite_rows),
            "token_mean_retention_percent": token_retention,
            "page_mean_retention_percent": page_retention,
            "page_minus_token_retention_pp": (
                page_retention - token_retention
            ),
        })
    SUMMARY.write_text(json.dumps({
        "setup": {
            "logical_selected_token_fraction": 0.10,
            "token_method": "exact q·k token top-k",
            "page_method": (
                "centroid per-plane page top-k, P=1024, 16 tokens/page"
            ),
            "performance": (
                "each method reselects batch under 100-ms SLO; Llama-3-8B TP=8"
            ),
            "compulsory_region": (
                "identical sink tokens and current page in both methods"
            ),
            "token_mode_validation": {
                "context_length": 8192,
                "budget_fraction": 1.0,
                "dense_ppl": 10.0216,
                "token_ppl": 10.0238,
                "relative_difference_percent": (
                    100 * (10.0238 / 10.0216 - 1)
                ),
            },
            "samples": {
                "LongBench": "20 examples per task, max prompt 31.5K",
                "RULER": "10 generated instances per task at 16K",
                "PG-19": (
                    "up to 12 non-overlapping windows per context, last 2048 "
                    "tokens scored"
                ),
                "Passkey": (
                    "4 random samples at each of 5 insertion depths at 128K"
                ),
            },
        },
        "quality_summary": {
            "categories": len(quality),
            "page_no_worse_categories": sum(
                r["page_minus_token_retention_pp"] >= 0 for r in quality
            ),
            "page_within_1_5pp_or_better_categories": sum(
                r["page_minus_token_retention_pp"] >= -1.5 for r in quality
            ),
            "minimum_page_minus_token_retention_pp": min(
                r["page_minus_token_retention_pp"] for r in quality
            ),
            "maximum_absolute_page_token_retention_gap_pp": max(
                abs(r["page_minus_token_retention_pp"]) for r in quality
            ),
            "mean_page_minus_token_retention_pp": mean(
                r["page_minus_token_retention_pp"] for r in quality
            ),
        },
        "suite_quality": suite_quality,
        "quality": quality,
        "performance": perf,
        "attention": attn,
    }, indent=2) + "\n")
    tex = [
        r"\begin{tabular}{@{}lllrrrr@{}}",
        r"\toprule",
        r"\textbf{Suite} & \textbf{Retrieval stress} & \textbf{Metric} & "
        r"\textbf{Full} & \textbf{Token} & \textbf{Page} & "
        r"\textbf{Page--Token (pp)} \\",
        r"\midrule",
    ]
    for r in quality:
        tex.append(
            f"{r['suite']} & {r['category']} & {r['metric']} & "
            f"{r['dense']:.2f} & {r['token']:.2f} & {r['page']:.2f} & "
            f"{r['page_minus_token_retention_pp']:+.2f} \\\\"
        )
    tex.extend([r"\bottomrule", r"\end{tabular}"])
    OUT_TEX.write_text("\n".join(tex) + "\n")

    md = [
        "# Matched 10% Token-vs-Page Sparsity",
        "",
        "Both selectors receive the same logical token budget and the same "
        "compulsory sink/current-page region. Token uses exact per-head "
        "q·k top-k; Page uses P=1024 plane-balanced centroid top-k over "
        "16-token (4-KB) pages.",
        "",
        "## Quality",
        "",
        "| Suite | Stress | Metric | Full | Token | Page | "
        "Page−Token retention (pp) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in quality:
        metric_md = (
            row["metric"].replace("$", "")
            .replace("\\uparrow", "↑")
            .replace("\\downarrow", "↓")
            .replace("\\%", "%")
        )
        md.append(
            f"| {row['suite']} | {row['category']} | "
            f"{metric_md} | {row['dense']:.2f} | "
            f"{row['token']:.2f} | {row['page']:.2f} | "
            f"{row['page_minus_token_retention_pp']:+.2f} |"
        )
    md.extend([
        "",
        "### Suite-level mean retention",
        "",
        "| Suite | Conditions | Token retention | Page retention | "
        "Page−Token (pp) |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in suite_quality:
        md.append(
            f"| {row['suite']} | {row['conditions']} | "
            f"{row['token_mean_retention_percent']:.2f}% | "
            f"{row['page_mean_retention_percent']:.2f}% | "
            f"{row['page_minus_token_retention_pp']:+.2f} |"
        )
    md.extend([
        "",
        "## Serving performance",
        "",
        "| Context | Token tok/s/GPU | Page tok/s/GPU | Page/Token |",
        "|---:|---:|---:|---:|",
    ])
    for row in perf:
        md.append(
            f"| {row['context_length']:,} | "
            f"{row['token_throughput_per_gpu']:.2f} | "
            f"{row['page_throughput_per_gpu']:.2f} | "
            f"{row['page_over_token']:.2f}× |"
        )
    md.extend([
        "",
        "Each method independently selects its highest-throughput batch under "
        "the 100-ms TPOT-p50 SLO; model=Llama-3-8B, TP=8.",
        "",
        "## Fixed-batch attention critical path",
        "",
        "| Context | Token (ms/token) | Page (ms/token) | Token/Page |",
        "|---:|---:|---:|---:|",
    ])
    for row in attn:
        md.append(
            f"| {row['context_length']:,} | {row['token_ms']:.3f} | "
            f"{row['page_ms']:.3f} | {row['token_over_page']:.2f}× |"
        )
    md.extend([
        "",
        "Fixed batch=64, TP=8. At the 10% point, token selection reads a "
        "full-K scoring scan plus amplified selected V pages (67.8% internal "
        "HBF traffic in total); SPLASH reads K/16 centroids plus selected "
        "K+V pages (13.1%).",
        "",
    ])
    OUT_MD.write_text("\n".join(md))

    ps.apply()
    token_c = ps.COLORS["token"]
    page_c = ps.COLORS["splash"]
    plt.rcParams.update({"axes.titlesize":7.0,"axes.labelsize":6.4,"xtick.labelsize":5.8,"ytick.labelsize":5.8,"legend.fontsize":6.2})
    fig, axes = plt.subplots(2, 2, figsize=(3.4, 3.0))

    x = np.arange(len(suite_quality))
    width = 0.38
    axes[0, 0].bar(
        x - width / 2,
        [r["token_mean_retention_percent"] for r in suite_quality],
        width, color=token_c, label="Exact token top-k",
    )
    axes[0, 0].bar(
        x + width / 2,
        [r["page_mean_retention_percent"] for r in suite_quality],
        width, color=page_c, label="4-KB page top-k",
    )
    # Show every constituent condition, not just the suite mean.
    for i, suite in enumerate(suites):
        suite_rows = [r for r in quality if r["suite"] == suite]
        axes[0, 0].scatter(
            np.full(len(suite_rows), i - width / 2),
            [r["token_retention_percent"] for r in suite_rows],
            s=10, facecolor="white", edgecolor="#333", linewidth=0.45,
            zorder=3,
        )
        axes[0, 0].scatter(
            np.full(len(suite_rows), i + width / 2),
            [r["page_retention_percent"] for r in suite_rows],
            s=10, facecolor="white", edgecolor="#333", linewidth=0.45,
            zorder=3,
        )
    axes[0, 0].axhline(100, color="#555", ls=":", lw=1)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(
        [
            f"{r['suite']}\n({r['conditions']} conditions)"
            for r in suite_quality
        ],
    )
    qvals = [
        r[k] for r in quality
        for k in ("token_retention_percent", "page_retention_percent")
    ]
    axes[0, 0].set_ylim(max(0, min(qvals) - 5), max(qvals) + 5)
    axes[0, 0].set_ylabel("Quality retained vs Full (%)")
    axes[0, 0].set_title("(a) Same 10% token budget: quality")
    axes[0, 0].legend(frameon=False, ncol=2, loc="lower left")
    ps.finish_axis(axes[0, 0], grid_axis="y")

    cx = np.arange(len(perf))
    clabel = [
        f"{r['context_length']//1024}K"
        if r["context_length"] < 1048576
        else f"{r['context_length']//1048576}M"
        for r in perf
    ]
    axes[0, 1].plot(
        cx, [r["token"] for r in perf], "-o",
        color=token_c, label="Exact token top-k",
    )
    axes[0, 1].plot(
        cx, [r["page"] for r in perf], "-o",
        color=page_c, label="4-KB page top-k",
    )
    axes[0, 1].set_xticks(cx)
    axes[0, 1].set_xticklabels(clabel)
    axes[0, 1].set_xlabel("Context length")
    axes[0, 1].set_ylabel("Throughput/GPU (normalized to H3)")
    axes[0, 1].set_title("(b) Same 10% token budget: performance")
    axes[0, 1].legend(frameon=False)
    ps.finish_axis(axes[0, 1])

    axes[1, 0].plot(
        cx, [r["token_ms"] for r in attn], "-o",
        color=token_c, label="Exact token top-k",
    )
    axes[1, 0].plot(
        cx, [r["page_ms"] for r in attn], "-o",
        color=page_c, label="4-KB page top-k",
    )
    axes[1, 0].set_xticks(cx)
    axes[1, 0].set_xticklabels(clabel)
    axes[1, 0].set_xlabel("Context length")
    axes[1, 0].set_ylabel("Attention critical path (ms/token)")
    axes[1, 0].set_title("(c) Attention-level latency")
    ps.finish_axis(axes[1, 0])

    labels = ["Exact token", "4-KB page"]
    score = [50.0, 3.125]
    selected = [17.8175, 10.0]
    xx = np.arange(2)
    axes[1, 1].bar(
        xx, score, color=ps.COLORS["token"], label="Scoring scan")
    axes[1, 1].bar(
        xx, selected, bottom=score, color=ps.COLORS["dense"],
        label="Selected data read",
    )
    axes[1, 1].set_xticks(xx)
    axes[1, 1].set_xticklabels(labels)
    axes[1, 1].set_ylabel("Internal HBF traffic (% of cold K+V)")
    axes[1, 1].set_title("(d) Why page granularity is faster")
    axes[1, 1].legend(frameon=False)
    ps.finish_axis(axes[1, 1], grid_axis="y")

    fig.tight_layout(w_pad=1.4, h_pad=1.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight", dpi=240)
    plt.close(fig)
    print(OUT)
    for r in quality:
        print(
            f"{r['category']:18s} token={r['token_retention_percent']:.2f}% "
            f"page={r['page_retention_percent']:.2f}%"
        )


if __name__ == "__main__":
    main()
