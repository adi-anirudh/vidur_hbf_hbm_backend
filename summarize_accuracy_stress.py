#!/usr/bin/env python3
"""Summarize the P=1024 accuracy stress suite by retrieval behavior."""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean


ACC = Path("/home/adityaan/splash_accuracy_tests")
OUT_JSON = Path("results/accuracy_stress_summary.json")
OUT_TEX = Path("results/accuracy_stress_table.tex")

LONG_FILE = ACC / "longbench_stress_p1024.jsonl"
RULER_FILE = ACC / "ruler_stress_p1024.jsonl"

LONG_GROUPS = (
    ("Long-document QA", ("narrativeqa", "qasper"), "F1"),
    ("Multi-hop QA", ("hotpotqa",), "F1"),
    ("Long summarization", ("gov_report",), "ROUGE-L"),
    ("Exact passage retrieval", ("passage_retrieval_en",), "Accuracy"),
    ("Counting / aggregation", ("passage_count",), "Accuracy"),
    ("Code completion", ("lcc",), "Similarity"),
)
RULER_GROUPS = (
    ("Distributed multi-fact", ("niah_multivalue", "niah_multiquery"), "Recall"),
    ("Variable tracing", ("vt",), "Recall"),
    ("Frequency aggregation", ("cwe",), "Recall"),
)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open() if line.strip()]


def aggregate(rows: list[dict], tasks: tuple[str, ...], value: str) -> dict:
    selected = [r for r in rows if r["task"] in tasks]
    by_mode = {}
    for mode in ("dense", "global", "plane"):
        vals = [float(r[value]) for r in selected if r["mode"] == mode]
        if vals:
            by_mode[mode] = mean(vals)
    return by_mode


def main() -> None:
    long_rows = read_jsonl(LONG_FILE)
    ruler_rows = read_jsonl(RULER_FILE)
    groups = []
    for label, tasks, metric in LONG_GROUPS:
        scores = aggregate(long_rows, tasks, "score")
        if len(scores) == 3:
            groups.append({
                "stress": label,
                "tasks": list(tasks),
                "metric": metric,
                "dense": scores["dense"],
                "global": scores["global"],
                "splash": scores["plane"],
                "splash_minus_global": scores["plane"] - scores["global"],
                "splash_retention_percent": (
                    100 * scores["plane"] / scores["dense"]
                    if scores["dense"] else 0
                ),
            })
    for label, tasks, metric in RULER_GROUPS:
        scores = aggregate(ruler_rows, tasks, "recall")
        if len(scores) == 3:
            groups.append({
                "stress": label,
                "tasks": list(tasks),
                "metric": metric,
                "dense": 100 * scores["dense"],
                "global": 100 * scores["global"],
                "splash": 100 * scores["plane"],
                "splash_minus_global": (
                    100 * (scores["plane"] - scores["global"])
                ),
                "splash_retention_percent": (
                    100 * scores["plane"] / scores["dense"]
                    if scores["dense"] else 0
                ),
            })

    payload = {
        "setup": {
            "model": "Llama-3.1-8B-Instruct",
            "planes": 1024,
            "page_tokens": 16,
            "physical_page_kb_per_k_or_v_stream": 4,
            "score_mode": "centroid",
            "selection_fraction": 0.10,
            "longbench_examples_per_task": 20,
            "ruler_samples_per_task": 10,
            "comparison": "dense vs global top-k vs per-plane top-k (SPLASH)",
        },
        "groups": groups,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        r"\begin{tabular}{@{}llrrr@{}}",
        r"\toprule",
        r"\textbf{Retrieval stress} & \textbf{Metric} & "
        r"\textbf{Full} & \textbf{Global} & \textbf{\sys} \\",
        r"\midrule",
    ]
    for group in groups:
        label = group["stress"].replace("/", r"/")
        lines.append(
            f"{label} & {group['metric']} & "
            f"{group['dense']:.1f} & {group['global']:.1f} & "
            f"{group['splash']:.1f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    OUT_TEX.write_text("\n".join(lines) + "\n")
    print(OUT_JSON)
    print(OUT_TEX)
    for group in groups:
        print(
            f"{group['stress']:25s} Full={group['dense']:.2f} "
            f"Global={group['global']:.2f} SPLASH={group['splash']:.2f}"
        )


if __name__ == "__main__":
    main()
