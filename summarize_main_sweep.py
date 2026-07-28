#!/usr/bin/env python3
"""Filter and summarize the exhaustive Blackwell sweep for paper use.

This is the sole post-processing gate between results/sweep_b200.csv and the
paper-facing plots. It removes duplicate rows and rejects operating points that
violate the frozen eight-GPU contract, including the critical rule that model
weights remain in HBM. It emits both the exact filtered CSV and a JSON audit.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from evaluation_common import (
    PLATFORM,
    capacity_feasible,
    model_geometry,
    throughput_per_gpu,
    weights_fit_hbm,
)

SOURCE = Path("results/sweep_b200.csv")
FILTERED = Path("results/sweep_b200_weight_valid.csv")
SUMMARY = Path("results/main_sweep_summary.json")
SLOS = (50.0, 100.0)


def geomean(values: list[float]) -> float | None:
    values = [v for v in values if v > 0]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def row_key(row: dict) -> tuple:
    return (
        row["model"],
        row["device"],
        int(row["batch"]),
        int(row["context_length"]),
        row["baseline"],
        int(row["tp"]),
    )


def point_key(row: dict) -> tuple:
    return (
        row["model"],
        int(row["context_length"]),
        int(row["batch"]),
        int(row["tp"]),
    )


def workload_key(row: dict) -> tuple:
    return row["model"], int(row["context_length"])


def invalid_reason(row: dict) -> str | None:
    try:
        batch = int(row["batch"])
        context = int(row["context_length"])
        tp = int(row["tp"])
    except (KeyError, TypeError, ValueError):
        return "malformed"
    if row.get("device") != PLATFORM.device:
        return "wrong_device"
    if tp > PLATFORM.gpus_per_node:
        return "tp_exceeds_8gpu_node"
    if tp > model_geometry(row["model"])[3]:
        return "tp_exceeds_model_kv_head_cap"
    if not weights_fit_hbm(row["model"], tp):
        return "weights_do_not_fit_hbm"
    tier = "hbm" if row["baseline"] == "HBM-only" else "hbf"
    if not capacity_feasible(row["model"], batch, context, tp, tier):
        return "kv_capacity"
    return None


def best_by_workload(rows: list[dict], baseline: str, slo: float) -> dict:
    result = {}
    for row in rows:
        if row["baseline"] != baseline:
            continue
        tpot = float(row["tpot_p50_ms"])
        if tpot > slo:
            continue
        key = workload_key(row)
        throughput = throughput_per_gpu(
            int(row["batch"]), int(row["tp"]), tpot
        )
        if key not in result or throughput > result[key]["throughput_per_gpu"]:
            result[key] = {
                "batch": int(row["batch"]),
                "tp": int(row["tp"]),
                "tpot_p50_ms": tpot,
                "tpot_p99_ms": float(row["tpot_p99_ms"]),
                "throughput_per_gpu": throughput,
            }
    return result


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--filtered", type=Path, default=FILTERED)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    args = parser.parse_args()
    source_path = args.source
    filtered_path = args.filtered
    summary_path = args.summary

    raw = list(csv.DictReader(source_path.open()))
    reasons = Counter()
    dedup = {}
    duplicate_rows = 0
    ok_seen = 0
    for row in raw:
        if row.get("status") != "OK":
            reasons["non_ok"] += 1
            continue
        ok_seen += 1
        reason = invalid_reason(row)
        if reason:
            reasons[reason] += 1
            continue
        key = row_key(row)
        if key in dedup:
            duplicate_rows += 1
        dedup[key] = row

    rows = list(dedup.values())
    rows.sort(key=row_key)
    with filtered_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=raw[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "source": str(source_path),
        "filtered": str(filtered_path),
        "contract": {
            "device": PLATFORM.device,
            "gpus_per_node": PLATFORM.gpus_per_node,
            "weights": "HBM only",
            "activations_hbm_reserve_gb_per_gpu": (
                PLATFORM.activation_reserve_gb_per_gpu
            ),
            "kv": "remaining HBM then HBF",
            "throughput_per_gpu": "batch * 1000 / (TP * TPOT_p50_ms)",
        },
        "row_audit": {
            "raw_rows": len(raw),
            "raw_ok_rows": ok_seen,
            "filtered_unique_ok_rows": len(rows),
            "duplicate_valid_rows_removed": duplicate_rows,
            "excluded": dict(sorted(reasons.items())),
        },
        "coverage": {
            "models": sorted({r["model"] for r in rows}),
            "num_models": len({r["model"] for r in rows}),
            "contexts": sorted({int(r["context_length"]) for r in rows}),
            "batches": sorted({int(r["batch"]) for r in rows}),
            "tensor_parallel_degrees": sorted({int(r["tp"]) for r in rows}),
            "baselines": sorted({r["baseline"] for r in rows}),
        },
        "slo": {},
    }

    by_baseline_point = {
        baseline: {point_key(r): r for r in rows if r["baseline"] == baseline}
        for baseline in ("Dense", "Sparse", "Naive", "HBM-only")
    }

    for slo in SLOS:
        dense = best_by_workload(rows, "Dense", slo)
        splash = best_by_workload(rows, "Sparse", slo)
        naive = best_by_workload(rows, "Naive", slo)
        hbm = best_by_workload(rows, "HBM-only", slo)
        paired = sorted(dense.keys() & splash.keys())
        gains = [
            splash[key]["throughput_per_gpu"] / dense[key]["throughput_per_gpu"]
            for key in paired
        ]

        matched_ratios = []
        splash_op_ratios = []
        splash_only_points = 0
        both_points = 0
        dense_only_points = 0
        neither_points = 0
        all_points = (
            set(by_baseline_point["Dense"]) | set(by_baseline_point["Sparse"])
        )
        for key in all_points:
            dr = by_baseline_point["Dense"].get(key)
            sr = by_baseline_point["Sparse"].get(key)
            d_meets = bool(dr and float(dr["tpot_p50_ms"]) <= slo)
            s_meets = bool(sr and float(sr["tpot_p50_ms"]) <= slo)
            if d_meets and s_meets:
                both_points += 1
            elif s_meets:
                splash_only_points += 1
            elif d_meets:
                dense_only_points += 1
            else:
                neither_points += 1
            if dr and sr:
                matched_ratios.append(
                    float(dr["tpot_p50_ms"]) / float(sr["tpot_p50_ms"])
                )

        for key, sop in splash.items():
            dr = by_baseline_point["Dense"].get((
                key[0], key[1], sop["batch"], sop["tp"]
            ))
            if dr:
                splash_op_ratios.append(
                    float(dr["tpot_p50_ms"]) / sop["tpot_p50_ms"]
                )

        payload["slo"][str(int(slo))] = {
            "workloads_splash_feasible": len(splash),
            "workloads_dense_feasible": len(dense),
            "workloads_naive_feasible": len(naive),
            "workloads_hbm_only_feasible": len(hbm),
            "workloads_paired_dense_splash": len(paired),
            "optimized_throughput_speedup_splash_vs_dense": {
                "geomean": geomean(gains),
                "median": statistics.median(gains) if gains else None,
                "p95": percentile(gains, 0.95),
                "maximum": max(gains) if gains else None,
                "minimum": min(gains) if gains else None,
            },
            "same_configuration_slo_counts": {
                "both": both_points,
                "splash_only": splash_only_points,
                "dense_only": dense_only_points,
                "neither": neither_points,
            },
            "matched_configuration_tpot_speedup_all_rows": {
                "count": len(matched_ratios),
                "geomean": geomean(matched_ratios),
                "median": (
                    statistics.median(matched_ratios)
                    if matched_ratios else None
                ),
                "maximum": max(matched_ratios) if matched_ratios else None,
                "minimum": min(matched_ratios) if matched_ratios else None,
            },
            "tpot_speedup_at_splash_operating_point": {
                "count": len(splash_op_ratios),
                "geomean": geomean(splash_op_ratios),
                "median": (
                    statistics.median(splash_op_ratios)
                    if splash_op_ratios else None
                ),
                "maximum": max(splash_op_ratios) if splash_op_ratios else None,
                "minimum": min(splash_op_ratios) if splash_op_ratios else None,
            },
        }

    summary_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
