#!/usr/bin/env python3
"""Byte-level traffic accounting for the mechanism ablation.

The output separates GPU-visible HBM traffic, internal HBF media reads, bytes
crossing the HBF--GPU link, and busiest-plane amplification. Values correspond
to each system's SLO-selected operating point in ablation_matrix_summary.json.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from evaluation_common import PLATFORM, hbm_kv_fraction
from sweep_capacity import kv_bytes


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results" / "ablation_matrix_summary.json"
OUT_CSV = ROOT / "results" / "mechanism_traffic.csv"
OUT_JSON = ROOT / "results" / "mechanism_traffic.json"
MODEL = "meta-llama/Meta-Llama-3-8B"
TP = 8
SPARSITY = 0.10
TOKEN_AMP = 3.5635
TOKEN_PLANE_IMBALANCE = 1.3405
PLANE_IMBALANCE = 2.175


def traffic(system: str, cold_kv: float) -> dict[str, float]:
    cold_k = cold_kv / 2.0
    cold_v = cold_kv / 2.0
    centroid_scan = cold_k / PLATFORM.tokens_per_selection_page
    if system == "Dense":
        return {
            "hbf_metadata_scan_bytes": 0.0,
            "hbf_selected_data_read_bytes": cold_kv,
            "hbf_internal_read_bytes": cold_kv,
            "hbf_to_gpu_link_bytes": cold_kv,
            "busiest_plane_load_over_balanced": 1.0,
        }
    if system == "TokenGranular":
        selected_v = min(cold_v, cold_v * SPARSITY * TOKEN_AMP)
        return {
            "hbf_metadata_scan_bytes": cold_k,
            "hbf_selected_data_read_bytes": selected_v,
            "hbf_internal_read_bytes": cold_k + selected_v,
            "hbf_to_gpu_link_bytes": selected_v,
            "busiest_plane_load_over_balanced": TOKEN_PLANE_IMBALANCE,
        }
    if system == "FullPageScore":
        selected_v = cold_v * SPARSITY
        return {
            "hbf_metadata_scan_bytes": cold_k,
            "hbf_selected_data_read_bytes": selected_v,
            "hbf_internal_read_bytes": cold_k + selected_v,
            "hbf_to_gpu_link_bytes": selected_v,
            "busiest_plane_load_over_balanced": 1.0,
        }

    selected_kv = cold_kv * SPARSITY
    return {
        "hbf_metadata_scan_bytes": centroid_scan,
        "hbf_selected_data_read_bytes": selected_kv,
        "hbf_internal_read_bytes": centroid_scan + selected_kv,
        "hbf_to_gpu_link_bytes": selected_kv,
        "busiest_plane_load_over_balanced": (
            PLANE_IMBALANCE if system == "PlaneImbalance" else 1.0
        ),
    }


def main() -> None:
    source = json.loads(SOURCE.read_text())
    rows = []
    for context in source["contexts"]:
        ctx = int(context["context_length"])
        for system, point in context["systems"].items():
            batch = int(point["batch"])
            hot_fraction = hbm_kv_fraction(MODEL, batch, ctx, TP)
            logical_kv = float(kv_bytes(MODEL, batch, ctx)) / TP
            hbm_kv = logical_kv * hot_fraction
            cold_kv = logical_kv - hbm_kv
            detail = traffic(system, cold_kv)
            rows.append({
                "model": MODEL,
                "context_length": ctx,
                "batch": batch,
                "tp": TP,
                "system": system,
                "logical_kv_bytes_per_decode_step_per_gpu": logical_kv,
                "hbm_kv_read_bytes": hbm_kv,
                "cold_hbf_kv_bytes": cold_kv,
                **detail,
                "hbf_internal_read_over_cold_kv": (
                    detail["hbf_internal_read_bytes"] / cold_kv
                    if cold_kv else 0.0
                ),
                "hbf_link_over_cold_kv": (
                    detail["hbf_to_gpu_link_bytes"] / cold_kv
                    if cold_kv else 0.0
                ),
            })

    with OUT_CSV.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    OUT_JSON.write_text(json.dumps({
        "schema_version": 1,
        "semantics": {
            "unit": "bytes per decode step per GPU, across all layers",
            "metadata": "centroid scan reads K/16; full-page scoring reads full K",
            "token_granular": (
                "full-K scoring plus 3.5635x amplification on selected V pages; "
                "the independently measured busiest-plane/mean ratio is 1.3405x"
            ),
            "plane_imbalance": (
                "does not change aggregate selected bytes; 2.175x is the "
                "measured busiest-plane critical-path load"
            ),
        },
        "rows": rows,
    }, indent=2) + "\n")
    print(OUT_CSV)
    print(OUT_JSON)


if __name__ == "__main__":
    main()
