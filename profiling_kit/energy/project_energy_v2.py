#!/usr/bin/env python3
"""SPLASH energy model v2: component-complete, measurement-anchored.

What changes relative to ``project_provisional_hybrid.py``:

1. The GPU form is KEPT and now measurement-validated, not replaced.  On 132
   Blackwell attention-decode shapes (RTX PRO 6000, static subtracted) board
   power x time predicts dynamic board energy to 0.61 % wAPE.  A bytes-linear
   form fits equally well (0.65 %), and corr(bytes, time) = 0.999998, so the
   two are not distinguishable and the GPU-internal HBM/logic split is NOT
   identifiable from board telemetry.  Consequence: HBM pJ/B is a diagnostic
   fraction of the board term and is never added on top of it.  The v1
   sensitivity swept HBM over three values and got bit-identical ratios; that
   axis was vacuous and has been replaced by axes that actually move.

2. SPLASH is now charged for its own hardware.  v1 omitted:
     * the centroid scan -- every decode step streams all HBF-resident
       centroids through the flash array.  That is KV/32 bytes, i.e. +31.25 %
       on top of a 10 % selection.
     * KV-offload program and erase energy.
     * the NMA MAC array, the SRAM protocol bridge, and the D2D link.
   Naive Sparse is likewise charged for its full-key selection scan, which the
   latency model already pays for but the energy model did not.  In v1
   ``hbf_physical == hbf_useful`` for Sparse, so no selection traffic of any
   kind was billed.

Run with ``--ablate`` to price each newly charged term separately.

Usage:
    .venv/bin/python profiling_kit/energy/project_energy_v2.py --ablate
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sweep_capacity import HBM_GB, weight_bytes  # noqa: E402

MODELS = [
    ("Qwen/Qwen3-235B-A22B", "Qwen3-235B"),
    ("mistralai/Mixtral-8x22B-v0.1", "Mixtral-8x22B"),
    ("deepseek-ai/deepseek-llm-67b-chat", "DeepSeek-67B"),
    ("meta-llama/Meta-Llama-3.1-405B", "Llama3.1-405B"),
]
BASELINES = ["Dense", "Naive", "Sparse"]

# Page geometry from Table I: 16 token positions per 4 KB page, K and V in
# separate pages, one FP16 d-vector centroid per key page.
TOKENS_PER_PAGE = 16
CENTROID_FRACTION_OF_KV = 1.0 / (2 * TOKENS_PER_PAGE)  # K is half of KV -> KV/32
PAGE_BYTES = 4096


@dataclass
class Coefficients:
    """Every coefficient carries an explicit provenance tag in PROVENANCE."""

    # --- GPU board -------------------------------------------------------
    gpu_board_w: float = 700.0           # total board power during decode
    hbm_pj_per_byte: float = 11.93       # diagnostic only, not additive
    # --- HBF media -------------------------------------------------------
    hbf_read_pj_per_byte: float = 64.0   # 8 pJ/bit, matches FlashAccel
    hbf_program_read_ratio: float = 10.0  # 100 us vs 10 us per op
    hbf_erase_uj_per_block: float = 300.0
    pages_per_block: int = 256
    # --- SPLASH on-stack logic ------------------------------------------
    sram_bridge_pj_per_byte: float = 1.0
    d2d_link_pj_per_byte: float = 1.0
    nma_pj_per_mac: float = 0.2          # FP16 MAC at 16 nm

    @property
    def hbf_program_pj_per_byte(self) -> float:
        return self.hbf_read_pj_per_byte * self.hbf_program_read_ratio

    @property
    def hbf_erase_pj_per_byte(self) -> float:
        block_bytes = self.pages_per_block * PAGE_BYTES
        return self.hbf_erase_uj_per_block * 1e6 / block_bytes


PROVENANCE = {
    "gpu_board_w": "swept 450-1000 W. MEASURED anchor: 192.4 W dynamic for "
    "attention decode on an RTX PRO 6000 Blackwell (600 W part, 32% of TDP); "
    "the same ratio on a 1000 W B200 plus static lands near 470 W, so 700 W is "
    "conservative",
    "hbm_pj_per_byte": "DIAGNOSTIC ONLY, inside the board term. Ujeniya et al. arXiv:2604.11391 H100 rail, activity-attributed "
    "(results/energy_h100/source_calibrated_separation.json); high case 23.9 pJ/B "
    "= FlashAccel's cited HBM3e 2.99 pJ/bit",
    "gpu_board_form": "MEASURED: P*t predicts dynamic board energy to 0.61% wAPE "
    "over 132 Blackwell attention-decode shapes (n=132). A bytes-linear form fits "
    "to 0.65%; corr(bytes,time)=0.999998 so they are not separable and the "
    "HBM/logic split is not identifiable from board telemetry",
    "hbf_read_pj_per_byte": "8 pJ/bit; independently matches FlashAccel Table 2",
    "hbf_program_read_ratio": "HBFSim energy_table.toml, 100 us program vs 10 us read per op",
    "hbf_erase_uj_per_block": "HBFSim energy_table.toml",
    "sram_bridge_pj_per_byte": "HBFSim interconnect_pj_per_byte, CACTI-class large SRAM",
    "d2d_link_pj_per_byte": "HBFSim host_bus_pj_per_byte, short-reach on-package D2D",
    "nma_pj_per_mac": "FP16 MAC at 16 nm",
}


def geomean(values) -> float:
    positive = [float(v) for v in values if v > 0 and np.isfinite(v)]
    if not positive:
        return float("nan")
    return math.exp(sum(math.log(v) for v in positive) / len(positive))


def select_operating_points(frame: pd.DataFrame, slo_ms: float) -> pd.DataFrame:
    """Same selection rule as the v1 model, so the comparison is like-for-like."""
    names = {name for name, _ in MODELS}
    sel = frame.loc[
        frame["model"].isin(names)
        & frame["baseline"].isin(BASELINES)
        & frame["tpot_p50_ms"].le(slo_ms)
    ].copy()
    sel["model_weight_bytes"] = sel["model"].map(weight_bytes)
    sel = sel.loc[
        sel["model_weight_bytes"] / sel["tp"] + 4e9 <= HBM_GB["blackwell"] * 1e9
    ].copy()
    sel["goodput_per_gpu"] = sel["batch"] / (sel["tpot_p50_ms"] / 1000.0) / sel["tp"]
    sel.sort_values("goodput_per_gpu", ascending=False, inplace=True)
    return sel.drop_duplicates(["model", "context_length", "baseline"])


def add_traffic_terms(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive the byte counters the v1 traffic export never produced."""
    r = frame.copy()

    # Full HBF-resident KV per output token, recovered from the selection ratio.
    # For Dense sparsity_fraction == 1, so this is an identity there.
    r["hbf_resident_kv_bytes"] = (
        r["hbf_useful_bytes_per_output_token"] / r["sparsity_fraction"]
    )

    # SPLASH: the NMA streams every centroid through the array each step.
    is_sparse = r["baseline"].eq("Sparse")
    r["centroid_scan_bytes"] = np.where(
        is_sparse, r["hbf_resident_kv_bytes"] * CENTROID_FRACTION_OF_KV, 0.0
    )

    # Naive Sparse: token-granular selection must score every key, so it reads
    # the whole key half of the history out of the array each step.
    is_naive = r["baseline"].eq("Naive")
    r["selection_scan_bytes"] = np.where(
        is_naive, r["hbf_resident_kv_bytes"] * 0.5, 0.0
    ) + r["centroid_scan_bytes"]

    # Every baseline stores history in HBF, so every baseline programs the KV of
    # each generated token once, plus SPLASH's centroid.  KV per token is the
    # resident footprint divided by the context it covers.
    kv_bytes_per_token = (
        r["hbf_resident_kv_bytes"] / r["effective_attention_context"].clip(lower=1)
    )
    r["hbf_program_bytes"] = kv_bytes_per_token * np.where(
        is_sparse, 1.0 + CENTROID_FRACTION_OF_KV, 1.0
    )

    # Total array traffic actually sensed, per output token, per GPU shard.
    r["hbf_array_read_bytes"] = (
        r["hbf_physical_bytes_per_output_token"] + r["selection_scan_bytes"]
    )

    # NMA work: one FP16 MAC per centroid element scored.
    r["nma_macs"] = r["centroid_scan_bytes"] / 2.0  # FP16 -> 2 B per element
    return r


def add_energy(frame: pd.DataFrame, c: Coefficients) -> pd.DataFrame:
    r = frame.copy()
    tp = r["tp"]
    scale = tp  # per-GPU counters -> whole-request totals

    r["hbm_bytes_total"] = (
        r["hbm_read_bytes_per_output_token"] + r["hbm_write_bytes_per_output_token"]
    ) * scale + r["model_weight_bytes"] / r["batch"]

    # --- GPU board -------------------------------------------------------
    # Board power x execution time.  This form is measurement-validated: on 132
    # Blackwell attention-decode shapes it predicts dynamic board energy to
    # 0.61 % wAPE.  A bytes-linear form fits equally well (0.65 %) because the
    # kernel is bandwidth-bound and corr(bytes, time) = 0.999998, so the two are
    # not distinguishable and the GPU-internal HBM/logic split is NOT
    # identifiable from board telemetry.  HBM energy is therefore reported as a
    # diagnostic fraction of the board term, never added on top of it.
    r["e_gpu_board"] = (
        c.gpu_board_w * tp * (r["tpot_p50_ms"] / 1000.0) / r["batch"]
    )
    r["hbm_diagnostic_j"] = r["hbm_bytes_total"] * c.hbm_pj_per_byte * 1e-12
    r["hbm_fraction_of_board"] = r["hbm_diagnostic_j"] / r["e_gpu_board"]
    r["e_gpu"] = r["e_gpu_board"]

    # --- HBF media --------------------------------------------------------
    r["e_hbf_read"] = r["hbf_array_read_bytes"] * scale * c.hbf_read_pj_per_byte * 1e-12
    r["e_hbf_program"] = (
        r["hbf_program_bytes"] * scale * c.hbf_program_pj_per_byte * 1e-12
    )
    r["e_hbf_erase"] = r["hbf_program_bytes"] * scale * c.hbf_erase_pj_per_byte * 1e-12

    # --- SPLASH on-stack logic -------------------------------------------
    # Bridge stages every retrieved page once in and once out.
    r["e_bridge"] = (
        2.0
        * r["hbf_physical_bytes_per_output_token"]
        * scale
        * c.sram_bridge_pj_per_byte
        * 1e-12
    )
    r["e_link"] = (
        r["hbf_link_bytes_per_output_token"] * scale * c.d2d_link_pj_per_byte * 1e-12
    )
    r["e_nma"] = r["nma_macs"] * scale * c.nma_pj_per_mac * 1e-12

    r["e_total"] = (
        r["e_gpu"]
        + r["e_hbf_read"]
        + r["e_hbf_program"]
        + r["e_hbf_erase"]
        + r["e_bridge"]
        + r["e_link"]
        + r["e_nma"]
    )
    return r


COMPONENTS = [
    "e_gpu_board",
    "e_hbf_read",
    "e_hbf_program",
    "e_hbf_erase",
    "e_bridge",
    "e_link",
    "e_nma",
]


def ratios(energy: pd.DataFrame) -> dict[str, float]:
    pivot = energy.pivot(
        index=["model", "context_length"], columns="baseline", values="e_total"
    ).dropna()
    return {
        "dense_over_splash": geomean(pivot["Dense"] / pivot["Sparse"]),
        "naive_over_splash": geomean(pivot["Naive"] / pivot["Sparse"]),
        "pairs": int(len(pivot)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--traffic", type=Path,
        default=Path("results/energy_b200/sweep_component_traffic.csv"))
    ap.add_argument("--slo-ms", type=float, default=100.0)
    ap.add_argument(
        "--output", type=Path,
        default=Path("results/energy_b200/energy_v2.csv"))
    ap.add_argument(
        "--summary", type=Path,
        default=Path("results/energy_b200/energy_v2_summary.json"))
    ap.add_argument("--ablate", action="store_true",
                    help="price each newly charged term separately")
    args = ap.parse_args()

    traffic = pd.read_csv(args.traffic)
    selected = add_traffic_terms(select_operating_points(traffic, args.slo_ms))

    base = Coefficients()
    energy = add_energy(selected, base)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    energy.to_csv(args.output, index=False)

    ref = ratios(energy)
    print(f"operating points: {len(energy)}   model-context pairs: {ref['pairs']}")
    print(f"BASELINE  H3/SPLASH={ref['dense_over_splash']:.2f}x  "
          f"Naive/SPLASH={ref['naive_over_splash']:.2f}x")

    print("\n--- component share of total, by baseline (mean) ---")
    share = energy.groupby("baseline")[COMPONENTS].sum()
    share = share.div(share.sum(axis=1), axis=0)
    print((share * 100).round(2).to_string())

    if args.ablate:
        print("\n--- what each newly charged term costs (H3/SPLASH ratio) ---")
        variants = {
            "v1 accounting (no scan, no write, no logic)": dict(
                scan=False, write=False, logic=False),
            "+ centroid scan (SPLASH) and full-key scan (Naive)": dict(
                scan=True, write=False, logic=False),
            "+ HBF program and erase": dict(scan=True, write=True, logic=False),
            "+ bridge, link, NMA  (= v2)": dict(scan=True, write=True, logic=True),
        }
        for label, flags in variants.items():
            t = selected.copy()
            if not flags["scan"]:
                t["selection_scan_bytes"] = 0.0
                t["centroid_scan_bytes"] = 0.0
                t["hbf_array_read_bytes"] = t["hbf_physical_bytes_per_output_token"]
                t["nma_macs"] = 0.0
            if not flags["write"]:
                t["hbf_program_bytes"] = 0.0
            e = add_energy(t, base)
            if not flags["logic"]:
                e["e_total"] = e["e_total"] - e["e_bridge"] - e["e_link"] - e["e_nma"]
            rr = ratios(e)
            print(f"  {label:52} H3/SPLASH={rr['dense_over_splash']:.2f}x  "
                  f"Naive/SPLASH={rr['naive_over_splash']:.2f}x")

    # --- sensitivity ------------------------------------------------------
    grid = {
        "gpu_board_w": [450.0, 700.0, 1000.0],
        "hbf_read_pj_per_byte": [32.0, 64.0, 96.0],
        "hbf_program_read_ratio": [5.0, 10.0, 20.0],
        "sram_bridge_pj_per_byte": [0.5, 1.0, 2.0],
    }
    cases = []
    for gw in grid["gpu_board_w"]:
        for hr in grid["hbf_read_pj_per_byte"]:
            for pr in grid["hbf_program_read_ratio"]:
                for sb in grid["sram_bridge_pj_per_byte"]:
                    c = Coefficients(
                        gpu_board_w=gw, hbf_read_pj_per_byte=hr,
                        hbf_program_read_ratio=pr, sram_bridge_pj_per_byte=sb)
                    rr = ratios(add_energy(selected, c))
                    cases.append({
                        "gpu_board_w": gw, "hbf_read_pj_per_byte": hr,
                        "hbf_program_read_ratio": pr,
                        "sram_bridge_pj_per_byte": sb, **rr})
    d = [x["dense_over_splash"] for x in cases]
    n = [x["naive_over_splash"] for x in cases]
    print(f"\n--- sensitivity over {len(cases)} coefficient combinations ---")
    print(f"H3/SPLASH    {min(d):.2f}x - {max(d):.2f}x   (baseline {ref['dense_over_splash']:.2f}x)")
    print(f"Naive/SPLASH {min(n):.2f}x - {max(n):.2f}x   (baseline {ref['naive_over_splash']:.2f}x)")

    summary = {
        "schema_version": 2,
        "status": "component_complete_measurement_anchored_form",
        "selected_operating_points": int(len(energy)),
        "model_context_pairs": ref["pairs"],
        "selection": (
            f"maximum tokens/s/GPU under the {args.slo_ms:g} ms TPOT SLO after "
            "requiring weights plus 4 GB reserve to fit in per-GPU HBM"),
        "coefficients": asdict(base),
        "provenance": PROVENANCE,
        "included": [
            "GPU board power x execution time (measurement-validated form)",
            "HBF array reads including the centroid scan (SPLASH) and the "
            "full-key selection scan (Naive Sparse)",
            "HBF program and erase for KV offload",
            "SRAM protocol bridge",
            "HBF-to-GPU D2D link",
            "NMA MAC array",
        ],
        "omitted": ["FTL control logic", "cooling and host energy"],
        "baseline_case": ref,
        "sensitivity_cases": cases,
        "sensitivity_envelope": {
            "dense_over_splash_min": min(d), "dense_over_splash_max": max(d),
            "naive_over_splash_min": min(n), "naive_over_splash_max": max(n),
        },
        "component_share_percent": (share * 100).round(3).to_dict(),
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nwrote {args.output} and {args.summary}")


if __name__ == "__main__":
    main()
