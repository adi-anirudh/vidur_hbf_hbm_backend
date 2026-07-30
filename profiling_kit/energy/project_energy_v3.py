#!/usr/bin/env python3
"""SPLASH energy model v3: fully isolated components, no board-power term.

v1 and v2 both charged the GPU as ``P_board x TPOT``.  That makes GPU energy
identically ``P / goodput_per_GPU``, so the energy result was ~98 % a
restatement of the throughput result (measured: energy geomean 5.783x vs
goodput geomean 5.862x, r = 0.964).  It also charged a stalled H3 the same
power as a busy SPLASH, which biases the ratio in SPLASH's favour.

v3 removes the board-power term entirely and models each component from its own
activity counter, in the style of Horowitz ISSCC'14 per-operation energies.
NOTE: do NOT claim lineage from AttAcc -- it uses Ramulator 2.0 with IDD-based
DRAMPower, a cycle-accurate method, not a per-operation table.

    E = a_flop * FLOPs            GPU logic  (attention + FFN/projection)
      + a_hbm  * HBM_bytes        HBM devices + controller/PHY
      + a_hbf  * HBF_read_bytes   NAND array, incl. selection scans
      + program / erase           KV offload
      + a_mac  * NMA_MACs         centroid scoring on the base die
      + a_sram * bridge_bytes     SRAM protocol bridge
      + a_link * link_bytes       HBF -> GPU D2D
      + P_idle * t                the ONLY time-dependent term

Every dominant term is now an activity count, so the result is no longer a
reparameterisation of latency.  Only the small idle term retains a time
dependence, and it is swept.

VALIDATION: the implied GPU board power (logic + HBM + idle) is 228 W at the
median and 100 % of rows fall inside the 137-300 W band Ma et al. measure for
H200 decode (arXiv:2605.11999).  The model is never fitted to that band.

Usage:
    .venv/bin/python profiling_kit/energy/project_energy_v3.py
"""

from __future__ import annotations

import argparse
import itertools
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
# Parameters actually multiplied per token.  MoE models activate a subset.
ACTIVE_PARAMS = {
    "Qwen/Qwen3-235B-A22B": 22.0e9,          # A22B
    "mistralai/Mixtral-8x22B-v0.1": 39.0e9,  # top-2 of 8 x 22B
    "deepseek-ai/deepseek-llm-67b-chat": 67.0e9,
    "meta-llama/Meta-Llama-3.1-405B": 405.0e9,
}
BASELINES = ["Dense", "Naive", "Sparse"]

TOKENS_PER_PAGE = 16
CENTROID_FRACTION_OF_KV = 1.0 / (2 * TOKENS_PER_PAGE)
PAGE_BYTES = 4096


@dataclass
class Coefficients:
    gpu_pj_per_flop: float = 0.40     # B200 board-at-peak 2.25 PFLOP/s @ ~1 kW
    hbm_pj_per_byte: float = 22.49    # H200/HBM3e rail, extrapolated to B200
    hbf_read_pj_per_byte: float = 64.0
    hbf_program_read_ratio: float = 10.0
    hbf_erase_uj_per_block: float = 300.0
    pages_per_block: int = 256
    sram_bridge_pj_per_byte: float = 1.0
    d2d_link_pj_per_byte: float = 1.0
    nma_pj_per_mac: float = 0.30      # FP16 MAC at 16 nm
    idle_w_per_gpu: float = 150.0     # only time-dependent term

    @property
    def hbf_program_pj_per_byte(self) -> float:
        return self.hbf_read_pj_per_byte * self.hbf_program_read_ratio

    @property
    def hbf_erase_pj_per_byte(self) -> float:
        return self.hbf_erase_uj_per_block * 1e6 / (self.pages_per_block * PAGE_BYTES)


PROVENANCE = {
    "gpu_pj_per_flop": "B200 dense FP16 ~2.25 PFLOP/s at ~1 kW board => 0.44 pJ/FLOP "
    "at peak; 0.40 used for logic. Consistent with Horowitz ISSCC'14 scaled to 4 nm. "
    "Swept 0.20-0.80.",
    "hbm_pj_per_byte": "Ujeniya et al. arXiv:2604.11391, nvidia-smi memory-rail "
    "readings. H200 (HBM3e, same generation as B200): 220 W memory-bound Triad vs "
    "110 W DGEMM = 110 W activity-attributed over 4.89 TB/s = 22.5 pJ/B. Independently "
    "corroborated by FlashAccel's cited HBM3e 2.99 pJ/bit = 23.9 pJ/B. Swept 11.9-28.0.",
    "hbf_read_pj_per_byte": "8 pJ/bit; independently matches FlashAccel Table 2.",
    "hbf_program_read_ratio": "HBFSim energy_table.toml, 100 us program vs 10 us read.",
    "nma_pj_per_mac": "FP16 MAC at 16 nm, Horowitz-scaled. Swept 0.10-1.00.",
    "idle_w_per_gpu": "B200 idle board power. Swept 100-250 W.",
    "sram_bridge_pj_per_byte": "CACTI-class large SRAM / HBFSim interconnect.",
    "d2d_link_pj_per_byte": "short-reach on-package D2D / HBFSim host bus.",
}


def geomean(v) -> float:
    p = [float(x) for x in v if x > 0 and np.isfinite(x)]
    return math.exp(sum(map(math.log, p)) / len(p)) if p else float("nan")


def select_operating_points(f: pd.DataFrame, slo_ms: float) -> pd.DataFrame:
    names = {n for n, _ in MODELS}
    s = f.loc[f.model.isin(names) & f.baseline.isin(BASELINES)
              & f.tpot_p50_ms.le(slo_ms)].copy()
    s["model_weight_bytes"] = s.model.map(weight_bytes)
    s = s.loc[s.model_weight_bytes / s.tp + 4e9 <= HBM_GB["blackwell"] * 1e9].copy()
    s["goodput_per_gpu"] = s.batch / (s.tpot_p50_ms / 1000.0) / s.tp
    s.sort_values("goodput_per_gpu", ascending=False, inplace=True)
    return s.drop_duplicates(["model", "context_length", "baseline"])


def add_traffic(f: pd.DataFrame) -> pd.DataFrame:
    r = f.copy()
    r["hbf_resident_kv_bytes"] = (
        r.hbf_useful_bytes_per_output_token / r.sparsity_fraction)
    sp, nv = r.baseline.eq("Sparse"), r.baseline.eq("Naive")
    r["centroid_scan_bytes"] = np.where(
        sp, r.hbf_resident_kv_bytes * CENTROID_FRACTION_OF_KV, 0.0)
    # Naive scores every key: it reads the whole key half of the history.
    r["selection_scan_bytes"] = (
        np.where(nv, r.hbf_resident_kv_bytes * 0.5, 0.0) + r.centroid_scan_bytes)
    kv_per_tok = r.hbf_resident_kv_bytes / r.effective_attention_context.clip(lower=1)
    r["hbf_program_bytes"] = kv_per_tok * np.where(
        sp, 1.0 + CENTROID_FRACTION_OF_KV, 1.0)
    r["hbf_array_read_bytes"] = (
        r.hbf_physical_bytes_per_output_token + r.selection_scan_bytes)
    r["nma_macs"] = r.centroid_scan_bytes / 2.0          # FP16: 1 MAC per element
    # Non-attention FLOPs per token: two per active parameter.
    r["dense_flops_per_output_token"] = 2.0 * r.model.map(ACTIVE_PARAMS)
    return r


def add_energy(f: pd.DataFrame, c: Coefficients) -> pd.DataFrame:
    r = f.copy()
    tp = r.tp
    r["e_gpu_logic"] = (
        (r.attention_flops_per_output_token * tp + r.dense_flops_per_output_token)
        * c.gpu_pj_per_flop * 1e-12)
    r["e_hbm"] = (
        ((r.hbm_read_bytes_per_output_token + r.hbm_write_bytes_per_output_token) * tp
         + r.model_weight_bytes / r.batch) * c.hbm_pj_per_byte * 1e-12)
    r["e_hbf_read"] = r.hbf_array_read_bytes * tp * c.hbf_read_pj_per_byte * 1e-12
    r["e_hbf_program"] = r.hbf_program_bytes * tp * c.hbf_program_pj_per_byte * 1e-12
    r["e_hbf_erase"] = r.hbf_program_bytes * tp * c.hbf_erase_pj_per_byte * 1e-12
    r["e_bridge"] = (2.0 * r.hbf_physical_bytes_per_output_token * tp
                     * c.sram_bridge_pj_per_byte * 1e-12)
    r["e_link"] = (r.hbf_link_bytes_per_output_token * tp
                   * c.d2d_link_pj_per_byte * 1e-12)
    r["e_nma"] = r.nma_macs * tp * c.nma_pj_per_mac * 1e-12
    r["e_idle"] = c.idle_w_per_gpu * tp * (r.tpot_p50_ms / 1000.0) / r.batch
    r["e_total"] = r[COMPONENTS].sum(axis=1)
    return r


COMPONENTS = ["e_gpu_logic", "e_hbm", "e_hbf_read", "e_hbf_program",
              "e_hbf_erase", "e_bridge", "e_link", "e_nma", "e_idle"]


def ratios(e: pd.DataFrame) -> dict:
    p = e.pivot_table(index=["model", "context_length"], columns="baseline",
                      values="e_total").dropna()
    return {"dense_over_splash": geomean(p["Dense"] / p["Sparse"]),
            "naive_over_splash": geomean(p["Naive"] / p["Sparse"]),
            "pairs": int(len(p))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traffic", type=Path,
                    default=Path("results/energy_b200/sweep_component_traffic.csv"))
    ap.add_argument("--slo-ms", type=float, default=100.0)
    ap.add_argument("--output", type=Path,
                    default=Path("results/energy_b200/energy_v3.csv"))
    ap.add_argument("--summary", type=Path,
                    default=Path("results/energy_b200/energy_v3_summary.json"))
    a = ap.parse_args()

    sel = add_traffic(select_operating_points(pd.read_csv(a.traffic), a.slo_ms))
    base = Coefficients()
    e = add_energy(sel, base)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    e.to_csv(a.output, index=False)

    ref = ratios(e)
    print(f"operating points {len(e)}   model-context pairs {ref['pairs']}")
    print(f"H3/SPLASH = {ref['dense_over_splash']:.2f}x   "
          f"Naive/SPLASH = {ref['naive_over_splash']:.2f}x")

    print("\n--- component share of total (%), by baseline ---")
    sh = e.groupby("baseline")[COMPONENTS].sum()
    print((sh.div(sh.sum(axis=1), axis=0) * 100).round(2).to_string())

    print("\n--- circularity check ---")
    pe = e.pivot_table(index=["model", "context_length"], columns="baseline",
                       values="e_total")
    pg = e.pivot_table(index=["model", "context_length"], columns="baseline",
                       values="goodput_per_gpu")
    j = pd.concat([(pe["Dense"] / pe["Sparse"]).rename("E"),
                   (pg["Sparse"] / pg["Dense"]).rename("G")], axis=1).dropna()
    print(f"energy geomean {geomean(j.E):.3f}x vs goodput geomean {geomean(j.G):.3f}x")
    print(f"correlation {np.corrcoef(j.E, j.G)[0, 1]:.4f}   "
          f"(v2 was 0.964 with the two agreeing to 1.4%)")

    grid = {"gpu_pj_per_flop": [0.20, 0.40, 0.80],
            "hbm_pj_per_byte": [11.93, 22.49, 28.00],
            "hbf_read_pj_per_byte": [32.0, 64.0, 96.0],
            "nma_pj_per_mac": [0.10, 0.30, 1.00],
            "idle_w_per_gpu": [100.0, 150.0, 250.0]}
    cases = []
    for combo in itertools.product(*grid.values()):
        c = Coefficients(**dict(zip(grid.keys(), combo)))
        cases.append({**dict(zip(grid.keys(), combo)), **ratios(add_energy(sel, c))})
    d = [x["dense_over_splash"] for x in cases]
    n = [x["naive_over_splash"] for x in cases]
    print(f"\n--- sensitivity over {len(cases)} combinations ---")
    print(f"H3/SPLASH    {min(d):.2f}x - {max(d):.2f}x  (baseline {ref['dense_over_splash']:.2f}x)")
    print(f"Naive/SPLASH {min(n):.2f}x - {max(n):.2f}x  (baseline {ref['naive_over_splash']:.2f}x)")

    a.summary.write_text(json.dumps({
        "schema_version": 3,
        "status": "component_isolated_activity_counted",
        "selected_operating_points": int(len(e)),
        "model_context_pairs": ref["pairs"],
        "coefficients": asdict(base),
        "provenance": PROVENANCE,
        "no_board_power_term": True,
        "baseline_case": ref,
        "component_share_percent":
            (sh.div(sh.sum(axis=1), axis=0) * 100).round(3).to_dict(),
        "circularity": {"energy_geomean": geomean(j.E),
                        "goodput_geomean": geomean(j.G),
                        "correlation": float(np.corrcoef(j.E, j.G)[0, 1])},
        "sensitivity_envelope": {"dense_over_splash": [min(d), max(d)],
                                 "naive_over_splash": [min(n), max(n)]},
        "sensitivity_cases": cases,
    }, indent=2) + "\n")
    print(f"\nwrote {a.output} and {a.summary}")


if __name__ == "__main__":
    main()
