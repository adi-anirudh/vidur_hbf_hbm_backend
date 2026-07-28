#!/usr/bin/env python3
"""End-to-end co-design ablation across long-context lengths.

The matrix changes only the retrieval mechanism. Hardware, TP, batch choices,
HBM placement, HBF bandwidth, and SLO are inherited from evaluation_common.py.
At each context, the plotting script independently chooses each system's
highest-throughput batch under the 100 ms TPOT-p50 SLO.
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from evaluation_common import (
    BATCHES,
    PLATFORM,
    capacity_feasible,
    hbm_kv_fraction,
)
from sweep_capacity import weight_bytes

ROOT = Path(__file__).parent
PY = sys.executable
RUNNER = str(ROOT / "run_point_hbf.py")
TOML = str(ROOT / "configs/hbf_paper.toml")
CACHE = str(ROOT / "results" / "predictor_cache_eval")
OUT = ROOT / "results" / "ablation_matrix.csv"

MODEL = "meta-llama/Meta-Llama-3-8B"
TP = 8
CONTEXTS = (131072, 262144, 524288, 1048576, 2097152)
# Measured by measure_codesign.py using the final 1024-plane layout at 10%.
AMP = 3.5635
IMBALANCE = 2.175
SYSTEMS = (
    ("Dense", ("--sparsity_fraction", "1.0")),
    ("TokenGranular", (
        "--sparsity_fraction", "0.1", "--hbf_sra",
        "--sparse_read_amplification", str(AMP),
    )),
    ("FullPageScore", ("--sparsity_fraction", "0.1", "--hbf_sra")),
    ("PlaneImbalance", (
        "--sparsity_fraction", "0.1",
        "--plane_imbalance_factor", str(IMBALANCE),
    )),
    ("SPLASH", ("--sparsity_fraction", "0.1")),
)
FIELDS = (
    "model", "device", "tp", "batch", "context_length", "system",
    "sparsity_fraction", "hbm_kv_fraction", "weights_hbm_gb_per_gpu",
    "backing_bw_gbps", "read_amplification", "plane_imbalance",
    "full_key_scoring", "tpot_p50_ms", "tpot_p99_ms", "status", "wall_s",
)


def run_point(job: dict, timeout_s: int) -> dict:
    cmd = [
        PY, RUNNER,
        "--model", MODEL,
        "--device", PLATFORM.device,
        "--batch_size", str(job["batch"]),
        "--context_length", str(job["context_length"]),
        "--decode_tokens", "4",
        "--num_requests", str(job["batch"]),
        "--tensor_parallel_size", str(TP),
        "--hbm_kv_fraction", str(job["hbm_kv_fraction"]),
        "--backing_bw_gbps", str(PLATFORM.effective_hbf_bw_gbps()),
        "--hbfsim_toml", TOML,
        "--cache_dir", CACHE,
        *job["flags"],
    ]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s, cwd=ROOT
        )
        wall = time.perf_counter() - t0
        m = re.search(
            r"RESULT status=OK tpot_p50_ms=([\d.]+) tpot_p99_ms=([\d.]+)",
            p.stdout,
        )
        if m:
            return {
                "tpot_p50_ms": m.group(1),
                "tpot_p99_ms": m.group(2),
                "status": "OK",
                "wall_s": f"{wall:.1f}",
            }
        tail = (p.stderr or p.stdout).strip().splitlines()
        reason = tail[-1][-120:] if tail else "no-result"
        return {
            "tpot_p50_ms": "",
            "tpot_p99_ms": "",
            "status": f"FAIL({p.returncode}):{reason}",
            "wall_s": f"{wall:.1f}",
        }
    except subprocess.TimeoutExpired:
        return {
            "tpot_p50_ms": "",
            "tpot_p99_ms": "",
            "status": f"TIMEOUT>{timeout_s}s",
            "wall_s": "",
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    done = set()
    if args.resume and OUT.exists():
        for row in csv.DictReader(OUT.open()):
            if row["status"] == "OK":
                done.add((
                    int(row["context_length"]), int(row["batch"]), row["system"]
                ))

    jobs = []
    for context in CONTEXTS:
        for batch in BATCHES:
            if not capacity_feasible(MODEL, batch, context, TP, "hbf"):
                continue
            hot = hbm_kv_fraction(MODEL, batch, context, TP, "hbf")
            for system, flags in SYSTEMS:
                if (context, batch, system) in done:
                    continue
                jobs.append({
                    "model": MODEL,
                    "device": PLATFORM.device,
                    "tp": TP,
                    "batch": batch,
                    "context_length": context,
                    "system": system,
                    "sparsity_fraction": 1.0 if system == "Dense" else 0.1,
                    "hbm_kv_fraction": f"{hot:.8f}",
                    "weights_hbm_gb_per_gpu": (
                        f"{weight_bytes(MODEL) / TP / 1e9:.4f}"
                    ),
                    "backing_bw_gbps": (
                        f"{PLATFORM.effective_hbf_bw_gbps():.4f}"
                    ),
                    "read_amplification": AMP if system == "TokenGranular" else 1.0,
                    "plane_imbalance": (
                        IMBALANCE if system == "PlaneImbalance" else 1.0
                    ),
                    "full_key_scoring": (
                        system in {"TokenGranular", "FullPageScore"}
                    ),
                    "flags": flags,
                })

    mode = "a" if args.resume and OUT.exists() else "w"
    with OUT.open(mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
        print(f"ablation matrix points: {len(jobs)}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(run_point, job, args.timeout): job for job in jobs
            }
            for i, fut in enumerate(as_completed(futures), 1):
                job = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    result = {
                        "tpot_p50_ms": "",
                        "tpot_p99_ms": "",
                        "status": f"ERR:{type(exc).__name__}:{exc}",
                        "wall_s": "",
                    }
                writer.writerow({**job, **result})
                fh.flush()
                if i % 25 == 0 or i == len(jobs):
                    print(f"  {i}/{len(jobs)}", flush=True)
    print(f"ABLATION MATRIX DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
