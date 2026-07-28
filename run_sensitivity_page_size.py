#!/usr/bin/env python3
"""End-to-end physical HBF page-size sensitivity.

Only physically valid NAND pages are tested.  FP16 K and V are stored in
separate head streams.  At head dimension 128, one token occupies 256 B in
either stream, so 4/8/16/32-KB physical pages contain 16/32/64/128 tokens.
Selection keeps a fixed 10% of KV tokens and reads both corresponding streams.
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from evaluation_common import (
    BATCHES,
    PLATFORM,
    capacity_feasible,
    hbm_kv_fraction,
)


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_point_hbf.py"
OUT = ROOT / "results" / "sensitivity_page_size.csv"
MODEL = "meta-llama/Meta-Llama-3-8B"
TP = 8
CONTEXT = 1_048_576
PAGE_CONFIGS = ((4, 16), (8, 32), (16, 64), (32, 128))
FIELDS = (
    "model", "tp", "context_length", "batch", "page_kb", "page_tokens",
    "selection_fraction", "centroid_fraction_of_kv", "hbm_kv_fraction",
    "tpot_p50_ms", "tpot_p99_ms", "status",
)


def run(job: dict) -> dict:
    command = [
        sys.executable, str(RUNNER),
        "--model", MODEL,
        "--device", PLATFORM.device,
        "--batch_size", str(job["batch"]),
        "--context_length", str(CONTEXT),
        "--decode_tokens", "4",
        "--num_requests", str(job["batch"]),
        "--tensor_parallel_size", str(TP),
        "--sparsity_fraction", str(job["selection_fraction"]),
        "--selection_page_tokens", str(job["page_tokens"]),
        "--hbm_kv_fraction", str(job["hbm_kv_fraction"]),
        "--backing_bw_gbps", str(PLATFORM.effective_hbf_bw_gbps()),
        "--hbfsim_toml", str(ROOT / "configs" / "hbf_paper.toml"),
        "--cache_dir", str(ROOT / "results" / "predictor_cache_eval"),
    ]
    process = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=900
    )
    match = re.search(
        r"RESULT status=OK tpot_p50_ms=([\d.]+) tpot_p99_ms=([\d.]+)",
        process.stdout,
    )
    if not match:
        tail = (process.stderr or process.stdout).strip().splitlines()
        reason = tail[-1][-100:] if tail else "no-result"
        return {
            "tpot_p50_ms": "",
            "tpot_p99_ms": "",
            "status": f"FAIL({process.returncode}):{reason}",
        }
    return {
        "tpot_p50_ms": match.group(1),
        "tpot_p99_ms": match.group(2),
        "status": "OK",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    jobs = []
    for batch in BATCHES:
        if not capacity_feasible(MODEL, batch, CONTEXT, TP, "hbf"):
            continue
        hot = hbm_kv_fraction(MODEL, batch, CONTEXT, TP, "hbf")
        for page_kb, page_tokens in PAGE_CONFIGS:
            jobs.append({
                "model": MODEL,
                "tp": TP,
                "context_length": CONTEXT,
                "batch": batch,
                "page_kb": page_kb,
                "page_tokens": page_tokens,
                "selection_fraction": PLATFORM.sparse_kv_fraction,
                "centroid_fraction_of_kv": 1.0 / (2.0 * page_tokens),
                "hbm_kv_fraction": f"{hot:.8f}",
            })
    with OUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run, job): job for job in jobs}
            for index, future in enumerate(as_completed(futures), 1):
                writer.writerow({**futures[future], **future.result()})
                stream.flush()
                if index % 8 == 0 or index == len(jobs):
                    print(f"  {index}/{len(jobs)}", flush=True)
    print(OUT)


if __name__ == "__main__":
    main()
