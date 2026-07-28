#!/usr/bin/env python3
"""End-to-end selection-budget sensitivity at fixed model, context, and TP."""
from __future__ import annotations

import csv
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from evaluation_common import BATCHES, PLATFORM, capacity_feasible, hbm_kv_fraction


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "run_point_hbf.py"
OUT = ROOT / "results" / "sensitivity_budget_multictx.csv"
MODEL = "meta-llama/Meta-Llama-3-8B"
TP = 8
CONTEXTS = (131_072, 524_288, 1_048_576)
FRACTIONS = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
FIELDS = (
    "model", "tp", "context_length", "batch", "selection_fraction",
    "hbm_kv_fraction", "tpot_p50_ms", "tpot_p99_ms", "status",
)


def run(job: dict) -> dict:
    command = [
        sys.executable, str(RUNNER), "--model", MODEL, "--device",
        PLATFORM.device, "--batch_size", str(job["batch"]),
        "--context_length", str(job["context_length"]), "--decode_tokens", "4",
        "--num_requests", str(job["batch"]), "--tensor_parallel_size", str(TP),
        "--sparsity_fraction", str(job["selection_fraction"]),
        "--hbm_kv_fraction", str(job["hbm_kv_fraction"]),
        "--backing_bw_gbps", "8000", "--hbfsim_toml",
        str(ROOT / "configs" / "hbf_paper.toml"), "--cache_dir",
        str(ROOT / "results" / "predictor_cache_eval"),
    ]
    process = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, timeout=900
    )
    match = re.search(
        r"RESULT status=OK tpot_p50_ms=([\d.]+) tpot_p99_ms=([\d.]+)",
        process.stdout,
    )
    if not match:
        return {"tpot_p50_ms": "", "tpot_p99_ms": "",
                "status": f"FAIL({process.returncode})"}
    return {"tpot_p50_ms": match.group(1), "tpot_p99_ms": match.group(2),
            "status": "OK"}


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    jobs = []
    for context in CONTEXTS:
        for batch in BATCHES:
            if not capacity_feasible(MODEL, batch, context, TP, "hbf"):
                continue
            hot = hbm_kv_fraction(MODEL, batch, context, TP)
            for fraction in FRACTIONS:
                jobs.append({
                    "model": MODEL, "tp": TP, "context_length": context,
                    "batch": batch, "selection_fraction": fraction,
                    "hbm_kv_fraction": f"{hot:.8f}",
                })
    with OUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run, job): job for job in jobs}
            for index, future in enumerate(as_completed(futures), 1):
                writer.writerow({**futures[future], **future.result()})
                stream.flush()
                if index % 12 == 0 or index == len(jobs):
                    print(f"  {index}/{len(jobs)}", flush=True)
    print(OUT)


if __name__ == "__main__":
    main()
