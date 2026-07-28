#!/usr/bin/env python3
"""Bound the effect of Blackwell TP collective fixed latency on SLO throughput."""
from __future__ import annotations

import csv
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from evaluation_common import BATCHES, PLATFORM, capacity_feasible, hbm_kv_fraction


ROOT = Path(__file__).resolve().parent
PY = sys.executable
RUNNER = ROOT / "run_point_hbf.py"
GENERATOR = ROOT / "generate_blackwell_network_profile.py"
OUT = ROOT / "results" / "sensitivity_tp_comm.csv"
PROFILE_ROOT = ROOT / "results" / "network_sensitivity"
MODEL_TP = {
    "meta-llama/Meta-Llama-3-8B": 8,
    "meta-llama/Meta-Llama-3-70B": 8,
    "Qwen/Qwen3-235B-A22B": 4,
}
CONTEXT = 1_048_576
LATENCIES_US = (0.0, 3.0, 10.0, 30.0)
SYSTEMS = (("H3", 1.0), ("SPLASH", 0.1))
FIELDS = (
    "model", "tp", "context_length", "batch", "system", "device_latency_us",
    "link_bandwidth_gbps", "hbm_kv_fraction", "tpot_p50_ms", "tpot_p99_ms",
    "status",
)


def generate_profiles() -> dict[float, Path]:
    result = {}
    for latency in LATENCIES_US:
        directory = PROFILE_ROOT / f"latency_{latency:g}us"
        subprocess.run([
            PY, str(GENERATOR), "--output-dir", str(directory),
            "--bandwidth-gbps", "1800", "--latency-us", str(latency),
        ], check=True, cwd=ROOT, capture_output=True, text=True)
        result[latency] = directory / "all_reduce.csv"
    return result


def run(job: dict) -> dict:
    command = [
        PY, str(RUNNER), "--model", job["model"], "--device", PLATFORM.device,
        "--batch_size", str(job["batch"]), "--context_length", str(CONTEXT),
        "--decode_tokens", "4", "--num_requests", str(job["batch"]),
        "--tensor_parallel_size", str(job["tp"]),
        "--sparsity_fraction", str(job["sparsity_fraction"]),
        "--hbm_kv_fraction", str(job["hbm_kv_fraction"]),
        "--backing_bw_gbps", "8000", "--hbfsim_toml",
        str(ROOT / "configs" / "hbf_paper.toml"), "--cache_dir",
        str(ROOT / "results" / "predictor_cache_eval"),
        "--all_reduce_input_file", str(job["profile"]),
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
    profiles = generate_profiles()
    jobs = []
    for model, tp in MODEL_TP.items():
        for batch in BATCHES:
            if not capacity_feasible(model, batch, CONTEXT, tp, "hbf"):
                continue
            hot = hbm_kv_fraction(model, batch, CONTEXT, tp)
            for system, sparsity in SYSTEMS:
                for latency, profile in profiles.items():
                    jobs.append({
                        "model": model, "tp": tp, "context_length": CONTEXT,
                        "batch": batch, "system": system,
                        "sparsity_fraction": sparsity,
                        "device_latency_us": latency,
                        "link_bandwidth_gbps": 1800.0,
                        "hbm_kv_fraction": f"{hot:.8f}", "profile": profile,
                    })
    with OUT.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run, job): job for job in jobs}
            for index, future in enumerate(as_completed(futures), 1):
                writer.writerow({**futures[future], **future.result()})
                stream.flush()
                if index % 24 == 0 or index == len(jobs):
                    print(f"  {index}/{len(jobs)}", flush=True)
    print(OUT)


if __name__ == "__main__":
    main()
