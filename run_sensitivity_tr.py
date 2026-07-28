#!/usr/bin/env python3
"""End-to-end NAND-read-latency sensitivity for the paper.

This study changes one physical parameter: per-page NAND read latency tR.
The aggregate HBF bandwidth is derived from geometry and then capped by the
8 TB/s GPU-facing path:

  BW = min(8 TB/s, 8 stacks * 1024 planes/stack * 4096 B / tR).

TP is fixed within each model to isolate the memory parameter: TP=8 for the
two Llama models and TP=4 for Qwen3-235B, whose four KV heads cap TP at four.
For every (model, context, tR, system), batch 1..128 is swept and the paper's
100 ms TPOT-p50 throughput/GPU operating point is selected by the plotter.
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
    TR_SENSITIVITY_NS,
    capacity_feasible,
    hbm_kv_fraction,
)
from sweep_capacity import weight_bytes

ROOT = Path(__file__).parent
PY = sys.executable
RUNNER = str(ROOT / "run_point_hbf.py")
TOML = str(ROOT / "configs/hbf_paper.toml")
CACHE = str(ROOT / "results" / "predictor_cache_eval")
OUT = ROOT / "results" / "sensitivity_tr.csv"

MODELS = (
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-70B",
    "Qwen/Qwen3-235B-A22B",
)
CONTEXTS = (262144, 1048576, 2097152)
MODEL_TP = {
    "meta-llama/Meta-Llama-3-8B": 8,
    "meta-llama/Meta-Llama-3-70B": 8,
    "Qwen/Qwen3-235B-A22B": 4,
}
SYSTEMS = (
    ("H3", 1.0),
    ("SPLASH", PLATFORM.sparse_kv_fraction),
)
FIELDS = (
    "model", "device", "tp", "batch", "context_length", "system",
    "sparsity_fraction", "hbm_kv_fraction", "weights_hbm_gb_per_gpu",
    "hbf_stacks_per_gpu", "planes_per_stack", "page_bytes", "tr_ns",
    "media_bw_gbps", "host_bw_cap_gbps", "effective_hbf_bw_gbps",
    "tpot_p50_ms", "tpot_p99_ms", "status", "wall_s",
)


def run_point(job: dict, timeout_s: int) -> dict:
    cmd = [
        PY, RUNNER,
        "--model", job["model"],
        "--device", PLATFORM.device,
        "--batch_size", str(job["batch"]),
        "--context_length", str(job["context_length"]),
        "--decode_tokens", "4",
        "--num_requests", str(job["batch"]),
        "--tensor_parallel_size", str(job["tp"]),
        "--sparsity_fraction", str(job["sparsity_fraction"]),
        "--hbm_kv_fraction", str(job["hbm_kv_fraction"]),
        "--backing_bw_gbps", str(job["effective_hbf_bw_gbps"]),
        "--hbfsim_toml", TOML,
        "--cache_dir", CACHE,
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


def prewarm(models: tuple[str, ...]) -> None:
    """Populate predictor tables serially before parallel read-only use."""
    for model in models:
        tp = MODEL_TP[model]
        if not capacity_feasible(model, 1, CONTEXTS[0], tp, "hbf"):
            continue
        hot = hbm_kv_fraction(model, 1, CONTEXTS[0], tp)
        job = {
            "model": model,
            "tp": tp,
            "batch": 1,
            "context_length": CONTEXTS[0],
            "sparsity_fraction": 1.0,
            "hbm_kv_fraction": hot,
            "effective_hbf_bw_gbps": PLATFORM.effective_hbf_bw_gbps(),
        }
        result = run_point(job, 900)
        if result["status"] != "OK":
            raise RuntimeError(f"predictor prewarm failed for {model}: {result}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    prewarm(MODELS)
    done = set()
    if args.resume and OUT.exists():
        for r in csv.DictReader(OUT.open()):
            if r["status"] == "OK":
                done.add((
                    r["model"], int(r["batch"]), int(r["context_length"]),
                    r["system"], float(r["tr_ns"]),
                ))

    jobs = []
    for model in MODELS:
        tp = MODEL_TP[model]
        for context in CONTEXTS:
            for batch in BATCHES:
                if not capacity_feasible(model, batch, context, tp, "hbf"):
                    continue
                hot = hbm_kv_fraction(model, batch, context, tp)
                for tr_ns in TR_SENSITIVITY_NS:
                    media_bw = PLATFORM.media_bw_gbps(tr_ns)
                    effective_bw = PLATFORM.effective_hbf_bw_gbps(tr_ns)
                    for system, sparsity in SYSTEMS:
                        key = (model, batch, context, system, tr_ns)
                        if key in done:
                            continue
                        jobs.append({
                            "model": model,
                            "device": PLATFORM.device,
                            "tp": tp,
                            "batch": batch,
                            "context_length": context,
                            "system": system,
                            "sparsity_fraction": sparsity,
                            "hbm_kv_fraction": f"{hot:.8f}",
                            "weights_hbm_gb_per_gpu": (
                                f"{weight_bytes(model) / tp / 1e9:.4f}"
                            ),
                            "hbf_stacks_per_gpu": PLATFORM.hbf_stacks_per_gpu,
                            "planes_per_stack": PLATFORM.hbf_planes_per_stack,
                            "page_bytes": PLATFORM.hbf_page_bytes,
                            "tr_ns": tr_ns,
                            "media_bw_gbps": f"{media_bw:.4f}",
                            "host_bw_cap_gbps": PLATFORM.hbf_host_bw_cap_gbps,
                            "effective_hbf_bw_gbps": f"{effective_bw:.4f}",
                        })

    mode = "a" if args.resume and OUT.exists() else "w"
    with OUT.open(mode, newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if mode == "w":
            writer.writeheader()
        print(f"sensitivity points: {len(jobs)}", flush=True)
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
    print(f"SENSITIVITY DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
