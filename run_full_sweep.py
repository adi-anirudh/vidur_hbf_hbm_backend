#!/usr/bin/env python3
"""
Full-factorial HBF paper sweep: 8 models x 4 GPUs x 8 batch x 9 ctx = 2304
points, each run for Dense (sparsity=1.0) and Sparse (sparsity=0.1).

For every point:
  * pick min TP in {1,2,4,8} from combined HBM+HBF capacity (sweep_capacity);
    INFEASIBLE if even TP=8 doesn't fit.
  * require a real compute trace for (model, device); else NO_TRACE.
  * otherwise run the in-repo hbf_linear_regression predictor (run_point_hbf.py)
    at the chosen TP and record TPOT (dense + sparse).

Results stream to results/full_sweep.csv (resumable: existing rows are skipped).
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from sweep_capacity import select_tp

ROOT = Path(__file__).parent
PY = sys.executable
RUNNER = str(ROOT / "run_point_hbf.py")
TOML = str(ROOT / "configs/hbf_paper.toml")
OUT = ROOT / "results" / "full_sweep.csv"

MODELS = [
    "microsoft/phi-2",
    "mistralai/Mistral-7B-v0.1",
    "meta-llama/Meta-Llama-3-8B",
    "mistralai/Mixtral-8x7B-v0.1",
    "deepseek-ai/deepseek-llm-67b-chat",
    "meta-llama/Meta-Llama-3-70B",
    "Qwen/Qwen-72B",
    "Qwen/Qwen2-72B",
]
DEVICES = ["a40", "a100", "h100", "h200"]
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]
CONTEXTS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
BASELINES = [("Dense", 1.0), ("Sparse", 0.1)]

DECODE_TOKENS = 4
NUM_REQUESTS = 8

FIELDS = ["model", "device", "batch", "context_length", "baseline",
          "tp", "footprint_gb", "sparsity_fraction",
          "tpot_p50_ms", "tpot_p99_ms", "status", "wall_s"]


def has_trace(model: str, device: str) -> bool:
    d = ROOT / "data" / "profiling" / "compute" / device / model
    return (d / "mlp.csv").exists() and (d / "attention.csv").exists()


def run_point(model, device, batch, ctx, tp, label, sparsity, timeout_s):
    cmd = [PY, RUNNER, "--model", model, "--device", device,
           "--batch_size", str(batch), "--context_length", str(ctx),
           "--decode_tokens", str(DECODE_TOKENS), "--num_requests", str(NUM_REQUESTS),
           "--tensor_parallel_size", str(tp), "--sparsity_fraction", str(sparsity),
           "--hbfsim_toml", TOML]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, cwd=str(ROOT))
        wall = time.perf_counter() - t0
        m = re.search(r"RESULT status=OK tpot_p50_ms=([\d.]+) tpot_p99_ms=([\d.]+)", p.stdout)
        if m:
            return dict(tpot_p50_ms=m.group(1), tpot_p99_ms=m.group(2),
                        status="OK", wall_s=f"{wall:.1f}")
        return dict(tpot_p50_ms="", tpot_p99_ms="",
                    status=f"FAIL({p.returncode})", wall_s=f"{wall:.1f}")
    except subprocess.TimeoutExpired:
        return dict(tpot_p50_ms="", tpot_p99_ms="", status=f"TIMEOUT>{timeout_s}s", wall_s="")


def job_key(r):
    return (r["model"], r["device"], int(r["batch"]), int(r["context_length"]), r["baseline"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=1800)
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if OUT.exists():
        with open(OUT) as f:
            for r in csv.DictReader(f):
                if r.get("status") not in (None, "", "PENDING"):
                    done.add(job_key(r))
    new_file = not OUT.exists()
    fh = open(OUT, "a", newline="")
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    if new_file:
        w.writeheader(); fh.flush()

    # Build job list; resolve TP/feasibility/trace up front (cheap), then only
    # dispatch sim subprocesses for runnable points.
    runnable, n_skip = [], 0
    for model in MODELS:
        for device in DEVICES:
            trace = has_trace(model, device)
            for batch in BATCHES:
                for ctx in CONTEXTS:
                    tp, foot, ok = select_tp(model, device, batch, ctx)
                    for label, sp in BASELINES:
                        base = dict(model=model, device=device, batch=batch,
                                    context_length=ctx, baseline=label, tp=tp,
                                    footprint_gb=f"{foot:.1f}", sparsity_fraction=sp,
                                    tpot_p50_ms="", tpot_p99_ms="", wall_s="")
                        if job_key({**base, "batch": batch, "context_length": ctx}) in done:
                            continue
                        if not ok:
                            w.writerow({**base, "status": "INFEASIBLE"}); n_skip += 1
                        elif not trace:
                            w.writerow({**base, "status": "NO_TRACE"}); n_skip += 1
                        else:
                            runnable.append(base)
    fh.flush()
    print(f"runnable sim points: {len(runnable)}   pre-marked (infeasible/no-trace/done): {n_skip}")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_point, b["model"], b["device"], b["batch"],
                          b["context_length"], b["tp"], b["baseline"],
                          b["sparsity_fraction"], args.timeout): b
                for b in runnable}
        n = 0
        for fut in as_completed(futs):
            b = futs[fut]; res = fut.result()
            w.writerow({**b, **res}); fh.flush()
            n += 1
            if n % 20 == 0:
                print(f"  {n}/{len(runnable)} done")
    fh.close()
    print(f"sweep complete -> {OUT}")


if __name__ == "__main__":
    main()
