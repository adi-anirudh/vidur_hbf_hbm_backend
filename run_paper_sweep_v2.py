#!/usr/bin/env python3.10
"""
Comprehensive HBF paper sweep.

Runs all (model, device, baseline, batch_size, context_length) combinations,
filtered by feasibility (wall-clock drain estimate < MAX_DRAIN_WALL_S seconds).

Results saved to results/paper_sweep_<timestamp>.csv  (incremental, safe to Ctrl-C).
"""

import csv
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

PYTHON = "/home/yaswanth/miniconda3/envs/scissors/bin/python3.10"
RUNNER = str(Path(__file__).parent / "run_hbf_simulation.py")
TOML   = str(Path(__file__).parent / "configs/hbf_paper.toml")

# ---------------------------------------------------------------------------
# Sweep dimensions
# ---------------------------------------------------------------------------

MODELS = [
    "microsoft/phi-2",
    "mistralai/Mistral-7B-v0.1",
    "meta-llama/Meta-Llama-3-8B",
    "mistralai/Mixtral-8x7B-v0.1",
    "internlm/internlm-20b",
    "deepseek-ai/deepseek-llm-67b-chat",
    "meta-llama/Llama-2-70b-hf",
    "meta-llama/Meta-Llama-3-70B",
    "Qwen/Qwen2-72B",
    "Qwen/Qwen-72B",
    "meta-llama/Meta-Llama-3.1-405B",
    "codellama/CodeLlama-34b-Instruct-hf",
    "meta-llama/Llama-2-7b-hf",
]

DEVICES = ["a100", "h100", "a40"]

BASELINES = [
    # (label, kv_read_fraction)
    # Dense and NaiveSparse share kv_frac=1.0; the runner deduplicates them.
    ("Dense",       1.0),
    ("NaiveSparse", 1.0),
    ("Sparse",      0.1),
]

# Unique kv fractions to actually simulate (Dense/NaiveSparse share kv_frac=1.0)
_UNIQUE_FRACS = sorted(set(f for _, f in BASELINES), reverse=True)  # [1.0, 0.1]

BATCH_SIZES = [4, 8, 16, 32, 64, 128]

CONTEXT_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
# 10M excluded: ~42 min/drain at best, completely infeasible

NUM_REQUESTS = 8    # per simulation (gives 8×decode_tokens decode events)
DECODE_TOKENS = 4   # per request

N_WORKERS = 8       # parallel simulations

# Seconds per drain call above which we skip the combination.
MAX_DRAIN_WALL_S = 60.0

# ---------------------------------------------------------------------------
# Model KV geometry (used for feasibility + max_drain_cycles)
# ---------------------------------------------------------------------------

MODEL_INFO = {
    "microsoft/phi-2":                    dict(n_kv=32, h_dim=80,  n_lay=32),
    "mistralai/Mistral-7B-v0.1":          dict(n_kv=8,  h_dim=128, n_lay=32),
    "meta-llama/Meta-Llama-3-8B":         dict(n_kv=8,  h_dim=128, n_lay=32),
    "mistralai/Mixtral-8x7B-v0.1":        dict(n_kv=8,  h_dim=128, n_lay=32),
    "internlm/internlm-20b":              dict(n_kv=40, h_dim=128, n_lay=48),
    "deepseek-ai/deepseek-llm-67b-chat":  dict(n_kv=8,  h_dim=128, n_lay=95),
    "meta-llama/Llama-2-70b-hf":          dict(n_kv=8,  h_dim=128, n_lay=80),
    "meta-llama/Meta-Llama-3-70B":        dict(n_kv=8,  h_dim=128, n_lay=80),
    "Qwen/Qwen2-72B":                     dict(n_kv=8,  h_dim=128, n_lay=80),
    "Qwen/Qwen-72B":                      dict(n_kv=64, h_dim=128, n_lay=80),
    "meta-llama/Meta-Llama-3.1-405B":     dict(n_kv=8,  h_dim=128, n_lay=126),
    "codellama/CodeLlama-34b-Instruct-hf":dict(n_kv=8,  h_dim=128, n_lay=48),
    "meta-llama/Llama-2-7b-hf":           dict(n_kv=32, h_dim=128, n_lay=32),
}

BLOCK_SIZE_TOKENS = 16
HBF_BW_GBps      = 768.0
FREQ_GHZ          = 1.0

# Empirical: ~0.49 μs per NAND-page event in C++ simulator
# Derived from: Llama-2-7b (2048 pages/block) → ~1ms per packet
COST_PER_NAND_PAGE_US = 0.49


def block_bytes(model: str) -> int:
    info = MODEL_INFO[model]
    per_tok = 2 * info["n_kv"] * info["h_dim"] * 2   # K+V, FP16
    return BLOCK_SIZE_TOKENS * per_tok * info["n_lay"]


def nand_pages_per_block(model: str) -> int:
    return block_bytes(model) // 4096


def estimate_drain_wall_ms(model: str, ctx: int, batch: int, kv_frac: float) -> float:
    n_blocks = math.ceil(ctx / BLOCK_SIZE_TOKENS)
    n_packets = int(batch * n_blocks * kv_frac)
    return n_packets * nand_pages_per_block(model) * COST_PER_NAND_PAGE_US / 1000.0


def max_drain_cycles(model: str, ctx: int, batch: int, kv_frac: float) -> int:
    n_blocks = math.ceil(ctx / BLOCK_SIZE_TOKENS)
    total_bytes = batch * n_blocks * block_bytes(model) * kv_frac
    flash_cycles = int(total_bytes / (HBF_BW_GBps * 1e9) * FREQ_GHZ * 1e9)
    return max(1_000_000_000, flash_cycles * 3)   # 3× safety margin


def is_feasible(model: str, ctx: int, batch: int, kv_frac: float) -> bool:
    # Dense (kv_frac=1.0) is the worst case; Sparse is 10× faster.
    wall_dense = estimate_drain_wall_ms(model, ctx, batch, 1.0)
    return wall_dense / 1000.0 <= MAX_DRAIN_WALL_S


# ---------------------------------------------------------------------------
# Generate combinations
# ---------------------------------------------------------------------------

def generate_jobs():
    """Generate simulation jobs, deduplicating Dense/NaiveSparse (same kv_frac).

    Each job has an 'emit_labels' list — labels to write to CSV for that run.
    This avoids running the same simulation twice for Dense and NaiveSparse.
    """
    # Map kv_frac → list of baseline labels that share it
    frac_to_labels: dict = {}
    for label, frac in BASELINES:
        frac_to_labels.setdefault(frac, []).append(label)

    jobs = []
    for model in MODELS:
        if model not in MODEL_INFO:
            continue
        for device in DEVICES:
            org, mname = model.split("/", 1)
            data_dir = Path(f"data/profiling/compute/{device}/{org}/{mname}")
            if not data_dir.exists():
                continue
            for ctx in CONTEXT_LENGTHS:
                for batch in BATCH_SIZES:
                    if not is_feasible(model, ctx, batch, 1.0):
                        continue
                    for kv_frac, labels in frac_to_labels.items():
                        mdc = max_drain_cycles(model, ctx, batch, kv_frac)
                        jobs.append(dict(
                            model=model,
                            device=device,
                            emit_labels=labels,          # may be ["Dense","NaiveSparse"]
                            context_length=ctx,
                            batch_size=batch,
                            kv_read_fraction=kv_frac,
                            max_drain_cycles=mdc,
                        ))

    # Sort for cache locality: same (model, device, ctx) → same RF model hash
    jobs.sort(key=lambda j: (j["model"], j["device"], j["context_length"], j["batch_size"], j["kv_read_fraction"]))
    return jobs


# ---------------------------------------------------------------------------
# Run one simulation, return metrics dict
# ---------------------------------------------------------------------------

def run_one(job: dict, timeout_s: int = 4200) -> list:
    """Run one simulation. Returns a list of result dicts (one per emit_label)."""
    cmd = [
        PYTHON, RUNNER,
        "--model",            job["model"],
        "--device",           job["device"],
        "--context_length",   str(job["context_length"]),
        "--batch_size",       str(job["batch_size"]),
        "--num_requests",     str(NUM_REQUESTS),
        "--decode_tokens",    str(DECODE_TOKENS),
        "--decode_only",
        "--kv_read_fraction", str(job["kv_read_fraction"]),
        "--hbm_kv_fraction",  "0.0",
        "--max_drain_cycles", str(job["max_drain_cycles"]),
        "--hbfsim_toml",      TOML,
        "--qps",              "1000",
    ]

    base = dict(
        model=job["model"],
        device=job["device"],
        context_length=job["context_length"],
        batch_size=job["batch_size"],
        kv_read_fraction=job["kv_read_fraction"],
        ttft_p50_ms="N/A", ttft_p99_ms="N/A",
        tpot_p50_ms="N/A", tpot_p99_ms="N/A",
        throughput_tok_s="N/A",
        hbf_stall_ms="N/A",
        status="PENDING",
        wall_s="",
    )

    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True, text=True,
            timeout=timeout_s,
            cwd=str(Path(__file__).parent),
        )
        wall = time.perf_counter() - t0
        base["wall_s"] = f"{wall:.1f}"

        out = proc.stdout + proc.stderr
        if proc.returncode != 0:
            base["status"] = f"FAIL({proc.returncode})"
        else:
            def _grab(pattern, text):
                m = re.search(pattern, text)
                return m.group(1).strip() if m else "N/A"

            base["ttft_p50_ms"]      = _grab(r"TTFT p50 \(ms\)\s+([\d.\-]+)", out)
            base["ttft_p99_ms"]      = _grab(r"TTFT p99 \(ms\)\s+([\d.\-]+)", out)
            base["tpot_p50_ms"]      = _grab(r"TPOT p50 \(ms\)\s+([\d.\-]+)", out)
            base["tpot_p99_ms"]      = _grab(r"TPOT p99 \(ms\)\s+([\d.\-]+)", out)
            base["throughput_tok_s"] = _grab(r"Throughput \(tok/s\)\s+([\d.\-]+)", out)
            base["hbf_stall_ms"]     = _grab(r"HBF stall total \(ms\)\s+([\d.\-]+)", out)
            base["status"] = "OK"
    except subprocess.TimeoutExpired:
        base["status"] = f"TIMEOUT>{timeout_s}s"
        base["wall_s"] = f">{timeout_s}"
    except Exception as e:
        base["status"] = f"ERROR:{e}"

    # Fan-out to one row per emit_label
    rows = []
    for label in job["emit_labels"]:
        rows.append({**base, "baseline": label})
    return rows


# ---------------------------------------------------------------------------
# Parallel runner
# ---------------------------------------------------------------------------

FIELDNAMES = [
    "model", "device", "baseline", "context_length", "batch_size", "kv_read_fraction",
    "ttft_p50_ms", "ttft_p99_ms", "tpot_p50_ms", "tpot_p99_ms",
    "throughput_tok_s", "hbf_stall_ms",
    "status", "wall_s",
]


def run_sweep(jobs, out_csv: str, n_workers: int = N_WORKERS):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    write_header = not os.path.exists(out_csv)
    csv_lock = threading.Lock()

    job_queue = queue.Queue()
    for j in jobs:
        job_queue.put(j)

    total = len(jobs)
    done_count = [0]
    ok_count   = [0]

    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
            f.flush()

        def worker():
            while True:
                try:
                    job = job_queue.get_nowait()
                except queue.Empty:
                    return
                rows = run_one(job)
                with csv_lock:
                    done_count[0] += 1
                    status = rows[0]["status"] if rows else "ERROR"
                    if status == "OK":
                        ok_count[0] += 1
                    for row in rows:
                        writer.writerow(row)
                    f.flush()
                    pct = 100 * done_count[0] / total
                    short = f"{job['model'].split('/')[-1]}/{job['device']}/ctx={job['context_length']//1024}K/bs={job['batch_size']}/frac={job['kv_read_fraction']}"
                    first = rows[0] if rows else {}
                    print(f"[{done_count[0]:4d}/{total}] {pct:5.1f}%  {status:18s}  {short}  "
                          f"TPOT={first.get('tpot_p50_ms','N/A'):>8}ms  stall={first.get('hbf_stall_ms','N/A'):>8}ms",
                          flush=True)
                job_queue.task_done()

        threads = [threading.Thread(target=worker, daemon=True) for _ in range(n_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    print(f"\nDone. {ok_count[0]}/{total} succeeded. Results → {out_csv}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, datetime

    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--workers", type=int, default=N_WORKERS)
    ap.add_argument("--dry_run", action="store_true",
                    help="Print jobs without running them")
    ap.add_argument("--models",  nargs="*", default=None,
                    help="Subset of models to run (substring match)")
    ap.add_argument("--devices", nargs="*", default=None)
    ap.add_argument("--ctx_max", type=int, default=0,
                    help="Skip contexts larger than this (0 = no limit)")
    ap.add_argument("--resume", default=None,
                    help="Path to existing CSV; skip jobs already marked OK in it.")
    args_cli = ap.parse_args()

    os.chdir(str(Path(__file__).parent))

    jobs = generate_jobs()

    # Build set of already-completed (model, device, kv_frac, ctx, batch) keys.
    completed_keys: set = set()
    if args_cli.resume and os.path.exists(args_cli.resume):
        import csv as _csv
        with open(args_cli.resume) as _f:
            for _row in _csv.DictReader(_f):
                if _row.get("status") == "OK":
                    completed_keys.add((
                        _row["model"], _row["device"],
                        _row["kv_read_fraction"],
                        _row["context_length"], _row["batch_size"],
                    ))
        print(f"Resuming: {len(completed_keys)} sim-runs already OK in {args_cli.resume}")

    if args_cli.models:
        jobs = [j for j in jobs
                if any(m in j["model"] for m in args_cli.models)]
    if args_cli.devices:
        jobs = [j for j in jobs if j["device"] in args_cli.devices]
    if args_cli.ctx_max:
        jobs = [j for j in jobs if j["context_length"] <= args_cli.ctx_max]

    if completed_keys:
        before = len(jobs)
        jobs = [j for j in jobs
                if (j["model"], j["device"],
                    str(j["kv_read_fraction"]),
                    str(j["context_length"]), str(j["batch_size"]))
                not in completed_keys]
        print(f"Skipping {before - len(jobs)} already-OK runs.")

    n_rows = sum(len(j["emit_labels"]) for j in jobs)
    print(f"Total simulation runs: {len(jobs)}  (→ {n_rows} CSV rows after label fan-out)")

    # Estimate total wall time
    total_wall_est = 0.0
    for j in jobs:
        total_wall_est += estimate_drain_wall_ms(
            j["model"], j["context_length"], j["batch_size"], j["kv_read_fraction"]
        ) * DECODE_TOKENS / 1000.0   # seconds
    parallel_est = total_wall_est / args_cli.workers
    print(f"Estimated wall time: {parallel_est/3600:.1f}h with {args_cli.workers} workers "
          f"(sequential: {total_wall_est/3600:.1f}h)")

    if args_cli.dry_run:
        from collections import defaultdict
        per_model = defaultdict(lambda: {"sims": 0, "rows": 0})
        for j in jobs:
            per_model[j["model"]]["sims"] += 1
            per_model[j["model"]]["rows"] += len(j["emit_labels"])
        print(f"{'Model':<55} {'sims':>5} {'rows':>5}")
        for m in sorted(per_model):
            print(f"  {m:<53} {per_model[m]['sims']:>5} {per_model[m]['rows']:>5}")
        sys.exit(0)

    if args_cli.resume:
        out_csv = args_cli.resume
    else:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join(args_cli.out_dir, f"paper_sweep_{ts}.csv")
    print(f"Writing to: {out_csv}\n")

    run_sweep(jobs, out_csv, n_workers=args_cli.workers)
