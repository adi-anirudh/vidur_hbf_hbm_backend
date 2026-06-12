#!/usr/bin/env python3.9
"""
Multi-GPU sweep to fill in configs infeasible on single GPU due to KV overflow.

Determines minimum TP (2, 4, 8) for each missing (model, device, batch, ctx),
simulates only those jobs, writes paper_results_multigpu.csv with a num_gpus column.

Usage:
    python3.9 run_multigpu_sweep.py [--workers N] [--resume] [--dry-run]
"""

import argparse
import csv
import glob
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# ---------------------------------------------------------------------------
# Hardware / model constants — must match run_paper_sweep.py
# ---------------------------------------------------------------------------

GPU_MEM_GB = {"a100": 80.0, "h100": 80.0, "a40": 45.0, "h200": 141.0}
GPU_OVERHEAD_GB = 2.0
HBF_CAPACITY_GB = 206.0   # 768 planes × 256 blocks × 256 pages × 4 KB ≈ 206 GB

MODEL_INFO = {
    "microsoft/phi-2":                   dict(layers=32,  kv_heads=32, head_dim=80,  weights_gb=5.4),
    "mistralai/Mistral-7B-v0.1":         dict(layers=32,  kv_heads=8,  head_dim=128, weights_gb=14.0),
    "meta-llama/Meta-Llama-3-8B":        dict(layers=32,  kv_heads=8,  head_dim=128, weights_gb=16.0),
    "mistralai/Mixtral-8x7B-v0.1":       dict(layers=32,  kv_heads=8,  head_dim=128, weights_gb=93.0),
    "deepseek-ai/deepseek-llm-67b-chat": dict(layers=95,  kv_heads=8,  head_dim=128, weights_gb=134.0),
}

# Devices where each model fits on a SINGLE GPU (TP=1 baseline)
SINGLE_GPU_DEVICES = {
    "microsoft/phi-2":                   ["a100", "h100", "a40", "h200"],
    "mistralai/Mistral-7B-v0.1":         ["a100", "h100", "a40", "h200"],
    "meta-llama/Meta-Llama-3-8B":        ["a100", "h100", "a40", "h200"],
    "mistralai/Mixtral-8x7B-v0.1":       ["h200"],
    "deepseek-ai/deepseek-llm-67b-chat": ["h200"],
}

BATCHES       = [1, 2, 4, 8, 16, 32, 64, 128]
CTX_POINTS    = [4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144, 524_288, 1_048_576]
DECODE_TOKENS = 8
BLOCK_SIZE    = 16
SPARSITY      = 0.10
BASELINES     = ["Dense", "NaiveSparse", "Sparse"]

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_hbf_simulation.py")
TOML   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs/hbf_paper.toml")
PYTHON = "/home/yaswanth/miniconda3/envs/scissors/bin/python3.10"


# ---------------------------------------------------------------------------
# KV / feasibility helpers
# ---------------------------------------------------------------------------

def kv_per_token_bytes(model):
    p = MODEL_INFO[model]
    return 2 * p["kv_heads"] * p["head_dim"] * 2   # FP16 K+V per layer per token

def total_kv_gb(model, batch, ctx):
    p = MODEL_INFO[model]
    return batch * ctx * kv_per_token_bytes(model) * p["layers"] / 1e9

def avail_hbm_per_gpu(model, device, tp):
    """HBM available for KV on one GPU given TP."""
    return max(0.0, GPU_MEM_GB[device] - MODEL_INFO[model]["weights_gb"] / tp - GPU_OVERHEAD_GB)

def hbf_needed_per_gpu(model, device, batch, ctx, tp):
    kv_per_gpu = total_kv_gb(model, batch, ctx) / tp
    avail = avail_hbm_per_gpu(model, device, tp)
    return max(0.0, kv_per_gpu - avail)

def compute_hbm_frac(model, device, batch, ctx, tp):
    avail = avail_hbm_per_gpu(model, device, tp)
    kv_per_gpu = total_kv_gb(model, batch, ctx) / tp
    if kv_per_gpu <= 0:
        return 1.0
    return min(1.0, avail / kv_per_gpu)

def min_tp_needed(model, device, batch, ctx):
    """Minimum TP (power-of-2) that makes this config feasible. Returns None if TP=8 not enough."""
    for tp in [2, 4, 8]:
        if hbf_needed_per_gpu(model, device, batch, ctx, tp) <= HBF_CAPACITY_GB:
            return tp
    return None


# ---------------------------------------------------------------------------
# Find missing configs
# ---------------------------------------------------------------------------

def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for r in csv.DictReader(f):
            done.add((r["model"], r["device"], str(r["batch"]), str(r["ctx"]),
                      r["baseline"], str(r.get("num_gpus", "1"))))
    return done

def find_missing(single_gpu_csv, multigpu_csv):
    """Return list of (model, device, batch, ctx, tp) that need multi-GPU simulation."""
    # What we already have in single-GPU sweep
    have_single = set()
    if os.path.exists(single_gpu_csv):
        with open(single_gpu_csv) as f:
            for r in csv.DictReader(f):
                if r.get("tpot_p50") and "FAILED" not in str(r.get("note", "")):
                    have_single.add((r["model"], r["device"], str(r["batch"]), str(r["ctx"])))

    # What we already have in multi-GPU sweep
    have_multi = set()
    if os.path.exists(multigpu_csv):
        with open(multigpu_csv) as f:
            for r in csv.DictReader(f):
                if r.get("tpot_p50") and "FAILED" not in str(r.get("note", "")):
                    have_multi.add((r["model"], r["device"], str(r["batch"]),
                                    str(r["ctx"]), str(r.get("num_gpus", "?"))))

    missing = []
    for model, devices in SINGLE_GPU_DEVICES.items():
        info = MODEL_INFO[model]
        for device in devices:
            for ctx in CTX_POINTS:
                for batch in BATCHES:
                    # Already covered by single-GPU sweep?
                    if (model, device, str(batch), str(ctx)) in have_single:
                        continue
                    # Single-GPU feasible but just not run yet? Skip — that's run_paper_sweep's job.
                    hbf1 = hbf_needed_per_gpu(model, device, batch, ctx, tp=1)
                    weights_fit = info["weights_gb"] + GPU_OVERHEAD_GB <= GPU_MEM_GB[device]
                    if weights_fit and hbf1 <= HBF_CAPACITY_GB:
                        continue  # single-GPU feasible, not our job
                    # Single-GPU infeasible — find min TP
                    tp = min_tp_needed(model, device, batch, ctx)
                    if tp is None:
                        continue  # even TP=8 not enough → skip
                    if (model, device, str(batch), str(ctx), str(tp)) in have_multi:
                        continue
                    missing.append((model, device, batch, ctx, tp))
    return missing


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def _run_one(job):
    model, device, batch, ctx, baseline, tp, job_id = job

    hbm_frac = compute_hbm_frac(model, device, batch, ctx, tp)
    num_kv   = math.ceil(batch * math.ceil(ctx / BLOCK_SIZE) / 0.99) + 64

    if baseline == "Dense":
        kv_read_fraction = 1.0
    elif baseline == "NaiveSparse":
        kv_read_fraction = 1.0
    else:
        kv_read_fraction = SPARSITY

    cmd = [
        PYTHON, SCRIPT,
        "--model",                  model,
        "--device",                 device,
        "--batch_size",             str(batch),
        "--context_length",         str(ctx),
        "--decode_tokens",          str(DECODE_TOKENS),
        "--qps",                    "1000",
        "--decode_only",
        "--num_requests",           str(batch),
        "--kv_read_fraction",       str(kv_read_fraction),
        "--hbm_kv_fraction",        str(hbm_frac),
        "--num_kv_blocks",          str(num_kv),
        "--hbfsim_toml",            TOML,
        "--tensor_parallel_size",   str(tp),
    ]

    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True,
                         cwd=os.path.dirname(SCRIPT))
    elapsed = time.time() - t0

    out_dir = None
    flash_stall_ms = None
    for line in (res.stdout or "").splitlines():
        if "Output:" in line:
            out_dir = line.split("Output:")[-1].strip()
        if "HBF stall total" in line:
            try:
                flash_stall_ms = float(line.split()[-1])
            except (ValueError, IndexError):
                pass

    metrics = _parse_csv(out_dir) if out_dir else None
    ok = metrics is not None and res.returncode == 0
    if not ok:
        err = (res.stderr or res.stdout or "").strip()[-300:]
        metrics = dict(ttft_p50=None, ttft_p99=None, tpot_p50=None,
                       tpot_p99=None, throughput_tps=None)
        note = f"FAILED: {err}"
    else:
        note = f"sim {elapsed:.0f}s TP={tp}"

    return dict(
        model=model, device=device, batch=batch, ctx=ctx, baseline=baseline,
        num_gpus=tp,
        hbm_frac=round(hbm_frac, 4), simulated=True,
        **metrics, flash_stall_ms=flash_stall_ms, note=note,
    )


def _parse_csv(output_dir):
    if not output_dir:
        return None
    csvs = glob.glob(f"{output_dir}/**/request_metrics.csv", recursive=True)
    if not csvs:
        return None
    with open(csvs[0]) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None

    def pct(col, scale=1000):
        vals = sorted(float(r[col]) for r in rows if r.get(col, ""))
        if not vals:
            return None, None
        n = len(vals)
        return vals[n // 2] * scale, vals[min(n-1, int(n*0.99))] * scale

    ttft_p50, ttft_p99 = pct("prefill_e2e_time")
    tpot_p50, tpot_p99 = pct("decode_time_execution_plus_preemption_normalized")

    tput = None
    try:
        by_id = sorted(rows, key=lambda r: int(r["Request Id"]))
        t = 0.0; arrivals = []
        for r in by_id:
            t += float(r.get("request_inter_arrival_delay") or 0)
            arrivals.append(t)
        comps = [a + float(r["request_e2e_time"]) for a, r in zip(arrivals, by_id)]
        dur   = max(comps) - min(arrivals)
        dtoks = sum(float(r.get("request_num_decode_tokens", 0)) for r in rows)
        tput  = dtoks / dur if dur > 0 else None
    except Exception:
        pass

    return dict(ttft_p50=ttft_p50, ttft_p99=ttft_p99,
                tpot_p50=tpot_p50, tpot_p99=tpot_p99,
                throughput_tps=tput)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers",  type=int, default=4)
    ap.add_argument("--resume",   action="store_true")
    ap.add_argument("--dry-run",  action="store_true",
                    help="Print jobs that would run without executing them")
    args = ap.parse_args()

    single_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_results.csv")
    out_csv    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_results_multigpu.csv")

    missing = find_missing(single_csv, out_csv)
    print(f"Missing multi-GPU configs to fill: {len(missing)}")

    # Count by TP tier
    from collections import Counter
    by_tp = Counter(tp for _, _, _, _, tp in missing)
    for tp in sorted(by_tp):
        print(f"  TP={tp}: {by_tp[tp]} (model,device,batch,ctx) combos × 3 baselines = {by_tp[tp]*3} jobs")

    if args.dry_run:
        print("\nDry run — jobs that would execute:")
        for model, device, batch, ctx, tp in sorted(missing)[:40]:
            hbf = hbf_needed_per_gpu(model, device, batch, ctx, tp)
            print(f"  {model.split('/')[-1]:20} {device:5} B={batch:3} ctx={ctx//1024:5}K  TP={tp}  hbf/GPU={hbf:.1f}GB")
        if len(missing) > 40:
            print(f"  ... and {len(missing)-40} more")
        return

    # Build sim jobs (Dense only when all-HBM; copy others)
    sim_jobs  = []
    copy_rows = []
    seen      = set()

    done_keys = set()
    existing  = []
    if args.resume and os.path.exists(out_csv):
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                existing.append(row)
                done_keys.add((row["model"], row["device"],
                               str(row["batch"]), str(row["ctx"]),
                               row["baseline"], str(row.get("num_gpus", ""))))

    for model, device, batch, ctx, tp in missing:
        hbm_frac   = compute_hbm_frac(model, device, batch, ctx, tp)
        all_in_hbm = hbm_frac >= 1.0
        for bl in BASELINES:
            key = (model, device, str(batch), str(ctx), bl, str(tp))
            if key in seen or key in done_keys:
                continue
            seen.add(key)
            if all_in_hbm and bl != "Dense":
                copy_rows.append((model, device, batch, ctx, bl, tp))
            else:
                sim_jobs.append((model, device, batch, ctx, bl, tp, len(sim_jobs)))

    print(f"\nSim jobs: {len(sim_jobs)}  |  Copy-from-Dense: {len(copy_rows)}")
    print(f"Workers:  {args.workers}\n")

    fields = ["model", "device", "num_gpus", "batch", "ctx", "baseline",
              "hbm_frac", "simulated",
              "ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99",
              "throughput_tps", "flash_stall_ms", "note"]

    rows = list(existing)
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, j): j for j in sim_jobs}
        for fut in as_completed(futures):
            job = futures[fut]
            model, device, batch, ctx, baseline, tp, _ = job
            label = f"{model.split('/')[-1]:20} {device} TP={tp} B={batch:3} ctx={ctx//1024}K {baseline}"
            try:
                row    = fut.result()
                status = "OK" if row.get("tpot_p50") else "FAILED"
            except Exception as e:
                row    = dict(model=model, device=device, num_gpus=tp,
                              batch=batch, ctx=ctx, baseline=baseline,
                              hbm_frac=compute_hbm_frac(model, device, batch, ctx, tp),
                              simulated=True, ttft_p50=None, ttft_p99=None,
                              tpot_p50=None, tpot_p99=None,
                              throughput_tps=None, flash_stall_ms=None,
                              note=f"EXCEPTION: {e}")
                status = "ERROR"
            rows.append(row)
            completed += 1
            print(f"  [{completed}/{len(sim_jobs)}] {label}  hbm={row.get('hbm_frac','?')}  → {status}")

            if status == "OK" and baseline == "Dense":
                for (m, d, b, c, cbl, ctp) in copy_rows:
                    if (m, d, b, c, ctp) == (model, device, batch, ctx, tp):
                        copy = dict(row)
                        copy["baseline"] = cbl
                        copy["note"] = f"copy-from-Dense (hbm_frac=1.0) TP={tp}"
                        rows.append(copy)

            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)

    print(f"\nDone. {len(rows)} rows → {out_csv}")


if __name__ == "__main__":
    main()
