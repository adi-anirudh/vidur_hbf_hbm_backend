#!/usr/bin/env python3.9
"""
Comprehensive paper experiment sweep for HBF long-context inference.

Three baselines (all share the same HBM hot window):
  Dense      : HBM dense + HBF dense reads (sparsity=1.0)
  NaiveSparse: HBM dense + HBF naive sequential scan
  Sparse     : HBM dense + HBF plane-optimized Top-K (sparsity=0.1)

HBM fraction = min(1.0, available_GPU_mem / total_KV_size)
  - HBM holds dense+init KV until full; remainder goes to HBF
  - Optimisation: when hbm_frac >= 1.0, all three baselines are identical
    (no HBF access); only Dense is simulated and results are copied.

Full sweep: all models × all GPUs × 8 context lengths × 3 baselines.
Context lengths: 4K, 8K, 16K, 32K, 64K, 128K, 1M, 10M (all simulated).
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
# Hardware constants (must match hbf_default.toml)
# ---------------------------------------------------------------------------
TOTAL_PLANES = 1024     # 64 planes/die × 16 dies/stack × 1 stack
TR_NS        = 4096     # tR = 4 µs
PAGE_BYTES   = 4096     # 4 KB NAND page
BLOCK_SIZE   = 16       # Vidur KV block = 16 tokens
SPARSITY     = 0.10     # Top-K: 10% of HBF blocks accessed

GPU_MEM_GB = {
    "a100": 80.0,
    "h100": 80.0,
    "a40":  45.0,
    "h200": 141.0,
}
HBM_BW_GBPS = {
    "a100": 2000.0,
    "h100": 3350.0,
    "a40":  696.0,
    "h200": 4800.0,
}

# ---------------------------------------------------------------------------
# Model table  (10 diverse models across families, sizes, KV-head counts)
# ---------------------------------------------------------------------------
# kv_heads × head_dim × layers × 2 bytes × 2 (K+V) = KV bytes / token
#
#  Model                    Size  KV-heads  arch      profiling source
#  ─────────────────────────────────────────────────────────────────────
#  phi-2                    2.7B  32 (MHA)  dense     real
#  Mistral-7B-v0.1          7B     8 (GQA)  dense     → Meta-Llama-3-8B
#  Meta-Llama-3-8B          8B     8 (GQA)  dense     real
#  Mixtral-8x7B-v0.1       46.7B   8 (GQA)  MoE       → Meta-Llama-3-8B
#  internlm2-20b           20B     8 (GQA)  dense     real
#  deepseek-llm-67b-chat   67B     8 (GQA)  dense     → Llama-2-70b-hf
#  Meta-Llama-3-70B        70B     8 (GQA)  dense     real
#  Qwen-72B                72B    64 (MHA)  dense     real
#  Qwen2-72B               72B     8 (GQA)  dense     → Llama-2-70b-hf
#  Meta-Llama-3.1-405B    405B     8 (GQA)  dense     synthetic
# ---------------------------------------------------------------------------
MODEL_INFO = {
    "microsoft/phi-2": dict(
        layers=32,  kv_heads=32, head_dim=80,  weights_gb=5.4),
    "mistralai/Mistral-7B-v0.1": dict(
        layers=32,  kv_heads=8,  head_dim=128, weights_gb=14.0),
    "meta-llama/Meta-Llama-3-8B": dict(
        layers=32,  kv_heads=8,  head_dim=128, weights_gb=16.0),
    "mistralai/Mixtral-8x7B-v0.1": dict(
        layers=32,  kv_heads=8,  head_dim=128, weights_gb=93.0),
    "deepseek-ai/deepseek-llm-67b-chat": dict(
        layers=95,  kv_heads=8,  head_dim=128, weights_gb=134.0),
    "meta-llama/Meta-Llama-3-70B": dict(
        layers=80,  kv_heads=8,  head_dim=128, weights_gb=140.0),
    "Qwen/Qwen-72B": dict(
        layers=80,  kv_heads=64, head_dim=128, weights_gb=144.0),
    "Qwen/Qwen2-72B": dict(
        layers=80,  kv_heads=8,  head_dim=128, weights_gb=145.0),
    "meta-llama/Meta-Llama-3.1-405B": dict(
        layers=126, kv_heads=8,  head_dim=128, weights_gb=810.0),
}
GPU_OVERHEAD_GB  = 2.0
HBF_CAPACITY_GB  = (TOTAL_PLANES * 256 * 256 * PAGE_BYTES) / 1e9   # ≈ 206 GB

ALL_MODELS    = list(MODEL_INFO.keys())
ALL_GPUS      = ["a100", "h100", "a40", "h200"]
CTX_POINTS    = [4_096, 8_192, 16_384, 32_768, 65_536, 131_072, 262_144, 524_288,
                 1_048_576, 10_485_760]
BATCH_SIZES   = [1, 2, 4, 8, 16, 32, 64, 128]
DECODE_TOKENS = 8

SCRIPT  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_hbf_simulation.py")
TOML    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs/hbf_paper.toml")
PYTHON  = "/home/yaswanth/miniconda3/envs/scissors/bin/python3.10"
NUM_REQ = 8    # decode-only: all requests start fully prefilled → fast, clean decode metrics


# ---------------------------------------------------------------------------
# KV geometry
# ---------------------------------------------------------------------------

def kv_per_token_bytes(model):
    p = MODEL_INFO[model]
    return 2 * p["kv_heads"] * p["head_dim"] * 2   # FP16 K+V

def kv_block_bytes(model):
    return kv_per_token_bytes(model) * BLOCK_SIZE

def pages_per_kv_block(model):
    return max(1, math.ceil(kv_block_bytes(model) / PAGE_BYTES))

def compute_hbm_frac(model, device, batch, ctx):
    """Fraction of total KV that fits in HBM after weights (using TP)."""
    gpu_gb   = GPU_MEM_GB[device]
    info     = MODEL_INFO[model]
    num_gpus = max(1, math.ceil(info["weights_gb"] / gpu_gb))
    available_gb = max(0.0, gpu_gb - info["weights_gb"] / num_gpus - GPU_OVERHEAD_GB)
    total_kv_gb  = batch * ctx * kv_per_token_bytes(model) * info["layers"] / 1e9
    if total_kv_gb <= 0:
        return 1.0
    return min(1.0, available_gb / total_kv_gb)


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def _run_one(job):
    model, device, batch, ctx, baseline, num_req, job_id = job

    hbm_frac = compute_hbm_frac(model, device, batch, ctx)
    num_kv   = math.ceil(batch * math.ceil(ctx / BLOCK_SIZE) / 0.99) + 64

    # NaiveSparse = sequential scan until last needed block ≈ all blocks read
    if baseline == "Dense":
        kv_read_fraction = 1.0
        effective_hbm_frac = hbm_frac
    elif baseline == "NaiveSparse":
        kv_read_fraction = 1.0   # worst-case scan reads essentially all blocks
        effective_hbm_frac = hbm_frac
    else:  # Sparse
        kv_read_fraction = SPARSITY
        # When KV overflows HBM into HBF, Dense reads all HBF overflow;
        # Sparse reads SPARSITY fraction of HBF overflow (10x less).
        # Achieved by scaling the HBM tier split by SPARSITY so that
        # Sparse's accessed blocks span HBM/HBF in the same ratio as Dense.
        effective_hbm_frac = hbm_frac * SPARSITY

    cmd = [
        PYTHON, SCRIPT,
        "--model",             model,
        "--device",            device,
        "--batch_size",        str(batch),
        "--context_length",    str(ctx),
        "--decode_tokens",     str(DECODE_TOKENS),
        "--qps",               "1000",
        "--decode_only",
        "--num_requests",      str(batch),
        "--kv_read_fraction",  str(kv_read_fraction),
        "--hbm_kv_fraction",   str(effective_hbm_frac),
        "--num_kv_blocks",     str(num_kv),
        "--hbfsim_toml",       TOML,
    ]

    t0  = time.time()
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
        note = f"sim {elapsed:.0f}s"

    return dict(
        model=model, device=device, batch=batch, ctx=ctx, baseline=baseline,
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
# Build job list
# ---------------------------------------------------------------------------

BASELINES = ["Dense", "NaiveSparse", "Sparse"]


def is_feasible(model, device, batch, ctx):
    """KV must fit in HBM + HBF combined; skip otherwise."""
    info      = MODEL_INFO[model]
    total_kv_gb = batch * ctx * kv_per_token_bytes(model) * info["layers"] / 1e9
    num_gpus  = max(1, math.ceil(info["weights_gb"] / GPU_MEM_GB[device]))
    avail_gb  = max(0.0, GPU_MEM_GB[device] - info["weights_gb"] / num_gpus - GPU_OVERHEAD_GB)
    hbf_needed = max(0.0, total_kv_gb - avail_gb)
    return hbf_needed <= HBF_CAPACITY_GB


def build_jobs(resume_keys=set()):
    """
    Full cross-product: all models × all GPUs × 8 ctx × 6 batch sizes × 3 baselines.

    Optimisation: when hbm_frac >= 1.0 (all KV fits in HBM) the three
    baselines produce identical results.  We simulate only Dense and
    synthesise NaiveSparse / Sparse by copying later.

    Feasibility: skip (model, GPU, ctx, batch) combos where HBF would need
    to hold more than HBF_CAPACITY_GB of KV (≈ 206 GB, one stack).
    """
    sim_jobs = []
    copy_rows = []
    seen = {}

    def add(model, device, batch, ctx):
        if not is_feasible(model, device, batch, ctx):
            return
        hbm_frac   = compute_hbm_frac(model, device, batch, ctx)
        all_in_hbm = hbm_frac >= 1.0

        for bl in BASELINES:
            key = (model, device, str(batch), str(ctx), bl)
            if key in seen:
                continue
            seen[key] = True

            if all_in_hbm and bl != "Dense":
                copy_rows.append((model, device, batch, ctx, bl))
                continue

            if key not in resume_keys:
                jid = len(sim_jobs)
                sim_jobs.append((model, device, batch, ctx, bl, batch, jid))

    for model in ALL_MODELS:
        for gpu in ALL_GPUS:
            for ctx in CTX_POINTS:
                for batch in BATCH_SIZES:
                    add(model, gpu, batch, ctx)

    return sim_jobs, copy_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resume",  action="store_true")
    args = ap.parse_args()

    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paper_results.csv")

    done_keys = set()
    existing  = []
    if args.resume and os.path.exists(out_csv):
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                existing.append(row)
                done_keys.add((row["model"], row["device"],
                               str(row["batch"]), str(row["ctx"]), row["baseline"]))
        print(f"Resuming: {len(done_keys)} results already in {out_csv}")

    sim_jobs, copy_rows = build_jobs(resume_keys=done_keys)

    print("\nHBM fractions sample (LLaMA-3-8B, A100, batch=32):")
    for ctx in CTX_POINTS:
        f = compute_hbm_frac("meta-llama/Meta-Llama-3-8B", "a100", 32, ctx)
        feas = is_feasible("meta-llama/Meta-Llama-3-8B", "a100", 32, ctx)
        tag = "(all HBM)" if f >= 1.0 else f"(HBF={1-f:.1%})"
        skip = "" if feas else "  SKIP (exceeds HBF cap)"
        print(f"  ctx={ctx//1024 if ctx < 1_000_000 else ctx//1_048_576:4}{'K' if ctx < 1_000_000 else 'M'}  "
              f"hbm_frac={f:.3f}  {tag}{skip}")

    print(f"\nSim jobs: {len(sim_jobs)}  |  Copy-from-Dense rows: {len(copy_rows)}")
    print(f"Workers:  {args.workers}\n")

    fields = ["model", "device", "batch", "ctx", "baseline", "hbm_frac", "simulated",
              "ttft_p50", "ttft_p99", "tpot_p50", "tpot_p99",
              "throughput_tps", "flash_stall_ms", "note"]

    rows = list(existing)
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_one, j): j for j in sim_jobs}
        for fut in as_completed(futures):
            job    = futures[fut]
            model_short = job[0].split("/")[-1]
            label  = f"{model_short} {job[1]} B={job[2]} ctx={job[3]//1024}K {job[4]}"
            try:
                row    = fut.result()
                status = "OK" if row.get("tpot_p50") else "FAILED"
            except Exception as e:
                row    = dict(model=job[0], device=job[1], batch=job[2], ctx=job[3],
                              baseline=job[4],
                              hbm_frac=compute_hbm_frac(job[0], job[1], job[2], job[3]),
                              simulated=True, ttft_p50=None, ttft_p99=None,
                              tpot_p50=None, tpot_p99=None,
                              throughput_tps=None, flash_stall_ms=None,
                              note=f"EXCEPTION: {e}")
                status = "ERROR"
            rows.append(row)
            completed += 1
            hf = row.get("hbm_frac", "?")
            print(f"  [{completed}/{len(sim_jobs)}] {label}  hbm={hf}  → {status}  ({row.get('note', '')})")

            # Synthesise copy rows for this (model, device, batch, ctx) Dense run
            if status == "OK" and job[4] == "Dense":
                for (m, d, b, c, copy_bl) in copy_rows:
                    if (m, d, b, c) == (job[0], job[1], job[2], job[3]):
                        copy = dict(row)
                        copy["baseline"] = copy_bl
                        copy["note"]     = f"copy-from-Dense (hbm_frac=1.0)"
                        rows.append(copy)

            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                w.writeheader()
                w.writerows(rows)

    print(f"\nDone. Results: {out_csv}  ({len(rows)} total rows)")
    _print_tables(rows)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

SLO_THRESHOLDS = [30, 50, 100, 200]   # TPOT p50 thresholds in ms


def _fmt(v, decimals=1):
    try:    return f"{float(v):.{decimals}f}"
    except: return "N/A"

def _slo_tag(v, thr):
    try:    return "P" if float(v) <= thr else "F"
    except: return "?"

def _speedup(dense_tpot, bl_tpot):
    try:    return f"{float(dense_tpot)/float(bl_tpot):.2f}x"
    except: return "N/A"

def _stall_frac(tpot_ms, stall_ms):
    try:
        t, s = float(tpot_ms), float(stall_ms or 0)
        # stall_ms is cumulative across all requests; tpot_ms is per-token median
        return f"{s/(t*8*8)*100:.0f}%" if t > 0 else "N/A"
    except: return "N/A"


def _print_tables(rows):
    slo_hdr = " ".join(f"S{t:>3}" for t in SLO_THRESHOLDS)
    hdr = (f"{'':38} {'Baseline':<12} {'HBM%':>5} "
           f"{'TPOT50':>7} {'TPOT99':>7} {'Thpt':>8} "
           f"{'Stall':>8} {'Spdup':>6}  {slo_hdr}")
    sep = "-" * 118

    def _row_dense_tpot(r, filt_rows):
        dense = [x for x in filt_rows
                 if x["model"] == r["model"] and x["device"] == r["device"]
                 and x["batch"] == r["batch"] and x["ctx"] == r["ctx"]
                 and x["baseline"] == "Dense"]
        return dense[0]["tpot_p50"] if dense else None

    def tbl(title, filt, key_fn, label_fn):
        print("\n" + "=" * 118)
        print(title)
        print(hdr); print(sep)
        filt_rows = [r for r in rows if filt(r)]
        for r in sorted(filt_rows, key=lambda x: (key_fn(x), x["baseline"])):
            hf      = float(r.get("hbm_frac") or 0)
            sim     = "" if str(r.get("simulated")) == "True" else "*"
            dt      = _row_dense_tpot(r, filt_rows)
            spdup   = _speedup(dt, r["tpot_p50"]) if r["baseline"] != "Dense" else "—"
            slo_str = " ".join(f"{_slo_tag(r['tpot_p50'], t):>4}" for t in SLO_THRESHOLDS)
            stall   = _fmt(r.get("flash_stall_ms") or 0)
            print(f"{label_fn(r)+sim:<38} {r['baseline']:<12} {hf*100:>4.0f}% "
                  f"{_fmt(r['tpot_p50']):>7} {_fmt(r['tpot_p99']):>7} "
                  f"{_fmt(r['throughput_tps']):>8} "
                  f"{stall:>8} {spdup:>6}  {slo_str}")

    tbl("TABLE 1: Context Sweep  (LLaMA-3-8B, A100, batch=32)",
        lambda r: (r["model"] == "meta-llama/Meta-Llama-3-8B" and r["device"] == "a100"
                   and str(r["batch"]) == "32"),
        lambda r: int(r["ctx"]),
        lambda r: f"ctx={'%dK' % (int(r['ctx'])//1024) if int(r['ctx']) < 1_000_000 else '%dM' % (int(r['ctx'])//1_048_576)}")

    tbl("TABLE 2: Model Sweep  (A100, batch=32, ctx=128K)",
        lambda r: (r["device"] == "a100" and str(r["batch"]) == "32"
                   and str(r["ctx"]) == "131072"),
        lambda r: r["model"],
        lambda r: r["model"].split("/")[-1])

    tbl("TABLE 3: GPU Sweep  (LLaMA-3-8B, batch=32, ctx=128K)",
        lambda r: (r["model"] == "meta-llama/Meta-Llama-3-8B"
                   and str(r["batch"]) == "32"
                   and str(r["ctx"]) == "131072"),
        lambda r: r["device"],
        lambda r: r["device"])

    tbl("TABLE 4: Batch Sweep  (LLaMA-3-8B, A100, ctx=128K)",
        lambda r: (r["model"] == "meta-llama/Meta-Llama-3-8B" and r["device"] == "a100"
                   and str(r["ctx"]) == "131072"),
        lambda r: int(r["batch"]),
        lambda r: f"batch={r['batch']}")

    tbl("TABLE 5: Batch Sweep  (Qwen-72B, H100, ctx=1M)  [High-KV MHA model]",
        lambda r: (r["model"] == "Qwen/Qwen-72B" and r["device"] == "h100"
                   and str(r["ctx"]) == "1048576"),
        lambda r: int(r["batch"]),
        lambda r: f"batch={r['batch']}")

    print(f"\n* = analytical  |  SLO thresholds (P=pass,F=fail): {SLO_THRESHOLDS} ms TPOT p50")
    print(f"Spdup = Dense_TPOT / Baseline_TPOT  |  Stall = cumulative HBF stall (ms)")

    # SLO compliance summary across ALL results
    _print_slo_summary(rows)


def _print_slo_summary(rows):
    """Cross-baseline SLO compliance and speedup summary."""
    ok = [r for r in rows if r.get("tpot_p50") and "FAILED" not in str(r.get("note", ""))]
    if not ok:
        return

    print("\n" + "=" * 118)
    print("SLO COMPLIANCE SUMMARY  (fraction of sim points meeting threshold, by baseline)")
    print(f"{'Baseline':<14}", end="")
    for t in SLO_THRESHOLDS:
        print(f"  {'TPOT<'+str(t)+'ms':>10}", end="")
    print(f"  {'median TPOT':>12}  {'median Thpt':>12}  {'P99 TPOT':>10}")
    print("-" * 118)

    import statistics
    for bl in ["Dense", "NaiveSparse", "Sparse"]:
        bl_rows = [r for r in ok if r["baseline"] == bl]
        if not bl_rows:
            continue
        tpots = [float(r["tpot_p50"]) for r in bl_rows]
        tpot99s = [float(r["tpot_p99"]) for r in bl_rows]
        thpts = [float(r["throughput_tps"]) for r in bl_rows if r.get("throughput_tps")]
        n = len(bl_rows)
        compliance = [f"{sum(1 for r in bl_rows if float(r['tpot_p50']) <= t)/n*100:>9.0f}%" for t in SLO_THRESHOLDS]
        print(f"{bl:<14}", end="")
        for c in compliance:
            print(f"  {c:>10}", end="")
        med_tpot = statistics.median(tpots)
        med_thpt = statistics.median(thpts) if thpts else float('nan')
        med_p99  = statistics.median(tpot99s)
        print(f"  {med_tpot:>11.1f}ms  {med_thpt:>10.1f}/s  {med_p99:>9.1f}ms")

    # Speedup distribution
    print()
    dense_map = {(r["model"], r["device"], r["batch"], r["ctx"]): float(r["tpot_p50"])
                 for r in ok if r["baseline"] == "Dense"}
    for bl in ["NaiveSparse", "Sparse"]:
        speedups = []
        for r in ok:
            if r["baseline"] != bl:
                continue
            key = (r["model"], r["device"], r["batch"], r["ctx"])
            if key in dense_map:
                try:
                    speedups.append(dense_map[key] / float(r["tpot_p50"]))
                except (ValueError, ZeroDivisionError):
                    pass
        if speedups:
            s = sorted(speedups)
            n = len(s)
            print(f"Speedup vs Dense → {bl:<14}: "
                  f"median={statistics.median(s):.2f}x  "
                  f"p25={s[n//4]:.2f}x  p75={s[3*n//4]:.2f}x  "
                  f"max={s[-1]:.2f}x  (n={n})")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
