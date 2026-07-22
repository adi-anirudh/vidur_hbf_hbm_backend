#!/usr/bin/env python3
"""Single-model motivation data (batch 16 -> KV spills out of HBM at ~80K, giving a
wide divergence region for the intro figure). Three backends: HBM-only, HBF dense (H3),
HBF+SPLASH. Each uses the fewest GPUs (min TP) that hold its footprint; HBM-only in HBM
alone (walls out), HBF backends in HBM+HBF. SPLASH reads 10% of the cold KV.
Output: results/motivation.csv"""
from __future__ import annotations
import csv, re, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from sweep_capacity import weight_bytes, HBM_GB, HBF_GB

ROOT = Path(__file__).parent; PY = sys.executable
RUNNER = str(ROOT / "run_point_hbf.py"); TOML = str(ROOT / "configs/hbf_paper.toml")
CACHE = str(ROOT / "pred_cache"); OUT = ROOT / "results" / "motivation.csv"
MODEL = "meta-llama/Meta-Llama-3-70B"; DEVICE = "blackwell"; BATCH = 8; HBF_BW = "8000"
KV_PER_TOK = 2 * 8 * 128 * 80 * 2                       # 70B: 8 KV heads x 128 x 80 layers

SYSTEMS = [("HBM-only", 1.0, "hbm", 1.0), ("HBF", 1.0, "hbf", 1.0), ("HBF+SPLASH", 0.1, "hbf", 1.0)]
CONTEXTS = [4096, 8192, 16384, 32768, 65536, 98304, 131072, 196608, 262144,
            393216, 524288, 786432, 1048576]     # cap at 1M: 70B stays TP=1 in HBF (no kinks)
FIELDS = ["store", "context_length", "batch", "tp", "hbm_kv_fraction", "sparsity",
          "backing_bw", "tpot_ms", "throughput_tok_s", "status"]


def kv_total(ctx): return BATCH * ctx * KV_PER_TOK


def min_tp(ctx, tier):
    cap = (HBM_GB[DEVICE] if tier == "hbm" else HBM_GB[DEVICE] + HBF_GB) * 1e9
    total = weight_bytes(MODEL) + kv_total(ctx)
    for tp in (1, 2, 4, 8):
        if total <= tp * cap:
            return tp
    return None


def hbm_frac(ctx, tp, tier):
    if tier == "hbm":
        return 1.0
    avail = HBM_GB[DEVICE] * 1e9 - weight_bytes(MODEL) / tp - 4e9
    return max(0.0, min(1.0, avail / (kv_total(ctx) / tp)))


def run_point(store, sp, tier, amp, ctx, timeout_s=1200):
    tp = min_tp(ctx, tier)
    if tp is None:
        return dict(tp="", hbm_kv_fraction="", sparsity=sp, backing_bw=HBF_BW, tpot_ms="",
                    throughput_tok_s="", status="INFEASIBLE")
    frac = hbm_frac(ctx, tp, tier)
    cmd = [PY, RUNNER, "--model", MODEL, "--device", DEVICE, "--batch_size", str(BATCH),
           "--context_length", str(ctx), "--decode_tokens", "4", "--num_requests", str(BATCH),
           "--tensor_parallel_size", str(tp), "--sparsity_fraction", str(sp),
           "--sparse_read_amplification", str(amp),
           "--hbm_kv_fraction", f"{frac:.6f}", "--backing_bw_gbps", HBF_BW,
           "--hbfsim_toml", TOML, "--cache_dir", CACHE]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, cwd=str(ROOT))
        m = re.search(r"RESULT status=OK tpot_p50_ms=([\d.]+)", p.stdout)
        if m:
            t = float(m.group(1))
            return dict(tp=tp, hbm_kv_fraction=f"{frac:.4f}", sparsity=sp, backing_bw=HBF_BW,
                        tpot_ms=f"{t:.4f}", throughput_tok_s=f"{BATCH*1000.0/t:.4f}", status="OK")
        return dict(tp=tp, hbm_kv_fraction=f"{frac:.4f}", sparsity=sp, backing_bw=HBF_BW,
                    tpot_ms="", throughput_tok_s="", status=f"FAIL({p.returncode})")
    except subprocess.TimeoutExpired:
        return dict(tp=tp, hbm_kv_fraction=f"{frac:.4f}", sparsity=sp, backing_bw=HBF_BW,
                    tpot_ms="", throughput_tok_s="", status="TIMEOUT")


def main():
    print("warm cache..."); run_point("HBF", 1.0, "hbf", 1.0, CONTEXTS[0])
    jobs = [(s, sp, tier, amp, c) for (s, sp, tier, amp) in SYSTEMS for c in CONTEXTS]
    print(f"{len(jobs)} points (batch {BATCH})", flush=True)
    fh = open(OUT, "w", newline=""); w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(run_point, s, sp, tier, amp, c): (s, c) for (s, sp, tier, amp, c) in jobs}
        for fut in as_completed(futs):
            s, c = futs[fut]; res = fut.result()
            w.writerow(dict(store=s, context_length=c, batch=BATCH, **res)); fh.flush()
    fh.close(); print("MOTIVATION DATA DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
