#!/usr/bin/env python3
"""Corrected Blackwell (B200-class) sweep: HBM 192 GB @ 8 TB/s, HBF 3072 GB (16x)
@ 8 TB/s aggregate. Long-context subset, 8 context points 128K..2M, TP swept up to
min(8, num_kv_heads) on one eight-GPU node so both the 50 ms and 100 ms SLO
operating points are measured. Dense (H3) + SPLASH. Streams to
results/sweep_b200.csv (resumable).

HBF bandwidth is set via --backing_bw_gbps 8000 (flat aggregate = plane model at
saturation). Device 'blackwell' now carries the corrected 192 GB / 8 TB/s HBM."""
from __future__ import annotations
import csv, re, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from functools import lru_cache
from sweep_capacity import weight_bytes, HBM_GB, HBF_GB
from vidur.config.model_config import BaseModelConfig

ROOT = Path(__file__).parent; PY = sys.executable
RUNNER = str(ROOT / "run_point_hbf.py"); TOML = str(ROOT / "configs/hbf_paper.toml")
OUT = ROOT / "results" / "sweep_b200.csv"
CACHE = str(ROOT / "results" / "predictor_cache_eval")

# Complete paper sweep: ten architectures spanning 2.7B--405B, dense/MoE, and
# MHA/GQA. Keep this list aligned with results/main_sweep_summary.json.
MODELS = ["microsoft/phi-2", "mistralai/Mistral-7B-v0.1",
          "meta-llama/Meta-Llama-3-8B", "mistralai/Mixtral-8x7B-v0.1",
          "deepseek-ai/deepseek-llm-67b-chat",
          "meta-llama/Meta-Llama-3-70B", "Qwen/Qwen2-72B",
          "mistralai/Mixtral-8x22B-v0.1", "Qwen/Qwen3-235B-A22B",
          "meta-llama/Meta-Llama-3.1-405B"]
DEVICES = ["blackwell"]
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]
CONTEXTS = [131072, 196608, 262144, 393216, 524288, 786432, 1048576, 2097152]
TP_CHOICES = (1, 2, 4, 8)
# (label, sparsity, tier). HBM-only holds the ENTIRE KV in HBM -> capacity-bounded TP
# and frac=1.0 (no HBF). It is forced onto more GPUs than the HBF baselines and OOMs
# once even max-TP can't fit the KV in HBM, so it comes out worse per-GPU than H3.
BASELINES = [("Dense", 1.0, "hbf"), ("Sparse", 0.1, "hbf"), ("HBM-only", 1.0, "hbm"),
             ("Naive", 0.1, "hbf")]
HBF_BW = "8000"
# Naive token-granular sparse = SPLASH minus its two co-design wins: full-page
# scoring (--hbf_sra: scan ALL cold K, no centroid) + read amplification
# (--sparse_read_amplification). AMP measured @128K on Llama-3.1-8B (codesign
# ablation); measured with the final 1024-plane layout at 10% selection.
NAIVE_AMP = "3.5635"
FIELDS = ["model", "device", "batch", "context_length", "baseline", "tp", "footprint_gb",
          "sparsity_fraction", "tpot_p50_ms", "tpot_p99_ms", "status", "wall_s"]


@lru_cache(maxsize=None)
def _dimcache(model):
    c = BaseModelConfig.create_from_name(model)
    tp_cap = 1 if getattr(c, "no_tensor_parallel", False) else min(32, c.num_kv_heads)
    return c.num_layers, c.num_kv_heads, c.head_size(), tp_cap

def feasible_tps(model, device, batch, ctx, tier="hbf"):
    n_lay, n_kv, head_dim, tp_cap = _dimcache(model)
    total = weight_bytes(model) + batch * ctx * n_lay * 2 * n_kv * head_dim * 2
    valid = []
    for tp in TP_CHOICES:
        if tp > tp_cap:
            continue
        # HBF is a KV tier: model weights and the activation reserve must fit in
        # HBM even when the combined HBM+HBF capacity is otherwise sufficient.
        weights_and_reserve_per_gpu = weight_bytes(model) / tp + 4e9
        if weights_and_reserve_per_gpu > HBM_GB[device] * 1e9:
            continue
        kv_per_gpu = (total - weight_bytes(model)) / tp
        hbm_kv_capacity = (
            HBM_GB[device] * 1e9 - weights_and_reserve_per_gpu
        )
        if tier == "hbm":
            fits = kv_per_gpu <= hbm_kv_capacity
        else:
            cold_kv = max(0.0, kv_per_gpu - hbm_kv_capacity)
            # One FP16 centroid per 16-token K page: 1/16 of K,
            # hence 1/32 of the K+V cold-KV footprint.
            fits = cold_kv * (1.0 + 1.0 / 32.0) <= HBF_GB * 1e9
        if fits:
            valid.append(tp)
    return valid, total / 1e9


def hbm_kv_fraction(model, device, batch, ctx, tp, tier="hbf"):
    if tier == "hbm":
        return 1.0                                                    # entire KV resident in HBM
    # Fraction of the KV that fits in HBM (the sliding window, read densely); the
    # rest is cold KV in HBF. HBM budget/GPU = HBM - weights/TP - activation buffer.
    n_lay, n_kv, head_dim, _ = _dimcache(model)
    kv_bytes = batch * ctx * n_lay * 2 * n_kv * head_dim * 2           # total KV (all GPUs)
    hbm_kv_avail_per_gpu = HBM_GB[device] * 1e9 - weight_bytes(model) / tp - 4e9
    kv_per_gpu = kv_bytes / tp
    return max(0.0, min(1.0, hbm_kv_avail_per_gpu / kv_per_gpu)) if kv_per_gpu else 1.0


def run_point(model, device, batch, ctx, tp, label, sp, tier, timeout_s):
    frac = hbm_kv_fraction(model, device, batch, ctx, tp, tier)
    cmd = [PY, RUNNER, "--model", model, "--device", device, "--batch_size", str(batch),
           "--context_length", str(ctx), "--decode_tokens", "4", "--num_requests", str(batch),
           "--tensor_parallel_size", str(tp), "--sparsity_fraction", str(sp),
           "--hbm_kv_fraction", f"{frac:.6f}", "--backing_bw_gbps", HBF_BW,
           "--hbfsim_toml", TOML, "--cache_dir", CACHE]
    # Shared cache writes are atomic and lock-protected; after the first point for a
    # model/TP, the remaining grid is read-only and avoids retraining identical tables.
    if label == "Naive":
        cmd += ["--hbf_sra", "--sparse_read_amplification", NAIVE_AMP]
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, cwd=str(ROOT))
        wall = time.perf_counter() - t0
        m = re.search(r"RESULT status=OK tpot_p50_ms=([\d.]+) tpot_p99_ms=([\d.]+)", p.stdout)
        if m: return dict(tpot_p50_ms=m.group(1), tpot_p99_ms=m.group(2), status="OK", wall_s=f"{wall:.1f}")
        return dict(tpot_p50_ms="", tpot_p99_ms="", status=f"FAIL({p.returncode})", wall_s=f"{wall:.1f}")
    except subprocess.TimeoutExpired:
        return dict(tpot_p50_ms="", tpot_p99_ms="", status=f"TIMEOUT>{timeout_s}s", wall_s="")


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=900); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-model", choices=MODELS)
    ap.add_argument("--only-context", type=int, choices=CONTEXTS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out = args.out
    done = set()
    if out.exists():
        for r in csv.DictReader(open(out)):
            if r["status"] == "OK" and r["tp"]:
                done.add((r["model"], r["device"], int(r["batch"]), int(r["context_length"]), r["baseline"], int(r["tp"])))
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists()
    fh = open(out, "a", newline=""); w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
    if new: w.writeheader()
    runnable = []
    for model in MODELS:
        if args.only_model and model != args.only_model:
            continue
        for device in DEVICES:
            for batch in BATCHES:
                for ctx in CONTEXTS:
                    if args.only_context and ctx != args.only_context:
                        continue
                    for label, sp, tier in BASELINES:
                        tps, foot = feasible_tps(model, device, batch, ctx, tier)
                        for tp in tps:
                            if (model, device, batch, ctx, label, tp) in done: continue
                            runnable.append(dict(model=model, device=device, batch=batch, context_length=ctx,
                                                 baseline=label, tp=tp, footprint_gb=f"{foot:.1f}",
                                                 sparsity_fraction=sp, tier=tier))
    if args.limit: runnable = runnable[:args.limit]
    print(f"runnable points: {len(runnable)} (already done {len(done)})", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_point, b["model"], b["device"], b["batch"], b["context_length"],
                          b["tp"], b["baseline"], b["sparsity_fraction"], b["tier"], args.timeout): b for b in runnable}
        n = 0
        for fut in as_completed(futs):
            b = futs[fut]
            try: res = fut.result()
            except Exception as e: res = dict(tpot_p50_ms="", tpot_p99_ms="", status=f"ERR:{type(e).__name__}", wall_s="")
            w.writerow({**b, **res}); fh.flush(); n += 1
            if n % 25 == 0: print(f"  {n}/{len(runnable)}", flush=True)
    fh.close(); print("SWEEP DONE ->", out, flush=True)


if __name__ == "__main__":
    main()
