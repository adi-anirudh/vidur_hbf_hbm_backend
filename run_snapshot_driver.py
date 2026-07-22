#!/usr/bin/env python3
"""Latency/throughput snapshot at one long-context operating point (1M ctx, batch-32)
for the selected models x all baselines. Emits mean + p99 TPOT and per-GPU throughput.
HBM-only is capacity-infeasible at this point (KV can't fit in HBM at any TP) -> marked
infeasible. Writes results/snapshot.csv."""
import csv, re, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from run_sweep_b200 import hbm_kv_fraction, feasible_tps, _dimcache

ROOT = Path(__file__).parent; PY = sys.executable
RUNNER = str(ROOT / "run_point_hbf.py"); TOML = str(ROOT / "configs/hbf_paper.toml")
OUT = ROOT / "results" / "snapshot.csv"
CTX = 1048576; BATCH = 32; AMP = "3.5636"

MODELS = [("meta-llama/Meta-Llama-3-8B", "Llama-3-8B"),
          ("mistralai/Mixtral-8x7B-v0.1", "Mixtral-8x7B"),
          ("deepseek-ai/deepseek-llm-67b-chat", "DeepSeek-67B"),
          ("meta-llama/Meta-Llama-3-70B", "Llama-3-70B"),
          ("Qwen/Qwen3-235B-A22B", "Qwen3-235B")]
# (label, sparsity, tier, sra)
SYS = [("HBM-only", 1.0, "hbm", False), ("Dense", 1.0, "hbf", False),
       ("Naive", 0.1, "hbf", True), ("Sparse", 0.1, "hbf", False)]


def run(model, tp, label, sp, tier, sra):
    frac = hbm_kv_fraction(model, "blackwell", BATCH, CTX, tp, tier)
    cmd = [PY, RUNNER, "--model", model, "--device", "blackwell", "--batch_size", str(BATCH),
           "--context_length", str(CTX), "--decode_tokens", "4", "--num_requests", str(BATCH),
           "--tensor_parallel_size", str(tp), "--sparsity_fraction", str(sp),
           "--hbm_kv_fraction", f"{frac:.6f}", "--backing_bw_gbps", "8000", "--hbfsim_toml", TOML]
    if sra:
        cmd += ["--hbf_sra", "--sparse_read_amplification", AMP]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=900, cwd=str(ROOT))
    m = re.search(r"tpot_mean_ms=([\d.]+) tpot_p95_ms=[\d.]+ ", p.stdout)
    p99 = re.search(r"tpot_p99_ms=([\d.]+)", p.stdout)
    if not (m and p99):
        return None
    return float(m.group(1)), float(p99.group(1))


def one(mk, md, label, sp, tier, sra):
    tp = min(8, _dimcache(mk)[3])
    feas, _ = feasible_tps(mk, "blackwell", BATCH, CTX, tier)
    if not feas:                       # capacity-infeasible (HBM-only at long ctx)
        return dict(model=md, system=label, tp="", tpot_mean="", tpot_p99="",
                    thr_mean="", thr_p99="", feasible=0)
    r = run(mk, tp, label, sp, tier, sra)
    if r is None:
        return dict(model=md, system=label, tp=tp, tpot_mean="", tpot_p99="",
                    thr_mean="", thr_p99="", feasible=0)
    mean, p99 = r
    return dict(model=md, system=label, tp=tp, tpot_mean=f"{mean:.3f}", tpot_p99=f"{p99:.3f}",
                thr_mean=f"{BATCH * 1000.0 / mean / tp:.3f}",
                thr_p99=f"{BATCH * 1000.0 / p99 / tp:.3f}", feasible=1)


def main():
    jobs = [(mk, md, lab, sp, tier, sra) for mk, md in MODELS for lab, sp, tier, sra in SYS]
    rows = []
    with ProcessPoolExecutor(max_workers=5) as ex:
        futs = [ex.submit(one, *j) for j in jobs]
        for f in as_completed(futs):
            rows.append(f.result())
    fields = ["model", "system", "tp", "tpot_mean", "tpot_p99", "thr_mean", "thr_p99", "feasible"]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for r in sorted(rows, key=lambda r: (r["model"], r["system"])):
            w.writerow(r)
    print("SNAPSHOT DONE ->", OUT)
    for r in sorted(rows, key=lambda r: (r["model"], r["system"])):
        print(f"  {r['model']:14s} {r['system']:9s} tpot_mean={r['tpot_mean'] or 'x':>8} "
              f"p99={r['tpot_p99'] or 'x':>8} thr_mean={r['thr_mean'] or 'x':>7}")


if __name__ == "__main__":
    main()
