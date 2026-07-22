#!/usr/bin/env python3
"""Co-design ablation at 128K on Blackwell: sweep batch for five systems at TP=8
so each can be loaded to its throughput-maximizing point under the decode SLO.
Same SLO operating-point methodology as the headline throughput figure.

  Dense (H3)         sparsity 1.0
  SPLASH             sparsity 0.1
  Full-page scoring  sparsity 0.1 + hbf_sra
  Token-granular     sparsity 0.1 + amp   = 3.56   (measured, real Llama-3.1-8B attn @128K)
  Plane imbalance    sparsity 0.1 + imbal = 1.92   (measured, real Llama-3.1-8B attn @128K)
"""
from __future__ import annotations
import csv, re, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent; PY = sys.executable
RUNNER = str(ROOT / "run_point_hbf.py"); TOML = str(ROOT / "configs/hbf_paper.toml")
CACHE = str(ROOT / "pred_cache"); OUT = ROOT / "results" / "ablation.csv"

MODEL = "meta-llama/Meta-Llama-3-8B"; DEVICE = "blackwell"; TP = 8; CTX = 131072
BATCHES = [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 320, 384, 512]
AMP = 3.5636; IMB = 1.9234          # measured @128K (codesign_factors.jsonl)
SYSTEMS = [
    ("Dense",          ["--sparsity_fraction", "1.0"]),
    ("SPLASH",         ["--sparsity_fraction", "0.1"]),
    ("FullPageScore",  ["--sparsity_fraction", "0.1", "--hbf_sra"]),
    # token-granular scores every token -> same full-K scan as full-page scoring
    # (--hbf_sra), PLUS the amplified page read (scattered tokens).
    ("TokenGranular",  ["--sparsity_fraction", "0.1", "--hbf_sra", "--sparse_read_amplification", str(AMP)]),
    ("PlaneImbalance", ["--sparsity_fraction", "0.1", "--plane_imbalance_factor", str(IMB)]),
]
FIELDS = ["model", "device", "tp", "batch", "context_length", "system",
          "amp", "imbalance", "tpot_p50_ms", "tpot_p99_ms", "status", "wall_s"]


def run_point(system, flags, batch, timeout_s=900):
    cmd = [PY, RUNNER, "--model", MODEL, "--device", DEVICE,
           "--batch_size", str(batch), "--context_length", str(CTX),
           "--decode_tokens", "4", "--num_requests", str(batch),
           "--tensor_parallel_size", str(TP), "--hbfsim_toml", TOML,
           "--cache_dir", CACHE] + flags
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, cwd=str(ROOT))
        wall = time.perf_counter() - t0
        m = re.search(r"RESULT status=OK tpot_p50_ms=([\d.]+) tpot_p99_ms=([\d.]+)", p.stdout)
        if m:
            return dict(tpot_p50_ms=m.group(1), tpot_p99_ms=m.group(2), status="OK", wall_s=f"{wall:.1f}")
        return dict(tpot_p50_ms="", tpot_p99_ms="", status=f"FAIL({p.returncode})", wall_s=f"{wall:.1f}")
    except subprocess.TimeoutExpired:
        return dict(tpot_p50_ms="", tpot_p99_ms="", status=f"TIMEOUT>{timeout_s}s", wall_s="")


def main():
    print("warming predictor cache...", flush=True)
    run_point("Dense", ["--sparsity_fraction", "1.0"], 1)
    jobs = [(s, flags, b) for (s, flags) in SYSTEMS for b in BATCHES]
    print(f"{len(jobs)} points ({MODEL} tp={TP} ctx={CTX//1024}K, batch sweep)", flush=True)
    fh = open(OUT, "w", newline=""); w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader()
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(run_point, s, flags, b): (s, b) for (s, flags, b) in jobs}
        for fut in as_completed(futs):
            s, b = futs[fut]; res = fut.result()
            row = dict(model=MODEL, device=DEVICE, tp=TP, batch=b, context_length=CTX,
                       system=s, amp=AMP, imbalance=IMB, **res)
            w.writerow(row); fh.flush()
            print(f"  {s:14s} b={b:<3d} -> {res['status']} tpot={res.get('tpot_p50_ms','')}", flush=True)
    fh.close(); print("ABLATION DONE ->", OUT, flush=True)


if __name__ == "__main__":
    main()
