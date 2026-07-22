#!/usr/bin/env python3
"""Capture GPU decode-step component breakdown for all 5 models at ONE operating point
(1M ctx, batch 16, per-model max TP), for H3 / Naive / SPLASH. Uses breakdown_one.py
(which picks the fullest decode step). Writes results/breakdown_all.csv."""
import csv, json, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from run_sweep_b200 import _dimcache

CTX, B = 1048576, 16
MODELS = [("meta-llama/Meta-Llama-3-8B", "Llama-3-8B"),
          ("mistralai/Mixtral-8x7B-v0.1", "Mixtral-8×7B"),
          ("deepseek-ai/deepseek-llm-67b-chat", "DeepSeek-67B"),
          ("meta-llama/Meta-Llama-3-70B", "Llama-3-70B"),
          ("Qwen/Qwen3-235B-A22B", "Qwen3-235B")]
SYS = [("H3", "1.0", "hbf", "0"), ("Naive", "0.1", "hbf", "1"), ("SPLASH", "0.1", "hbf", "0")]


def run(mk, md, label, sp, tier, sra):
    tp = min(8, _dimcache(mk)[3])
    p = subprocess.run([sys.executable, "breakdown_one.py", label, mk, str(CTX), str(B),
                        str(tp), sp, tier, sra], capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("RESULT"):
            _, lab, js = line.split(" ", 2)
            d = json.loads(js); d["model"] = md; d["system"] = label; d["tp"] = tp
            return d
    return None


jobs = [(mk, md, *s) for mk, md in MODELS for s in SYS]
with ThreadPoolExecutor(max_workers=5) as ex:
    res = [r for r in ex.map(lambda a: run(*a), jobs) if r]

fields = ["model", "system", "tp", "kv_read", "attn_proj", "mlp", "kv_write", "comm",
          "norms", "model_time_ms"]
with open("results/breakdown_all.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in res:
        w.writerow(r)
print(f"BREAKDOWN_ALL DONE -> {len(res)} rows")
for r in res:
    print(f"  {r['model']:13s} {r['system']:7s} kv={r['kv_read']:7.1f} model={r['model_time_ms']:7.1f}")
