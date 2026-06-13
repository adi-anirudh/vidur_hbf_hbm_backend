# -*- coding: utf-8 -*-
"""
Validate the ANALYTICAL compute model against REAL profiled GEMM times on every
(model, device) cell we have measured data for.

For each weight-load GEMM (QKV, O, MLP up/gate, MLP down) at each profiled
(num_tokens, TP), compare:
    analytical = max(FLOPs / peak_TFLOPS, weight_bytes / HBM_bandwidth)   [roofline]
    actual     = profiled median time
Report relative error, split by regime:
    decode-scale (num_tokens <= 128, memory-bound) vs prefill-scale (compute-bound).

This settles whether analytical compute is paper-defensible (within ~1%) or
whether real per-(model,device) traces are required.
"""
from __future__ import annotations
import math
import pandas as pd

# device peak specs (from vidur/config/device_sku_config.py)
TFLOPS = {"a40": 150e12, "a100": 312e12, "h100": 1000e12}
BW_BPS = {"a40": 696e9, "a100": 2039e9, "h100": 3350e9}
FP16 = 2

CELLS = [
    ("a40", "microsoft/phi-2"), ("a100", "microsoft/phi-2"), ("h100", "microsoft/phi-2"),
    ("a40", "Qwen/Qwen-72B"), ("a100", "Qwen/Qwen-72B"), ("h100", "Qwen/Qwen-72B"),
    ("a100", "meta-llama/Meta-Llama-3-8B"), ("a100", "meta-llama/Meta-Llama-3-70B"),
]


def gemm_specs(row):
    """Return {op: (flops_per_token_factor, weight_elems)} for the 4 weight GEMMs,
    already divided by TP."""
    nq, nkv, emb, exp = int(row.n_head), int(row.n_kv_head), int(row.n_embd), int(row.n_expanded_embd)
    gated = bool(row.use_gated_mlp)
    tp = int(row.num_tensor_parallel_workers)
    hd = emb // nq
    qkv_out = (nq + 2 * nkv) * hd
    up_out = (2 if gated else 1) * exp
    # (weight_elems, flops_coeff)  with flops = 2 * num_tokens * weight_elems
    specs = {
        "attn_pre_proj":  emb * qkv_out,
        "attn_post_proj": (nq * hd) * emb,
        "mlp_up_proj":    emb * up_out,
        "mlp_down_proj":  exp * emb,
    }
    return {k: v / tp for k, v in specs.items()}


def analytical_ms(weight_elems, num_tokens, tflops, bw):
    flops = 2 * num_tokens * weight_elems
    wbytes = weight_elems * FP16
    t = max(flops / tflops, wbytes / bw)   # seconds
    return t * 1e3


def main():
    OPS = ["attn_pre_proj", "attn_post_proj", "mlp_up_proj", "mlp_down_proj"]
    print(f"{'cell':<40}{'regime':<10}{'op':<16}{'n':>5}{'med_err%':>10}{'p90_err%':>10}")
    print("-" * 92)
    overall = {"decode": [], "prefill": []}
    for dev, model in CELLS:
        df = pd.read_csv(f"data/profiling/compute/{dev}/{model}/mlp.csv")
        tfl, bw = TFLOPS[dev], BW_BPS[dev]
        for regime, mask in [("decode", df.num_tokens <= 128), ("prefill", df.num_tokens > 128)]:
            sub = df[mask]
            if len(sub) == 0:
                continue
            for op in OPS:
                col = f"time_stats.{op}.median"
                errs = []
                for _, row in sub.iterrows():
                    we = gemm_specs(row)[op]
                    a = analytical_ms(we, int(row.num_tokens), tfl, bw)
                    actual = float(row[col])
                    if actual > 0:
                        errs.append(abs(a - actual) / actual * 100)
                if not errs:
                    continue
                errs.sort()
                med = errs[len(errs) // 2]
                p90 = errs[min(len(errs) - 1, int(len(errs) * 0.9))]
                overall[regime].extend(errs)
                print(f"{dev+'/'+model.split('/')[-1]:<40}{regime:<10}{op:<16}{len(errs):>5}{med:>9.1f}%{p90:>9.1f}%")
    print("-" * 92)
    for regime in ("decode", "prefill"):
        e = sorted(overall[regime])
        if e:
            med = e[len(e)//2]; p90 = e[min(len(e)-1,int(len(e)*0.9))]; mx = e[-1]
            print(f"OVERALL {regime:<8}: median {med:.1f}%   p90 {p90:.1f}%   max {mx:.1f}%   (n={len(e)})")


if __name__ == "__main__":
    main()
