# -*- coding: utf-8 -*-
"""
Harvest project-vajra HF profiling datasets into Vidur's profiling schema.

Vajra publishes mlp.csv / attention.csv in a NEWER schema:
  mlp:        fused norms (input_norm_fused, post_attention_norm_fused),
              no `add`, no `use_gated_mlp`.
  attention:  single `attention_forward` (+ `save_kv_cache`) instead of
              attn_prefill / attn_decode / attn_kv_cache_save / reshapes.

This maps both to the Vidur columns the sklearn predictor expects:
  input_norm_fused          -> input_layernorm
  post_attention_norm_fused -> post_attention_layernorm
  add                       -> 0  (residual add; negligible, real serving fuses it)
  use_gated_mlp             -> from --gated
  attention_forward         -> attn_prefill (is_prefill rows) / attn_decode (else)
  save_kv_cache             -> attn_kv_cache_save
  attn_input/output_reshape -> 0

NB: for the decode-only sweep the HBF predictor OVERRIDES attn_decode + kv_save
(flash / HBM) and weight-load GEMMs (HBM bw); the live decode contribution of
this data is the kept norms/act/rope from mlp.csv — which are real vajra
measurements on the target GPU. So harvested decode TPOT is faithful.

Usage:
  python convert_vajra.py --dataset <vajra-ds> --device h200 --model meta-llama/Meta-Llama-3-8B --gated
"""
from __future__ import annotations
import argparse, io, lzma, os, urllib.request
import pandas as pd

BASE = "https://huggingface.co/datasets/project-vajra/{ds}/resolve/main/{f}"
REF = "data/profiling/compute/a100/meta-llama/Meta-Llama-3-8B"  # reference schema


def _read_xz(ds, fname):
    url = BASE.format(ds=ds, f=fname + ".xz")
    raw = urllib.request.urlopen(url, timeout=120).read()
    return pd.read_csv(io.BytesIO(lzma.decompress(raw)))


def _stats(prefix):
    return [f"time_stats.{prefix}.{s}" for s in ("min", "max", "mean", "median", "std")]


def convert_mlp(ds, gated):
    df = _read_xz(ds, "mlp.csv")
    # variant A: fused norms -> rename to Vidur names. variant B: already standard.
    if "time_stats.input_norm_fused.median" in df.columns:
        ren = {}
        for a, b in (("input_norm_fused", "input_layernorm"),
                     ("post_attention_norm_fused", "post_attention_layernorm")):
            for s in ("min", "max", "mean", "median", "std"):
                ren[f"time_stats.{a}.{s}"] = f"time_stats.{b}.{s}"
        df = df.rename(columns=ren)
    if "use_gated_mlp" not in df:
        df["use_gated_mlp"] = bool(gated)
    ref_cols = pd.read_csv(f"{REF}/mlp.csv", nrows=0).columns.tolist()
    for c in ref_cols:
        if c not in df:
            df[c] = 0.0   # e.g. `add` (residual; negligible, fused in real serving)
    return df[ref_cols]


def convert_attention(ds):
    df = _read_xz(ds, "attention.csv")
    isp = df["is_prefill"].astype(bool)
    if "time_stats.attention_forward.median" in df.columns:
        # variant A: single attention_forward -> split by is_prefill
        for s in ("min", "max", "mean", "median", "std"):
            fwd = f"time_stats.attention_forward.{s}"
            df[f"time_stats.attn_prefill.{s}"] = df[fwd].where(isp, 0.0)
            df[f"time_stats.attn_decode.{s}"]  = df[fwd].where(~isp, 0.0)
            if f"time_stats.save_kv_cache.{s}" in df:
                df[f"time_stats.attn_kv_cache_save.{s}"] = df[f"time_stats.save_kv_cache.{s}"]
    # variant B/C already have attn_prefill/attn_decode/attn_kv_cache_save; keep them.
    ref_cols = pd.read_csv(f"{REF}/attention.csv", nrows=0).columns.tolist()
    for c in ref_cols:
        if c not in df:
            df[c] = 0.0
    return df[ref_cols]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--gated", action="store_true")
    args = ap.parse_args()
    dst = f"data/profiling/compute/{args.device}/{args.model}"
    os.makedirs(dst, exist_ok=True)
    mlp = convert_mlp(args.dataset, args.gated)
    att = convert_attention(args.dataset)
    mlp.to_csv(f"{dst}/mlp.csv", index=False)
    att.to_csv(f"{dst}/attention.csv", index=False)
    print(f"OK {args.device}/{args.model}: mlp {len(mlp)} rows, attention {len(att)} rows -> {dst}")


if __name__ == "__main__":
    main()
