#!/usr/bin/env python3
"""Reproduce the HBF predictor's SRA flat-BW branch to pick a clean example for the
'sparse selection cannot hide inside HBM sliding' figure. No new model -- mirrors
hbf_execution_time_predictor._get_attention_decode_execution_time (backing_bw>0, sra)."""
import math
from vidur.config.model_config import BaseModelConfig
from sweep_capacity import weight_bytes, HBM_GB, HBF_GB

BLOCK = 16
HBM_BW = 8000.0      # blackwell mem_bandwidth_gbps (bytes/ns)
HBF_BW = 8000.0      # --backing_bw_gbps
AMP = 3.5636
SPARSITY = 0.1
NMP_FPB = 2.0        # nmp_flops_per_byte (for the honest compute-bound scan term)


def geom(model):
    c = BaseModelConfig.create_from_name(model)
    return c.num_layers, c.num_q_heads, c.num_kv_heads, c.head_size()


def hbm_kv_fraction(model, batch, ctx, tp):
    n_lay, _, n_kv, hd = geom(model)
    kv_bytes = batch * ctx * n_lay * 2 * n_kv * hd * 2
    avail = HBM_GB["blackwell"] * 1e9 - weight_bytes(model) / tp - 4e9
    kv_per_gpu = kv_bytes / tp
    return max(0.0, min(1.0, avail / kv_per_gpu)) if kv_per_gpu else 1.0


def times(model, batch, ctx, tp):
    n_lay, n_q, n_kv, hd = geom(model)
    frac = hbm_kv_fraction(model, batch, ctx, tp)
    tkv = batch * math.ceil(ctx / BLOCK)                 # blocks per layer
    hbm_blocks = max(1, round(tkv * frac))
    hbf_blocks = max(0, tkv - hbm_blocks)
    kv_block_bytes = (2 * BLOCK * n_kv * hd * 2) // tp
    half = kv_block_bytes / 2.0
    thbf = hbf_blocks * n_lay
    t_hbm = hbm_blocks * n_lay * kv_block_bytes / (HBM_BW * 1e6)          # ms, hot window read
    t_scan = thbf * half / (HBF_BW * 1e6)                                # ms, scan all cold K (read-bound)
    # honest compute-bound scan term (max in _hbf_sra_score_ms): GQA_ratio/nmp x read
    t_scan_comp = (2 * hd * max(1, n_q // tp) * thbf * BLOCK) / (NMP_FPB * HBF_BW * 1e6)
    t_read = thbf * min(1.0, SPARSITY * AMP) * half / (HBF_BW * 1e6)      # sparse V read after select
    return dict(frac=frac, tp=tp, t_hbm=t_hbm, t_scan=t_scan, t_scan_comp=t_scan_comp,
                t_read=t_read, idle=max(0.0, t_scan - t_hbm),
                idle_comp=max(0.0, max(t_scan, t_scan_comp) - t_hbm))


if __name__ == "__main__":
  CANDS = [
    ("meta-llama/Meta-Llama-3-8B", 8, 262144, 1),
    ("meta-llama/Meta-Llama-3-8B", 16, 524288, 1),
    ("meta-llama/Meta-Llama-3-8B", 8, 1048576, 1),
    ("deepseek-ai/deepseek-llm-67b-chat", 16, 262144, 4),
    ("meta-llama/Meta-Llama-3-70B", 16, 262144, 8),
    ("meta-llama/Meta-Llama-3.1-405B", 32, 262144, 8),
    ("meta-llama/Meta-Llama-3.1-405B", 16, 524288, 8),
  ]
  for m, b, ctx, tp in CANDS:
    r = times(m, b, ctx, tp)
    print(f"{m.split('/')[-1]:20} b={b:3} ctx={ctx//1024}K tp={tp}  "
          f"hotfrac={r['frac']*100:5.1f}%  t_hbm={r['t_hbm']*1000:7.1f}us  "
          f"t_scan={r['t_scan']*1000:8.1f}us  idle={r['idle']*1000:8.1f}us  "
          f"scan/hbm={r['t_scan']/r['t_hbm']:5.1f}x  (comp-bound scan={r['t_scan_comp']*1000:8.1f}us)")
