# -*- coding: utf-8 -*-
"""
Capacity-driven tensor-parallel (TP) selection for the HBF paper sweep.

For each (model, device, batch, context) point, pick the minimum TP in {1,2,4,8}
such that the combined per-GPU memory (weights + full KV cache, sharded by TP)
fits in TP x (HBM_per_gpu + HBF_per_gpu). Fixes the earlier bug where TP was
statically 1 and batch capped at 8: where HBM+HBF is not enough, TP scales up.

  per-GPU footprint = (weight_bytes + kv_bytes) / TP
  fits if           (weight_bytes + kv_bytes) <= TP x (HBM_gb + HBF_gb) x 1e9

KV cache can only shard across TP up to num_kv_heads, so TP is capped at
min(8, num_kv_heads). If no TP <= cap fits, the point is INFEASIBLE.
"""
from __future__ import annotations

from vidur.config.model_config import BaseModelConfig

# Per-GPU HBM (GB), from device_sku_config.py
HBM_GB = {"a40": 45, "a100": 80, "h100": 80, "h200": 141, "blackwell": 96}
# Per-GPU HBF stack (GB): 768 planes x 256 blocks x 256 pages x 4096 B  (hbf_paper.toml)
HBF_GB = 768 * 256 * 256 * 4096 / 1e9   # ~206.16

# Total stored parameters (for weight memory), fp16. From the workload set.
PARAMS = {
    "microsoft/phi-2":                    2.7e9,
    "mistralai/Mistral-7B-v0.1":          7.24e9,
    "meta-llama/Meta-Llama-3-8B":         8.03e9,
    "mistralai/Mixtral-8x7B-v0.1":        46.7e9,   # MoE: all experts resident
    "deepseek-ai/deepseek-llm-67b-chat":  67.0e9,
    "meta-llama/Meta-Llama-3-70B":        70.6e9,
    "Qwen/Qwen-72B":                      72.3e9,
    "Qwen/Qwen2-72B":                     72.7e9,
    # frontier / MoE additions (total stored params, fp16)
    "meta-llama/Meta-Llama-3.1-405B":          405.0e9,
    "meta-llama/Llama-3.3-70B-Instruct":       70.6e9,
    "mistralai/Mixtral-8x22B-v0.1":            141.0e9,
    "Qwen/Qwen3-235B-A22B":                    235.0e9,
    "Qwen/Qwen3-Coder-480B-A35B-Instruct":     480.0e9,
}

TP_CANDIDATES = [1, 2, 4, 8]
BYTES_FP16 = 2


def _dims(model: str):
    c = BaseModelConfig.create_from_name(model)
    head_dim = c.head_size()  # explicit head_dim if set (Qwen3), else emb//q
    return c.num_layers, c.num_kv_heads, head_dim


def kv_bytes(model: str, batch: int, ctx: int) -> int:
    n_lay, n_kv, head_dim = _dims(model)
    # K and V, fp16, all layers, all batch, full context
    return batch * ctx * n_lay * 2 * n_kv * head_dim * BYTES_FP16


def weight_bytes(model: str) -> int:
    return int(PARAMS[model] * BYTES_FP16)


def select_tp(model: str, device: str, batch: int, ctx: int):
    """Return (tp, footprint_gb, feasible). tp is None if infeasible."""
    _, n_kv, _ = _dims(model)
    total = weight_bytes(model) + kv_bytes(model, batch, ctx)
    cap_bytes = (HBM_GB[device] + HBF_GB) * 1e9
    # Models flagged no_tensor_parallel (e.g. phi-2) can't shard -> TP=1 only;
    # if they don't fit at TP=1 they're INFEASIBLE (not a higher TP). Otherwise
    # cap TP at the KV-head count (can't shard KV beyond num_kv_heads).
    if getattr(BaseModelConfig.create_from_name(model), "no_tensor_parallel", False):
        tp_cap = 1
    else:
        tp_cap = min(8, n_kv)
    for tp in TP_CANDIDATES:
        if tp > tp_cap:
            break
        if total <= tp * cap_bytes:
            return tp, total / 1e9, True
    return None, total / 1e9, False


if __name__ == "__main__":
    # sanity check: a few representative points
    print(f"HBF per GPU = {HBF_GB:.1f} GB")
    cases = [
        ("meta-llama/Meta-Llama-3-8B", "a100", 8, 4096),
        ("meta-llama/Meta-Llama-3-8B", "a100", 128, 131072),
        ("meta-llama/Meta-Llama-3-70B", "a40", 32, 262144),
        ("Qwen/Qwen2-72B", "h200", 128, 1048576),
        ("Qwen/Qwen2-72B", "a40", 128, 1048576),
        ("microsoft/phi-2", "a40", 1, 4096),
        ("mistralai/Mixtral-8x7B-v0.1", "h100", 64, 524288),
    ]
    print(f"{'model':<36}{'dev':>6}{'B':>5}{'ctx':>9}{'foot_GB':>10}{'TP':>5}{'feasible':>10}")
    for m, d, b, c in cases:
        tp, foot, ok = select_tp(m, d, b, c)
        print(f"{m:<36}{d:>6}{b:>5}{c:>9}{foot:>10.1f}{str(tp):>5}{str(ok):>10}")
