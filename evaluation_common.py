#!/usr/bin/env python3
"""Shared, paper-facing evaluation contract.

Every sensitivity and ablation driver imports this file so hardware geometry,
capacity checks, HBM placement, tensor parallelism, batch choices, and SLO
selection cannot silently diverge between figures.

Units:
  * capacities use decimal GB where Vidur device SKUs do;
  * bandwidth uses GB/s (numerically bytes/ns);
  * NAND geometry uses bytes and ns.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from functools import lru_cache

from sweep_capacity import HBF_GB, HBM_GB, kv_bytes, weight_bytes
from vidur.config.model_config import BaseModelConfig


@dataclass(frozen=True)
class EvaluationPlatform:
    device: str = "blackwell"
    gpus_per_node: int = 8
    hbm_gb_per_gpu: float = 192.0
    hbm_bw_gbps_per_gpu: float = 8000.0
    hbf_gb_per_gpu: float = 3072.0
    hbf_stacks_per_gpu: int = 8
    hbf_planes_per_stack: int = 1024
    hbf_page_bytes: int = 4096
    hbf_nominal_tr_ns: float = 4096.0
    hbf_host_bw_cap_gbps: float = 8000.0
    activation_reserve_gb_per_gpu: float = 4.0
    kv_precision_bytes: int = 2
    tokens_per_selection_page: int = 16
    sparse_kv_fraction: float = 0.10
    tpot_slo_ms: float = 100.0

    @property
    def hbf_total_planes_per_gpu(self) -> int:
        return self.hbf_stacks_per_gpu * self.hbf_planes_per_stack

    def media_bw_gbps(self, tr_ns: float | None = None) -> float:
        """Raw aggregate media bandwidth across the eight HBF stacks."""
        tr = self.hbf_nominal_tr_ns if tr_ns is None else float(tr_ns)
        return self.hbf_total_planes_per_gpu * self.hbf_page_bytes / tr

    def effective_hbf_bw_gbps(self, tr_ns: float | None = None) -> float:
        """Host-visible bandwidth: min(media capability, 8 TB/s link cap)."""
        return min(self.media_bw_gbps(tr_ns), self.hbf_host_bw_cap_gbps)

    def manifest(self) -> dict:
        d = asdict(self)
        d["hbf_total_planes_per_gpu"] = self.hbf_total_planes_per_gpu
        d["hbf_nominal_media_bw_gbps"] = self.media_bw_gbps()
        d["hbf_nominal_effective_bw_gbps"] = self.effective_hbf_bw_gbps()
        d["capacity_policy"] = {
            "weights": "HBM only, sharded by TP",
            "activations": "4 GB/GPU HBM reserve",
            "kv": "remaining HBM first; overflow in HBF, sharded by TP",
            "hbf_centroid_metadata": (
                "one FP16 d-vector per 16-token K page; "
                "6.25% of K or 3.125% of K+V"
            ),
        }
        d["operating_point_policy"] = (
            "For each model/context/system, maximize batch*1000/(TP*TPOT_p50_ms) "
            "over batch and TP subject to TPOT_p50 <= 100 ms and capacity."
        )
        return d


PLATFORM = EvaluationPlatform()
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
TP_CHOICES = (1, 2, 4, 8)
TR_SENSITIVITY_NS = (2048.0, 4096.0, 8192.0, 16384.0)


@lru_cache(maxsize=None)
def model_geometry(model: str) -> tuple[int, int, int, int]:
    c = BaseModelConfig.create_from_name(model)
    tp_cap = 1 if getattr(c, "no_tensor_parallel", False) else min(
        PLATFORM.gpus_per_node, c.num_kv_heads
    )
    return c.num_layers, c.num_kv_heads, c.head_size(), tp_cap


def weights_fit_hbm(model: str, tp: int) -> bool:
    """HBF stores KV, not model weights."""
    used = weight_bytes(model) / tp / 1e9 + PLATFORM.activation_reserve_gb_per_gpu
    return used <= PLATFORM.hbm_gb_per_gpu


def capacity_feasible(model: str, batch: int, context: int, tp: int, tier: str) -> bool:
    """Exact capacity rule used before an operating point can be selected."""
    if tp < 1 or tp > PLATFORM.gpus_per_node or tp > model_geometry(model)[3]:
        return False
    if not weights_fit_hbm(model, tp):
        return False
    weights_gb_per_gpu = weight_bytes(model) / tp / 1e9
    kv_gb_per_gpu = kv_bytes(model, batch, context) / tp / 1e9
    hbm_kv_capacity_gb = (
        PLATFORM.hbm_gb_per_gpu
        - weights_gb_per_gpu
        - PLATFORM.activation_reserve_gb_per_gpu
    )
    if tier == "hbm":
        return kv_gb_per_gpu <= hbm_kv_capacity_gb
    cold_kv_gb = max(0.0, kv_gb_per_gpu - hbm_kv_capacity_gb)
    centroid_fraction_of_kv = (
        1.0 / (2.0 * PLATFORM.tokens_per_selection_page)
    )
    return (
        cold_kv_gb * (1.0 + centroid_fraction_of_kv)
        <= PLATFORM.hbf_gb_per_gpu
    )


def feasible_tps(model: str, batch: int, context: int, tier: str = "hbf") -> list[int]:
    return [
        tp for tp in TP_CHOICES
        if capacity_feasible(model, batch, context, tp, tier)
    ]


def hbm_kv_fraction(
    model: str, batch: int, context: int, tp: int, tier: str = "hbf"
) -> float:
    if tier == "hbm":
        return 1.0
    n_layers, n_kv_heads, head_dim, _ = model_geometry(model)
    total_kv = (
        batch * context * n_layers * 2 * n_kv_heads * head_dim
        * PLATFORM.kv_precision_bytes
    )
    kv_per_gpu = total_kv / tp
    available = (
        PLATFORM.hbm_gb_per_gpu * 1e9
        - weight_bytes(model) / tp
        - PLATFORM.activation_reserve_gb_per_gpu * 1e9
    )
    if kv_per_gpu <= 0:
        return 1.0
    return max(0.0, min(1.0, available / kv_per_gpu))


def throughput_per_gpu(batch: int, tp: int, tpot_ms: float) -> float:
    return batch * 1000.0 / (tp * tpot_ms)


def best_slo_point(rows: list[dict], slo_ms: float | None = None) -> dict | None:
    slo = PLATFORM.tpot_slo_ms if slo_ms is None else float(slo_ms)
    candidates = [
        r for r in rows
        if r.get("status") == "OK"
        and float(r["tpot_p50_ms"]) <= slo
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda r: throughput_per_gpu(
            int(r["batch"]), int(r["tp"]), float(r["tpot_p50_ms"])
        ),
    )
