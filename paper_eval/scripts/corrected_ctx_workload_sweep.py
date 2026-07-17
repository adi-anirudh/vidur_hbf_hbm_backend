#!/usr/bin/env python3
"""Corrected context x workload sweep for heterogeneous HBM/HBF MoE serving.

This is intentionally independent from the legacy paper_eval evaluators.  It models:

* DeepSeek-V3's real MLA projection geometry (absorbed MLA decode path).
* The three dense FFN layers and the vocabulary projection.
* An explicit per-pool capacity ledger.
* Balanced, fixed expert placement as the static baseline.
* Per-layer integer weighted scheduling as the dynamic baseline.
* Global gate-mass pruning with every token's top-1 expert protected.
* Layer-synchronous MoE latency: sum_l(max_gpu(load[l, gpu])).
* Integer attention/MoE GPU splits under a fixed total GPU budget.

The router and gate scores are synthetic.  Pruning quality, scheduler overhead, kernel
efficiency, topology effects, and HBF energy are not measured; the report labels these
limitations.  The output is an analytical design-space study, not a hardware result.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Model and hardware
# ---------------------------------------------------------------------------

# Official DeepSeek-V3 671B geometry.
D = 7168
N_LAYERS = 61
N_DENSE = 3
N_MOE = N_LAYERS - N_DENSE
N_HEADS = 128
Q_LORA = 1536
KV_LORA = 512
QK_NOPE = 128
QK_ROPE = 64
V_HEAD = 128
DENSE_INTERMEDIATE = 18432
MOE_INTERMEDIATE = 2048
N_EXPERTS = 256
TOP_K = 8
N_SHARED_EXPERTS = 1
VOCAB = 129280

# The study explicitly uses FP8 weights and FP8 compressed MLA KV.
WEIGHT_BYTES = 1
KV_BYTES_PER_TOKEN_LAYER = (KV_LORA + QK_ROPE)  # 576 B, FP8
ACT_BYTES = 2  # bf16 dispatch/combine payload

# User-selected design point: equal memory bandwidths.
BW_HBM_GBPS = 1024.0
BW_HBF_GBPS = 1024.0
LINK_BW_GBPS = 900.0
GPU_FLOPS = 1_000e12  # 1 PFLOP/s bf16/fp16 analytical peak

HBM_CAP_GB = 72.0
HBF_CAP_GB = 722.0
POOL_RUNTIME_GB = 4.0

# MLA projection weights from the official absorbed-MLA implementation:
# wq_a, wq_b, wkv_a, wkv_b, wo.
ATTN_WEIGHT_BYTES_LAYER = WEIGHT_BYTES * (
    D * Q_LORA
    + Q_LORA * N_HEADS * (QK_NOPE + QK_ROPE)
    + D * (KV_LORA + QK_ROPE)
    + KV_LORA * N_HEADS * (QK_NOPE + V_HEAD)
    + N_HEADS * V_HEAD * D
)
DENSE_WEIGHT_BYTES_LAYER = WEIGHT_BYTES * 3 * D * DENSE_INTERMEDIATE
PER_EXPERT_BYTES = WEIGHT_BYTES * 3 * D * MOE_INTERMEDIATE
SHARED_WEIGHT_BYTES_LAYER = N_SHARED_EXPERTS * PER_EXPERT_BYTES
ROUTED_EXPERT_BYTES_TOTAL = PER_EXPERT_BYTES * N_EXPERTS * N_MOE
SHARED_EXPERT_BYTES_TOTAL = SHARED_WEIGHT_BYTES_LAYER * N_MOE
EMBED_WEIGHT_BYTES = WEIGHT_BYTES * VOCAB * D
LM_HEAD_WEIGHT_BYTES = WEIGHT_BYTES * VOCAB * D

BASE_WEIGHT_BYTES = (
    ATTN_WEIGHT_BYTES_LAYER * N_LAYERS
    + DENSE_WEIGHT_BYTES_LAYER * N_DENSE
    + EMBED_WEIGHT_BYTES
    + LM_HEAD_WEIGHT_BYTES
)

# Synthetic router, retained from the previous study so changes isolate the evaluator.
ROUTER_GROUPS = 16
ROUTER_HOT = 32
ROUTER_STRUCTURE = 0.9
GATE_LOGIT_SIGMA = 1.5
ROUTING_TRIALS = 4

CONTEXTS = (8192, 16384, 32768, 65536, 131072)
CONCURRENCIES = (32, 64, 128, 256, 512, 1024)
RETAINED_MASS = (1.00, 0.99, 0.98, 0.95, 0.90)
SLOS_MS = (100.0, 200.0)
MAX_TOTAL_GPUS = 256
MAX_MOE_GPUS = 64

SYSTEM_HBM = "HBM-colocated-static"
SYSTEM_HBF = "HBF-colocated-static"
SYSTEM_STATIC = "HBM-attn/HBF-MoE-static"
SYSTEM_DYNAMIC = "HBM-attn/HBF-MoE-dynamic"


def bytes_to_ms(num_bytes: float, bw_gbps: float) -> float:
    """GB/s is numerically bytes/ns; convert ns to ms."""
    return num_bytes / bw_gbps / 1e6


def flops_to_ms(flops: float) -> float:
    return flops / (GPU_FLOPS / 1e3)


def gb(num_bytes: float) -> float:
    return num_bytes / 1e9


def kv_session_bytes(context: int) -> int:
    return context * KV_BYTES_PER_TOKEN_LAYER * N_LAYERS


def attention_capacity_sessions(context: int, capacity_gb: float) -> int:
    available = capacity_gb - POOL_RUNTIME_GB - gb(BASE_WEIGHT_BYTES)
    return max(0, math.floor(available / gb(kv_session_bytes(context))))


def colocated_capacity_ok(context: int, batch_per_gpu: int, n_gpus: int, cap_gb: float) -> bool:
    # Attention/dense/embedding/head are sequence-DP copies. Routed experts are
    # balanced shards; the shared expert is replicated on every worker.
    used = (
        POOL_RUNTIME_GB
        + gb(BASE_WEIGHT_BYTES)
        + gb(SHARED_EXPERT_BYTES_TOTAL)
        + gb(ROUTED_EXPERT_BYTES_TOTAL) / n_gpus
        + gb(kv_session_bytes(context)) * batch_per_gpu
    )
    return used <= cap_gb + 1e-12


def moe_capacity_ok(n_gpus: int, dynamic: bool) -> bool:
    routed = gb(ROUTED_EXPERT_BYTES_TOTAL) if dynamic else gb(ROUTED_EXPERT_BYTES_TOTAL) / n_gpus
    used = POOL_RUNTIME_GB + gb(SHARED_EXPERT_BYTES_TOTAL) + routed
    return used <= HBF_CAP_GB + 1e-12


def attention_stage_ms(context: int, batch_per_gpu: int, bw_gbps: float) -> float:
    """Per-decode-step attention-pool time on one sequence-DP worker.

    Each transformer layer is a roofline max of weight+KV bytes and absorbed-MLA
    FLOPs. Dense FFNs and the LM head are separate roofline operations.
    """
    b = batch_per_gpu
    layer_mem = bytes_to_ms(
        ATTN_WEIGHT_BYTES_LAYER + b * context * KV_BYTES_PER_TOKEN_LAYER,
        bw_gbps,
    )

    projection_flops = 2 * ATTN_WEIGHT_BYTES_LAYER / WEIGHT_BYTES * b
    # Absorbed MLA: Q_nope dot latent KV + Q_rope dot PE, then probabilities
    # times latent values. Softmax is added as ~5 FLOPs per score.
    context_flops = (
        2 * b * context * N_HEADS * (KV_LORA + QK_ROPE + KV_LORA)
        + 5 * b * context * N_HEADS
    )
    layer_compute = flops_to_ms(projection_flops + context_flops)
    attention = N_LAYERS * max(layer_mem, layer_compute)

    dense_mem = bytes_to_ms(DENSE_WEIGHT_BYTES_LAYER, bw_gbps)
    dense_compute = flops_to_ms(2 * DENSE_WEIGHT_BYTES_LAYER / WEIGHT_BYTES * b)
    dense = N_DENSE * max(dense_mem, dense_compute)

    head_mem = bytes_to_ms(LM_HEAD_WEIGHT_BYTES, bw_gbps)
    head_compute = flops_to_ms(2 * (LM_HEAD_WEIGHT_BYTES / WEIGHT_BYTES) * b)
    return attention + dense + max(head_mem, head_compute)


# ---------------------------------------------------------------------------
# Synthetic routing and pruning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LayerRouting:
    routes: tuple[tuple[int, ...], ...]
    gate_weights: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class PrunedLayer:
    routes: tuple[tuple[int, ...], ...]
    token_counts: tuple[int, ...]
    retained: tuple[int, ...]


def generate_layer_routing(sessions: int, rng: np.random.Generator) -> LayerRouting:
    hot = [rng.choice(N_EXPERTS, ROUTER_HOT, replace=False) for _ in range(ROUTER_GROUPS)]
    routes: list[tuple[int, ...]] = []
    weights: list[tuple[float, ...]] = []
    for _ in range(sessions):
        group = int(rng.integers(ROUTER_GROUPS))
        chosen: set[int] = set()
        while len(chosen) < TOP_K:
            if rng.random() < ROUTER_STRUCTURE:
                chosen.add(int(rng.choice(hot[group])))
            else:
                chosen.add(int(rng.integers(N_EXPERTS)))
        expert_ids = tuple(sorted(chosen))
        logits = rng.normal(0.0, GATE_LOGIT_SIGMA, TOP_K)
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        routes.append(expert_ids)
        weights.append(tuple(float(x) for x in probs))
    return LayerRouting(tuple(routes), tuple(weights))


@lru_cache(maxsize=None)
def routing_samples(sessions: int) -> tuple[tuple[LayerRouting, ...], ...]:
    trials: list[tuple[LayerRouting, ...]] = []
    for trial in range(ROUTING_TRIALS):
        rng = np.random.default_rng(11_000_003 + sessions * 101 + trial * 1_000_003)
        trials.append(tuple(generate_layer_routing(sessions, rng) for _ in range(N_MOE)))
    return tuple(trials)


def prune_layer(layer: LayerRouting, retained_mass: float) -> PrunedLayer:
    mass = np.zeros(N_EXPERTS, dtype=np.float64)
    protected: set[int] = set()
    for experts, weights in zip(layer.routes, layer.gate_weights):
        top = int(np.argmax(weights))
        protected.add(experts[top])
        for expert, weight in zip(experts, weights):
            mass[expert] += weight

    active = np.flatnonzero(mass > 0)
    if retained_mass >= 1.0 - 1e-12:
        retained = set(int(x) for x in active)
    else:
        retained = set()
        accumulated = 0.0
        target = retained_mass * float(mass.sum())
        for expert in active[np.argsort(-mass[active], kind="stable")]:
            retained.add(int(expert))
            accumulated += float(mass[expert])
            if accumulated >= target:
                break
        retained.update(protected)

    counts = np.zeros(N_EXPERTS, dtype=np.int64)
    pruned_routes: list[tuple[int, ...]] = []
    for experts in layer.routes:
        kept = tuple(expert for expert in experts if expert in retained)
        # Top-1 protection guarantees non-empty routes.
        assert kept
        pruned_routes.append(kept)
        for expert in kept:
            counts[expert] += 1
    return PrunedLayer(
        routes=tuple(pruned_routes),
        token_counts=tuple(int(x) for x in counts),
        retained=tuple(sorted(retained)),
    )


@lru_cache(maxsize=None)
def pruned_samples(sessions: int, retained_mass_milli: int) -> tuple[tuple[PrunedLayer, ...], ...]:
    retained_mass = retained_mass_milli / 1000.0
    return tuple(
        tuple(prune_layer(layer, retained_mass) for layer in trial)
        for trial in routing_samples(sessions)
    )


@lru_cache(maxsize=None)
def fixed_owner(layer_index: int, n_gpus: int) -> tuple[int, ...]:
    """Balanced fixed placement: every shard has floor/ceil(E/M) experts."""
    permutation = np.random.default_rng(91_000_019 + layer_index * 65_537).permutation(N_EXPERTS)
    owner = np.empty(N_EXPERTS, dtype=np.int64)
    owner[permutation] = np.arange(N_EXPERTS) % n_gpus
    return tuple(int(x) for x in owner)


def dynamic_lpt_owner(layer: PrunedLayer, n_gpus: int, bw_gbps: float) -> tuple[int, ...]:
    """Integer weighted longest-processing-time assignment for retained experts."""
    jobs: list[tuple[float, int]] = []
    for expert in layer.retained:
        read = bytes_to_ms(PER_EXPERT_BYTES, bw_gbps)
        compute = flops_to_ms(layer.token_counts[expert] * 6 * D * MOE_INTERMEDIATE)
        jobs.append((read + compute, expert))
    jobs.sort(reverse=True)
    loads = [0.0] * n_gpus
    owner = [-1] * N_EXPERTS
    for cost, expert in jobs:
        gpu = min(range(n_gpus), key=lambda idx: (loads[idx], idx))
        owner[expert] = gpu
        loads[gpu] += cost
    return tuple(owner)


@dataclass(frozen=True)
class MoeProfile:
    execution_ms: float
    aggregate_visit_bytes: float
    mean_retained_experts: float
    mean_routes_per_token: float

    def stage_ms(self, attention_gpus: int, moe_gpus: int) -> float:
        links = max(1, min(attention_gpus, moe_gpus))
        return self.execution_ms + bytes_to_ms(self.aggregate_visit_bytes, links * LINK_BW_GBPS)


@lru_cache(maxsize=None)
def moe_profile(sessions: int, retained_mass_milli: int, n_gpus: int, dynamic: bool, bw_int: int) -> MoeProfile:
    bw_gbps = float(bw_int)
    trial_execution: list[float] = []
    trial_visit_bytes: list[float] = []
    retained_counts: list[int] = []
    route_counts: list[float] = []

    for trial in pruned_samples(sessions, retained_mass_milli):
        execution_sum = 0.0
        visit_bytes_sum = 0.0
        for layer_index, layer in enumerate(trial):
            owner = (
                dynamic_lpt_owner(layer, n_gpus, bw_gbps)
                if dynamic
                else fixed_owner(layer_index, n_gpus)
            )
            gpu_loads = [0.0] * n_gpus
            for expert in layer.retained:
                gpu = owner[expert]
                read = bytes_to_ms(PER_EXPERT_BYTES, bw_gbps)
                compute = flops_to_ms(layer.token_counts[expert] * 6 * D * MOE_INTERMEDIATE)
                gpu_loads[gpu] += read + compute

            # Shared expert is replicated; tokens are evenly divided across the pool.
            shared = bytes_to_ms(SHARED_WEIGHT_BYTES_LAYER, bw_gbps)
            shared += flops_to_ms((sessions / n_gpus) * 6 * D * MOE_INTERMEDIATE)
            execution_sum += max(gpu_loads) + shared

            visits = 0
            retained_routes = 0
            for token_route in layer.routes:
                token_owners = {owner[expert] for expert in token_route}
                visits += len(token_owners)
                retained_routes += len(token_route)
            # Each owner visit sends an activation and returns a combined partial.
            visit_bytes_sum += 2 * visits * D * ACT_BYTES
            retained_counts.append(len(layer.retained))
            route_counts.append(retained_routes / sessions)
        trial_execution.append(execution_sum)
        trial_visit_bytes.append(visit_bytes_sum)

    return MoeProfile(
        execution_ms=float(np.mean(trial_execution)),
        aggregate_visit_bytes=float(np.mean(trial_visit_bytes)),
        mean_retained_experts=float(np.mean(retained_counts)),
        mean_routes_per_token=float(np.mean(route_counts)),
    )


# ---------------------------------------------------------------------------
# System enumeration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Point:
    context: int
    sessions: int
    retained_mass: float
    system: str
    gpus: int
    attention_gpus: int
    moe_gpus: int
    attention_ms: float
    moe_ms: float
    tbt_ms: float
    mean_retained_experts: float
    mean_routes_per_token: float


def colocated_point(context: int, sessions: int, n_gpus: int, hbf: bool) -> Point | None:
    if n_gpus < 1 or n_gpus > N_EXPERTS:
        return None
    batch = math.ceil(sessions / n_gpus)
    cap = HBF_CAP_GB if hbf else HBM_CAP_GB
    bw = BW_HBF_GBPS if hbf else BW_HBM_GBPS
    if not colocated_capacity_ok(context, batch, n_gpus, cap):
        return None
    profile = moe_profile(sessions, 1000, n_gpus, False, int(bw))
    attention = attention_stage_ms(context, batch, bw)
    moe = profile.stage_ms(n_gpus, n_gpus)
    return Point(
        context, sessions, 1.0, SYSTEM_HBF if hbf else SYSTEM_HBM,
        n_gpus, n_gpus, n_gpus, attention, moe, attention + moe,
        profile.mean_retained_experts, profile.mean_routes_per_token,
    )


def best_heterogeneous_point(
    context: int,
    sessions: int,
    retained_mass: float,
    total_gpus: int,
    dynamic: bool,
) -> Point | None:
    retained_mass_milli = int(round(retained_mass * 1000))
    best: Point | None = None
    max_m = min(MAX_MOE_GPUS, total_gpus - 1, N_EXPERTS)
    for moe_gpus in range(1, max_m + 1):
        attention_gpus = total_gpus - moe_gpus
        if attention_gpus < 1 or not moe_capacity_ok(moe_gpus, dynamic):
            continue
        cap_sessions = attention_capacity_sessions(context, HBM_CAP_GB)
        if cap_sessions < 1:
            continue
        batch = math.ceil(sessions / attention_gpus)
        if batch > cap_sessions:
            continue
        attention = attention_stage_ms(context, batch, BW_HBM_GBPS)
        profile = moe_profile(
            sessions, retained_mass_milli, moe_gpus, dynamic, int(BW_HBF_GBPS)
        )
        moe = profile.stage_ms(attention_gpus, moe_gpus)
        point = Point(
            context=context,
            sessions=sessions,
            retained_mass=retained_mass,
            system=SYSTEM_DYNAMIC if dynamic else SYSTEM_STATIC,
            gpus=total_gpus,
            attention_gpus=attention_gpus,
            moe_gpus=moe_gpus,
            attention_ms=attention,
            moe_ms=moe,
            tbt_ms=max(attention, moe),
            mean_retained_experts=profile.mean_retained_experts,
            mean_routes_per_token=profile.mean_routes_per_token,
        )
        if best is None or (point.tbt_ms, point.moe_gpus) < (best.tbt_ms, best.moe_gpus):
            best = point
    return best


def enumerate_points() -> list[Point]:
    points: list[Point] = []
    for sessions in CONCURRENCIES:
        print(f"precomputing routing/MoE profiles: S={sessions}", flush=True)
        # Force profiles once; context length does not affect MoE routing/work.
        for retained in RETAINED_MASS:
            milli = int(round(retained * 1000))
            for n_gpus in range(1, MAX_MOE_GPUS + 1):
                moe_profile(sessions, milli, n_gpus, False, int(BW_HBF_GBPS))
                moe_profile(sessions, milli, n_gpus, True, int(BW_HBF_GBPS))

        for context in CONTEXTS:
            print(f"  enumerating C={context // 1024}K", flush=True)
            for gpus in range(2, MAX_TOTAL_GPUS + 1):
                for hbf in (False, True):
                    point = colocated_point(context, sessions, gpus, hbf)
                    if point is not None:
                        points.append(point)
                for retained in RETAINED_MASS:
                    for dynamic in (False, True):
                        point = best_heterogeneous_point(
                            context, sessions, retained, gpus, dynamic
                        )
                        if point is not None:
                            points.append(point)
    return points


def point_key(point: Point) -> tuple[int, int, float, str, int]:
    return (point.context, point.sessions, point.retained_mass, point.system, point.gpus)


def write_csv(points: Sequence[Point], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "context_tokens", "context_k", "sessions", "retained_mass", "system",
        "gpus", "attention_gpus", "moe_gpus", "attention_ms", "moe_ms",
        "tbt_ms", "mean_retained_experts", "mean_routes_per_token",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for p in points:
            writer.writerow({
                "context_tokens": p.context,
                "context_k": p.context // 1024,
                "sessions": p.sessions,
                "retained_mass": f"{p.retained_mass:.2f}",
                "system": p.system,
                "gpus": p.gpus,
                "attention_gpus": p.attention_gpus,
                "moe_gpus": p.moe_gpus,
                "attention_ms": f"{p.attention_ms:.6f}",
                "moe_ms": f"{p.moe_ms:.6f}",
                "tbt_ms": f"{p.tbt_ms:.6f}",
                "mean_retained_experts": f"{p.mean_retained_experts:.6f}",
                "mean_routes_per_token": f"{p.mean_routes_per_token:.6f}",
            })


def best_at_or_below(points: Iterable[Point], budget: int) -> Point | None:
    eligible = [p for p in points if p.gpus <= budget]
    return min(eligible, key=lambda p: (p.tbt_ms, p.gpus)) if eligible else None


def min_at_slo(points: Iterable[Point], slo_ms: float) -> Point | None:
    eligible = [p for p in points if p.tbt_ms <= slo_ms]
    return min(eligible, key=lambda p: (p.gpus, p.tbt_ms)) if eligible else None


def group_points(points: Sequence[Point]) -> dict[tuple[int, int, float, str], list[Point]]:
    grouped: dict[tuple[int, int, float, str], list[Point]] = {}
    for point in points:
        grouped.setdefault(
            (point.context, point.sessions, point.retained_mass, point.system), []
        ).append(point)
    return grouped


def choose_fixed_budgets(grouped: dict[tuple[int, int, float, str], list[Point]]) -> dict[int, int]:
    """Use the fair static heterogeneous baseline at 32K and 100 ms.

    The budget is rounded up to a multiple of four.  If 100 ms is unreachable,
    use the minimum-latency point's GPU count, also rounded up.
    """
    budgets: dict[int, int] = {}
    reference_context = 32768
    for sessions in CONCURRENCIES:
        candidates = grouped[(reference_context, sessions, 1.0, SYSTEM_STATIC)]
        at_slo = min_at_slo(candidates, 100.0)
        chosen = at_slo.gpus if at_slo else min(candidates, key=lambda p: p.tbt_ms).gpus
        budgets[sessions] = min(MAX_TOTAL_GPUS, max(4, math.ceil(chosen / 4) * 4))
    return budgets


def fmt_point(point: Point | None) -> str:
    if point is None:
        return "—"
    if point.system in (SYSTEM_HBM, SYSTEM_HBF):
        return f"{point.tbt_ms:.1f} ({point.gpus}G)"
    return f"{point.tbt_ms:.1f} ({point.attention_gpus}A:{point.moe_gpus}M)"


def write_markdown(points: Sequence[Point], path: Path, budgets: dict[int, int]) -> None:
    grouped = group_points(points)
    lines: list[str] = []
    lines.extend([
        "# Corrected context × workload sweep",
        "",
        "> Fresh analytical generator; legacy `master_sweep.md` was not read or overwritten.",
        "> Routing and pruning are synthetic. These are design-space results, not measured hardware results.",
        "",
        "## Scope and corrections",
        "",
        f"- Contexts: `{', '.join(str(c // 1024) + 'K' for c in CONTEXTS)}`.",
        f"- Concurrent decode sessions: `{', '.join(map(str, CONCURRENCIES))}`.",
        f"- Retained gate mass: `{', '.join(f'{c:.2f}' for c in RETAINED_MASS)}`; every token's top-1 is protected.",
        f"- HBM/HBF bandwidth: `{BW_HBM_GBPS:.0f}/{BW_HBF_GBPS:.0f} GB/s`; link: `{LINK_BW_GBPS:.0f} GB/s`.",
        "- Static EP uses balanced fixed integer placement, never independent random ownership.",
        "- Dynamic EP uses integer weighted LPT assignment after routing/pruning.",
        "- MoE latency is the sum of the per-layer straggler times.",
        "- Attention uses official DeepSeek-V3 absorbed-MLA geometry and a memory/compute roofline.",
        "- Three dense FFNs, embeddings, LM head, shared experts, and runtime capacity are explicit.",
        "- Heterogeneous systems use steady-state `max(T_attention, T_MoE)` overlap.",
        "",
        "## Capacity ledger",
        "",
        "| Component | Size | Placement |",
        "|---|---:|---|",
        f"| MLA weights, 61 layers | {gb(ATTN_WEIGHT_BYTES_LAYER * N_LAYERS):.2f} GB | attention/colocated GPU |",
        f"| Dense FFNs, 3 layers | {gb(DENSE_WEIGHT_BYTES_LAYER * N_DENSE):.2f} GB | attention/colocated GPU |",
        f"| Embedding + LM head | {gb(EMBED_WEIGHT_BYTES + LM_HEAD_WEIGHT_BYTES):.2f} GB | attention/colocated GPU |",
        f"| Routed experts, 58 layers | {gb(ROUTED_EXPERT_BYTES_TOTAL):.2f} GB | sharded static / fully replicated dynamic |",
        f"| Shared expert, 58 layers | {gb(SHARED_EXPERT_BYTES_TOTAL):.2f} GB | replicated on MoE GPUs |",
        f"| Runtime reserve | {POOL_RUNTIME_GB:.2f} GB | each pool GPU |",
        "",
        "HBM attention capacity in sessions per GPU:",
        "",
        "| Context | Sessions/GPU |",
        "|---:|---:|",
    ])
    for context in CONTEXTS:
        lines.append(f"| {context // 1024}K | {attention_capacity_sessions(context, HBM_CAP_GB)} |")

    lines.extend([
        "",
        "## Deterministic fixed budgets",
        "",
        "Budgets are the smallest multiple of four at which the **unpruned balanced-static heterogeneous baseline** reaches 100 ms at 32K. They are selected without looking at dynamic/pruned results.",
        "",
        "| Sessions | Budget |",
        "|---:|---:|",
    ])
    for sessions, budget in budgets.items():
        lines.append(f"| {sessions} | {budget} |")

    lines.extend([
        "",
        "## Best TBT within the fixed budget",
        "",
        "Cells show `TBT ms (A:M)` for heterogeneous systems and `TBT ms (total G)` for colocated systems.",
        "",
        "| Ctx | S | Budget | HBM coloc | HBF coloc | Static c=1.00 | Dynamic c=1.00 | Static c=.98 | Dynamic c=.98 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for context in CONTEXTS:
        for sessions in CONCURRENCIES:
            budget = budgets[sessions]
            cells = []
            for retained, system in (
                (1.0, SYSTEM_HBM), (1.0, SYSTEM_HBF),
                (1.0, SYSTEM_STATIC), (1.0, SYSTEM_DYNAMIC),
                (0.98, SYSTEM_STATIC), (0.98, SYSTEM_DYNAMIC),
            ):
                cells.append(fmt_point(best_at_or_below(grouped.get((context, sessions, retained, system), []), budget)))
            lines.append(
                f"| {context // 1024}K | {sessions} | {budget} | " + " | ".join(cells) + " |"
            )

    for slo_ms in SLOS_MS:
        lines.extend([
            "",
            f"## Minimum GPUs at {slo_ms:.0f} ms",
            "",
            "| Ctx | S | HBM coloc | HBF coloc | Static c=1.00 | Dynamic c=1.00 | Static c=.98 | Dynamic c=.98 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for context in CONTEXTS:
            for sessions in CONCURRENCIES:
                cells = []
                for retained, system in (
                    (1.0, SYSTEM_HBM), (1.0, SYSTEM_HBF),
                    (1.0, SYSTEM_STATIC), (1.0, SYSTEM_DYNAMIC),
                    (0.98, SYSTEM_STATIC), (0.98, SYSTEM_DYNAMIC),
                ):
                    point = min_at_slo(
                        grouped.get((context, sessions, retained, system), []), slo_ms
                    )
                    cells.append("—" if point is None else str(point.gpus))
                lines.append(
                    f"| {context // 1024}K | {sessions} | " + " | ".join(cells) + " |"
                )

    lines.extend([
        "",
        "## Pruning sensitivity at the fixed budgets",
        "",
        "Improvement compares dynamic against balanced static at the same retained-mass target and GPU budget.",
        "",
        "| Retained mass | Mean kept experts | Median TBT improvement | Max TBT improvement | Cases releasing ≥1 MoE GPU |",
        "|---:|---:|---:|---:|---:|",
    ])
    for retained in RETAINED_MASS:
        improvements: list[float] = []
        kept: list[float] = []
        released = 0
        for context in CONTEXTS:
            for sessions in CONCURRENCIES:
                budget = budgets[sessions]
                static = best_at_or_below(grouped[(context, sessions, retained, SYSTEM_STATIC)], budget)
                dynamic = best_at_or_below(grouped[(context, sessions, retained, SYSTEM_DYNAMIC)], budget)
                if static and dynamic:
                    improvements.append(100 * (1 - dynamic.tbt_ms / static.tbt_ms))
                    kept.append(dynamic.mean_retained_experts)
                    if dynamic.moe_gpus < static.moe_gpus:
                        released += 1
        lines.append(
            f"| {retained:.2f} | {np.mean(kept):.1f} | {np.median(improvements):.2f}% | "
            f"{np.max(improvements):.2f}% | {released}/{len(improvements)} |"
        )

    lines.extend([
        "",
        "## Limitations",
        "",
        "- Synthetic structured routing and synthetic softmax gate scores; no real DeepSeek trace yet.",
        "- Retained gate mass is not a quality guarantee; perplexity and task accuracy are unmeasured.",
        "- Dynamic scheduling overhead is currently zero, so dynamic rows are an achievable analytical bound.",
        "- Peak bandwidth/FLOPs rooflines omit kernel efficiency, contention, topology, and control overhead.",
        "- Full attention/MoE overlap is a steady-state pipeline assumption.",
        "- Both memory tiers intentionally use the user-selected equal 1024 GB/s design point.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def make_heatmaps(points: Sequence[Point], outdir: Path, budgets: dict[int, int]) -> None:
    grouped = group_points(points)
    outdir.mkdir(parents=True, exist_ok=True)
    contexts_k = [c // 1024 for c in CONTEXTS]
    sessions = list(CONCURRENCIES)

    for retained in (1.0, 0.98, 0.90):
        gain = np.full((len(sessions), len(CONTEXTS)), np.nan)
        moe_delta = np.full_like(gain, np.nan)
        for i, s in enumerate(sessions):
            for j, context in enumerate(CONTEXTS):
                budget = budgets[s]
                static = best_at_or_below(grouped[(context, s, retained, SYSTEM_STATIC)], budget)
                dynamic = best_at_or_below(grouped[(context, s, retained, SYSTEM_DYNAMIC)], budget)
                if static and dynamic:
                    gain[i, j] = 100 * (1 - dynamic.tbt_ms / static.tbt_ms)
                    moe_delta[i, j] = static.moe_gpus - dynamic.moe_gpus

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for ax, data, title, cmap in (
            (axes[0], gain, "Dynamic TBT improvement over balanced static (%)", "YlGn"),
            (axes[1], moe_delta, "MoE GPUs released to attention", "PuBu"),
        ):
            image = ax.imshow(data, aspect="auto", cmap=cmap)
            ax.set_xticks(range(len(contexts_k)), [f"{x}K" for x in contexts_k])
            ax.set_yticks(range(len(sessions)), sessions)
            ax.set_xlabel("context length")
            ax.set_ylabel("concurrent sessions")
            ax.set_title(title)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if np.isfinite(data[i, j]):
                        suffix = "%" if ax is axes[0] else ""
                        ax.text(j, i, f"{data[i, j]:.1f}{suffix}", ha="center", va="center", fontsize=8)
            fig.colorbar(image, ax=ax, shrink=0.8)
        fig.suptitle(f"Corrected heterogeneous sweep · retained mass c={retained:.2f}")
        fig.savefig(outdir / f"dynamic_vs_static_c{int(retained * 100):03d}.png", dpi=150)
        plt.close(fig)


def validate_constants() -> None:
    assert KV_BYTES_PER_TOKEN_LAYER == 576
    assert N_MOE == 58
    assert max(fixed_owner(0, 8).count(i) for i in range(8)) == 32
    assert min(fixed_owner(0, 8).count(i) for i in range(8)) == 32
    assert moe_capacity_ok(1, True), "full FP8 expert replica must fit in a 722 GB HBF pool GPU"
    assert attention_capacity_sessions(8192, HBM_CAP_GB) > attention_capacity_sessions(131072, HBM_CAP_GB)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parents[1] / "results"),
        help="directory for corrected CSV/Markdown outputs",
    )
    parser.add_argument(
        "--figure-dir",
        default=str(Path(__file__).resolve().parents[1] / "figures" / "corrected"),
        help="directory for corrected figures",
    )
    args = parser.parse_args()
    validate_constants()
    points = enumerate_points()
    grouped = group_points(points)
    budgets = choose_fixed_budgets(grouped)
    output_dir = Path(args.output_dir)
    csv_path = output_dir / "corrected_ctx_workload_sweep.csv"
    md_path = output_dir / "corrected_ctx_workload_sweep.md"
    write_csv(points, csv_path)
    write_markdown(points, md_path, budgets)
    make_heatmaps(points, Path(args.figure_dir), budgets)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")
    print(f"wrote figures under {args.figure_dir}")


if __name__ == "__main__":
    main()
