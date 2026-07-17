#!/usr/bin/env python3
"""Trace-backed hybrid HBM/HBF MoE balancer analysis.

This is an isolated design check for the "middle ground" system:

* attention stays on HBM GPUs;
* the MoE tier has M GPUs total;
* H of those M MoE GPUs are upgraded to HBF and hold the full routed expert set;
* the remaining HBM MoE GPUs hold a train-profiled static expert subset;
* HBF GPUs can absorb any expert job at each microbatch/layer/decode step.

The script reuses the no-pruning R1 selected-expert traces and the same raw
attention/MoE timing constants used by the Task-3 A/M selector. It does not run
Vidur and therefore reports raw pipeline-step estimates, not queueing p95.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import corrected_ctx_workload_sweep as base  # noqa: E402
from vidur.execution_time_predictor.moe_trace import (  # noqa: E402
    EXPERTS,
    LAYERS,
    MoETraceRouteStore,
)


DEFAULT_TRACE_ROOT = PAPER / "experiments" / "deepseek_r1_awq_no_prune_replay"
DEFAULT_CANDIDATES = (
    PAPER
    / "experiments"
    / "deepseek_r1_awq_task3_balanced_am"
    / "full_mu2_exact_jobs8"
    / "results"
    / "all_splits.csv"
)
DEFAULT_OUT = (
    PAPER
    / "experiments"
    / "deepseek_r1_awq_task3_hybrid_moe_balancers"
    / "initial_mu2_profiledA"
)

PROFILED_ATTENTION_GPUS = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16}
GIB = 1024**3
PER_EXPERT_COLLECTION_GIB = base.PER_EXPERT_BYTES * LAYERS / GIB
SHARED_COLLECTION_GIB = base.SHARED_WEIGHT_BYTES_LAYER * LAYERS / GIB
HBM_ROUTED_CAP_EXPERTS = max(
    0, int(math.floor((base.HBM_CAP_GB - SHARED_COLLECTION_GIB) / PER_EXPERT_COLLECTION_GIB))
)


def int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("empty integer list")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--candidates-csv", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--microbatch-count", type=int, default=2)
    parser.add_argument("--trace-max-requests", type=int, default=512)
    parser.add_argument("--trace-profile-max-requests", type=int, default=0)
    parser.add_argument("--hbf-labels", default="0,1,2,4,all")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--slo-ms", type=float, default=100.0)
    return parser.parse_args()


def microbatch_slices(sessions: int, microbatch_count: int) -> tuple[tuple[int, int], ...]:
    if not 1 <= microbatch_count <= sessions:
        raise ValueError(f"bad microbatch_count={microbatch_count} for sessions={sessions}")
    quotient, remainder = divmod(sessions, microbatch_count)
    bounds: list[tuple[int, int]] = []
    start = 0
    for index in range(microbatch_count):
        size = quotient + (1 if index < remainder else 0)
        bounds.append((start, start + size))
        start += size
    return tuple(bounds)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def hbf_labels_for(moe_gpus: int, requested: tuple[str, ...]) -> list[tuple[str, int]]:
    labels: list[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for label in requested:
        if label == "all":
            pair = ("all", moe_gpus)
        else:
            hbf_gpus = int(label)
            if hbf_gpus > moe_gpus:
                continue
            pair = (label, hbf_gpus)
        if pair not in seen:
            labels.append(pair)
            seen.add(pair)
    return labels


def profiled_hbm_owner(profile: np.ndarray, hbm_gpus: int) -> np.ndarray:
    """Capacity-limited train-profiled HBM ownership for one layer.

    Each HBM GPU can hold at most HBM_ROUTED_CAP_EXPERTS routed experts per
    layer, after reserving space for the shared expert collection. If the HBM
    side cannot cover all experts, cold experts remain unassigned and must run
    on HBF balancers.
    """

    owner = np.full(EXPERTS, -1, np.int16)
    if hbm_gpus <= 0 or HBM_ROUTED_CAP_EXPERTS <= 0:
        return owner

    remaining = np.full(hbm_gpus, HBM_ROUTED_CAP_EXPERTS, np.int32)
    loads = np.zeros(hbm_gpus, np.float64)
    for expert in np.lexsort((np.arange(EXPERTS), -profile)):
        candidates = np.flatnonzero(remaining > 0)
        if candidates.size == 0:
            break
        gpu = int(candidates[np.argmin(loads[candidates])])
        owner[int(expert)] = gpu
        remaining[gpu] -= 1
        loads[gpu] += float(profile[int(expert)])
    return owner


def raw_step(attention_ms: float, moe_ms: float, microbatch_count: int) -> float:
    if microbatch_count == 1:
        return attention_ms + moe_ms
    return max(attention_ms, moe_ms)


@dataclass(frozen=True)
class HybridTask:
    workload: str
    sessions: int
    attention_gpus: int
    moe_gpus: int
    hbf_label: str
    hbf_gpus: int


_ROUTES: np.ndarray | None = None
_PROFILE: np.ndarray | None = None
_DECODE_TOKENS = 0
_MICROBATCH_COUNT = 2
_WORKLOAD = ""
_OWNER_CACHE: dict[int, np.ndarray] = {}


def init_worker(
    trace_root: str,
    workload: str,
    max_requests: int,
    profile_max_requests: int,
    decode_tokens: int,
    microbatch_count: int,
) -> None:
    global _ROUTES, _PROFILE, _DECODE_TOKENS, _MICROBATCH_COUNT, _WORKLOAD, _OWNER_CACHE

    store = MoETraceRouteStore(
        root=trace_root,
        workload=workload,
        max_requests=max_requests,
        profile_max_requests=profile_max_requests,
    )
    if not store.routes:
        raise RuntimeError(f"no trace routes loaded for workload={workload}")
    used_decode = min(decode_tokens, min(route.route.shape[0] for route in store.routes))
    _ROUTES = np.stack(
        [route.route[:used_decode].astype(np.int16, copy=False) for route in store.routes],
        axis=0,
    )
    _PROFILE = store.profile.astype(np.float64, copy=True)
    _DECODE_TOKENS = used_decode
    _MICROBATCH_COUNT = microbatch_count
    _WORKLOAD = workload
    _OWNER_CACHE = {}


def hbm_owners_for(hbm_gpus: int) -> np.ndarray:
    assert _PROFILE is not None
    cached = _OWNER_CACHE.get(hbm_gpus)
    if cached is None:
        cached = np.stack(
            [profiled_hbm_owner(_PROFILE[layer], hbm_gpus) for layer in range(LAYERS)]
        )
        _OWNER_CACHE[hbm_gpus] = cached
    return cached


def assign_hybrid(
    counts: np.ndarray,
    hbm_owner: np.ndarray,
    moe_gpus: int,
    hbm_gpus: int,
    hbf_gpus: int,
    read_ms: float,
    compute_unit_ms: float,
) -> tuple[np.ndarray, np.ndarray, bool]:
    active = np.flatnonzero(counts)
    owner = np.full(EXPERTS, -1, np.int16)
    loads = np.zeros(moe_gpus, np.float64)
    infeasible = False

    if active.size == 0:
        return owner, loads, infeasible

    hbf_devices = np.arange(hbm_gpus, moe_gpus, dtype=np.int16)
    scores = np.where(counts > 0, read_ms + compute_unit_ms * counts, 0.0)
    order = active[np.lexsort((active, -counts[active], -scores[active]))]
    for expert in order:
        static_gpu = int(hbm_owner[int(expert)])
        if hbf_gpus == moe_gpus:
            candidates = np.arange(moe_gpus, dtype=np.int16)
        elif hbf_gpus == 0:
            if static_gpu < 0:
                infeasible = True
                continue
            candidates = np.array([static_gpu], dtype=np.int16)
        elif static_gpu >= 0:
            candidates = np.concatenate((np.array([static_gpu], dtype=np.int16), hbf_devices))
        else:
            candidates = hbf_devices

        candidate_loads = loads[candidates]
        best_load = float(candidate_loads.min())
        best_candidates = candidates[np.isclose(candidate_loads, best_load)]
        if static_gpu in best_candidates:
            gpu = static_gpu
        else:
            gpu = int(best_candidates[0])
        owner[int(expert)] = gpu
        loads[gpu] += float(scores[int(expert)])

    return owner, loads, infeasible


def worker(task_tuple: tuple[str, int, int, int, str, int]) -> dict[str, object]:
    workload, sessions, attention_gpus, moe_gpus, hbf_label, hbf_gpus = task_tuple
    assert _ROUTES is not None

    hbm_gpus = moe_gpus - hbf_gpus
    read_ms = base.bytes_to_ms(base.PER_EXPERT_BYTES, base.BW_HBF_GBPS)
    compute_unit_ms = base.flops_to_ms(6 * base.D * base.MOE_INTERMEDIATE)
    shared_read_ms = base.bytes_to_ms(base.SHARED_WEIGHT_BYTES_LAYER, base.BW_HBF_GBPS)
    links = max(1, min(attention_gpus, moe_gpus))

    hbm_owners = hbm_owners_for(hbm_gpus) if hbm_gpus > 0 else np.full(
        (LAYERS, EXPERTS), -1, np.int16
    )
    route_count = int(_ROUTES.shape[0])
    bounds = microbatch_slices(sessions, _MICROBATCH_COUNT)

    step_moe: list[float] = []
    step_execution: list[float] = []
    step_transfer: list[float] = []
    active_experts: list[float] = []
    max_gpu_experts: list[float] = []
    max_gpu_tokens: list[float] = []
    max_hbm_gpu_experts: list[float] = []
    max_hbf_gpu_experts: list[float] = []
    hbf_expert_fraction: list[float] = []
    hbf_load_fraction: list[float] = []
    owner_visits: list[float] = []
    infeasible_events = 0
    event_count = 0

    for decode_step in range(_DECODE_TOKENS):
        execution_sum = 0.0
        transfer_bytes_sum = 0.0
        for start, end in bounds:
            request_ids = np.arange(start, end, dtype=np.int64) % route_count
            mb_size = int(end - start)
            selected = _ROUTES[request_ids, decode_step]
            for layer in range(LAYERS):
                routes = selected[:, layer, :]
                counts = np.bincount(routes.reshape(-1), minlength=EXPERTS)
                active = np.flatnonzero(counts)
                if active.size == 0:
                    continue
                event_count += 1

                owner, _, infeasible = assign_hybrid(
                    counts,
                    hbm_owners[layer],
                    moe_gpus,
                    hbm_gpus,
                    hbf_gpus,
                    read_ms,
                    compute_unit_ms,
                )
                if infeasible or np.any(owner[active] < 0):
                    infeasible_events += 1
                    continue

                assigned = owner[active]
                token_loads = np.bincount(
                    assigned, weights=counts[active], minlength=moe_gpus
                ).astype(np.float64)
                expert_fetches = np.bincount(assigned, minlength=moe_gpus).astype(np.float64)
                gpu_loads = expert_fetches * read_ms + token_loads * compute_unit_ms
                shared = shared_read_ms + base.flops_to_ms(
                    (mb_size / moe_gpus) * 6 * base.D * base.MOE_INTERMEDIATE
                )
                execution_sum += float(gpu_loads.max() + shared)

                token_owners = owner[routes]
                visits = 0
                for row in token_owners:
                    visits += len(set(int(x) for x in row if x >= 0))
                transfer_bytes_sum += 2 * visits * base.D * base.ACT_BYTES

                active_experts.append(float(active.size))
                max_gpu_experts.append(float(expert_fetches.max()))
                max_gpu_tokens.append(float(token_loads.max()))
                hbm_fetches = expert_fetches[:hbm_gpus] if hbm_gpus else np.array([0.0])
                hbf_fetches = expert_fetches[hbm_gpus:] if hbf_gpus else np.array([0.0])
                max_hbm_gpu_experts.append(float(hbm_fetches.max()))
                max_hbf_gpu_experts.append(float(hbf_fetches.max()))
                hbf_expert_fraction.append(
                    float(hbf_fetches.sum() / expert_fetches.sum()) if expert_fetches.sum() else 0.0
                )
                hbf_load = float(gpu_loads[hbm_gpus:].sum()) if hbf_gpus else 0.0
                total_load = float(gpu_loads.sum())
                hbf_load_fraction.append(hbf_load / total_load if total_load else 0.0)
                owner_visits.append(float(visits / mb_size if mb_size else 0.0))

        transfer_ms = base.bytes_to_ms(transfer_bytes_sum, links * base.LINK_BW_GBPS)
        step_execution.append(execution_sum)
        step_transfer.append(transfer_ms)
        step_moe.append(execution_sum + transfer_ms)

    hbm_capacity_covers_all = hbm_gpus * HBM_ROUTED_CAP_EXPERTS >= EXPERTS
    hbf_full_replica_fits = (base.ROUTED_EXPERT_BYTES_TOTAL + base.SHARED_EXPERT_BYTES_TOTAL) / GIB <= (
        base.HBF_CAP_GB + 1e-9
    )
    capacity_ok = (
        hbf_full_replica_fits
        if hbf_gpus > 0
        else hbm_capacity_covers_all and infeasible_events == 0
    )

    return {
        "workload": workload,
        "sessions": sessions,
        "attention_gpus": attention_gpus,
        "moe_gpus": moe_gpus,
        "hbf_label": hbf_label,
        "hbf_gpus": hbf_gpus,
        "hbm_gpus": hbm_gpus,
        "hbm_routed_cap_experts_per_gpu": HBM_ROUTED_CAP_EXPERTS,
        "hbm_capacity_covers_all_experts": hbm_capacity_covers_all,
        "hbf_full_replica_fits": hbf_full_replica_fits,
        "capacity_ok": capacity_ok,
        "infeasible_events": infeasible_events,
        "event_count": event_count,
        "raw_moe_ms_mean": float(np.mean(step_moe)),
        "raw_moe_ms_p95": float(np.percentile(step_moe, 95)),
        "moe_execution_ms_mean": float(np.mean(step_execution)),
        "moe_transfer_ms_mean": float(np.mean(step_transfer)),
        "active_experts_mean": float(np.mean(active_experts)) if active_experts else 0.0,
        "max_gpu_experts_mean": float(np.mean(max_gpu_experts)) if max_gpu_experts else 0.0,
        "max_gpu_tokens_mean": float(np.mean(max_gpu_tokens)) if max_gpu_tokens else 0.0,
        "max_hbm_gpu_experts_p95": float(np.percentile(max_hbm_gpu_experts, 95))
        if max_hbm_gpu_experts
        else 0.0,
        "max_hbf_gpu_experts_p95": float(np.percentile(max_hbf_gpu_experts, 95))
        if max_hbf_gpu_experts
        else 0.0,
        "hbf_expert_fraction_mean": float(np.mean(hbf_expert_fraction))
        if hbf_expert_fraction
        else 0.0,
        "hbf_load_fraction_mean": float(np.mean(hbf_load_fraction)) if hbf_load_fraction else 0.0,
        "owner_visits_mean": float(np.mean(owner_visits)) if owner_visits else 0.0,
    }


def candidate_rows(path: Path, requested_hbf_labels: tuple[str, ...]) -> list[dict[str, object]]:
    raw_rows = read_csv(path)
    candidates: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for row in raw_rows:
        if row.get("policy") != "cost_dynamic":
            continue
        attention_gpus = int(row["attention_gpus"])
        moe_gpus = int(row["moe_gpus"])
        if attention_gpus not in PROFILED_ATTENTION_GPUS:
            continue
        if row.get("capacity_ok", "True") not in {"True", "true", "1"}:
            continue
        for hbf_label, hbf_gpus in hbf_labels_for(moe_gpus, requested_hbf_labels):
            key = (
                row["workload"],
                int(row["context_length"]),
                int(row["sessions"]),
                int(row["gpu_budget"]),
                attention_gpus,
                moe_gpus,
                hbf_label,
                hbf_gpus,
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "workload": row["workload"],
                    "context_length": int(row["context_length"]),
                    "sessions": int(row["sessions"]),
                    "microbatch_count": int(row["microbatch_count"]),
                    "gpu_budget": int(row["gpu_budget"]),
                    "attention_gpus": attention_gpus,
                    "moe_gpus": moe_gpus,
                    "hbf_label": hbf_label,
                    "hbf_gpus": hbf_gpus,
                    "source_raw_attention_ms": float(row["raw_attention_ms"]),
                    "source_capacity_ok": row["capacity_ok"],
                }
            )
    return candidates


def select_best(rows: list[dict[str, object]], microbatch_count: int) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            row["workload"],
            row["context_length"],
            row["sessions"],
            row["gpu_budget"],
            row["hbf_label"],
        )
        groups.setdefault(key, []).append(row)

    best_rows: list[dict[str, object]] = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        feasible = [row for row in group if row["capacity_ok"]]
        if not feasible:
            continue
        best = min(feasible, key=lambda row: (row["raw_step_ms_mean"], row["moe_gpus"], row["hbf_gpus"]))
        out = dict(best)
        out["selection_reason"] = "min_raw_step_capacity_ok"
        out["slo_ok"] = float(out["raw_step_ms_mean"]) <= 100.0
        best_rows.append(out)
    return best_rows


def add_comparisons(best_rows: list[dict[str, object]]) -> None:
    by_key: dict[tuple[object, ...], dict[str, dict[str, object]]] = {}
    for row in best_rows:
        key = (
            row["workload"],
            row["context_length"],
            row["sessions"],
            row["gpu_budget"],
        )
        by_key.setdefault(key, {})[str(row["hbf_label"])] = row

    for group in by_key.values():
        static = group.get("0")
        all_hbf = group.get("all")
        static_step = float(static["raw_step_ms_mean"]) if static else math.nan
        all_step = float(all_hbf["raw_step_ms_mean"]) if all_hbf else math.nan
        denom = static_step - all_step
        for row in group.values():
            step = float(row["raw_step_ms_mean"])
            row["step_reduction_vs_h0_pct"] = (
                100.0 * (static_step - step) / static_step if static and static_step > 0 else math.nan
            )
            row["all_hbf_gap_pct"] = (
                100.0 * (step - all_step) / all_step if all_hbf and all_step > 0 else math.nan
            )
            row["recovered_all_hbf_gain_pct"] = (
                100.0 * (static_step - step) / denom if static and all_hbf and denom > 1e-9 else math.nan
            )


def markdown_table(rows: list[dict[str, object]], fields: list[str]) -> list[str]:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        cells = []
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                if math.isnan(value):
                    cells.append("n/a")
                else:
                    cells.append(f"{value:.2f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_summary(outdir: Path, best_rows: list[dict[str, object]], all_rows: list[dict[str, object]]) -> None:
    results = outdir / "results"
    by_label: dict[str, list[dict[str, object]]] = {}
    for row in best_rows:
        by_label.setdefault(str(row["hbf_label"]), []).append(row)

    label_summary: list[dict[str, object]] = []
    for label in sorted(by_label, key=lambda x: (x == "all", int(x) if x.isdigit() else 999)):
        rows = by_label[label]
        label_summary.append(
            {
                "hbf_label": label,
                "best_cells": len(rows),
                "avg_raw_step_ms": float(np.mean([float(r["raw_step_ms_mean"]) for r in rows])),
                "avg_raw_moe_ms": float(np.mean([float(r["raw_moe_ms_mean"]) for r in rows])),
                "slo_ok_cells": sum(bool(r["slo_ok"]) for r in rows),
                "avg_hbf_load_fraction": float(np.mean([float(r["hbf_load_fraction_mean"]) for r in rows])),
                "avg_hbf_expert_fraction": float(
                    np.mean([float(r["hbf_expert_fraction_mean"]) for r in rows])
                ),
                "avg_recovered_gain_pct": float(
                    np.nanmean([float(r["recovered_all_hbf_gain_pct"]) for r in rows])
                )
                if any(not math.isnan(float(r["recovered_all_hbf_gain_pct"])) for r in rows)
                else math.nan,
            }
        )

    by_budget_label: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in best_rows:
        by_budget_label.setdefault((int(row["gpu_budget"]), str(row["hbf_label"])), []).append(row)
    budget_summary: list[dict[str, object]] = []
    for (budget, label), rows in sorted(by_budget_label.items()):
        budget_summary.append(
            {
                "budget": budget,
                "hbf_label": label,
                "cells": len(rows),
                "avg_step_ms": float(np.mean([float(r["raw_step_ms_mean"]) for r in rows])),
                "slo_ok_cells": sum(bool(r["slo_ok"]) for r in rows),
                "avg_A": float(np.mean([int(r["attention_gpus"]) for r in rows])),
                "avg_M": float(np.mean([int(r["moe_gpus"]) for r in rows])),
            }
        )

    lines = [
        "# Hybrid HBM/HBF MoE Balancer Check",
        "",
        "This is a raw trace-backed check, not a Vidur queueing run.",
        "",
        "## Model",
        "",
        f"- R1 no-pruning selected-expert traces, {LAYERS} MoE layers, {EXPERTS} routed experts, top-8.",
        f"- HBM and HBF expert bandwidth are both `{base.BW_HBM_GBPS:.0f} GB/s`.",
        f"- One routed expert collection across all layers is `{PER_EXPERT_COLLECTION_GIB:.2f} GiB`.",
        f"- Shared expert collection reserve is `{SHARED_COLLECTION_GIB:.2f} GiB`.",
        f"- HBM usable capacity is `{base.HBM_CAP_GB:.1f} GiB`, so each HBM MoE GPU can hold `{HBM_ROUTED_CAP_EXPERTS}` routed expert slots/layer after the shared reserve.",
        f"- HBF usable capacity is `{base.HBF_CAP_GB:.1f} GiB`; a full routed+shared replica is `{(base.ROUTED_EXPERT_BYTES_TOTAL + base.SHARED_EXPERT_BYTES_TOTAL) / GIB:.2f} GiB`.",
        "",
        "## Average best-cell summary",
        "",
        *markdown_table(
            label_summary,
            [
                "hbf_label",
                "best_cells",
                "avg_raw_step_ms",
                "avg_raw_moe_ms",
                "slo_ok_cells",
                "avg_hbf_load_fraction",
                "avg_hbf_expert_fraction",
                "avg_recovered_gain_pct",
            ],
        ),
        "",
        "## Budget summary",
        "",
        *markdown_table(
            budget_summary,
            ["budget", "hbf_label", "cells", "avg_step_ms", "slo_ok_cells", "avg_A", "avg_M"],
        ),
        "",
        "## Interpretation guardrails",
        "",
        "- `hbf_label=0` is capacity-aware all-HBM static MoE. It is absent where the HBM side cannot store all routed experts.",
        "- `hbf_label=1/2/4` keeps total MoE GPUs fixed at the chosen M and upgrades only that many MoE GPUs to HBF full replicas.",
        "- `hbf_label=all` is the all-HBF dynamic upper bound.",
        "- `recovered_all_hbf_gain_pct` is only defined where a feasible HBM-only baseline exists for the same context/S/budget cell.",
    ]
    (results / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(outdir: Path, best_rows: list[dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    labels = [label for label in ["0", "1", "2", "4", "all"] if any(str(r["hbf_label"]) == label for r in best_rows)]
    budgets = sorted({int(r["gpu_budget"]) for r in best_rows})
    avg_by_budget_label = {}
    for budget in budgets:
        for label in labels:
            rows = [r for r in best_rows if int(r["gpu_budget"]) == budget and str(r["hbf_label"]) == label]
            if rows:
                avg_by_budget_label[(budget, label)] = np.mean([float(r["raw_step_ms_mean"]) for r in rows])

    x = np.arange(len(labels))
    width = 0.18
    plt.figure(figsize=(9, 4.8))
    for idx, budget in enumerate(budgets):
        ys = [avg_by_budget_label.get((budget, label), np.nan) for label in labels]
        plt.bar(x + (idx - (len(budgets) - 1) / 2) * width, ys, width, label=f"B={budget}")
    plt.xticks(x, labels)
    plt.ylabel("Average best raw step (ms)")
    plt.xlabel("HBF MoE GPUs upgraded")
    plt.title("Hybrid HBF balancers: best raw step after A/M selection")
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(figdir / "fig1_best_raw_step_by_hbf_count.png", dpi=180)
    plt.close()

    rows = [
        r
        for r in best_rows
        if str(r["hbf_label"]) in {"1", "2", "4"}
        and not math.isnan(float(r["recovered_all_hbf_gain_pct"]))
    ]
    if rows:
        plt.figure(figsize=(8, 4.5))
        data = []
        plot_labels = []
        for label in ["1", "2", "4"]:
            vals = [float(r["recovered_all_hbf_gain_pct"]) for r in rows if str(r["hbf_label"]) == label]
            if vals:
                data.append(vals)
                plot_labels.append(label)
        plt.boxplot(data, tick_labels=plot_labels, showfliers=False)
        plt.axhline(100, color="0.4", linestyle="--", linewidth=1)
        plt.ylabel("Recovered all-HBF gain (%)")
        plt.xlabel("HBF MoE GPUs upgraded")
        plt.title("How much of all-HBF dynamic gain partial HBF recovers")
        plt.tight_layout()
        plt.savefig(figdir / "fig2_recovered_gain_distribution.png", dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    requested_hbf_labels = tuple(part.strip() for part in args.hbf_labels.split(",") if part.strip())
    candidates = candidate_rows(args.candidates_csv, requested_hbf_labels)
    if not candidates:
        raise RuntimeError(f"no candidates loaded from {args.candidates_csv}")

    task_keys = sorted(
        {
            (
                str(row["workload"]),
                int(row["sessions"]),
                int(row["attention_gpus"]),
                int(row["moe_gpus"]),
                str(row["hbf_label"]),
                int(row["hbf_gpus"]),
            )
            for row in candidates
        }
    )
    workloads = sorted({key[0] for key in task_keys})
    if len(workloads) != 1:
        raise RuntimeError(f"this script currently expects one workload per run, got {workloads}")
    workload = workloads[0]

    task_results: dict[tuple[object, ...], dict[str, object]] = {}
    if args.jobs <= 1:
        init_worker(
            str(args.trace_root),
            workload,
            args.trace_max_requests,
            args.trace_profile_max_requests,
            args.decode_tokens,
            args.microbatch_count,
        )
        for index, task in enumerate(task_keys, start=1):
            task_results[task] = worker(task)
            if index % 50 == 0:
                print(f"completed {index}/{len(task_keys)} hybrid MoE tasks", flush=True)
    else:
        with ProcessPoolExecutor(
            max_workers=args.jobs,
            initializer=init_worker,
            initargs=(
                str(args.trace_root),
                workload,
                args.trace_max_requests,
                args.trace_profile_max_requests,
                args.decode_tokens,
                args.microbatch_count,
            ),
        ) as pool:
            future_to_task = {pool.submit(worker, task): task for task in task_keys}
            for index, future in enumerate(as_completed(future_to_task), start=1):
                task = future_to_task[future]
                task_results[task] = future.result()
                if index % 50 == 0 or index == len(future_to_task):
                    print(f"completed {index}/{len(future_to_task)} hybrid MoE tasks", flush=True)

    all_rows: list[dict[str, object]] = []
    for row in candidates:
        key = (
            str(row["workload"]),
            int(row["sessions"]),
            int(row["attention_gpus"]),
            int(row["moe_gpus"]),
            str(row["hbf_label"]),
            int(row["hbf_gpus"]),
        )
        moe = task_results[key]
        out = dict(row)
        out.update(moe)
        out["raw_step_ms_mean"] = raw_step(
            float(out["source_raw_attention_ms"]),
            float(out["raw_moe_ms_mean"]),
            args.microbatch_count,
        )
        out["raw_step_ms_p95"] = raw_step(
            float(out["source_raw_attention_ms"]),
            float(out["raw_moe_ms_p95"]),
            args.microbatch_count,
        )
        out["slo_ok"] = float(out["raw_step_ms_mean"]) <= args.slo_ms
        all_rows.append(out)

    best_rows = select_best(all_rows, args.microbatch_count)
    add_comparisons(best_rows)

    results = args.outdir / "results"
    write_csv(results / "all_hybrid_candidates.csv", all_rows)
    write_csv(results / "best_by_hbf_label.csv", best_rows)
    write_summary(args.outdir, best_rows, all_rows)
    write_plots(args.outdir, best_rows)
    print(f"wrote {len(all_rows)} candidate rows and {len(best_rows)} best rows to {args.outdir}")


if __name__ == "__main__":
    main()
