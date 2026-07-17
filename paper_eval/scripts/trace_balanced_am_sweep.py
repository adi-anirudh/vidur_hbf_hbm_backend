#!/usr/bin/env python3
"""Trace-backed raw-step A/M optimizer for FlexEP.

This is intentionally separate from the Vidur Task-3 wrapper.

Purpose:
  * use real selected-expert traces for MoE load;
  * use the corrected analytical attention model from corrected_microbatch_sweep;
  * choose A/M by the pipeline resource-balance objective:

        raw_step_ms = max(raw_attention_ms, raw_moe_ms)   for mu >= 2

The output is an A/M-selection input for later Vidur validation. It is not a
Vidur serving-throughput result.
"""

from __future__ import annotations

import argparse
import csv
import json
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
import corrected_microbatch_sweep as micro  # noqa: E402
from vidur.execution_time_predictor.moe_trace import (  # noqa: E402
    EXPERTS,
    LAYERS,
    MoETraceRouteStore,
)


DEFAULT_TRACE_ROOT = PAPER / "experiments" / "deepseek_r1_awq_no_prune_replay"
DEFAULT_OUT = PAPER / "experiments" / "deepseek_r1_awq_task3_balanced_am"

POLICIES = ("static", "active_count_dynamic", "token_count_dynamic", "cost_dynamic")


def int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("empty integer list")
    return values


def str_list(text: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("empty string list")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--context-lengths",
        type=int_list,
        default=(8192, 16384, 32768, 65536, 131072),
    )
    parser.add_argument("--sessions", type=int_list, default=(8, 16, 32, 64, 128))
    parser.add_argument("--gpu-budgets", type=int_list, default=(8, 16, 24, 32))
    parser.add_argument("--microbatch-count", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--workloads", type=str_list, default=("mixed",))
    parser.add_argument("--policies", type=str_list, default=("static", "cost_dynamic"))
    parser.add_argument(
        "--budget-mode",
        choices=("at_most", "exact"),
        default="at_most",
        help=(
            "'at_most' matches the older analytical best-within-budget selector. "
            "'exact' requires A+M to equal the budget."
        ),
    )
    parser.add_argument("--trace-max-requests", type=int, default=512)
    parser.add_argument("--trace-profile-max-requests", type=int, default=0)
    parser.add_argument("--hbm-bw-gbps", type=float, default=base.BW_HBM_GBPS)
    parser.add_argument("--hbf-bw-gbps", type=float, default=base.BW_HBF_GBPS)
    parser.add_argument("--link-bw-gbps", type=float, default=base.LINK_BW_GBPS)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--slo-ms", type=float, default=100.0)
    parser.add_argument("--max-moe-gpus", type=int, default=0)
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


def profiled_static_owner(profile: np.ndarray, moe_gpus: int) -> np.ndarray:
    """Train-profiled fixed ownership with floor/ceil expert capacity.

    The Vidur helper requires M to divide 256. The old analytical evaluator did
    not. This owner keeps the fixed-placement idea but permits arbitrary M by
    assigning ceil/floor capacity and greedily balancing train-profiled token
    mass.
    """

    base_capacity, extra = divmod(EXPERTS, moe_gpus)
    remaining = np.full(moe_gpus, base_capacity, np.int32)
    if extra:
        remaining[:extra] += 1
    loads = np.zeros(moe_gpus, np.float64)
    owner = np.full(EXPERTS, -1, np.int16)
    for expert in np.lexsort((np.arange(EXPERTS), -profile)):
        candidates = np.flatnonzero(remaining > 0)
        gpu = int(candidates[np.argmin(loads[candidates])])
        owner[int(expert)] = gpu
        remaining[gpu] -= 1
        loads[gpu] += float(profile[int(expert)])
    return owner


def dynamic_owner(counts: np.ndarray, moe_gpus: int, policy: str) -> np.ndarray:
    active = np.flatnonzero(counts)
    owner = np.full(EXPERTS, -1, np.int16)
    if active.size == 0:
        return owner

    read = base.bytes_to_ms(base.PER_EXPERT_BYTES, _HBF_BW_GBPS)
    compute_unit = base.flops_to_ms(6 * base.D * base.MOE_INTERMEDIATE)

    if policy == "active_count_dynamic":
        score = np.where(counts > 0, 1.0, 0.0)
    elif policy == "token_count_dynamic":
        score = counts.astype(np.float64)
    elif policy == "cost_dynamic":
        score = np.where(counts > 0, read + compute_unit * counts, 0.0)
    else:
        raise ValueError(f"unsupported dynamic policy: {policy}")

    loads = np.zeros(moe_gpus, np.float64)
    for expert in active[np.lexsort((active, -counts[active], -score[active]))]:
        gpu = int(np.argmin(loads))
        owner[int(expert)] = gpu
        loads[gpu] += float(score[int(expert)])
    return owner


@dataclass(frozen=True)
class MoeResult:
    workload: str
    sessions: int
    attention_gpus: int
    moe_gpus: int
    policy: str
    raw_moe_ms_mean: float
    raw_moe_ms_p95: float
    moe_execution_ms_mean: float
    moe_transfer_ms_mean: float
    active_experts_mean: float
    max_gpu_experts_mean: float
    max_gpu_tokens_mean: float
    owner_visits_mean: float


_ROUTES: np.ndarray | None = None
_PROFILE: np.ndarray | None = None
_WORKLOAD = ""
_DECODE_TOKENS = 0
_MICROBATCH_COUNT = 0
_HBF_BW_GBPS = base.BW_HBF_GBPS
_LINK_BW_GBPS = base.LINK_BW_GBPS
_STATIC_OWNER_CACHE: dict[int, np.ndarray] = {}


def init_worker(
    trace_root: str,
    workload: str,
    max_requests: int,
    profile_max_requests: int,
    decode_tokens: int,
    microbatch_count: int,
    hbf_bw_gbps: float,
    link_bw_gbps: float,
) -> None:
    global _ROUTES, _PROFILE, _WORKLOAD, _DECODE_TOKENS, _MICROBATCH_COUNT
    global _HBF_BW_GBPS, _LINK_BW_GBPS, _STATIC_OWNER_CACHE

    store = MoETraceRouteStore(
        root=trace_root,
        workload=workload,
        max_requests=max_requests,
        profile_max_requests=profile_max_requests,
    )
    if not store.routes:
        raise RuntimeError(f"no trace routes loaded for workload={workload}")

    min_decode = min(route.route.shape[0] for route in store.routes)
    used_decode = min(decode_tokens, min_decode)
    _ROUTES = np.stack(
        [route.route[:used_decode].astype(np.int16, copy=False) for route in store.routes],
        axis=0,
    )
    _PROFILE = store.profile.astype(np.float64, copy=True)
    _WORKLOAD = workload
    _DECODE_TOKENS = used_decode
    _MICROBATCH_COUNT = microbatch_count
    _HBF_BW_GBPS = hbf_bw_gbps
    _LINK_BW_GBPS = link_bw_gbps
    _STATIC_OWNER_CACHE = {}


def owner_for(layer: int, counts: np.ndarray, moe_gpus: int, policy: str) -> np.ndarray:
    assert _PROFILE is not None
    if policy == "static":
        cached = _STATIC_OWNER_CACHE.get(moe_gpus)
        if cached is None:
            cached = np.stack(
                [profiled_static_owner(_PROFILE[idx], moe_gpus) for idx in range(LAYERS)]
            )
            _STATIC_OWNER_CACHE[moe_gpus] = cached
        owner = np.full(EXPERTS, -1, np.int16)
        active = np.flatnonzero(counts)
        owner[active] = cached[layer, active]
        return owner
    return dynamic_owner(counts, moe_gpus, policy)


def worker_moe(task: tuple[int, int, int, str]) -> dict[str, object]:
    sessions, attention_gpus, moe_gpus, policy = task
    assert _ROUTES is not None

    read_ms = base.bytes_to_ms(base.PER_EXPERT_BYTES, _HBF_BW_GBPS)
    compute_unit_ms = base.flops_to_ms(6 * base.D * base.MOE_INTERMEDIATE)
    shared_read_ms = base.bytes_to_ms(base.SHARED_WEIGHT_BYTES_LAYER, _HBF_BW_GBPS)
    links = max(1, min(attention_gpus, moe_gpus))

    step_moe: list[float] = []
    step_execution: list[float] = []
    step_transfer: list[float] = []
    active_experts: list[float] = []
    max_gpu_experts: list[float] = []
    max_gpu_tokens: list[float] = []
    owner_visits: list[float] = []

    route_count = int(_ROUTES.shape[0])
    bounds = microbatch_slices(sessions, _MICROBATCH_COUNT)

    for decode_step in range(_DECODE_TOKENS):
        execution_sum = 0.0
        transfer_bytes_sum = 0.0
        for start, end in bounds:
            request_ids = np.arange(start, end, dtype=np.int64) % route_count
            mb_size = int(end - start)
            selected = _ROUTES[request_ids, decode_step]  # [B, layers, topk]
            for layer in range(LAYERS):
                routes = selected[:, layer, :]
                counts = np.bincount(routes.reshape(-1), minlength=EXPERTS)
                active = np.flatnonzero(counts)
                if active.size == 0:
                    continue

                owner = owner_for(layer, counts, moe_gpus, policy)
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
                owner_visits.append(float(visits / mb_size if mb_size else 0.0))

        transfer_ms = base.bytes_to_ms(transfer_bytes_sum, links * _LINK_BW_GBPS)
        moe_ms = execution_sum + transfer_ms
        step_execution.append(execution_sum)
        step_transfer.append(transfer_ms)
        step_moe.append(moe_ms)

    result = MoeResult(
        workload=_WORKLOAD,
        sessions=sessions,
        attention_gpus=attention_gpus,
        moe_gpus=moe_gpus,
        policy=policy,
        raw_moe_ms_mean=float(np.mean(step_moe)),
        raw_moe_ms_p95=float(np.percentile(step_moe, 95)),
        moe_execution_ms_mean=float(np.mean(step_execution)),
        moe_transfer_ms_mean=float(np.mean(step_transfer)),
        active_experts_mean=float(np.mean(active_experts)) if active_experts else 0.0,
        max_gpu_experts_mean=float(np.mean(max_gpu_experts)) if max_gpu_experts else 0.0,
        max_gpu_tokens_mean=float(np.mean(max_gpu_tokens)) if max_gpu_tokens else 0.0,
        owner_visits_mean=float(np.mean(owner_visits)) if owner_visits else 0.0,
    )
    return result.__dict__


def candidate_splits(max_budget: int, max_moe_gpus: int) -> list[tuple[int, int]]:
    limit = max_moe_gpus if max_moe_gpus > 0 else max_budget - 1
    splits = []
    for total in range(2, max_budget + 1):
        for moe_gpus in range(1, min(limit, total - 1) + 1):
            attention_gpus = total - moe_gpus
            splits.append((attention_gpus, moe_gpus))
    return sorted(set(splits), key=lambda pair: (sum(pair), pair[1], pair[0]))


def attention_capacity_ok(context: int, sessions: int, attention_gpus: int) -> bool:
    capacity = base.attention_capacity_sessions(context, base.HBM_CAP_GB)
    return capacity >= 1 and math.ceil(sessions / attention_gpus) <= capacity


def raw_step(attention_ms: float, moe_ms: float, microbatch_count: int) -> float:
    if microbatch_count == 1:
        return attention_ms + moe_ms
    return max(attention_ms, moe_ms)


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def select_best(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            row["workload"],
            row["context_length"],
            row["sessions"],
            row["gpu_budget"],
            row["policy"],
        )
        groups.setdefault(key, []).append(row)

    best_rows = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        feasible = [row for row in group if row["capacity_ok"]]
        if feasible:
            best = min(
                feasible,
                key=lambda row: (
                    float(row["raw_step_ms_mean"]),
                    float(row["raw_step_ms_p95"]),
                    int(row["gpus"]),
                    int(row["moe_gpus"]),
                ),
            )
            reason = "min_raw_step"
        else:
            best = min(
                group,
                key=lambda row: (
                    float("inf")
                    if row["raw_step_ms_mean"] == ""
                    else float(row["raw_step_ms_mean"]),
                    int(row["gpus"]),
                ),
            )
            reason = "no_capacity_feasible"
        out = dict(best)
        out["selection_reason"] = reason
        out["slo_feasible"] = (
            bool(out["capacity_ok"]) and float(out["raw_step_ms_mean"]) <= float(out["slo_ms"])
        )
        best_rows.append(out)
    return best_rows


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Trace-backed raw-step A/M optimization",
        "",
        "Objective: choose A/M by minimizing `max(raw_attention_ms, raw_moe_ms)` for `mu >= 2`.",
        "This is an A/M resource-balance sweep, not a Vidur throughput run.",
        "",
        "| Policy | Budget | Ctx | S | GPUs | A | M | Attn ms | MoE ms | Step ms | SLO feasible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(
        rows,
        key=lambda r: (
            str(r["policy"]),
            int(r["gpu_budget"]),
            int(r["context_length"]),
            int(r["sessions"]),
        ),
    ):
        lines.append(
            f"| {row['policy']} | {row['gpu_budget']} | "
            f"{int(row['context_length']) // 1024}K | {row['sessions']} | "
            f"{row['gpus']} | {row['attention_gpus']} | {row['moe_gpus']} | "
            f"{float(row['raw_attention_ms']):.2f} | "
            f"{float(row['raw_moe_ms_mean']):.2f} | "
            f"{float(row['raw_step_ms_mean']):.2f} | "
            f"{'yes' if row['slo_feasible'] else 'no'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "results").mkdir(exist_ok=True)

    for policy in args.policies:
        if policy not in POLICIES:
            raise ValueError(f"unknown policy {policy!r}; expected one of {POLICIES}")

    max_budget = max(args.gpu_budgets)
    splits = candidate_splits(max_budget, args.max_moe_gpus)
    tasks = [
        (sessions, attention_gpus, moe_gpus, policy)
        for sessions in args.sessions
        for attention_gpus, moe_gpus in splits
        for policy in args.policies
    ]

    all_rows: list[dict[str, object]] = []
    for workload in args.workloads:
        print(
            f"loading traces and computing raw MoE: workload={workload}, "
            f"tasks={len(tasks)}, jobs={args.jobs}",
            flush=True,
        )
        moe_rows: dict[tuple[int, int, int, str], dict[str, object]] = {}
        if args.jobs == 1:
            init_worker(
                str(args.trace_root),
                workload,
                args.trace_max_requests,
                args.trace_profile_max_requests,
                args.decode_tokens,
                args.microbatch_count,
                args.hbf_bw_gbps,
                args.link_bw_gbps,
            )
            for index, task in enumerate(tasks, start=1):
                result = worker_moe(task)
                moe_rows[(task[0], task[1], task[2], task[3])] = result
                if index % 100 == 0 or index == len(tasks):
                    print(f"  completed {index}/{len(tasks)} raw-MoE tasks", flush=True)
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
                    args.hbf_bw_gbps,
                    args.link_bw_gbps,
                ),
            ) as executor:
                future_to_task = {executor.submit(worker_moe, task): task for task in tasks}
                for index, future in enumerate(as_completed(future_to_task), start=1):
                    task = future_to_task[future]
                    result = future.result()
                    moe_rows[(task[0], task[1], task[2], task[3])] = result
                    if index % 100 == 0 or index == len(tasks):
                        print(f"  completed {index}/{len(tasks)} raw-MoE tasks", flush=True)

        for context in args.context_lengths:
            for sessions in args.sessions:
                for budget in args.gpu_budgets:
                    for attention_gpus, moe_gpus in splits:
                        gpus = attention_gpus + moe_gpus
                        if args.budget_mode == "exact" and gpus != budget:
                            continue
                        if args.budget_mode == "at_most" and gpus > budget:
                            continue
                        cap_ok = attention_capacity_ok(context, sessions, attention_gpus)
                        attention_ms = micro.attention_work_ms(
                            context,
                            sessions,
                            attention_gpus,
                            args.microbatch_count,
                            args.hbm_bw_gbps,
                        )
                        for policy in args.policies:
                            moe = moe_rows[(sessions, attention_gpus, moe_gpus, policy)]
                            raw_moe = float(moe["raw_moe_ms_mean"])
                            raw_moe_p95 = float(moe["raw_moe_ms_p95"])
                            row = {
                                "workload": workload,
                                "context_length": context,
                                "sessions": sessions,
                                "microbatch_count": args.microbatch_count,
                                "sessions_per_microbatch": sessions / args.microbatch_count,
                                "gpu_budget": budget,
                                "gpus": gpus,
                                "attention_gpus": attention_gpus,
                                "moe_gpus": moe_gpus,
                                "policy": policy,
                                "budget_mode": args.budget_mode,
                                "capacity_ok": cap_ok,
                                "raw_attention_ms": attention_ms,
                                "raw_moe_ms_mean": raw_moe,
                                "raw_moe_ms_p95": raw_moe_p95,
                                "raw_step_ms_mean": raw_step(
                                    attention_ms, raw_moe, args.microbatch_count
                                ),
                                "raw_step_ms_p95": raw_step(
                                    attention_ms, raw_moe_p95, args.microbatch_count
                                ),
                                "moe_execution_ms_mean": moe["moe_execution_ms_mean"],
                                "moe_transfer_ms_mean": moe["moe_transfer_ms_mean"],
                                "active_experts_mean": moe["active_experts_mean"],
                                "max_gpu_experts_mean": moe["max_gpu_experts_mean"],
                                "max_gpu_tokens_mean": moe["max_gpu_tokens_mean"],
                                "owner_visits_mean": moe["owner_visits_mean"],
                                "hbm_bw_gbps": args.hbm_bw_gbps,
                                "hbf_bw_gbps": args.hbf_bw_gbps,
                                "link_bw_gbps": args.link_bw_gbps,
                                "slo_ms": args.slo_ms,
                            }
                            all_rows.append(row)

    best_rows = select_best(all_rows)
    write_csv(args.outdir / "results" / "all_splits.csv", all_rows)
    write_csv(args.outdir / "results" / "best_by_policy.csv", best_rows)
    flex_rows = [row for row in best_rows if row["policy"] == "cost_dynamic"]
    if flex_rows:
        write_csv(args.outdir / "results" / "best_flexep_cost_dynamic.csv", flex_rows)
    write_summary(args.outdir / "results" / "summary.md", best_rows)
    (args.outdir / "run_config.json").write_text(
        json.dumps(
            {
                "trace_root": str(args.trace_root),
                "context_lengths": list(args.context_lengths),
                "sessions": list(args.sessions),
                "gpu_budgets": list(args.gpu_budgets),
                "microbatch_count": args.microbatch_count,
                "decode_tokens": args.decode_tokens,
                "workloads": list(args.workloads),
                "policies": list(args.policies),
                "budget_mode": args.budget_mode,
                "trace_max_requests": args.trace_max_requests,
                "trace_profile_max_requests": args.trace_profile_max_requests,
                "hbm_bw_gbps": args.hbm_bw_gbps,
                "hbf_bw_gbps": args.hbf_bw_gbps,
                "link_bw_gbps": args.link_bw_gbps,
                "slo_ms": args.slo_ms,
                "jobs": args.jobs,
                "objective": "minimize max(raw_attention_ms, raw_moe_ms)",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.outdir / "README.md").write_text(
        "\n".join(
            [
                "# Trace-backed raw-step A/M sweep",
                "",
                "This folder is separate from the Vidur budget sweep.",
                "",
                "Selection objective:",
                "",
                "`raw_step_ms = max(raw_attention_ms, raw_moe_ms)` for `mu >= 2`.",
                "",
                "Outputs:",
                "",
                "- `results/all_splits.csv`: every tested A/M candidate.",
                "- `results/best_by_policy.csv`: selected A/M per context/session/budget/policy.",
                "- `results/best_flexep_cost_dynamic.csv`: cost-dynamic rows only.",
                "- `results/summary.md`: compact human-readable table.",
                "",
                "No Vidur throughput or p95 scheduling result is inferred here.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(all_rows)} candidate rows and {len(best_rows)} best rows under {args.outdir}")


if __name__ == "__main__":
    main()
