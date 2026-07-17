#!/usr/bin/env python3
"""Task 3: no-pruning R1 trace-backed Vidur evaluation.

This script keeps Vidur as the outer serving simulator and enables the
trace-backed MoE path in HBFLinearRegressionExecutionTimePredictor. Outputs are
kept isolated under paper_eval/experiments/deepseek_r1_awq_task3_vidur.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import glob
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PAPER = Path(__file__).resolve().parents[1]
REPO = PAPER.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DEFAULT_TRACE_ROOT = PAPER / "experiments" / "deepseek_r1_awq_no_prune_replay"
DEFAULT_OUT = PAPER / "experiments" / "deepseek_r1_awq_task3_vidur"
POLICIES = ("static", "active_count_dynamic", "token_count_dynamic", "cost_dynamic")
POLICY_LABELS = {
    "static": "Static",
    "active_count_dynamic": "Active-count dyn",
    "token_count_dynamic": "Token-count dyn",
    "cost_dynamic": "Cost dyn",
}
POLICY_COLORS = {
    "static": "#777777",
    "active_count_dynamic": "#D55E00",
    "token_count_dynamic": "#0072B2",
    "cost_dynamic": "#009E73",
}


def int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(x) for x in text.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("empty integer list")
    return values


def str_list(text: str) -> tuple[str, ...]:
    values = tuple(x.strip() for x in text.split(",") if x.strip())
    if not values:
        raise argparse.ArgumentTypeError("empty string list")
    return values


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-root", type=Path, default=DEFAULT_TRACE_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--context-lengths", type=int_list, default=(32768,))
    parser.add_argument("--sessions", type=int_list, default=(8, 16, 32, 64, 128))
    parser.add_argument("--microbatch-count", type=int, default=2)
    parser.add_argument("--moe-gpus", type=int_list, default=(8, 16, 32))
    parser.add_argument("--attention-gpus", type=int, default=8)
    parser.add_argument("--gpu-budgets", type=int_list, default=(8, 16, 24, 32))
    parser.add_argument(
        "--optimize-am",
        action="store_true",
        help=(
            "Sweep valid attention/MoE splits A+M=B for every budget in "
            "--gpu-budgets and select the best split under the SLO."
        ),
    )
    parser.add_argument("--policies", type=str_list, default=POLICIES)
    parser.add_argument("--workloads", type=str_list, default=("mixed",))
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--slo-ms", type=float, default=100.0)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--device", default="h100")
    parser.add_argument("--network-device", default="h100_pairwise_nvlink")
    parser.add_argument("--hbf-config", default="configs/hbf_paper_722g.toml")
    parser.add_argument("--trace-max-requests", type=int, default=512)
    parser.add_argument("--trace-profile-max-requests", type=int, default=0)
    parser.add_argument("--trace-read-bw-gbps", type=float, default=1024.0)
    parser.add_argument("--disagg-link-bw-gbps", type=float, default=450.0)
    parser.add_argument("--overlap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of Vidur runs to execute in parallel. Use 1 for serial execution.",
    )
    parser.add_argument("--max-runs", type=int, default=0)
    return parser.parse_args()


def vidur_argv(
    args: argparse.Namespace,
    run_base: Path,
    diagnostics_path: Path,
    workload: str,
    context: int,
    sessions: int,
    attention_gpus: int,
    moe_gpus: int,
    policy: str,
) -> list[str]:
    sessions_per_microbatch = sessions // args.microbatch_count
    granularity = max(16, context // 2048)
    num_blocks = math.ceil(
        sessions * math.ceil((context + args.decode_tokens) / 16) / 0.99
    ) + 64
    overlap_flag = (
        "--h_b_f_linear_regression_execution_time_predictor_config_disagg_microbatch_overlap"
        if args.overlap
        else "--no-h_b_f_linear_regression_execution_time_predictor_config_disagg_microbatch_overlap"
    )
    return [
        "run_task3_trace_vidur.py",
        "--replica_config_model_name",
        args.model,
        "--replica_config_device",
        args.device,
        "--replica_config_network_device",
        args.network_device,
        "--replica_config_num_pipeline_stages",
        "1",
        "--replica_config_tensor_parallel_size",
        str(attention_gpus),
        "--replica_scheduler_config_type",
        "sarathi",
        "--sarathi_scheduler_config_batch_size_cap",
        str(sessions_per_microbatch),
        "--sarathi_scheduler_config_chunk_size",
        str(min(context, 4096)),
        "--sarathi_scheduler_config_num_blocks",
        str(num_blocks),
        "--request_generator_config_type",
        "synthetic",
        "--length_generator_config_type",
        "fixed",
        "--fixed_request_length_generator_config_prefill_tokens",
        str(context),
        "--fixed_request_length_generator_config_decode_tokens",
        str(args.decode_tokens),
        "--interval_generator_config_type",
        "static",
        "--synthetic_request_generator_config_num_requests",
        str(sessions),
        "--synthetic_request_generator_config_decode_only",
        "--execution_time_predictor_config_type",
        "hbf_linear_regression",
        "--h_b_f_linear_regression_execution_time_predictor_config_hbfsim_config_path",
        args.hbf_config,
        "--h_b_f_linear_regression_execution_time_predictor_config_kv_cache_prediction_granularity",
        str(granularity),
        "--h_b_f_linear_regression_execution_time_predictor_config_prediction_max_tokens_per_request",
        str(context + args.decode_tokens + granularity + 16),
        "--h_b_f_linear_regression_execution_time_predictor_config_prediction_max_prefill_chunk_size",
        str(min(context, 4096)),
        "--h_b_f_linear_regression_execution_time_predictor_config_prediction_max_batch_size",
        str(max(args.sessions)),
        "--h_b_f_linear_regression_execution_time_predictor_config_num_experts",
        "256",
        "--h_b_f_linear_regression_execution_time_predictor_config_moe_top_k",
        "8",
        "--h_b_f_linear_regression_execution_time_predictor_config_moe_intermediate",
        "2048",
        "--h_b_f_linear_regression_execution_time_predictor_config_moe_dtype_bytes",
        "1",
        "--h_b_f_linear_regression_execution_time_predictor_config_shared_expert_intermediate",
        "2048",
        "--h_b_f_linear_regression_execution_time_predictor_config_disagg_num_moe_gpus",
        str(moe_gpus),
        "--h_b_f_linear_regression_execution_time_predictor_config_disagg_expert_placement",
        "replicated",
        "--h_b_f_linear_regression_execution_time_predictor_config_disagg_routing",
        "affinity",
        overlap_flag,
        "--h_b_f_linear_regression_execution_time_predictor_config_disagg_link_bw_gbps",
        str(args.disagg_link_bw_gbps),
        "--h_b_f_linear_regression_execution_time_predictor_config_moe_trace_path",
        str(args.trace_root),
        "--h_b_f_linear_regression_execution_time_predictor_config_moe_trace_policy",
        policy,
        "--h_b_f_linear_regression_execution_time_predictor_config_moe_trace_workload",
        workload,
        "--h_b_f_linear_regression_execution_time_predictor_config_moe_trace_max_requests",
        str(max(args.trace_max_requests, sessions)),
        "--h_b_f_linear_regression_execution_time_predictor_config_moe_trace_profile_max_requests",
        str(args.trace_profile_max_requests),
        "--h_b_f_linear_regression_execution_time_predictor_config_moe_trace_read_bw_gbps",
        str(args.trace_read_bw_gbps),
        "--h_b_f_linear_regression_execution_time_predictor_config_moe_trace_diagnostics_path",
        str(diagnostics_path),
        "--metrics_config_output_dir",
        str(run_base),
        "--no-metrics_config_write_json_trace",
        "--no-metrics_config_store_plots",
        "--no-metrics_config_enable_chrome_trace",
        "--no-metrics_config_store_operation_metrics",
        "--no-metrics_config_store_token_completion_metrics",
        "--no-metrics_config_store_utilization_metrics",
        "--log_level",
        "warning",
    ]


def run_vidur(argv: list[str]) -> Path:
    reset_vidur_entity_ids()

    from vidur.config import SimulationConfig
    from vidur.simulator import Simulator
    from vidur.utils.random import set_seeds

    saved = sys.argv[:]
    sys.argv = argv
    try:
        config = SimulationConfig.create_from_cli_args()
        set_seeds(config.seed)
        simulator = Simulator(config)
        simulator.run()
        simulator._write_output()
        try:
            atexit.unregister(simulator._write_output)
        except Exception:
            pass
        return Path(config.metrics_config.output_dir)
    finally:
        sys.argv = saved


def reset_vidur_entity_ids() -> None:
    """Reset Vidur process-global counters before an in-process sweep run.

    Vidur assumes a fresh Python process for each simulation. If several
    simulations run in one sweep process, class-level entity IDs keep
    increasing, while the synthetic global scheduler still assigns requests to
    replica 0. Resetting these counters preserves the single-run semantics for
    each matrix point.
    """

    from vidur.entities.batch import Batch
    from vidur.entities.batch_stage import BatchStage
    from vidur.entities.cluster import Cluster
    from vidur.entities.execution_time import ExecutionTime
    from vidur.entities.replica import Replica
    from vidur.entities.request import Request
    from vidur.events.base_event import BaseEvent

    for cls in (Batch, BatchStage, Cluster, ExecutionTime, Replica, Request):
        cls._id = -1
    BaseEvent._id = 0


def read_metrics(output_dir: Path) -> dict[str, float | int | str]:
    matches = glob.glob(str(output_dir / "**" / "request_metrics.csv"), recursive=True)
    if not matches:
        raise FileNotFoundError(f"request_metrics.csv not found under {output_dir}")
    with open(matches[0], encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty request metrics: {matches[0]}")

    tpot = [
        float(row["decode_time_execution_plus_preemption_normalized"]) * 1000.0
        for row in rows
        if row.get("decode_time_execution_plus_preemption_normalized")
    ]
    arrivals = []
    completions = []
    arrival = 0.0
    decode_tokens = 0.0
    for row in sorted(rows, key=lambda r: int(float(r["Request Id"]))):
        delay = row.get("request_inter_arrival_delay")
        if delay:
            arrival += float(delay)
        arrivals.append(arrival)
        completions.append(arrival + float(row["request_e2e_time"]))
        decode_tokens += float(row["request_num_decode_tokens"])
    duration = max(completions) - min(arrivals)
    return {
        "requests": len(rows),
        "tpot_p50_ms": float(np.percentile(tpot, 50)),
        "tpot_p95_ms": float(np.percentile(tpot, 95)),
        "tpot_p99_ms": float(np.percentile(tpot, 99)),
        "throughput_tok_s": float(decode_tokens / duration) if duration > 0 else 0.0,
        "vidur_output_dir": str(output_dir),
    }


def summarize_diagnostics(path: Path) -> dict[str, float]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    def mean(key: str) -> float:
        vals = [float(row[key]) for row in rows if key in row]
        return float(np.mean(vals)) if vals else 0.0
    return {
        "moe_layer_time_ms_mean": mean("moe_layer_time_ms_mean"),
        "active_experts_mean": mean("active_experts_mean"),
        "max_fetched_experts_p95_mean": mean("max_fetched_experts_p95"),
        "hbf_fetch_fraction_mean": mean("hbf_fetch_fraction_mean"),
        "hbf_load_fraction_mean": mean("hbf_load_fraction_mean"),
        "reassignment_fraction_mean": mean("reassignment_fraction_mean"),
    }


def _read_profiled_tp_sizes(path: Path) -> set[int]:
    if not path.exists():
        raise FileNotFoundError(f"missing profiling table: {path}")
    values: set[int] = set()
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            raw = row.get("num_tensor_parallel_workers")
            if raw not in (None, ""):
                values.add(int(float(raw)))
    if not values:
        raise ValueError(f"no num_tensor_parallel_workers values in {path}")
    return values


def profiled_attention_gpu_counts(args: argparse.Namespace) -> set[int]:
    profile_dir = REPO / "data" / "profiling" / "compute" / args.device / Path(args.model)
    attention = _read_profiled_tp_sizes(profile_dir / "attention.csv")
    mlp = _read_profiled_tp_sizes(profile_dir / "mlp.csv")
    values = attention & mlp
    if not values:
        raise ValueError(f"no shared profiled TP sizes in {profile_dir}")
    return values


def valid_budget_splits(args: argparse.Namespace) -> list[dict[str, int]]:
    if not args.optimize_am:
        return [
            {
                "gpu_budget": args.attention_gpus + moe_gpus,
                "attention_gpus": args.attention_gpus,
                "moe_gpus": moe_gpus,
            }
            for moe_gpus in args.moe_gpus
        ]

    profiled = profiled_attention_gpu_counts(args)
    run_points: list[dict[str, int]] = []
    for budget in args.gpu_budgets:
        candidates: list[dict[str, int]] = []
        for moe_gpus in range(1, budget):
            attention_gpus = budget - moe_gpus
            if 256 % moe_gpus != 0:
                continue
            if attention_gpus not in profiled:
                continue
            candidates.append(
                {
                    "gpu_budget": budget,
                    "attention_gpus": attention_gpus,
                    "moe_gpus": moe_gpus,
                }
            )
        if not candidates:
            raise ValueError(
                f"no valid A/M splits for budget {budget}; profiled A values are "
                f"{sorted(profiled)} and M must divide 256"
            )
        run_points.extend(sorted(candidates, key=lambda row: (row["moe_gpus"], row["attention_gpus"])))
    return run_points


def metric_float(row: dict[str, object], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def select_best_rows(rows: list[dict[str, object]], slo_ms: float) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            row.get("workload"),
            row.get("context_length"),
            row.get("sessions"),
            row.get("gpu_budget"),
            row.get("policy"),
        )
        groups.setdefault(key, []).append(row)

    best_rows: list[dict[str, object]] = []
    for key, group in sorted(groups.items(), key=lambda item: item[0]):
        ok = [row for row in group if row.get("status", "ok") == "ok"]
        if not ok:
            row = dict(group[0])
            row["slo_feasible"] = False
            row["selection_reason"] = "no_valid_split"
            best_rows.append(row)
            continue
        feasible = [row for row in ok if metric_float(row, "tpot_p95_ms") <= slo_ms]
        if feasible:
            best = min(
                feasible,
                key=lambda row: (
                    -metric_float(row, "throughput_tok_s"),
                    metric_float(row, "tpot_p95_ms"),
                    -int(row.get("attention_gpus", 0)),
                ),
            )
            reason = "max_throughput_under_slo"
        else:
            best = min(
                ok,
                key=lambda row: (
                    metric_float(row, "tpot_p95_ms"),
                    -metric_float(row, "throughput_tok_s"),
                    -int(row.get("attention_gpus", 0)),
                ),
            )
            reason = "lowest_p95_no_slo_feasible"
        row = dict(best)
        row["slo_feasible"] = bool(feasible)
        row["selection_reason"] = reason
        best_rows.append(row)
    return best_rows


def write_budget_summary(path: Path, rows: list[dict[str, object]], slo_ms: float) -> None:
    lines = [
        "# Task 3 GPU-budget A/M optimization summary",
        "",
        "Scope: no-pruning R1 trace-backed Vidur replay with microbatch_count=2.",
        f"Best split rule: maximize throughput among p95 TPOT <= {slo_ms:g} ms; "
        "if none is feasible, report the lowest-p95 split.",
        "",
        "| Policy | Budget | Ctx | S | Best A | Best M | p95 TPOT (ms) | Throughput (tok/s) | SLO feasible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(
        rows,
        key=lambda row: (
            str(row.get("policy")),
            int(row.get("gpu_budget", 0)),
            int(row.get("context_length", 0)),
            int(row.get("sessions", 0)),
        ),
    ):
        if row.get("workload") != "mixed":
            continue
        tpot = metric_float(row, "tpot_p95_ms")
        throughput = metric_float(row, "throughput_tok_s")
        tpot_text = "—" if math.isnan(tpot) else f"{tpot:.2f}"
        throughput_text = "—" if math.isnan(throughput) else f"{throughput:.1f}"
        feasible = "yes" if row.get("slo_feasible") else "no"
        lines.append(
            f"| {POLICY_LABELS.get(str(row.get('policy')), row.get('policy'))} "
            f"| {row.get('gpu_budget')} | {int(row.get('context_length', 0)) // 1024}K "
            f"| {row.get('sessions')} | {row.get('attention_gpus')} | {row.get('moe_gpus')} "
            f"| {tpot_text} | {throughput_text} | {feasible} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_budget_results(rows: list[dict[str, object]], outdir: Path, slo_ms: float) -> None:
    rows = [row for row in rows if row.get("status", "ok") == "ok"]
    if not rows:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    workloads = sorted({str(row["workload"]) for row in rows})
    contexts = sorted({int(row["context_length"]) for row in rows})
    sessions = sorted({int(row["sessions"]) for row in rows})
    budgets = sorted({int(row["gpu_budget"]) for row in rows})

    for workload in workloads:
        policy = "cost_dynamic" if any(row["policy"] == "cost_dynamic" for row in rows) else str(rows[0]["policy"])
        selected = {
            (int(row["gpu_budget"]), int(row["context_length"]), int(row["sessions"])): row
            for row in rows
            if row["workload"] == workload and row["policy"] == policy
        }
        if selected:
            fig, axes = plt.subplots(1, len(budgets), figsize=(4.2 * len(budgets), 3.8), sharey=True)
            axes = np.atleast_1d(axes)
            for ax, budget in zip(axes, budgets):
                matrix = np.full((len(sessions), len(contexts)), np.nan)
                for yi, session in enumerate(sessions):
                    for xi, context in enumerate(contexts):
                        row = selected.get((budget, context, session))
                        if not row:
                            continue
                        matrix[yi, xi] = int(row["moe_gpus"]) / budget
                im = ax.imshow(matrix, aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
                for yi, session in enumerate(sessions):
                    for xi, context in enumerate(contexts):
                        row = selected.get((budget, context, session))
                        label = "—" if not row else f"{row['attention_gpus']}/{row['moe_gpus']}"
                        ax.text(xi, yi, label, ha="center", va="center", color="white", fontsize=7)
                ax.set_xticks(range(len(contexts)), [f"{c//1024}K" for c in contexts])
                ax.set_yticks(range(len(sessions)), [str(s) for s in sessions])
                ax.set_title(f"B={budget}")
                ax.set_xlabel("context")
                ax.grid(False)
            axes[0].set_ylabel("sessions")
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="M / budget")
            fig.suptitle(f"Best A/M split ({POLICY_LABELS.get(policy, policy)}), {workload}")
            fig.savefig(outdir / f"fig3_best_am_map_{workload}_{policy}.png", dpi=220, bbox_inches="tight")
            plt.close(fig)

            fig, axes = plt.subplots(1, len(budgets), figsize=(4.2 * len(budgets), 3.7), sharey=True)
            axes = np.atleast_1d(axes)
            for ax, budget in zip(axes, budgets):
                for session in sessions:
                    y = []
                    for context in contexts:
                        row = selected.get((budget, context, session))
                        feasible = row and metric_float(row, "tpot_p95_ms") <= slo_ms
                        y.append(metric_float(row, "throughput_tok_s") if feasible else np.nan)
                    ax.plot([c // 1024 for c in contexts], y, "o-", label=f"S={session}")
                ax.set_title(f"B={budget}")
                ax.set_xlabel("context (K)")
                ax.grid(alpha=0.25)
            axes[0].set_ylabel(f"throughput at p95 <= {slo_ms:g} ms (tok/s)")
            axes[-1].legend(frameon=False, fontsize=7)
            fig.suptitle(f"Best throughput under SLO ({POLICY_LABELS.get(policy, policy)}), {workload}")
            fig.tight_layout()
            fig.savefig(outdir / f"fig4_best_throughput_slo_{workload}_{policy}.png", dpi=220)
            plt.close(fig)

        by_policy = {
            (str(row["policy"]), int(row["gpu_budget"]), int(row["context_length"]), int(row["sessions"])): row
            for row in rows
            if row["workload"] == workload
        }
        if any(key[0] == "static" for key in by_policy) and any(key[0] == "cost_dynamic" for key in by_policy):
            fig, axes = plt.subplots(1, len(budgets), figsize=(4.2 * len(budgets), 3.8), sharey=True)
            axes = np.atleast_1d(axes)
            for ax, budget in zip(axes, budgets):
                matrix = np.full((len(sessions), len(contexts)), np.nan)
                for yi, session in enumerate(sessions):
                    for xi, context in enumerate(contexts):
                        st = by_policy.get(("static", budget, context, session))
                        dy = by_policy.get(("cost_dynamic", budget, context, session))
                        if not st or not dy:
                            continue
                        st_tpot = metric_float(st, "tpot_p95_ms")
                        dy_tpot = metric_float(dy, "tpot_p95_ms")
                        if st_tpot > 0 and not math.isnan(dy_tpot):
                            matrix[yi, xi] = (st_tpot - dy_tpot) / st_tpot * 100.0
                im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-25, vmax=25)
                for yi in range(len(sessions)):
                    for xi in range(len(contexts)):
                        value = matrix[yi, xi]
                        label = "—" if math.isnan(value) else f"{value:.0f}%"
                        ax.text(xi, yi, label, ha="center", va="center", fontsize=7)
                ax.set_xticks(range(len(contexts)), [f"{c//1024}K" for c in contexts])
                ax.set_yticks(range(len(sessions)), [str(s) for s in sessions])
                ax.set_title(f"B={budget}")
                ax.set_xlabel("context")
            axes[0].set_ylabel("sessions")
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="p95 TPOT reduction vs static (%)")
            fig.suptitle(f"Cost-dynamic vs static, each with own best A/M, {workload}")
            fig.savefig(outdir / f"fig5_cost_dynamic_vs_static_{workload}.png", dpi=220, bbox_inches="tight")
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(6.2, 3.8))
            for policy_name, marker in (("static", "o"), ("cost_dynamic", "s")):
                for budget in budgets:
                    y = []
                    for context in contexts:
                        feasible_sessions = [
                            session
                            for session in sessions
                            if (row := by_policy.get((policy_name, budget, context, session)))
                            and metric_float(row, "tpot_p95_ms") <= slo_ms
                        ]
                        y.append(max(feasible_sessions) if feasible_sessions else np.nan)
                    ax.plot(
                        [c // 1024 for c in contexts],
                        y,
                        marker + "-",
                        label=f"{POLICY_LABELS.get(policy_name, policy_name)} B={budget}",
                    )
            ax.set_xlabel("context (K)")
            ax.set_ylabel(f"max feasible S at p95 <= {slo_ms:g} ms")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False, fontsize=7, ncol=2)
            fig.tight_layout()
            fig.savefig(outdir / f"fig6_max_feasible_sessions_{workload}.png", dpi=220)
            plt.close(fig)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def plot_results(rows: list[dict[str, object]], outdir: Path, slo_ms: float) -> None:
    rows = [row for row in rows if row.get("status", "ok") == "ok"]
    if not rows:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    contexts = sorted({int(row["context_length"]) for row in rows})
    workloads = sorted({str(row["workload"]) for row in rows})
    for context in contexts:
        for workload in workloads:
            subset = [
                row
                for row in rows
                if int(row["context_length"]) == context and row["workload"] == workload
            ]
            if not subset:
                continue
            moe_values = sorted({int(row["moe_gpus"]) for row in subset})
            fig, axes = plt.subplots(1, len(moe_values), figsize=(4.5 * len(moe_values), 3.6), sharey=True)
            axes = np.atleast_1d(axes)
            for ax, moe_gpus in zip(axes, moe_values):
                data = [row for row in subset if int(row["moe_gpus"]) == moe_gpus]
                for policy in POLICIES:
                    series = sorted(
                        (row for row in data if row["policy"] == policy),
                        key=lambda row: int(row["sessions"]),
                    )
                    if not series:
                        continue
                    ax.plot(
                        [int(row["sessions"]) for row in series],
                        [float(row["tpot_p95_ms"]) for row in series],
                        "o-",
                        label=POLICY_LABELS[policy],
                        color=POLICY_COLORS[policy],
                    )
                ax.axhline(slo_ms, color="black", lw=0.8, ls=":")
                ax.set_title(f"M={moe_gpus}")
                ax.set_xlabel("sessions")
                ax.grid(alpha=0.25)
            axes[0].set_ylabel("p95 TPOT/TBT (ms)")
            axes[-1].legend(frameon=False, fontsize=8)
            fig.suptitle(f"Trace-backed Vidur no-pruning, {workload}, ctx={context//1024}K")
            fig.tight_layout()
            fig.savefig(outdir / f"fig1_tpot_p95_{workload}_ctx{context//1024}k.png", dpi=220)
            plt.close(fig)

            fig, axes = plt.subplots(1, len(moe_values), figsize=(4.5 * len(moe_values), 3.6), sharey=True)
            axes = np.atleast_1d(axes)
            for ax, moe_gpus in zip(axes, moe_values):
                data = [row for row in subset if int(row["moe_gpus"]) == moe_gpus]
                for policy in POLICIES:
                    series = sorted(
                        (row for row in data if row["policy"] == policy),
                        key=lambda row: int(row["sessions"]),
                    )
                    if not series:
                        continue
                    ax.plot(
                        [int(row["sessions"]) for row in series],
                        [
                            float(row["throughput_tok_s"])
                            if float(row["tpot_p95_ms"]) <= slo_ms
                            else np.nan
                            for row in series
                        ],
                        "o-",
                        label=POLICY_LABELS[policy],
                        color=POLICY_COLORS[policy],
                    )
                ax.set_title(f"M={moe_gpus}")
                ax.set_xlabel("sessions")
                ax.grid(alpha=0.25)
            axes[0].set_ylabel(f"throughput at p95 <= {slo_ms:g} ms (tok/s)")
            axes[-1].legend(frameon=False, fontsize=8)
            fig.suptitle(f"Trace-backed Vidur SLO throughput, {workload}, ctx={context//1024}K")
            fig.tight_layout()
            fig.savefig(outdir / f"fig2_throughput_slo_{workload}_ctx{context//1024}k.png", dpi=220)
            plt.close(fig)


def write_summary(path: Path, rows: list[dict[str, object]], slo_ms: float) -> None:
    rows = [row for row in rows if row.get("status", "ok") == "ok"]
    if not rows:
        path.write_text("# Task 3 trace-backed Vidur summary\n\nNo successful runs.\n", encoding="utf-8")
        return
    mixed = [
        row
        for row in rows
        if row["workload"] == "mixed" and int(row["context_length"]) == min(int(r["context_length"]) for r in rows)
    ]
    lines = [
        "# Task 3 trace-backed Vidur summary",
        "",
        "Scope: DeepSeek-R1-AWQ selected-expert traces, no pruning, Vidur decode-only serving.",
        "These are not DeepSeek-V3 measured hardware results.",
        "",
        f"SLO used for throughput masking: p95 TPOT <= {slo_ms:g} ms.",
        "",
        "| Policy | M | S | p95 TPOT (ms) | Throughput (tok/s) | Active experts | Churn |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(mixed, key=lambda r: (r["policy"], int(r["moe_gpus"]), int(r["sessions"]))):
        lines.append(
            f"| {POLICY_LABELS.get(str(row['policy']), row['policy'])} "
            f"| {row['moe_gpus']} | {row['sessions']} "
            f"| {float(row['tpot_p95_ms']):.2f} "
            f"| {float(row['throughput_tok_s']):.1f} "
            f"| {float(row.get('active_experts_mean', 0.0)):.1f} "
            f"| {100.0 * float(row.get('reassignment_fraction_mean', 0.0)):.1f}% |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cap_worker_threads() -> None:
    """Keep each parallel Vidur worker from over-subscribing BLAS threads."""

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(name, "1")


def build_run_specs(
    args: argparse.Namespace, run_points: list[dict[str, int]]
) -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    run_count = 0
    for workload in args.workloads:
        for context in args.context_lengths:
            for sessions in args.sessions:
                for split in run_points:
                    for policy in args.policies:
                        run_count += 1
                        if args.max_runs and run_count > args.max_runs:
                            return specs
                        attention_gpus = split["attention_gpus"]
                        moe_gpus = split["moe_gpus"]
                        gpu_budget = split["gpu_budget"]
                        run_name = (
                            f"{workload}_ctx{context}_S{sessions}_"
                            f"B{gpu_budget}_A{attention_gpus}_M{moe_gpus}_{policy}"
                        )
                        run_base = args.outdir / "vidur_runs" / run_name
                        diagnostics_path = run_base / "moe_trace_diagnostics.jsonl"
                        row: dict[str, object] = {
                            "workload": workload,
                            "context_length": context,
                            "sessions": sessions,
                            "microbatch_count": args.microbatch_count,
                            "sessions_per_microbatch": sessions // args.microbatch_count,
                            "gpu_budget": gpu_budget,
                            "attention_gpus": attention_gpus,
                            "moe_gpus": moe_gpus,
                            "policy": policy,
                            "decode_tokens": args.decode_tokens,
                            "status": "pending",
                            "error": "",
                        }
                        argv = vidur_argv(
                            args,
                            run_base,
                            diagnostics_path,
                            workload,
                            context,
                            sessions,
                            attention_gpus,
                            moe_gpus,
                            policy,
                        )
                        specs.append(
                            {
                                "run_name": run_name,
                                "argv": argv,
                                "diagnostics_path": str(diagnostics_path),
                                "row": row,
                            }
                        )
    return specs


def execute_run_spec(spec: dict[str, object]) -> dict[str, object]:
    cap_worker_threads()
    row = dict(spec["row"])
    try:
        output_dir = run_vidur(list(spec["argv"]))
        row.update(read_metrics(output_dir))
        row.update(summarize_diagnostics(Path(str(spec["diagnostics_path"]))))
        row["status"] = "ok"
    except Exception as exc:  # keep long sweeps moving
        row["status"] = "invalid"
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def execute_run_specs(specs: list[dict[str, object]], jobs: int) -> list[dict[str, object]]:
    if jobs < 1:
        raise ValueError("--jobs must be >= 1")
    if jobs == 1:
        rows: list[dict[str, object]] = []
        for spec in specs:
            print(f"running {spec['run_name']}", flush=True)
            row = execute_run_spec(spec)
            if row.get("status") != "ok":
                print(f"invalid {spec['run_name']}: {row.get('error')}", flush=True)
            rows.append(row)
        return rows

    cap_worker_threads()
    rows = []
    print(f"running {len(specs)} Vidur configs with jobs={jobs}", flush=True)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        future_to_spec = {pool.submit(execute_run_spec, spec): spec for spec in specs}
        for index, future in enumerate(as_completed(future_to_spec), start=1):
            spec = future_to_spec[future]
            try:
                row = future.result()
            except Exception as exc:
                row = dict(spec["row"])
                row["status"] = "invalid"
                row["error"] = f"WorkerCrash: {type(exc).__name__}: {exc}"
            rows.append(row)
            status = row.get("status")
            suffix = "" if status == "ok" else f" error={row.get('error')}"
            print(
                f"completed {index}/{len(specs)} {spec['run_name']} status={status}{suffix}",
                flush=True,
            )
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("workload")),
            int(row.get("context_length", 0)),
            int(row.get("sessions", 0)),
            int(row.get("gpu_budget", 0)),
            int(row.get("attention_gpus", 0)),
            int(row.get("moe_gpus", 0)),
            str(row.get("policy")),
        ),
    )


def write_readme(path: Path, args: argparse.Namespace) -> None:
    text = f"""# DeepSeek-R1-AWQ Task 3 Vidur evaluation

This folder is isolated from `paper_eval/figures/current`.

Scope:
- no pruning;
- R1-AWQ selected-expert traces used as V3-shaped routing traces;
- Vidur is the outer serving simulator;
- trace-backed MoE timing is enabled through `moe_trace_path`;
- HBM/HBF expert-read bandwidth is {args.trace_read_bw_gbps:g} GB/s.

Important limitation: the source traces contain selected expert IDs only. They
do not contain router weights, token IDs, arrivals, or measured latency, so this
experiment evaluates no-pruning serving sensitivity, not quality.

Files:
- `results/task3_vidur_summary.csv`: one row per Vidur run.
- `results/task3_vidur_summary.md`: compact mixed-workload table for fixed-split runs.
- `results/task3_budget_*.csv`: optimizer all-runs and best-split tables when `--optimize-am` is used.
- `results/task3_budget_summary.md`: compact best-A/M table when `--optimize-am` is used.
- `figures/*.png`: PNG-only figures.
- `vidur_runs/`: raw Vidur output directories and per-run MoE diagnostics.
- `--jobs N`: runs independent Vidur configs in parallel worker processes.
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = arguments()
    bad = [policy for policy in args.policies if policy not in POLICIES]
    if bad:
        raise ValueError(f"unknown policies: {bad}")
    if any(s % args.microbatch_count for s in args.sessions):
        raise ValueError("every session count must divide by microbatch_count")

    run_points = valid_budget_splits(args)
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "results").mkdir(exist_ok=True)
    (args.outdir / "figures").mkdir(exist_ok=True)
    (args.outdir / "vidur_runs").mkdir(exist_ok=True)
    write_readme(args.outdir / "README.md", args)

    specs = build_run_specs(args, run_points)
    if not specs:
        raise RuntimeError("no Vidur runs selected")
    rows = execute_run_specs(specs, args.jobs)

    if not rows:
        raise RuntimeError("no Vidur runs completed")

    write_csv(args.outdir / "results" / "task3_vidur_summary.csv", rows)

    if args.optimize_am:
        best_rows = select_best_rows(rows, args.slo_ms)
        flexep_rows = [row for row in best_rows if row.get("policy") == "cost_dynamic"]
        write_csv(args.outdir / "results" / "task3_budget_all_runs.csv", rows)
        write_csv(args.outdir / "results" / "task3_budget_best_by_policy.csv", best_rows)
        if flexep_rows:
            write_csv(args.outdir / "results" / "task3_budget_best_flexep.csv", flexep_rows)
        write_budget_summary(
            args.outdir / "results" / "task3_budget_summary.md",
            best_rows,
            args.slo_ms,
        )
        plot_budget_results(best_rows, args.outdir / "figures", args.slo_ms)
    else:
        write_summary(args.outdir / "results" / "task3_vidur_summary.md", rows, args.slo_ms)
        plot_results(rows, args.outdir / "figures", args.slo_ms)

    run_config = {
        "trace_root": str(args.trace_root),
        "context_lengths": list(args.context_lengths),
        "sessions": list(args.sessions),
        "microbatch_count": args.microbatch_count,
        "moe_gpus": list(args.moe_gpus),
        "attention_gpus": args.attention_gpus,
        "gpu_budgets": list(args.gpu_budgets),
        "optimize_am": args.optimize_am,
        "run_points": run_points,
        "policies": list(args.policies),
        "workloads": list(args.workloads),
        "decode_tokens": args.decode_tokens,
        "slo_ms": args.slo_ms,
        "jobs": args.jobs,
        "no_pruning": True,
        "simulator": "Vidur",
    }
    (args.outdir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} rows under {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
