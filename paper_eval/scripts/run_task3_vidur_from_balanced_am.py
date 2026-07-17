#!/usr/bin/env python3
"""Run Vidur on raw-step selected A/M splits.

This runner is intentionally separate from run_task3_trace_vidur.py. The old
Task 3 runner can optimize A/M internally using Vidur p95/throughput, which is
not the split-selection rule we want here. This script reads the raw-step sweep
candidates, filters to attention TP sizes that Vidur has local profiles for,
selects the lowest raw pipeline step per workload/context/S/budget/policy, and
then runs Vidur exactly on those selected splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER = SCRIPT_DIR.parent
REPO = PAPER.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import run_task3_trace_vidur as task3  # noqa: E402

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
    / "deepseek_r1_awq_task3_vidur"
    / "balanced_am_exact_profiledA_mu2_vidur"
)


def int_filter(text: str) -> tuple[int, ...] | None:
    text = text.strip()
    if not text or text.lower() == "all":
        return None
    values = tuple(int(x) for x in text.split(",") if x.strip())
    if not values:
        return None
    return values


def str_filter(text: str) -> tuple[str, ...] | None:
    text = text.strip()
    if not text or text.lower() == "all":
        return None
    values = tuple(x.strip() for x in text.split(",") if x.strip())
    if not values:
        return None
    return values


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
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--trace-root", type=Path, default=task3.DEFAULT_TRACE_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--context-lengths", type=int_filter, default=None)
    parser.add_argument("--sessions", type=int_filter, default=None)
    parser.add_argument("--gpu-budgets", type=int_filter, default=None)
    parser.add_argument("--policies", type=str_list, default=("static", "cost_dynamic"))
    parser.add_argument("--workloads", type=str_filter, default=("mixed",))
    parser.add_argument("--microbatch-count", type=int, default=2)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--slo-ms", type=float, default=100.0)
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V3")
    parser.add_argument("--device", default="h100")
    parser.add_argument("--network-device", default="h100_pairwise_nvlink")
    parser.add_argument("--hbf-config", default="configs/hbf_paper_722g.toml")
    parser.add_argument("--trace-max-requests", type=int, default=512)
    parser.add_argument("--trace-profile-max-requests", type=int, default=0)
    parser.add_argument("--trace-read-bw-gbps", type=float, default=1024.0)
    parser.add_argument(
        "--disagg-link-bw-gbps",
        type=float,
        default=900.0,
        help="Default matches the raw-step selector's link_bw_gbps column.",
    )
    parser.add_argument("--overlap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--allowed-attention-gpus",
        type=int_list,
        default=None,
        help="Override profiled attention TP sizes. Default: read local profiling table.",
    )
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--max-rows", type=int, default=0)
    return parser.parse_args()


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def as_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    return float(value)


def read_candidate_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def passes_filter(value: int | str, allowed: Iterable[int | str] | None) -> bool:
    return allowed is None or value in set(allowed)


def select_splits(
    args: argparse.Namespace,
    allowed_attention_gpus: set[int],
) -> list[dict[str, object]]:
    rows = read_candidate_rows(args.candidate_csv)
    grouped: dict[tuple[object, ...], list[dict[str, str]]] = {}
    wanted_policies = set(args.policies)
    wanted_contexts = None if args.context_lengths is None else set(args.context_lengths)
    wanted_sessions = None if args.sessions is None else set(args.sessions)
    wanted_budgets = None if args.gpu_budgets is None else set(args.gpu_budgets)
    wanted_workloads = None if args.workloads is None else set(args.workloads)

    for row in rows:
        if row.get("policy") not in wanted_policies:
            continue
        context = as_int(row, "context_length")
        sessions = as_int(row, "sessions")
        budget = as_int(row, "gpu_budget")
        workload = row.get("workload", "mixed")
        attention_gpus = as_int(row, "attention_gpus")
        if attention_gpus not in allowed_attention_gpus:
            continue
        if as_int(row, "microbatch_count") != args.microbatch_count:
            continue
        if not as_bool(row.get("capacity_ok", "true")):
            continue
        if not passes_filter(context, wanted_contexts):
            continue
        if not passes_filter(sessions, wanted_sessions):
            continue
        if not passes_filter(budget, wanted_budgets):
            continue
        if not passes_filter(workload, wanted_workloads):
            continue
        key = (workload, context, sessions, budget, row["policy"])
        grouped.setdefault(key, []).append(row)

    selected: list[dict[str, object]] = []
    for key, group in sorted(grouped.items(), key=lambda item: item[0]):
        best = min(
            group,
            key=lambda row: (
                as_float(row, "raw_step_ms_mean"),
                as_float(row, "raw_step_ms_p95"),
                as_int(row, "gpus"),
                as_int(row, "moe_gpus"),
            ),
        )
        selected.append(
            {
                "workload": best.get("workload", "mixed"),
                "context_length": as_int(best, "context_length"),
                "sessions": as_int(best, "sessions"),
                "microbatch_count": as_int(best, "microbatch_count"),
                "sessions_per_microbatch": as_float(best, "sessions_per_microbatch"),
                "gpu_budget": as_int(best, "gpu_budget"),
                "gpus": as_int(best, "gpus"),
                "attention_gpus": as_int(best, "attention_gpus"),
                "moe_gpus": as_int(best, "moe_gpus"),
                "policy": best["policy"],
                "budget_mode": best.get("budget_mode", "exact"),
                "source_raw_attention_ms": as_float(best, "raw_attention_ms"),
                "source_raw_moe_ms_mean": as_float(best, "raw_moe_ms_mean"),
                "source_raw_step_ms_mean": as_float(best, "raw_step_ms_mean"),
                "source_raw_step_ms_p95": as_float(best, "raw_step_ms_p95"),
                "source_selection_reason": "min_raw_step_profiled_attention",
            }
        )

    if args.max_rows:
        selected = selected[: args.max_rows]
    if not selected:
        raise RuntimeError("no selected splits after filtering candidate CSV")
    return selected


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    task3.write_csv(path, rows)


def build_specs(args: argparse.Namespace, selected: list[dict[str, object]]) -> list[dict[str, object]]:
    args.sessions = tuple(sorted({int(row["sessions"]) for row in selected}))
    specs: list[dict[str, object]] = []
    for row_in in selected:
        row = dict(row_in)
        workload = str(row["workload"])
        context = int(row["context_length"])
        sessions = int(row["sessions"])
        attention_gpus = int(row["attention_gpus"])
        moe_gpus = int(row["moe_gpus"])
        gpu_budget = int(row["gpu_budget"])
        policy = str(row["policy"])
        run_name = (
            f"{workload}_ctx{context}_S{sessions}_"
            f"B{gpu_budget}_A{attention_gpus}_M{moe_gpus}_{policy}"
        )
        run_base = args.outdir / "vidur_runs" / run_name
        diagnostics_path = run_base / "moe_trace_diagnostics.jsonl"
        row.update(
            {
                "decode_tokens": args.decode_tokens,
                "status": "pending",
                "error": "",
            }
        )
        argv = task3.vidur_argv(
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
        argv.extend(
            [
                "--h_b_f_linear_regression_execution_time_predictor_config_num_training_job_threads",
                "1",
            ]
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


def metric(row: dict[str, object], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def write_summary(
    path: Path,
    rows: list[dict[str, object]],
    selected: list[dict[str, object]],
    allowed_attention_gpus: set[int],
    args: argparse.Namespace,
) -> None:
    ok = [row for row in rows if row.get("status") == "ok"]
    lines = [
        "# Vidur replay from raw-step selected A/M splits",
        "",
        f"Candidate CSV: `{args.candidate_csv}`",
        f"Selected split CSV: `inputs/selected_profiled_attention_splits.csv`",
        f"Allowed attention GPU counts: `{sorted(allowed_attention_gpus)}`",
        f"Rows selected: `{len(selected)}`; Vidur rows OK: `{len(ok)}/{len(rows)}`.",
        f"SLO: p95 TPOT <= `{args.slo_ms:g}` ms.",
        f"Trace expert read BW: `{args.trace_read_bw_gbps:g}` GB/s; disagg link BW: `{args.disagg_link_bw_gbps:g}` GB/s.",
        "",
    ]
    if len(ok) != len(rows):
        lines.extend(["## Invalid rows", "", "| Workload | Ctx | S | B | A | M | Policy | Error |", "|---|---:|---:|---:|---:|---:|---|---|"])
        for row in rows:
            if row.get("status") == "ok":
                continue
            lines.append(
                f"| {row.get('workload')} | {int(row.get('context_length', 0)) // 1024}K "
                f"| {row.get('sessions')} | {row.get('gpu_budget')} | {row.get('attention_gpus')} "
                f"| {row.get('moe_gpus')} | {row.get('policy')} | {row.get('error')} |"
            )
        lines.append("")

    by_key = {
        (
            row.get("workload"),
            int(row.get("context_length", 0)),
            int(row.get("sessions", 0)),
            int(row.get("gpu_budget", 0)),
            row.get("policy"),
        ): row
        for row in ok
    }
    paired = []
    for key, static in by_key.items():
        if key[-1] != "static":
            continue
        dyn_key = key[:-1] + ("cost_dynamic",)
        dynamic = by_key.get(dyn_key)
        if dynamic is None:
            continue
        st = metric(static, "tpot_p95_ms")
        dy = metric(dynamic, "tpot_p95_ms")
        if st > 0 and not math.isnan(dy):
            paired.append((st - dy) / st * 100.0)
    if paired:
        lines.extend(
            [
                "## Dynamic-vs-static p95 TPOT reduction",
                "",
                f"Matched cells: `{len(paired)}`.",
                f"Average reduction: `{sum(paired) / len(paired):.2f}%`.",
                f"Best reduction: `{max(paired):.2f}%`.",
                f"Worst reduction: `{min(paired):.2f}%`.",
                "",
            ]
        )

    lines.extend(
        [
            "## Cost-dynamic max feasible sessions",
            "",
            "| Budget | Ctx | Max feasible S |",
            "|---:|---:|---:|",
        ]
    )
    budgets = sorted({int(row.get("gpu_budget", 0)) for row in ok})
    contexts = sorted({int(row.get("context_length", 0)) for row in ok})
    for budget in budgets:
        for context in contexts:
            feasible = [
                int(row.get("sessions", 0))
                for row in ok
                if row.get("policy") == "cost_dynamic"
                and int(row.get("gpu_budget", 0)) == budget
                and int(row.get("context_length", 0)) == context
                and metric(row, "tpot_p95_ms") <= args.slo_ms
            ]
            text = str(max(feasible)) if feasible else "—"
            lines.append(f"| {budget} | {context // 1024}K | {text} |")

    lines.extend(
        [
            "",
            "## S=128 cost-dynamic detail",
            "",
            "| B | Ctx | A | M | Raw step ms | Vidur p95 TPOT ms | Throughput tok/s | SLO |",
            "|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(
        [row for row in ok if row.get("policy") == "cost_dynamic" and int(row.get("sessions", 0)) == 128],
        key=lambda row: (int(row.get("gpu_budget", 0)), int(row.get("context_length", 0))),
    ):
        p95 = metric(row, "tpot_p95_ms")
        lines.append(
            f"| {row.get('gpu_budget')} | {int(row.get('context_length', 0)) // 1024}K "
            f"| {row.get('attention_gpus')} | {row.get('moe_gpus')} "
            f"| {metric(row, 'source_raw_step_ms_mean'):.2f} "
            f"| {p95:.2f} | {metric(row, 'throughput_tok_s'):.1f} "
            f"| {'yes' if p95 <= args.slo_ms else 'no'} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(path: Path, args: argparse.Namespace) -> None:
    path.write_text(
        f"""# Vidur from balanced A/M CSV

This folder is isolated from earlier Task 3 runs.

Flow:
1. Read raw-step candidates from `{args.candidate_csv}`.
2. Filter to attention GPU counts that local Vidur profiling supports.
3. Select the min raw-step A/M per workload/context/S/budget/policy.
4. Run Vidur exactly on those selected splits.

Outputs:
- `inputs/selected_profiled_attention_splits.csv`: exact A/M rows used as Vidur input.
- `results/task3_vidur_selected_profiledA.csv`: measured Vidur rows.
- `results/task3_budget_best_by_policy.csv`: same measured rows, named for compatibility with plotting.
- `results/task3_budget_summary.md`: compact summary.
- `figures/*.png`: PNG-only figures.
""",
        encoding="utf-8",
    )


def main() -> None:
    args = arguments()
    bad = [policy for policy in args.policies if policy not in task3.POLICIES]
    if bad:
        raise ValueError(f"unknown policies: {bad}")
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "inputs").mkdir(exist_ok=True)
    (args.outdir / "results").mkdir(exist_ok=True)
    (args.outdir / "figures").mkdir(exist_ok=True)
    (args.outdir / "vidur_runs").mkdir(exist_ok=True)

    if args.allowed_attention_gpus is None:
        allowed_attention_gpus = task3.profiled_attention_gpu_counts(args)
    else:
        allowed_attention_gpus = set(args.allowed_attention_gpus)

    selected = select_splits(args, allowed_attention_gpus)
    write_csv(args.outdir / "inputs" / "selected_profiled_attention_splits.csv", selected)
    specs = build_specs(args, selected)
    write_readme(args.outdir / "README.md", args)

    rows = task3.execute_run_specs(specs, args.jobs)
    write_csv(args.outdir / "results" / "task3_vidur_selected_profiledA.csv", rows)
    write_csv(args.outdir / "results" / "task3_budget_best_by_policy.csv", rows)
    flexep = [row for row in rows if row.get("policy") == "cost_dynamic"]
    if flexep:
        write_csv(args.outdir / "results" / "task3_budget_best_flexep.csv", flexep)
    task3.plot_budget_results(rows, args.outdir / "figures", args.slo_ms)
    write_summary(args.outdir / "results" / "task3_budget_summary.md", rows, selected, allowed_attention_gpus, args)

    run_config = {
        "candidate_csv": str(args.candidate_csv),
        "trace_root": str(args.trace_root),
        "outdir": str(args.outdir),
        "allowed_attention_gpus": sorted(allowed_attention_gpus),
        "selected_rows": len(selected),
        "policies": list(args.policies),
        "workloads": None if args.workloads is None else list(args.workloads),
        "context_lengths": None if args.context_lengths is None else list(args.context_lengths),
        "sessions": sorted({int(row["sessions"]) for row in selected}),
        "gpu_budgets": sorted({int(row["gpu_budget"]) for row in selected}),
        "microbatch_count": args.microbatch_count,
        "decode_tokens": args.decode_tokens,
        "slo_ms": args.slo_ms,
        "trace_read_bw_gbps": args.trace_read_bw_gbps,
        "disagg_link_bw_gbps": args.disagg_link_bw_gbps,
        "jobs": args.jobs,
        "simulator": "Vidur",
        "no_pruning": True,
    }
    (args.outdir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} Vidur rows under {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
