#!/usr/bin/env python3
"""Run Vidur on trace-backed hybrid HBM/HBF MoE balancer rows.

Input is the raw hybrid checker's best_by_hbf_label.csv. Each row already has a
selected A/M/H split. This runner only validates serving behavior in Vidur.
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

DEFAULT_INPUT = (
    PAPER
    / 'experiments'
    / 'deepseek_r1_awq_task3_hybrid_moe_balancers'
    / 'initial_mu2_profiledA'
    / 'results'
    / 'best_by_hbf_label.csv'
)
DEFAULT_OUT = (
    PAPER
    / 'experiments'
    / 'deepseek_r1_awq_task3_vidur'
    / 'hybrid_hbf_balancers_mu2_vidur'
)


def int_filter(text: str) -> tuple[int, ...] | None:
    text = text.strip()
    if not text or text.lower() == 'all':
        return None
    values = tuple(int(x) for x in text.split(',') if x.strip())
    return values or None


def str_filter(text: str) -> tuple[str, ...] | None:
    text = text.strip()
    if not text or text.lower() == 'all':
        return None
    values = tuple(x.strip() for x in text.split(',') if x.strip())
    return values or None


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--trace-root', type=Path, default=task3.DEFAULT_TRACE_ROOT)
    parser.add_argument('--outdir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--context-lengths', type=int_filter, default=None)
    parser.add_argument('--sessions', type=int_filter, default=None)
    parser.add_argument('--gpu-budgets', type=int_filter, default=None)
    parser.add_argument('--hbf-labels', type=str_filter, default=('1', '2', '4', 'all'))
    parser.add_argument('--workloads', type=str_filter, default=('mixed',))
    parser.add_argument('--microbatch-count', type=int, default=2)
    parser.add_argument('--decode-tokens', type=int, default=128)
    parser.add_argument('--slo-ms', type=float, default=100.0)
    parser.add_argument('--model', default='deepseek-ai/DeepSeek-V3')
    parser.add_argument('--device', default='h100')
    parser.add_argument('--network-device', default='h100_pairwise_nvlink')
    parser.add_argument('--hbf-config', default='configs/hbf_paper_722g.toml')
    parser.add_argument('--trace-max-requests', type=int, default=512)
    parser.add_argument('--trace-profile-max-requests', type=int, default=0)
    parser.add_argument('--trace-read-bw-gbps', type=float, default=1024.0)
    parser.add_argument('--disagg-link-bw-gbps', type=float, default=900.0)
    parser.add_argument('--overlap', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--hbm-experts-per-gpu', type=int, default=29)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--max-rows', type=int, default=0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    task3.write_csv(path, rows)


def passes(value: int | str, allowed: Iterable[int | str] | None) -> bool:
    return allowed is None or value in set(allowed)


def as_int(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def as_float(row: dict[str, str], key: str, default: float = math.nan) -> float:
    value = row.get(key, '')
    if value in ('', None):
        return default
    return float(value)


def select_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = read_csv(args.input_csv)
    wanted_contexts = None if args.context_lengths is None else set(args.context_lengths)
    wanted_sessions = None if args.sessions is None else set(args.sessions)
    wanted_budgets = None if args.gpu_budgets is None else set(args.gpu_budgets)
    wanted_hbf = None if args.hbf_labels is None else set(args.hbf_labels)
    wanted_workloads = None if args.workloads is None else set(args.workloads)
    selected: list[dict[str, object]] = []
    for row in rows:
        context = as_int(row, 'context_length')
        sessions = as_int(row, 'sessions')
        budget = as_int(row, 'gpu_budget')
        hbf_label = row['hbf_label']
        workload = row.get('workload', 'mixed')
        if as_int(row, 'microbatch_count') != args.microbatch_count:
            continue
        if not passes(context, wanted_contexts):
            continue
        if not passes(sessions, wanted_sessions):
            continue
        if not passes(budget, wanted_budgets):
            continue
        if not passes(hbf_label, wanted_hbf):
            continue
        if not passes(workload, wanted_workloads):
            continue
        selected.append(
            {
                'workload': workload,
                'context_length': context,
                'sessions': sessions,
                'microbatch_count': as_int(row, 'microbatch_count'),
                'sessions_per_microbatch': sessions / args.microbatch_count,
                'gpu_budget': budget,
                'gpus': budget,
                'attention_gpus': as_int(row, 'attention_gpus'),
                'moe_gpus': as_int(row, 'moe_gpus'),
                'hbf_label': hbf_label,
                'hbf_gpus': as_int(row, 'hbf_gpus'),
                'hbm_gpus': as_int(row, 'hbm_gpus'),
                'policy': 'hybrid_hbf_dynamic',
                'source_raw_attention_ms': as_float(row, 'source_raw_attention_ms'),
                'source_raw_moe_ms_mean': as_float(row, 'raw_moe_ms_mean'),
                'source_raw_step_ms_mean': as_float(row, 'raw_step_ms_mean'),
                'source_raw_step_ms_p95': as_float(row, 'raw_step_ms_p95'),
                'source_selection_reason': 'hybrid_raw_min_step_by_hbf_label',
                'decode_tokens': args.decode_tokens,
                'status': 'pending',
                'error': '',
            }
        )
    selected.sort(
        key=lambda r: (
            str(r['workload']),
            int(r['gpu_budget']),
            int(r['context_length']),
            int(r['sessions']),
            str(r['hbf_label']),
        )
    )
    if args.max_rows:
        selected = selected[: args.max_rows]
    if not selected:
        raise RuntimeError('no hybrid rows selected')
    return selected


def build_specs(args: argparse.Namespace, selected: list[dict[str, object]]) -> list[dict[str, object]]:
    args.sessions = tuple(sorted({int(row['sessions']) for row in selected}))
    specs = []
    for row in selected:
        workload = str(row['workload'])
        context = int(row['context_length'])
        sessions = int(row['sessions'])
        attention_gpus = int(row['attention_gpus'])
        moe_gpus = int(row['moe_gpus'])
        gpu_budget = int(row['gpu_budget'])
        hbf_label = str(row['hbf_label'])
        hbf_gpus = int(row['hbf_gpus'])
        run_name = (
            f'{workload}_ctx{context}_S{sessions}_B{gpu_budget}_'
            f'A{attention_gpus}_M{moe_gpus}_H{hbf_label}_hybrid'
        )
        run_base = args.outdir / 'vidur_runs' / run_name
        diagnostics_path = run_base / 'moe_trace_diagnostics.jsonl'
        argv = task3.vidur_argv(
            args,
            run_base,
            diagnostics_path,
            workload,
            context,
            sessions,
            attention_gpus,
            moe_gpus,
            'hybrid_hbf_dynamic',
        )
        argv.extend(
            [
                '--h_b_f_linear_regression_execution_time_predictor_config_moe_trace_hybrid_hbf_gpus',
                str(hbf_gpus),
                '--h_b_f_linear_regression_execution_time_predictor_config_moe_trace_hybrid_hbm_experts_per_gpu',
                str(args.hbm_experts_per_gpu),
                '--h_b_f_linear_regression_execution_time_predictor_config_num_training_job_threads',
                '1',
            ]
        )
        specs.append({'run_name': run_name, 'argv': argv, 'diagnostics_path': str(diagnostics_path), 'row': dict(row)})
    return specs


def metric(row: dict[str, object], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, default)
        if value in ('', None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def write_summary(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    ok = [row for row in rows if row.get('status') == 'ok']
    lines = [
        '# Hybrid HBM/HBF Vidur replay',
        '',
        f'Input CSV: `{args.input_csv}`',
        f'Rows OK: `{len(ok)}/{len(rows)}`.',
        f'SLO: p95 TPOT <= `{args.slo_ms:g}` ms.',
        f'HBM expert slots/GPU/layer: `{args.hbm_experts_per_gpu}`.',
        '',
    ]
    if len(ok) != len(rows):
        lines.extend(['## Invalid rows', '', '| B | Ctx | S | A | M | H | Error |', '|---:|---:|---:|---:|---:|---:|---|'])
        for row in rows:
            if row.get('status') == 'ok':
                continue
            lines.append(
                f"| {row.get('gpu_budget')} | {int(row.get('context_length', 0)) // 1024}K | "
                f"{row.get('sessions')} | {row.get('attention_gpus')} | {row.get('moe_gpus')} | "
                f"{row.get('hbf_label')} | {row.get('error')} |"
            )
        lines.append('')
    lines.extend(['## Max feasible S by budget/context/H', '', '| B | Ctx | H | Max feasible S |', '|---:|---:|---:|---:|'])
    for budget in sorted({int(row.get('gpu_budget', 0)) for row in ok}):
        for context in sorted({int(row.get('context_length', 0)) for row in ok}):
            for hbf_label in sorted({str(row.get('hbf_label')) for row in ok}, key=lambda x: (x == 'all', int(x) if x.isdigit() else 999)):
                feasible = [
                    int(row.get('sessions', 0))
                    for row in ok
                    if int(row.get('gpu_budget', 0)) == budget
                    and int(row.get('context_length', 0)) == context
                    and str(row.get('hbf_label')) == hbf_label
                    and metric(row, 'tpot_p95_ms') <= args.slo_ms
                ]
                if feasible:
                    lines.append(f'| {budget} | {context // 1024}K | {hbf_label} | {max(feasible)} |')
    lines.extend(['', '## S=128 detail', '', '| B | Ctx | A | M | H | Raw step | Vidur p95 TPOT | tok/s | SLO |', '|---:|---:|---:|---:|---:|---:|---:|---:|---|'])
    for row in sorted([r for r in ok if int(r.get('sessions', 0)) == 128], key=lambda r: (int(r.get('gpu_budget', 0)), int(r.get('context_length', 0)), str(r.get('hbf_label')))):
        p95 = metric(row, 'tpot_p95_ms')
        lines.append(
            f"| {row.get('gpu_budget')} | {int(row.get('context_length', 0)) // 1024}K | "
            f"{row.get('attention_gpus')} | {row.get('moe_gpus')} | {row.get('hbf_label')} | "
            f"{metric(row, 'source_raw_step_ms_mean'):.2f} | {p95:.2f} | "
            f"{metric(row, 'throughput_tok_s'):.1f} | {'yes' if p95 <= args.slo_ms else 'no'} |"
        )
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    args = arguments()
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / 'inputs').mkdir(exist_ok=True)
    (args.outdir / 'results').mkdir(exist_ok=True)
    (args.outdir / 'figures').mkdir(exist_ok=True)
    (args.outdir / 'vidur_runs').mkdir(exist_ok=True)
    selected = select_rows(args)
    write_csv(args.outdir / 'inputs' / 'selected_hybrid_vidur_rows.csv', selected)
    specs = build_specs(args, selected)
    rows = task3.execute_run_specs(specs, args.jobs)
    write_csv(args.outdir / 'results' / 'hybrid_vidur_rows.csv', rows)
    task3.plot_budget_results(rows, args.outdir / 'figures', args.slo_ms)
    write_summary(args.outdir / 'results' / 'summary.md', rows, args)
    run_config = {
        'input_csv': str(args.input_csv),
        'trace_root': str(args.trace_root),
        'selected_rows': len(selected),
        'microbatch_count': args.microbatch_count,
        'decode_tokens': args.decode_tokens,
        'hbf_labels': None if args.hbf_labels is None else list(args.hbf_labels),
        'context_lengths': None if args.context_lengths is None else list(args.context_lengths),
        'sessions': None if args.sessions is None else list(args.sessions),
        'gpu_budgets': None if args.gpu_budgets is None else list(args.gpu_budgets),
        'hbm_experts_per_gpu': args.hbm_experts_per_gpu,
        'slo_ms': args.slo_ms,
        'jobs': args.jobs,
    }
    (args.outdir / 'run_config.json').write_text(json.dumps(run_config, indent=2) + '\n', encoding='utf-8')
    print(f'wrote {len(rows)} hybrid Vidur rows under {args.outdir}', flush=True)


if __name__ == '__main__':
    main()
