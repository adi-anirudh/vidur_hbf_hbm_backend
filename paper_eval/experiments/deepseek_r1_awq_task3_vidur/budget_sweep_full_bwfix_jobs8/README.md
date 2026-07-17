# DeepSeek-R1-AWQ Task 3 Vidur evaluation

This folder is isolated from `paper_eval/figures/current`.

Scope:
- no pruning;
- R1-AWQ selected-expert traces used as V3-shaped routing traces;
- Vidur is the outer serving simulator;
- trace-backed MoE timing is enabled through `moe_trace_path`;
- HBM/HBF expert-read bandwidth is 1024 GB/s.

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
