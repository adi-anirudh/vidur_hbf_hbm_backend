# Vidur from balanced A/M CSV

This folder is isolated from earlier Task 3 runs.

Flow:
1. Read raw-step candidates from `/home/agasthi/vidur/vidur_hbf_hbm_backend/paper_eval/experiments/deepseek_r1_awq_task3_balanced_am/full_mu2_exact_jobs8/results/all_splits.csv`.
2. Filter to attention GPU counts that local Vidur profiling supports.
3. Select the min raw-step A/M per workload/context/S/budget/policy.
4. Run Vidur exactly on those selected splits.

Outputs:
- `inputs/selected_profiled_attention_splits.csv`: exact A/M rows used as Vidur input.
- `results/task3_vidur_selected_profiledA.csv`: measured Vidur rows.
- `results/task3_budget_best_by_policy.csv`: same measured rows, named for compatibility with plotting.
- `results/task3_budget_summary.md`: compact summary.
- `figures/*.png`: PNG-only figures.
