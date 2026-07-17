# Hybrid HBM/HBF Vidur replay

Input CSV: `/home/agasthi/vidur/vidur_hbf_hbm_backend/paper_eval/experiments/deepseek_r1_awq_task3_hybrid_moe_balancers/initial_mu2_profiledA/results/best_by_hbf_label.csv`
Rows OK: `6/6`.
SLO: p95 TPOT <= `100` ms.
HBM expert slots/GPU/layer: `29`.

## Max feasible S by budget/context/H

| B | Ctx | H | Max feasible S |
|---:|---:|---:|---:|
| 16 | 8K | 2 | 64 |
| 16 | 8K | 4 | 64 |
| 16 | 8K | all | 64 |
| 16 | 64K | 2 | 64 |
| 16 | 64K | 4 | 64 |
| 16 | 64K | all | 64 |

## S=128 detail

| B | Ctx | A | M | H | Raw step | Vidur p95 TPOT | tok/s | SLO |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
