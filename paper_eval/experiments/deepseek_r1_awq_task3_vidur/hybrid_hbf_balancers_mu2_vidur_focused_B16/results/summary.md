# Hybrid HBM/HBF Vidur replay

Input CSV: `/home/agasthi/vidur/vidur_hbf_hbm_backend/paper_eval/experiments/deepseek_r1_awq_task3_hybrid_moe_balancers/initial_mu2_profiledA/results/best_by_hbf_label.csv`
Rows OK: `30/30`.
SLO: p95 TPOT <= `100` ms.
HBM expert slots/GPU/layer: `29`.

## Max feasible S by budget/context/H

| B | Ctx | H | Max feasible S |
|---:|---:|---:|---:|
| 16 | 8K | 2 | 64 |
| 16 | 8K | 4 | 128 |
| 16 | 8K | all | 128 |
| 16 | 16K | 2 | 64 |
| 16 | 16K | 4 | 64 |
| 16 | 16K | all | 128 |
| 16 | 32K | 2 | 64 |
| 16 | 32K | 4 | 64 |
| 16 | 32K | all | 64 |
| 16 | 64K | 2 | 64 |
| 16 | 64K | 4 | 64 |
| 16 | 64K | all | 64 |

## S=128 detail

| B | Ctx | A | M | H | Raw step | Vidur p95 TPOT | tok/s | SLO |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 16 | 8K | 1 | 15 | 2 | 62.40 | 101.26 | 1264.0 | no |
| 16 | 8K | 2 | 14 | 4 | 61.29 | 88.01 | 1454.4 | yes |
| 16 | 8K | 2 | 14 | all | 58.25 | 85.68 | 1493.9 | yes |
| 16 | 16K | 2 | 14 | 2 | 64.20 | 101.03 | 1267.0 | no |
| 16 | 16K | 2 | 14 | 4 | 62.40 | 101.02 | 1267.1 | no |
| 16 | 16K | 3 | 13 | all | 61.99 | 92.13 | 1389.3 | yes |
| 16 | 32K | 4 | 12 | 2 | 71.21 | 104.20 | 1228.4 | no |
| 16 | 32K | 4 | 12 | 4 | 69.30 | 103.09 | 1241.6 | no |
| 16 | 32K | 4 | 12 | all | 66.29 | 101.72 | 1258.4 | no |
| 16 | 64K | 6 | 10 | 2 | 84.14 | 125.72 | 1018.1 | no |
| 16 | 64K | 6 | 10 | 4 | 83.30 | 126.19 | 1014.3 | no |
| 16 | 64K | 6 | 10 | all | 77.74 | 124.21 | 1030.5 | no |
| 16 | 128K | 12 | 4 | 2 | 265.05 | 393.87 | 325.0 | no |
| 16 | 128K | 12 | 4 | 4 | 183.05 | 264.44 | 484.0 | no |
| 16 | 128K | 12 | 4 | all | 183.05 | 264.44 | 484.0 | no |
