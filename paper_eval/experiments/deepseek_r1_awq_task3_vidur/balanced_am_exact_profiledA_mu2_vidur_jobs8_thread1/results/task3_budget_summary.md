# Vidur replay from raw-step selected A/M splits

Candidate CSV: `/home/agasthi/vidur/vidur_hbf_hbm_backend/paper_eval/experiments/deepseek_r1_awq_task3_balanced_am/full_mu2_exact_jobs8/results/all_splits.csv`
Selected split CSV: `inputs/selected_profiled_attention_splits.csv`
Allowed attention GPU counts: `[1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]`
Rows selected: `198`; Vidur rows OK: `198/198`.
SLO: p95 TPOT <= `100` ms.
Trace expert read BW: `1024` GB/s; disagg link BW: `900` GB/s.

## Dynamic-vs-static p95 TPOT reduction

Matched cells: `99`.
Average reduction: `12.99%`.
Best reduction: `24.87%`.
Worst reduction: `0.00%`.

## Cost-dynamic max feasible sessions

| Budget | Ctx | Max feasible S |
|---:|---:|---:|
| 8 | 8K | 32 |
| 8 | 16K | 32 |
| 8 | 32K | 32 |
| 8 | 64K | 16 |
| 8 | 128K | 16 |
| 16 | 8K | 128 |
| 16 | 16K | 128 |
| 16 | 32K | 64 |
| 16 | 64K | 64 |
| 16 | 128K | 32 |
| 24 | 8K | 128 |
| 24 | 16K | 128 |
| 24 | 32K | 128 |
| 24 | 64K | 128 |
| 24 | 128K | 64 |
| 32 | 8K | 128 |
| 32 | 16K | 128 |
| 32 | 32K | 128 |
| 32 | 64K | 128 |
| 32 | 128K | 128 |

## S=128 cost-dynamic detail

| B | Ctx | A | M | Raw step ms | Vidur p95 TPOT ms | Throughput tok/s | SLO |
|---:|---:|---:|---:|---:|---:|---:|---|
| 8 | 8K | 1 | 7 | 109.05 | 159.14 | 804.3 | no |
| 8 | 16K | 2 | 6 | 124.78 | 182.31 | 702.1 | no |
| 8 | 32K | 3 | 5 | 147.88 | 217.08 | 589.6 | no |
| 8 | 64K | 6 | 2 | 358.21 | 515.24 | 248.4 | no |
| 16 | 8K | 2 | 14 | 58.25 | 85.68 | 1493.9 | yes |
| 16 | 16K | 3 | 13 | 61.99 | 92.13 | 1389.3 | yes |
| 16 | 32K | 4 | 12 | 66.29 | 101.72 | 1258.4 | no |
| 16 | 64K | 6 | 10 | 77.74 | 124.21 | 1030.5 | no |
| 16 | 128K | 12 | 4 | 183.05 | 264.44 | 484.0 | no |
| 24 | 8K | 3 | 21 | 41.66 | 61.60 | 2077.9 | yes |
| 24 | 16K | 4 | 20 | 44.41 | 65.57 | 1952.0 | yes |
| 24 | 32K | 7 | 17 | 48.91 | 72.61 | 1762.8 | yes |
| 24 | 64K | 10 | 14 | 57.91 | 86.37 | 1482.0 | yes |
| 24 | 128K | 14 | 10 | 77.64 | 114.93 | 1113.7 | no |
| 32 | 8K | 5 | 27 | 33.73 | 48.83 | 2621.6 | yes |
| 32 | 16K | 8 | 24 | 36.98 | 53.26 | 2403.4 | yes |
| 32 | 32K | 10 | 22 | 42.17 | 58.22 | 2198.6 | yes |
| 32 | 64K | 14 | 18 | 48.91 | 69.13 | 1851.7 | yes |
| 32 | 128K | 16 | 16 | 62.40 | 98.57 | 1298.5 | yes |
