# Hybrid HBM/HBF MoE Balancer Check

This is a raw trace-backed check, not a Vidur queueing run.

## Model

- R1 no-pruning selected-expert traces, 58 MoE layers, 256 routed experts, top-8.
- HBM and HBF expert bandwidth are both `1024 GB/s`.
- One routed expert collection across all layers is `2.38 GiB`.
- Shared expert collection reserve is `2.38 GiB`.
- HBM usable capacity is `72.0 GiB`, so each HBM MoE GPU can hold `29` routed expert slots/layer after the shared reserve.
- HBF usable capacity is `722.0 GiB`; a full routed+shared replica is `611.38 GiB`.

## Average best-cell summary

| hbf_label | best_cells | avg_raw_step_ms | avg_raw_moe_ms | slo_ok_cells | avg_hbf_load_fraction | avg_hbf_expert_fraction | avg_recovered_gain_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 74 | 43.38 | 42.40 | 74 | 0.00 | 0.00 | 0.00 |
| 1 | 99 | 71.01 | 69.62 | 87 | 0.16 | 0.16 | 52.78 |
| 2 | 99 | 57.69 | 55.99 | 88 | 0.27 | 0.27 | 69.50 |
| 4 | 97 | 47.67 | 45.54 | 91 | 0.45 | 0.45 | 80.37 |
| all | 99 | 50.91 | 48.14 | 92 | 1.00 | 1.00 | 100.00 |

## Budget summary

| budget | hbf_label | cells | avg_step_ms | slo_ok_cells | avg_A | avg_M |
| --- | --- | --- | --- | --- | --- | --- |
| 8 | 1 | 24 | 149.46 | 13 | 1.88 | 6.12 |
| 8 | 2 | 24 | 104.20 | 14 | 2.00 | 6.00 |
| 8 | 4 | 22 | 70.25 | 17 | 1.86 | 6.14 |
| 8 | all | 24 | 85.18 | 18 | 2.29 | 5.71 |
| 16 | 0 | 24 | 50.29 | 24 | 3.54 | 12.46 |
| 16 | 1 | 25 | 61.61 | 24 | 4.64 | 11.36 |
| 16 | 2 | 25 | 53.94 | 24 | 5.04 | 10.96 |
| 16 | 4 | 25 | 49.82 | 24 | 5.48 | 10.52 |
| 16 | all | 25 | 48.28 | 24 | 6.08 | 9.92 |
| 24 | 0 | 25 | 42.36 | 25 | 6.92 | 17.08 |
| 24 | 1 | 25 | 40.15 | 25 | 8.52 | 15.48 |
| 24 | 2 | 25 | 39.23 | 25 | 9.08 | 14.92 |
| 24 | 4 | 25 | 38.50 | 25 | 9.92 | 14.08 |
| 24 | all | 25 | 37.58 | 25 | 10.96 | 13.04 |
| 32 | 0 | 25 | 37.77 | 25 | 10.32 | 21.68 |
| 32 | 1 | 25 | 35.95 | 25 | 11.92 | 20.08 |
| 32 | 2 | 25 | 35.27 | 25 | 12.48 | 19.52 |
| 32 | 4 | 25 | 34.83 | 25 | 13.16 | 18.84 |
| 32 | all | 25 | 33.99 | 25 | 14.12 | 17.88 |

## Interpretation guardrails

- `hbf_label=0` is capacity-aware all-HBM static MoE. It is absent where the HBM side cannot store all routed experts.
- `hbf_label=1/2/4` keeps total MoE GPUs fixed at the chosen M and upgrades only that many MoE GPUs to HBF full replicas.
- `hbf_label=all` is the all-HBF dynamic upper bound.
- `recovered_all_hbf_gain_pct` is only defined where a feasible HBM-only baseline exists for the same context/S/budget cell.
