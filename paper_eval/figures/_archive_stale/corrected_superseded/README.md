# Corrected μ=2 figures

These figures use `results/corrected_microbatch_sweep.csv`.

- Heterogeneous static and dynamic systems are fixed at two microbatches.
- Colocated baselines select their best swept microbatch count for fairness.
- HBM and HBF bandwidth are both 1024 GB/s.
- Routing/pruning are synthetic and pipeline overlap is an analytical steady-state bound.
- Fixed budgets by workload: `{32: 8, 64: 12, 128: 16, 256: 20, 512: 28, 1024: 40}`.

Both PNG and vector PDF versions are generated.
