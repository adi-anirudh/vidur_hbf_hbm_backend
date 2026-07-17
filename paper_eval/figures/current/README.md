# Current corrected figures

These figures use `results/corrected_microbatch_sweep.csv`.

- Heterogeneous static and dynamic systems are fixed at two microbatches.
- Colocated baselines select their best swept microbatch count for fairness.
- HBM and HBF bandwidth are both 1024 GB/s.
- Routing/pruning are synthetic and pipeline overlap is an analytical steady-state bound.
- Fixed budgets by workload: `{32: 8, 64: 12, 128: 16, 256: 20, 512: 28, 1024: 40}`.

## Figure numbering

1. Synthetic router-mass concentration.
2. Synthetic reduction from active-before-pruning to retained-after-pruning experts versus workload.
3. Microbatch-count sensitivity.
4. Dynamic EP versus balanced static EP at matched c=1.00 and c=.98.
5. Common-budget GPU–TBT Pareto curves at 128K context, with 100 ms crossings.
6. End-to-end TBT advantage-regime zoom, without pruning and with c=.98 pruning.
7. Minimum GPUs at 100 ms for S=32, 64, 128, and 256, without and with pruning.

Unnumbered diagnostic: achieved TBT at method-specific minimum-GPU points; not a fixed-budget comparison.

Both PNG and vector PDF versions are generated.
