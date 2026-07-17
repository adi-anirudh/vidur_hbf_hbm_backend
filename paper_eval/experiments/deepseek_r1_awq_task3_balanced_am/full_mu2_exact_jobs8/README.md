# Trace-backed raw-step A/M sweep

This folder is separate from the Vidur budget sweep.

Selection objective:

`raw_step_ms = max(raw_attention_ms, raw_moe_ms)` for `mu >= 2`.

Outputs:

- `results/all_splits.csv`: every tested A/M candidate.
- `results/best_by_policy.csv`: selected A/M per context/session/budget/policy.
- `results/best_flexep_cost_dynamic.csv`: cost-dynamic rows only.
- `results/summary.md`: compact human-readable table.

No Vidur throughput or p95 scheduling result is inferred here.
