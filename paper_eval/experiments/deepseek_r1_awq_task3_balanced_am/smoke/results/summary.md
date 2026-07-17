# Trace-backed raw-step A/M optimization

Objective: choose A/M by minimizing `max(raw_attention_ms, raw_moe_ms)` for `mu >= 2`.
This is an A/M resource-balance sweep, not a Vidur throughput run.

| Policy | Budget | Ctx | S | GPUs | A | M | Attn ms | MoE ms | Step ms | SLO feasible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cost_dynamic | 8 | 8K | 8 | 8 | 2 | 6 | 27.55 | 27.76 | 27.76 | yes |
| static | 8 | 8K | 8 | 8 | 1 | 7 | 28.67 | 35.90 | 35.90 | yes |
