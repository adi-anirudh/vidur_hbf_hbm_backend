# Task 3 trace-backed Vidur summary

Scope: DeepSeek-R1-AWQ selected-expert traces, no pruning, Vidur decode-only serving.
These are not DeepSeek-V3 measured hardware results.

SLO used for throughput masking: p95 TPOT <= 100 ms.

| Policy | M | S | p95 TPOT (ms) | Throughput (tok/s) | Active experts | Churn |
|---|---:|---:|---:|---:|---:|---:|
| Cost dyn | 16 | 128 | 72.46 | 1766.4 | 199.4 | 71.3% |
