# Task 3 trace-backed Vidur summary

Scope: DeepSeek-R1-AWQ selected-expert traces, no pruning, Vidur decode-only serving.
These are not DeepSeek-V3 measured hardware results.

SLO used for throughput masking: p95 TPOT <= 100 ms.

| Policy | M | S | p95 TPOT (ms) | Throughput (tok/s) | Active experts | Churn |
|---|---:|---:|---:|---:|---:|---:|
| Active-count dyn | 16 | 64 | 52.14 | 1227.4 | 132.9 | 40.3% |
| Cost dyn | 16 | 64 | 52.10 | 1228.5 | 132.9 | 69.0% |
| Static | 16 | 64 | 66.01 | 969.5 | 132.9 | 0.0% |
| Token-count dyn | 16 | 64 | 54.70 | 1169.9 | 132.9 | 52.1% |
