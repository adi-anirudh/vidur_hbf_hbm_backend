# Corrected context × workload sweep

> Fresh analytical generator; legacy `master_sweep.md` was not read or overwritten.
> Routing and pruning are synthetic. These are design-space results, not measured hardware results.

## Scope and corrections

- Contexts: `8K, 16K, 32K, 64K, 128K`.
- Concurrent decode sessions: `32, 64, 128, 256, 512, 1024`.
- Retained gate mass: `1.00, 0.99, 0.98, 0.95, 0.90`; every token's top-1 is protected.
- HBM/HBF bandwidth: `1024/1024 GB/s`; link: `900 GB/s`.
- Static EP uses balanced fixed integer placement, never independent random ownership.
- Dynamic EP uses integer weighted LPT assignment after routing/pruning.
- MoE latency is the sum of the per-layer straggler times.
- Attention uses official DeepSeek-V3 absorbed-MLA geometry and a memory/compute roofline.
- Three dense FFNs, embeddings, LM head, shared experts, and runtime capacity are explicit.
- Heterogeneous systems use steady-state `max(T_attention, T_MoE)` overlap.

## Capacity ledger

| Component | Size | Placement |
|---|---:|---|
| MLA weights, 61 layers | 11.41 GB | attention/colocated GPU |
| Dense FFNs, 3 layers | 1.19 GB | attention/colocated GPU |
| Embedding + LM head | 1.85 GB | attention/colocated GPU |
| Routed experts, 58 layers | 653.91 GB | sharded static / fully replicated dynamic |
| Shared expert, 58 layers | 2.55 GB | replicated on MoE GPUs |
| Runtime reserve | 4.00 GB | each pool GPU |

HBM attention capacity in sessions per GPU:

| Context | Sessions/GPU |
|---:|---:|
| 8K | 186 |
| 16K | 93 |
| 32K | 46 |
| 64K | 23 |
| 128K | 11 |

## Deterministic fixed budgets

Budgets are the smallest multiple of four at which the **unpruned balanced-static heterogeneous baseline** reaches 100 ms at 32K. They are selected without looking at dynamic/pruned results.

| Sessions | Budget |
|---:|---:|
| 32 | 8 |
| 64 | 8 |
| 128 | 12 |
| 256 | 16 |
| 512 | 20 |
| 1024 | 32 |

## Best TBT within the fixed budget

Cells show `TBT ms (A:M)` for heterogeneous systems and `TBT ms (total G)` for colocated systems.

| Ctx | S | Budget | HBM coloc | HBF coloc | Static c=1.00 | Dynamic c=1.00 | Static c=.98 | Dynamic c=.98 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 32 | 8 | — | 73.0 (8G) | 65.8 (1A:7M) | 56.4 (1A:7M) | 53.6 (1A:7M) | 43.8 (1A:7M) |
| 8K | 64 | 8 | — | 87.7 (8G) | 81.8 (1A:7M) | 74.5 (1A:7M) | 67.9 (1A:7M) | 58.9 (1A:7M) |
| 8K | 128 | 12 | — | 72.2 (12G) | 61.6 (1A:11M) | 57.2 (1A:11M) | 54.3 (1A:11M) | 49.2 (1A:11M) |
| 8K | 256 | 16 | 61.2 (16G) | 61.2 (16G) | 52.0 (2A:14M) | 49.4 (2A:14M) | 49.2 (2A:14M) | 44.4 (3A:13M) |
| 8K | 512 | 20 | 57.3 (20G) | 57.3 (20G) | 49.2 (4A:16M) | 47.7 (5A:15M) | 47.2 (5A:15M) | 42.6 (5A:15M) |
| 8K | 1024 | 32 | 47.2 (32G) | 47.2 (32G) | 39.0 (12A:20M) | 38.4 (12A:20M) | 38.9 (12A:20M) | 37.4 (12A:20M) |
| 16K | 32 | 8 | — | 74.1 (8G) | 65.8 (1A:7M) | 56.4 (1A:7M) | 53.6 (1A:7M) | 43.8 (1A:7M) |
| 16K | 64 | 8 | — | 90.0 (8G) | 81.8 (1A:7M) | 74.5 (1A:7M) | 67.9 (1A:7M) | 58.9 (1A:7M) |
| 16K | 128 | 12 | — | 75.3 (12G) | 66.4 (2A:10M) | 61.7 (2A:10M) | 58.2 (2A:10M) | 50.8 (2A:10M) |
| 16K | 256 | 16 | 65.7 (16G) | 65.7 (16G) | 58.6 (4A:12M) | 56.0 (4A:12M) | 53.7 (4A:12M) | 49.2 (4A:12M) |
| 16K | 512 | 20 | 64.6 (20G) | 64.6 (20G) | 55.4 (7A:13M) | 54.9 (7A:13M) | 54.8 (7A:13M) | 51.7 (8A:12M) |
| 16K | 1024 | 32 | 56.2 (32G) | 56.2 (32G) | 49.2 (16A:16M) | 49.2 (16A:16M) | 49.2 (16A:16M) | 47.5 (17A:15M) |
| 32K | 32 | 8 | — | 76.4 (8G) | 65.8 (1A:7M) | 56.4 (1A:7M) | 53.6 (1A:7M) | 49.2 (1A:7M) |
| 32K | 64 | 8 | — | 94.5 (8G) | 93.0 (2A:6M) | 85.8 (2A:6M) | 76.8 (2A:6M) | 67.7 (2A:6M) |
| 32K | 128 | 12 | — | 81.4 (12G) | 72.7 (3A:9M) | 68.0 (3A:9M) | 63.1 (3A:9M) | 61.6 (3A:9M) |
| 32K | 256 | 16 | — | 74.7 (16G) | 68.7 (6A:10M) | 65.9 (6A:10M) | 62.7 (6A:10M) | 61.6 (6A:10M) |
| 32K | 512 | 20 | — | 79.2 (20G) | 86.2 (12A:8M) | 85.6 (12A:8M) | 81.0 (12A:8M) | 75.1 (12A:8M) |
| 32K | 1024 | 32 | — | 74.2 (32G) | 81.9 (23A:9M) | 81.2 (23A:9M) | 79.1 (23A:9M) | 74.2 (23A:9M) |
| 64K | 32 | 8 | — | 80.9 (8G) | 73.7 (2A:6M) | 65.0 (2A:6M) | 59.2 (2A:6M) | 50.3 (2A:6M) |
| 64K | 64 | 8 | — | 103.5 (8G) | 109.5 (3A:5M) | 102.0 (3A:5M) | 89.5 (3A:5M) | 80.3 (3A:5M) |
| 64K | 128 | 12 | — | 93.8 (12G) | 104.7 (6A:6M) | 99.6 (6A:6M) | 88.9 (6A:6M) | 81.2 (6A:6M) |
| 64K | 256 | 16 | — | 92.7 (16G) | 161.4 (12A:4M) | 158.2 (12A:4M) | 141.0 (12A:4M) | 133.5 (12A:4M) |
| 64K | 512 | 20 | — | 108.5 (20G) | — | — | — | — |
| 64K | 1024 | 32 | — | 110.1 (32G) | — | — | — | — |
| 128K | 32 | 8 | — | 89.9 (8G) | 86.0 (3A:5M) | 77.2 (3A:5M) | 69.0 (3A:5M) | 62.7 (3A:5M) |
| 128K | 64 | 8 | — | 121.5 (8G) | 255.4 (6A:2M) | 249.2 (6A:2M) | 202.0 (6A:2M) | 194.9 (6A:2M) |
| 128K | 128 | 12 | — | 118.5 (12G) | — | — | — | — |
| 128K | 256 | 16 | — | 128.7 (16G) | — | — | — | — |
| 128K | 512 | 20 | — | 166.9 (20G) | — | — | — | — |
| 128K | 1024 | 32 | — | 182.1 (32G) | — | — | — | — |

## Minimum GPUs at 100 ms

| Ctx | S | HBM coloc | HBF coloc | Static c=1.00 | Dynamic c=1.00 | Static c=.98 | Dynamic c=.98 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 32 | 14 | 6 | 6 | 5 | 5 | 4 |
| 8K | 64 | 14 | 7 | 7 | 7 | 6 | 5 |
| 8K | 128 | 14 | 8 | 8 | 8 | 7 | 6 |
| 8K | 256 | 15 | 9 | 9 | 9 | 8 | 8 |
| 8K | 512 | 16 | 10 | 10 | 10 | 10 | 9 |
| 8K | 1024 | 19 | 13 | 14 | 14 | 13 | 13 |
| 16K | 32 | 14 | 6 | 6 | 5 | 5 | 4 |
| 16K | 64 | 14 | 8 | 7 | 7 | 6 | 5 |
| 16K | 128 | 15 | 9 | 9 | 8 | 8 | 7 |
| 16K | 256 | 16 | 10 | 10 | 10 | 9 | 9 |
| 16K | 512 | 19 | 12 | 13 | 13 | 13 | 12 |
| 16K | 1024 | 25 | 16 | 20 | 20 | 19 | 19 |
| 32K | 32 | 14 | 6 | 6 | 5 | 5 | 4 |
| 32K | 64 | 15 | 8 | 8 | 8 | 7 | 6 |
| 32K | 128 | 16 | 10 | 10 | 9 | 9 | 8 |
| 32K | 256 | 19 | 12 | 13 | 13 | 12 | 12 |
| 32K | 512 | 25 | 16 | 19 | 19 | 19 | 18 |
| 32K | 1024 | 37 | 23 | 31 | 31 | 30 | 30 |
| 64K | 32 | 15 | 7 | 7 | 6 | 6 | 5 |
| 64K | 64 | 16 | 9 | 9 | 9 | 8 | 7 |
| 64K | 128 | 19 | 12 | 13 | 12 | 12 | 11 |
| 64K | 256 | 26 | 16 | 19 | 19 | 18 | 18 |
| 64K | 512 | 37 | 23 | 30 | 30 | 30 | 29 |
| 64K | 1024 | 61 | 37 | 53 | 53 | 52 | 52 |
| 128K | 32 | 16 | 8 | 8 | 7 | 7 | 6 |
| 128K | 64 | 21 | 11 | 12 | 12 | 11 | 10 |
| 128K | 128 | 26 | 15 | 19 | 18 | 18 | 17 |
| 128K | 256 | 37 | 24 | 31 | 31 | 30 | 30 |
| 128K | 512 | 64 | 37 | 54 | 54 | 54 | 53 |
| 128K | 1024 | 114 | 64 | 102 | 102 | 101 | 101 |

## Minimum GPUs at 200 ms

| Ctx | S | HBM coloc | HBF coloc | Static c=1.00 | Dynamic c=1.00 | Static c=.98 | Dynamic c=.98 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8K | 32 | 14 | 3 | 3 | 3 | 3 | 3 |
| 8K | 64 | 14 | 3 | 4 | 4 | 4 | 3 |
| 8K | 128 | 14 | 4 | 5 | 4 | 4 | 4 |
| 8K | 256 | 15 | 4 | 6 | 6 | 5 | 5 |
| 8K | 512 | 16 | 5 | 7 | 7 | 6 | 6 |
| 8K | 1024 | 19 | 6 | 10 | 10 | 10 | 10 |
| 16K | 32 | 14 | 3 | 3 | 3 | 3 | 3 |
| 16K | 64 | 14 | 4 | 4 | 4 | 4 | 3 |
| 16K | 128 | 15 | 4 | 6 | 5 | 5 | 5 |
| 16K | 256 | 16 | 5 | 7 | 7 | 6 | 6 |
| 16K | 512 | 19 | 6 | 10 | 10 | 9 | 9 |
| 16K | 1024 | 25 | 7 | 16 | 16 | 16 | 16 |
| 32K | 32 | 14 | 3 | 3 | 3 | 3 | 3 |
| 32K | 64 | 15 | 4 | 5 | 5 | 5 | 4 |
| 32K | 128 | 16 | 5 | 7 | 6 | 6 | 6 |
| 32K | 256 | 19 | 6 | 10 | 10 | 9 | 9 |
| 32K | 512 | 25 | 7 | 16 | 16 | 15 | 15 |
| 32K | 1024 | 37 | 11 | 27 | 27 | 27 | 27 |
| 64K | 32 | 15 | 3 | 4 | 4 | 4 | 4 |
| 64K | 64 | 16 | 4 | 6 | 6 | 6 | 5 |
| 64K | 128 | 19 | 5 | 10 | 9 | 9 | 9 |
| 64K | 256 | 26 | 7 | 16 | 16 | 15 | 15 |
| 64K | 512 | 37 | 11 | 27 | 27 | 26 | 26 |
| 64K | 1024 | 61 | 17 | 49 | 49 | 49 | 49 |
| 128K | 32 | 16 | 3 | 5 | 5 | 5 | 5 |
| 128K | 64 | 21 | 5 | 9 | 9 | 9 | 8 |
| 128K | 128 | 26 | 7 | 16 | 15 | 15 | 15 |
| 128K | 256 | 37 | 10 | 28 | 28 | 27 | 27 |
| 128K | 512 | 64 | 17 | 51 | 51 | 50 | 50 |
| 128K | 1024 | 114 | 30 | 98 | 98 | 98 | 98 |

## Pruning sensitivity at the fixed budgets

Improvement compares dynamic against balanced static at the same retained-mass target and GPU budget.

| Retained mass | Mean kept experts | Median TBT improvement | Max TBT improvement | Cases releasing ≥1 MoE GPU |
|---:|---:|---:|---:|---:|
| 1.00 | 213.1 | 5.71% | 14.23% | 1/24 |
| 0.99 | 185.5 | 8.81% | 16.82% | 3/24 |
| 0.98 | 175.9 | 8.96% | 18.24% | 3/24 |
| 0.95 | 159.4 | 9.32% | 19.81% | 9/24 |
| 0.90 | 144.0 | 11.85% | 21.41% | 5/24 |

## Limitations

- Synthetic structured routing and synthetic softmax gate scores; no real DeepSeek trace yet.
- Retained gate mass is not a quality guarantee; perplexity and task accuracy are unmeasured.
- Dynamic scheduling overhead is currently zero, so dynamic rows are an achievable analytical bound.
- Peak bandwidth/FLOPs rooflines omit kernel efficiency, contention, topology, and control overhead.
- Full attention/MoE overlap is a steady-state pipeline assumption.
- Both memory tiers intentionally use the user-selected equal 1024 GB/s design point.
