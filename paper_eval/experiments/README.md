# Isolated paper experiments

This directory contains exploratory or trace-driven experiments that are **not** part of `paper_eval/figures/current` unless they are explicitly promoted later.

## DeepSeek-R1-AWQ normalized trace — no pruning

- [Format and reproduction guide](deepseek_r1_awq_no_prune_replay/README.md)
- [Normalization summary](deepseek_r1_awq_no_prune_replay/summary.json)
- [Request manifest](deepseek_r1_awq_no_prune_replay/manifests/requests.csv)
- [Discarded final-entry manifest](deepseek_r1_awq_no_prune_replay/manifests/discarded_entries.csv)
- [Shard checksum manifest](deepseek_r1_awq_no_prune_replay/manifests/shards.csv)
- [Held-out replay results](deepseek_r1_awq_no_prune_replay/results/load_balance_summary.md)
- [Decisive static-versus-dynamic figure](deepseek_r1_awq_no_prune_replay/figures/fig1_p95_slowest_gpu_work.png)
- [Workload sensitivity figure](deepseek_r1_awq_no_prune_replay/figures/fig2_p95_reduction_by_workload.png)

Scope: 24,062 DeepSeek-R1-AWQ request traces normalized into 94 compact
`uint8` shards for held-out, no-pruning per-microbatch replay.

## DeepSeek-V2-Lite real trace — no pruning

- [Experiment guide](deepseek_v2_lite_no_prune/README.md)
- [Human-readable results](deepseek_v2_lite_no_prune/results/load_balance_summary.md)
- [Machine-readable results](deepseek_v2_lite_no_prune/results/load_balance_summary.csv)
- [Routing characterization](deepseek_v2_lite_no_prune/figures/fig1_routing_characterization.png)
- [Dynamic versus profile-fixed scheduling](deepseek_v2_lite_no_prune/figures/fig2_dynamic_vs_profile_fixed.png)
- [Fetch versus routed-token balancing](deepseek_v2_lite_no_prune/figures/fig3_fetch_vs_compute_balance.png)

Scope: DeepSeek-V2-Lite, WikiText-103, 256-token traces, held-out replay, no pruning. This is a MoE scheduling experiment rather than a DeepSeek-V3 or end-to-end TBT result.

## DeepSeek-V2-Lite trace — FP8 expert-storage sensitivity

- [Experiment guide](deepseek_v2_lite_no_prune_fp8_sensitivity/README.md)
- [Human-readable results](deepseek_v2_lite_no_prune_fp8_sensitivity/results/load_balance_summary.md)
- [Dynamic versus profile-fixed scheduling](deepseek_v2_lite_no_prune_fp8_sensitivity/figures/fig2_dynamic_vs_profile_fixed.png)
- [Fetch versus routed-token balancing](deepseek_v2_lite_no_prune_fp8_sensitivity/figures/fig3_fetch_vs_compute_balance.png)

Scope: the same measured BF16 routing decisions replayed with FP8 expert storage (`1 byte/parameter`). This is a storage/timing sensitivity, not routing collected from an FP8 V2-Lite execution.
