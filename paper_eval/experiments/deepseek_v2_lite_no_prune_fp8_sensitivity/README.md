# DeepSeek-V2-Lite real-trace, no-pruning experiment

This directory is intentionally separate from `paper_eval/figures/current`. Nothing here is a current DeepSeek-V3 paper result.

## What this tests

The replay compares profile-optimized fixed expert ownership against per-microbatch dynamic LPT assignment. Every selected routed expert executes; there is no pruning. The first half of sequences profiles fixed placement and the second half is held out for evaluation.

## What this does not test

- It is not a DeepSeek-V3 trace or a 256-expert result.
- It does not model attention, network communication, scheduler overhead, or end-to-end TBT.
- It does not validate routing at long context; the trace length is 256 tokens.
- V2-Lite's complete routed-expert collection fits in conventional HBM, so this validates scheduling behavior rather than HBF capacity necessity.

## Inputs and assumptions

- Trace: `/home/agasthi/vidur/vidur_hbf_hbm_backend/paper_eval/figures/current/deepseek_trace_run/deepseek_traces.npz`
- Trace SHA-256: `7dcc669787d0a30a2c7ee6b235b834fbdc219b6095991363e3e893e58ca2542d`
- Model revision: `604d5664dddd88a0433dbae533b7fe9472482de0`
- Trace collection dtype: `bfloat16`.
- Expert dimensions: hidden `2048`, intermediate `1408`.
- Modeled expert storage: `1` bytes/parameter; one routed expert is `8.651` MB.
- HBM/HBF analytical bandwidth: `1024` GB/s.
- Analytical compute: `1000` TFLOP/s.
- At this roofline, one expert read costs the same time as approximately `488.3` expert-token computations.
- Routed experts are indivisible jobs; hot-expert splitting is disabled.
- Shared-expert work is included as the same balanced cost for both methods.

## Files

- `run_config.json`: exact replay inputs and assumptions.
- `results/load_balance_summary.csv`: machine-readable results.
- `results/load_balance_summary.md`: compact human-readable table.
- `figures/fig1_routing_characterization.*`: active-expert union and token skew.
- `figures/fig2_dynamic_vs_profile_fixed.*`: mean and p95 modeled MoE-stack reductions.
- `figures/fig3_fetch_vs_compute_balance.*`: slowest-GPU fetch and routed-token reductions.

## Headline within this limited experiment

The largest mean modeled reduction is `18.92%` at microbatch size `8` with `16` MoE GPUs. Interpret this only as a trace-driven scheduling result under the assumptions above.
Under the selected roofline, expert reads dominate expert-token compute. Dynamic assignment continues to reduce routed-token imbalance at large microbatches, but that does not materially reduce modeled latency once nearly every expert is fetched.

## Reproduce

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mplconfig \
  .venv/bin/python paper_eval/scripts/replay_deepseek_v2_no_prune.py
```
