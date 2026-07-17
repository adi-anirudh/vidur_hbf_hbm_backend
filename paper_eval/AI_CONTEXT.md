# AI Context

Last updated: 2026-07-17

## Project

FlexEP is a systems-research project evaluating heterogeneous HBM/HBF LLM MoE serving inside the Vidur simulator. The current promising direction is a hybrid MoE tier: attention runs on HBM GPUs, most MoE GPUs remain HBM with capacity-limited static expert shards, and a small number of HBF MoE GPUs hold full expert replicas and dynamically absorb per-microbatch expert-load imbalance.

The paper direction is not “HBF is faster than HBM.” In the current model, HBM and HBF expert-read bandwidth are both `1024 GB/s`. HBF is useful because its capacity enables full expert replication and therefore flexible dynamic expert assignment.

## Current Goal

Prepare a defensible ASPLOS-style FlexEP paper around the no-pruning hybrid HBM/HBF MoE balancer architecture. The strongest current claim is that four HBF full-replica balancers recover most all-HBF dynamic EP performance while avoiding upgrading the entire MoE tier.

## Current Branch

`gpu_clustering`

Worktree base from `git worktree list`: `a336807`

## Last Completed

- Created a complete Claude handoff: `paper_eval/HANDOFF_CLAUDE_2026-07-17.md`.
- Implemented and ran raw trace-backed hybrid HBM/HBF MoE analysis.
- Implemented Vidur support for `moe_trace_policy="hybrid_hbf_dynamic"`.
- Added hybrid config fields:
  - `moe_trace_hybrid_hbf_gpus`
  - `moe_trace_hybrid_hbm_experts_per_gpu`
- Added/used dedicated runner: `paper_eval/scripts/run_hybrid_hbf_vidur.py`.
- Completed full hybrid Vidur sweep:
  - Folder: `paper_eval/experiments/deepseek_r1_awq_task3_vidur/hybrid_hbf_balancers_mu2_vidur_full_B8_16_24_32_S64_128/`
  - Rows: `154/154` OK.
  - Main result: H=4 HBF balancers are close to all-HBF dynamic:
    - Overall avg p95 gap: `4.90%`
    - B=16 avg p95 gap: `3.00%`
    - B=24 avg p95 gap: `3.76%`
    - B=32 avg p95 gap: `2.52%`
  - B=8 is underprovisioned and should not be framed as a success.

## Next Task

1. Add small unit tests for the hybrid assignment path:
   - `limited_static_owners`
   - `assign_hybrid_experts`
   - `MoETraceRouteStore.limited_owners` if feasible without loading large traces.
2. Create one paper-ready PNG figure from:
   - `paper_eval/experiments/deepseek_r1_awq_task3_vidur/hybrid_hbf_balancers_mu2_vidur_full_B8_16_24_32_S64_128/results/hybrid_vidur_rows.csv`
   - Show p95 TPOT or p95 gap vs all-HBF as HBF balancer count varies over `H={1,2,4,all}`.
   - Focus on B={16,24,32}; exclude or clearly mark B=8 as underprovisioned.

## Constraints

- Do not reintroduce pruning into the main story.
- Do not claim true DeepSeek-V3 traces; current traces are R1-AWQ selected-expert traces used as V3/R1-shaped routing traces.
- Do not claim HBF bandwidth advantage; HBM and HBF expert-read bandwidth are both modeled as `1024 GB/s`.
- Do not delete or move:
  - `paper_eval/experiments/deepseek_r1_awq_trace_raw/` (~46G raw trace payload)
  - `paper_eval/experiments/deepseek_r1_awq_no_prune_replay/` (~1.8G normalized replay)
- Use `.venv/bin/python -B` for Python commands.
- Prefer PNG-only figures.
- Keep generated outputs isolated under `paper_eval/experiments/...` until explicitly paper-ready.
- Be careful with git status: many files are untracked or pre-existing modified. Do not blindly commit everything.

## Important Decisions

- The main research claim has pivoted from all-HBF MoE to hybrid HBM/HBF MoE with a small HBF full-replica balancer pool.
- `H=4` is the current strongest design point.
- `B=8` is too constrained in the current S={64,128} grid.
- Current evaluation is no-pruning.
- Vidur is used for serving behavior/p95; raw trace replay is used for faster design-space checks.
- Current expert jobs are whole-expert assignments; hot-expert splitting is not implemented.

## Open Issues

- `paper_eval/` is untracked in git; important scripts/results need to be committed or backed up.
- No unit tests yet for the hybrid assignment path.
- HBM-only/cache-based baseline needs to be strengthened for reviewer defense.
- Paper draft likely still contains stale pruning/older framing and should be updated.
- Some old figure folders are stale; do not reuse figures without checking their source.
- Need a clean paper-ready figure and narrative for hybrid HBF count.

## Important Files

- `paper_eval/HANDOFF_CLAUDE_2026-07-17.md`
- `paper_eval/TASKS.md`
- `vidur/config/config.py`
- `vidur/execution_time_predictor/hbf_execution_time_predictor.py`
- `vidur/execution_time_predictor/moe_trace.py`
- `vidur/memory_backends/moe_flash.py`
- `paper_eval/scripts/hybrid_hbf_balancer_analysis.py`
- `paper_eval/scripts/run_hybrid_hbf_vidur.py`
- `paper_eval/scripts/run_task3_trace_vidur.py`
- `paper_eval/experiments/deepseek_r1_awq_task3_vidur/hybrid_hbf_balancers_mu2_vidur_full_B8_16_24_32_S64_128/results/full_comparison.md`
- `paper_eval/experiments/deepseek_r1_awq_task3_vidur/hybrid_hbf_balancers_mu2_vidur_full_B8_16_24_32_S64_128/results/hybrid_vidur_rows.csv`

## Recommended Model-Switch Prompt

Read `README.md` if present, then `paper_eval/AI_CONTEXT.md`, `paper_eval/TASKS.md`, and `paper_eval/HANDOFF_CLAUDE_2026-07-17.md`. Also inspect the last few commits and `git status`. Confirm your understanding in a few bullet points. Do not start coding until you identify the current task, relevant files, and risks.
