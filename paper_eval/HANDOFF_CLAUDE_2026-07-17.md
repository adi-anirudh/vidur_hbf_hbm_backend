# FlexEP / Hybrid HBM-HBF MoE Handoff for Claude

Date: 2026-07-17  
Repository: `/home/agasthi/vidur/vidur_hbf_hbm_backend`  
Current branch: `gpu_clustering`  
Current commit/worktree base reported by `git worktree list`: `a336807`

This handoff is written for an AI coding assistant with no prior context.

---

## 1. Project Overview

### Purpose

This project evaluates **FlexEP**, a heterogeneous multi-node LLM serving architecture for Mixture-of-Experts (MoE) models. The current paper direction is:

> FlexEP uses HBM GPUs for attention and a hybrid HBM/HBF MoE tier where most MoE GPUs keep capacity-limited static expert shards, while a small number of high-capacity HBF GPUs hold full expert replicas and dynamically absorb per-microbatch expert-load imbalance.

The key insight is **not** that HBF has higher bandwidth. In this model HBM and HBF expert-read bandwidth are both treated as `1024 GB/s`. HBF is useful because its larger capacity allows full routed-expert replication, which enables dynamic expert assignment/load balancing across MoE GPUs.

### Current Architecture

The architecture currently being evaluated:

1. **Attention subsystem**
   - Runs on HBM GPUs.
   - Context/KV time scales with context length.
   - Attention and MoE are disaggregated.

2. **MoE subsystem**
   - Total MoE GPUs: `M`.
   - `H` of those `M` GPUs are HBF full-expert balancers.
   - Remaining `M-H` GPUs are HBM GPUs with train-profiled, capacity-limited static expert ownership.
   - Per microbatch/layer/decode step:
     - Active expert counts are read from real selected-expert traces.
     - Each expert job can run on its static HBM owner if present, or on any HBF balancer.
     - Greedy LPT-style cost-aware assignment balances `expert fetch + expert compute`.

3. **Simulator**
   - Vidur remains the outer serving simulator.
   - Vidur handles request batches, decode-only serving, queueing, TPOT/TBT, throughput, and p95 latency.
   - The MoE layer timing path is replaced with trace-backed routing/timing when `moe_trace_path` is set.

### Tech Stack

- Python 3 via repo virtualenv: `.venv/bin/python`
- Vidur simulator code under `vidur/`
- NumPy, pandas-like CSV processing via stdlib `csv`
- Matplotlib for PNG figures
- scikit-learn inside Vidur predictor training
- R1-AWQ selected-expert traces, normalized into NumPy shards
- HBF/HBM memory modeling inside `vidur/execution_time_predictor/hbf_execution_time_predictor.py`

### Important Dependencies / Data

Important data/config:

- `configs/hbf_paper_722g.toml`
  - HBF config with large usable HBF capacity.
- `paper_eval/experiments/deepseek_r1_awq_trace_raw/`
  - Raw downloaded trace payload, about `46G`.
  - Do not delete/move casually.
- `paper_eval/experiments/deepseek_r1_awq_no_prune_replay/`
  - Normalized replay folder, about `1.8G`.
  - Contains normalized shards and manifests used by Vidur trace replay.
- `paper_eval/experiments/deepseek_r1_awq_task3_balanced_am/full_mu2_exact_jobs8/results/all_splits.csv`
  - Raw-step A/M candidate sweep input used by later hybrid selection.

Important model assumptions:

- R1-AWQ traces are used as **DeepSeek-V3-shaped selected-expert traces**, not true DeepSeek-V3 traces.
- Trace shape/semantics: 256 routed experts, top-8, 58 MoE layers.
- No pruning in the current main story.
- `microbatch_count = 2`.
- Expert dtype modeled as FP8 for V3/R1-style experts.
- HBM/HBF expert read bandwidth both `1024 GB/s`.
- Disaggregated attention↔MoE link bandwidth in these runs: `900 GB/s`.

---

## 2. Current State

### Completed

#### A. R1 trace pipeline

Raw trace download and normalized replay artifacts exist.

Key folders:

- Raw payload:
  - `paper_eval/experiments/deepseek_r1_awq_trace_raw/`
- Normalized replay:
  - `paper_eval/experiments/deepseek_r1_awq_no_prune_replay/`

The replay folder contains normalized selected-expert routes for no-pruning experiments. These traces drive both raw analytical checks and Vidur runs.

#### B. Raw trace-backed hybrid MoE analysis

Implemented:

- `paper_eval/scripts/hybrid_hbf_balancer_analysis.py`

Output:

- `paper_eval/experiments/deepseek_r1_awq_task3_hybrid_moe_balancers/initial_mu2_profiledA/`

Important result file:

- `results/summary.md`
- `results/best_by_hbf_label.csv`
- `results/paired_comparisons.csv`
- `results/paired_summary.md`

Raw-step result summary:

| HBF label | Meaning | Avg raw step |
|---|---|---:|
| `0` | capacity-aware all-HBM static MoE | 43.38 ms on feasible cells only |
| `1` | 1 HBF balancer | 71.01 ms |
| `2` | 2 HBF balancers | 57.69 ms |
| `4` | 4 HBF balancers | 47.67 ms |
| `all` | all-HBF dynamic upper bound | 50.91 ms |

Important guardrail: the averages above are not all directly comparable because `hbf_label=0` is absent where HBM-only cannot store all routed experts.

Paired raw-step conclusion:

- H=4 recovers most of all-HBF dynamic benefit.
- H=1 is weak.
- H=2 is intermediate.
- H=4 is the clean middle-ground design point.

#### C. Vidur hybrid HBM/HBF policy

Implemented a new trace-backed policy:

- `moe_trace_policy = "hybrid_hbf_dynamic"`

New config fields:

- `moe_trace_hybrid_hbf_gpus`
- `moe_trace_hybrid_hbm_experts_per_gpu`

Key modified/new files:

- `vidur/config/config.py`
- `vidur/execution_time_predictor/hbf_execution_time_predictor.py`
- `vidur/execution_time_predictor/moe_trace.py`
- `paper_eval/scripts/run_task3_trace_vidur.py`
- `paper_eval/scripts/run_hybrid_hbf_vidur.py`

#### D. Focused B=16 hybrid Vidur run

Output:

- `paper_eval/experiments/deepseek_r1_awq_task3_vidur/hybrid_hbf_balancers_mu2_vidur_focused_B16/`

Result:

- 30/30 Vidur rows OK.
- H=4 was only `3.00%` average p95 TPOT slower than all-HBF.
- H=2 was `10.38%` average slower.

#### E. Full hybrid Vidur run

Command that was run:

```bash
.venv/bin/python -B paper_eval/scripts/run_hybrid_hbf_vidur.py \
  --outdir paper_eval/experiments/deepseek_r1_awq_task3_vidur/hybrid_hbf_balancers_mu2_vidur_full_B8_16_24_32_S64_128 \
  --gpu-budgets 8,16,24,32 \
  --context-lengths 8192,16384,32768,65536,131072 \
  --sessions 64,128 \
  --hbf-labels 1,2,4,all \
  --jobs 4
```

Output:

- `paper_eval/experiments/deepseek_r1_awq_task3_vidur/hybrid_hbf_balancers_mu2_vidur_full_B8_16_24_32_S64_128/`

Important files:

- `results/hybrid_vidur_rows.csv`
- `results/paired_vs_all_hbf.csv`
- `results/full_comparison.md`
- `results/summary.md`
- `figures/fig3_best_am_map_mixed_hybrid_hbf_dynamic.png`
- `figures/fig4_best_throughput_slo_mixed_hybrid_hbf_dynamic.png`

Full Vidur result:

- 154/154 rows OK.
- H=4 is the best partial-HBF design point.

Overall p95 TPOT gap vs all-HBF:

| HBF balancers | Cells | Avg p95 gap | Median p95 gap | SLO cells / all-HBF SLO cells |
|---:|---:|---:|---:|---:|
| 1 | 39 | 32.95% | 9.73% | 23/25 |
| 2 | 39 | 13.88% | 6.92% | 23/25 |
| 4 | 37 | 4.90% | 3.39% | 24/25 |

By budget, H=4 gap vs all-HBF:

| Budget | H=4 avg p95 gap | H=4 median p95 gap |
|---:|---:|---:|
| B=8 | 12.66% | 12.25% |
| B=16 | 3.00% | 2.55% |
| B=24 | 3.76% | 3.41% |
| B=32 | 2.52% | 2.72% |

Interpretation:

- B=8 is too underprovisioned for S=64/128 under 100ms p95 SLO; even all-HBF misses SLO.
- B=16 is the interesting tight-budget region.
- B=24 and B=32 are strong: H=4 nearly matches all-HBF.
- H=1 is too weak.
- H=2 is sometimes acceptable but not robust.
- H=4 is the defensible paper design point.

### Currently In Progress

The project is conceptually in a paper-refinement stage:

- The current technical direction has shifted from “all MoE GPUs are HBF” to “small HBF full-replica balancer pool in an otherwise HBM MoE tier.”
- The next work should make the evidence and paper framing rigorous enough for ASPLOS.

No simulator process is currently expected to be running from this handoff.

### Remaining Work

High-priority remaining tasks:

1. Create paper-ready figures for the hybrid story.
2. Strengthen baselines, especially HBM-only/cache-based alternatives.
3. Clean up git status and separate unrelated older changes from current FlexEP changes.
4. Update `paper_eval/PAPER.md` to focus on hybrid HBM/HBF balancers, no pruning.
5. Add tests for the new hybrid assignment path.
6. Verify no old stale figures are being used in the paper.
7. Optionally run one more sweep over `H={0,1,2,4,8,all}` or more S values if needed.

---

## 3. Recent Changes

### Files Modified / Added Recently

Tracked modified files relevant to the current hybrid implementation:

- `vidur/config/config.py`
  - Added hybrid MoE trace config fields.
  - Added `hybrid_hbf_dynamic` to allowed `moe_trace_policy` values.

- `vidur/execution_time_predictor/hbf_execution_time_predictor.py`
  - Imports and uses hybrid expert assignment.
  - Uses `moe_trace_hybrid_hbf_gpus` and `moe_trace_hybrid_hbm_experts_per_gpu`.
  - Emits hybrid diagnostics:
    - `hybrid_hbf_gpus`
    - `hybrid_hbm_gpus`
    - `hbf_fetch_fraction_mean`
    - `hbf_load_fraction_mean`

Untracked but important files/directories:

- `vidur/execution_time_predictor/moe_trace.py`
  - Trace helper for R1 routes.
  - Includes static owners, limited HBM owners, dynamic assignment, and hybrid assignment.

- `paper_eval/scripts/hybrid_hbf_balancer_analysis.py`
  - Raw trace-backed hybrid MoE analysis.

- `paper_eval/scripts/run_hybrid_hbf_vidur.py`
  - Dedicated Vidur runner for hybrid HBM/HBF selected rows.

- `paper_eval/scripts/run_task3_trace_vidur.py`
  - Shared Task-3 Vidur helper. It was patched to summarize hybrid diagnostic fields.

- `paper_eval/experiments/deepseek_r1_awq_task3_hybrid_moe_balancers/initial_mu2_profiledA/`
  - Raw hybrid analysis outputs.

- `paper_eval/experiments/deepseek_r1_awq_task3_vidur/hybrid_hbf_balancers_mu2_vidur_full_B8_16_24_32_S64_128/`
  - Full hybrid Vidur sweep outputs.

Other modified tracked files visible in git status may be older/pre-existing work and should not be blindly bundled without review:

- `.vscode/settings.json`
- `data/profiling/network/h100_dgx/all_reduce.csv`
- `vidur/config/model_config.py`
- `vidur/entities/replica.py`
- `vidur/execution_time_predictor/sklearn_execution_time_predictor.py`
- `vidur/utils/param_counter.py`

Other untracked files/directories visible in git status:

- `configs/hbf_paper_722g.toml`
- `data/profiling/compute/h100/deepseek-ai/DeepSeek-V3/`
- `paper_eval/`
- `verify_moe_flash.py`
- `vidur/memory_backends/moe_flash.py`

### Major Design Decisions

#### Decision 1: Pivot from all-HBF MoE to hybrid HBM/HBF MoE

Reason:

- All-HBF is expensive and easier to attack as unrealistic.
- Hybrid HBM/HBF gives a more practical systems story:
  - Most MoE GPUs stay HBM.
  - A small HBF pool provides full expert coverage.
  - Dynamic assignment uses that capacity to smooth load imbalance.

#### Decision 2: Treat HBM and HBF bandwidth as equal

Reason:

- The user explicitly clarified that HBM and HBF bandwidth should both be modeled as `1 TB/s`.
- Therefore the paper must not claim HBF wins due to bandwidth.
- HBF wins due to capacity/full-replica flexibility.

#### Decision 3: Focus on no-pruning

Reason:

- Pruning adds quality/PPL complications and makes the story broader/weaker.
- The current promising evidence already exists without pruning.
- The ASPLOS submission should first make the no-pruning hybrid systems story airtight.

#### Decision 4: Use R1-AWQ traces as V3-shaped traces, not true V3

Reason:

- True DeepSeek-V3 traces were not available/feasible in this environment.
- R1-AWQ traces provide selected expert IDs with the right shape: 256 experts, top-8, 58 MoE layers.
- The paper must state this carefully and avoid claiming true V3 traces.

#### Decision 5: Use Vidur for serving behavior, not just raw load balance

Reason:

- Raw trace replay proves load balancing, but not end-to-end serving behavior.
- Vidur adds batching, queueing, TPOT/p95, throughput, and SLO feasibility.

---

## 4. Repository Structure

### Important Directories

- `vidur/`
  - Main simulator and execution-time predictor code.

- `vidur/config/`
  - Dataclass configs and CLI-config plumbing.

- `vidur/execution_time_predictor/`
  - Execution-time predictors, including HBF/HBM timing and trace-backed MoE logic.

- `vidur/memory_backends/`
  - HBF/HBM memory backend utilities and MoE flash math.

- `configs/`
  - HBF simulator/config files.

- `data/profiling/`
  - Profiling tables used by Vidur predictors.

- `paper_eval/`
  - Research/evaluation scripts, experiment outputs, paper draft, figures.

- `paper_eval/scripts/`
  - All current research scripts and runners.

- `paper_eval/experiments/`
  - Generated outputs. Some are large and should be treated carefully.

- `paper_eval/figures/current/`
  - Intended paper-ready figure area, but do not assume all figures there are current.

### Key Files

Core simulator modifications:

- `vidur/config/config.py`
- `vidur/execution_time_predictor/hbf_execution_time_predictor.py`
- `vidur/execution_time_predictor/moe_trace.py`
- `vidur/memory_backends/moe_flash.py`

Core evaluation scripts:

- `paper_eval/scripts/hybrid_hbf_balancer_analysis.py`
- `paper_eval/scripts/run_hybrid_hbf_vidur.py`
- `paper_eval/scripts/run_task3_trace_vidur.py`
- `paper_eval/scripts/run_task3_vidur_from_balanced_am.py`
- `paper_eval/scripts/trace_balanced_am_sweep.py`
- `paper_eval/scripts/replay_deepseek_r1_no_prune.py`
- `paper_eval/scripts/normalize_deepseek_r1_traces.py`

Important current result files:

- `paper_eval/experiments/deepseek_r1_awq_task3_vidur/hybrid_hbf_balancers_mu2_vidur_full_B8_16_24_32_S64_128/results/full_comparison.md`
- `paper_eval/experiments/deepseek_r1_awq_task3_vidur/hybrid_hbf_balancers_mu2_vidur_full_B8_16_24_32_S64_128/results/hybrid_vidur_rows.csv`
- `paper_eval/experiments/deepseek_r1_awq_task3_vidur/hybrid_hbf_balancers_mu2_vidur_full_B8_16_24_32_S64_128/results/paired_vs_all_hbf.csv`
- `paper_eval/experiments/deepseek_r1_awq_task3_hybrid_moe_balancers/initial_mu2_profiledA/results/summary.md`
- `paper_eval/experiments/deepseek_r1_awq_task3_hybrid_moe_balancers/initial_mu2_profiledA/results/best_by_hbf_label.csv`

Paper/planning files:

- `paper_eval/PAPER.md`
- `paper_eval/PLAN.md`
- `paper_eval/HANDOFF.md`
- `paper_eval/HANDOFF_CLAUDE_2026-07-17.md`  ← this file

### Entry Points

Raw hybrid analysis:

```bash
.venv/bin/python -B paper_eval/scripts/hybrid_hbf_balancer_analysis.py \
  --jobs 8 \
  --outdir paper_eval/experiments/deepseek_r1_awq_task3_hybrid_moe_balancers/initial_mu2_profiledA
```

Full hybrid Vidur sweep:

```bash
.venv/bin/python -B paper_eval/scripts/run_hybrid_hbf_vidur.py \
  --outdir paper_eval/experiments/deepseek_r1_awq_task3_vidur/hybrid_hbf_balancers_mu2_vidur_full_B8_16_24_32_S64_128 \
  --gpu-budgets 8,16,24,32 \
  --context-lengths 8192,16384,32768,65536,131072 \
  --sessions 64,128 \
  --hbf-labels 1,2,4,all \
  --jobs 4
```

Compile-check key modified files:

```bash
.venv/bin/python -B -m py_compile \
  vidur/config/config.py \
  vidur/execution_time_predictor/moe_trace.py \
  vidur/execution_time_predictor/hbf_execution_time_predictor.py \
  paper_eval/scripts/run_task3_trace_vidur.py \
  paper_eval/scripts/run_hybrid_hbf_vidur.py \
  paper_eval/scripts/hybrid_hbf_balancer_analysis.py
```

---

## 5. Current Branch and Git Status

### Current Branch

`gpu_clustering`

### Worktree

`git worktree list` reports:

```text
/home/agasthi/vidur/vidur_hbf_hbm_backend  a336807 [gpu_clustering]
```

### Stashes

`git stash list` returned no stashes.

### Git Status Snapshot

`git status --short --branch` reported:

```text
## gpu_clustering
 M .vscode/settings.json
 M data/profiling/network/h100_dgx/all_reduce.csv
 M vidur/config/config.py
 M vidur/config/model_config.py
 M vidur/entities/replica.py
 M vidur/execution_time_predictor/hbf_execution_time_predictor.py
 M vidur/execution_time_predictor/sklearn_execution_time_predictor.py
 M vidur/utils/param_counter.py
?? configs/hbf_paper_722g.toml
?? data/profiling/compute/h100/deepseek-ai/DeepSeek-V3/
?? paper_eval/
?? verify_moe_flash.py
?? vidur/execution_time_predictor/moe_trace.py
?? vidur/memory_backends/moe_flash.py
```

Important caution:

- `paper_eval/` is untracked as a directory in this git repo.
- Some tracked modifications predate the hybrid work and should be reviewed before committing.
- Do not blindly commit all modified tracked files unless the user confirms.

### Files That Should Be Committed Together

For the hybrid HBM/HBF feature, commit together:

- `vidur/config/config.py`
- `vidur/execution_time_predictor/hbf_execution_time_predictor.py`
- `vidur/execution_time_predictor/moe_trace.py`
- `vidur/memory_backends/moe_flash.py`
- `configs/hbf_paper_722g.toml`
- `paper_eval/scripts/hybrid_hbf_balancer_analysis.py`
- `paper_eval/scripts/run_hybrid_hbf_vidur.py`
- `paper_eval/scripts/run_task3_trace_vidur.py`
- Relevant result summaries:
  - `paper_eval/experiments/deepseek_r1_awq_task3_hybrid_moe_balancers/initial_mu2_profiledA/results/*.md`
  - `paper_eval/experiments/deepseek_r1_awq_task3_hybrid_moe_balancers/initial_mu2_profiledA/results/*.csv`
  - `paper_eval/experiments/deepseek_r1_awq_task3_vidur/hybrid_hbf_balancers_mu2_vidur_full_B8_16_24_32_S64_128/results/*.md`
  - `paper_eval/experiments/deepseek_r1_awq_task3_vidur/hybrid_hbf_balancers_mu2_vidur_full_B8_16_24_32_S64_128/results/*.csv`
  - PNG figures if selected for paper use.

Do not necessarily commit:

- Large raw trace payloads.
- Large normalized trace shards unless the repo policy permits.
- Old/stale figure folders.
- `.vscode/settings.json` unless intentional.
- Unrelated profiling/model config changes unless confirmed relevant.

---

## 6. Open Issues

### Known Bugs / Risks

1. **Sandbox issue in Codex environment**
   - Some commands fail with:
     - `bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`
   - Workaround used in Codex: rerun read/write commands outside sandbox with approval.
   - Claude may not see this issue depending on environment.

2. **`paper_eval/` is untracked**
   - Many important scripts/results are not git-tracked.
   - Risk of losing work if not committed/backed up.

3. **No unit tests yet for hybrid assignment**
   - `assign_hybrid_experts` needs direct tests:
     - H=all should match all-dynamic assignment behavior closely.
     - H=0/limited behavior should not assign experts to nonexistent devices.
     - Cold experts with no HBM owner should go to HBF.
     - Churn should be computed correctly.

4. **Modeling of HBM static shard capacity is simplified**
   - `moe_trace_hybrid_hbm_experts_per_gpu` default is `29`.
   - This comes from 72GiB HBM, FP8 expert collection size, and shared expert reserve.
   - Verify exact memory math before paper submission.

5. **True V3 traces are not used**
   - Current traces are R1-AWQ selected-expert traces.
   - Must not overclaim DeepSeek-V3 trace validity.

6. **B=8 results are poor for S=64/128**
   - B=8 misses 100ms SLO even for all-HBF in current grid.
   - Do not frame B=8 as a success region.

7. **No pruning in current main claim**
   - Old paper/results may mention pruning.
   - Current recommended paper direction should remove or de-emphasize pruning.

8. **Some figures may be stale**
   - Earlier figure folders exist and may not match the current hybrid story.
   - Do not reuse old figures without checking the generating script and result source.

### Technical Debt

- Result folders are numerous and hard to navigate.
- There is no manifest marking “paper-ready” vs exploratory outputs.
- `run_task3_trace_vidur.py` is both a runner and helper module; new scripts import it.
- Some untracked scripts contain important logic and should be organized/committed.
- Need a clean top-level `README` for current FlexEP experiment flow.

### TODOs

- Add tests for `limited_static_owners` and `assign_hybrid_experts`.
- Add a paper-specific figure generator for hybrid results.
- Update `PAPER.md` to the hybrid HBM/HBF story.
- Create a concise table comparing:
  - all-HBM static
  - hybrid H=1/2/4
  - all-HBF dynamic
- Verify memory capacity assumptions and add them to paper text.
- Add a “do not use” archive or README for stale figure folders.

### Edge Cases Not Yet Handled

- `hbf_label=4` can be absent for some B=8/M=2 rows because H cannot exceed M.
- HBM-only (`hbf_label=0`) is not in the Vidur hybrid full sweep; it exists in raw analysis, not full Vidur.
- The hybrid Vidur run compares partial HBF against all-HBF, not directly against HBM-only static in every region.
- Current scheduling assigns whole expert jobs; it does not split a hot expert across GPUs.
- Current selected rows are based on raw-step A/M/H selection; Vidur is then run on those selections. Vidur itself is not optimizing A/M/H online.

---

## 7. Implementation Details

### Critical Modules / Functions

#### `vidur/execution_time_predictor/moe_trace.py`

Important constants:

- `LAYERS = 58`
- `EXPERTS = 256`
- `TOPK = 8`

Important classes/functions:

- `MoETraceRouteStore`
  - Loads normalized R1-AWQ trace shards.
  - Provides:
    - `routes`
    - `profile`
    - `owners(moe_gpus)`
    - `limited_owners(hbm_gpus, experts_per_gpu)`
    - `counts_for(request_ids, decode_steps, layer)`
    - `selected_for(request_id, decode_step, layer)`

- `static_owners(profile, moe_gpus)`
  - Train-profiled fixed expert ownership.
  - Supports arbitrary `M`, not only divisors of 256.

- `limited_static_owners(profile, hbm_gpus, experts_per_gpu)`
  - Assigns only up to `experts_per_gpu` routed experts per HBM GPU.
  - Unassigned experts have owner `-1` and require HBF in hybrid mode.

- `assign_experts(...)`
  - Existing static/dynamic trace-backed assignment.

- `assign_hybrid_experts(...)`
  - New hybrid assignment.
  - Candidates for each active expert:
    - static HBM owner if present;
    - all HBF balancer devices;
    - HBF only if the expert has no HBM owner.
  - Greedy cost-aware LPT over whole expert jobs.
  - Cost score is `cost_per_expert + cost_per_token * counts[expert]`.

#### `vidur/execution_time_predictor/hbf_execution_time_predictor.py`

Critical method:

- `_moe_trace_layer_time_ms(self, batch)`

Data flow:

1. Extract decode request IDs and decode steps from Vidur batch.
2. For each MoE layer:
   - Get expert counts from `MoETraceRouteStore.counts_for`.
   - Assign experts using:
     - `assign_experts` for `static`, `active_count_dynamic`, `token_count_dynamic`, `cost_dynamic`;
     - `assign_hybrid_experts` for `hybrid_hbf_dynamic`.
   - Convert assignment to per-GPU expert fetches and token loads.
   - Compute routed expert fetch and GEMM time.
   - Add shared expert fetch/compute.
   - Add attention↔MoE activation transfer for disaggregated mode.
   - Apply overlap behavior if configured.
3. Return average MoE layer time to Vidur.
4. Optionally write JSONL diagnostics.

New hybrid diagnostics:

- `hybrid_hbf_gpus`
- `hybrid_hbm_gpus`
- `hbf_fetch_fraction_mean`
- `hbf_load_fraction_mean`

#### `vidur/config/config.py`

New fields in `HBFLinearRegressionExecutionTimePredictorConfig`:

- `moe_trace_hybrid_hbf_gpus: int = 0`
- `moe_trace_hybrid_hbm_experts_per_gpu: int = 29`

Allowed `moe_trace_policy` values now include:

- `static`
- `active_count_dynamic`
- `token_count_dynamic`
- `cost_dynamic`
- `hybrid_hbf_dynamic`

#### `paper_eval/scripts/hybrid_hbf_balancer_analysis.py`

Purpose:

- Fast raw trace-backed design check.
- Selects/evaluates HBF counts before running full Vidur.

Outputs:

- `all_hybrid_candidates.csv`
- `best_by_hbf_label.csv`
- `paired_comparisons.csv`
- `summary.md`
- PNG figures.

#### `paper_eval/scripts/run_hybrid_hbf_vidur.py`

Purpose:

- Consumes `best_by_hbf_label.csv` from raw hybrid analysis.
- Builds Vidur command args for each selected `(B, ctx, S, A, M, H)` row.
- Runs Vidur in parallel.
- Writes:
  - `inputs/selected_hybrid_vidur_rows.csv`
  - `results/hybrid_vidur_rows.csv`
  - `results/summary.md`
  - PNG figures via shared plotting helper.

### Important Interfaces / Data Flow

End-to-end flow:

```text
Raw R1 trace payload
  -> normalize_deepseek_r1_traces.py
  -> deepseek_r1_awq_no_prune_replay/normalized/shards
  -> MoETraceRouteStore
  -> raw hybrid analysis selects A/M/H
  -> run_hybrid_hbf_vidur.py
  -> Vidur simulation
  -> hybrid_vidur_rows.csv + full_comparison.md
```

Important selected-row path:

```text
paper_eval/experiments/deepseek_r1_awq_task3_hybrid_moe_balancers/
  initial_mu2_profiledA/results/best_by_hbf_label.csv
```

This CSV is the input to:

```text
paper_eval/scripts/run_hybrid_hbf_vidur.py
```

### Configuration / Environment Variables

Common command environment:

- Use `.venv/bin/python -B`.
- `PYTHONDONTWRITEBYTECODE=1` is useful but not required.
- `MPLCONFIGDIR=/tmp/mplconfig` can avoid matplotlib cache issues.

Vidur runner important CLI args:

- `--trace-root`
  - Defaults to `paper_eval/experiments/deepseek_r1_awq_no_prune_replay`.
- `--hbf-config`
  - Defaults to `configs/hbf_paper_722g.toml`.
- `--trace-read-bw-gbps`
  - Default `1024.0`.
- `--disagg-link-bw-gbps`
  - Default `900.0`.
- `--hbm-experts-per-gpu`
  - Default `29`.
- `--jobs`
  - Parallel worker count.

The runner appends:

```text
--h_b_f_linear_regression_execution_time_predictor_config_num_training_job_threads 1
```

Reason:

- Prevents nested sklearn/joblib oversubscription/hangs when running many Vidur processes.

---

## 8. Next Recommended Steps

### Ordered Checklist

1. **Do not run new sweeps first. Read the current result summaries.**
   - Start with:
     - `paper_eval/experiments/deepseek_r1_awq_task3_vidur/hybrid_hbf_balancers_mu2_vidur_full_B8_16_24_32_S64_128/results/full_comparison.md`
     - `paper_eval/experiments/deepseek_r1_awq_task3_hybrid_moe_balancers/initial_mu2_profiledA/results/summary.md`

2. **Add tests for the hybrid assignment path.**
   - Add tests for:
     - `limited_static_owners`
     - `assign_hybrid_experts`
     - `MoETraceRouteStore.limited_owners`
   - Keep tests small and synthetic.

3. **Make one paper-ready hybrid figure.**
   - Recommended figure:
     - X-axis: HBF balancer count (`1`, `2`, `4`, `all`)
     - Y-axis: p95 TPOT or p95 gap vs all-HBF
     - Lines/facets: budgets B=16,24,32
   - Exclude or visually separate B=8 because it is underprovisioned.

4. **Update `paper_eval/PAPER.md`.**
   - Remove/de-emphasize pruning.
   - Center the claim on hybrid HBM/HBF balancers.
   - Explicitly state:
     - HBM/HBF bandwidth equal.
     - HBF advantage is capacity/full-expert replication.
     - R1-AWQ traces are V3-shaped, not true V3.

5. **Create defensible baselines.**
   - Needed baselines:
     - HBM-only static/capacity-limited MoE.
     - all-HBF dynamic upper bound.
     - hybrid H={1,2,4}.
   - Reviewer attack to address:
     - “Could HBM just cache the active experts?”
   - Consider a cached-active-set HBM baseline or at least a quantitative explanation.

6. **Clean repository state.**
   - Separate current hybrid work from old/unrelated modified files.
   - Commit or archive important untracked `paper_eval` scripts/results.
   - Do not commit huge trace payloads unless intended.

7. **Prepare ASPLOS story.**
   - The paper should not be “HBF beats HBM.”
   - The paper should be “small HBF full-replica balancer pool recovers most all-HBF dynamic EP performance.”

### Prerequisites

- Ensure `.venv` works.
- Ensure normalized trace folder exists:
  - `paper_eval/experiments/deepseek_r1_awq_no_prune_replay/`
- Ensure HBF config exists:
  - `configs/hbf_paper_722g.toml`
- Ensure sufficient disk for result folders.

### Potential Pitfalls

- Accidentally reusing stale figures from older pruning/long-context experiments.
- Claiming DeepSeek-V3 traces when using R1-AWQ traces.
- Claiming HBF bandwidth advantage when bandwidth is equal in this model.
- Overfitting the story to B=8, which is not a success region.
- Ignoring all-HBM/caching baselines.
- Running massive sweeps before adding tests and cleaning the paper claim.

---

## 9. Context Another AI Should Know

### Coding Conventions

- Prefer local scripts under `paper_eval/scripts/` for research workflows.
- Keep generated outputs isolated under `paper_eval/experiments/...`.
- Use PNG-only figures where possible.
- Use CSV/Markdown summaries for reproducibility.
- Avoid mixing exploratory outputs into `paper_eval/figures/current` until paper-ready.
- Use `.venv/bin/python -B`.
- Keep long-running Vidur sweeps parallel via `--jobs`, but cap sklearn inner threads.

### Existing Patterns to Preserve

- `run_hybrid_hbf_vidur.py` imports helper functions from `run_task3_trace_vidur.py`.
- Vidur configs are passed through CLI arg names derived from dataclass fields.
- Diagnostics are written per run as `moe_trace_diagnostics.jsonl`.
- Summary CSVs are written with stdlib `csv`.
- Result folders contain:
  - `inputs/`
  - `results/`
  - `figures/`
  - `vidur_runs/`
  - `run_config.json`

### Non-Obvious Assumptions

- The current claim is **capacity-driven**, not bandwidth-driven.
- HBF full replica fits:
  - routed+shared expert collection is about `611.38 GiB`;
  - HBF usable capacity is `722 GiB`.
- HBM per-GPU routed expert capacity after shared reserve is modeled as `29` expert slots/layer.
- R1-AWQ traces are accepted only as selected-expert routing traces shaped like DeepSeek-V3/R1, not as quality traces.
- Current evaluation is no-pruning.
- Microbatch count is fixed at 2.

---

## 10. Exact Prompt for Claude

Paste this directly into Claude:

```text
You are taking over a research coding project in /home/agasthi/vidur/vidur_hbf_hbm_backend. Please act as a careful systems-research coding assistant. Do not assume prior context; read paper_eval/HANDOFF_CLAUDE_2026-07-17.md first, then inspect the mentioned files before editing.

Project summary:
This repo evaluates FlexEP, a heterogeneous LLM serving architecture for MoE models. The current promising direction is a hybrid HBM/HBF MoE tier: attention runs on HBM GPUs, most MoE GPUs remain HBM with capacity-limited static expert shards, and a small number of HBF MoE GPUs hold full expert replicas and act as dynamic per-microbatch balancers. HBM and HBF expert-read bandwidth are both modeled as 1 TB/s, so HBF's advantage is capacity/full-expert replication, not bandwidth. Current evaluation uses R1-AWQ selected-expert traces as DeepSeek-V3/R1-shaped routing traces: 256 experts, top-8, 58 MoE layers. No pruning is part of the main current claim.

Current state:
- Hybrid raw trace analysis exists in paper_eval/scripts/hybrid_hbf_balancer_analysis.py.
- Hybrid Vidur runner exists in paper_eval/scripts/run_hybrid_hbf_vidur.py.
- Vidur supports moe_trace_policy=\"hybrid_hbf_dynamic\" through changes in vidur/config/config.py, vidur/execution_time_predictor/moe_trace.py, and vidur/execution_time_predictor/hbf_execution_time_predictor.py.
- Full hybrid Vidur sweep completed 154/154 rows OK under paper_eval/experiments/deepseek_r1_awq_task3_vidur/hybrid_hbf_balancers_mu2_vidur_full_B8_16_24_32_S64_128/.
- Key result: H=4 HBF balancers are close to all-HBF dynamic: average p95 TPOT gap 4.90% overall; by budget B=16: 3.00%, B=24: 3.76%, B=32: 2.52%. B=8 is underprovisioned and should not be framed as a success.

Your next task:
1. First, add small unit tests for the new hybrid assignment path:
   - limited_static_owners
   - assign_hybrid_experts
   - MoETraceRouteStore.limited_owners if feasible without loading large traces
2. Then create one paper-ready PNG figure from the completed full hybrid Vidur CSV:
   - Input: paper_eval/experiments/deepseek_r1_awq_task3_vidur/hybrid_hbf_balancers_mu2_vidur_full_B8_16_24_32_S64_128/results/hybrid_vidur_rows.csv
   - Show p95 TPOT or p95 gap vs all-HBF as HBF balancer count varies over H={1,2,4,all}
   - Focus on B={16,24,32}; either exclude B=8 or mark it as underprovisioned.
   - Save outputs in a clearly named isolated folder under paper_eval/experiments/...; do not overwrite older paper figures.

Important constraints:
- Preserve the existing architecture and coding style.
- Do not reintroduce pruning into the main story.
- Do not claim true DeepSeek-V3 traces; call them R1-AWQ selected-expert traces used as V3/R1-shaped routing traces.
- Do not claim HBF bandwidth advantage; bandwidth is equal in the model.
- Do not delete or move the raw trace payload or normalized replay folders.
- Be careful with git status: many files are untracked or pre-existing modified; do not blindly commit everything.
- Use .venv/bin/python -B for commands.
- Prefer PNG-only figures.

Please start by reading the handoff file and summarizing your understanding before making changes.
```

---

## AI_CONTEXT

```json
{
  "project_summary": "FlexEP evaluates hybrid HBM/HBF LLM MoE serving. Current claim: a small HBF full-expert balancer pool in an otherwise HBM MoE tier recovers most all-HBF dynamic EP performance. HBM/HBF expert bandwidth are equal; HBF advantage is capacity/full replication.",
  "architecture": "Attention on HBM GPUs; disaggregated MoE tier with M GPUs total; H HBF GPUs hold full expert replicas; M-H HBM GPUs hold capacity-limited static expert shards; per-microbatch trace-backed cost-aware assignment balances expert fetch+compute; Vidur provides serving/queueing/p95.",
  "important_files": [
    "vidur/config/config.py",
    "vidur/execution_time_predictor/hbf_execution_time_predictor.py",
    "vidur/execution_time_predictor/moe_trace.py",
    "vidur/memory_backends/moe_flash.py",
    "paper_eval/scripts/hybrid_hbf_balancer_analysis.py",
    "paper_eval/scripts/run_hybrid_hbf_vidur.py",
    "paper_eval/scripts/run_task3_trace_vidur.py",
    "paper_eval/experiments/deepseek_r1_awq_task3_vidur/hybrid_hbf_balancers_mu2_vidur_full_B8_16_24_32_S64_128/results/full_comparison.md"
  ],
  "recent_changes": [
    "Added moe_trace_policy=hybrid_hbf_dynamic.",
    "Added config fields moe_trace_hybrid_hbf_gpus and moe_trace_hybrid_hbm_experts_per_gpu.",
    "Added limited_static_owners and assign_hybrid_experts for hybrid HBM/HBF assignment.",
    "Added raw hybrid analysis and dedicated hybrid Vidur runner.",
    "Completed full hybrid Vidur sweep: 154/154 OK; H=4 avg p95 gap vs all-HBF 4.90% overall, 3.00% for B=16, 3.76% for B=24, 2.52% for B=32."
  ],
  "pending_tasks": [
    "Add unit tests for hybrid assignment helpers.",
    "Create paper-ready hybrid HBF-count figure from full Vidur CSV.",
    "Update PAPER.md to focus on no-pruning hybrid HBM/HBF balancer story.",
    "Strengthen HBM-only/cache baseline.",
    "Clean git/untracked files and separate unrelated changes."
  ],
  "known_issues": [
    "paper_eval is untracked; many important scripts/results are not committed.",
    "R1-AWQ traces are not true DeepSeek-V3 traces; wording must be careful.",
    "B=8 is underprovisioned and not a success region.",
    "No unit tests yet for hybrid assignment.",
    "Some old figures/results are stale."
  ],
  "coding_conventions": [
    "Use .venv/bin/python -B.",
    "Keep outputs isolated under paper_eval/experiments.",
    "Prefer CSV plus Markdown summaries and PNG-only figures.",
    "Do not mix exploratory outputs into paper_eval/figures/current until paper-ready.",
    "Preserve Vidur CLI/dataclass config pattern."
  ],
  "constraints": [
    "No pruning in current main story.",
    "HBM and HBF bandwidth both 1024 GB/s.",
    "Do not delete/move raw or normalized trace folders.",
    "Do not overclaim DeepSeek-V3 traces.",
    "Do not blindly commit unrelated modified files."
  ],
  "next_task": "Read this handoff, add tests for limited_static_owners/assign_hybrid_experts, then generate one paper-ready PNG figure showing p95 TPOT or p95 gap vs all-HBF across H={1,2,4,all}, focusing on B={16,24,32}."
}
```
