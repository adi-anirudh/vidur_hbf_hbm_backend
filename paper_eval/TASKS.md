# TASKS

Last updated: 2026-07-17

This file is the source of truth for current project tasks.

## In Progress

- Prepare the hybrid HBM/HBF FlexEP story for a possible ASPLOS submission.
  - Current focus: no-pruning hybrid HBF balancers, not pruning or all-HBF-only.

## Next

1. Add unit tests for the hybrid assignment implementation.
   - Test `limited_static_owners` capacity behavior.
   - Test `assign_hybrid_experts` candidate-device behavior.
   - Test that experts without HBM owner are assigned to HBF devices when HBF exists.
   - Test that H=all behaves like fully dynamic assignment at a high level.

2. Create one paper-ready hybrid figure.
   - Input: `paper_eval/experiments/deepseek_r1_awq_task3_vidur/hybrid_hbf_balancers_mu2_vidur_full_B8_16_24_32_S64_128/results/hybrid_vidur_rows.csv`
   - Show p95 TPOT or p95 gap vs all-HBF for `H={1,2,4,all}`.
   - Focus on `B={16,24,32}`.
   - Save under a new isolated folder in `paper_eval/experiments/...`.
   - PNG only.

3. Update `paper_eval/PAPER.md`.
   - Reframe around hybrid HBM/HBF balancers.
   - Remove or de-emphasize pruning.
   - State bandwidth equality clearly.
   - State R1-AWQ trace limitation clearly.

4. Strengthen baselines.
   - Add or analyze HBM-only static/capacity-limited baseline.
   - Address likely reviewer question: “Could HBM just cache active experts?”

5. Clean repository state.
   - Identify which modified tracked files belong to FlexEP hybrid work.
   - Identify which untracked outputs should be committed, ignored, or archived.
   - Do not commit raw trace payloads unless explicitly intended.

## Backlog

- Add a clean README for the current experiment flow.
- Add a manifest distinguishing paper-ready figures from exploratory figures.
- Run optional additional hybrid sweep with `H={0,1,2,4,8,all}` if needed.
- Consider smaller S values for B=8 if a budget-sensitivity story needs low-budget data.
- Consider network-bandwidth sensitivity after the core story is stable.
- Consider hot-expert splitting only after whole-expert hybrid assignment is fully evaluated.
- Validate memory-capacity math for HBM expert slots per GPU (`29`) before paper submission.
- Update `paper_eval/PLAN.md` if the research plan changes.

## Done

- Downloaded/organized DeepSeek-R1-AWQ selected-expert trace artifacts.
- Normalized R1 traces for no-pruning replay.
- Implemented trace-backed raw A/M selector.
- Corrected attention context-length scaling in the A/M selection flow.
- Implemented raw hybrid HBM/HBF MoE balancer analysis.
- Implemented Vidur `hybrid_hbf_dynamic` MoE trace policy.
- Added hybrid Vidur runner.
- Ran focused B=16 hybrid Vidur validation: 30/30 OK.
- Ran full hybrid Vidur sweep B={8,16,24,32}, ctx={8K,16K,32K,64K,128K}, S={64,128}, H={1,2,4,all}: 154/154 OK.
- Created detailed handoff: `paper_eval/HANDOFF_CLAUDE_2026-07-17.md`.
- Created persistent project-state files: `paper_eval/AI_CONTEXT.md` and `paper_eval/TASKS.md`.

## Known Risks

- `paper_eval/` is untracked; important work can be lost if not committed/backed up.
- Some tracked modified files may be unrelated older changes.
- B=8 is not a success region in current full sweep.
- Current traces are R1-AWQ selected-expert traces, not true DeepSeek-V3 traces.
- No tests yet for the hybrid scheduling helpers.
- Old figures may be stale.

## Session Shutdown Checklist

Before ending a future session:

1. Update `paper_eval/AI_CONTEXT.md`.
2. Update `paper_eval/TASKS.md`.
3. Commit coherent changes if the user approves and the worktree is clean enough.
4. Write a concise handoff summarizing what changed and why.
5. Mention any commands still running or outputs still being generated.
