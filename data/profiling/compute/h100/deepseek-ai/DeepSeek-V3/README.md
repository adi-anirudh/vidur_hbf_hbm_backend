# DeepSeek-V3 profiling tables (derived, not measured)

These CSVs carry the **measured timings of meta-llama/Llama-2-70b-hf on H100**
with DeepSeek-V3's geometry stamped into the filter columns (`n_embd=7168`,
`n_head/n_q_head=224`, `n_kv_head=9`, `n_expanded_embd=18432`) so that
`SklearnExecutionTimePredictor._load_compute_df/_load_attention_df` accept the
rows for the `DeepSeekV3ModelConfig`.

This is the same approximation the repo already uses via symlinks (Mixtral →
Meta-Llama-3-8B, deepseek-llm-67b → Llama-2-70b-hf), made explicit because the
geometry filters would reject a plain symlink. Timings are slightly
conservative (70B is D=8192 vs V3's 7168).

Intended use: HBF/MoE decode-only studies, where the only surviving sklearn
terms are the small compute-bound ops (norms, rope, add, activations) —
attention decode, KV-cache save, and the MLP/MoE path are replaced by the
HBF predictor's cycle-accurate/analytical models.

Regenerate with: stamp the columns above onto
`data/profiling/compute/h100/meta-llama/Llama-2-70b-hf/{mlp,attention}.csv`.

## TP=16 rows (derived 2026-07-08)

TP=16 rows in `mlp.csv`/`attention.csv` are the TP=8 rows with all
`time_stats.*` halved (first-order Megatron scaling; optimistic for small
kernels). TP=16 rows in `data/profiling/network/h100_dgx/all_reduce.csv` are
the TP=8 rows × 1.0714 (ring-collective 2(n−1)/n factor), assuming a flat
16-GPU NVLink domain — a real 2-node DGX crossing would be slower.
