# HBF paper sweep — cloud profiling kit

The full sweep is 8 models × 4 GPUs (A40/A100/H100/H200) × 8 batch × 9 ctx,
Dense + Sparse. Compute traces (`mlp.csv`, `attention.csv`) must be measured on
**each physical target GPU** — Vidur's profiler runs real kernels; there is no
analytical mode, and cross-architecture scaling is not faithful (15–37% error).
Only ~7/32 (model,device) cells currently have real traces; this kit produces
the rest. Once traces land, the sweep runs on CPU.

## What you need per SKU
A cloud instance whose GPU **is** the target SKU. Suggested:
- A100 80GB SXM, H100 80GB SXM, H200 141GB — widely available.
- A40 48GB — available on several clouds.
- ≥1 GPU is enough (TP profiling needs only 1 GPU, just slower). 70B/72B models
  need a GPU with enough memory to instantiate one reference layer; 80GB is safe.

## Setup (once per instance)
```bash
# 1. sarathi-serve (the profiler's kernel backend), branch `vidur`
git clone https://github.com/microsoft/sarathi-serve && cd sarathi-serve
git checkout vidur
python -m venv env && source env/bin/activate
pip install -e .            # follow sarathi-serve README for CUDA-matched torch/flashinfer
cd ..
# 2. this repo, same venv
git clone <this repo> && cd vidur_hbf_hbm_backend
pip install -e .
```

## Run (per SKU)
```bash
# on the A100 box:
./profiling_kit/profile_device.sh a100
# on the H100 box:
./profiling_kit/profile_device.sh h100
# A40, H200 similarly. Use a second arg for multi-GPU: ./profile_device.sh h100 4
```
This writes `data/profiling/compute/<device>/<org>/<model>/{mlp,attention}.csv`.
Commit those, pull them onto this machine, and re-run:
```bash
PYTHONPATH=. .venv/bin/python run_full_sweep.py --workers 8
```
`run_full_sweep.py` is resumable — it only runs cells whose traces newly appeared
(NO_TRACE → OK), skips INFEASIBLE (capacity) and already-done points.

## ⚠️ Fix before spending cloud hours (model-config fidelity)
The fork's `vidur/config/model_config.py` has values that will bake error into
the traces if left as-is:

1. **`vocab_size` looks wrong** for several models (affects embedding / lm_head
   GEMM, which is per decode token):
   - Mistral-7B-v0.1: should be 32000 (config has 128256)
   - Mixtral-8x7B-v0.1: should be 32000 (config has 128256)
   - deepseek-llm-67b-chat: should be 102400 (config has 32768)
   - Qwen2-72B: should be 152064 (config has 32768)
   Verify each against the HF `config.json` and correct before profiling.

2. **Mixtral-8x7B is MoE** but Vidur's reference `GPTModel` profiles a **dense**
   MLP — it will undercount Mixtral's MLP/expert cost. Decide how to handle
   (profile dense and note the caveat, or extend the reference model for MoE)
   before treating Mixtral results as paper-grade.

## Cost: use DEFAULT ranges (do NOT profile to 1M ctx)
The HBF predictor overrides attn_decode + kv_save with the flash/HBM models, and
decode-only runs skip prefill. Verified: phi-2 runs correctly at 1M context using
attention data that only reaches 4096. So profile with the **default
`max_seq_len=4096` / `max_model_len=4096`** (what `profile_device.sh` uses) — this
keeps per-model profiling to minutes on one GPU and avoids the huge KV allocations
(and OOM risk) that profiling long contexts on a 70B/72B would cause. Do not pass
large `--max_seq_len`; the long-context decode physics comes from the analytical
flash model, not the profiled table.

## Notes
- Compute profiling is **device-only** (network/TP-collective cost is separate;
  add `network` profiling if you later model TP communication — see docs/profiling.md).
- The sweep's TP per point is chosen by `sweep_capacity.py` (min TP in {1,2,4,8}
  s.t. weights+KV ≤ TP×(HBM+HBF)); profiling at num_gpus=1 still covers all TP
  because per-worker shapes are derived analytically.
