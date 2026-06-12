#!/usr/bin/env bash
# Turnkey compute profiling for the HBF paper sweep.
#
# Run ON a cloud instance whose GPU is the target SKU. It profiles MLP +
# attention for all 8 workload models and drops the CSVs into
#   data/profiling/compute/<DEVICE>/<org>/<model>/{mlp,attention}.csv
# After this lands in the repo, run_full_sweep.py picks the cells up (re-run it).
#
# Prereqs (see README.md): sarathi-serve `vidur` branch installed in the active
# venv, this repo pip-installed (`pip install -e .`), and you are running on the
# physical target GPU (the SKU label is YOUR responsibility — Vidur measures
# whatever GPU is present and you label the folder).
#
# Usage:
#   ./profile_device.sh a100          # 1 GPU (slower but sufficient)
#   ./profile_device.sh h100 4        # use 4 GPUs to parallelize
set -euo pipefail

DEVICE="${1:?usage: profile_device.sh <device-label> [num_gpus]}"
NUM_GPUS="${2:-1}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

MODELS=(
  "microsoft/phi-2"
  "mistralai/Mistral-7B-v0.1"
  "meta-llama/Meta-Llama-3-8B"
  "mistralai/Mixtral-8x7B-v0.1"
  "deepseek-ai/deepseek-llm-67b-chat"
  "meta-llama/Meta-Llama-3-70B"
  "Qwen/Qwen-72B"
  "Qwen/Qwen2-72B"
)

echo ">> Profiling MLP for ${#MODELS[@]} models on device label '$DEVICE' (num_gpus=$NUM_GPUS)"
python vidur/profiling/mlp/main.py --models "${MODELS[@]}" --num_gpus "$NUM_GPUS"
echo ">> Profiling attention"
python vidur/profiling/attention/main.py --models "${MODELS[@]}" --num_gpus "$NUM_GPUS"

# Copy newest outputs into the device-labelled profiling tree.
MLP_DIR="$(ls -dt profiling_outputs/mlp/*/ | head -1)"
ATT_DIR="$(ls -dt profiling_outputs/attention/*/ | head -1)"
echo ">> Latest MLP run: $MLP_DIR"
echo ">> Latest attention run: $ATT_DIR"

for m in "${MODELS[@]}"; do
  dest="data/profiling/compute/$DEVICE/$m"
  mkdir -p "$dest"
  cp "$MLP_DIR/$m/mlp.csv"        "$dest/mlp.csv"
  cp "$ATT_DIR/$m/attention.csv" "$dest/attention.csv"
  echo "   placed $dest/{mlp,attention}.csv"
done

echo ">> Done. Commit data/profiling/compute/$DEVICE and re-run run_full_sweep.py."
