#!/usr/bin/env bash
# One-shot setup + profiling for a fresh RunPod / Lambda / Vast GPU pod.
#
# Pick a pod with a CUDA 12.1 (or 12.x) base image and the target GPU:
#   A100 80GB | H100 80GB | H200 141GB | A40 48GB   (one GPU is enough)
#
# Usage on the pod:
#   export REPO_URL=https://github.com/adi-anirudh/vidur_hbf_hbm_backend   # YOUR fork, with the model_config fixes pushed
#   export REPO_BRANCH=main                       # branch that has the corrected configs
#   export DEVICE=a100                            # label matching the GPU you rented
#   bash setup_pod.sh
#
# Output: data/profiling/compute/$DEVICE/<org>/<model>/{mlp,attention}.csv
# and a tarball  profiling_$DEVICE.tar.gz  to download / push back.
set -euo pipefail

: "${DEVICE:?set DEVICE=a100|h100|h200|a40 (label for the GPU you rented)}"
REPO_URL="${REPO_URL:?set REPO_URL to your fork}"
REPO_BRANCH="${REPO_BRANCH:-main}"
WORK="${WORK:-/workspace}"
cd "$WORK"

echo "==> GPU check"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --- Python 3.10 env via micromamba (self-contained; image-agnostic) ---
if ! command -v micromamba >/dev/null 2>&1; then
  echo "==> installing micromamba"
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -C /usr/local bin/micromamba
fi
export MAMBA_ROOT_PREFIX="$WORK/mamba"
eval "$(micromamba shell hook -s bash)"
micromamba create -y -p "$WORK/env" python=3.10
micromamba activate "$WORK/env"

# --- sarathi-serve (vidur branch): torch 2.3 + flashinfer cu121 ---
echo "==> installing sarathi-serve (vidur branch)"
git clone https://github.com/microsoft/sarathi-serve.git
cd sarathi-serve && git checkout vidur
pip install -e . --extra-index-url https://flashinfer.ai/whl/cu121/torch2.3/
cd "$WORK"

# --- this repo (your fork WITH the corrected model_config dims) ---
echo "==> installing simulator repo"
git clone -b "$REPO_BRANCH" "$REPO_URL" vidur_repo
cd vidur_repo
pip install -e .

# --- profile all 8 models on this GPU, default ranges (<=4096 is sufficient) ---
echo "==> profiling on label '$DEVICE'"
bash profiling_kit/profile_device.sh "$DEVICE" 1

# --- package for return ---
tar -czf "$WORK/profiling_$DEVICE.tar.gz" -C "$WORK/vidur_repo" "data/profiling/compute/$DEVICE"
echo "==> DONE. Download $WORK/profiling_$DEVICE.tar.gz, or push the branch:"
echo "    cd vidur_repo && git checkout -b profiling-$DEVICE && git add data/profiling/compute/$DEVICE && git commit -m 'profiling: $DEVICE' && git push origin profiling-$DEVICE"
