#!/bin/bash
# Run the MCLMC_238 repro inside the gigalens shifter container on a GPU node.
# Usage: bash run_repro_on_gpu.sh <tag> [extra-env-pairs...]
# Example: bash run_repro_on_gpu.sh autotune-off XLA_FLAGS=--xla_gpu_autotune_level=0
set -u
TAG="${1:-default}"
shift || true
EXTRA_ENV=""
for kv in "$@"; do EXTRA_ENV="$EXTRA_ENV $kv"; done

LOG="/global/homes/l/linusu/GIGALens-Code/alternate_inference/repro_${TAG}.log"
DMESG_BEFORE="/tmp/dmesg_before_${TAG}.txt"
DMESG_AFTER="/tmp/dmesg_after_${TAG}.txt"

echo "=== TAG=$TAG host=$(hostname) date=$(date -Iseconds)" | tee "$LOG"
echo "=== EXTRA_ENV=$EXTRA_ENV" | tee -a "$LOG"

# Run inside shifter, same as the notebook kernel
shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2025-06-07 \
    bash -c "
        set -u
        export PYTHONPATH=\"\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages\"
        export REPRO_TAG='$TAG'
        $EXTRA_ENV
        echo '--- env relevant ---'
        env | egrep '^(JAX_|XLA_|CUDA_|TF_|REPRO_)' | sort
        echo '--- nvidia-smi (before) ---'
        nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv,noheader
        echo '--- starting python ---'
        python -u /global/homes/l/linusu/GIGALens-Code/alternate_inference/repro_mclmc_238.py
        ec=\$?
        echo \"--- python exit code: \$ec ---\"
        echo '--- nvidia-smi (after) ---'
        nvidia-smi --query-gpu=index,memory.free,memory.used --format=csv,noheader
        exit \$ec
    " 2>&1 | tee -a "$LOG"

EC=${PIPESTATUS[0]}
echo "=== exit code: $EC" | tee -a "$LOG"
echo "=== signal name (if killed): $(kill -l $((EC-128)) 2>/dev/null || echo n/a)" | tee -a "$LOG"
exit $EC
