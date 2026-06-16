#!/bin/bash
# E1c compilation-seam probe — CPU only, login node, inside the shifter image.
# Mirrors run_e1_cpu.sh. Serial.
set -u
IMAGE="docker:ghcr.io/nvidia/jax:jax-2026-04-13"
DBG="/global/homes/l/linusu/GIGALens-Code/experiments/mclmc/debugging"
SCRIPT="$DBG/e1c_compilation_seam.py"
OUT="$DBG/diagnosis_2026-06/e1c"
PYP="/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/homes/l/linusu/GIGALens-Code/src:$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages"
mkdir -p "$OUT"
LOG="$OUT/run_cpu.log"

shifter --image="$IMAGE" bash -c "
  set -u
  export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' XLA_PYTHON_CLIENT_PREALLOCATE=false
  export PYTHONPATH=$PYP
  set -x
  python -u $SCRIPT --device cpu --precision float32 --n-max 25 --anchor-class both --K 8 || exit 1
" 2>&1 | tee "$LOG"
echo "=== e1c cpu exit ${PIPESTATUS[0]} ===" | tee -a "$LOG"
