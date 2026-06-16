#!/bin/bash
# E1b gradient audit — CPU only, inside the jax-0.10 shifter container
# (same image/PYTHONPATH as run_e1_cpu.sh). Serial.
set -u
IMAGE="docker:ghcr.io/nvidia/jax:jax-2026-04-13"
DBG="/global/homes/l/linusu/GIGALens-Code/experiments/mclmc/debugging"
SCRIPT="$DBG/e1b_grad_audit.py"
OUT="$DBG/diagnosis_2026-06/e1b"
PYP="/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/homes/l/linusu/GIGALens-Code/src:$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages"
mkdir -p "$OUT"

shifter --image="$IMAGE" bash -c "
  set -u
  export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' XLA_PYTHON_CLIENT_PREALLOCATE=false
  export PYTHONPATH=$PYP
  set -x
  for cls in bootstrap run_a_late run_b_late prior_far ; do
    for prec in float64 float32 ; do
      python -u $SCRIPT dump --anchor-class \$cls --n-max 25 --precision \$prec --K 8 || exit 1
    done
  done
  for prec in float64 float32 ; do
    python -u $SCRIPT dump --anchor-class bootstrap --n-max 10 --precision \$prec --K 8 || exit 1
  done
  python -u $SCRIPT analyze || exit 1
"
