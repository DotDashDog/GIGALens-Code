#!/bin/bash
# E1 stage-wise noise attribution — CPU only, inside the jax-0.10 shifter
# container (same image/PYTHONPATH as the D-series CPU runners). Serial.
set -u
IMAGE="docker:ghcr.io/nvidia/jax:jax-2026-04-13"
DBG="/global/homes/l/linusu/GIGALens-Code/experiments/mclmc/debugging"
SCRIPT="$DBG/e1_stage_noise.py"
OUT="$DBG/diagnosis_2026-06/e1"
PYP="/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/homes/l/linusu/GIGALens-Code/src:$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages"
mkdir -p "$OUT"

shifter --image="$IMAGE" bash -c "
  set -u
  export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' XLA_PYTHON_CLIENT_PREALLOCATE=false
  export PYTHONPATH=$PYP
  set -x
  # --- DUMP: all anchor classes x both precisions, n_max=25 ---
  for cls in bootstrap run_a_late run_b_late prior_far ; do
    for prec in float64 float32 ; do
      python -u $SCRIPT dump --anchor-class \$cls --n-max 25 --precision \$prec --K 16 || exit 1
    done
  done
  # --- bootstrap n_max=10 healthy comparator ---
  for prec in float64 float32 ; do
    python -u $SCRIPT dump --anchor-class bootstrap --n-max 10 --precision \$prec --K 16 || exit 1
  done
  # --- RAY: logp along a ray through a frozen run_a anchor (both precisions) ---
  for prec in float64 float32 ; do
    python -u $SCRIPT ray --anchor-class run_a_late --anchor-idx 0 --n-max 25 --precision \$prec || exit 1
    python -u $SCRIPT ray --anchor-class bootstrap  --anchor-idx 0 --n-max 25 --precision \$prec || exit 1
  done
  # --- ANALYZE + plots ---
  python -u $SCRIPT analyze || exit 1
  python -u $SCRIPT ray-plot || exit 1
"
