#!/bin/bash
# T16 (carousel B1) -- T0 seed-variance for BOTH carousel arms.
# Runs run_t0_seed_variance.py on carousel_min_old and carousel_min_new with the
# STANDARD config (8 chains x 2000 burnin / 2000 results) for seeds 1,2,3.
# Follows run_t0.sh / run_t14b.sh pattern EXACTLY (shifter image, PYTHONPATH,
# JAX_ENABLE_X64=1, container python, srun --overlap against an existing JOBID).
#
# NOTE: the carousel B1 pre-registration sets 3 seeds; run_t0_seed_variance.py's
# default guard is N>=4 (the original sys60/vela T0 spec), so we pass the
# behavior-preserving optional flag --min-seeds 3 (sys60/vela default is
# unchanged). This is an explicit, logged design choice, not a silent override.
#
# Usage:
#   ./run_t16.sh <JOBID> <CUDA_VISIBLE_DEVICES> [SEEDS]
# e.g.
#   ./run_t16.sh 55420000 0,1,2,3 1,2,3
set -euo pipefail

JOBID="${1:?need JOBID as arg 1}"
CUDA="${2:?need CUDA_VISIBLE_DEVICES as arg 2}"
SEEDS="${3:-1,2,3}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OLD_DATA_DIR="$HERE/systems/carousel_min_old"
NEW_DATA_DIR="$HERE/systems/carousel_min_new"

srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=4 --time=180 \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export CUDA_VISIBLE_DEVICES=$CUDA
    export JAX_ENABLE_X64=1
    cd $HERE

    echo '=== T16 B1: T0 seed-variance OLD arm (carousel_min_old) ==='
    /usr/bin/python3 run_t0_seed_variance.py \
      --data-dir '$OLD_DATA_DIR' --seeds '$SEEDS' --min-seeds 3 \
      --out-dir ./results_carousel/old/t0

    echo '=== T16 B1: T0 seed-variance NEW arm (carousel_min_new) ==='
    /usr/bin/python3 run_t0_seed_variance.py \
      --data-dir '$NEW_DATA_DIR' --seeds '$SEEDS' --min-seeds 3 \
      --out-dir ./results_carousel/new/t0
  "
