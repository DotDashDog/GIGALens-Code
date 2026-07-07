#!/bin/bash
# T0 seed-variance calibration launcher.
# Follows the EXACT pattern of experiments/sim_carousel/_h1h2_diag/grun_gpu.sh:
# same shifter image, same PYTHONPATH ordering, JAX_ENABLE_X64=1, container python.
#
# Usage:
#   ./run_t0.sh <JOBID> <CUDA_VISIBLE_DEVICES> [SEEDS] [OUT_DIR] [DATA_DIR]
# e.g.
#   ./run_t0.sh 55099115 0,1,2,3 1,2,3,4 ./t0_out \
#       /global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag
set -euo pipefail

JOBID="${1:?need JOBID as arg 1}"
CUDA="${2:?need CUDA_VISIBLE_DEVICES as arg 2}"
SEEDS="${3:-1,2,3,4}"
OUT_DIR="${4:-./t0_out}"
DATA_DIR="${5:-/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=4 \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
  bash -lc "
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export JAX_ENABLE_X64=1
    export CUDA_VISIBLE_DEVICES=$CUDA
    cd $HERE
    /usr/bin/python3 run_t0_seed_variance.py --data-dir '$DATA_DIR' --seeds '$SEEDS' --out-dir '$OUT_DIR'
  "
