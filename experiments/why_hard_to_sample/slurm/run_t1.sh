#!/bin/bash
# T1 Gaussian-clone launcher.
# Follows the EXACT pattern of experiments/sim_carousel/_h1h2_diag/grun_gpu.sh:
# same shifter image, same PYTHONPATH ordering, JAX_ENABLE_X64=1, container python.
#
# Assumes build_clone.py has already produced CLONE_NPZ (pure-numpy, run on the
# login node or in-container -- no GPU needed for that step).
#
# Usage:
#   ./run_t1.sh <JOBID> <CUDA_VISIBLE_DEVICES> <CLONE_NPZ> [SEED] [OUT_DIR] [DATA_DIR]
# e.g.
#   ./run_t1.sh 55099115 0,1,2,3 ./clone.npz 1 ./t1_out \
#       /global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag
set -euo pipefail

JOBID="${1:?need JOBID as arg 1}"
CUDA="${2:?need CUDA_VISIBLE_DEVICES as arg 2}"
CLONE_NPZ="${3:?need CLONE_NPZ as arg 3}"
SEED="${4:-1}"
OUT_DIR="${5:-./t1_out}"
DATA_DIR="${6:-/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=4 \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
  bash -lc "
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export JAX_ENABLE_X64=1
    export CUDA_VISIBLE_DEVICES=$CUDA
    cd $HERE
    /usr/bin/python3 run_t1_clone.py --clone '$CLONE_NPZ' --data-dir '$DATA_DIR' --out-dir '$OUT_DIR' --seed $SEED
  "
