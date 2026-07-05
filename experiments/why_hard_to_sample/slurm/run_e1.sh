#!/bin/bash
# E1 Fisher-metric survey (T6) + bijector curvature contribution (T7) launcher.
# Follows the EXACT pattern of run_t3.sh (same shifter image, PYTHONPATH,
# container python). float64-ONLY arm -- this experiment has no float32 arm
# (JAX_ENABLE_X64=1). The script itself runs the HARD chi^2 render-path gate at
# startup and RAISES if the render does not reconcile with log_prob's aux.
#
# Usage:
#   ./run_e1.sh <JOBID> <CUDA_VISIBLE_DEVICES> [OUT_DIR] [SEED] [DATA_DIR]
# e.g.
#   ./run_e1.sh 55396852 0 ./e1_out 1 \
#       /global/u1/l/linusu/GIGALens-Code/.claude/worktrees/why-hard-t0t1/experiments/why_hard_to_sample/systems/sys60
set -euo pipefail

JOBID="${1:?need JOBID as arg 1}"
CUDA="${2:?need CUDA_VISIBLE_DEVICES as arg 2}"
OUT_DIR="${3:-./e1_out}"
SEED="${4:-1}"
DATA_DIR="${5:-/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/why-hard-t0t1/experiments/why_hard_to_sample/systems/sys60}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Fixed inputs (sys60 T0/T1 artifacts).
SAMPLES="$HERE/results_t0t1/sys60/clone_source.npz"
CLONE="$HERE/results_t0t1/sys60/clone.npz"

srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
  bash -lc "
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export CUDA_VISIBLE_DEVICES=$CUDA
    cd $HERE

    echo '=== E1 T6/T7 float64 (Fisher-metric survey) ==='
    export JAX_ENABLE_X64=1
    /usr/bin/python3 e1_fisher_survey.py \
      --data-dir '$DATA_DIR' \
      --samples '$SAMPLES' \
      --clone '$CLONE' \
      --out-dir '$OUT_DIR' \
      --seed $SEED
  "
