#!/bin/bash
# T8 (breathing transects) + T9 (xi vs lambda1) launcher.
# Follows the EXACT pattern of run_e1.sh / run_t3.sh (same shifter image,
# PYTHONPATH, container python). float64-ONLY (JAX_ENABLE_X64=1) -- neither
# experiment has a float32 arm. Both scripts run the HARD chi^2 render-path gate
# at startup (reusing E1's verified render path) and RAISE if it does not
# reconcile with log_prob's aux. T8 then T9 run SEQUENTIALLY in the same shell.
#
# Usage:
#   ./run_t8_t9.sh <JOBID> <CUDA_VISIBLE_DEVICES> [OUT_ROOT] [SEED] [DATA_DIR]
# e.g.
#   ./run_t8_t9.sh 55401964 0 ./results_t0t1/sys60 1 \
#       /global/u1/l/linusu/GIGALens-Code/.claude/worktrees/why-hard-t0t1/experiments/why_hard_to_sample/systems/sys60
set -euo pipefail

JOBID="${1:?need JOBID as arg 1}"
CUDA="${2:?need CUDA_VISIBLE_DEVICES as arg 2}"
OUT_ROOT="${3:-./results_t0t1/sys60}"
SEED="${4:-1}"
DATA_DIR="${5:-/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/why-hard-t0t1/experiments/why_hard_to_sample/systems/sys60}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Fixed inputs (sys60 T0/T1/T3 artifacts + reference MCLMC run).
SAMPLES="$HERE/results_t0t1/sys60/clone_source.npz"
CLONE="$HERE/results_t0t1/sys60/clone.npz"
T3_JSON="$HERE/results_t0t1/sys60/t3/float64/t3_results_float64.json"
RUN_DIR="/global/homes/l/linusu/GIGALens-Code/results/testsys60/mclmc"

T8_OUT="$OUT_ROOT/t8"
T9_OUT="$OUT_ROOT/t9"

srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
  bash -lc "
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export CUDA_VISIBLE_DEVICES=$CUDA
    export JAX_ENABLE_X64=1
    cd $HERE

    echo '=== T8 breathing transects (float64) ==='
    /usr/bin/python3 t8_breathing_transects.py \
      --data-dir '$DATA_DIR' \
      --t3-json '$T3_JSON' \
      --clone '$CLONE' \
      --samples '$SAMPLES' \
      --out-dir '$T8_OUT' \
      --seed $SEED

    echo '=== T9 xi vs lambda1 (float64) ==='
    /usr/bin/python3 t9_xi_lambda.py \
      --data-dir '$DATA_DIR' \
      --run-dir '$RUN_DIR' \
      --out-dir '$T9_OUT' \
      --seed $SEED
  "
