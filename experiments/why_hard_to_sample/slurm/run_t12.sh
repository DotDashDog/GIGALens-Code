#!/bin/bash
# T12 flank-crossing / pixelation check launcher.
# Follows the EXACT pattern of run_t3.sh / run_t10_t11.sh (same shifter image,
# PYTHONPATH, container python). float64-ONLY (JAX_ENABLE_X64=1) -- no float32 arm.
# The script runs the HARD chi^2 render-path gate at startup for EACH model build
# (supersample=2 AND the supersample=4 control) and RAISES if either fails to
# reconcile against that build's own log_prob aux.
#
# Part 1 (dial scans, ss2) + Part 2 (ss4 control) + Part 3 (census cross-check) all
# run in a single invocation. Cost: ~4x97 Jacobians + ~2100 renders; minutes on 1 GPU.
#
# Usage:
#   ./run_t12.sh <JOBID> <CUDA_VISIBLE_DEVICES> [OUT_DIR] [SEED] [DATA_DIR]
# e.g.
#   ./run_t12.sh 55406521 0 ./results_t0t1/sys60/t12 0 \
#       /global/u1/l/linusu/GIGALens-Code/.claude/worktrees/why-hard-t0t1/experiments/why_hard_to_sample/systems/sys60
set -euo pipefail

JOBID="${1:?need JOBID as arg 1}"
CUDA="${2:?need CUDA_VISIBLE_DEVICES as arg 2}"
OUT_DIR="${3:-./results_t0t1/sys60/t12}"
SEED="${4:-0}"
DATA_DIR="${5:-/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/why-hard-t0t1/experiments/why_hard_to_sample/systems/sys60}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Fixed inputs (sys60 reference MCLMC run + T10 artifacts).
RUN_DIR="/global/homes/l/linusu/GIGALens-Code/results/testsys60/mclmc"
T10_DIR="$HERE/results_t0t1/sys60/t10"

srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 --time=60 \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
  bash -lc "
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export CUDA_VISIBLE_DEVICES=$CUDA
    export JAX_ENABLE_X64=1
    cd $HERE

    echo '=== T12 flank-crossing / pixelation check (float64) ==='
    t0=\$(date +%s)
    /usr/bin/python3 t12_flank_crossing.py \
      --data-dir '$DATA_DIR' \
      --run-dir '$RUN_DIR' \
      --t10-dir '$T10_DIR' \
      --out-dir '$OUT_DIR' \
      --seed $SEED
    t1=\$(date +%s)
    echo \"[wall] T12 = \$((t1 - t0)) s\"
  "
