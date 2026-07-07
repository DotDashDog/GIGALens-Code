#!/bin/bash
# T10 (on-ridge spike census) + T11 (render-space spike localization) launcher.
# Follows the EXACT pattern of run_t3.sh / run_t8_t9.sh (same shifter image,
# PYTHONPATH, container python). float64-ONLY (JAX_ENABLE_X64=1) -- neither
# experiment has a float32 arm. Both scripts run the HARD chi^2 render-path gate
# at startup (reusing E1's verified render path) and RAISE if it does not
# reconcile with log_prob's aux. T10 runs FIRST and writes spike_list.json; T11
# then CONSUMES that spike_list.json. They run SEQUENTIALLY in the same shell.
#
# T10 is the expensive step (~2048 batched forward-mode Jacobians of a 6400x22
# map). We give the srun a generous time budget and print wall-time per script.
#
# Usage:
#   ./run_t10_t11.sh <JOBID> <CUDA_VISIBLE_DEVICES> [OUT_ROOT] [SEED] [DATA_DIR]
# e.g.
#   ./run_t10_t11.sh 55404453 0 ./results_t0t1/sys60 1 \
#       /global/u1/l/linusu/GIGALens-Code/.claude/worktrees/why-hard-t0t1/experiments/why_hard_to_sample/systems/sys60
set -euo pipefail

JOBID="${1:?need JOBID as arg 1}"
CUDA="${2:?need CUDA_VISIBLE_DEVICES as arg 2}"
OUT_ROOT="${3:-./results_t0t1/sys60}"
SEED="${4:-1}"
DATA_DIR="${5:-/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/why-hard-t0t1/experiments/why_hard_to_sample/systems/sys60}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Fixed inputs (sys60 reference MCLMC run + E1 artifacts).
RUN_DIR="/global/homes/l/linusu/GIGALens-Code/results/testsys60/mclmc"
E1_JSON="$HERE/results_t0t1/sys60/e1/e1_results.json"

T10_OUT="$OUT_ROOT/t10"
T11_OUT="$OUT_ROOT/t11"
SPIKE_LIST="$T10_OUT/spike_list.json"

srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 --time=60 \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
  bash -lc "
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export CUDA_VISIBLE_DEVICES=$CUDA
    export JAX_ENABLE_X64=1
    cd $HERE

    echo '=== T10 on-ridge spike census (float64) ==='
    t0=\$(date +%s)
    /usr/bin/python3 t10_spike_census.py \
      --data-dir '$DATA_DIR' \
      --run-dir '$RUN_DIR' \
      --out-dir '$T10_OUT' \
      --seed $SEED \
      --segment-len 256 \
      --n-segments 8
    t1=\$(date +%s)
    echo \"[wall] T10 = \$((t1 - t0)) s\"

    echo '=== T11 render-space spike localization (float64) ==='
    /usr/bin/python3 t11_spike_pixels.py \
      --data-dir '$DATA_DIR' \
      --run-dir '$RUN_DIR' \
      --spike-list '$SPIKE_LIST' \
      --e1-json '$E1_JSON' \
      --out-dir '$T11_OUT'
    t2=\$(date +%s)
    echo \"[wall] T11 = \$((t2 - t1)) s ; total = \$((t2 - t0)) s\"
  "
