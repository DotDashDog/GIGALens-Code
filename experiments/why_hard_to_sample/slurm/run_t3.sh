#!/bin/bash
# T3 multi-scale transect scan launcher.
# Follows the EXACT pattern of run_t0.sh: same shifter image, PYTHONPATH,
# container python. Precision is process-global in jax, so this runs the script
# TWICE in the SAME srun/shifter shell:
#   1) float64 FULL  (JAX_ENABLE_X64=1, all ~7 directions)         -> $OUT_DIR/float64
#   2) float32 CTRL  (JAX_ENABLE_X64=0, --allow-float32, reduced)  -> $OUT_DIR/float32
# The float32 arm is O5's known noise floor -- the positive control that the
# diagnostic CAN see roughness; it MUST blow up.
#
# Usage:
#   ./run_t3.sh <JOBID> <CUDA_VISIBLE_DEVICES> [OUT_DIR] [SEED] [DATA_DIR]
# e.g.
#   ./run_t3.sh 55392666 0 ./t3_out 1 \
#       /global/u1/l/linusu/GIGALens-Code/.claude/worktrees/why-hard-t0t1/experiments/why_hard_to_sample/systems/sys60
set -euo pipefail

JOBID="${1:?need JOBID as arg 1}"
CUDA="${2:?need CUDA_VISIBLE_DEVICES as arg 2}"
OUT_DIR="${3:-./t3_out}"
SEED="${4:-1}"
DATA_DIR="${5:-/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/why-hard-t0t1/experiments/why_hard_to_sample/systems/sys60}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Fixed inputs (sys60 T0/T1 artifacts + reference run diagnostics).
SAMPLES="$HERE/results_t0t1/sys60/clone_source.npz"
CLONE="$HERE/results_t0t1/sys60/clone.npz"
REFDIAG="/global/homes/l/linusu/GIGALens-Code/results/testsys60/mclmc/diagnostics.npz"

srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
  bash -lc "
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export CUDA_VISIBLE_DEVICES=$CUDA
    cd $HERE

    echo '=== T3 float64 (full) ==='
    export JAX_ENABLE_X64=1
    /usr/bin/python3 t3_transects.py \
      --data-dir '$DATA_DIR' \
      --samples '$SAMPLES' \
      --clone '$CLONE' \
      --ref-diagnostics '$REFDIAG' \
      --out-dir '$OUT_DIR/float64' \
      --seed $SEED

    echo '=== T3 float32 (positive control, reduced) ==='
    export JAX_ENABLE_X64=0
    # Clean control: systems/sys60/system.py skips its x64 forcing under this
    # env var, so the model is BUILT float32 (constants included), not flipped
    # after the fact. t3_transects.py still records energy_dtype_achieved.
    export WHTS_FLOAT32_CONTROL=1
    /usr/bin/python3 t3_transects.py \
      --data-dir '$DATA_DIR' \
      --samples '$SAMPLES' \
      --clone '$CLONE' \
      --ref-diagnostics '$REFDIAG' \
      --out-dir '$OUT_DIR/float32' \
      --seed $SEED \
      --allow-float32
  "
