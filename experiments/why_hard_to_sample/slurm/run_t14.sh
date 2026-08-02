#!/bin/bash
# T14 -- exact Hessian vs GN comb along the T12 top-spike dial (ss2 model), on the
# ORIGINAL ss2 data vs the re-simulated d' (ss128) data. Follows the run_t3.sh /
# run_t12.sh pattern: same shifter image, PYTHONPATH, container python. float64-ONLY
# (JAX_ENABLE_X64=1) -- a SINGLE float64 pass (no float32 arm). The script re-runs
# the HARD chi^2 render-path gate at startup for EACH target (against that target's
# OWN observed/err map) and RAISES if either fails to reconcile.
#
# Cost: ~2x97 Jacobians + ~2x97 22-dim Hessians + ~2x97 renders; minutes on 1 GPU.
#
# Usage:
#   ./run_t14.sh <JOBID> <CUDA_VISIBLE_DEVICES> [OUT_DIR] [SEED]
# e.g.
#   ./run_t14.sh 55412000 0 ./results_t0t1/sys60/t14 0
set -euo pipefail

JOBID="${1:?need JOBID as arg 1}"
CUDA="${2:?need CUDA_VISIBLE_DEVICES as arg 2}"
OUT_DIR="${3:-./results_t0t1/sys60/t14}"
SEED="${4:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Fixed inputs (sys60 reference MCLMC run + T10 artifacts + the two targets).
OLD_DATA_DIR="$HERE/systems/sys60"                 # original ss2 data
NEW_SYS_DIR="$HERE/systems/sys60_ss16data"         # d' (ss128) via build_prob_model
RUN_DIR="/global/homes/l/linusu/GIGALens-Code/results/testsys60/mclmc.stale-20260703T111618"  # original reference (rotated by user reruns 2026-07-03)
T10_DIR="$HERE/results_t0t1/sys60/t10"

srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 --time=60 \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export CUDA_VISIBLE_DEVICES=$CUDA
    export JAX_ENABLE_X64=1
    cd $HERE

    echo '=== T14 exact-Hessian vs GN-comb dial (float64) ==='
    t0=\$(date +%s)
    /usr/bin/python3 t14_hessian_dial.py \
      --old-data-dir '$OLD_DATA_DIR' \
      --new-sys-dir '$NEW_SYS_DIR' \
      --run-dir '$RUN_DIR' \
      --t10-dir '$T10_DIR' \
      --out-dir '$OUT_DIR' \
      --seed $SEED
    t1=\$(date +%s)
    echo \"[wall] T14 = \$((t1 - t0)) s\"
  "
