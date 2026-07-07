#!/bin/bash
# T14b -- curvature along each posterior's OWN chains (CORRECTS T14's off-typical-set
# dial). ss2 model on ss2 data (OLD) vs ss2 model on d'/ss128 data (NEW), each walked
# along its OWN seed-1 results-phase chains. Follows the run_t14.sh / run_t12.sh
# pattern: same shifter image, PYTHONPATH, container python. float64-ONLY
# (JAX_ENABLE_X64=1) -- a SINGLE float64 pass (no float32 arm). The script re-runs the
# HARD chi^2 render-path gate at startup for EACH target (against that target's OWN
# observed/err map) and RAISES if either fails to reconcile.
#
# Cost: ~2x256 Jacobians + ~2x256 22-dim Hessians + ~2x256 renders; minutes on 1 GPU.
#
# Usage:
#   ./run_t14b.sh <JOBID> <CUDA_VISIBLE_DEVICES> [OUT_DIR] [SEED]
# e.g.
#   ./run_t14b.sh 55420000 0 ./results_t0t1/sys60/t14b 0
set -euo pipefail

JOBID="${1:?need JOBID as arg 1}"
CUDA="${2:?need CUDA_VISIBLE_DEVICES as arg 2}"
OUT_DIR="${3:-./results_t0t1/sys60/t14b}"
SEED="${4:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Fixed inputs (two targets + their own seed-1 runs + the frozen std reference).
OLD_DATA_DIR="$HERE/systems/sys60"                 # RE-PINNED ss2 data (OLD model)
NEW_SYS_DIR="$HERE/systems/sys60_ss16data"         # d' (ss128) via build_modelling_sequence
OLD_RUN="$HERE/results_t0t1/sys60/t0/t0_seed1.npz"                 # OLD chains + xi
NEW_RUN="$HERE/resim/sys60_ss16/arm_2/t0/t0_seed1.npz"            # NEW chains + xi
STD_REF="/global/homes/l/linusu/GIGALens-Code/results/testsys60/mclmc.stale-20260703T111618/arrays.npz"  # COMMON std_z

srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 --time=60 \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export CUDA_VISIBLE_DEVICES=$CUDA
    export JAX_ENABLE_X64=1
    cd $HERE

    echo '=== T14b chain-locus curvature census (float64) ==='
    t0=\$(date +%s)
    /usr/bin/python3 t14b_chain_census.py \
      --old-data-dir '$OLD_DATA_DIR' \
      --new-sys-dir '$NEW_SYS_DIR' \
      --old-run '$OLD_RUN' \
      --new-run '$NEW_RUN' \
      --std-ref '$STD_REF' \
      --out-dir '$OUT_DIR' \
      --seed $SEED
    t1=\$(date +%s)
    echo \"[wall] T14b = \$((t1 - t0)) s\"
  "
