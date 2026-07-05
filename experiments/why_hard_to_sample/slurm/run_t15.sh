#!/bin/bash
# T15 -- carousel minimal-case NFW-reparameterization decomposition (Phase A).
# Runs t15_carousel_decompose.py once on 1 GPU: z->physical for both reference
# runs, crowding census, basin/main log-prob split, ridge curvature, overlay
# plots + summary.json. Follows run_t14b.sh's pattern EXACTLY (same shifter
# image, PYTHONPATH ordering, JAX_ENABLE_X64=1, container python, srun --overlap
# against an already-allocated JOBID). float64-ONLY.
#
# Cost: bijector pushforwards (analytic) + ~4 image renders (b,m x 2 arms);
# seconds-to-a-minute on 1 GPU.
#
# Usage:
#   ./run_t15.sh <JOBID> <CUDA_VISIBLE_DEVICES> [OUT_DIR]
# e.g.
#   ./run_t15.sh 55420000 0 ./results_carousel/phaseA
set -euo pipefail

JOBID="${1:?need JOBID as arg 1}"
CUDA="${2:?need CUDA_VISIBLE_DEVICES as arg 2}"
OUT_DIR="${3:-./results_carousel/phaseA}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Fixed inputs: the two carousel system modules (built by the parallel subagent)
# and the two read-only MCLMC reference-run dirs (arrays.npz -> samples_z).
OLD_DATA_DIR="$HERE/systems/carousel_min_old"   # NFW_ELLIPSE / alpha_Rs
NEW_DATA_DIR="$HERE/systems/carousel_min_new"   # NFW_ELLIPSE_EINSTEIN / theta_E
OLD_RUN="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/minimal_case_oldbij/mclmc"
NEW_RUN="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/minimal_case_newbij/mclmc"

srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 --time=60 \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export CUDA_VISIBLE_DEVICES=$CUDA
    export JAX_ENABLE_X64=1
    cd $HERE

    echo '=== T15 carousel NFW-reparam decomposition (float64) ==='
    t0=\$(date +%s)
    /usr/bin/python3 t15_carousel_decompose.py \
      --old-data-dir '$OLD_DATA_DIR' \
      --new-data-dir '$NEW_DATA_DIR' \
      --old-run '$OLD_RUN' \
      --new-run '$NEW_RUN' \
      --out-dir '$OUT_DIR'
    t1=\$(date +%s)
    echo \"[wall] T15 = \$((t1 - t0)) s\"
  "
