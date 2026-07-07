#!/bin/bash
# T19 -- carousel minimal-case Gram-conditioning vs xi-spike test (Phase C).
# Runs t19_gram_xi.py once on 1 GPU for BOTH arms: reproduces the frac(xi>10)
# alignment gate, stratified point selection (fixed seed), per-point weighted-Gram
# conditioning cond(G)/lmin(G) and Gauss-Newton curvature lambda1_GN (jacfwd through
# the lstsq solve), registered ratios/Spearman metrics, npz + summary.json + PNGs.
# Follows run_t15.sh's pattern EXACTLY (same shifter image, PYTHONPATH ordering,
# JAX_ENABLE_X64=1, container python, srun --overlap against an already-allocated
# JOBID). float64-ONLY.
#
# Cost: ~512 selected points/arm x (1 design build + 14 jacfwd passes); old arm has
# 2 variants (pooled + basin-excluded) sharing rendered points. ~30-45 min on 1 GPU.
#
# Usage:
#   ./run_t19.sh <JOBID> <CUDA_VISIBLE_DEVICES> [OUT_DIR]
# e.g.
#   ./run_t19.sh 55420000 0 ./results_carousel/phaseC
set -euo pipefail

JOBID="${1:?need JOBID as arg 1}"
CUDA="${2:?need CUDA_VISIBLE_DEVICES as arg 2}"
OUT_DIR="${3:-./results_carousel/phaseC}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 --time=60 \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export CUDA_VISIBLE_DEVICES=$CUDA
    export JAX_ENABLE_X64=1
    cd $HERE

    echo '=== T19 carousel Gram-conditioning vs xi spikes (float64) ==='
    t0=\$(date +%s)
    /usr/bin/python3 t19_gram_xi.py \
      --arm both \
      --out-dir '$OUT_DIR'
    t1=\$(date +%s)
    echo \"[wall] T19 = \$((t1 - t0)) s\"
  "
