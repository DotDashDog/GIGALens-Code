#!/bin/bash
# T17 (carousel B2) -- Gaussian-clone build + T1 clone run for BOTH carousel arms.
# For each arm: build_clone.py fits mean+full-cov to the user's 10k reference
# MCLMC z-samples (arrays.npz samples_z, post-burn-in-only), then run_t1_clone.py
# samples that clone with the STANDARD config (8x2000/2000, seed 1).
#
# CRITICAL REGISTERED CONSTRAINT (kept, unchanged from run_t1_clone.py): the
# clone run's qz is the REAL arm's qz from load_target(data-dir) -- NEVER the
# fitted clone covariance. run_t1_clone.py line ~90 already loads qz that way;
# for the carousel arms load_target returns each arm's real qz.
#
# OLD arm: fit the clone EXCLUDING chain 0 entirely (--exclude-chains 0): chain 0
# is a metastable transient (parked 3-4 sigma for the first half of its draws)
# whose inclusion would inflate the covariance. NEW arm: all chains.
# This exclusion is a logged design choice (also recorded in the clone manifest).
#
# Follows run_t1.sh / run_t14b.sh pattern EXACTLY (shifter image, PYTHONPATH,
# JAX_ENABLE_X64=1, container python, srun --overlap against an existing JOBID).
#
# Usage:
#   ./run_t17.sh <JOBID> <CUDA_VISIBLE_DEVICES> [SEED]
# e.g.
#   ./run_t17.sh 55420000 0,1,2,3 1
set -euo pipefail

JOBID="${1:?need JOBID as arg 1}"
CUDA="${2:?need CUDA_VISIBLE_DEVICES as arg 2}"
SEED="${3:-1}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

OLD_DATA_DIR="$HERE/systems/carousel_min_old"
NEW_DATA_DIR="$HERE/systems/carousel_min_new"
OLD_RUN="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/minimal_case_oldbij/mclmc"
NEW_RUN="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/minimal_case_newbij/mclmc"
# samples_z is post-burn-in-only (8, 10000, 14): pool all 10000 draws.
NR=10000

srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=4 --time=180 \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export CUDA_VISIBLE_DEVICES=$CUDA
    export JAX_ENABLE_X64=1
    cd $HERE

    mkdir -p ./results_carousel/old/t1 ./results_carousel/new/t1

    echo '=== T17 B2 OLD arm: build clone (EXCLUDE chain 0) ==='
    /usr/bin/python3 build_clone.py \
      --source-run '$OLD_RUN' --num-results $NR --exclude-chains 0 \
      --data-dir '$OLD_DATA_DIR' \
      --out ./results_carousel/old/t1/clone.npz
    echo '=== T17 B2 OLD arm: T1 clone run (qz = real OLD arm qz) ==='
    /usr/bin/python3 run_t1_clone.py \
      --clone ./results_carousel/old/t1/clone.npz \
      --data-dir '$OLD_DATA_DIR' \
      --out-dir ./results_carousel/old/t1 --seed $SEED

    echo '=== T17 B2 NEW arm: build clone (all chains) ==='
    /usr/bin/python3 build_clone.py \
      --source-run '$NEW_RUN' --num-results $NR \
      --data-dir '$NEW_DATA_DIR' \
      --out ./results_carousel/new/t1/clone.npz
    echo '=== T17 B2 NEW arm: T1 clone run (qz = real NEW arm qz) ==='
    /usr/bin/python3 run_t1_clone.py \
      --clone ./results_carousel/new/t1/clone.npz \
      --data-dir '$NEW_DATA_DIR' \
      --out-dir ./results_carousel/new/t1 --seed $SEED
  "
