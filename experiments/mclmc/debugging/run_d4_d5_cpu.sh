#!/bin/bash
# Run a single D4/D5 synthetic diagnostic inside the jax-0.10 shifter container,
# CPU ONLY (the node's GPUs are shared).  All args after the image setup are
# forwarded to d4_d5_synthetic.py.
#
# Usage: bash run_d4_d5_cpu.sh --experiment <tag> [--cond ...] [--qz-init ...] ...
set -u
IMAGE="docker:ghcr.io/nvidia/jax:jax-2026-04-13"
SCRIPT="/global/homes/l/linusu/GIGALens-Code/experiments/mclmc/debugging/d4_d5_synthetic.py"

shifter --image="$IMAGE" bash -c "
    set -u
    export JAX_PLATFORMS=cpu
    export CUDA_VISIBLE_DEVICES=''
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/homes/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    python -u $SCRIPT $*
"
