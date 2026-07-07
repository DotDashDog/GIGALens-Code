#!/bin/bash
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -G 1
#SBATCH -q debug
#SBATCH -A deepsrch_g
#SBATCH -t 15
#SBATCH --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13
#SBATCH -o /global/homes/l/linusu/.claude/jobs/8a0cd9d7/tmp/t29_%j.out
# T29 -- NFW_ELLIPSE_SLOPE gates. Submitted with sbatch (returns instantly; no
# salloc-wait kill window -- see the T28 relaunch saga in the log).
set -euo pipefail
# HARDCODED: under sbatch BASH_SOURCE is the spooled copy in /var/spool/slurmd,
# so dirname-based resolution breaks (cost one 16 s FAILED job to learn).
HERE="/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/why-hard-t0t1/experiments/why_hard_to_sample"

srun --ntasks=1 --gpus=1 shifter --module=gpu,nccl-plugin \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export JAX_ENABLE_X64=1
    cd $HERE
    /usr/bin/python3 t29_slope_class_gpu.py
  "
