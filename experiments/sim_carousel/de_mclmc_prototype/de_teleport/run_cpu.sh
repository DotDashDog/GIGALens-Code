#!/bin/bash
# CPU-only run wrapper for the de_teleport prototype (mirrors the orchestrator env).
# Usage:  bash run_cpu.sh <script.py> [args...]
export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/u1/l/linusu/GIGALens-Code/src:/global/u1/l/linusu/gigalens/src:$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
exec shifter --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 /usr/bin/python3 "$@"
