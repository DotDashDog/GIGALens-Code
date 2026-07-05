#!/bin/bash
# T21 -- "typical-set init" arm for both carousel minimal-case posteriors + their
# matched Gaussian-clone controls. Runs t21_typical_init.py once on 1 GPU:
#   both arms x {real seeds 1,2,3 + clone control seed 1} from a typical-set init.
# Follows run_t15.sh's pattern EXACTLY (same shifter image, PYTHONPATH ordering,
# JAX_ENABLE_X64=1, container python, srun --overlap against an already-allocated
# JOBID, CUDA_VISIBLE_DEVICES hardcoded to 0).
#
# f64-conv is the NEW STANDARD for this phase: export WHTS_CONV_PRECISION=float64
# so the carousel system module renders the convolution in float64 (T22 hook).
#
# NOTE for the orchestrator: request the hbm80g pool. f64 conv DOUBLES the
# convolution buffers, and 8-chain MCLMC + f64 conv is UNTESTED on the 40GB card
# (the per-point ~52MB intermediates are why logp selection is chunked to 16).
#
# Cost: 2 arms x (3 real + 1 clone) STANDARD MCLMC runs (8 chains, 2000/2000)
# plus two 512-draw chunked logp selections. --time=90.
#
# Usage:
#   ./run_t21.sh <JOBID> [OUT_DIR]
# e.g.
#   ./run_t21.sh 55420000 ./results_carousel/phaseC/t21
set -euo pipefail

JOBID="${1:?need JOBID as arg 1}"
OUT_DIR="${2:-./results_carousel/phaseC/t21}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 --time=90 \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    # srun --gpus=1 renumbers the single allocated GPU to device 0, so hardcode 0.
    export CUDA_VISIBLE_DEVICES=0
    export JAX_ENABLE_X64=1
    # f64 conv is the new standard for this phase (T22 discriminator hook); the
    # carousel system module reads WHTS_CONV_PRECISION -- the driver never sets it.
    export WHTS_CONV_PRECISION=float64
    cd $HERE

    echo '=== T21 typical-set init (both arms, real+clone; conv float64) ==='
    t0=\$(date +%s)
    /usr/bin/python3 t21_typical_init.py --arm both --stage both --out-dir '$OUT_DIR'
    t1=\$(date +%s)
    echo \"[wall] T21 = \$((t1 - t0)) s\"
  "
