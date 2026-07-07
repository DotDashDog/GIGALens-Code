#!/bin/bash
# T18 -- carousel minimal-case MAP-quality arm (OLD system).
# Runs t18_map_arm.py in sequence on 1 GPU: (1) --stage map (replicated MAP,
# num_steps=5000, seed 42, same optimizer as the reference), (2) --stage quality
# (D1/D2/D3 diagnostic on ref-old / ref-new / improved MAPs + calibration), then
# (3) --stage seeds (STANDARD MCLMC off the improved MAP for seeds 1,2,3 +
# displaced-chain census). Follows run_t15.sh's pattern EXACTLY (same shifter
# image, PYTHONPATH ordering, JAX_ENABLE_X64=1, container python, srun --overlap
# against an already-allocated JOBID). float64-ONLY.
#
# Cost: MAP 5000 steps ~10x the reference 58s ~= 10 min; quality ~1-2 min
# (grad+hessian x3 + batched loglike); seeds ~6 min each (3 seeds) ~= 18 min.
# Budget the srun --time accordingly (~45 min headroom below).
#
# Usage:
#   ./run_t18.sh <JOBID> <CUDA_VISIBLE_DEVICES>
# e.g.
#   ./run_t18.sh 55420000 0
set -euo pipefail

JOBID="${1:?need JOBID as arg 1}"
CUDA="${2:?need CUDA_VISIBLE_DEVICES as arg 2}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 --time=45 \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export CUDA_VISIBLE_DEVICES=$CUDA
    export JAX_ENABLE_X64=1
    cd $HERE

    echo '=== T18 stage: map (replicated MAP, num_steps=5000, float64) ==='
    t0=\$(date +%s)
    /usr/bin/python3 t18_map_arm.py --stage map
    t1=\$(date +%s)
    echo \"[wall] T18 map = \$((t1 - t0)) s\"

    echo '=== T18 stage: quality (D1/D2/D3 on 3 MAPs + calibration) ==='
    t2=\$(date +%s)
    /usr/bin/python3 t18_map_arm.py --stage quality
    t3=\$(date +%s)
    echo \"[wall] T18 quality = \$((t3 - t2)) s\"

    echo '=== T18 stage: seeds (STANDARD MCLMC off improved MAP + census) ==='
    t4=\$(date +%s)
    /usr/bin/python3 t18_map_arm.py --stage seeds
    t5=\$(date +%s)
    echo \"[wall] T18 seeds = \$((t5 - t4)) s\"
    echo \"[wall] T18 total = \$((t5 - t0)) s\"
  "
