#!/bin/bash
# E1c compilation-seam probe on the GPU node via srun --overlap + flock.
# Mirrors run_d3_on_gpu.sh exactly.
# Usage: bash run_e1c_on_gpu.sh [precision] [anchor-class]
set -euo pipefail

PRECISION="${1:-float32}"
ANCHOR_CLASS="${2:-both}"

LOCK=/global/homes/l/linusu/.claude/mclmc_gpu.lock
JOBID=$(squeue -u linusu -h -o %i -n jupyter | head -1)
if [ -z "$JOBID" ]; then
    echo "ERROR: no jupyter job found for linusu. Cannot srun --overlap." >&2
    exit 1
fi
echo "=== E1c precision=${PRECISION} anchor=${ANCHOR_CLASS} jobid=${JOBID} host=$(hostname) date=$(date -Iseconds) ==="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
E1C_SCRIPT="${SCRIPT_DIR}/e1c_compilation_seam.py"
LOG_DIR="${SCRIPT_DIR}/diagnosis_2026-06/e1c"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/run_gpu_${PRECISION}_${ANCHOR_CLASS}.log"

PYTHONPATH_VAL="/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/homes/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages"

echo "=== output log: $LOG ==="
echo "=== acquiring GPU lock $LOCK ==="

flock "$LOCK" \
    srun --overlap --jobid="$JOBID" \
        --ntasks=1 --gpus=1 \
        shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
        bash -c "
            set -euo pipefail
            export XLA_PYTHON_CLIENT_PREALLOCATE=false
            export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
            export PYTHONPATH=\"${PYTHONPATH_VAL}\"
            echo '=== E1c env ===' >&2
            env | grep -E '^(JAX_|XLA_|CUDA_|TF_|PYTHONPATH)' | sort >&2
            python -u '${E1C_SCRIPT}' --device gpu --precision '${PRECISION}' --n-max 25 --anchor-class '${ANCHOR_CLASS}' --K 8
        " 2>&1 | tee "$LOG"

EC=${PIPESTATUS[0]}
echo "=== E1c gpu ${PRECISION} ${ANCHOR_CLASS}: exit ${EC} ===" | tee -a "$LOG"
exit $EC
