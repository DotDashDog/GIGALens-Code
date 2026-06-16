#!/bin/bash
# Run D3 EEVPD measurement on the GPU node via srun --overlap + flock.
# Usage: bash run_d3_on_gpu.sh <n_max> <precision> [extra args to d3 script]
# Example:
#   bash run_d3_on_gpu.sh 25 float32
#   bash run_d3_on_gpu.sh 25 float64
#   bash run_d3_on_gpu.sh 10 float32
#   bash run_d3_on_gpu.sh 10 float64
set -euo pipefail

N_MAX="${1:?Usage: run_d3_on_gpu.sh <n_max> <precision>}"
PRECISION="${2:?Usage: run_d3_on_gpu.sh <n_max> <precision>}"
shift 2 || true
EXTRA_ARGS="$*"

LOCK=/global/homes/l/linusu/.claude/mclmc_gpu.lock
JOBID=$(squeue -u linusu -h -o %i -n jupyter | head -1)
if [ -z "$JOBID" ]; then
    echo "ERROR: no jupyter job found for linusu. Cannot srun --overlap." >&2
    exit 1
fi
echo "=== D3 n_max=${N_MAX} precision=${PRECISION} jobid=${JOBID} host=$(hostname) date=$(date -Iseconds) ==="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
D3_SCRIPT="${SCRIPT_DIR}/d3_eevpd_vs_eps.py"
LOG_DIR="${SCRIPT_DIR}/diagnosis_2026-06/d3"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/run_nmax${N_MAX}_${PRECISION}.log"

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
            export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
            export PYTHONPATH=\"${PYTHONPATH_VAL}\"
            echo '=== D3 env ===' >&2
            env | grep -E '^(JAX_|XLA_|CUDA_|TF_|PYTHONPATH)' | sort >&2
            python -u '${D3_SCRIPT}' --n-max '${N_MAX}' --precision '${PRECISION}' ${EXTRA_ARGS}
        " 2>&1 | tee "$LOG"

EC=${PIPESTATUS[0]}
echo "=== D3 n_max=${N_MAX} ${PRECISION}: exit ${EC} ===" | tee -a "$LOG"
exit $EC
