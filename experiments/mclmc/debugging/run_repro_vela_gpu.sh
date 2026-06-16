#!/bin/bash
# Run repro_vela_shapelets.py on the interactive GPU node via srun --overlap + flock.
# Models run_d3_on_gpu.sh. Usage:
#   bash run_repro_vela_gpu.sh <out_tag> [extra args to repro_vela_shapelets.py ...]
# Example:
#   bash run_repro_vela_gpu.sh ds1e-6_hp --n-max 25 --diag-scale 1e-6 --high-precision \
#        --num-burnin-steps 2000 --num-results 500 --bootstrap-map-samples 16 \
#        --out-dir diagnosis_2026-06/fix1/diag_sweep/ds1e-6_hp
set -euo pipefail

OUT_TAG="${1:?Usage: run_repro_vela_gpu.sh <out_tag> [extra args...]}"
shift || true
EXTRA_ARGS="$*"

LOCK=/global/homes/l/linusu/.claude/mclmc_gpu.lock
JOBID=$(squeue -u linusu -h -o %i -n jupyter | head -1)
if [ -z "$JOBID" ]; then
    echo "ERROR: no jupyter job found for linusu. Cannot srun --overlap." >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPRO="${SCRIPT_DIR}/repro_vela_shapelets.py"
LOG_DIR="${SCRIPT_DIR}/diagnosis_2026-06/fix1/diag_sweep"
mkdir -p "$LOG_DIR"
LOG="${LOG_DIR}/${OUT_TAG}.log"

PYTHONPATH_VAL="/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/homes/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages"

echo "=== repro_vela ${OUT_TAG} jobid=${JOBID} host=$(hostname) date=$(date -Iseconds) ===" | tee "$LOG"
echo "=== args: ${EXTRA_ARGS} ===" | tee -a "$LOG"
echo "=== output log: $LOG ===" | tee -a "$LOG"
echo "=== acquiring GPU lock $LOCK ===" | tee -a "$LOG"

flock "$LOCK" \
    srun --overlap --jobid="$JOBID" \
        --ntasks=1 --gpus=1 \
        shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
        bash -c "
            set -euo pipefail
            export XLA_PYTHON_CLIENT_PREALLOCATE=false
            export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
            export PYTHONPATH=\"${PYTHONPATH_VAL}\"
            python -u '${REPRO}' ${EXTRA_ARGS}
        " 2>&1 | tee -a "$LOG"

EC=${PIPESTATUS[0]}
echo "=== repro_vela ${OUT_TAG}: exit ${EC} ===" | tee -a "$LOG"
exit $EC
