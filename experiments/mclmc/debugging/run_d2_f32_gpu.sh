#!/bin/bash
# Run D2 noise floor (float32) on GPU node via srun --overlap.
set -euo pipefail

JOBID=$(squeue -u linusu -h -o %i -n jupyter | head -1)
echo "[run_d2_f32] jobid=$JOBID host=$(hostname) date=$(date -Iseconds)"

LOG_DIR="/global/homes/l/linusu/GIGALens-Code/experiments/mclmc/debugging/diagnosis_2026-06/d1_d2"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/d2_f32_gpu_run.log"

echo "[run_d2_f32] Waiting for GPU lock..."
flock /global/homes/l/linusu/.claude/mclmc_gpu.lock \
    srun --overlap --jobid="$JOBID" \
        --ntasks=1 --cpus-per-task=16 \
        shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
        bash -c "
            export PYTHONPATH=\"/global/homes/l/linusu/sidecar_jax_upgrade:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages\"
            export XLA_PYTHON_CLIENT_PREALLOCATE=false
            export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
            export JAX_PLATFORMS=cuda
            echo '--- versions ---'
            python -c 'import jax, jaxlib; print(\"jax\", jax.__version__, \"jaxlib\", jaxlib.__version__)'
            echo '--- running D2 float32 ---'
            python -u /global/homes/l/linusu/GIGALens-Code/experiments/mclmc/debugging/d2_noise_floor_f32.py
            ec=\$?
            echo '--- D2 f32 exit code:' \$ec '---'
            exit \$ec
        " 2>&1 | tee "$LOG_FILE"

EC=${PIPESTATUS[0]}
echo "[run_d2_f32] exit code: $EC"
exit $EC
