#!/bin/bash
# Combined runner: D1 + D2-float32 in a single GPU srun.
# D2-float64 runs separately (CPU, separate process, different lock acquisition).
# This avoids multiple lock acquisitions for the sequential GPU work.
set -euo pipefail

JOBID=$(squeue -u linusu -h -o %i -n jupyter | head -1)
echo "[d1_d2_combined] jobid=$JOBID host=$(hostname) date=$(date -Iseconds)"

LOG_DIR="/global/homes/l/linusu/GIGALens-Code/experiments/mclmc/debugging/diagnosis_2026-06/d1_d2"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/d1_d2_combined_run.log"

echo "[d1_d2_combined] Acquiring GPU lock and running D1 then D2-f32..."

flock /global/homes/l/linusu/.claude/mclmc_gpu.lock \
    srun --overlap --jobid="$JOBID" \
        --ntasks=1 --cpus-per-task=16 \
        shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
        bash -c "
            export PYTHONPATH=\"/global/homes/l/linusu/sidecar_jax_upgrade:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages\"
            export XLA_PYTHON_CLIENT_PREALLOCATE=false
            export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
            # Let JAX auto-detect (CUDA is available on the GPU node)
            echo '--- versions ---'
            python -c 'import jax, jaxlib; print(\"jax\", jax.__version__, \"jaxlib\", jaxlib.__version__)'
            python -c 'import tensorflow_probability as tfp; print(\"tfp\", tfp.__version__)'

            echo '============================================================'
            echo 'PHASE: D1 finiteness audit'
            echo '============================================================'
            python -u /global/homes/l/linusu/GIGALens-Code/experiments/mclmc/debugging/d1_finiteness_audit.py
            D1_EC=\$?
            echo \"D1 exit code: \$D1_EC\"

            echo '============================================================'
            echo 'PHASE: D2 noise floor (float32)'
            echo '============================================================'
            python -u /global/homes/l/linusu/GIGALens-Code/experiments/mclmc/debugging/d2_noise_floor_f32.py
            D2_EC=\$?
            echo \"D2 f32 exit code: \$D2_EC\"

            # Exit with combined code
            if [ \$D1_EC -ne 0 ] || [ \$D2_EC -ne 0 ]; then
                exit 1
            fi
            exit 0
        " 2>&1 | tee "$LOG_FILE"

EC=${PIPESTATUS[0]}
echo "[d1_d2_combined] exit code: $EC"
exit $EC
