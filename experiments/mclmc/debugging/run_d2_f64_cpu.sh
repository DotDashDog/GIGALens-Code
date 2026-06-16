#!/bin/bash
# Run D2 noise floor (float64) on CPU login node.
# No GPU needed; JAX_PLATFORMS=cpu is set explicitly.
# Must run in a SEPARATE process from float32 (never toggle x64 mid-process).
set -euo pipefail

echo "[run_d2_f64] host=$(hostname) date=$(date -Iseconds)"

LOG_DIR="/global/homes/l/linusu/GIGALens-Code/experiments/mclmc/debugging/diagnosis_2026-06/d1_d2"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/d2_f64_cpu_run.log"

# Run inside the same shifter image for consistent libraries
JOBID=$(squeue -u linusu -h -o %i -n jupyter | head -1)

echo "[run_d2_f64] Waiting for GPU lock (needed for shifter)..."
flock /global/homes/l/linusu/.claude/mclmc_gpu.lock \
    srun --overlap --jobid="$JOBID" \
        --ntasks=1 --cpus-per-task=16 \
        shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
        bash -c "
            export PYTHONPATH=\"/global/homes/l/linusu/sidecar_jax_upgrade:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages\"
            export XLA_PYTHON_CLIENT_PREALLOCATE=false
            export JAX_PLATFORMS=cpu
            echo '--- versions ---'
            python -c 'import jax, jaxlib; print(\"jax\", jax.__version__, \"jaxlib\", jaxlib.__version__)'
            echo '--- running D2 float64 ---'
            python -u /global/homes/l/linusu/GIGALens-Code/experiments/mclmc/debugging/d2_noise_floor_f64.py
            ec=\$?
            echo '--- D2 f64 exit code:' \$ec '---'
            exit \$ec
        " 2>&1 | tee "$LOG_FILE"

EC=${PIPESTATUS[0]}
echo "[run_d2_f64] exit code: $EC"
exit $EC
