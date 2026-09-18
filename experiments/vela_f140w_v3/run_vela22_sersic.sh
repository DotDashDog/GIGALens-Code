#!/bin/bash
# vela22 single-Sersic-source fit on one interactive 4-GPU Perlmutter node (2026-09-17).
# Allocates its own node (salloc --no-shell), runs ONE process that sees all 4 GPUs
# (MAP shards its multi-start batch and MCLMC shards its chains across devices
# internally), then releases the allocation. Log: $LOG.
#
#   experiments/vela_f140w_v3/run_vela22_sersic.sh            # allocate + run
#   ACCOUNT=m5362_g WALL=02:00:00 experiments/.../run_vela22_sersic.sh
set -u
REPO=/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/vela-generator-v2
GL=/global/u1/l/linusu/gigalens/.claude/worktrees/dev-curvature/src   # dev + curvature driver (see fit yaml header)
PY=/global/homes/l/linusu/.conda/envs/gigalens_env/bin/python
YAML=$REPO/experiments/vela_f140w_v3/fit_sersic_vela22.yaml
OUT=/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3
SHARD=${SHARD:-6/10}                 # manifest index 6 of 10 = vela22_cam12_a0.400_rep00
ACCOUNT=${ACCOUNT:-deepsrch_g}
WALL=${WALL:-04:00:00}
LOG=${LOG:-$OUT/runs/vela22_sersic_v1_$(date +%Y%m%d_%H%M%S).log}
mkdir -p "$(dirname "$LOG")"

echo "[$(date)] requesting 1 node x 4 GPU (hbm80g), $WALL, account $ACCOUNT" | tee -a "$LOG"
ALLOC_OUT="$(salloc --no-shell -N1 -C "gpu&hbm80g" -G 4 -q interactive -A "$ACCOUNT" -t "$WALL" 2>&1)"
echo "$ALLOC_OUT" | tee -a "$LOG"
JOBID=$(echo "$ALLOC_OUT" | grep -oP 'Granted job allocation \K[0-9]+' | head -1)
if [ -z "${JOBID:-}" ]; then echo "ERROR: no job id from salloc" | tee -a "$LOG"; exit 1; fi
trap 'echo "[$(date)] releasing allocation $JOBID" | tee -a "$LOG"; scancel "$JOBID"' EXIT
echo "jobid=$JOBID" | tee -a "$LOG"

srun --overlap --jobid="$JOBID" -N1 -n1 --gpus=4 --gpu-bind=none bash -c "
  export PYTHONPATH=$GL:$REPO/src JAX_ENABLE_X64=1 MPLBACKEND=Agg
  cd $REPO
  $PY -c 'import jax, gigalens; print(\"jax\", jax.__version__, \"devices\", jax.devices()); print(\"gigalens\", gigalens.__file__)'
  $PY -u -m gigalens_research.simtests run $YAML --output-dir $OUT --shard $SHARD
" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "[$(date)] run exited with $RC" | tee -a "$LOG"
exit $RC
