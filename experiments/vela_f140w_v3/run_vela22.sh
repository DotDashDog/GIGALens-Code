#!/bin/bash
# vela22 fits on one interactive 4-GPU Perlmutter node (2026-09-17). Allocates its own node
# (salloc --no-shell), runs ONE process per shard that sees all 4 GPUs (MAP shards its
# multi-start batch and MCLMC shards its chains across devices internally), then releases
# the allocation. Log: $LOG.
#
#   experiments/vela_f140w_v3/run_vela22.sh                       # default: Sersic, truth-free MAP start
#   YAML=.../fit_sersic_truthinit_vela22.yaml SHARDS=6/10 experiments/vela_f140w_v3/run_vela22.sh
#   YAML=.../fit_shapelets_truthinit_vela22.yaml SHARDS="24/40 25/40 26/40 27/40" .../run_vela22.sh
#   ACCOUNT=m5362_g WALL=02:00:00 ...
set -u
REPO=/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/vela-generator-v2
GL=${GL:-/global/u1/l/linusu/gigalens/src}   # linusu-dev-merge: curvature driver (PR #115 merged 2026-09-17) + TruncatedDiskNormal (PR #128)
PY=/global/homes/l/linusu/.conda/envs/gigalens_env/bin/python
YAML=${YAML:-$REPO/experiments/vela_f140w_v3/fit_sersic_vela22.yaml}
OUT=/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3
SHARDS=${SHARDS:-6/10}               # run list slices runs[i::N]; vela22 is manifest index 6 of 10
ACCOUNT=${ACCOUNT:-deepsrch_g}
WALL=${WALL:-04:00:00}
LOG=${LOG:-$OUT/runs/vela22_$(basename "$YAML" .yaml)_$(date +%Y%m%d_%H%M%S).log}
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
  unset JAX_PLATFORMS   # 2026-09-19: an inherited JAX_PLATFORMS=cpu (from a CPU smoke test in the calling shell) ran a whole fit on the node CPU until the wall limit
  cd $REPO
  $PY -c 'import jax, gigalens; print(\"jax\", jax.__version__, \"devices\", jax.devices()); print(\"gigalens\", gigalens.__file__)'
  for S in $SHARDS; do
    echo \"[\$(date)] === shard \$S of $YAML\"
    $PY -u -m gigalens_research.simtests run $YAML --output-dir $OUT --shard \$S || exit 1
  done
" 2>&1 | tee -a "$LOG"
RC=${PIPESTATUS[0]}
echo "[$(date)] run exited with $RC" | tee -a "$LOG"
exit $RC
