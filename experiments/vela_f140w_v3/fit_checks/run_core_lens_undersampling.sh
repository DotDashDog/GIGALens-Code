#!/bin/bash
# Lens-light undersampling diagnostic for every system of a dataset, one GPU process per system
# (the ss=64 simulators exhaust an A100 across systems and kill a login-node process). Allocates one
# GPU, loops the manifest, releases. 2026-09-19: tied-core set -> tied_core_lens_undersampling.json.
#
#   DS=<dataset dir> OUT=<json> experiments/vela_f140w_v3/fit_checks/run_core_lens_undersampling.sh
set -u
REPO=/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/vela-generator-v2
GL=${GL:-/global/u1/l/linusu/gigalens/src}
PY=/global/homes/l/linusu/.conda/envs/gigalens_env/bin/python
DS=${DS:-/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset}
OUT=${OUT:-$REPO/experiments/vela_f140w_v3/fit_checks/tied_core_lens_undersampling.json}
ACCOUNT=${ACCOUNT:-deepsrch_g}; WALL=${WALL:-00:45:00}
LOG=${LOG:-/global/homes/l/linusu/.claude/jobs/41490d07/tmp/tied_undersampling.log}
SIDS=$($PY -c "import json,sys; m=json.load(open('$DS/manifest.json')); s=m['system_ids'] if isinstance(m,dict) and 'system_ids' in m else [x if isinstance(x,str) else x['system_id'] for x in (m['systems'] if isinstance(m,dict) else m)]; print(' '.join(s))")
echo "[$(date)] systems: $SIDS" | tee -a "$LOG"
ALLOC_OUT="$(salloc --no-shell -N1 -C "gpu&hbm80g" -G 1 -q interactive -A "$ACCOUNT" -t "$WALL" 2>&1)"; echo "$ALLOC_OUT" | tee -a "$LOG"
JOBID=$(echo "$ALLOC_OUT" | grep -oP 'Granted job allocation \K[0-9]+' | head -1)
if [ -z "${JOBID:-}" ]; then echo "ERROR: no job id from salloc" | tee -a "$LOG"; exit 1; fi
trap 'echo "[$(date)] releasing allocation $JOBID" | tee -a "$LOG"; scancel "$JOBID"' EXIT
for SID in $SIDS; do
  echo "[$(date)] === $SID" | tee -a "$LOG"
  srun --overlap --jobid="$JOBID" -N1 -n1 --gpus=1 bash -c "
    unset JAX_PLATFORMS; export PYTHONPATH=$GL:$REPO/src JAX_ENABLE_X64=1 MPLBACKEND=Agg; cd $REPO
    $PY -u experiments/vela_f140w_v3/fit_checks/core_lens_undersampling.py $DS $SID $OUT" 2>&1 | grep -v -i warning | tee -a "$LOG"
done
echo "[$(date)] diagnostic finished" | tee -a "$LOG"
