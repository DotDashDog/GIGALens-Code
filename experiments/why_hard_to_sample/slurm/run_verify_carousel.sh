#!/bin/bash
# GPU self-check of the two carousel system modules (qz hash + chi2-at-z_best).
set -u
HARNESS=/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/why-hard-t0t1/experiments/why_hard_to_sample
IMG=docker:ghcr.io/nvidia/jax:jax-2026-04-13
PP=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages

salloc --no-shell -N 1 -C gpu -G 1 -t 00:25:00 -q interactive -A deepsrch_g 2>&1 | tee /tmp/vc_salloc.$$ | head -2
JOBID=$(grep -oP 'salloc: Granted job allocation \K[0-9]+' /tmp/vc_salloc.$$ || grep -oP 'job \K[0-9]+' /tmp/vc_salloc.$$ | head -1)
rm -f /tmp/vc_salloc.$$
if [ -z "${JOBID:-}" ]; then echo "NO JOBID"; exit 1; fi
echo "jobid=$JOBID"

RC=0
for ARM in old new; do
  echo "=== verify carousel_min_$ARM ==="
  srun --overlap --jobid=$JOBID -N1 -n1 --gpus=1 shifter --image=$IMG \
    bash -c "export PYTHONPATH=$PP JAX_ENABLE_X64=1; /usr/bin/python3 $HARNESS/systems/carousel_min_$ARM/system.py --verify" || RC=1
done
scancel $JOBID
echo "allocation $JOBID released"
echo "EXIT=$RC"
exit $RC
