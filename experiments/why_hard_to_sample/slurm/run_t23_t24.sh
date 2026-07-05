#!/bin/bash
# T23 + T24 -- momentum-space xi accounting (T23) + encounter-geometry census
# (T24), both on the SAVED T21 chains. Shares ONE hbm80g interactive GPU
# allocation. Registered 2026-07-03 (docs/logs/why-hard-to-sample.md).
#
# Follows run_t21.sh's shifter image / PYTHONPATH ordering / env exactly, with
# the T21 out-dir mishap DESIGNED OUT: the python scripts HARDCODE their output
# dirs (results_carousel/phaseC/t23, .../t24). NO out-dir is ever passed on the
# CLI (that is the bug that turned run_t21.sh's OUT_DIR into a literal "0").
#
# This script allocates its OWN GPU (salloc --no-shell), runs both stages via
# srun --overlap, then RELEASES the allocation (scancel) and echoes the exit
# status. f64 conv is the phase standard (WHTS_CONV_PRECISION=float64).
#
# Cost estimate (chunked HVPs, hbm80g):
#   T23 ~ 2*(<=512 spike + <=512 calm) + 2*14 dense-Hessian HVPs  ~ 2.1k HVPs
#   T24 ~ 2*(8 chains * 256 steps) + 2*(12 iters * 24 pts) + 2*14 ~ 4.7k HVPs
#   total ~ 6.8k HVPs; each HVP ~2-3x a logp+grad -> expect ~30-60 min wall.
#
# Usage:
#   ./run_t23_t24.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== allocating 1x hbm80g GPU (interactive, deepsrch_g) ==="
# salloc --no-shell prints "salloc: Granted job allocation <JOBID>" on stderr;
# parse that directly (do NOT guess via squeue -- the user may hold other jobs).
ALLOC_OUT="$(salloc --no-shell -N1 -C "gpu&hbm80g" -G 1 -q interactive -A deepsrch_g -t 150 2>&1)"
echo "$ALLOC_OUT"
JOBID="$(printf '%s\n' "$ALLOC_OUT" | grep -oE 'Granted job allocation [0-9]+' | grep -oE '[0-9]+' | tail -1)"
if [[ -z "$JOBID" ]]; then
  echo "ERROR: could not determine allocated JOBID from salloc output"; exit 1
fi
echo "=== allocated JOBID=$JOBID ==="

RC=0
srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 --time=150 \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    # srun --gpus=1 renumbers the single allocated GPU to device 0 -- MUST be 0.
    export CUDA_VISIBLE_DEVICES=0
    export JAX_ENABLE_X64=1
    export WHTS_CONV_PRECISION=float64
    cd $HERE

    echo '=== SMOKE pass (--limit 16; aborts run on wiring failure) ==='
    /usr/bin/python3 t23_momentum_gpu.py --arm both --limit 16
    /usr/bin/python3 t24_census_gpu.py --arm both --limit 16
    echo '=== SMOKE OK ==='

    echo '=== T23 momentum-space xi accounting (both arms; conv float64) ==='
    t0=\$(date +%s)
    /usr/bin/python3 t23_momentum_gpu.py --arm both
    t1=\$(date +%s)
    echo \"[wall] T23 = \$((t1 - t0)) s\"

    echo '=== T24 encounter-geometry census (both arms + clones) ==='
    /usr/bin/python3 t24_census_gpu.py --arm both
    t2=\$(date +%s)
    echo \"[wall] T24 = \$((t2 - t1)) s ; TOTAL = \$((t2 - t0)) s\"
  " || RC=$?

echo "=== releasing allocation JOBID=$JOBID ==="
scancel "$JOBID" || true
echo "=== run_t23_t24 EXIT status = $RC ==="
exit $RC

# ---------------------------------------------------------------------------
# SMOKE variant (tiny, verifies wiring end-to-end before the full run):
#   srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 --time=20 \
#     shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
#     bash -lc "
#       export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
#       export XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 JAX_ENABLE_X64=1 WHTS_CONV_PRECISION=float64
#       cd $HERE
#       /usr/bin/python3 t23_momentum_gpu.py --arm both --limit 16
#       /usr/bin/python3 t24_census_gpu.py  --arm both --limit 16
#     "
#   # then, on the login node:
#   /global/homes/l/linusu/.conda/envs/gigalens_multinode_env/bin/python t23_t24_analyze.py --smoke
# ---------------------------------------------------------------------------
