#!/bin/bash
# Run C (ratio-coordinates grouped prior) launcher: interactive GPU allocation
# on Perlmutter, same idiom as run_dspl_free_r2.sh (which ran successfully for
# Run A): salloc --no-shell + srun into the canonical Shifter container
# (docs/env_setup.md). Approved by grader (user) 2026-07-11 — see the Run C
# design checkpoint in docs/logs/sample-cosmology-dspl.md. The python script
# itself additionally refuses to run without a passing gate JSON.
#
# NOTE: PYTHONPATH points at THIS script's repo checkout (resolved from the
# script location), so running from a worktree uses the worktree's src.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_SRC="$(cd "$HERE/../../src" && pwd)"

echo "=== allocating 1x hbm80g GPU (interactive, deepsrch_g) ==="
ALLOC_OUT="$(salloc --no-shell -N1 -C "gpu&hbm80g" -G 1 -q interactive -A deepsrch_g -t 60 2>&1)"
echo "$ALLOC_OUT"
JOBID="$(printf '%s\n' "$ALLOC_OUT" | grep -oE 'Granted job allocation [0-9]+' | grep -oE '[0-9]+' | tail -1)"
if [[ -z "$JOBID" ]]; then
  echo "ERROR: could not determine allocated JOBID from salloc output"; exit 1
fi
echo "=== allocated JOBID=$JOBID ==="

trap 'echo "=== releasing allocation JOBID=$JOBID (trap) ==="; scancel "$JOBID" 2>/dev/null || true' EXIT

RC=0
srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 --time=60 \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:$REPO_SRC:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export CUDA_VISIBLE_DEVICES=0
    export JAX_ENABLE_X64=1
    cd $HERE

    echo '=== Run C step 1: dspl_ratio_coords.py --run (MAP + MCLMC, GPU) ==='
    t0=\$(date +%s)
    /usr/bin/python3 dspl_ratio_coords.py --run --confirm-run-c-approved
    t1=\$(date +%s)
    echo \"[wall] dspl_ratio_coords.py --run = \$((t1 - t0)) s\"

    echo '=== Run C step 2: dspl_ratio_coords_analysis.py (pre-registered analysis) ==='
    /usr/bin/python3 dspl_ratio_coords_analysis.py
    t2=\$(date +%s)
    echo \"[wall] analysis = \$((t2 - t1)) s ; TOTAL = \$((t2 - t0)) s\"
  " || RC=$?

echo "=== run_dspl_ratio_coords EXIT status = $RC ==="
echo "=== outputs: ~/GIGALens-Code/results/sample_cosmology/dspl_ratio_coords/ ==="
exit $RC
