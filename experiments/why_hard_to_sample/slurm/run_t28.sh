#!/bin/bash
# T28 -- observable-slope prior (s ~ Uniform(0, 0.75)) swap: implementation gates
# + 3-seed standard MCLMC. Registered 2026-07-04 (docs/logs/why-hard-to-sample.md).
#
# Follows run_t25_t26.sh's shifter image / PYTHONPATH / env EXACTLY (gigalens comes
# from /global/homes/l/linusu/gigalens/src -- the worktree has no src/ of its own;
# the main-checkout src path is kept identical to the verified T25/T26 runs). Output
# dir is HARDCODED in the python (results_carousel/phaseC/t28); NO out-dir on the CLI.
#
# Flow inside ONE hbm80g interactive allocation:
#   0. t28_sprior_transform.py           (build Rs(s) leaf + login gates; numpy)
#   1. t28_run_gpu.py --limit 200        (GATES + seed-1 SMOKE, end-to-end wiring)
#        - nonzero exit (any hard gate fails) => ABORT before the full battery.
#   2. t28_run_gpu.py                     (GATES re-check + FULL 3 seeds {1,2,3})
#   3. scancel (also on error via trap).
# conv float64 is the phase standard (WHTS_CONV_PRECISION=float64), as in T26.
#
# Cost estimate (hbm80g): gates ~a few renders (<2 min); smoke seed-1 200/200
# (~2-3 min); 3 real 8x2000/2000 runs ~ the T21 new-arm wall each (~15-20 min) ->
# TOTAL ~60-75 min. -t 120 has ample margin.
#
# Usage:  ./run_t28.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== allocating 1x hbm80g GPU (interactive, deepsrch_g) ==="
ALLOC_OUT="$(salloc --no-shell -N1 -C "gpu&hbm80g" -G 1 -q interactive -A deepsrch_g -t 120 2>&1)"
echo "$ALLOC_OUT"
JOBID="$(printf '%s\n' "$ALLOC_OUT" | grep -oE 'Granted job allocation [0-9]+' | grep -oE '[0-9]+' | tail -1)"
if [[ -z "$JOBID" ]]; then
  echo "ERROR: could not determine allocated JOBID from salloc output"; exit 1
fi
echo "=== allocated JOBID=$JOBID ==="

# release the allocation on ANY exit (success, error, or Ctrl-C)
trap 'echo "=== releasing allocation JOBID=$JOBID (trap) ==="; scancel "$JOBID" 2>/dev/null || true' EXIT

RC=0
srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 --time=120 \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export CUDA_VISIBLE_DEVICES=0
    export JAX_ENABLE_X64=1
    export WHTS_CONV_PRECISION=float64
    cd $HERE

    echo '=== T28.0: build Rs(s) leaf + login gates (numpy) ==='
    /usr/bin/python3 t28_sprior_transform.py

    echo '=== T28.1: GATES + seed-1 SMOKE (--limit 200) ==='
    t0=\$(date +%s)
    /usr/bin/python3 t28_run_gpu.py --limit 200
    t1=\$(date +%s)
    echo \"[wall] T28 gates+smoke = \$((t1 - t0)) s\"
    echo '=== T28 SMOKE OK (gates passed, end-to-end wiring verified) ==='

    echo '=== T28.2: FULL 3 seeds {1,2,3} (8x2000/2000, dev 5e-4, conv f64) ==='
    /usr/bin/python3 t28_run_gpu.py
    t2=\$(date +%s)
    echo \"[wall] T28 full = \$((t2 - t1)) s ; TOTAL = \$((t2 - t0)) s\"
  " || RC=$?

echo "=== run_t28 EXIT status = $RC ==="
echo "=== analyze on the login node with:"
echo "    /global/homes/l/linusu/.conda/envs/gigalens_multinode_env/bin/python t28_analyze.py"
exit $RC
