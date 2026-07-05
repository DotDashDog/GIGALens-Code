#!/bin/bash
# T25 + T26 -- Rs conditional-width profile + Route A/B transform derivation
# (T25) and the acceptance battery on BOTH routes (T26). Shares ONE hbm80g
# interactive GPU allocation. Registered 2026-07-03 (docs/logs/why-hard-to-sample.md).
#
# Follows run_t23_t24.sh's shifter image / PYTHONPATH ordering / env EXACTLY.
# Output dirs are HARDCODED in the python scripts (results_carousel/phaseC/t25,
# .../t26) -- NO out-dir is ever passed on the CLI (the run_t21.sh "0" mishap is
# designed out here as in run_t23_t24.sh).
#
# Flow inside the single allocation:
#   1. t25_profile_gpu.py                  (lambda_Rs profile; writes profile.npz)
#   2. t25_transforms.py                   (derive Route A/B + FAMILY GATES)
#        - its exit code GATES continuation: nonzero (F-T25a / profile missing /
#          NO route passes gates) => block ALL of T26 and exit.
#        - per-route gate failures skip ONLY that route (routes_passed.json).
#   3. t26_battery_gpu.py --route R --limit 200   (SMOKE, each passing route)
#   4. t26_battery_gpu.py --route R               (FULL battery, each passing route)
#   5. scancel; echo exit status.
# conv float64 is the phase standard (WHTS_CONV_PRECISION=float64).
#
# Cost estimate (hbm80g):
#   T25: ~96 on-floor + ~128 below-floor + 1 z_init HVPs (~225) + derivations
#        + 2 routes x family gates (~150 renders) -> ~15-20 min GPU.
#   T26 per route: 3 real (8x2000/2000) + clone fit + 1 clone run.
#        ~ the T21 new-arm wall per run; 2 routes x (3+1) runs + 2 smoke.
#        -> ~2-2.5 h for both routes; well within -t 230.
#
# Usage:  ./run_t25_t26.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "=== allocating 1x hbm80g GPU (interactive, deepsrch_g) ==="
ALLOC_OUT="$(salloc --no-shell -N1 -C "gpu&hbm80g" -G 1 -q interactive -A deepsrch_g -t 230 2>&1)"
echo "$ALLOC_OUT"
JOBID="$(printf '%s\n' "$ALLOC_OUT" | grep -oE 'Granted job allocation [0-9]+' | grep -oE '[0-9]+' | tail -1)"
if [[ -z "$JOBID" ]]; then
  echo "ERROR: could not determine allocated JOBID from salloc output"; exit 1
fi
echo "=== allocated JOBID=$JOBID ==="

RC=0
srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 --time=230 \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export CUDA_VISIBLE_DEVICES=0
    export JAX_ENABLE_X64=1
    export WHTS_CONV_PRECISION=float64
    cd $HERE

    echo '=== SMOKE: T25 profile (--limit 3) + transforms (--no-gates guard off) ==='
    /usr/bin/python3 t25_profile_gpu.py --limit 3
    echo '=== SMOKE OK (profile wiring) ==='

    echo '=== T25a: Rs conditional-width profile (conv float64) ==='
    t0=\$(date +%s)
    /usr/bin/python3 t25_profile_gpu.py
    t1=\$(date +%s)
    echo \"[wall] T25 profile = \$((t1 - t0)) s\"

    echo '=== T25b: derive Route A/B transforms + FAMILY GATES ==='
    set +e
    /usr/bin/python3 t25_transforms.py
    T25RC=\$?
    set -e
    echo \"[t25_transforms exit] = \$T25RC\"
    if [ \$T25RC -ne 0 ]; then
      echo \"*** t25_transforms exit \$T25RC (F-T25a / profile missing / no route passed gates)\"
      echo \"*** BLOCKING ALL of T26 per the pre-registered gate. ***\"
      exit \$T25RC
    fi

    PASSING=\$(/usr/bin/python3 -c \"import json;d=json.load(open('results_carousel/phaseC/t25/routes_passed.json'));print(' '.join(r for r in ('A','B') if d.get(r)))\")
    echo \"=== routes passing family gates: [\$PASSING] ===\"
    if [ -z \"\$PASSING\" ]; then
      echo '*** no route passed the family gates -> nothing to run in T26 ***'
      exit 5
    fi

    for R in \$PASSING; do
      echo \"=== T26 SMOKE route \$R (--limit 200) ===\"
      /usr/bin/python3 t26_battery_gpu.py --route \$R --limit 200
    done
    echo '=== T26 SMOKE OK ==='

    t2=\$(date +%s)
    for R in \$PASSING; do
      echo \"=== T26 FULL battery route \$R (8x2000/2000, seeds 1,2,3 + clone) ===\"
      /usr/bin/python3 t26_battery_gpu.py --route \$R
    done
    t3=\$(date +%s)
    echo \"[wall] T26 full = \$((t3 - t2)) s ; TOTAL = \$((t3 - t0)) s\"
  " || RC=$?

echo "=== releasing allocation JOBID=$JOBID ==="
scancel "$JOBID" || true
echo "=== run_t25_t26 EXIT status = $RC ==="
echo "=== analyze on the login node with:"
echo "    /global/homes/l/linusu/.conda/envs/gigalens_multinode_env/bin/python t26_analyze.py"
exit $RC
