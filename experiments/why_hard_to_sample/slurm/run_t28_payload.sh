#!/bin/bash
# T28 payload: the srun sequence from run_t28.sh, but against an EXISTING
# allocation (JOBID passed as $1). Written 2026-07-04 after the launcher's
# salloc-wait phase proved fragile under the harness (three killed launchers;
# see the log's T28 gate-correction + relaunch notes): the allocation request
# is now made/monitored separately and this payload only needs a granted JOBID.
# scancel on exit is retained (trap) so the allocation never outlives the run.
#
# Usage:  ./run_t28_payload.sh <JOBID>
set -euo pipefail

JOBID="${1:?usage: run_t28_payload.sh <JOBID>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

trap 'echo "=== releasing allocation JOBID=$JOBID (trap) ==="; scancel "$JOBID" 2>/dev/null || true' EXIT

RC=0
srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 \
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
exit $RC
