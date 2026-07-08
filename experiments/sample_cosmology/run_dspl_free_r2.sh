#!/bin/bash
# Run A (free-r2 reparameterization) launcher: interactive GPU allocation on
# Perlmutter, matching this repo's established idiom for a single interactive
# GPU node (cf. experiments/why_hard_to_sample/slurm/run_t28.sh) and the
# canonical Shifter image / PYTHONPATH from docs/env_setup.md.
#
# DO NOT RUN until the grader has approved the design checkpoint ("Run A") in
# docs/logs/sample-cosmology-dspl.md AND the pre-registered equivalence gate
# (dspl_r2_equivalence_gate.py) has been run and PASSED -- as of this writing
# that gate PASSED (max rel diff ~1.5e-14 over 64 matched draws; see
# results/sample_cosmology/dspl_free_r2/equivalence_gate.json), but the
# sampler itself is still awaiting grader approval per the pre-run-checklist.
#
# ASSUMPTIONS FLAGGED FOR REVIEW (copied from run_t28.sh's idiom -- this repo
# has no prior sample_cosmology-specific GPU launcher to copy from instead):
#   -A deepsrch_g       : NERSC account/repo charged. Confirm this is still
#                         the right one for this work before running.
#   -q interactive       : interactive QOS (short wall-clock, immediate start,
#                         no queue wait). Switch to a batch QOS (`-q regular`
#                         + sbatch) if this should NOT be interactive.
#   -C "gpu&hbm80g"       : any Perlmutter GPU node with 80GB HBM. Not load
#                         bearing for this small system (100x100 pixel grid,
#                         21/20-param model) -- a plain "gpu" constraint would
#                         likely also work; kept for parity with the repo's
#                         other interactive GPU runs.
#   -t 60                : wall-clock minutes. The baseline notebook's
#                         (Om0,w0)-parameterized run (same MAP steps, same
#                         8x10000+10000 MCLMC config) took ~130s (MAP) + ~131s
#                         (MCLMC) = ~5 minutes end-to-end on GPU; 60 minutes is
#                         a large (>10x) margin, not a measured estimate for
#                         THIS (r2) parameterization specifically.
#
# Usage:  ./run_dspl_free_r2.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== allocating 1x hbm80g GPU (interactive, deepsrch_g) ==="
ALLOC_OUT="$(salloc --no-shell -N1 -C "gpu&hbm80g" -G 1 -q interactive -A deepsrch_g -t 60 2>&1)"
echo "$ALLOC_OUT"
JOBID="$(printf '%s\n' "$ALLOC_OUT" | grep -oE 'Granted job allocation [0-9]+' | grep -oE '[0-9]+' | tail -1)"
if [[ -z "$JOBID" ]]; then
  echo "ERROR: could not determine allocated JOBID from salloc output"; exit 1
fi
echo "=== allocated JOBID=$JOBID ==="

# release the allocation on ANY exit (success, error, or Ctrl-C)
trap 'echo "=== releasing allocation JOBID=$JOBID (trap) ==="; scancel "$JOBID" 2>/dev/null || true' EXIT

RC=0
srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 --time=60 \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export CUDA_VISIBLE_DEVICES=0
    export JAX_ENABLE_X64=1
    cd $HERE

    echo '=== Run A step 1: dspl_free_r2.py --run (MAP + MCLMC, GPU) ==='
    t0=\$(date +%s)
    /usr/bin/python3 dspl_free_r2.py --run
    t1=\$(date +%s)
    echo \"[wall] dspl_free_r2.py --run = \$((t1 - t0)) s\"

    echo '=== Run A step 2: dspl_r2_reconstruct.py (post-run analysis) ==='
    /usr/bin/python3 dspl_r2_reconstruct.py
    t2=\$(date +%s)
    echo \"[wall] dspl_r2_reconstruct.py = \$((t2 - t1)) s ; TOTAL = \$((t2 - t0)) s\"
  " || RC=$?

echo "=== run_dspl_free_r2 EXIT status = $RC ==="
echo "=== outputs: ~/GIGALens-Code/results/sample_cosmology/dspl_free_r2/ ==="
exit $RC
