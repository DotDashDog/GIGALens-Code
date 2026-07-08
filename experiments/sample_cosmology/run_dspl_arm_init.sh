#!/bin/bash
# Run B (arm-initialized frozen-metric MCLMC) launcher: interactive GPU
# allocation on Perlmutter, matching this repo's established idiom for THIS
# experiment family -- copied from the sibling `run_dspl_free_r2.sh` (Run A's
# launcher, same directory/campaign) rather than re-derived, so the
# account/queue/constraint/PYTHONPATH choices are consistent across the two
# pre-registered runs in `docs/logs/sample-cosmology-dspl.md`.
#
# DO NOT RUN the "map"/"all" modes until the grader has approved the Run B
# design checkpoint in docs/logs/sample-cosmology-dspl.md. `dspl_arm_init.py`
# itself enforces this at the script level (refuses to run without
# --confirm-run-b-approved); this launcher does NOT pass that flag for you --
# you must add it explicitly to CONFIRM you are the grader approving the run
# (see Usage below). The "toy" mode (mechanics validation only) has no such
# gate and needs no GPU allocation.
#
# ASSUMPTIONS FLAGGED FOR REVIEW (copied from run_dspl_free_r2.sh's idiom,
# itself copied from experiments/why_hard_to_sample/slurm/run_t28.sh -- this
# repo has no Run-B-specific measurement to derive these from independently):
#   -A deepsrch_g        : NERSC account/repo charged. Confirm this is still
#                          the right one before running.
#   -q interactive        : interactive QOS (short wall-clock, immediate
#                          start, no queue wait). Switch to `-q regular` +
#                          sbatch if this should NOT be interactive.
#   -C "gpu&hbm80g"        : any Perlmutter GPU node with 80GB HBM. Not load-
#                          bearing for this small system; kept for parity
#                          with Run A's launcher and the repo's other
#                          interactive GPU runs.
#   -G 1 / --gpus=1        : Stage 3 (`frozen_metric_mclmc` in
#                          dspl_arm_init.py) deliberately does NOT shard
#                          chains across devices (no adaptation state needs
#                          cross-device sync -- see that function's
#                          docstring), so requesting more than 1 GPU would not
#                          be used. This is a genuine difference from the
#                          baseline notebook's run, flagged rather than
#                          silently assumed equivalent in parallelism.
#   -t 30                 : wall-clock minutes. NOT a measured estimate for
#                          this parameterization -- the design checkpoint
#                          describes Run B as "small profile-MAP + one
#                          8-chain 10k-RESULT MCLMC run (~half the baseline)",
#                          and the baseline's full 8x10000-burnin+10000-result
#                          adapted run took ~131s on GPU (per
#                          dspl_cosmology_newapi.ipynb); Run B's Stage 3 has NO
#                          adaptation/burn-in at all (see mclmc.py note in
#                          dspl_arm_init.py), so it should be considerably
#                          cheaper per-step than that, plus the profile-MAP
#                          stage (~130s per the notebook's MAP timing). 30
#                          minutes is a large margin over that sum, not a
#                          tight bound.
#
# Usage:
#   ./run_dspl_arm_init.sh toy
#       CPU-only mechanics validation (no GPU allocation, no --confirm flag).
#   ./run_dspl_arm_init.sh map --confirm-run-b-approved
#       Stage 1 (profile-MAP) + Stage 2 (init assembly) only; prints z_init,
#       does not sample. Still gated (real compute).
#   ./run_dspl_arm_init.sh all --confirm-run-b-approved
#       Full Run B (Stages 1-3) + `dspl_arm_init_analysis.py` post-run
#       analysis, end to end. ONLY after grader approval.
set -euo pipefail

# NOTE: keep this usage string free of literal '}' — inside ${1:?...} an
# unescaped '}' terminates the expansion early and corrupts MODE.
MODE="${1:?Usage: run_dspl_arm_init.sh toy|map|all [--confirm-run-b-approved] [extra args]}"
shift || true
EXTRA_ARGS=("$@")

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONPATH_VAL="/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages"

if [[ "$MODE" == "toy" ]]; then
  echo "=== dspl_arm_init.py --run toy (CPU-only mechanics check; no GPU allocation) ==="
  shifter --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 bash -lc "
    set -euo pipefail
    export PYTHONPATH=${PYTHONPATH_VAL}
    export JAX_PLATFORMS=cpu
    export JAX_ENABLE_X64=1
    cd $HERE
    /usr/bin/python3 dspl_arm_init.py --run toy
  "
  exit $?
fi

if [[ "$MODE" != "map" && "$MODE" != "all" ]]; then
  echo "ERROR: unknown mode '$MODE' (expected toy|map|all)" >&2
  exit 1
fi

echo "=== allocating 1x hbm80g GPU (interactive, deepsrch_g) ==="
ALLOC_OUT="$(salloc --no-shell -N1 -C "gpu&hbm80g" -G 1 -q interactive -A deepsrch_g -t 30 2>&1)"
echo "$ALLOC_OUT"
JOBID="$(printf '%s\n' "$ALLOC_OUT" | grep -oE 'Granted job allocation [0-9]+' | grep -oE '[0-9]+' | tail -1)"
if [[ -z "$JOBID" ]]; then
  echo "ERROR: could not determine allocated JOBID from salloc output"; exit 1
fi
echo "=== allocated JOBID=$JOBID ==="

# release the allocation on ANY exit (success, error, or Ctrl-C)
trap 'echo "=== releasing allocation JOBID=$JOBID (trap) ==="; scancel "$JOBID" 2>/dev/null || true' EXIT

RC=0
srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 --time=30 \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH=${PYTHONPATH_VAL}
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export CUDA_VISIBLE_DEVICES=0
    export JAX_ENABLE_X64=1
    cd $HERE

    echo '=== Run B step 1: dspl_arm_init.py --run ${MODE} ${EXTRA_ARGS[*]} ==='
    t0=\$(date +%s)
    /usr/bin/python3 dspl_arm_init.py --run ${MODE} ${EXTRA_ARGS[*]}
    t1=\$(date +%s)
    echo \"[wall] dspl_arm_init.py --run ${MODE} = \$((t1 - t0)) s\"

    if [[ '${MODE}' == 'all' ]]; then
      echo '=== Run B step 2: dspl_arm_init_analysis.py (post-run analysis) ==='
      /usr/bin/python3 dspl_arm_init_analysis.py
      t2=\$(date +%s)
      echo \"[wall] dspl_arm_init_analysis.py = \$((t2 - t1)) s ; TOTAL = \$((t2 - t0)) s\"
    fi
  " || RC=$?

echo "=== run_dspl_arm_init ($MODE) EXIT status = $RC ==="
echo "=== outputs: ~/GIGALens-Code/results/sample_cosmology/dspl_arm_init/ ==="
exit $RC
