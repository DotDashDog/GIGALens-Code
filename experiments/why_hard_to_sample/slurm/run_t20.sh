#!/bin/bash
# T20 -- carousel minimal-case log-posterior MICRO-TEXTURE probe (Phase C).
#
# Runs, against an already-allocated JOBID (srun --overlap), each step in its own
# srun with its own --time (total budget <= 2h; t3 sys60 runs were ~10-20 min):
#   (1) t20_prepare_inputs.py  -> inputs/{old,new}_samples.npz (position<-samples_z)
#   (2) t3 (via t20_run_t3.py) carousel_min_old  float64 -> t3_old_f64
#   (3) t3 (via t20_run_t3.py) carousel_min_new  float64 -> t3_new_f64
#   (4) t3 float32 CONTROL on old (JAX_ENABLE_X64=0 + WHTS_FLOAT32_CONTROL=1
#       + --allow-float32) -> t3_old_f32   [positive control; noise floor MUST blow up]
#   (5) t20_step_segments.py --arm both -> t20/  (in-step segments + micro-transects)
#
# t20_run_t3.py is a NON-EDITING wrapper: it overrides t3's sys60-specific
# WORST_ESS_PARAMS (which contains .../gamma, ABSENT on the NFW carousel arms, so
# stock t3 would AssertionError) with carousel axes. See its docstring.
#
# jax precision is PROCESS-GLOBAL, so the float32 control is a SEPARATE python
# process (fresh interpreter) with JAX_ENABLE_X64=0.
#
# Usage:
#   ./run_t20.sh <JOBID> <CUDA> [OUT_BASE]
# e.g.
#   ./run_t20.sh 55420000 0 ./results_carousel/phaseC/t20
set -euo pipefail

JOBID="${1:?need JOBID as arg 1}"
CUDA="${2:?need CUDA_VISIBLE_DEVICES as arg 2 (informational; see note below)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_BASE="${3:-$HERE/results_carousel/phaseC/t20}"

SEED=20260703
IMG="docker:ghcr.io/nvidia/jax:jax-2026-04-13"
PYPATH="/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages"

# Reference (READ-ONLY) diagnostics -- already carry step_size+L in t3's layout.
REFDIAG_OLD="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/minimal_case_oldbij/mclmc/diagnostics.npz"
REFDIAG_NEW="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/minimal_case_newbij/mclmc/diagnostics.npz"
CLONE_OLD="$HERE/results_carousel/old/t1/clone.npz"
CLONE_NEW="$HERE/results_carousel/new/t1/clone.npz"
SYS_OLD="$HERE/systems/carousel_min_old"
SYS_NEW="$HERE/systems/carousel_min_new"
SAMP_OLD="$OUT_BASE/inputs/old_samples.npz"
SAMP_NEW="$OUT_BASE/inputs/new_samples.npz"

# NOTE (a prior run failed on this): inside `srun --gpus=1` exactly ONE GPU is
# bound to the step and it ALWAYS appears as device 0, regardless of the $CUDA
# arg. So we export CUDA_VISIBLE_DEVICES=0 (NOT $CUDA). $CUDA is kept only for
# interface parity with the sibling run_t*.sh launchers.
GPUENV="export CUDA_VISIBLE_DEVICES=0"   # must be 0 under srun --gpus=1

# ---- (1) prepare t3 input adapters (numpy-only; small) --------------------
srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 --time=5 \
  shifter --module=gpu,nccl-plugin --image="$IMG" bash -lc "
    set -euo pipefail
    export PYTHONPATH=$PYPATH
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    $GPUENV
    cd $HERE
    echo '=== T20 (1) prepare inputs ==='
    /usr/bin/python3 t20_prepare_inputs.py --arm both --out-dir '$OUT_BASE/inputs'
  "

# ---- (2) t3 carousel_min_old float64 --------------------------------------
srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 --time=30 \
  shifter --module=gpu,nccl-plugin --image="$IMG" bash -lc "
    set -euo pipefail
    export PYTHONPATH=$PYPATH
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    $GPUENV
    export JAX_ENABLE_X64=1
    cd $HERE
    echo '=== T20 (2) t3 old float64 ==='
    /usr/bin/python3 t20_run_t3.py \
      --data-dir '$SYS_OLD' --samples '$SAMP_OLD' --clone '$CLONE_OLD' \
      --ref-diagnostics '$REFDIAG_OLD' --out-dir '$OUT_BASE/t3_old_f64' --seed $SEED
  "

# ---- (3) t3 carousel_min_new float64 --------------------------------------
srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 --time=30 \
  shifter --module=gpu,nccl-plugin --image="$IMG" bash -lc "
    set -euo pipefail
    export PYTHONPATH=$PYPATH
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    $GPUENV
    export JAX_ENABLE_X64=1
    cd $HERE
    echo '=== T20 (3) t3 new float64 ==='
    /usr/bin/python3 t20_run_t3.py \
      --data-dir '$SYS_NEW' --samples '$SAMP_NEW' --clone '$CLONE_NEW' \
      --ref-diagnostics '$REFDIAG_NEW' --out-dir '$OUT_BASE/t3_new_f64' --seed $SEED
  "

# ---- (4) t3 float32 CONTROL on old arm ------------------------------------
srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 --time=20 \
  shifter --module=gpu,nccl-plugin --image="$IMG" bash -lc "
    set -euo pipefail
    export PYTHONPATH=$PYPATH
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    $GPUENV
    export JAX_ENABLE_X64=0
    export WHTS_FLOAT32_CONTROL=1
    cd $HERE
    echo '=== T20 (4) t3 old float32 CONTROL ==='
    /usr/bin/python3 t20_run_t3.py \
      --data-dir '$SYS_OLD' --samples '$SAMP_OLD' --clone '$CLONE_OLD' \
      --ref-diagnostics '$REFDIAG_OLD' --out-dir '$OUT_BASE/t3_old_f32' \
      --seed $SEED --allow-float32
  "

# ---- (5) t20 in-step segments + micro-transects (both arms, float64) ------
srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 --time=30 \
  shifter --module=gpu,nccl-plugin --image="$IMG" bash -lc "
    set -euo pipefail
    export PYTHONPATH=$PYPATH
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    $GPUENV
    export JAX_ENABLE_X64=1
    cd $HERE
    echo '=== T20 (5) step-segments + micro-transects ==='
    /usr/bin/python3 t20_step_segments.py --arm both --out-dir '$OUT_BASE'
  "

echo '[run_t20] all steps submitted/completed.'
