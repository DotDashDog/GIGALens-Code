#!/bin/bash
# T13' -- re-simulated sys60 (data ss16) x model-fidelity 2x2 launcher.
# Follows the run_t3.sh pattern: same shifter image, PYTHONPATH, container python;
# runs the whole T13' sequence in ONE srun/shifter shell. `set -euo pipefail` so a
# GATE-A (or any) failure ABORTS everything (t13_resim exits nonzero on gate failure).
#
# Sequence:
#   1) t13_resim               -> resim/sys60_ss16/observed_ss16.npz (GATES A + B)
#   2) arm ss2 pipeline (MAP->SVI on d')
#   3) arm ss2 mclmc seeds 1..4
#   4) arm ss4 pipeline
#   5) arm ss4 mclmc seeds 1..4
#   6) summaries (ss2, ss4)
#   7) comb identity check (ss2 model on d')  [Step 6]
#
# STRICT SEPARATION: every product is written under $OUT_DIR (resim/sys60_ss16);
# data/, results/, and the MAIN checkout are never written.
#
# Usage:
#   ./run_t13.sh <JOBID> <CUDA_VISIBLE_DEVICES> [OUT_DIR]
# e.g.
#   ./run_t13.sh 55408000 0 ./resim/sys60_ss16
set -euo pipefail

JOBID="${1:?need JOBID as arg 1}"
CUDA="${2:?need CUDA_VISIBLE_DEVICES as arg 2}"
OUT_DIR="${3:-./resim/sys60_ss16}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SYS60="$HERE/systems/sys60"
SYS60_SS16="$HERE/systems/sys60_ss16data"
OLD_T10="$HERE/results_t0t1/sys60/t10"
OLD_RUN="/global/homes/l/linusu/GIGALens-Code/results/testsys60/mclmc"

srun --overlap --jobid="$JOBID" --ntasks=1 --gpus=1 \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
  bash -lc "
    set -euo pipefail
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export CUDA_VISIBLE_DEVICES=$CUDA
    export JAX_ENABLE_X64=1
    cd $HERE

    echo '=== T13 Step 0-3: re-simulate + GATES A/B (aborts on failure) ==='
    /usr/bin/python3 t13_resim.py --data-dir '$SYS60' --out-dir '$OUT_DIR'

    echo '=== T13 Step 4: arm ss2 pipeline (MAP->SVI on d-prime) ==='
    /usr/bin/python3 t13_arms.py --stage pipeline --arm ss2 \
      --out-dir '$OUT_DIR' --data-dir '$SYS60_SS16'

    for S in 1 2 3 4; do
      echo \"=== T13 Step 5: arm ss2 mclmc seed \$S ===\"
      /usr/bin/python3 t13_arms.py --stage mclmc --arm ss2 --seed \$S \
        --out-dir '$OUT_DIR' --data-dir '$SYS60_SS16'
    done

    echo '=== T13 Step 4: arm ss4 pipeline (MAP->SVI on d-prime) ==='
    /usr/bin/python3 t13_arms.py --stage pipeline --arm ss4 --map-samples 50 --svi-nvi 125 \
      --out-dir '$OUT_DIR' --data-dir '$SYS60_SS16'

    for S in 1 2 3 4; do
      echo \"=== T13 Step 5: arm ss4 mclmc seed \$S ===\"
      /usr/bin/python3 t13_arms.py --stage mclmc --arm ss4 --seed \$S \
        --out-dir '$OUT_DIR' --data-dir '$SYS60_SS16'
    done

    echo '=== T13 Step 5: summaries ==='
    /usr/bin/python3 t13_arms.py --stage summary --arm ss2 \
      --out-dir '$OUT_DIR' --data-dir '$SYS60_SS16'
    /usr/bin/python3 t13_arms.py --stage summary --arm ss4 \
      --out-dir '$OUT_DIR' --data-dir '$SYS60_SS16'

    echo '=== T13 Step 6: comb identity check (ss2 model on d-prime) ==='
    /usr/bin/python3 t13_arms.py --stage comb \
      --out-dir '$OUT_DIR' --data-dir '$SYS60_SS16' \
      --old-t10-dir '$OLD_T10' --old-run-dir '$OLD_RUN'

    echo '=== T13 sequence complete ==='
  "
