#!/bin/bash -l
#SBATCH --job-name=ps_v3_debug
#SBATCH --account=deepsrch_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=/global/homes/l/linusu/slurm_logs/ps_v3_debug_%j.out
# Three-system pilot of the v3 point-source SBC arms on the multiplicity-term
# PR branches (gigalens PR #126, GIGALens-Code PR #29), NOT the merged mains:
# PYTHONPATH puts the two worktrees' src ahead of the editable-install .pth.
# DEBUG-QOS variant (30 min): the reduced-budget 3-system double pilot
# (campaign_v3_debug_double.yaml). RUN_TESTS=1 also runs the container unit
# tests + GPU cost benchmark first (done 2026-09-04: 12 passed; term costs
# +10 ms/step at 8 chains, +47 ms at 1000 particles on A100).
set -euo pipefail
GL=/global/u1/l/linusu/gigalens/.claude/worktrees/multiplicity-term
GLC=/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/multiplicity-term
EXP=$GLC/experiments/hundred_point_sources
export GL GLC EXP RUN_TESTS
echo "Job $SLURM_JOB_ID on $(hostname), $(date)"
srun -u --cpu-bind=cores --gpu-bind=none \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 bash -lc '
    set -euo pipefail
    export PYTHONPATH="$GL/src:$GLC/src:/global/homes/l/linusu/sidecar_jax_upgrade:$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages"
    export MPLBACKEND=Agg
    PY=/usr/bin/python3
    $PY -c "import jax, gigalens, gigalens_research; print(jax.__version__, jax.devices()); print(gigalens.__file__); print(gigalens_research.__file__)"
    nvidia-smi -L || true
    if [ "${RUN_TESTS:-0}" = "1" ]; then
      echo "=== [1] unit tests (CPU, as the files pin) $(date)"
      cd $GL && $PY -m pytest -q tests/unit/test_point_source_multiplicity.py 2>&1 | tail -3
      cd $GLC && $PY -m pytest -q src/gigalens_research/simtests/tests/multiplicity_selection_test.py 2>&1 | tail -3
      echo "=== [2] GPU cost benchmark $(date)"
      $PY $EXP/bench_count_gpu.py
    fi
    Y=$EXP/campaign_v3_debug_double.yaml
    echo "=== [3] debug double pilot: generate $(date)"
    $PY -m gigalens_research.simtests generate "$Y"
    echo "=== [3] debug double pilot: run $(date)"
    $PY -m gigalens_research.simtests run "$Y" --shard 0/1
    echo "=== [3] debug double pilot: status $(date)"
    $PY -m gigalens_research.simtests status "$Y"
    echo "=== done $(date)"
  '
