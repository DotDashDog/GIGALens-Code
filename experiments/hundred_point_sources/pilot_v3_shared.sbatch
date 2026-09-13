#!/bin/bash -l
#SBATCH --job-name=ps_v3_pilot_sh
#SBATCH --account=deepsrch_g
#SBATCH --constraint=gpu
#SBATCH --qos=shared
#SBATCH --time=08:00:00
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --output=/global/homes/l/linusu/slurm_logs/ps_v3_pilot_sh_%A_%a.out
# Three-system pilot of the v3 point-source SBC arms on the multiplicity-term
# PR branches (gigalens PR #126, GIGALens-Code PR #29), NOT the merged mains:
# PYTHONPATH puts the two worktrees' src ahead of the editable-install .pth.
# SHARED-QOS variant (1 GPU per task; the solo sampler is launch-bound, so one
# A100 is as fast as four). A 3-task array: task i runs system i of each arm (--shard i/3) on its own
# node, because the solo sampler is launch-latency bound (~0.6 s per MAMS draw
# regardless of chain count) and 8x(5000+5000) draws is hours per system.
# Resume-friendly (re-submit continues). Tests/benchmark ran in the debug job.
set -euo pipefail
GL=/global/u1/l/linusu/gigalens/.claude/worktrees/multiplicity-term
GLC=/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/multiplicity-term
EXP=$GLC/experiments/hundred_point_sources
export GL GLC EXP SLURM_ARRAY_TASK_ID
echo "Job $SLURM_JOB_ID on $(hostname), $(date)"
srun -u --cpu-bind=cores --gpu-bind=none \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 bash -lc '
    set -euo pipefail
    export PYTHONPATH="$GL/src:$GLC/src:/global/homes/l/linusu/sidecar_jax_upgrade:$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages"
    export MPLBACKEND=Agg
    PY=/usr/bin/python3
    $PY -c "import jax, gigalens, gigalens_research; print(jax.__version__, jax.devices()); print(gigalens.__file__); print(gigalens_research.__file__)"
    nvidia-smi -L || true
    for ARM in double quad; do
      Y=$EXP/campaign_v3_pilot_$ARM.yaml
      echo "=== [3] $ARM pilot: generate $(date)"
      $PY -m gigalens_research.simtests generate "$Y"
      echo "=== [3] $ARM pilot: run $(date)"
      $PY -m gigalens_research.simtests run "$Y" --shard $SLURM_ARRAY_TASK_ID/3
      echo "=== [3] $ARM pilot: status/aggregate $(date)"
      $PY -m gigalens_research.simtests status "$Y"
    done
    echo "=== done $(date)"
  '
