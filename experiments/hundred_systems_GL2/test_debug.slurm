#!/bin/bash -l
#SBATCH --job-name=gl2_simtest
#SBATCH --account=deepsrch_g
#SBATCH --constraint=gpu
#SBATCH --qos=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --array=0-4
#SBATCH --output=/global/homes/l/linusu/GIGALens-Code/slurm_logs/gl2_simtest_%A_%a.out

# Quick debug test: run 5 systems (array tasks 0-4) from the hundred_sersic_debug
# campaign. Reduced parameters (map_n_samples=200) fit comfortably on 1 GPU.
# Each task processes exactly 1 system via --shard i/100.

set -euo pipefail

CAMPAIGN_YAML=/global/homes/l/linusu/GIGALens-Code/experiments/hundred_systems_GL2/campaign_debug.yaml

SHARD_I="${SLURM_ARRAY_TASK_ID}"
SHARD_N=100

echo "=== GL2 simtest debug ==="
echo "Job:   $SLURM_JOB_ID  task $SHARD_I/$SHARD_N"
echo "Node:  $(hostname)"
echo "GPU:   $CUDA_VISIBLE_DEVICES"
echo "Time:  $(date -u)"

srun -u --cpu-bind=cores --gpu-bind=none \
  shifter --module=gpu,nccl-plugin \
          --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
          bash -lc '
    set -euo pipefail
    export PYTHONPATH="/global/homes/l/linusu/sidecar_jax_upgrade:$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages"
    export PYTHONPATH="$HOME/gigalens/src:$HOME/GIGALens-Code/src:$PYTHONPATH"

    echo "Python: $(python --version)"
    echo "Shard:  '"${SHARD_I}/${SHARD_N}"'"
    nvidia-smi -L || true

    python -m gigalens_research.simtests run \
        '"$CAMPAIGN_YAML"' \
        --shard '"${SHARD_I}/${SHARD_N}"'

    echo "Done at $(date -u)"
  '
