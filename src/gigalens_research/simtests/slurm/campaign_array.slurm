#!/bin/bash -l
#SBATCH --job-name=simtests
#SBATCH --account=deepsrch_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --array=0-99                     # override: --array=0-<N-1>%<max_concurrent>
#SBATCH --output=slurm_logs/simtests_%A_%a.out

# Simtests campaign array job
# ===========================
# Sharding is over SYSTEMS: each array task owns a slice of the systems and
# runs ALL sweep points of each owned system sequentially (this lets pipeline
# variants share the per-system trunk MAP/SVI race-free).
# The --array index maps to --shard $SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT.
#
# Typical submission:
#   N=100   # total number of SYSTEMS (not systems × sweep points)
#   sbatch --array=0-$((N-1))%32 campaign_array.slurm
#
# Override knobs at submit time:
#   sbatch --export=ALL,CAMPAIGN_YAML=<path>,OUTPUT_DIR=<dir> campaign_array.slurm

set -euo pipefail

# Tunable knobs (override with --export=ALL,VAR=value).
: "${CAMPAIGN_YAML:=campaign.yaml}"
: "${OUTPUT_DIR:=}"          # empty = use output_dir from campaign YAML
: "${EXTRA_ARGS:=}"          # any extra args passed to `simtests run`

mkdir -p "$HOME/slurm_logs"

# Each array task processes a 1/N slice of all runs.
SHARD_I="${SLURM_ARRAY_TASK_ID}"
SHARD_N="${SLURM_ARRAY_TASK_COUNT}"
SHARD="${SHARD_I}/${SHARD_N}"

echo "Job:      $SLURM_JOB_ID  task $SHARD_I of $SHARD_N"
echo "Campaign: $CAMPAIGN_YAML"
echo "Shard:    $SHARD"
echo "Node:     $(hostname)"

export CAMPAIGN_YAML OUTPUT_DIR SHARD EXTRA_ARGS

srun -u --cpu-bind=cores --gpu-bind=none \
  shifter --module=gpu,nccl-plugin \
          --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
          bash -lc '
    set -euo pipefail
    export PYTHONPATH="/global/homes/l/linusu/sidecar_jax_upgrade:$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages"
    export PYTHONPATH="$HOME/gigalens/src:$HOME/GIGALens-Code/src:$PYTHONPATH"
    echo "Python:   $(python --version)"
    echo "Shard:    $SHARD"
    nvidia-smi -L || true

    OUTPUT_ARG=""
    if [ -n "$OUTPUT_DIR" ]; then
        OUTPUT_ARG="--output-dir $OUTPUT_DIR"
    fi

    python -m gigalens_research.simtests run \
        "$CAMPAIGN_YAML" \
        --shard "$SHARD" \
        $OUTPUT_ARG \
        $EXTRA_ARGS
  '
