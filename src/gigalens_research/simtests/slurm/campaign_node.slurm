#!/bin/bash -l
#SBATCH --job-name=simtests_node
#SBATCH --account=deepsrch_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm_logs/simtests_node_%j.out

# Single-node, whole-campaign run
# ===============================
# One node (4 GPUs) runs ALL of a campaign's (system × sweep) runs sequentially;
# each run uses all 4 GPUs (gigalens shards the MCLMC chains across the device
# mesh). Idempotent + resume-friendly: re-submitting skips runs whose run.json
# already records status=ok, so if the 24h window isn't enough just `sbatch`
# again to continue from the content-addressed cache.
#
#   sbatch --export=ALL,CAMPAIGN_YAML=<abs path to campaign.yaml> campaign_node.slurm

set -euo pipefail

: "${CAMPAIGN_YAML:?set CAMPAIGN_YAML=/abs/path/to/campaign.yaml}"
: "${OUTPUT_DIR:=}"          # empty = use output_dir from campaign YAML
: "${EXTRA_ARGS:=}"          # extra args passed to `simtests run`

mkdir -p "$HOME/slurm_logs"

echo "Job:      $SLURM_JOB_ID"
echo "Campaign: $CAMPAIGN_YAML"
echo "Node:     $(hostname)"

export CAMPAIGN_YAML OUTPUT_DIR EXTRA_ARGS

srun -u --cpu-bind=cores --gpu-bind=none \
  shifter --module=gpu,nccl-plugin \
          --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
          bash -lc '
    set -euo pipefail
    export PYTHONPATH="/global/homes/l/linusu/sidecar_jax_upgrade:$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages"
    export PYTHONPATH="$HOME/gigalens/src:$HOME/GIGALens-Code/src:$PYTHONPATH"
    echo "Python:   $(python --version)"
    nvidia-smi -L || true

    OUTPUT_ARG=""
    if [ -n "$OUTPUT_DIR" ]; then
        OUTPUT_ARG="--output-dir $OUTPUT_DIR"
    fi

    # Idempotent dataset adapt (skips if already generated), then run every
    # (system, sweep) on this one node. --shard 0/1 = all runs, sequentially.
    python -m gigalens_research.simtests generate "$CAMPAIGN_YAML" $OUTPUT_ARG
    python -m gigalens_research.simtests run "$CAMPAIGN_YAML" --shard 0/1 $OUTPUT_ARG $EXTRA_ARGS
    python -m gigalens_research.simtests status "$CAMPAIGN_YAML" $OUTPUT_ARG
  '
