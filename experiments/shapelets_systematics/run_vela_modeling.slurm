#!/bin/bash -l
#SBATCH --job-name=vela_modeling
#SBATCH --account=deepsrch_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --output=slurm_logs/vela_modeling_%j.out

# Single-node Vela modeling on the regular qos. Submits run_vela_modeling.py
# inside the canonical shifter container (matches the notebook kernel).
#
# Override at submit time, e.g.:
#   sbatch --export=ALL,SIM_NUMS="01 03",REPS="0",N_MAX=15 run_vela_modeling.slurm

set -euo pipefail

# Tunable knobs (override via --export=ALL,VAR=value at sbatch time).
: "${SIM_NUMS:=01 03 04 07 08 10 15 21 22 23 25 26}"
: "${REPS:=0 1 2 3 4}"
: "${N_MAX:=30}"
: "${SUPERSAMPLE:=1}"
: "${NUM_PIX:=200}"
: "${NUM_BURNIN:=2000}"
: "${NUM_RESULTS:=2000}"
: "${N_HMC:=8}"
# Set SKIP_EXISTING=1 to skip (sim_num, rep) pairs whose save_dir already
# has mclmc_samples.npy. Handy for resuming a sweep that died partway.
: "${SKIP_EXISTING:=0}"
: "${EXTRA_ARGS:=--no-progress-bar}"

if [ "${SKIP_EXISTING}" = "1" ] || [ "${SKIP_EXISTING}" = "true" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --skip-existing"
fi

cd /global/homes/l/linusu/GIGALens-Code/experiments/shapelets_systematics

mkdir -p "$HOME/slurm_logs"

# Pass our knobs through to the shifter shell verbatim.
export SIM_NUMS REPS N_MAX SUPERSAMPLE NUM_PIX NUM_BURNIN NUM_RESULTS N_HMC EXTRA_ARGS

srun -u --cpu-bind=cores --gpu-bind=none \
  shifter --module=gpu,nccl-plugin \
          --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
          bash -lc '
    set -euo pipefail
    export PYTHONPATH="/global/homes/l/linusu/sidecar_jax_upgrade:$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages"
    echo "Hostname: $(hostname)"
    nvidia-smi -L || true
    python run_vela_modeling.py \
        --sim-nums $SIM_NUMS \
        --reps $REPS \
        --n-max $N_MAX \
        --supersample $SUPERSAMPLE \
        --num-pix $NUM_PIX \
        --num-burnin-steps $NUM_BURNIN \
        --num-results $NUM_RESULTS \
        --n-hmc $N_HMC \
        $EXTRA_ARGS
  '
