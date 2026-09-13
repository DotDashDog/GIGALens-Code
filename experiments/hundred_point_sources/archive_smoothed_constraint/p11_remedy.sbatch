#!/bin/bash
# P-11 fine-operator remedy (docs/logs/point-source-sbc.md, P-11 checkpoint).
# One es0 node, 4x GRTX2080TI (11 GB). Memory case: P-10 measured
# peak_gpu_bytes = 1.35 GB on every system at chunk 7 (56 batched chains);
# the fine config keeps per-tile quadrature size identical (768^2/64 =
# 384^2/16 = 9216 points) and runs chunk 1 (8 chains), so the peak is
# strictly below the measured 1.35 GB. Host: P-10 MaxRSS 12.4 GB at 4
# concurrent slices < 90 GB requested.
#SBATCH -J p11_fine
#SBATCH -A ac_cosmo
#SBATCH -p es0
#SBATCH -q es_normal
#SBATCH --gres=gpu:GRTX2080TI:4
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=90G
#SBATCH -t 16:00:00
#SBATCH -o /global/scratch/users/linusupson/simtests_results/hundred_point_sources_v2_double/launch_mc/logs_p11/%x-%j.out

set -u
source /etc/profile.d/modules.sh
module load miniforge3/25.9.1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gigalens_modern_env
# Both repos pinned at the P-10 commits (GIGALens-Code 4358ebb, gigalens
# d321d3c): the live checkouts have moved (ModellingSequence migration), and
# the remedy comparison is only controlled if nothing but the operator
# resolution differs from mc1.
export PYTHONPATH=$HOME/GIGALens-Code-worktree-p10/src:$HOME/gigalens-worktree-p10/src
export GIGALENS_ALLOW_LEGACY_JAX=1
export XDG_CACHE_HOME=/global/scratch/users/linusupson/.cache
export CUDA_CACHE_PATH=/global/scratch/users/linusupson/.cache/nv
export PIP_CACHE_DIR=/global/scratch/users/linusupson/.cache/pip
export MPLCONFIGDIR=/global/scratch/users/linusupson/.cache/matplotlib

EXP=$HOME/GIGALens-Code/experiments/hundred_point_sources
YAML=$EXP/campaign_v2_double_mc_fine.yaml
ANL=$EXP/p11_analysis.py
LOGS=/global/scratch/users/linusupson/simtests_results/hundred_point_sources_v2_double/launch_mc/logs_p11
mkdir -p "$LOGS"

# CPU side deliverable, concurrent with the GPU phase: truth recount at
# search_window 12 for all 100 systems (generator counted within 6).
JAX_PLATFORMS=cpu nice -n 5 python "$ANL" recount > "$LOGS/recount.log" 2>&1 &
RECOUNT_PID=$!

# GPU slices: mc2 reruns, one system per invocation (targets are
# non-contiguous in the manifest), sequential within a GPU.
run_slice () {
  local gpu=$1; shift
  for off in "$@"; do
    CUDA_VISIBLE_DEVICES=$gpu python -m gigalens_research.simtests.experiments.batched_campaign_run \
      "$YAML" --offset "$off" --n-systems 1 --chunk 1 \
      > "$LOGS/mc2_off${off}.log" 2>&1 \
      || { echo "P11 SLICE $off FAILED"; return 1; }
    echo "P11 SLICE $off DONE"
  done
}
run_slice 0 57 35 & P0=$!
run_slice 1 86 38 & P1=$!
run_slice 2 99 67 & P2=$!
# GPU3 has the single shortest slice; it then runs the mc1 unpenalized
# loglik re-rank (all 100 systems) while GPUs 0-2 finish their seconds.
( run_slice 3 69 \
  && CUDA_VISIBLE_DEVICES=3 python "$ANL" rerank > "$LOGS/rerank.log" 2>&1 \
  && echo "P11 RERANK DONE" || echo "P11 RERANK FAILED" ) & P3=$!

FAIL=0
for p in $P0 $P1 $P2 $P3; do wait "$p" || FAIL=1; done
wait "$RECOUNT_PID" || { echo "P11 RECOUNT FAILED"; FAIL=1; }

# Post-rerun audit of the 7 mc2 posteriors (fine N_eff + 2x rung +
# lenstronomy cross-check + mc2 unpenalized loglik rank).
CUDA_VISIBLE_DEVICES=0 python "$ANL" audit > "$LOGS/audit.log" 2>&1 \
  || { echo "P11 AUDIT FAILED"; FAIL=1; }

[ "$FAIL" -eq 0 ] && echo "P11 JOB DONE" || echo "P11 JOB FAILED"
