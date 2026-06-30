#!/bin/bash
JOBID=55099115
GPUS=${2:-1}
srun --overlap --jobid=$JOBID --ntasks=1 --gpus=$GPUS \
  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
  bash -lc "
    export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/homes/l/linusu/gigalens/src:/global/u1/l/linusu/GIGALens-Code/src:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export JAX_ENABLE_X64=1
    export TF_GPU_ALLOCATOR=cuda_malloc_async
    /usr/bin/python3 $1
  "
