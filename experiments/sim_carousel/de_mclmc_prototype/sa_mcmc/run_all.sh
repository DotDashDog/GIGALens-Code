#!/bin/bash
export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:/global/u1/l/linusu/GIGALens-Code/src:/global/u1/l/linusu/gigalens/src:$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages
export JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu
cd /global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/de_mclmc_prototype/sa_mcmc
SH="shifter --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 /usr/bin/python3 -u"
echo "### ATTRIB"; $SH /tmp/attrib.py 2>&1 | grep -E "b=|Error|Trace"
echo "### COMBINED"; $SH /tmp/refac.py 2>&1 | grep -E "INVAR|Error|Trace"
echo "### CURVED"; R=1500 $SH validate_curved.py 2>&1 | grep -vE "Warning|slow_op|constant|XLA|guards|^$|metadata|%"
echo "### TINY"
: > log_tiny.txt
for cfg in "0.03 mixture 1.0" "0.03 mixture 2.5" "0.03 gaussian 1.0" "0.001 gaussian 1.0"; do
  echo "--CFG $cfg"; R=3000 $SH tiny_mode_test.py $cfg 2>&1 | grep -E "est_occupancy|truth=|PROPOSED"
done
echo "### ALL_COMPLETE"
