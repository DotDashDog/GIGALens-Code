#!/bin/bash
# Run repro with sidecar-overlaid tfp-nightly + a chosen JAX container image.
set -u
TAG="${1:-default}"
IMAGE="${2:-docker:ghcr.io/nvidia/jax:jax-2026-04-13}"
shift 2 || true
EXTRA_ENV=""
for kv in "$@"; do EXTRA_ENV="$EXTRA_ENV $kv"; done

LOG="/global/homes/l/linusu/GIGALens-Code/alternate_inference/repro_${TAG}.log"

echo "=== TAG=$TAG host=$(hostname) image=$IMAGE date=$(date -Iseconds)" | tee "$LOG"
echo "=== EXTRA_ENV=$EXTRA_ENV" | tee -a "$LOG"

shifter --module=gpu,nccl-plugin --image=$IMAGE \
    bash -c "
        set -u
        # Sidecar tfp-nightly first, then conda env. Order matters.
        export PYTHONPATH=\"/global/homes/l/linusu/sidecar_jax_upgrade:\$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages\"
        export REPRO_TAG='$TAG'
        $EXTRA_ENV
        echo '--- versions ---'
        python -c 'import jax, jaxlib; print(\"jax\", jax.__version__, \"jaxlib\", jaxlib.__version__)'
        python -c 'import tensorflow_probability as tfp; print(\"tfp\", tfp.__version__)'
        python -c 'import blackjax; print(\"blackjax\", blackjax.__version__)'
        echo '--- starting python ---'
        python -u /global/homes/l/linusu/GIGALens-Code/alternate_inference/repro_mclmc_238.py
        ec=\$?
        echo \"--- python exit code: \$ec ---\"
        exit \$ec
    " 2>&1 | tee -a "$LOG"

EC=${PIPESTATUS[0]}
echo "=== exit code: $EC" | tee -a "$LOG"
exit $EC
