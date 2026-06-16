#!/bin/bash
# D-fix1 diagnostic sweep: under high_precision (mixed float64 likelihood), vary the
# isotropic initial metric scale (diag_scale) at n_max=25, plus an n_max=10 control.
# Tests whether the tune1 eps-crash + freeze is driven by init-metric mis-scaling
# vs residual model noise. Each run dumps the full instrumented Hist.
set -uo pipefail
cd "$(dirname "$0")"
SWEEP_DIR=diagnosis_2026-06/fix1/diag_sweep
mkdir -p "$SWEEP_DIR"

COMMON="--num-burnin-steps 2000 --num-results 500 --n-chains 8 --seed 0 --bootstrap-map-samples 16 --high-precision"

run () {
    local tag="$1"; shift
    echo ">>>>>>>>>> START ${tag} $(date -Iseconds)"
    bash run_repro_vela_gpu.sh "${tag}" "$@" --out-dir "${SWEEP_DIR}/${tag}"
    echo ">>>>>>>>>> END   ${tag} exit=$? $(date -Iseconds)"
}

# n_max=25, isotropic init-metric scale sweep
run n25_ds1e-4_hp --n-max 25 --diag-scale 1e-4 $COMMON
run n25_ds1e-6_hp --n-max 25 --diag-scale 1e-6 $COMMON
run n25_ds1e-8_hp --n-max 25 --diag-scale 1e-8 $COMMON
run n25_ds1e-10_hp --n-max 25 --diag-scale 1e-10 $COMMON

# n_max=10 control (should stay healthy under high_precision)
run n10_ds1e-8_hp --n-max 10 --diag-scale 1e-8 $COMMON

echo "ALL SWEEP RUNS DONE $(date -Iseconds)"
