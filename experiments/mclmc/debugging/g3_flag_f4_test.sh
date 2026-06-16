#!/bin/bash
# G3: validate (a) the new likelihood_precision="float64" flag reproduces the healthy n25 run,
# and (b) F4 (--regularize-mm) does not regress it. GPU, serialized via run_repro_vela_gpu.sh.
set -uo pipefail
cd "$(dirname "$0")"
OUT=diagnosis_2026-06/fix1/flag_f4
mkdir -p "$OUT"
# --no-bootstrap: skip the float64 bootstrap MAP (OOMs on the GPU shared with the live notebook);
# use the truth point as the qz center. Irrelevant to the sampler-side float64/F4 validation.
COMMON="--n-max 25 --diag-scale 1e-8 --likelihood-precision float64 --no-bootstrap \
  --num-burnin-steps 2000 --num-results 500 --n-chains 8 --seed 0"

echo ">>>>> n25 float64, NO F4 (flag control) $(date -Iseconds)"
bash run_repro_vela_gpu.sh n25_f64_noF4 $COMMON --out-dir "$OUT/n25_f64_noF4"

echo ">>>>> n25 float64, WITH F4 (regression) $(date -Iseconds)"
bash run_repro_vela_gpu.sh n25_f64_F4 $COMMON --regularize-mm --out-dir "$OUT/n25_f64_F4"

echo "G3 DONE $(date -Iseconds)"
