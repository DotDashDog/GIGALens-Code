"""T20 -- prepare t3_transects.py-compatible input npz files for the carousel arms.

t3_transects.py's three input loaders expect these key layouts (verified by
reading t3_transects.py):

  --samples        : draw_typical_point() -> np.load(path)["position"], shape
                     (chains, steps, dim). It reshapes to (-1, dim) and draws ONE
                     seeded typical-set point.
  --clone          : main() -> np.load(path)["cov"], shape (dim, dim) (also uses
                     nothing else from the clone here; eigh(cov) gives stiff/soft).
  --ref-diagnostics: read_ref_step_size() -> np.load(path)["step_size"] (chains,
                     steps) and optional ["L"] (chains, steps).

The carousel reference artifacts:
  <ref>/mclmc/arrays.npz      : key "samples_z"  (8,10000,14) float64  <-- RENAME
  <ref>/mclmc/diagnostics.npz : keys "step_size","L" (8,20000) float64  <-- AS-IS
  results_carousel/{arm}/t1/clone.npz : keys mean/cov/cholesky (T17)     <-- AS-IS

So the ONLY key-layout mismatch is --samples: t3 wants "position", carousel has
"samples_z". This script writes an adapter npz per arm containing
  position <- samples_z   (identical bytes, renamed key)
under HARNESS/results_carousel/phaseC/t20/inputs/. clone.npz and diagnostics.npz
already match t3's expected layout and are passed to t3 DIRECTLY (read-only), so
they are NOT copied here -- run_t20.sh points --clone / --ref-diagnostics at the
existing files. This keeps the sim_carousel checkout untouched and avoids editing
t3_transects.py.

numpy-only; safe to run on the login node.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SIM = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests"

# arm -> reference mclmc dir (READ-ONLY source of arrays.npz/diagnostics.npz)
REF_MCLMC = {
    "old": os.path.join(_SIM, "minimal_case_oldbij", "mclmc"),
    "new": os.path.join(_SIM, "minimal_case_newbij", "mclmc"),
}
# arm -> existing T17 clone (has the "cov" key t3 needs); used by t3 directly.
CLONE = {
    "old": os.path.join(_HERE, "results_carousel", "old", "t1", "clone.npz"),
    "new": os.path.join(_HERE, "results_carousel", "new", "t1", "clone.npz"),
}


def prepare_arm(arm, out_dir):
    ref = REF_MCLMC[arm]
    arrays = os.path.join(ref, "arrays.npz")
    diag = os.path.join(ref, "diagnostics.npz")
    clone = CLONE[arm]
    for p in (arrays, diag, clone):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"[t20_prepare] missing {p}")

    a = np.load(arrays)
    if "samples_z" not in a.files:
        raise KeyError(f"'samples_z' not in {arrays}; keys={a.files}")
    samples_z = np.asarray(a["samples_z"], dtype=np.float64)   # (chains,steps,dim)
    if samples_z.ndim != 3:
        raise ValueError(f"samples_z must be 3-D; got {samples_z.shape}")

    os.makedirs(out_dir, exist_ok=True)
    out_samples = os.path.join(out_dir, f"{arm}_samples.npz")
    # KEY MAPPING: position <- samples_z (rename only; identical values).
    np.savez(out_samples, position=samples_z)

    # sanity: diagnostics already has t3's expected keys
    d = np.load(diag)
    for k in ("step_size", "L"):
        if k not in d.files:
            raise KeyError(f"'{k}' not in {diag}; keys={d.files}")
    c = np.load(clone)
    if "cov" not in c.files:
        raise KeyError(f"'cov' not in {clone}; keys={c.files}")

    info = {
        "arm": arm,
        "samples_adapter": os.path.abspath(out_samples),
        "samples_key_mapping": "position <- samples_z",
        "samples_shape": list(samples_z.shape),
        "clone_used_directly": os.path.abspath(clone),
        "ref_diagnostics_used_directly": os.path.abspath(diag),
        "note": ("clone.npz (cov) and diagnostics.npz (step_size,L) already match "
                 "t3's expected key layout -> passed to t3 directly, not copied."),
    }
    print(f"[t20_prepare] {arm}: wrote {out_samples}  (position <- samples_z "
          f"{samples_z.shape})")
    return info


def main(argv=None):
    p = argparse.ArgumentParser(description="T20 prepare t3 inputs (carousel)")
    p.add_argument("--arm", choices=["old", "new", "both"], default="both")
    p.add_argument("--out-dir",
                   default=os.path.join(_HERE, "results_carousel", "phaseC",
                                        "t20", "inputs"))
    args = p.parse_args(argv)
    arms = ["old", "new"] if args.arm == "both" else [args.arm]
    infos = [prepare_arm(a, args.out_dir) for a in arms]
    manifest = {
        "experiment": "T20 prepare t3 inputs",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "arms": infos,
    }
    mpath = os.path.join(args.out_dir, "t20_inputs_manifest.json")
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[t20_prepare] wrote {mpath}")


if __name__ == "__main__":
    main()
