"""Carousel minimal case -- NEW arm, ROUTE A reparameterization (T25/T26).

Pre-registered 2026-07-03 (docs/logs/why-hard-to-sample.md, "T25 ... + T26 ...").
This is a THIN variant of ../carousel_min_new: it builds the SAME target (same
NFW_ELLIPSE_EINSTEIN family, same priors, same data, same simulator) and then
wraps the unconstraining bijector with a scalar monotone leaf ``w: u -> z`` on the
z_Rs column ONLY. The leaf's knots come from the Route-A artifact

    results_carousel/phaseC/t25/transform_A.npz

(variance-stabilizing coordinate u(z) = int sqrt(lambda_hat(z')) dz'). Because the
map is a PURE coordinate change composed before the existing chain -- with
log|w'(u)| added to the forward-log-det-jacobian -- the theta-space posterior is
byte-identical to carousel_min_new (verified by the T25 family/identity gates).

qz-HASH PINNING IS DELIBERATELY BYPASSED HERE (documented):
  carousel_min_common.load_target asserts stable_hash(rebuilt qz) == the pinned
  reference-run hash. That guard is correct for the BASELINE coordinates, but no
  reference MCLMC run exists in these REPARAMETERIZED coordinates, so the hash can
  never match by construction. We therefore build the prob_model via common's
  PRIVATE builders and skip the qz-hash assertion. This does NOT weaken the
  baseline systems' guards (they are untouched, in their own modules); it is a
  separate code path used only by the T26 route runner. The default qz returned
  here is a fresh MVNDiag centred at the MAP z_best MAPPED into u-coords -- the
  T26 runner overrides it with the typical-set-init qz' anyway.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SYSTEMS = os.path.dirname(_HERE)
_HARNESS = os.path.dirname(_SYSTEMS)
_COMMON_PY = os.path.join(_SYSTEMS, "carousel_min_common.py")
_NEW_PY = os.path.join(_SYSTEMS, "carousel_min_new", "system.py")

ROUTE = "A"
ARTIFACT = os.path.join(_HARNESS, "results_carousel", "phaseC", "t25",
                        "transform_A.npz")

if _HARNESS not in sys.path:
    sys.path.insert(0, _HARNESS)


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _common():
    return _load(_COMMON_PY, "carousel_min_common")


def _newmod():
    return _load(_NEW_PY, "carousel_min_new_baseline")


def build_route_target(artifact=ARTIFACT):
    """Build the wrapped target WITHOUT the qz-hash guard. Returns a dict with:
      model_seq, prob_model, dim, param_names, leaf, col, sha256, z_best, ref_dir.
    prob_model.bij is the ColumnReparamBijector (u -> theta)."""
    import jax
    jax.config.update("jax_enable_x64", True)   # sampling needs f64 (as load_target)

    from reparam_bijector import attach_reparam_leaf

    common = _common()
    newmod = _newmod()
    if not os.path.isfile(artifact):
        raise FileNotFoundError(
            f"[route {ROUTE}] transform artifact not found: {artifact}. Run "
            "t25_transforms.py first (it writes transform_A.npz).")

    nfw0 = newmod._nfw0()                        # IDENTICAL NFW family + priors
    model_seq, prob_model = common._build_prob_model(nfw0, None)
    z_best = common._load_z_best(newmod.REF_DIR)
    dim = int(z_best.shape[0])
    param_names = common._recover_param_names(prob_model, dim)

    print(f"[route {ROUTE}] building reparameterized new-arm target "
          f"(qz-hash pinning BYPASSED -- reparam coords have no reference run)")
    leaf, col, sha = attach_reparam_leaf(prob_model, param_names, artifact)

    return {
        "model_seq": model_seq, "prob_model": prob_model, "dim": dim,
        "param_names": param_names, "leaf": leaf, "col": col, "sha256": sha,
        "z_best": np.asarray(z_best, np.float64), "ref_dir": newmod.REF_DIR,
        "route": ROUTE, "artifact": os.path.abspath(artifact),
    }


def map_z_to_u(z, leaf, col):
    """Baseline z-vector -> route u-vector (only the Rs column changes)."""
    u = np.asarray(z, np.float64).copy()
    u[col] = leaf.inverse(np.asarray(u[col]))
    return u


def map_u_to_z(u, leaf, col):
    """Route u-vector -> baseline z-vector (only the Rs column changes)."""
    import jax.numpy as jnp
    z = np.asarray(u, np.float64).copy()
    z[col] = float(np.asarray(leaf.forward(jnp.asarray(z[col]))))
    return z


def load_target(supersample=None):
    """common-compatible 5-tuple (model_seq, qz, z_center, dim, param_names).

    qz is a fresh MVNDiag centred at the MAP z_best MAPPED to u-coords (NOT hash
    pinned -- see module docstring). The T26 runner supplies its own qz'."""
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions

    if supersample is not None:
        raise ValueError("route systems reproduce the notebook supersample only")
    b = build_route_target()
    u_center = map_z_to_u(b["z_best"], b["leaf"], b["col"])
    qz = tfd.MultivariateNormalDiag(
        loc=jnp.asarray(u_center), scale_diag=jnp.full(b["dim"], 1e-3))
    print(f"[route {ROUTE}] default qz = MVNDiag(u_center, 1e-3) (BYPASS-pinned)")
    return b["model_seq"], qz, u_center, b["dim"], b["param_names"]


if __name__ == "__main__":
    print(f"carousel_min_new{ROUTE}: reparameterized new arm; artifact={ARTIFACT}")
