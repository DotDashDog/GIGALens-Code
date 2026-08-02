"""Carousel minimal case -- NEW arm, ROUTE B reparameterization (T25/T26).

Pre-registered 2026-07-03 (docs/logs/why-hard-to-sample.md, "T25 ... + T26 ...").
THIN variant of ../carousel_min_new (same NFW_ELLIPSE_EINSTEIN family, priors,
data, simulator) with the z_Rs column wrapped by a scalar monotone leaf
``w: u -> z`` whose knots come from the Route-B artifact

    results_carousel/phaseC/t25/transform_B.npz

(observable-anchored coordinate: u_B propto the NFW deflection slope
s(Rs) = dln(alpha)/dln(r) at the fixed reference arc radius theta_E*). Pure
coordinate change -> theta-space posterior byte-identical to carousel_min_new
(T25 family/identity gates verify it).

qz-hash pinning is BYPASSED here for the same, documented reason as Route A (no
reference MCLMC run exists in the reparameterized coordinates); the baseline
guards are untouched. See ../carousel_min_newA/system.py for the full rationale.
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

ROUTE = "B"
ARTIFACT = os.path.join(_HARNESS, "results_carousel", "phaseC", "t25",
                        "transform_B.npz")

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
      prob_model, dim, param_names, leaf, col, sha256, z_best, ref_dir."""
    import jax
    jax.config.update("jax_enable_x64", True)

    from reparam_bijector import attach_reparam_leaf

    common = _common()
    newmod = _newmod()
    if not os.path.isfile(artifact):
        raise FileNotFoundError(
            f"[route {ROUTE}] transform artifact not found: {artifact}. Run "
            "t25_transforms.py first (it writes transform_B.npz).")

    nfw0 = newmod._nfw0()
    prob_model = common._build_prob_model(nfw0, None)
    z_best = common._load_z_best(newmod.REF_DIR)
    dim = int(z_best.shape[0])
    param_names = common._recover_param_names(prob_model, dim)

    print(f"[route {ROUTE}] building reparameterized new-arm target "
          f"(qz-hash pinning BYPASSED -- reparam coords have no reference run)")
    leaf, col, sha = attach_reparam_leaf(prob_model, param_names, artifact)

    return {
        "prob_model": prob_model, "dim": dim,
        "param_names": param_names, "leaf": leaf, "col": col, "sha256": sha,
        "z_best": np.asarray(z_best, np.float64), "ref_dir": newmod.REF_DIR,
        "route": ROUTE, "artifact": os.path.abspath(artifact),
    }


def map_z_to_u(z, leaf, col):
    u = np.asarray(z, np.float64).copy()
    u[col] = leaf.inverse(np.asarray(u[col]))
    return u


def map_u_to_z(u, leaf, col):
    import jax.numpy as jnp
    z = np.asarray(u, np.float64).copy()
    z[col] = float(np.asarray(leaf.forward(jnp.asarray(z[col]))))
    return z


def load_target(supersample=None):
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
    return b["prob_model"], qz, u_center, b["dim"], b["param_names"]


if __name__ == "__main__":
    print(f"carousel_min_new{ROUTE}: reparameterized new arm; artifact={ARTIFACT}")
