"""Carousel minimal case -- NEW arm with the T28 OBSERVABLE-SLOPE PRIOR SWAP.

Pre-registered 2026-07-04 (docs/logs/why-hard-to-sample.md, "T28"). A THIN variant
of ../carousel_min_new that is IDENTICAL in every respect (same NFW_ELLIPSE_EINSTEIN
family, same data, same simulator, same OTHER 13 priors) EXCEPT that the Rs prior
component

    Rs ~ Uniform(20, 100)

is REPLACED by a prior set directly on the observable deflection slope
s(Rs) = d ln alpha/d ln r at r = theta_E* = 13.8127:

    Rs prior := tfd.TransformedDistribution(tfd.Uniform(0.0, 0.75), bijector=RsOfS)

where RsOfS.forward = the monotone PCHIP leaf s -> Rs (results_carousel/phaseC/t28/
transform_sprior.npz) and RsOfS.inverse/ildj are the ANALYTIC jnp slope s(Rs) and
log|ds/dRs| (t28_common). This is an INTENTIONAL MODELING CHANGE -- the theta-space
posterior is NOT preserved (cf. T25/T26 which were pure coordinate changes).

WHY THIS IS CORRECT FOR THE HOT PATH (traced; see t28_common docstring):
  ProbModel.log_prior(z) = prior.log_prob(bij.forward(z)) + bij.forward_log_det_jacobian(z)
  * prior.log_prob evaluates the Rs slot as the TransformedDistribution's constrained
    density  -log(0.75) + log|ds/dRs|_analytic(Rs)  (uses ONLY analytic inverse/ildj).
  * bij = Chain([prior.experimental_default_event_space_bijector(), pack]); the Rs
    slot's default bijector composes to  RsOfS o Sigmoid(0,0.75), so the SAMPLING
    chart is unconstrained-s (flat-in-s), and its fldj is the leaf's own derivative.
  Under change of variables the leaf Jacobian cancels EXACTLY, so the sampled
  physical posterior is  pi(Rs) ∝ L(Rs)·(1/0.75)·(ds/dRs)_analytic(Rs)  -- the
  intended prior×likelihood (GPU gates G2 prior-identity + G3 loglike-identity).

qz-hash pinning is BYPASSED (as in carousel_min_newA/B): no reference MCLMC run
exists under the swapped prior. The baseline guards (carousel_min_new) are untouched.

NEW FILE ONLY -- does not modify carousel_min_common.py or any existing system.
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

if _HARNESS not in sys.path:
    sys.path.insert(0, _HARNESS)

from t28_common import ARTIFACT, S_HI, S_LO, load_leaf  # noqa: E402


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _common():
    return _load(_COMMON_PY, "carousel_min_common")


def _newmod():
    return _load(_NEW_PY, "carousel_min_new_baseline")


def _sprior_nfw0(leaf):
    """NEW-arm NFW mass Component with the Rs prior REPLACED by the s-prior. All
    OTHER params (theta_E, e1, e2, center_x, center_y) are copied VERBATIM from
    carousel_min_new._nfw0() (single source of truth -> no drift), so only the Rs
    slot changes."""
    from gigalens.jax.scene import Component
    from t28_common import make_sprior_Rs_distribution

    base_comp = _newmod()._nfw0()           # identical NFW family + priors
    priors = dict(base_comp.priors)         # copy the 6-param prior dict
    if "Rs" not in priors:
        raise KeyError(f"expected 'Rs' in NFW priors; got {sorted(priors)}")
    priors["Rs"] = make_sprior_Rs_distribution(leaf)   # <-- the ONLY change
    # reuse the SAME profile instance (identical renderer)
    return Component(base_comp.profile, priors)


def build_sprior_target(artifact=ARTIFACT):
    """Build the s-prior target (qz-hash guard bypassed). Returns a dict:
      prob_model, dim, param_names, leaf, rs_col, z_best, ref_dir, artifact."""
    import jax
    jax.config.update("jax_enable_x64", True)   # sampling needs f64

    common = _common()
    newmod = _newmod()
    leaf = load_leaf(artifact)                  # eager-materialized inside make_RsOfS

    nfw0 = _sprior_nfw0(leaf)
    prob_model = common._build_prob_model(nfw0, None)
    z_best = common._load_z_best(newmod.REF_DIR)   # baseline MAP (for init mapping)
    dim = int(z_best.shape[0])
    param_names = common._recover_param_names(prob_model, dim)
    rs_col = _rs_col(param_names)

    from reparam_bijector import sha256_file
    sha = sha256_file(artifact)
    print(f"[sprior] built NEW-arm target with Rs~TransformedDist(Uniform"
          f"({S_LO},{S_HI}), RsOfS); INTENTIONAL prior change (not a coord change)")
    print(f"[sprior] qz-hash pinning BYPASSED (no reference run under swapped prior)")
    print(f"[sprior] artifact = {os.path.abspath(artifact)}")
    print(f"[sprior] artifact sha256 = {sha}")
    print(f"[sprior] Rs column = {rs_col} (param {param_names[rs_col]})")

    return {
        "prob_model": prob_model, "dim": dim,
        "param_names": param_names, "leaf": leaf, "rs_col": rs_col,
        "sha256": sha, "z_best": np.asarray(z_best, np.float64),
        "ref_dir": newmod.REF_DIR, "artifact": os.path.abspath(artifact),
    }


def _rs_col(param_names):
    names = [str(n) for n in param_names]
    cand = [i for i, n in enumerate(names)
            if (n == "Rs" or n.endswith("Rs")) and not n.endswith("alpha_Rs")]
    if len(cand) != 1:
        raise ValueError(f"Rs column not unique in {names}: {cand}")
    return cand[0]


# ---------------------------------------------------------------------------
# init-mapping helpers (physical Rs <-> s <-> unconstrained-s)
# ---------------------------------------------------------------------------
def baseline_z_to_theta(z, baseline_pm, dim):
    """Baseline z-vector -> constrained theta dict via the BASELINE bijector."""
    import jax.numpy as jnp
    z = np.asarray(z, np.float64)
    return baseline_pm.bij.forward([jnp.asarray(z[j][None]) for j in range(dim)])


def map_baseline_z_to_u(z_init, baseline_pm, sprior_pm, dim):
    """Map a baseline-chart z_init (physical MAP-ish point) into the sprior chart.

    Robust route: baseline z -> physical theta (baseline bij.forward) -> sprior u
    (sprior bij.inverse). Uses tfp's OWN inverse (RsOfS._inverse = analytic s(Rs),
    Sigmoid(0,0.75).inverse), so it does not rely on a hand-written chart formula.
    Only the Rs column differs between charts (all other priors identical); this is
    asserted. Returns (u_init, info)."""
    import jax.numpy as jnp
    z_init = np.asarray(z_init, np.float64)
    theta_b = baseline_z_to_theta(z_init, baseline_pm, dim)
    u_cols = sprior_pm.bij.inverse(theta_b)                 # list of (1,) columns
    u_init = np.array([float(np.asarray(c).reshape(-1)[0]) for c in u_cols], np.float64)
    rs_col = _rs_col(sorted(str(k) for k in theta_b.keys()))
    dz = u_init - z_init
    dz[rs_col] = 0.0
    max_other = float(np.max(np.abs(dz)))
    info = {"rs_col": rs_col, "z_Rs": float(z_init[rs_col]),
            "u_Rs": float(u_init[rs_col]), "max_abs_other_coord_change": max_other}
    return u_init, info


# ---------------------------------------------------------------------------
# analytic chart helpers (physical <-> s <-> unconstrained) -- for tests/plots
# ---------------------------------------------------------------------------
def Rs_to_s(Rs):
    """physical Rs -> observable s (numpy analytic)."""
    from t25_transforms import slope_s_of_Rs
    from t28_common import THETA_E_STAR
    return np.asarray(slope_s_of_Rs(np.asarray(Rs, np.float64), THETA_E_STAR), np.float64)


def s_to_u(s):
    """observable s in (0,0.75) -> unconstrained u = Sigmoid(0,0.75).inverse(s)."""
    s = np.asarray(s, np.float64)
    return np.log((s - S_LO) / (S_HI - s))


def u_to_s(u):
    """unconstrained u -> s = S_LO + (S_HI-S_LO)*sigmoid(u)."""
    u = np.asarray(u, np.float64)
    return S_LO + (S_HI - S_LO) / (1.0 + np.exp(-u))


def s_to_Rs(s, leaf):
    """observable s -> physical Rs via the leaf spline (numpy forward)."""
    return leaf._forward_np(np.asarray(s, np.float64))


# ---------------------------------------------------------------------------
# common-compatible loader
# ---------------------------------------------------------------------------
def load_target(supersample=None):
    """common-compatible 5-tuple (prob_model, qz, z_center, dim, param_names).

    z_center = baseline MAP z_best MAPPED into the sprior chart; qz is a fresh
    MVNDiag there (the T28 runner supplies its own typical-set qz')."""
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions

    if supersample is not None:
        raise ValueError("sprior system reproduces the notebook supersample only")
    b = build_sprior_target()
    # map the baseline MAP center into the sprior chart (needs the baseline pm)
    common = _common()
    newmod = _newmod()
    base_pm, _qz, _zc, _dim, _names = common.load_target(
        newmod._nfw0, newmod.REF_DIR, newmod.PINNED_QZ_HASH)
    u_center, _info = map_baseline_z_to_u(
        b["z_best"], base_pm, b["prob_model"], b["dim"])
    qz = tfd.MultivariateNormalDiag(
        loc=jnp.asarray(u_center), scale_diag=jnp.full(b["dim"], 1e-3))
    print(f"[sprior] default qz = MVNDiag(u_center, 1e-3) (BYPASS-pinned)")
    return b["prob_model"], qz, u_center, b["dim"], b["param_names"]


if __name__ == "__main__":
    print(f"carousel_min_sprior: T28 observable-slope prior swap; artifact={ARTIFACT}")
