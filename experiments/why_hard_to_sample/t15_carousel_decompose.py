"""T15 -- carousel minimal-case NFW reparameterization decomposition (Phase A).

Pre-registered decomposition of WHY the two MCLMC reference runs for the
"carousel minimal case" (OLD arm: NFW_ELLIPSE with alpha_Rs; NEW arm:
NFW_ELLIPSE_EINSTEIN with theta_E) differ in sampling difficulty. Every
threshold below is REGISTERED (P1/F1, P2/F2, P3, P4) and encoded as a module
constant; NONE is tuned here. This script PROPOSES numbers/plots
(UNCERTIFIED) and does not adjudicate -- a grader inspects the artifacts.

Runs on a GPU node inside the Shifter container (JAX_ENABLE_X64=1). All outputs
go to results_carousel/phaseA/.

Interfaces this script codes against (verified against source):
  * common.load_target(data_dir) -> (prob_model, qz, z_center, dim, param_names)
    (common.py:211). data_dir = systems/carousel_min_{old,new} (built by a
    parallel subagent with the sys60 system-module interface).
  * prob_model exposes (scene_prob_model.py):
      - log_like(z)  -> (log_like, red_chi2)   [:274]  (LIKELIHOOD only)
      - log_prior(z) -> log_prior + fwd_log_det_jacobian [:308] (PRIOR only)
      - log_prob(z)  -> (log_like+log_prior, red_chi2)   [:313]
      - bij           = model.bijector (scene.py:258, a tfb.Chain)
      - model._unique = {ukey: tfd.Distribution} (scene.py:206) the prior dict
    log_like/log_prior take z of shape (N, dim) (they do list(z.T)).
  * bij.forward(list(z.T)) -> FLAT dict {ukey: array-(N,)} (scene.py:285).
  * bij.inverse({ukey: array-(1,)}) -> LIST of columns (each (1,)); stack ->
    z of shape (1, dim). KNOWN PITFALL (verified): inverse consumes a FLAT dict
    of (1,)-shaped arrays, NOT a nested pytree; the returned list is in
    sorted-key (sampler-column) order.
  * Parameter names via the C-8-safe sorted-key zero-probe:
    sorted(bij.forward(list(zeros.T)).keys()) == sampler column order.

theta_E <-> alpha_Rs conversion (REUSED, not re-derived) from nfw.py:
  OLD NFW_ELLIPSE rho0        = alpha_Rs / (4 Rs^2 (1-log2))      (nfw.py:106)
  NEW NFW_ELLIPSE_EINSTEIN rho0 = theta_E^2 / (4 Rs^3 g(theta_E/Rs)) (nfw.py:146)
  The two profiles are physically identical iff rho0 matches, so at fixed Rs:
    alpha_Rs = (1-log2) * theta_E^2 / (Rs * g(theta_E/Rs))          [closed form]
    theta_E  : solve theta_E^2 / g(theta_E/Rs) = alpha_Rs*Rs/(1-log2) [bisection]
  g(.) is the EXACT gigalens NFW().g_ (nfw.py:36) -- reused, never re-derived.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from common import assert_x64, load_target

# ---------------------------------------------------------------------------
# REGISTERED constants (pre-registered thresholds -- DO NOT TUNE)
# ---------------------------------------------------------------------------
P1_THETA_E_REF = 13.0    # P1: expected main-mode theta_E scale (arcsec)
P1_MARGIN = 2.0          # P1 predicts |theta_E(b) - 13| >= 2
F1_MARGIN = 1.0          # F1 fires if |theta_E(b) - 13| < 1 AND ...
F1_LOGLIKE_NATS = 10.0   #      ... delta_loglike(b vs m) >= 10 nats
P2_RATIO = 0.5           # P2: sagitta / transverse_sigma >= 0.5
F2_RATIO = 0.15          # F2: sagitta / transverse_sigma < 0.15
P3_SIGMA = 1.0           # P3: 1D-marginal shift > 1 pooled sigma flags leakage
CROWD_FLAG = 0.3         # P4: bound-crowding fraction > 0.3 is flagged

BASIN_DRAWS = 2500       # basin b_z = chain-0 post-burn-in draws [0:2500]
RIDGE_EXCLUDE_CHAIN0 = 5000  # oldbij ridge excludes chain-0 draws [0:5000]

OUT_SUBDIR = "results_carousel/phaseA"


# ---------------------------------------------------------------------------
# NFW theta_E <-> alpha_Rs conversion (reuses gigalens NFW().g_ exactly)
# ---------------------------------------------------------------------------
def _make_conversion():
    """Return (theta_E_to_alpha_Rs, alpha_Rs_to_theta_E) closures that call the
    EXACT gigalens NFW().g_ (nfw.py:36). Both are vectorized over numpy arrays."""
    import jax.numpy as jnp
    from gigalens.jax.profiles.mass.nfw import NFW
    _nfw = NFW()
    LOG2 = float(np.log(2.0))

    def g_of(x):
        return np.asarray(_nfw.g_(jnp.asarray(np.asarray(x, dtype=np.float64))),
                          dtype=np.float64)

    def theta_E_to_alpha_Rs(theta_E, Rs):
        theta_E = np.asarray(theta_E, dtype=np.float64)
        Rs = np.asarray(Rs, dtype=np.float64)
        return (1.0 - LOG2) * theta_E ** 2 / (Rs * g_of(theta_E / Rs))

    def alpha_Rs_to_theta_E(alpha_Rs, Rs, lo=1e-4, hi=500.0, iters=90):
        """Monotone bisection on theta_E^2/g(theta_E/Rs) = alpha_Rs*Rs/(1-log2).
        The LHS is monotone increasing in theta_E (verified), so a fixed bracket
        + bisection converges to ~machine precision (validated round-trip 1e-15)."""
        alpha_Rs = np.atleast_1d(np.asarray(alpha_Rs, dtype=np.float64))
        Rs = np.broadcast_to(np.asarray(Rs, dtype=np.float64), alpha_Rs.shape).copy()
        target = alpha_Rs * Rs / (1.0 - LOG2)
        lo_a = np.full_like(alpha_Rs, lo)
        hi_a = np.full_like(alpha_Rs, hi)

        def f(te):
            return te ** 2 / g_of(te / Rs) - target

        f_lo, f_hi = f(lo_a), f(hi_a)
        if np.any(f_lo > 0) or np.any(f_hi < 0):
            raise ValueError(
                "alpha_Rs_to_theta_E: root not bracketed in "
                f"[{lo},{hi}] for some sample (f_lo>0: {int(np.sum(f_lo>0))}, "
                f"f_hi<0: {int(np.sum(f_hi<0))}); widen the bracket.")
        for _ in range(iters):
            mid = 0.5 * (lo_a + hi_a)
            fm = f(mid)
            pos = fm > 0
            hi_a = np.where(pos, mid, hi_a)
            lo_a = np.where(pos, lo_a, mid)
        return 0.5 * (lo_a + hi_a)

    return theta_E_to_alpha_Rs, alpha_Rs_to_theta_E


# ---------------------------------------------------------------------------
# names + z<->physical (C-8-safe sorted-key convention)
# ---------------------------------------------------------------------------
def derive_names(prob_model, dim):
    """C-8-safe param names = sorted keys of the zero-probe bijector output.
    sorted() reproduces the sampler column order (scene.py pack_sequence_as +
    JAX tree flatten sort dict keys); NEVER trust names.npy (reversed)."""
    import jax.numpy as jnp
    probe = np.zeros((1, dim), dtype=np.float64)
    out = prob_model.bij.forward(list(jnp.asarray(probe).T))
    if not isinstance(out, dict) or not out or any(
            isinstance(v, (dict, list)) for v in out.values()):
        raise TypeError(f"bij.forward did not return a flat scalar-key dict: {type(out)}")
    names = sorted(str(k) for k in out.keys())
    if len(names) != dim or len(set(names)) != dim:
        raise ValueError(f"recovered {len(names)} unique names != dim {dim}")
    return names


def z_to_physical(prob_model, z, names, chunk=20000):
    """z (N, dim) -> theta (N, dim) in `names` (sorted-key) column order.
    Chunked bijector.forward; analytic, cheap, batched."""
    import jax.numpy as jnp
    z = np.asarray(z, dtype=np.float64)
    N = z.shape[0]
    cols = {n: np.empty(N, dtype=np.float64) for n in names}
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        out = prob_model.bij.forward(list(jnp.asarray(z[s:e]).T))
        for n in names:
            cols[n][s:e] = np.asarray(out[n], dtype=np.float64).reshape(-1)
    return np.stack([cols[n] for n in names], axis=1)


def physical_dict_to_z(prob_model, phys_dict, names):
    """{name: scalar} (ALL names) -> z of shape (1, dim). Verified pitfall:
    bij.inverse consumes a FLAT dict of (1,)-shaped arrays; returns a LIST of
    columns in sorted-key order -> stack to z."""
    import jax.numpy as jnp
    flat = {n: jnp.asarray([float(phys_dict[n])], dtype=jnp.float64) for n in names}
    cols = prob_model.bij.inverse(flat)  # list of (1,) columns, sorted-key order
    z = np.stack([np.asarray(c, dtype=np.float64).reshape(-1)[0] for c in cols])
    return z.reshape(1, -1)


# ---------------------------------------------------------------------------
# bound-crowding (SAME convention as t2_zspace_diagnostics.compute_bound_crowding:347)
# ---------------------------------------------------------------------------
def bound_crowding(prob_model, theta, names):
    """Per-bounded-param fraction of mass in the outer-1% band (1% at EACH edge)
    of the PRIOR BOX -- byte-for-byte the metric in
    t2_zspace_diagnostics.compute_bound_crowding (lines 347-377), but reading the
    prior dict off model._unique (scene API) instead of build_model.prob_model."""
    unique = getattr(prob_model.model, "_unique", None)
    if not unique:
        joint = getattr(prob_model, "prior", None)
        unique = dict(getattr(joint, "model", {})) if joint is not None else None
    if not unique:
        raise AttributeError("could not locate prior dict (model._unique / prior.model)")
    rows = []
    for i, name in enumerate(names):
        dist = unique.get(name)
        low = getattr(dist, "low", None) if dist is not None else None
        high = getattr(dist, "high", None) if dist is not None else None
        if dist is None:
            rows.append({"index": i, "param": name, "status": "no_matching_prior_entry",
                         "crowding_frac": None, "bounds": None})
            continue
        if low is None or high is None:
            rows.append({"index": i, "param": name, "status": "unbounded_prior_no_box",
                         "crowding_frac": None, "bounds": None})
            continue
        lo = float(np.asarray(low)); hi = float(np.asarray(high))
        width = hi - lo
        if not (width > 0):
            rows.append({"index": i, "param": name, "status": "degenerate_box",
                         "crowding_frac": None, "bounds": [lo, hi]})
            continue
        vals = theta[:, i]
        outer_lo = lo + 0.01 * width
        outer_hi = hi - 0.01 * width
        frac = float(np.mean((vals <= outer_lo) | (vals >= outer_hi)))
        rows.append({"index": i, "param": name, "status": "ok",
                     "bounds": [lo, hi], "crowding_frac": frac,
                     "flagged": bool(frac > CROWD_FLAG)})
    return rows


# ---------------------------------------------------------------------------
# ridge curvature (step 6)
# ---------------------------------------------------------------------------
def ridge_curvature(Rs, alpha_Rs):
    """(Rs, alpha_Rs) cloud -> (sagitta, transverse_sigma, ratio, fit info).
    Standardize each axis by its own std, 2-D PCA, fit y=a+b*x+c*x^2 (y=PC2,
    x=PC1); sagitta = |c| * (2*sigma_x)^2 = |c|*4*sigma_x^2; transverse_sigma =
    std of quadratic-fit residuals; ratio = sagitta / transverse_sigma."""
    Rs = np.asarray(Rs, dtype=np.float64)
    alpha_Rs = np.asarray(alpha_Rs, dtype=np.float64)
    X = np.stack([Rs, alpha_Rs], axis=1)
    Xs = (X - X.mean(axis=0)) / X.std(axis=0)
    cov = np.cov(Xs, rowvar=False)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]        # PC1 = largest variance
    V = V[:, order]
    proj = Xs @ V                       # columns: PC1, PC2
    x = proj[:, 0]; y = proj[:, 1]
    c, b, a = np.polyfit(x, y, 2)       # y = c x^2 + b x + a
    sigma_x = float(np.std(x))
    sagitta = float(abs(c) * 4.0 * sigma_x ** 2)
    resid = y - (c * x ** 2 + b * x + a)
    transverse_sigma = float(np.std(resid))
    ratio = float(sagitta / transverse_sigma) if transverse_sigma > 0 else float("inf")
    return {
        "sagitta": sagitta, "transverse_sigma": transverse_sigma, "ratio": ratio,
        "quad_c": float(c), "quad_b": float(b), "quad_a": float(a),
        "sigma_x_pc1": sigma_x, "n": int(X.shape[0]),
        "meets_P2": bool(ratio >= P2_RATIO), "fires_F2": bool(ratio < F2_RATIO),
    }


# ---------------------------------------------------------------------------
# plots (matplotlib only -- no seaborn)
# ---------------------------------------------------------------------------
def _scatter_overlay(ax, layers, xlab, ylab, title):
    for xs, ys, color, label, alpha, size in layers:
        ax.scatter(xs, ys, s=size, c=color, alpha=alpha, label=label,
                   edgecolors="none", rasterized=True)
    ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title)
    ax.legend(fontsize=7, markerscale=3)


def make_plots(out_dir, data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []
    # (a) 2-D (Rs, alpha_Rs) and (Rs, theta_E)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    _scatter_overlay(
        axes[0],
        [(data["old_bulk_Rs"], data["old_bulk_alpha"], "tab:blue", "oldbij bulk", 0.15, 3),
         (data["new_pf_Rs"], data["new_pf_alpha"], "tab:green", "newbij pushforward", 0.15, 3),
         (data["old_basin_Rs"], data["old_basin_alpha"], "tab:red", "oldbij chain-0 Q1 (basin)", 0.4, 5)],
        "Rs", "alpha_Rs", "(Rs, alpha_Rs)")
    axes[0].scatter([data["b_Rs"]], [data["b_alpha"]], marker="*", s=260, c="black",
                    zorder=5, label="basin point b")
    _scatter_overlay(
        axes[1],
        [(data["old_bulk_Rs"], data["old_bulk_thetaE"], "tab:blue", "oldbij bulk", 0.15, 3),
         (data["new_native_Rs"], data["new_native_thetaE"], "tab:green", "newbij (native)", 0.15, 3),
         (data["old_basin_Rs"], data["old_basin_thetaE"], "tab:red", "oldbij chain-0 Q1 (basin)", 0.4, 5)],
        "Rs", "theta_E", "(Rs, theta_E)")
    axes[1].scatter([data["b_Rs"]], [data["theta_E_b"]], marker="*", s=260, c="black", zorder=5)
    axes[1].axhline(P1_THETA_E_REF, color="gray", ls="--", lw=1,
                    label=f"theta_E={P1_THETA_E_REF}")
    fig.suptitle("T15 (a) NFW (Rs, alpha_Rs) / (Rs, theta_E) overlays")
    fig.tight_layout()
    p = os.path.join(out_dir, "t15a_ridge_overlay.png"); fig.savefig(p, dpi=120); plt.close(fig)
    paths.append(p)

    # (b) 14-panel 1-D marginals in OLD physical coords: oldbij bulk vs newbij pushforward
    names = data["old_names"]; dim = len(names)
    ncols = 4; nrows = int(np.ceil(dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
    axes = axes.ravel()
    for i, name in enumerate(names):
        ax = axes[i]
        ov = data["old_bulk_theta"][:, i]; nv = data["new_pf_theta_old"][:, i]
        lo = min(ov.min(), nv.min()); hi = max(ov.max(), nv.max())
        bins = np.linspace(lo, hi, 60)
        ax.hist(ov, bins=bins, density=True, histtype="step", color="tab:blue", label="oldbij bulk")
        ax.hist(nv, bins=bins, density=True, histtype="step", color="tab:green", label="newbij pf")
        ax.axvline(data["b_theta_old"][i], color="tab:red", ls="--", lw=1, label="basin b")
        shift = data["p3_shifts"][i]
        star = " *" if (data["p3_is_nfw"][i] is False and shift > P3_SIGMA) else ""
        ax.set_title(f"{name.split('/')[-1]}\nshift={shift:.2f}sig{star}", fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=6)
    for ax in axes[dim:]:
        ax.axis("off")
    fig.suptitle(f"T15 (b) OLD-coord 1-D marginals; max shift={data['p3_max_shift']:.2f}sig "
                 f"({data['p3_max_shift_param']})", y=1.0)
    fig.tight_layout()
    p = os.path.join(out_dir, "t15b_marginals_oldcoords.png"); fig.savefig(p, dpi=120); plt.close(fig)
    paths.append(p)

    # (c) crowding bar chart
    fig, ax = plt.subplots(figsize=(12, 5))
    labels, old_vals, new_vals = [], [], []
    for i, name in enumerate(data["old_names"]):
        oc = data["crowd_old"][i]["crowding_frac"]
        # match by FULL name (leaf names collide across components); the old arm's
        # alpha_Rs pairs with the new arm's theta_E
        target = data["thetaE_key"] if name == data["alpha_key"] else name
        nc = next((r["crowding_frac"] for r in data["crowd_new"]
                   if r["param"] == target), None)
        if oc is None and nc is None:
            continue
        labels.append(name.split("/")[-1])
        old_vals.append(oc if oc is not None else 0.0)
        new_vals.append(nc if nc is not None else 0.0)
    xpos = np.arange(len(labels))
    ax.bar(xpos - 0.2, old_vals, width=0.4, color="tab:blue", label="oldbij")
    ax.bar(xpos + 0.2, new_vals, width=0.4, color="tab:green", label="newbij")
    ax.axhline(CROWD_FLAG, color="red", ls="--", lw=1, label=f"flag={CROWD_FLAG}")
    ax.set_xticks(xpos); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("outer-1% crowding fraction"); ax.set_title("T15 (c) bound-crowding census")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(out_dir, "t15c_crowding.png"); fig.savefig(p, dpi=120); plt.close(fig)
    paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# reference-run samples
# ---------------------------------------------------------------------------
def load_samples_z(mclmc_dir):
    """(8, 10000, 14) float64 post-burn-in samples_z from a reference mclmc dir."""
    p = os.path.join(mclmc_dir, "arrays.npz")
    d = np.load(p)
    if "samples_z" not in d.files:
        raise KeyError(f"{p} missing samples_z; has {d.files}")
    sz = np.asarray(d["samples_z"], dtype=np.float64)
    if sz.ndim != 3:
        raise ValueError(f"samples_z must be (chains, draws, dim); got {sz.shape}")
    return sz


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old-data-dir", required=True,
                    help="systems/carousel_min_old (NFW_ELLIPSE / alpha_Rs)")
    ap.add_argument("--new-data-dir", required=True,
                    help="systems/carousel_min_new (NFW_ELLIPSE_EINSTEIN / theta_E)")
    ap.add_argument("--old-run", required=True,
                    help="oldbij mclmc dir (arrays.npz samples_z)")
    ap.add_argument("--new-run", required=True,
                    help="newbij mclmc dir (arrays.npz samples_z)")
    ap.add_argument("--out-dir", default=None,
                    help=f"output dir (default: <harness>/{OUT_SUBDIR})")
    args = ap.parse_args()

    assert_x64()  # fail loudly before touching the model if not float64
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.out_dir or os.path.join(here, OUT_SUBDIR)
    os.makedirs(out_dir, exist_ok=True)

    theta_E_to_alpha_Rs, alpha_Rs_to_theta_E = _make_conversion()

    # -- 1. load both systems + names + cross-map --------------------------
    old_pm, _oqz, _oc, old_dim, _ = load_target(args.old_data_dir)
    new_pm, _nqz, _nc, new_dim, _ = load_target(args.new_data_dir)
    old_names = derive_names(old_pm, old_dim)
    new_names = derive_names(new_pm, new_dim)
    if old_dim != new_dim:
        raise ValueError(f"dim mismatch old={old_dim} new={new_dim}")

    # cross-map: differ in exactly one param (alpha_Rs vs theta_E)
    old_only = [n for n in old_names if n not in set(new_names)]
    new_only = [n for n in new_names if n not in set(old_names)]
    if not (len(old_only) == 1 and len(new_only) == 1):
        raise ValueError(f"arms differ in != 1 param: old_only={old_only} new_only={new_only}")
    alpha_key = old_only[0]; thetaE_key = new_only[0]
    if not alpha_key.endswith("alpha_Rs"):
        raise ValueError(f"old-only key not alpha_Rs: {alpha_key}")
    if not thetaE_key.endswith("theta_E"):
        raise ValueError(f"new-only key not theta_E: {thetaE_key}")
    prefix_old = alpha_key[: -len("alpha_Rs")]
    prefix_new = thetaE_key[: -len("theta_E")]
    if prefix_old != prefix_new:
        raise ValueError(f"NFW arm path prefix mismatch: {prefix_old!r} vs {prefix_new!r}")
    Rs_key = prefix_old + "Rs"
    if Rs_key not in old_names or Rs_key not in new_names:
        raise ValueError(f"Rs sibling key {Rs_key} missing from an arm")
    i_alpha = old_names.index(alpha_key)
    i_Rs_old = old_names.index(Rs_key)
    i_thetaE = new_names.index(thetaE_key)
    i_Rs_new = new_names.index(Rs_key)
    newbij_z0_name = new_names[0]  # newbij worst-ESS coordinate z[0]
    print(f"[t15] alpha_Rs key={alpha_key} (idx {i_alpha}); theta_E key={thetaE_key} "
          f"(idx {i_thetaE}); Rs key={Rs_key}")
    print(f"[t15] newbij z[0] physical name = {newbij_z0_name}")

    # -- 2. z -> physical for ALL samples (both runs) ----------------------
    old_sz = load_samples_z(args.old_run)   # (8, 10000, 14)
    new_sz = load_samples_z(args.new_run)
    old_z = old_sz.reshape(-1, old_dim)
    new_z = new_sz.reshape(-1, new_dim)
    theta_old = z_to_physical(old_pm, old_z, old_names)   # (N, dim) old coords
    theta_new = z_to_physical(new_pm, new_z, new_names)   # (N, dim) new coords
    np.savez(os.path.join(out_dir, "theta_old.npz"),
             theta=theta_old, names=np.array(old_names))
    np.savez(os.path.join(out_dir, "theta_new.npz"),
             theta=theta_new, names=np.array(new_names))
    print(f"[t15] saved theta_old {theta_old.shape}, theta_new {theta_new.shape}")

    # -- 3. crowding census ------------------------------------------------
    crowd_old = bound_crowding(old_pm, theta_old, old_names)
    crowd_new = bound_crowding(new_pm, theta_new, new_names)
    p4_flags = {
        "old": [r["param"] for r in crowd_old if r.get("flagged")],
        "new": [r["param"] for r in crowd_new if r.get("flagged")],
    }
    print(f"[t15] P4 crowding flags (>{CROWD_FLAG}): {p4_flags}")

    # -- 4. basin / main-mode points --------------------------------------
    b_z = old_sz[0, 0:BASIN_DRAWS, :].mean(axis=0)          # (dim,)
    m_z = old_sz[1:8, :, :].reshape(-1, old_dim).mean(axis=0)
    b_theta = z_to_physical(old_pm, b_z[None], old_names)[0]  # old coords
    m_theta = z_to_physical(old_pm, m_z[None], old_names)[0]
    b_Rs, b_alpha = float(b_theta[i_Rs_old]), float(b_theta[i_alpha])
    m_Rs, m_alpha = float(m_theta[i_Rs_old]), float(m_theta[i_alpha])
    theta_E_b = float(alpha_Rs_to_theta_E(b_alpha, b_Rs)[0])
    theta_E_m = float(alpha_Rs_to_theta_E(m_alpha, m_Rs)[0])
    print(f"[t15] basin b: Rs={b_Rs:.4f} alpha_Rs={b_alpha:.4f} -> theta_E={theta_E_b:.4f}")
    print(f"[t15] main  m: Rs={m_Rs:.4f} alpha_Rs={m_alpha:.4f} -> theta_E={theta_E_m:.4f}")

    # -- 5. log-prob split at b and m under BOTH arms ----------------------
    # OLD arm: b, m are native old-arm z points.
    import jax.numpy as jnp
    z_bm_old = np.stack([b_z, m_z], axis=0)   # (2, dim)
    ll_old, _ = old_pm.log_like(jnp.asarray(z_bm_old))
    lp_old = old_pm.log_prior(jnp.asarray(z_bm_old))
    ll_old = np.asarray(ll_old, dtype=np.float64).reshape(-1)
    lp_old = np.asarray(lp_old, dtype=np.float64).reshape(-1)

    # NEW arm: map each old point's physical -> swap alpha_Rs->theta_E -> new bij inverse.
    def old_phys_to_new_z(theta_old_pt, Rs_val, alpha_val):
        te = float(alpha_Rs_to_theta_E(alpha_val, Rs_val)[0])
        new_phys = {}
        for j, name in enumerate(new_names):
            if name == thetaE_key:
                new_phys[name] = te
            else:
                new_phys[name] = float(theta_old_pt[old_names.index(name)])
        z_new = physical_dict_to_z(new_pm, new_phys, new_names)  # (1, dim)
        # round-trip verify: z_new -> physical -> matches new_phys to 1e-8
        rt = z_to_physical(new_pm, z_new, new_names)[0]
        err = max(abs(rt[k] - new_phys[new_names[k]]) for k in range(new_dim))
        if err > 1e-8:
            raise ValueError(f"new-arm round-trip error {err:.3e} > 1e-8 (theta_E={te})")
        return z_new[0], te, float(err)

    z_b_new, te_b_chk, rt_b = old_phys_to_new_z(b_theta, b_Rs, b_alpha)
    z_m_new, te_m_chk, rt_m = old_phys_to_new_z(m_theta, m_Rs, m_alpha)
    print(f"[t15] new-arm round-trip err: b={rt_b:.2e} m={rt_m:.2e}")
    z_bm_new = np.stack([z_b_new, z_m_new], axis=0)
    ll_new, _ = new_pm.log_like(jnp.asarray(z_bm_new))
    lp_new = new_pm.log_prior(jnp.asarray(z_bm_new))
    ll_new = np.asarray(ll_new, dtype=np.float64).reshape(-1)
    lp_new = np.asarray(lp_new, dtype=np.float64).reshape(-1)

    logp_split = {
        "old_arm": {"loglike_b": float(ll_old[0]), "loglike_m": float(ll_old[1]),
                    "logprior_b": float(lp_old[0]), "logprior_m": float(lp_old[1]),
                    "delta_loglike_b_minus_m": float(ll_old[0] - ll_old[1]),
                    "delta_logprior_b_minus_m": float(lp_old[0] - lp_old[1])},
        "new_arm": {"loglike_b": float(ll_new[0]), "loglike_m": float(ll_new[1]),
                    "logprior_b": float(lp_new[0]), "logprior_m": float(lp_new[1]),
                    "delta_loglike_b_minus_m": float(ll_new[0] - ll_new[1]),
                    "delta_logprior_b_minus_m": float(lp_new[0] - lp_new[1])},
    }

    # REGISTERED P1 / F1 (F1 uses the OLD-arm delta_loglike -- the basin lives there)
    dtheta_b = abs(theta_E_b - P1_THETA_E_REF)
    delta_loglike_bm = logp_split["old_arm"]["delta_loglike_b_minus_m"]
    meets_P1 = bool(dtheta_b >= P1_MARGIN)
    fires_F1 = bool((dtheta_b < F1_MARGIN) and (delta_loglike_bm >= F1_LOGLIKE_NATS))
    print(f"[t15] P1: |theta_E(b)-{P1_THETA_E_REF}|={dtheta_b:.3f} "
          f"(>= {P1_MARGIN}? {meets_P1}); F1 fires? {fires_F1} "
          f"(delta_loglike_bm={delta_loglike_bm:.3f})")

    # -- 6. ridge curvature ------------------------------------------------
    # newbij pushforward into OLD coords: Rs stays, alpha_Rs = conv(theta_E, Rs).
    new_Rs = theta_new[:, i_Rs_new]
    new_thetaE = theta_new[:, i_thetaE]
    new_alpha_pf = theta_E_to_alpha_Rs(new_thetaE, new_Rs)
    ridge_new = ridge_curvature(new_Rs, new_alpha_pf)
    # oldbij EXCLUDING chain-0 draws [0:5000]
    mask_keep = np.ones(old_sz.shape[:2], dtype=bool)
    mask_keep[0, 0:RIDGE_EXCLUDE_CHAIN0] = False
    keep_idx = mask_keep.reshape(-1)
    old_Rs_ridge = theta_old[keep_idx, i_Rs_old]
    old_alpha_ridge = theta_old[keep_idx, i_alpha]
    ridge_old = ridge_curvature(old_Rs_ridge, old_alpha_ridge)
    print(f"[t15] P2 ridge ratio: old={ridge_old['ratio']:.3f} new={ridge_new['ratio']:.3f} "
          f"(P2 >= {P2_RATIO}; F2 < {F2_RATIO})")

    # -- 7. plots ----------------------------------------------------------
    # bulk = chains 1-7 (the main mode, == m definition); basin = chain-0 [0:2500]
    bulk_idx = np.zeros(old_sz.shape[:2], dtype=bool); bulk_idx[1:8, :] = True
    bulk_idx = bulk_idx.reshape(-1)
    basin_mask = np.zeros(old_sz.shape[:2], dtype=bool); basin_mask[0, 0:BASIN_DRAWS] = True
    basin_mask = basin_mask.reshape(-1)
    old_bulk_theta = theta_old[bulk_idx]
    # newbij pushforward into OLD coords (full theta table w/ alpha in place of theta_E)
    new_pf_theta_old = theta_new.copy()
    new_pf_theta_old[:, i_thetaE] = new_alpha_pf  # column i_thetaE now holds alpha_Rs
    # REORDER columns from NEW-names sorted order into OLD-names sorted order.
    # The two orders differ (alpha_Rs sorts after Rs; theta_E sorts after e2), so
    # without this the P3 shifts and marginal panels would compare mismatched params.
    col_map = [i_thetaE if n == alpha_key else new_names.index(n) for n in old_names]
    new_pf_theta_old = new_pf_theta_old[:, col_map]
    # basin theta_E via per-sample inverse (basin cloud is small: 2500 pts)
    old_basin_theta = theta_old[basin_mask]
    old_basin_thetaE = alpha_Rs_to_theta_E(old_basin_theta[:, i_alpha], old_basin_theta[:, i_Rs_old])
    old_bulk_thetaE = alpha_Rs_to_theta_E(old_bulk_theta[:, i_alpha], old_bulk_theta[:, i_Rs_old])

    # P3: 1-D marginal shifts (old bulk vs new pushforward) in pooled sigma
    p3_shifts = np.zeros(old_dim)
    p3_is_nfw = []
    nfw_leaf = {"Rs", "alpha_Rs"}
    for i, name in enumerate(old_names):
        ov = old_bulk_theta[:, i]; nv = new_pf_theta_old[:, i]
        pooled = np.sqrt(0.5 * (ov.var() + nv.var()))
        p3_shifts[i] = abs(ov.mean() - nv.mean()) / pooled if pooled > 0 else 0.0
        p3_is_nfw.append(name.split("/")[-1] in nfw_leaf)
    # non-NFW max shift = P3 leakage headline
    non_nfw_shifts = [(p3_shifts[i], old_names[i]) for i in range(old_dim) if not p3_is_nfw[i]]
    p3_max_shift, p3_max_shift_param = max(non_nfw_shifts, key=lambda t: t[0])
    p3_leak_flags = [n for s, n in non_nfw_shifts if s > P3_SIGMA]
    print(f"[t15] P3 max non-NFW 1-D shift = {p3_max_shift:.3f} sigma ({p3_max_shift_param}); "
          f"leak flags: {p3_leak_flags}")

    plot_data = {
        "old_names": old_names,
        "old_bulk_theta": old_bulk_theta,
        "new_pf_theta_old": new_pf_theta_old,
        "b_theta_old": b_theta,
        "old_bulk_Rs": old_bulk_theta[:, i_Rs_old], "old_bulk_alpha": old_bulk_theta[:, i_alpha],
        "old_bulk_thetaE": old_bulk_thetaE,
        "old_basin_Rs": old_basin_theta[:, i_Rs_old], "old_basin_alpha": old_basin_theta[:, i_alpha],
        "old_basin_thetaE": old_basin_thetaE,
        "new_pf_Rs": new_Rs, "new_pf_alpha": new_alpha_pf,
        "new_native_Rs": new_Rs, "new_native_thetaE": new_thetaE,
        "b_Rs": b_Rs, "b_alpha": b_alpha, "theta_E_b": theta_E_b,
        "p3_shifts": p3_shifts, "p3_is_nfw": p3_is_nfw,
        "p3_max_shift": p3_max_shift, "p3_max_shift_param": p3_max_shift_param,
        "crowd_old": crowd_old, "crowd_new": crowd_new,
        "alpha_key": alpha_key, "thetaE_key": thetaE_key,
    }
    plot_paths = make_plots(out_dir, plot_data)
    print(f"[t15] wrote plots: {plot_paths}")

    # -- 8. summary.json ---------------------------------------------------
    summary = {
        "experiment": "T15_carousel_decompose_phaseA",
        "status": "proposed (UNCERTIFIED)",
        "old_data_dir": os.path.abspath(args.old_data_dir),
        "new_data_dir": os.path.abspath(args.new_data_dir),
        "old_run": os.path.abspath(args.old_run),
        "new_run": os.path.abspath(args.new_run),
        "dim": int(old_dim),
        "old_names": old_names,
        "new_names": new_names,
        "cross_map": {"alpha_Rs_key": alpha_key, "theta_E_key": thetaE_key,
                      "Rs_key": Rs_key, "shared_params": len(old_names) - 1},
        "newbij_z0_physical_name": newbij_z0_name,
        # step 4/5
        "basin_point_z_mean_draws": [0, BASIN_DRAWS],
        "b_physical": {old_names[k]: float(b_theta[k]) for k in range(old_dim)},
        "m_physical": {old_names[k]: float(m_theta[k]) for k in range(old_dim)},
        "theta_E_b": theta_E_b, "theta_E_m": theta_E_m,
        "logp_split": logp_split,
        "new_arm_roundtrip_err": {"b": rt_b, "m": rt_m},
        # REGISTERED P1/F1
        "P1_predicts_abs_thetaE_b_minus_13_ge_2": {
            "value_abs_diff": float(dtheta_b), "threshold": P1_MARGIN, "meets_P1": meets_P1},
        "F1": {"abs_diff": float(dtheta_b), "margin": F1_MARGIN,
               "delta_loglike_b_minus_m_old_arm": float(delta_loglike_bm),
               "nats_threshold": F1_LOGLIKE_NATS, "fires_F1": fires_F1},
        # REGISTERED P2/F2
        "P2_F2_ridge": {"old_arm": ridge_old, "new_arm": ridge_new,
                        "P2_threshold": P2_RATIO, "F2_threshold": F2_RATIO,
                        "old_excludes_chain0_draws": [0, RIDGE_EXCLUDE_CHAIN0]},
        # REGISTERED P3
        "P3_max_nonNFW_1d_shift_pooled_sigma": {
            "value": float(p3_max_shift), "param": p3_max_shift_param,
            "threshold": P3_SIGMA, "leak_flagged_params": p3_leak_flags,
            "all_shifts": {old_names[k]: float(p3_shifts[k]) for k in range(old_dim)}},
        # REGISTERED P4
        "P4_crowding_flag_threshold": CROWD_FLAG,
        "P4_flags": p4_flags,
        "crowding_old": crowd_old,
        "crowding_new": crowd_new,
        "plot_paths": plot_paths,
        "verdict": ("proposed (UNCERTIFIED) -- registered quantities computed; a grader "
                    "must inspect the plots/numbers. Movement toward a prediction is NOT "
                    "success."),
    }
    sp = os.path.join(out_dir, "summary.json")
    with open(sp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[t15] wrote {sp}")
    print("[t15] verdict: proposed (UNCERTIFIED).")


if __name__ == "__main__":
    main()
