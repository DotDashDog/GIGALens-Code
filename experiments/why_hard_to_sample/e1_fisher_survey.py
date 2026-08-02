"""E1 -- Fisher-metric survey (T6) + bijector curvature contribution (T7).

Pre-registered in docs/logs/why-hard-to-sample.md (checkpoints T6 and T7, both
approved by the human on 2026-07-02) and driven by the T3 finding: on sys60 the
posterior is a smooth, strongly CURVED stiff valley (T1 clone with identical
covariance samples ~130x faster; T3 log-density smooth to machine-relevant
scales; local curvature ~100x the sigma-scale average with a correlation length
~0.07 sigma). T6 asks: does the stiff subspace of the local Gauss-Newton metric
M(z) = J^T W J (J = image Jacobian, W = 1/err^2) ROTATE as the point moves along
the typical set, in all 22 dimensions? T7 asks: how much of that rotation is
manufactured by the diagonal bijector bending a straighter theta-space ridge?

This script PRODUCES artifacts and PROPOSED (UNCERTIFIED) verdict fields; it does
NOT adjudicate. A grader inspects the plots/JSON, not this summary.

Design (verbatim from the checkpoints):
  A. RENDER PATH + HARD chi^2 GATE. In mode="forward" the likelihood's model
     image is exactly SceneSimulator.simulate(params) with
     params = model.to_params(bij.forward(list(z.T))) (scene_prob_model.py
     _model_image/_dataset_chi2). We rebuild render(z) -> flattened model image
     and W = 1/err_map^2, then at >=3 typical-set points recompute
     chi^2 = sum W*(obs-render)^2 and compare to the aux from log_prob(z)[1].
     CONVENTION (reconciled from scene_prob_model.ProbModel.log_like):
     log_prob(z)[1] is the REDUCED chi^2 = chi2_total / event_size, with
     event_size = number of unmasked pixels (sys60: mask all-ones, 80*80=6400).
     So chi^2_from_aux = log_prob(z)[1] * prob_model.event_size. err_map is FIXED
     (Dataset builds it once from the OBSERVED image via background_rms=0.2,
     exp_time=100; it does NOT depend on z), so W is constant and M = J^T W J is
     the exact Gauss-Newton part of -Hessian(log_like). The Gaussian
     log-normalization is z-independent for the same reason. Gate: relative match
     must be <=1e-6 (expected ~1e-13); otherwise RAISE -- a wrong render path
     invalidates everything downstream.
  B. POINT SETS (seeded). 32 iid draws from pooled post-burn-in samples for
     loading statistics; same-chain lag pairs for rotation-vs-separation: lags
     {1,4,16,64,256,1024,4096}, 8 pairs per lag (random chain + start), recording
     every pair's actual ||Delta z||.
  C. METRIC + ROTATION (T6). J = jax.jacfwd(render)(z) (6400x22, forward-mode is
     ideal: 22 JVPs, out-dim >> in-dim); M = J^T (W[:,None]*J), symmetrized,
     float64. Per pair: principal angles between the k=3 STIFFEST subspaces
     (largest eigenvalues of M) and the angle between top eigenvectors. At 6 of
     the 32 iid points: full posterior Hessian H_post = -hessian(log_prob[0]) and
     r = ||P_stiff (H_post - M) P_stiff|| / ||P_stiff M P_stiff|| on the stiff
     subspace (P_stiff from M, k=3). H_post includes prior+logdet terms; that is
     fine for r's purpose -- r bounds EVERYTHING-not-GN (residual second-order +
     prior) on the stiff subspace. r<<1 => GN/manifold curvature dominant;
     r>~1 => residual/prior terms matter. Reported, not adjudicated.
  D. THETA-SPACE (T7). First verify the bijector is per-coordinate: J_bij =
     jacobian of the 22-vector map bij.forward (stacked in sorted-key /
     sampler-column order) -- off-diagonal must be ~0 (C-8 verification pattern);
     RAISE if not. Then s_i = dtheta_i/dz_i = diag(J_bij); J_theta = J_z / s
     (column-wise); M_theta = J_theta^T W J_theta. Standardize per coordinate by
     the posterior std (z std from samples, theta std from transformed samples):
     work in zhat = z/std_z and thetahat = theta/std_theta, so J_zhat = J_z*std_z,
     J_thetahat = J_theta*std_theta, and separations Delta zhat = Delta z/std_z,
     Delta thetahat = Delta theta/std_theta. Rotation-vs-separation in BOTH spaces
     on one axis; ratio of median top-vector rotation at matched separation bins.
  E. OUTPUTS. JSON (pair table, per-point spectra + loadings with param NAMES, r
     values, chi^2 verification numbers, verdict fields) + plots + printed
     summary restating T6's prediction (>=15 deg) / falsifier (<3 deg) and T7's
     ratio branches next to the measured numbers.

jax is imported ONLY inside functions so the module imports cleanly under a plain
conda python (no jax, no scipy) for offline smoke tests of the pure-numpy /
principal-angle machinery (--smoke). scipy.linalg.subspace_angles is used when
importable (matches the pre-registration); a numpy SVD implementation is the
offline fallback and they agree.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --- pre-registered constants (from the T3/T6/T7 checkpoints) --------------
EPS_MOVE = 1.28e-3    # per-step z-move (z-units), T3 log
H_STAR = 3.5e-3       # curvature transition scale h* (z-units), T3 log
LAGS = (1, 4, 16, 64, 256, 1024, 4096)
PAIRS_PER_LAG = 8
N_IID = 32
N_HESS = 6            # subset of the 32 iid points for the full-Hessian r read
K_STIFF = 3
T6_PREDICT_DEG = 15.0     # top-vector rotation at ~h* separations
T6_FALSIFY_DEG = 3.0      # median rotation < this at h*-scale => curved-valley read wrong
T7_JACKPOT_RATIO = 3.0    # z/theta rotation ratio >~ this => transform-induced curvature
TINY = 1e-300


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# Pure-numpy / linear-algebra machinery (no jax) -- offline-testable
# ===========================================================================

def gn_metric(J, W):
    """Gauss-Newton metric M = J^T diag(W) J, symmetrized, float64.

    J: (m, n) image Jacobian; W: (m,) pixel weights 1/err^2. This is exactly the
    GN part of the Hessian of 0.5*chi^2 = 0.5*sum W*(obs-render)^2."""
    J = np.asarray(J, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    M = J.T @ (W[:, None] * J)
    return 0.5 * (M + M.T)


def scale_cols(J, s):
    """Column-wise scale: (J * s)[:, i] = J[:, i] * s[i] (chain-rule reweight)."""
    return np.asarray(J, dtype=np.float64) * np.asarray(s, dtype=np.float64)[None, :]


def eig_desc(M):
    """Eigen-decomposition of a symmetric matrix, DESCENDING eigenvalues.
    For a curvature/precision matrix M, w[0]/V[:,0] is the STIFFEST direction
    (largest eigenvalue = highest curvature)."""
    w, V = np.linalg.eigh(np.asarray(M, dtype=np.float64))
    order = np.argsort(w)[::-1]
    return w[order], V[:, order]


def stiff_subspace(M, k):
    """Top-k (stiffest = largest-eigenvalue) eigenvectors of M as (n, k), plus the
    k eigenvalues (descending) and the full descending spectrum."""
    w, V = eig_desc(M)
    return V[:, :k], w[:k], w


def _principal_angles_numpy(A, B):
    """Principal angles (radians, ascending) between the column spaces of A and B
    via the standard QR + SVD algorithm (Bjorck-Golub). Offline fallback that
    matches scipy.linalg.subspace_angles for the well-separated subspaces here."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    QA, _ = np.linalg.qr(A)
    QB, _ = np.linalg.qr(B)
    s = np.linalg.svd(QA.T @ QB, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    return np.sort(np.arccos(s))  # ascending


def principal_angles_deg(A, B):
    """Principal angles in DEGREES (ascending). Uses scipy.linalg.subspace_angles
    when importable (the pre-registered routine); numpy fallback otherwise."""
    try:
        from scipy.linalg import subspace_angles
        ang = np.asarray(subspace_angles(np.asarray(A, float), np.asarray(B, float)))
        ang = np.sort(ang)
    except Exception:
        ang = _principal_angles_numpy(A, B)
    return np.rad2deg(ang)


def top_vector_angle_deg(va, vb):
    """Angle (deg) between two vectors, sign-insensitive (eigenvector sign is
    arbitrary): arccos(|va.vb| / (|va||vb|))."""
    va = np.asarray(va, dtype=np.float64)
    vb = np.asarray(vb, dtype=np.float64)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return float("nan")
    c = abs(float(va @ vb) / denom)
    return float(np.degrees(np.arccos(min(1.0, max(-1.0, c)))))


def stiff_projector(M, k):
    """Orthogonal projector P = Vs Vs^T onto the stiff (top-k) subspace of M."""
    Vs, _, _ = stiff_subspace(M, k)
    return Vs @ Vs.T, Vs


def r_value(H_post, M, k, ord_="fro"):
    """r = ||P (H_post - M) P|| / ||P M P|| on the stiff subspace of M (P from M).

    r << 1  => GN/manifold curvature dominates the stiff subspace;
    r >~ 1  => residual second-order + prior/logdet terms matter there.
    Frobenius norm by default; spectral (2-norm) also returned by the caller."""
    P, _ = stiff_projector(M, k)
    num = np.linalg.norm(P @ (np.asarray(H_post) - np.asarray(M)) @ P, ord_)
    den = np.linalg.norm(P @ np.asarray(M) @ P, ord_)
    return float(num / den) if den > 0 else float("inf")


def rotation_between_metrics(Ma, Mb, k):
    """Rotation summary between two GN metrics: principal angles (deg, ascending)
    of their stiff k-subspaces + max + mean, and the top-eigenvector angle."""
    Va, _, _ = stiff_subspace(Ma, k)
    Vb, _, _ = stiff_subspace(Mb, k)
    pa = principal_angles_deg(Va, Vb)
    return {
        "principal_angles_deg": pa.tolist(),
        "max_principal_angle_deg": float(np.max(pa)),
        "mean_principal_angle_deg": float(np.mean(pa)),
        "top_vector_angle_deg": top_vector_angle_deg(Va[:, 0], Vb[:, 0]),
    }


def binned_median(seps, vals, edges):
    """Median of vals within each [edges[i], edges[i+1]) bin of seps. Returns
    (centers, medians, counts); NaN median where a bin is empty."""
    seps = np.asarray(seps, dtype=np.float64)
    vals = np.asarray(vals, dtype=np.float64)
    centers, meds, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (seps >= lo) & (seps < hi)
        centers.append(np.sqrt(lo * hi))
        counts.append(int(m.sum()))
        meds.append(float(np.median(vals[m])) if m.any() else float("nan"))
    return np.asarray(centers), np.asarray(meds), np.asarray(counts)


# ===========================================================================
# Offline smoke tests (numpy only) -- validate the machinery without jax/GPU
# ===========================================================================

def _givens(n, i, j, phi):
    G = np.eye(n)
    c, s = np.cos(phi), np.sin(phi)
    G[i, i] = c; G[i, j] = -s; G[j, i] = s; G[j, j] = c
    return G


def smoke_principal_angles(verbose=True):
    """Rotate a known-metric's eigenbasis by a known angle and recover it.

    (A) rotate an in-subspace/out-of-subspace plane (2,3) by phi_s => the top-3
        stiff subspace's MAX principal angle must recover phi_s.
    (B) rotate the top vector against an out-of-subspace axis (0, n-1) by phi_t =>
        the top-eigenvector angle must recover phi_t."""
    rng = np.random.default_rng(0)
    n, k = 22, 3
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    eigs = np.geomspace(1e3, 1.0, n)          # descending => cols 0..2 are stiff
    M0 = Q @ np.diag(eigs) @ Q.T

    phi_s = np.radians(20.0)
    Qs = Q @ _givens(n, 2, 3, phi_s)          # moves span{0,1,2}
    Ms = Qs @ np.diag(eigs) @ Qs.T
    V0, _, _ = stiff_subspace(M0, k)
    Vs, _, _ = stiff_subspace(Ms, k)
    got_sub = float(np.max(principal_angles_deg(V0, Vs)))

    phi_t = np.radians(15.0)
    Qt = Q @ _givens(n, 0, n - 1, phi_t)      # rotates the top vector
    Mt = Qt @ np.diag(eigs) @ Qt.T
    Vt, _, _ = stiff_subspace(Mt, k)
    got_top = top_vector_angle_deg(V0[:, 0], Vt[:, 0])

    ok = abs(got_sub - 20.0) < 0.5 and abs(got_top - 15.0) < 0.5
    if verbose:
        print(f"[smoke] principal-angle recovery: subspace max angle "
              f"{got_sub:.3f} deg (expect 20.000); top-vector angle "
              f"{got_top:.3f} deg (expect 15.000) -> {'PASS' if ok else 'FAIL'}")
    return {"subspace_angle_deg": got_sub, "expected_subspace_deg": 20.0,
            "top_vector_angle_deg": got_top, "expected_top_vector_deg": 15.0,
            "pass": bool(ok)}


def smoke_transform_induced(verbose=True):
    """Diagonal-bijector toy: a STRAIGHT theta-space ridge (constant image
    Jacobian A => M_theta constant, zero rotation) warped by theta=exp(z) so that
    J_z = A*diag(exp(z)) and M_z rotates as the point moves. Confirms the
    diagnostic detects transform-induced curvature: z-space rotation >> theta.
    Standardization is exercised with per-coordinate std (constant here, so it
    cannot resurrect a rotation in theta)."""
    rng = np.random.default_rng(1)
    n, m, k = 4, 20, 2
    A = rng.standard_normal((m, n))           # image Jac wrt theta (constant ridge)
    W = np.ones(m)
    std = np.array([1.3, 0.7, 1.1, 0.9])       # per-coordinate standardization
    z_a = rng.standard_normal(n) * 0.3
    z_b = z_a + rng.standard_normal(n) * 0.06

    def metrics(z):
        theta = np.exp(z)
        s = theta                              # dtheta/dz for theta=exp(z)
        J_theta = A                            # constant
        J_z = scale_cols(J_theta, s)           # chain rule J_z = J_theta * s
        M_th = gn_metric(scale_cols(J_theta, std), W)   # thetahat metric
        M_z = gn_metric(scale_cols(J_z, std), W)        # zhat metric
        return M_th, M_z

    Mta, Mza = metrics(z_a)
    Mtb, Mzb = metrics(z_b)
    ang_theta = rotation_between_metrics(Mta, Mtb, k)["max_principal_angle_deg"]
    ang_z = rotation_between_metrics(Mza, Mzb, k)["max_principal_angle_deg"]
    ok = (ang_theta < 1e-6) and (ang_z > 1.0)
    if verbose:
        print(f"[smoke] transform-induced curvature: theta-space rotation "
              f"{ang_theta:.3e} deg (expect ~0, straight ridge); z-space rotation "
              f"{ang_z:.3f} deg (expect >0) -> {'PASS' if ok else 'FAIL'}")
    return {"theta_rotation_deg": ang_theta, "z_rotation_deg": ang_z,
            "pass": bool(ok)}


def run_smoke():
    print("=== E1 offline smoke tests (numpy only; no jax/GPU) ===")
    a = smoke_principal_angles()
    b = smoke_transform_induced()
    allpass = a["pass"] and b["pass"]
    print(f"[smoke] overall: {'PASS' if allpass else 'FAIL'}")
    return {"principal_angles": a, "transform_induced": b, "pass": bool(allpass)}


# ===========================================================================
# jax-backed builders (imported lazily; container-only)
# ===========================================================================

def build_jax_ops(prob_model, param_names):
    """Return a dict of jitted operators over the sys60 forward model:

      render(z)      -> (6400,) flattened model image  == the likelihood's image
      jac_render(z)  -> (6400, 22) image Jacobian (jacfwd)
      bij_vec(z)     -> (22,) theta = bij.forward(z) stacked in sorted-key order
      jac_bij(z)     -> (22, 22) bijector Jacobian (must be diagonal)
      bij_batch(Z)   -> (N, 22) batched theta for std estimation
      red_chi2(z)    -> scalar reduced chi^2 aux == log_prob(z)[1]
      hess_logp(z)   -> (22, 22) Hessian of log_prob(z)[0]
      W, obs         -> (6400,) pixel weights 1/err^2 and observed image
      event_size     -> int (unmasked pixel count)
    """
    import jax
    import jax.numpy as jnp

    pm = prob_model
    sim = pm.simulators[0]

    def render(z):
        x = pm.bij.forward(list(z.T))
        params = pm.model.to_params(x)
        im = sim.simulate(params)
        return im.reshape(-1)

    def bij_vec(z):
        out = pm.bij.forward(list(z.T))
        return jnp.stack([out[k] for k in param_names])

    def bij_batch(Z):
        out = pm.bij.forward(list(jnp.asarray(Z).T))
        return jnp.stack([out[k] for k in param_names], axis=-1)  # (N, 22)

    def red_chi2(z):
        return pm.log_prob(z)[1]

    def logp0(z):
        return pm.log_prob(z)[0]

    ops = {
        "render": jax.jit(render),
        "jac_render": jax.jit(jax.jacfwd(render)),
        "bij_vec": jax.jit(bij_vec),
        "jac_bij": jax.jit(jax.jacfwd(bij_vec)),
        "bij_batch": jax.jit(bij_batch),
        "red_chi2": jax.jit(red_chi2),
        "hess_logp": jax.jit(jax.hessian(logp0)),
    }

    err = np.asarray(pm.error_map, dtype=np.float64).reshape(-1)
    obs = np.asarray(pm.observed_image, dtype=np.float64).reshape(-1)
    ops["W"] = 1.0 / err ** 2
    ops["obs"] = obs
    ops["event_size"] = int(pm.event_size)
    return ops


# ===========================================================================
# Point sets
# ===========================================================================

def load_pooled(samples_path):
    d = np.load(samples_path)
    if "position" not in d.files:
        raise KeyError(f"'position' not in {samples_path}; keys={d.files}")
    pos = np.asarray(d["position"], dtype=np.float64)
    if pos.ndim != 3:
        raise ValueError(f"position must be (chains, steps, dim); got {pos.shape}")
    return pos


def draw_iid(pos, n, rng):
    n_chains, n_steps, dim = pos.shape
    flat = pos.reshape(-1, dim)
    idx = rng.choice(flat.shape[0], size=n, replace=False)
    return flat[idx], idx.tolist()


def build_pairs(pos, lags, per_lag, rng):
    n_chains, n_steps, dim = pos.shape
    pairs = []
    for lag in lags:
        if lag >= n_steps:
            continue
        for _ in range(per_lag):
            c = int(rng.integers(n_chains))
            s = int(rng.integers(0, n_steps - lag))
            pairs.append({"lag": int(lag), "chain": c, "start": s,
                          "za": pos[c, s].copy(), "zb": pos[c, s + lag].copy()})
    return pairs


# ===========================================================================
# Per-point metric assembly
# ===========================================================================

def metrics_at(z, ops, W, std_z, std_theta, verify_bij=False, bij_tol=1e-6):
    """Compute the raw, zhat and thetahat GN metrics at one point z.

    Returns dict with M_raw (raw z units, for r + spectrum), M_zhat, M_thetahat
    (standardized, for rotation), s (bijector diagonal), and diagnostics."""
    import numpy as _np
    z = _np.asarray(z, dtype=_np.float64)
    Jz = _np.asarray(ops["jac_render"](z), dtype=_np.float64)     # (6400, 22)
    Jb = _np.asarray(ops["jac_bij"](z), dtype=_np.float64)        # (22, 22)
    s = _np.diag(Jb).copy()

    diag_info = None
    if verify_bij:
        off = Jb - _np.diag(_np.diag(Jb))
        max_off = float(_np.max(_np.abs(off)))
        max_diag = float(_np.max(_np.abs(_np.diag(Jb))))
        ratio = max_off / max(max_diag, TINY)
        diag_info = {"max_abs_offdiag": max_off, "max_abs_diag": max_diag,
                     "offdiag_to_diag_ratio": ratio,
                     "min_abs_s": float(_np.min(_np.abs(s)))}
        if ratio > bij_tol:
            raise RuntimeError(
                f"[E1] bijector is NOT per-coordinate at the probe point: "
                f"max|offdiag|/max|diag| = {ratio:.3e} > {bij_tol:.1e}. The "
                "C-8 per-coordinate chain rule J_theta = J_z/diag(dz->theta) "
                "does not apply -- STOP and reconsider T7's construction.")

    M_raw = gn_metric(Jz, W)
    M_zhat = gn_metric(scale_cols(Jz, std_z), W)
    Jtheta = Jz / s[None, :]                       # J wrt theta (per-coord chain rule)
    M_thetahat = gn_metric(scale_cols(Jtheta, std_theta), W)
    return {"M_raw": M_raw, "M_zhat": M_zhat, "M_thetahat": M_thetahat,
            "s": s, "bij_diag": diag_info}


# ===========================================================================
# Plots
# ===========================================================================

def plot_rotation_vs_sep(pairs_out, sigma_bar, out_path):
    """Two panels: (top-vector angle | k=3 mean principal angle) vs standardized
    separation, z and theta overlaid, log-x, with eps_move and h* marked."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    sep_z = np.array([p["sep_zhat"] for p in pairs_out])
    sep_t = np.array([p["sep_thetahat"] for p in pairs_out])
    edges = np.geomspace(min(sep_z.min(), sep_t.min()) * 0.9,
                         max(sep_z.max(), sep_t.max()) * 1.1, 9)

    metrics = [("top_vector_angle_deg", "top-eigenvector rotation"),
               ("mean_principal_angle_deg", "k=3 mean principal angle")]
    for ax, (key, title) in zip(axes, metrics):
        az = np.array([p["z"][key] for p in pairs_out])
        at = np.array([p["theta"][key] for p in pairs_out])
        ax.scatter(sep_z, az, s=18, c="C0", alpha=0.5, label="z-space (pairs)")
        ax.scatter(sep_t, at, s=18, c="C1", alpha=0.5, marker="^",
                   label="theta-space (pairs)")
        cz, mz, _ = binned_median(sep_z, az, edges)
        ct, mt, _ = binned_median(sep_t, at, edges)
        ax.plot(cz, mz, "-o", c="C0", lw=2, label="z median")
        ax.plot(ct, mt, "-^", c="C1", lw=2, label="theta median")
        # sampler scales (z-units -> standardized by sigma_bar)
        for val, lab, col in ((EPS_MOVE, "eps_move", "green"),
                              (H_STAR, "h*", "purple")):
            ax.axvline(val / sigma_bar, color=col, ls=":", lw=1.2,
                       label=f"{lab}/sigma_bar")
        ax.axhline(T6_PREDICT_DEG, color="0.3", ls="--", lw=1,
                   label=f"T6 predict {T6_PREDICT_DEG:g} deg")
        ax.axhline(T6_FALSIFY_DEG, color="crimson", ls="--", lw=1,
                   label=f"T6 falsifier {T6_FALSIFY_DEG:g} deg")
        ax.set_xscale("log")
        ax.set_xlabel("standardized separation ||Delta hat||  (per-coord sigma units)")
        ax.set_ylabel("rotation angle (deg)")
        ax.set_title(title, fontsize=11)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("E1/T6-T7 stiff-subspace rotation vs on-ridge separation "
                 "(z vs theta) -- PROPOSED (UNCERTIFIED)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_loading_heatmap(loadings, names, out_path):
    """|top-eigenvector loadings| across the 32 iid points (M_zhat stiff vector).
    Rows = params (names), cols = points."""
    L = np.asarray(loadings).T  # (dim, n_points)
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(L, aspect="auto", cmap="viridis", vmin=0.0, vmax=float(L.max()))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("iid typical-set point index")
    ax.set_ylabel("parameter (z / sampler-column order)")
    ax.set_title("E1/T6 |top stiff-eigenvector loadings| across 32 points "
                 "(consistency of the stiff family) -- M_zhat", fontsize=11)
    fig.colorbar(im, ax=ax, label="|loading|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_spectra(spectra, clone_inv_spectrum, out_path):
    """M_raw eigenspectra at the 32 points (thin) overlaid with the clone
    covariance's INVERSE spectrum (thick). All descending, semilog-y."""
    fig, ax = plt.subplots(figsize=(9, 6))
    ranks = np.arange(1, len(clone_inv_spectrum) + 1)
    for i, w in enumerate(spectra):
        ax.semilogy(ranks, np.sort(w)[::-1], "-", color="0.6", lw=0.7, alpha=0.5,
                    label="M_raw eigenvalues (32 points)" if i == 0 else None)
    ax.semilogy(ranks, np.sort(clone_inv_spectrum)[::-1], "-", color="crimson",
                lw=2.5, marker="o", ms=4,
                label="clone cov INVERSE spectrum (1/eig Sigma)")
    ax.set_xlabel("eigenvalue rank (descending)")
    ax.set_ylabel("eigenvalue (log scale)")
    ax.set_title("E1/T6 local GN-metric spectra vs global precision "
                 "(clone Sigma^-1) -- PROPOSED (UNCERTIFIED)", fontsize=11)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ===========================================================================
# Main
# ===========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="E1 Fisher-metric survey (T6) + "
                                            "bijector curvature contribution (T7)")
    p.add_argument("--data-dir")
    p.add_argument("--samples", help="clone_source.npz (post-burn-in z-samples)")
    p.add_argument("--clone", help="clone.npz (mean/cov for sigma_bar + spectrum)")
    p.add_argument("--out-dir")
    p.add_argument("--seed", type=int)
    p.add_argument("--smoke", action="store_true",
                   help="run the offline numpy-only smoke tests and exit "
                        "(no jax/GPU/scipy needed)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.smoke:
        res = run_smoke()
        if not res["pass"]:
            raise SystemExit("[E1] smoke tests FAILED")
        return

    for req in ("data_dir", "samples", "clone", "out_dir", "seed"):
        if getattr(args, req) is None:
            raise ValueError(f"--{req.replace('_', '-')} is required (no default).")
    os.makedirs(args.out_dir, exist_ok=True)

    # float64 hard gate (reuse the repo rule)
    from common import assert_x64, load_target
    assert_x64()

    # --- inputs -------------------------------------------------------------
    pos = load_pooled(args.samples)
    n_chains, n_steps, dim = pos.shape
    flat = pos.reshape(-1, dim)
    std_z = np.std(flat, axis=0, ddof=1)
    sigma_bar = float(np.mean(std_z))
    clone = np.load(args.clone)
    clone_cov = np.asarray(clone["cov"], dtype=np.float64)
    if clone_cov.shape != (dim, dim):
        raise ValueError(f"clone cov shape {clone_cov.shape} != ({dim},{dim})")
    clone_inv_spectrum = 1.0 / np.linalg.eigvalsh(clone_cov)

    print(f"[E1] pos shape={pos.shape} sigma_bar={sigma_bar:.4e} "
          f"eps_move/sigma_bar={EPS_MOVE/sigma_bar:.3e} "
          f"h*/sigma_bar={H_STAR/sigma_bar:.3e}")

    prob_model, qz, z_center, dim2, param_names = load_target(args.data_dir)
    if dim2 != dim:
        raise ValueError(f"data-dir dim {dim2} != samples dim {dim}")
    ops = build_jax_ops(prob_model, param_names)
    W = ops["W"]
    obs = ops["obs"]
    event_size = ops["event_size"]
    print(f"[E1] event_size={event_size} (from prob_model); "
          f"names[0..3]={param_names[:4]}")

    # theta stds from transformed pooled samples (one batched forward)
    theta_all = np.asarray(ops["bij_batch"](flat), dtype=np.float64)
    std_theta = np.std(theta_all, axis=0, ddof=1)

    # --- point sets ---------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    iid_z, iid_idx = draw_iid(pos, N_IID, rng)
    pairs = build_pairs(pos, LAGS, PAIRS_PER_LAG, rng)
    print(f"[E1] {len(iid_z)} iid points; {len(pairs)} lag pairs "
          f"(lags={list(LAGS)} x {PAIRS_PER_LAG})")

    # --- Task A: HARD chi^2 verification gate at >=3 iid points --------------
    chi2_checks = []
    for i in range(3):
        z = iid_z[i]
        render = np.asarray(ops["render"](z), dtype=np.float64)
        chi2_mine = float(np.sum(W * (obs - render) ** 2))
        red_aux = float(ops["red_chi2"](z))
        chi2_from_aux = red_aux * event_size
        rel = abs(chi2_mine - chi2_from_aux) / max(abs(chi2_from_aux), TINY)
        chi2_checks.append({"iid_point": i, "chi2_recomputed": chi2_mine,
                            "reduced_chi2_aux": red_aux, "event_size": event_size,
                            "chi2_from_aux": chi2_from_aux, "relative_error": rel})
        print(f"[E1] chi2 gate pt{i}: recomputed={chi2_mine:.10e} "
              f"aux*event_size={chi2_from_aux:.10e} rel={rel:.3e}")
        if rel > 1e-6:
            raise RuntimeError(
                f"[E1] chi^2 reconciliation FAILED at iid point {i}: rel={rel:.3e}"
                " > 1e-6. The render path does not match the likelihood's model "
                "image -- STOP (a wrong render invalidates all metrics).")
    worst_rel = max(c["relative_error"] for c in chi2_checks)
    print(f"[E1] chi^2 gate PASSED (worst rel={worst_rel:.3e}; "
          f"{'<=1e-8' if worst_rel <= 1e-8 else '<=1e-6'})")

    # --- Task C/D: metrics at the 32 iid points -----------------------------
    iid_records = []
    iid_loadings = []      # |top M_zhat eigenvector| per point
    iid_M_raw_spectra = []
    verified_bij = None
    for i, z in enumerate(iid_z):
        mm = metrics_at(z, ops, W, std_z, std_theta,
                        verify_bij=(i == 0), bij_tol=1e-6)
        if i == 0:
            verified_bij = mm["bij_diag"]
            print(f"[E1] bijector diagonality: "
                  f"max|offdiag|/max|diag|={verified_bij['offdiag_to_diag_ratio']:.3e} "
                  f"min|s|={verified_bij['min_abs_s']:.3e}")
        Vz, wz, wz_all = stiff_subspace(mm["M_zhat"], K_STIFF)
        top_vec = Vz[:, 0]
        iid_loadings.append(np.abs(top_vec))
        w_raw = np.linalg.eigvalsh(mm["M_raw"])
        iid_M_raw_spectra.append(w_raw)
        # top-eigenvector loadings with names (sorted by |loading|)
        order = np.argsort(-np.abs(top_vec))
        loadings = [{"param": param_names[j], "index": int(j),
                     "loading": float(top_vec[j])} for j in order[:8]]
        iid_records.append({
            "iid_point": i,
            "flat_index": int(iid_idx[i]),
            "chain": int(iid_idx[i] // n_steps),
            "step": int(iid_idx[i] % n_steps),
            "M_zhat_top3_eigenvalues": wz.tolist(),
            "M_raw_top3_eigenvalues": np.sort(w_raw)[::-1][:3].tolist(),
            "top_eigenvector_loadings": loadings,
        })

    # --- Task C: r values on a subset of 6 iid points -----------------------
    r_records = []
    for i in range(min(N_HESS, len(iid_z))):
        z = iid_z[i]
        H_post = -np.asarray(ops["hess_logp"](z), dtype=np.float64)
        H_post = 0.5 * (H_post + H_post.T)
        M_raw = gn_metric(np.asarray(ops["jac_render"](z), dtype=np.float64), W)
        r_fro = r_value(H_post, M_raw, K_STIFF, "fro")
        r_spec = r_value(H_post, M_raw, K_STIFF, 2)
        r_records.append({"iid_point": i, "r_frobenius": r_fro,
                          "r_spectral": r_spec})
        print(f"[E1] r pt{i}: r_fro={r_fro:.4f} r_spec={r_spec:.4f}")
    r_fros = [r["r_frobenius"] for r in r_records]

    # --- Task C/D: rotation vs separation over all lag pairs ----------------
    pairs_out = []
    for p in pairs:
        za, zb = p["za"], p["zb"]
        ma = metrics_at(za, ops, W, std_z, std_theta)
        mb = metrics_at(zb, ops, W, std_z, std_theta)
        dz = zb - za
        theta_a = np.asarray(ops["bij_vec"](za), dtype=np.float64)
        theta_b = np.asarray(ops["bij_vec"](zb), dtype=np.float64)
        dtheta = theta_b - theta_a
        rot_z = rotation_between_metrics(ma["M_zhat"], mb["M_zhat"], K_STIFF)
        rot_t = rotation_between_metrics(ma["M_thetahat"], mb["M_thetahat"], K_STIFF)
        pairs_out.append({
            "lag": p["lag"], "chain": p["chain"], "start": p["start"],
            "dz_norm": float(np.linalg.norm(dz)),
            "sep_zhat": float(np.linalg.norm(dz / std_z)),
            "sep_thetahat": float(np.linalg.norm(dtheta / std_theta)),
            "z": rot_z, "theta": rot_t,
        })

    # --- summary reads (proposed, UNCERTIFIED) ------------------------------
    sep_z = np.array([p["sep_zhat"] for p in pairs_out])
    sep_t = np.array([p["sep_thetahat"] for p in pairs_out])
    top_z = np.array([p["z"]["top_vector_angle_deg"] for p in pairs_out])
    top_t = np.array([p["theta"]["top_vector_angle_deg"] for p in pairs_out])

    hstar_std = H_STAR / sigma_bar
    win = (0.5 * hstar_std, 2.0 * hstar_std)   # factor-2 window around h*-scale
    mask_z = (sep_z >= win[0]) & (sep_z < win[1])
    mask_t = (sep_t >= win[0]) & (sep_t < win[1])
    med_top_z_hstar = float(np.median(top_z[mask_z])) if mask_z.any() else float("nan")
    med_top_t_hstar = float(np.median(top_t[mask_t])) if mask_t.any() else float("nan")

    # matched-bin z/theta ratio of median top-vector rotation
    edges = np.geomspace(min(sep_z.min(), sep_t.min()) * 0.9,
                         max(sep_z.max(), sep_t.max()) * 1.1, 9)
    cz, mz, nz = binned_median(sep_z, top_z, edges)
    ct, mt, nt = binned_median(sep_t, top_t, edges)
    ratio_bins = []
    for c, a, b, na, nb in zip(cz, mz, mt, nz, nt):
        rr = float(a / b) if (np.isfinite(a) and np.isfinite(b) and b > 0) else None
        ratio_bins.append({"sep_center": float(c), "median_top_z_deg": float(a),
                           "median_top_theta_deg": float(b), "ratio_z_over_theta": rr,
                           "n_z": int(na), "n_theta": int(nb)})
    valid_ratios = [r["ratio_z_over_theta"] for r in ratio_bins
                    if r["ratio_z_over_theta"] is not None]
    median_ratio = float(np.median(valid_ratios)) if valid_ratios else float("nan")

    # --- plots --------------------------------------------------------------
    plot_rotation_vs_sep(pairs_out, sigma_bar,
                         os.path.join(args.out_dir, "e1_rotation_vs_sep.png"))
    plot_loading_heatmap(iid_loadings, param_names,
                         os.path.join(args.out_dir, "e1_loading_heatmap.png"))
    plot_spectra(iid_M_raw_spectra, clone_inv_spectrum,
                 os.path.join(args.out_dir, "e1_spectra.png"))

    # --- JSON ---------------------------------------------------------------
    doc = {
        "experiment": "E1 -- T6 Fisher-metric survey + T7 bijector curvature",
        "status": "proposed (UNCERTIFIED) -- grader inspects artifacts, not this summary",
        "timestamp_utc": _now(),
        "args": {k: v for k, v in vars(args).items()},
        "dim": int(dim),
        "param_names": param_names,
        "render_path": {
            "mode": "forward",
            "render": "SceneSimulator.simulate(model.to_params(bij.forward(list(z.T)))), flattened",
            "W": "1/err_map**2 (err_map FIXED: Dataset built it once from the OBSERVED "
                 "image via background_rms=0.2, exp_time=100; z-independent)",
            "chi2_aux_convention": "log_prob(z)[1] is REDUCED chi^2 = chi2_total/event_size; "
                                   "chi2_from_aux = aux * prob_model.event_size",
            "event_size": event_size,
            "amplitude_params_note": "mode=forward => Sersic Ie amplitudes are ORDINARY "
                                     "sampled params; they enter J(z) as ordinary columns "
                                     "(no lstsq layer).",
            "chi2_verification": chi2_checks,
            "chi2_gate_worst_relative_error": worst_rel,
        },
        "bijector_diagonality": verified_bij,
        "hessian_note": "H_post = -Hessian(log_prob[0]) includes prior + logdet terms; "
                        "for r this is intended -- r bounds EVERYTHING-not-GN (residual "
                        "second-order + prior) on the stiff subspace. M for r is the "
                        "RAW-z GN metric J^T W J.",
        "standardization": {
            "sigma_bar": sigma_bar,
            "std_z_from": "pooled post-burn-in samples (per-coordinate, ddof=1)",
            "std_theta_from": "bij.forward(pooled samples) (per-coordinate, ddof=1)",
            "z_space_units": "zhat = z/std_z ; J_zhat = J_z*std_z ; Delta zhat = Delta z/std_z",
            "theta_space_units": "thetahat = theta/std_theta ; J_thetahat = J_theta*std_theta",
            "std_z": std_z.tolist(),
            "std_theta": std_theta.tolist(),
        },
        "k_stiff": K_STIFF,
        "iid_points": iid_records,
        "r_values": {
            "records": r_records,
            "r_frobenius_range": [float(np.min(r_fros)), float(np.max(r_fros))],
            "interpretation": "r << 1 => GN/manifold curvature dominant on stiff subspace; "
                              "r >~ 1 => residual/prior terms matter. NOT adjudicated.",
        },
        "pairs": pairs_out,
        "sampler_scales": {
            "eps_move_zunits": EPS_MOVE, "h_star_zunits": H_STAR,
            "eps_move_over_sigma_bar": EPS_MOVE / sigma_bar,
            "h_star_over_sigma_bar": H_STAR / sigma_bar,
        },
        "T6": {
            "prediction": f"top-vector rotation >= {T6_PREDICT_DEG:g} deg at ~h* "
                          f"separations ({H_STAR:g} z-units), growing ~linearly below sigma",
            "falsifier": f"median rotation < {T6_FALSIFY_DEG:g} deg at h*-scale => the "
                         "curved-valley reading of T3 is wrong",
            "median_top_vector_rotation_z_at_hstar_deg": med_top_z_hstar,
            "hstar_window_standardized": list(win),
            "n_pairs_in_window_z": int(mask_z.sum()),
        },
        "T7": {
            "jackpot_branch": f"z/theta rotation ratio >~ {T7_JACKPOT_RATIO:g} => "
                              "transform-induced curvature is a major term (prior/bijector "
                              "redesign)",
            "innocent_branch": "ratio ~ 1 => curvature is intrinsic physics (reparameterize "
                               "the physical family)",
            "median_top_vector_rotation_theta_at_hstar_deg": med_top_t_hstar,
            "ratio_bins": ratio_bins,
            "median_ratio_z_over_theta": median_ratio,
        },
        "verdict_fields": {
            "T6_rotation": "proposed (UNCERTIFIED): compare median_top_vector_rotation_z_at_"
                           "hstar_deg to the T6 prediction/falsifier; read the rotation-vs-sep "
                           "plot.",
            "T7_bijector_contribution": "proposed (UNCERTIFIED): compare median_ratio_z_over_"
                                        "theta to the T7 branches.",
            "note": "this script does NOT adjudicate; a grader inspects plots + numbers.",
        },
        "plots": {
            "rotation_vs_sep": "e1_rotation_vs_sep.png",
            "loading_heatmap": "e1_loading_heatmap.png",
            "spectra": "e1_spectra.png",
        },
    }
    json_path = os.path.join(args.out_dir, "e1_results.json")
    with open(json_path, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(f"[E1] wrote {json_path}")

    # --- compact printed summary (PROPOSED / UNCERTIFIED) -------------------
    print("\n=== E1 summary (PROPOSED / UNCERTIFIED) ===")
    print(f"  chi^2 render gate: PASSED, worst relative error {worst_rel:.3e}")
    print(f"  bijector per-coordinate: max|offdiag|/max|diag| = "
          f"{verified_bij['offdiag_to_diag_ratio']:.3e} (per-coordinate confirmed)")
    print(f"  sigma_bar = {sigma_bar:.4e}; h*/sigma_bar = {hstar_std:.3e}; "
          f"eps_move/sigma_bar = {EPS_MOVE/sigma_bar:.3e}")
    print(f"  [T6] measured median TOP-VECTOR rotation (z) at h*-scale "
          f"({mask_z.sum()} pairs) = {med_top_z_hstar:.2f} deg")
    print(f"       T6 prediction >= {T6_PREDICT_DEG:g} deg ; "
          f"T6 falsifier < {T6_FALSIFY_DEG:g} deg")
    print(f"  [T7] measured median TOP-VECTOR rotation (theta) at h*-scale "
          f"({mask_t.sum()} pairs) = {med_top_t_hstar:.2f} deg")
    print(f"       z/theta median rotation ratio (matched bins) = {median_ratio:.2f} "
          f"; T7 jackpot >~ {T7_JACKPOT_RATIO:g}, innocent ~ 1")
    print(f"  [r]  Frobenius r range over {len(r_fros)} points = "
          f"[{min(r_fros):.3f}, {max(r_fros):.3f}] "
          f"(r<<1 => GN/manifold dominant; r>~1 => residual/prior matter)")
    print("[E1] done.")


if __name__ == "__main__":
    main()
