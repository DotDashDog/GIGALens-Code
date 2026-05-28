"""Diagnose the MCLMC step-size collapse with high-order shapelets + BackwardProbModel.

Central hypothesis (H1): at n_max >= ~20 the weighted shapelet+Sersic design
matrix X is so ill-conditioned that the normal-equation solve coeffs = (X^T X +
jitter)^{-1} X^T y -- and, crucially, its reverse-mode gradient w.r.t. the
nonlinear params z -- carries large relative error in float32 (and worse under
A100 TF32 matmul). The isokinetic integrator then uses a *wrong* force, so the
simulated trajectory cannot conserve the true energy at ANY step size. The
controller (which assumes Var[E] ~ eps^6) reads the floor-level energy error as
"step too big" and drives eps to the floor; handle_nans' x0.8 ratchet on
step_size_max accelerates the collapse.

This script runs three tests, each for several n_max, and compares
float32-default (TF32 on GPU), float32-highest (no TF32), and float64 for the
*linear algebra only* (physics stays float32):

  A. Conditioning of X and of the regularized gram; TF32-vs-f64 error in X^T X.
  B. Gradient accuracy: ||g_mode - g_ref|| / ||g_ref|| of grad(log_posterior).
     g_ref is the float64-solve gradient, validated against a float64 central
     finite difference.
  C. Energy-error vs step-size of the actual blackjax isokinetic kernel (the
     exact quantity the controller tunes on), for each mode -- does float64
     restore the eps^6 scaling?

Run inside the canonical shifter JAX-2026 container.
"""
from __future__ import annotations

import argparse
import os
import sys

SHAPELETS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SHAPELETS_DIR not in sys.path:
    sys.path.insert(0, SHAPELETS_DIR)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import functools
import json
import pickle
from os.path import expanduser
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)  # required for an honest float64 reference
import jax.numpy as jnp
from jax import lax
import tensorflow_probability.substrates.jax as tfp

import gigalens.jax.simulator as gsim
from gigalens.jax.inference import ModellingSequence
from gigalens.jax.model import BackwardProbModel
from gigalens.jax.profiles.light import sersic, shapelets
from gigalens.jax.profiles.mass import epl, shear
from gigalens.jax.simulator import LensSimulator
from gigalens.model import PhysicalModel
from gigalens.simulator import SimulatorConfig

tfd = tfp.distributions

# --- inlined from vela_utilities to avoid its plotting import chain ---
HOME = expanduser("~/")
DEFAULT_BACKGROUND_RMS = 0.002
DEFAULT_EXP_TIME = 2000.0
DEFAULT_NUM_PIX = 200
DEFAULT_SUPERSAMPLE = 1


def load_vela_sim_system(sim_num, rep, cam="12", num_pix=DEFAULT_NUM_PIX,
                         supersample=DEFAULT_SUPERSAMPLE, filter_tag="a0.500_f814w"):
    src_dir = os.path.join(HOME, "GIGALens-Code", "data", "vela_sources",
                           f"vela{sim_num}_cam{cam}_{filter_tag}")
    sys_dir = os.path.join(HOME, "GIGALens-Code", "data", "vela_sim_systems",
                           f"vela{sim_num}_cam{cam}_rep{int(rep):02d}_{filter_tag}")
    psf = np.load(os.path.join(src_dir, "psf.npy"))
    with open(os.path.join(src_dir, "metadata.json")) as f:
        meta = json.load(f)
    delta_pix = meta["instrument_pixel_scale_arcsec"]
    sim_config = SimulatorConfig(delta_pix=delta_pix, num_pix=num_pix,
                                 supersample=supersample, kernel=psf)
    observed_img = jnp.load(os.path.join(sys_dir, "lens_img.npy"))
    with open(os.path.join(sys_dir, "true_params"), "rb") as f:
        true_params = pickle.load(f)
    return observed_img, true_params, sim_config, meta


def fixed_prior(profile_params):
    prior_dists = {}
    for key in profile_params:
        val = jnp.squeeze(profile_params[key])
        prior_dists[key] = tfd.Uniform(val - 1e-6, val + 1e-6)
    return tfd.JointDistributionNamed(prior_dists)


def _src_model(use_shapelets, n_max):
    return (shapelets.ShapeletsFast(n_max=n_max, use_lstsq=True, interpolate=False)
            if use_shapelets else sersic.SersicEllipse(use_lstsq=True))


def free_source_fixed_lens_model(sim_config, observed_img, fixed_params,
                                 background_rms=DEFAULT_BACKGROUND_RMS,
                                 exp_time=DEFAULT_EXP_TIME, use_shapelets=True, n_max=10):
    lens_prior = tfd.JointDistributionSequential([fixed_prior(p) for p in fixed_params[0]])
    ie_less = fixed_params[1][0].copy()
    del ie_less["Ie"]
    lens_light_prior = tfd.JointDistributionSequential([fixed_prior(ie_less)])
    source_light_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            beta=tfd.LogNormal(jnp.log(0.7), 0.4),
            center_x=tfd.Normal(0, 0.01),
            center_y=tfd.Normal(0, 0.01),
        )),
    ])
    prior = tfd.JointDistributionSequential([lens_prior, lens_light_prior, source_light_prior])
    src_model = _src_model(use_shapelets, n_max)
    phys_model = PhysicalModel([epl.EPL(50), shear.Shear()],
                               [sersic.SersicEllipse(use_lstsq=True)], [src_model])
    lens_sim = LensSimulator(phys_model, sim_config, bs=1)
    prob_model = BackwardProbModel(prior, observed_img, background_rms=background_rms, exp_time=exp_time)
    model_seq = ModellingSequence(phys_model, prob_model, sim_config)
    return model_seq, lens_sim


# ---------------------------------------------------------------------------
# Build the (post conv+pool) weighted design tensor `ret` as a differentiable
# function of z, exactly mirroring LensSimulator.lstsq_simulate, but stop short
# of the linear solve so we can swap its precision.
# ---------------------------------------------------------------------------
def make_basis_fn(lens_sim, prob_model):
    """Return basis_fn(z) -> ret with shape (1, h, w, n_comp), differentiable in z.
    Mirrors the simulator's build_basis + conv_pool path."""
    bij = prob_model.bij

    def basis_fn(z):
        x = bij.forward(list(z.T))
        lens_params = x[0]
        lens_light_params, source_light_params = x[1], x[2]
        beta_x, beta_y = lens_sim._beta(lens_params)
        img = jnp.zeros((0, *lens_sim.img_X.shape))
        for lm, p in zip(lens_sim.phys_model.lens_light, lens_light_params):
            img = jnp.concatenate((img, lm.light(lens_sim.img_X, lens_sim.img_Y, **p)), axis=0)
        for lm, p in zip(lens_sim.phys_model.source_light, source_light_params):
            img = jnp.concatenate((img, lm.light(beta_x, beta_y, **p)), axis=0)
        img = jnp.nan_to_num(img)
        img = jnp.transpose(img, (3, 0, 1, 2))  # bs, n_comp, h, w
        if lens_sim.flat_kernel is not None:
            ret = gsim._shared_kernel_component_conv(img, lens_sim.flat_kernel)
        else:
            ret = img
        if lens_sim.supersample != 1:
            from objax.functional import average_pool_2d
            ret = average_pool_2d(ret, size=(lens_sim.supersample, lens_sim.supersample), padding="SAME")
        ret = jnp.transpose(ret, (0, 2, 3, 1))  # bs, h, w, n_comp
        return ret

    return basis_fn


def reconstruct(ret, observed_image, err_map, mode):
    """Weighted normal-equation reconstruct with controllable precision.

    mode:
      'f32_default' : status quo (global matmul precision; TF32 on A100)
      'f32_highest' : same in float32 but matmul precision forced 'highest'
      'f64'         : gram + solve in float64 (physics ret stays float32)
    """
    def _do(ret_, obs_, err_):
        W = (1 / err_)[..., jnp.newaxis]
        Y = jnp.reshape(obs_ * jnp.squeeze(W), (1, -1, 1))
        X = jnp.reshape((ret_ * W), (ret_.shape[0], -1, ret_.shape[-1]))
        Xt = jnp.transpose(X, (0, 2, 1))
        gram = Xt @ X
        rhs = Xt @ Y
        gram = gsim._regularize_gram(gram)
        coeffs = jax.vmap(jnp.linalg.solve)(gram, rhs)[..., 0]
        recon = jnp.sum(ret_ * coeffs[:, jnp.newaxis, jnp.newaxis, :], axis=-1)
        return jnp.squeeze(recon), jnp.squeeze(coeffs)

    if mode == "f64":
        # gram + solve in genuine float64 (physics ret arrives as float32).
        return _do(ret.astype(jnp.float64),
                   observed_image.astype(jnp.float64),
                   err_map.astype(jnp.float64))
    elif mode == "f32_highest":
        with jax.default_matmul_precision("highest"):
            return _do(ret, observed_image, err_map)
    else:  # f32_default (TF32 on A100)
        return _do(ret, observed_image, err_map)


def make_logpost_fn(lens_sim, prob_model, mode):
    """Differentiable scalar log-posterior at z (shape (1, n_dim)), with the
    linear solve done in the requested precision. Mirrors BackwardProbModel."""
    basis_fn = make_basis_fn(lens_sim, prob_model)
    bij = prob_model.bij
    observed_image = prob_model.observed_image
    err_map = prob_model.err_map
    npix = float(observed_image.size)

    def logpost(z):
        ret = basis_fn(z)
        # NOTE: BackwardProbModel uses lstsq_simulate(...)[0] directly -- NO
        # conversion_factor (only the forward simulate() applies it).
        recon, _ = reconstruct(ret, observed_image, err_map, mode)
        wdt = recon.dtype
        obs = observed_image.astype(wdt)
        err = err_map.astype(wdt)
        resid = (recon - obs) / err
        # Gaussian log-like == Independent(Normal(obs, err)).log_prob(recon).
        log_like = (-0.5 * jnp.sum(resid ** 2)
                    - jnp.sum(jnp.log(err))
                    - 0.5 * npix * jnp.log(2.0 * jnp.pi))
        x = bij.forward(list(z.T))
        log_prior = jnp.squeeze(prob_model.prior.log_prob(x)
                                + bij.forward_log_det_jacobian(list(z.T))).astype(wdt)
        chisq = jnp.mean(resid ** 2)
        return jnp.squeeze(log_like) + log_prior, chisq

    return logpost


# ---------------------------------------------------------------------------
# z0 selection: truth lens (pinned) + a source beta chosen by a quick chi^2 scan
# at center=0, so we sit in a data-supported (arc-reconstructing) region -- the
# regime the chains actually occupy.
# ---------------------------------------------------------------------------
def pick_z0(lens_sim, prob_model, refine_steps=200):
    """Pick z0 near the actual posterior mode the chains start from: truth lens
    + truth lens-light (pinned by the fixed-lens prior) + the best shapelet
    source. Beta scan for a good init, then Adam refine all dims in float64."""
    import optax
    sample = prob_model.prior.sample(seed=jax.random.PRNGKey(0))
    logpost = make_logpost_fn(lens_sim, prob_model, "f64")
    chi_j = jax.jit(lambda z: logpost(z)[1])
    best = None
    for beta in np.linspace(0.08, 1.2, 24):
        shp = dict(sample[2][0])
        shp["beta"] = jnp.asarray(np.float32(beta))
        shp["center_x"] = jnp.asarray(np.float32(0.0))
        shp["center_y"] = jnp.asarray(np.float32(0.0))
        twp = build_true_params_shp_from_sample(prob_model, sample, shp)
        z = jnp.stack(prob_model.bij.inverse(twp)).T  # (1, n_dim)
        c = float(chi_j(z))
        if best is None or c < best[0]:
            best = (c, np.asarray(z), float(beta))
    z = jnp.asarray(best[1])
    # Adam refine toward the mode (maximize log-post).
    val_grad = jax.jit(jax.value_and_grad(lambda zz: logpost(zz)[0]))
    opt = optax.adam(3e-3)
    st = opt.init(z)
    for _ in range(refine_steps):
        v, g = val_grad(z)
        updates, st = opt.update(-g, st)  # ascent
        z = optax.apply_updates(z, updates)
    chi_final = float(chi_j(z))
    return z, chi_final, best[2]


def build_true_params_shp_from_sample(prob_model, sample, shp):
    """Build a params pytree using the fixed-lens prior sample but overriding
    the source dict with `shp`. All leaves get a leading batch dim of 1."""
    params = [list(sample[0]), list(sample[1]), [shp]]
    params = jax.tree.map(
        lambda x: jnp.asarray(x)[None] if jnp.asarray(x).ndim == 0 else jnp.asarray(x),
        params,
    )
    return params


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_A_conditioning(lens_sim, prob_model, z0):
    basis_fn = make_basis_fn(lens_sim, prob_model)
    ret = np.asarray(basis_fn(z0))  # (1, h, w, n_comp)
    err = np.asarray(prob_model.err_map)
    obs = np.asarray(prob_model.observed_image)
    W = (1.0 / err)
    n_comp = ret.shape[-1]
    X = (ret[0] * W[..., None]).reshape(-1, n_comp)  # (npix, n_comp)

    X64 = X.astype(np.float64)
    s = np.linalg.svd(X64, compute_uv=False)
    condX = s[0] / s[-1] if s[-1] > 0 else np.inf

    XtX64 = X64.T @ X64
    diag_mean = np.mean(np.diag(XtX64))
    jitter = 1e-6 * max(diag_mean, 1.0)
    XtX_reg = XtX64 + jitter * np.eye(n_comp)
    cond_reg = np.linalg.cond(XtX_reg)

    # TF32-vs-f64 error in forming X^T X on the *device* (this is the GPU path).
    Xj = jnp.asarray(X.astype(np.float32))
    gram_tf32 = np.asarray(Xj.T @ Xj)  # default precision (TF32 on A100)
    with jax.default_matmul_precision("highest"):
        gram_hi = np.asarray(Xj.T @ Xj)
    rel_tf32 = np.linalg.norm(gram_tf32 - XtX64) / np.linalg.norm(XtX64)
    rel_hi = np.linalg.norm(gram_hi - XtX64) / np.linalg.norm(XtX64)

    # Predicted float32/TF32 solve relative error ~ cond_reg * unit_roundoff.
    print(f"  n_comp={n_comp}  cond(X)={condX:.3e}  cond(X^TX)~{condX**2:.3e}")
    print(f"  jitter={jitter:.3e}  cond(X^TX + jitter I)={cond_reg:.3e}")
    print(f"  rel err in X^TX:  TF32/default={rel_tf32:.3e}   highest={rel_hi:.3e}")
    print(f"  predicted solve rel-err  f32 (u=6e-8): {cond_reg*6e-8:.3e}"
          f"   TF32 (u=5e-4): {cond_reg*5e-4:.3e}")
    return dict(n_comp=n_comp, condX=condX, cond_reg=cond_reg,
                rel_tf32=rel_tf32, rel_hi=rel_hi)


def test_B_gradient(lens_sim, prob_model, z0):
    modes = ["f32_default", "f32_highest", "f64"]
    grads, vals, chis = {}, {}, {}
    for m in modes:
        lp = make_logpost_fn(lens_sim, prob_model, m)
        vg = jax.jit(jax.value_and_grad(lambda z: lp(z)[0]))
        (val), g = vg(z0)
        # recompute chisq
        _, chisq = jax.jit(lp)(z0)
        grads[m] = np.asarray(g).ravel()
        vals[m] = float(val)
        chis[m] = float(chisq)

    # Finite-difference reference (float64 solve), central differences.
    lp64 = make_logpost_fn(lens_sim, prob_model, "f64")
    f64_val = jax.jit(lambda z: lp64(z)[0])
    z0np = np.asarray(z0)
    n = z0np.shape[-1]
    h = 1e-3
    gfd = np.zeros(n)
    for i in range(n):
        zp = z0np.copy(); zp[0, i] += h
        zm = z0np.copy(); zm[0, i] -= h
        gfd[i] = (float(f64_val(jnp.asarray(zp))) - float(f64_val(jnp.asarray(zm)))) / (2 * h)

    gref = grads["f64"]
    relfd = np.linalg.norm(gref - gfd) / max(np.linalg.norm(gfd), 1e-30)
    print(f"  log-post values:  " + "  ".join(f"{m}={vals[m]:.6e}" for m in modes))
    print(f"  chi^2/pix:        " + "  ".join(f"{m}={chis[m]:.4e}" for m in modes))
    print(f"  ||g_f64 - g_FD64|| / ||g_FD64|| = {relfd:.3e}   (validates analytic f64 grad)")
    for m in modes:
        rel = np.linalg.norm(grads[m] - gref) / max(np.linalg.norm(gref), 1e-30)
        cos = float(np.dot(grads[m], gref) / (np.linalg.norm(grads[m]) * np.linalg.norm(gref) + 1e-30))
        print(f"  mode={m:12s} ||g||={np.linalg.norm(grads[m]):.3e}  "
              f"rel-err vs f64={rel:.3e}  cos(g,g_f64)={cos:.4f}")
    return dict(grads=grads, gref=gref, relfd=relfd)


def compute_hessian_metrics(lens_sim, prob_model, z0):
    """Hessian of the (float64) log-posterior at z0; returns eigenvalues of the
    negative-log-post Hessian (the natural metric) and a regularized posterior
    covariance = inv(H_pos)."""
    lp = make_logpost_fn(lens_sim, prob_model, "f64")
    f = lambda zf: lp(zf.reshape(1, -1))[0]
    H = jax.hessian(f)(z0.reshape(-1))      # Hessian of +log post
    H = np.asarray(H)
    Hneg = -0.5 * (H + H.T)                 # metric ~ Hessian of -log post
    evals, evecs = np.linalg.eigh(Hneg)
    # Regularize to PD for a usable mass matrix (clip tiny/neg eigenvalues).
    pos = evals[evals > 0]
    floor = (pos.min() if pos.size else 1.0) * 1e-3
    evals_clip = np.clip(evals, floor, None)
    cov = (evecs * (1.0 / evals_clip)) @ evecs.T   # inv(H_pos) = posterior cov
    cov = 0.5 * (cov + cov.T)
    return evals, cov


def test_E_curvature(lens_sim, prob_model, z0, n_max):
    evals, cov = compute_hessian_metrics(lens_sim, prob_model, z0)
    pos = evals[evals > 0]
    print(f"  Hessian(-logpost) eigenvalues (curvatures):")
    print(f"    n_pos={pos.size}/{evals.size}  min(+)={pos.min():.3e}  max={evals.max():.3e}")
    print(f"    curvature condition number = {evals.max()/pos.min():.3e}")
    print(f"    sqrt(1/max_eig) (stiffest length scale) = {1.0/np.sqrt(evals.max()):.3e}")
    print(f"    sqrt(1/min+_eig) (loosest length scale)  = {1.0/np.sqrt(pos.min()):.3e}")
    print(f"  -> a single scalar step size must straddle a {np.sqrt(evals.max()/pos.min()):.1e}x"
          f" range of length scales")
    print(f"  run's metric is isotropic cov=diag(1e-6) i.e. length scale 1e-3 in every dim")
    return evals, cov


def test_C_energy_vs_stepsize(lens_sim, prob_model, z0, dim, out_prefix, cov_good=None):
    """xi = dE^2/(dim*target) vs step_size using the exact isokinetic kernel
    (float32, matching the run), comparing the run's isotropic metric
    diag(1e-6) against a Hessian-informed posterior-covariance metric."""
    from gigalens_research.inference.blackjax_updated_utils import (
        _build_kernel_shardmap, isokinetic_mclachlan_smart, _single_init,
    )
    target = 5e-4
    L = float(np.sqrt(dim))
    step_sizes = np.logspace(-7, 1.0, 40)
    key = jax.random.key(0)
    z0f = z0.reshape(-1).astype(jnp.float32)

    metrics = {"iso diag(1e-6)": jnp.eye(dim, dtype=jnp.float32) * 1e-6}
    if cov_good is not None:
        metrics["Hessian cov"] = jnp.asarray(cov_good, dtype=jnp.float32)

    lp = make_logpost_fn(lens_sim, prob_model, "f32_default")
    logdensity = lambda z: lp(z.reshape(1, -1))[0]
    results = {}
    for name, inv_mm in metrics.items():
        state = _single_init(z0f, logdensity, key)
        kernel = _build_kernel_shardmap(logdensity, inv_mm, isokinetic_mclachlan_smart)
        kernel_j = jax.jit(lambda st, ss: kernel(rng_key=key, state=st, L=L, step_size=ss))
        xis = []
        for ss in step_sizes:
            _, info = kernel_j(state, jnp.float32(ss))
            dE = float(info.energy_change)
            xis.append((dE ** 2) / (dim * target) if np.isfinite(dE) else np.nan)
        results[name] = np.array(xis)
        finite = np.array(xis)[np.isfinite(xis)]
        # smallest step size at which xi first drops below 1 (controller's target)
        below = step_sizes[np.array(xis) < 1.0]
        ss_ok = below.max() if below.size else np.nan
        print(f"  metric={name:16s} xi in [{np.nanmin(xis):.2e},{np.nanmax(xis):.2e}]"
              f"  largest eps with xi<1: {ss_ok:.2e}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 5))
        colors = {"iso diag(1e-6)": "tab:red", "Hessian cov": "tab:green"}
        for name, xis in results.items():
            ax.loglog(step_sizes, xis, "o-", ms=4, color=colors.get(name), label=name)
        ax.axhline(1.0, color="gray", ls=":", label="target xi=1")
        ax.set_xlabel("step size"); ax.set_ylabel(r"$\xi=\Delta E^2/(d\cdot\mathrm{target})$")
        ax.legend(); ax.set_title(f"Energy error vs step size ({out_prefix})")
        fig.tight_layout()
        fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"energy_vs_step_{out_prefix}.png")
        fig.savefig(fn, dpi=110); print(f"  saved {fn}")
    except Exception as e:
        print(f"  (plot skipped: {e})")
    return results


def test_D_beta_sweep(lens_sim, prob_model, z0, n_max, out_prefix):
    """Sweep the shapelet scale beta (z[-3] = log beta) holding lens+centre at
    z0, and measure how cond(X) and the gradient error (TF32 and highest-f32 vs
    float64) grow. This maps the regime the chains explore against numerical
    breakdown of the autodiff force."""
    basis_fn = make_basis_fn(lens_sim, prob_model)
    err = np.asarray(prob_model.err_map)
    W = (1.0 / err)

    lp = {m: jax.jit(jax.value_and_grad(lambda z, m=m: make_logpost_fn(lens_sim, prob_model, m)(z)[0]))
          for m in ["f32_default", "f32_highest", "f64"]}

    z0np = np.asarray(z0)
    betas = np.geomspace(0.08, 3.0, 22)
    rows = []
    for b in betas:
        z = z0np.copy()
        z[0, -3] = np.log(b)  # LogNormal bijector: beta = exp(z)
        zj = jnp.asarray(z)

        ret = np.asarray(basis_fn(zj))
        X = (ret[0] * W[..., None]).reshape(-1, ret.shape[-1]).astype(np.float64)
        s = np.linalg.svd(X, compute_uv=False)
        condX = float(s[0] / s[-1]) if s[-1] > 0 else np.inf

        g = {}
        for m in ["f32_default", "f32_highest", "f64"]:
            _, gg = lp[m](zj)
            g[m] = np.asarray(gg).ravel().astype(np.float64)
        gref = g["f64"]
        nref = np.linalg.norm(gref) + 1e-30
        rel_tf32 = np.linalg.norm(g["f32_default"] - gref) / nref
        rel_hi = np.linalg.norm(g["f32_highest"] - gref) / nref
        cos_tf32 = float(np.dot(g["f32_default"], gref) / (np.linalg.norm(g["f32_default"]) * nref))
        rows.append((b, condX, np.linalg.norm(gref),
                     np.linalg.norm(g["f32_default"]), rel_tf32, rel_hi, cos_tf32))

    print("   beta     cond(X)    ||g_f64||   ||g_TF32||   relerr_TF32  relerr_hi  cos_TF32")
    for b, c, gf, gt, rt, rh, ct in rows:
        print(f"  {b:6.3f}  {c:.3e}  {gf:.3e}  {gt:.3e}  {rt:.3e}  {rh:.3e}  {ct:+.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        arr = np.array(rows)
        fig, ax1 = plt.subplots(figsize=(7.5, 5))
        ax1.loglog(arr[:, 0], arr[:, 4], "o-", color="tab:red", label="rel-err TF32 (default)")
        ax1.loglog(arr[:, 0], arr[:, 5], "s-", color="tab:orange", label="rel-err highest-f32")
        ax1.axhline(1.0, color="gray", ls=":", lw=1)
        ax1.set_xlabel("shapelet scale beta")
        ax1.set_ylabel("gradient rel-err vs float64")
        ax1.legend(loc="upper left")
        ax2 = ax1.twinx()
        ax2.loglog(arr[:, 0], arr[:, 1], "^--", color="tab:blue", alpha=0.6, label="cond(X)")
        ax2.set_ylabel("cond(X)", color="tab:blue")
        ax1.set_title(f"Gradient breakdown vs beta ({out_prefix})")
        fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"grad_vs_beta_{out_prefix}.png")
        fig.tight_layout(); fig.savefig(fn, dpi=110)
        print(f"   saved {fn}")
    except Exception as e:
        print(f"   (plot skipped: {e})")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim", default="01")
    p.add_argument("--rep", type=int, default=3)
    p.add_argument("--n-max-list", type=int, nargs="+", default=[10, 20, 30])
    p.add_argument("--tests", default="ABC")
    args = p.parse_args()

    print(f"JAX {jax.__version__}  devices={jax.devices()}")
    print(f"default matmul precision (None=>TF32 on A100): {jax.config.jax_default_matmul_precision}")

    observed_img, true_params, sim_config, _ = load_vela_sim_system(args.sim, args.rep, cam="12")

    for n_max in args.n_max_list:
        print(f"\n{'='*78}\n n_max = {n_max}\n{'='*78}")
        model_seq, lens_sim = free_source_fixed_lens_model(
            sim_config, observed_img, true_params,
            background_rms=DEFAULT_BACKGROUND_RMS, exp_time=DEFAULT_EXP_TIME,
            use_shapelets=True, n_max=n_max,
        )
        prob_model = model_seq.prob_model
        dim = int(jnp.stack(prob_model.bij.inverse(prob_model.prior.sample(seed=jax.random.PRNGKey(0)))).shape[0])

        z0, chi0, beta0 = pick_z0(lens_sim, prob_model)
        print(f" z0 chosen: beta={beta0:.3f}  chi^2/pix(f64)={chi0:.3e}  dim={dim}")

        if "A" in args.tests:
            print("\n--- TEST A: conditioning ---")
            test_A_conditioning(lens_sim, prob_model, z0)
        if "B" in args.tests:
            print("\n--- TEST B: gradient accuracy ---")
            test_B_gradient(lens_sim, prob_model, z0)
        if "D" in args.tests:
            print("\n--- TEST D: gradient breakdown vs beta ---")
            test_D_beta_sweep(lens_sim, prob_model, z0, n_max, f"{args.sim}_rep{args.rep:02d}_nmax{n_max}")
        cov_good = None
        if "E" in args.tests:
            print("\n--- TEST E: curvature / anisotropy ---")
            _, cov_good = test_E_curvature(lens_sim, prob_model, z0, n_max)
        if "C" in args.tests:
            print("\n--- TEST C: energy error vs step size (metric comparison) ---")
            if cov_good is None:
                _, cov_good = compute_hessian_metrics(lens_sim, prob_model, z0)
            test_C_energy_vs_stepsize(lens_sim, prob_model, z0, dim,
                                      f"{args.sim}_rep{args.rep:02d}_nmax{n_max}", cov_good=cov_good)


if __name__ == "__main__":
    main()
