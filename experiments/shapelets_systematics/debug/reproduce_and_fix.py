"""End-to-end reproduction of the MCLMC step-size collapse, and the fix.

Runs the *actual* production sampler (gigalens_research MCLMC_JIT, shard_map,
isokinetic MCLachlan, real BackwardProbModel.log_prob in float32) for a short
burn-in and records the per-step step size and xi, for several configs:

  - nmax30_iso     : exact notebook setup. qz = N(mode, scale_tril=diag(1e-3))
                     => isotropic inverse mass matrix diag(1e-6). EXPECT COLLAPSE.
  - nmax30_precond : same start mode, but qz covariance = Laplace posterior
                     covariance inv(H) at the mode (anisotropic preconditioner).
                     This sets BOTH the initial mass matrix and the init scatter
                     to the posterior geometry. EXPECT NO COLLAPSE.
  - nmax10_iso     : isotropic, low order -- threshold control (mild / no collapse).

The mass matrix is only a preconditioner: it does not change the target, so the
fix is statistically exact.

Run in the canonical shifter JAX-2026 container.
"""
from __future__ import annotations
import argparse
import os
import sys

DBG_DIR = os.path.dirname(os.path.abspath(__file__))
if DBG_DIR not in sys.path:
    sys.path.insert(0, DBG_DIR)

# importing the diagnostic module enables x64 and gives us the inlined model
# builders / mode finder / Hessian helper.
import diagnose_stepsize_collapse as D
import numpy as np
import jax
# The production sampler's Welford mass-matrix state is float32; importing the
# diagnostic module flips x64 on, which makes lax.cond branches disagree on
# dtype. Run this end-to-end reproduction in the production f32 regime. The mode
# finder / Hessian preconditioner are fine in f32 (a preconditioner need not be
# double precision).
jax.config.update("jax_enable_x64", False)
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp

from gigalens_research.inference.mclmc import MCLMC_JIT

tfd = tfp.distributions


def build_best_notebook(lens_sim, prob_model, true_params, refine_steps=200):
    """Reproduce the notebook's MCLMC start `best`: TRUTH lens + TRUTH lens-light
    (minus Ie) + a fixed-lens MAP source. Only the 3 source dims are optimised;
    the lens/lens-light stay at truth. In the free-lens model the joint mode's
    lens params are offset from truth (shapelets absorb residuals), so this start
    sits off-mode in the STIFF lens directions -- the actual run's condition."""
    import optax
    ll = dict(true_params[1][0]); ll.pop("Ie", None)

    def build_z(beta, cx, cy):
        src = dict(beta=jnp.float32(beta), center_x=jnp.float32(cx), center_y=jnp.float32(cy))
        params = [list(true_params[0]), [ll], [src]]
        params = jax.tree.map(lambda x: jnp.asarray(x).reshape(()), params)
        return jnp.stack(prob_model.bij.inverse(params)).reshape(1, -1)

    lp = D.make_logpost_fn(lens_sim, prob_model, "f32_default")
    chi = jax.jit(lambda z: lp(z)[1])
    best = None
    for beta in np.linspace(0.08, 1.2, 24):
        z = build_z(beta, 0.0, 0.0); c = float(chi(z))
        if best is None or c < best[0]:
            best = (c, np.asarray(z), float(beta))
    z = jnp.asarray(best[1])
    dim = z.shape[-1]
    mask = jnp.asarray([0.0] * (dim - 3) + [1.0, 1.0, 1.0], jnp.float32)  # source dims only
    vg = jax.jit(jax.value_and_grad(lambda zz: lp(zz)[0]))
    opt = optax.adam(3e-3); st = opt.init(z)
    for _ in range(refine_steps):
        v, g = vg(z); g = g * mask
        upd, st = opt.update(-g, st); z = optax.apply_updates(z, upd)
    return z, float(chi(z)), best[2]


def build_qz(z0, dim, kind, lens_sim, prob_model):
    loc = jnp.asarray(np.asarray(z0).reshape(-1), jnp.float32)
    if kind.startswith("iso"):
        # notebook: scale_tril = diag(1e-3)  => covariance diag(1e-6)
        scale_tril = jnp.eye(dim, dtype=jnp.float32) * 1e-3
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)
    elif kind.startswith("precond"):
        _, cov = D.compute_hessian_metrics(lens_sim, prob_model, jnp.asarray(z0))
        cov = np.asarray(cov, np.float64)
        cov = 0.5 * (cov + cov.T)
        # symmetrize + floor eigenvalues for a clean PD Cholesky
        w, V = np.linalg.eigh(cov)
        w = np.clip(w, w.max() * 1e-8, None)
        cov = (V * w) @ V.T
        Lc = np.linalg.cholesky(cov).astype(np.float32)
        return tfd.MultivariateNormalTriL(loc=loc, scale_tril=jnp.asarray(Lc))
    else:
        raise ValueError(kind)


def build_model(sim, rep, n_max):
    observed_img, true_params, sim_config, _ = D.load_vela_sim_system(sim, rep, cam="12")
    model_seq, lens_sim = D.vela_system_model(sim_config, observed_img, use_shapelets=True, n_max=n_max)
    prob_model = model_seq.prob_model
    dim = int(jnp.stack(prob_model.bij.inverse(prob_model.prior.sample(seed=jax.random.PRNGKey(0)))).shape[0])
    return model_seq, lens_sim, prob_model, true_params, dim


def run_one(sim, rep, n_max, kind, model_seq, lens_sim, prob_model, z0, dim,
            num_burnin, num_results, n_hmc, seed=0):
    gnorm = float(jnp.linalg.norm(jax.grad(
        lambda z: D.make_logpost_fn(lens_sim, prob_model, "f32_default")(z.reshape(1, -1))[0]
    )(jnp.asarray(z0).reshape(-1).astype(jnp.float32))))
    print(f"[{kind} nmax{n_max}] start ||grad||={gnorm:.3e}", flush=True)

    qz = build_qz(z0, dim, kind, lens_sim, prob_model)

    debug_hist = MCLMC_JIT(
        model_seq, qz,
        n_hmc=n_hmc, num_burnin_steps=num_burnin, num_results=num_results,
        desired_energy_variance=5e-4, frac_tune1=0.2, frac_tune2=0.6, frac_tune3=0.2,
        seed=seed, debug_output=True, progress_bar=False,
    )
    ss = np.asarray(debug_hist.step_size)   # (chains, steps)
    xi = np.asarray(debug_hist.xi)
    nonan = np.asarray(debug_hist.nonan)
    np.savez(os.path.join(DBG_DIR, f"repro_{sim}_rep{rep:02d}_nmax{n_max}_{kind}.npz"),
             step_size=ss, xi=xi, nonan=nonan)

    n1 = round(num_burnin * 0.2)
    ss_end_phase1 = ss[:, n1 - 1]
    ss_min_phase1 = ss[:, :n1].min(axis=1)
    final = ss[:, -1]
    print(f"[{kind} nmax{n_max}] eps: init~{ss[:,0].mean():.3e}  "
          f"phase1-end median={np.median(ss_end_phase1):.3e}  "
          f"phase1-min median={np.median(ss_min_phase1):.3e}  "
          f"final median={np.median(final):.3e}", flush=True)
    print(f"[{kind} nmax{n_max}] collapsed chains (eps<1e-4 by phase1 end): "
          f"{int((ss_end_phase1 < 1e-4).sum())}/{ss.shape[0]}   "
          f"nan-frac(phase1)={1.0 - nonan[:, :n1].mean():.3f}", flush=True)
    return dict(kind=kind, n_max=n_max, step_size=ss, xi=xi, n1=n1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim", default="01")
    p.add_argument("--rep", type=int, default=3)
    p.add_argument("--num-burnin", type=int, default=600)
    p.add_argument("--num-results", type=int, default=100)
    p.add_argument("--n-hmc", type=int, default=8)
    p.add_argument("--lens-perturb", type=float, default=0.5)
    p.add_argument("--configs", nargs="+",
                   default=["30:iso", "30:precond", "10:iso"])
    args = p.parse_args()

    print(f"JAX {jax.__version__} devices={jax.devices()}", flush=True)
    runs = []
    cache = {}  # n_max -> (model_seq, lens_sim, prob_model, z0, dim)
    for c in args.configs:
        parts = c.split(":"); nm = int(parts[0]); kind = parts[1]
        hard = (len(parts) > 2 and parts[2] == "hard")
        if nm not in cache:
            model_seq, lens_sim, prob_model, true_params, dim = build_model(args.sim, args.rep, nm)
            z0, chi0, beta0 = build_best_notebook(lens_sim, prob_model, true_params)
            print(f"[nmax{nm}] notebook start: beta={beta0:.3f} chi2/pix={chi0:.3e} dim={dim}", flush=True)
            cache[nm] = (model_seq, lens_sim, prob_model, z0, dim)
        model_seq, lens_sim, prob_model, z0, dim = cache[nm]
        z_start = np.asarray(z0).copy()
        tag = kind
        if hard:
            # mimic an imperfect MAP / a system with lens-shapelet degeneracy:
            # offset the (stiff) lens dims off the mode by ~0.5 in z-space.
            off = np.zeros_like(z_start)
            rng = np.random.default_rng(args.rep)
            off[0, :8] = args.lens_perturb * rng.standard_normal(8)
            z_start = z_start + off
            tag = kind + "_hard"
        runs.append(run_one(args.sim, args.rep, nm, tag,
                            model_seq, lens_sim, prob_model, jnp.asarray(z_start, jnp.float32), dim,
                            args.num_burnin, args.num_results, args.n_hmc))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5.5))
        def pick_color(kind):
            return "tab:green" if kind.startswith("precond") else "tab:red"
        for r in runs:
            c = pick_color(r["kind"])
            ss = r["step_size"]
            for ch in range(ss.shape[0]):
                ax.plot(ss[ch], color=c, alpha=0.5, lw=1,
                        label=f"nmax{r['n_max']} {r['kind']}" if ch == 0 else None)
            ax.axvline(r["n1"], color=c, ls=":", lw=1, alpha=0.7)
        ax.set_yscale("log"); ax.set_xlabel("burn-in step"); ax.set_ylabel("step size eps")
        ax.set_title("MCLMC step-size: isotropic (collapse) vs preconditioned (fixed)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fn = os.path.join(DBG_DIR, f"repro_stepsize_{args.sim}_rep{args.rep:02d}.png")
        fig.savefig(fn, dpi=120); print(f"saved {fn}", flush=True)
    except Exception as e:
        print(f"(plot skipped: {e})", flush=True)


if __name__ == "__main__":
    main()
