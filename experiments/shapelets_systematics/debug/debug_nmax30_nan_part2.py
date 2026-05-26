"""Part 2 of the n_max=30 NaN diagnostic.

Builds the full free-source / fixed-lens model and looks at the actual MAP
loss landscape and its gradients, mirroring `run_vela_modeling.run_one_system`
- sample n_samples starts from the prior
- evaluate log_prob and its gradient
- report how many samples produce NaN / Inf log_prob or grad
- sweep beta around the prior mean and plot chi^2 + grad-norm
"""

from __future__ import annotations

import argparse
import os
import sys

SHAPELETS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SHAPELETS_DIR not in sys.path:
    sys.path.insert(0, SHAPELETS_DIR)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

import gigalens.jax.simulator as sim

from vela_utilities import (
    DEFAULT_BACKGROUND_RMS,
    DEFAULT_EXP_TIME,
    free_source_fixed_lens_model,
    load_vela_sim_system,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim", default="01")
    p.add_argument("--rep", type=int, default=3)
    p.add_argument("--n-max", type=int, default=30)
    p.add_argument("--n-samples", type=int, default=8)
    args = p.parse_args()

    observed_img, true_params, sim_config, _ = load_vela_sim_system(
        args.sim, args.rep, cam="12",
    )
    model_seq, lens_sim = free_source_fixed_lens_model(
        sim_config, observed_img, true_params,
        background_rms=DEFAULT_BACKGROUND_RMS,
        exp_time=DEFAULT_EXP_TIME,
        use_shapelets=True, n_max=args.n_max,
    )
    prob_model = model_seq.prob_model

    lens_sim_batched = sim.LensSimulator(
        model_seq.phys_model, model_seq.sim_config, bs=args.n_samples,
    )

    def neg_log_prob(z_batched):
        lp, chisq = prob_model.log_prob(lens_sim_batched, z_batched)
        return -jnp.mean(lp) / float(observed_img.size), (lp, chisq)

    grad_fn = jax.jit(jax.value_and_grad(neg_log_prob, has_aux=True))

    print(f"--- multi-start scan at n_max={args.n_max} ---")
    for seed in range(3):
        start = prob_model.prior.sample(args.n_samples, seed=jax.random.PRNGKey(seed))
        z = jnp.stack(prob_model.bij.inverse(start)).T  # (n_samples, n_dim)
        (val, (lp, chisq)), grad = grad_fn(z)
        lp_np = np.asarray(lp)
        chisq_np = np.asarray(chisq)
        grad_np = np.asarray(grad)
        nan_lp = int(np.isnan(lp_np).sum())
        inf_lp = int(np.isinf(lp_np).sum())
        nan_grad_chains = int(np.any(np.isnan(grad_np), axis=1).sum())
        finite_chisq = chisq_np[np.isfinite(chisq_np)]
        print(f"seed={seed}: lp NaN/Inf = {nan_lp}/{inf_lp}, "
              f"chains with any-NaN grad = {nan_grad_chains}/{args.n_samples}")
        if finite_chisq.size:
            print(f"   finite chi^2 per-pixel: min={finite_chisq.min():.3e}, "
                  f"median={np.median(finite_chisq):.3e}, "
                  f"max={finite_chisq.max():.3e}, count={finite_chisq.size}")
        if nan_grad_chains > 0:
            # Inspect the parameter vectors that triggered the NaN gradients.
            bad_idx = np.where(np.any(np.isnan(grad_np), axis=1))[0]
            print(f"   bad-grad chain indices: {bad_idx[:5].tolist()}...")
            params_x = prob_model.bij.forward(list(z.T))
            beta_vals = np.asarray(params_x[2][0]["beta"])
            cx_vals = np.asarray(params_x[2][0]["center_x"])
            cy_vals = np.asarray(params_x[2][0]["center_y"])
            print(f"   bad-grad chain beta:        {beta_vals[bad_idx][:5]}")
            print(f"   bad-grad chain center_x:    {cx_vals[bad_idx][:5]}")
            print(f"   bad-grad chain center_y:    {cy_vals[bad_idx][:5]}")
        # Also report the corresponding chi^2 distribution by initial beta bin.
        params_x = prob_model.bij.forward(list(z.T))
        beta_vals = np.asarray(params_x[2][0]["beta"])
        if seed == 0:
            order = np.argsort(beta_vals)
            print("\n   first-seed sample (beta, chi^2, grad NaN?, lp NaN?):")
            for i in order[: min(10, args.n_samples)]:
                print(f"     beta={beta_vals[i]:8.4f} "
                      f"chi^2={chisq_np[i]:10.3e} "
                      f"grad-NaN={bool(np.any(np.isnan(grad_np[i])))} "
                      f"lp-NaN={bool(np.isnan(lp_np[i]))}")
            for i in order[-min(10, args.n_samples):]:
                print(f"     beta={beta_vals[i]:8.4f} "
                      f"chi^2={chisq_np[i]:10.3e} "
                      f"grad-NaN={bool(np.any(np.isnan(grad_np[i])))} "
                      f"lp-NaN={bool(np.isnan(lp_np[i]))}")

    print("\n--- explicit beta sweep at fixed (center_x, center_y) = (0, 0) ---")
    # Use the deterministic dimension layout from `prob_model.bij`.
    sample_one = prob_model.prior.sample(seed=jax.random.PRNGKey(0))
    z_template = jnp.stack(prob_model.bij.inverse(sample_one))  # (n_dim,)
    print(f"z_template shape: {z_template.shape}")
    # The last 3 entries are (beta_z, cx_z, cy_z) because the source-light
    # JointDistributionNamed was constructed with that ordering.
    beta_z_idx = -3
    cx_z_idx = -2
    cy_z_idx = -1
    base = np.array(z_template)
    base[cx_z_idx] = 0.0
    base[cy_z_idx] = 0.0

    # LogNormal bijector for beta: physical_beta = exp(log_beta).
    # Sweep log_beta from log(0.05) to log(5.0).
    log_betas = np.linspace(np.log(0.05), np.log(5.0), 25)
    rows = []
    lens_sim_one = sim.LensSimulator(
        model_seq.phys_model, model_seq.sim_config, bs=1,
    )
    def neg_log_prob_one(z_one):
        lp, chisq = prob_model.log_prob(lens_sim_one, z_one)
        return -jnp.mean(lp) / float(observed_img.size), (lp, chisq)
    grad_one = jax.jit(jax.value_and_grad(neg_log_prob_one, has_aux=True))

    for log_beta in log_betas:
        zb = np.array(base)
        zb[beta_z_idx] = log_beta
        zb = jnp.asarray(zb)[None]
        (val, (lp, chisq)), grad = grad_one(zb)
        rows.append((
            float(np.exp(log_beta)),
            float(np.asarray(chisq)[0]) if np.asarray(chisq).size else float("nan"),
            float(np.linalg.norm(np.asarray(grad)[0])),
            bool(np.isnan(np.asarray(grad)[0]).any()),
        ))
    print("    beta        chi^2/pix    ||grad||_2  grad-NaN?")
    for b, c, g, n in rows:
        print(f"   {b:8.4f}  {c:10.3e}  {g:10.3e}  {n}")


if __name__ == "__main__":
    main()
