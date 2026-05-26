"""Part 4: monkey-patch the lstsq solver and rerun a short MAP to confirm
the cure.

Replaces gigalens.jax.simulator._solve_normal_eq_with_fallback with one of:

  - "default"  : the original Cholesky + lstsq-fallback (status quo)
  - "solve"    : jnp.linalg.solve(XtX_regularized, XtY)
  - "chol1e-4" : Cholesky path with jitter_scale = 1e-4

then runs a 20-step MAP from a fixed seed on vela01_rep03 at n_max=30 and
prints the per-iteration min chi^2 across chains.
"""

from __future__ import annotations

import argparse
import functools
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
import optax

import gigalens.jax.simulator as gsim
from gigalens.jax.simulator import _regularize_gram, _triangular_solve_from_cholesky

from vela_utilities import (
    DEFAULT_BACKGROUND_RMS,
    DEFAULT_EXP_TIME,
    free_source_fixed_lens_model,
    load_vela_sim_system,
)


def make_solver(name):
    if name == "default":
        def _solver(gram, rhs):
            def solve_one(gram_i, rhs_i):
                gram_i = _regularize_gram(gram_i)
                chol = jnp.linalg.cholesky(gram_i)
                chol_solution = _triangular_solve_from_cholesky(chol, rhs_i)
                return jax.lax.cond(
                    jnp.any(jnp.isnan(chol_solution)),
                    lambda: jnp.linalg.lstsq(gram_i, rhs_i)[0],
                    lambda: chol_solution,
                )
            if gram.shape[0] == 1:
                return solve_one(gram[0], rhs[0])[jnp.newaxis, ...]
            return jax.vmap(solve_one)(gram, rhs)
        return _solver

    if name == "solve":
        def _solver(gram, rhs):
            def solve_one(gram_i, rhs_i):
                gram_i = _regularize_gram(gram_i)
                return jnp.linalg.solve(gram_i, rhs_i)
            if gram.shape[0] == 1:
                return solve_one(gram[0], rhs[0])[jnp.newaxis, ...]
            return jax.vmap(solve_one)(gram, rhs)
        return _solver

    if name == "chol1e-4":
        def _solver(gram, rhs):
            def solve_one(gram_i, rhs_i):
                gram_i = _regularize_gram(gram_i, jitter_scale=1e-4)
                chol = jnp.linalg.cholesky(gram_i)
                return _triangular_solve_from_cholesky(chol, rhs_i)
            if gram.shape[0] == 1:
                return solve_one(gram[0], rhs[0])[jnp.newaxis, ...]
            return jax.vmap(solve_one)(gram, rhs)
        return _solver

    raise ValueError(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--solver", choices=["default", "solve", "chol1e-4"], default="solve")
    p.add_argument("--n-max", type=int, default=30)
    p.add_argument("--n-samples", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=20)
    args = p.parse_args()

    # Monkey-patch.
    gsim._solve_normal_eq_with_fallback = make_solver(args.solver)
    print(f"using solver = {args.solver}")

    observed_img, true_params, sim_config, _ = load_vela_sim_system(
        "01", 3, cam="12",
    )
    model_seq, lens_sim = free_source_fixed_lens_model(
        sim_config, observed_img, true_params,
        background_rms=DEFAULT_BACKGROUND_RMS,
        exp_time=DEFAULT_EXP_TIME,
        use_shapelets=True, n_max=args.n_max,
    )
    prob_model = model_seq.prob_model

    # Mirror inference.ModellingSequence.MAP but on a single device, single
    # chain group (no shard_map), so we can run on the login-node CPU.
    bs = args.n_samples
    lens_sim = gsim.LensSimulator(model_seq.phys_model, model_seq.sim_config, bs=bs)

    start = prob_model.prior.sample(bs, seed=jax.random.PRNGKey(0))
    params = jnp.stack(prob_model.bij.inverse(start)).T  # (n_samples, n_dim)

    def loss(z):
        lp, chisq = prob_model.log_prob(lens_sim, z)
        return -jnp.mean(lp) / float(observed_img.size), (lp, chisq)

    loss_and_grad = jax.jit(jax.value_and_grad(loss, has_aux=True))

    optimizer = optax.adabelief(1e-2, b1=0.95, b2=0.99, nesterov=True)
    opt_state = optimizer.init(params)

    for step in range(args.num_steps):
        (val, (lp, chisq)), grad = loss_and_grad(params)
        chisq_np = np.asarray(chisq)
        grad_np = np.asarray(grad)
        n_nan_chains = int(np.any(np.isnan(grad_np), axis=1).sum())
        finite = chisq_np[np.isfinite(chisq_np)]
        min_chi = float(finite.min()) if finite.size else float("nan")
        median_chi = float(np.median(finite)) if finite.size else float("nan")
        print(f"step {step:3d}: min chi^2 = {min_chi:10.3e}  "
              f"median chi^2 = {median_chi:10.3e}  "
              f"chains alive = {finite.size}/{bs}  "
              f"chains with NaN grad = {n_nan_chains}/{bs}")
        updates, opt_state = optimizer.update(grad, opt_state)
        params = optax.apply_updates(params, updates)


if __name__ == "__main__":
    main()
