"""Part 3: drill into where the n_max=30 gradient becomes NaN.

The picture from part 2 is that the *forward* lstsq_simulate is well-behaved
but value_and_grad returns NaN gradients for almost every chain with
beta >~ 0.5 at n_max=30. The lstsq pipeline inside the simulator is

    XtX = X^T X
    coeffs = solve_one(XtX, X^T Y)
        where solve_one =
            jitter -> Cholesky -> two triangular solves
            then  lax.cond( any(NaN), lstsq_fallback, chol_solution )

`lax.cond`'s gradient follows the forward branch. Cholesky's VJP is known
to blow up when the input gram matrix is near-rank-deficient (it divides
by L_{ii}^2). So the hypothesis is:

    (A) X^T X is so ill-conditioned at n_max=30 that even though the
        Cholesky forward returns finite values (because of the tiny
        jitter), the Cholesky VJP produces NaN.

This script proves (A) by stepping through three variants of
`_solve_normal_eq_with_fallback` and reporting whether the gradient is
finite:

    1. The simulator's exact pipeline.
    2. Replace the Cholesky with `jnp.linalg.lstsq` everywhere.
    3. Replace the Cholesky with a regularized pseudoinverse based on
       jnp.linalg.solve(XtX + lambda I).

For each, we evaluate the chi^2-loss gradient on a single source
parameter sample drawn from the prior and report
  - chi^2 (forward)
  - grad NaN?
  - grad norm
We also sweep jitter_scale from 1e-6 (status quo) up to 1e-2 to see if
larger jitter is enough to make the Cholesky VJP finite.
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

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

import gigalens.jax.simulator as sim

from vela_utilities import (
    DEFAULT_BACKGROUND_RMS,
    DEFAULT_EXP_TIME,
    free_source_fixed_lens_model,
    load_vela_sim_system,
)


def _regularize_gram(gram, jitter_scale):
    gram = 0.5 * (gram + jnp.swapaxes(gram, -1, -2))
    diag_mean = jnp.mean(jnp.diagonal(gram, axis1=-2, axis2=-1), axis=-1, keepdims=True)
    jitter = jitter_scale * jnp.maximum(diag_mean, 1.0)
    return gram + jitter[..., jnp.newaxis] * jnp.eye(gram.shape[-1], dtype=gram.dtype)


def solve_cholesky(gram, rhs, jitter_scale):
    gram = _regularize_gram(gram, jitter_scale)
    chol = jnp.linalg.cholesky(gram)
    y = lax.linalg.triangular_solve(chol, rhs, left_side=True, lower=True)
    return lax.linalg.triangular_solve(
        jnp.swapaxes(chol, -1, -2), y, left_side=True, lower=False
    )


def solve_lstsq(gram, rhs, jitter_scale):
    gram = _regularize_gram(gram, jitter_scale)
    return jnp.linalg.lstsq(gram, rhs)[0]


def solve_solve(gram, rhs, jitter_scale):
    """Symmetric positive-definite solve using `jnp.linalg.solve`. This uses an
    LU factorization with partial pivoting whose VJP is well-conditioned even
    for near-singular matrices, since it only requires an inverse-solve
    against the same matrix, never explicit division by tiny diagonal entries.
    """
    gram = _regularize_gram(gram, jitter_scale)
    return jnp.linalg.solve(gram, rhs)


def build_simulate_with_solver(lens_sim, solver, jitter_scale):
    """Re-implement lstsq_simulate using a chosen lstsq solver. Mirrors
    gigalens.jax.simulator.LensSimulator.lstsq_simulate but with a swappable
    `solver(gram, rhs, jitter_scale)` callable."""
    @jax.jit
    def simulate(params, observed_image, err_map):
        lens_params = params[0]
        lens_light_params = params[1]
        source_light_params = params[2]

        beta_x, beta_y = lens_sim._beta(lens_params)
        img = jnp.zeros((0, *lens_sim.img_X.shape))
        for lm, p in zip(lens_sim.phys_model.lens_light, lens_light_params):
            img = jnp.concatenate((img, lm.light(lens_sim.img_X, lens_sim.img_Y, **p)), axis=0)
        for lm, p in zip(lens_sim.phys_model.source_light, source_light_params):
            img = jnp.concatenate((img, lm.light(beta_x, beta_y, **p)), axis=0)
        img = jnp.nan_to_num(img)
        img = jnp.transpose(img, (3, 0, 1, 2))  # bs, n_comp, h, w
        if lens_sim.flat_kernel is not None:
            bs, depth, h, w = img.shape
            folded = jnp.reshape(img, (bs * depth, 1, h, w))
            conv = lax.conv(folded, lens_sim.flat_kernel, (1, 1), "SAME")
            img = jnp.reshape(conv, (bs, depth, h, w))
        if lens_sim.supersample != 1:
            from gigalens.jax.simulator import average_pool_2d
            img = average_pool_2d(
                img, size=(lens_sim.supersample, lens_sim.supersample), padding="SAME",
            )
        ret = jnp.transpose(img, (0, 2, 3, 1))  # bs, h, w, n_comp

        W = (1 / err_map)[..., jnp.newaxis]
        Y = jnp.reshape(observed_image * jnp.squeeze(W), (1, -1, 1))
        X = jnp.reshape((ret * W), (ret.shape[0], -1, ret.shape[-1]))
        Xt = jnp.transpose(X, (0, 2, 1))
        gram = Xt @ X
        rhs = Xt @ Y
        # solver expects (gram[i], rhs[i])
        coeffs = solver(gram[0], rhs[0], jitter_scale)[..., 0]
        recon = jnp.sum(ret * coeffs[None, None, None, :], axis=-1)
        return jnp.squeeze(recon) * lens_sim.conversion_factor

    return simulate


def make_loss(simulate, observed_image, err_map):
    def loss(params):
        recon = simulate(params, observed_image, err_map)
        resid = (recon - observed_image) / err_map
        return jnp.mean(resid ** 2)
    return loss


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim", default="01")
    p.add_argument("--rep", type=int, default=3)
    args = p.parse_args()

    observed_img, true_params, sim_config, _ = load_vela_sim_system(
        args.sim, args.rep, cam="12",
    )

    # We test n_max=20 (known-good in MAP) and n_max=30 (failing).
    results = []
    for n_max in (20, 30):
        prob_model, lens_sim = free_source_fixed_lens_model(
            sim_config, observed_img, true_params,
            background_rms=DEFAULT_BACKGROUND_RMS,
            exp_time=DEFAULT_EXP_TIME,
            use_shapelets=True, n_max=n_max,
        )
        err_map = prob_model.err_map

        # Two parameter points: one with beta ~ 0.16 (works) and one with
        # beta ~ 0.7 (fails at n_max=30 per part 2).
        sample = prob_model.prior.sample(seed=jax.random.PRNGKey(0))
        params_lo = jax.tree.map(lambda x: jnp.asarray(x)[None] if jnp.asarray(x).ndim == 0 else jnp.asarray(x), sample)
        sample2 = prob_model.prior.sample(seed=jax.random.PRNGKey(42))
        params_hi = jax.tree.map(lambda x: jnp.asarray(x)[None] if jnp.asarray(x).ndim == 0 else jnp.asarray(x), sample2)

        # Force a specific beta by patching the source dict.
        def set_beta(params, beta):
            new = jax.tree.map(lambda x: x, params)
            new[2][0]["beta"] = jnp.asarray(beta)[None] if jnp.ndim(beta) == 0 else jnp.asarray(beta)
            new[2][0]["center_x"] = jnp.zeros((1,), dtype=jnp.float32)
            new[2][0]["center_y"] = jnp.zeros((1,), dtype=jnp.float32)
            return new

        for beta_test in (0.16, 0.45, 0.7, 1.5):
            params = set_beta(params_lo, beta_test)
            for solver_name, solver in [
                ("cholesky", solve_cholesky),
                ("lstsq", solve_lstsq),
                ("solve", solve_solve),
            ]:
                for jitter_scale in (1e-6, 1e-4, 1e-2):
                    simulate = build_simulate_with_solver(lens_sim, solver, jitter_scale)
                    loss = make_loss(simulate, observed_img, err_map)
                    try:
                        val, grad = jax.value_and_grad(loss)(params)
                        # grad is a nested list - flatten leaves
                        leaves, _ = jax.tree_util.tree_flatten(grad)
                        flat = np.concatenate([np.asarray(l).ravel() for l in leaves])
                        gn = float(np.linalg.norm(flat[np.isfinite(flat)]))
                        nan_count = int(np.isnan(flat).sum())
                    except Exception as e:
                        val = float("nan")
                        gn = float("nan")
                        nan_count = -1
                    results.append({
                        "n_max": n_max,
                        "beta": beta_test,
                        "solver": solver_name,
                        "jitter": jitter_scale,
                        "loss": float(val),
                        "grad_norm": gn,
                        "grad_nan_count": nan_count,
                    })
                    print(f"n_max={n_max:2d} beta={beta_test:5.2f} "
                          f"solver={solver_name:9s} jitter={jitter_scale:1.0e} "
                          f"loss={float(val):.3e} "
                          f"grad_nan={nan_count:4d} grad_norm={gn:.3e}")

    print("\n--- summary: count of grad_nan>0 cases per (n_max, solver, jitter) ---")
    import collections
    counts = collections.Counter()
    for r in results:
        if r["grad_nan_count"] > 0:
            counts[(r["n_max"], r["solver"], r["jitter"])] += 1
    for k, c in sorted(counts.items()):
        print(f"  {k}: {c} bad cases")


if __name__ == "__main__":
    main()
