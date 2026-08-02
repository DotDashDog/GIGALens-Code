"""Laplace / Hessian surrogate posterior for GIGALens.

:func:`HessianSurrogate` takes a point estimate ``z_best`` in unconstrained
parameter space and returns a ``tfd.MultivariateNormalTriL`` whose covariance
is the inverse of the negative log-posterior Hessian evaluated at that point —
the *Laplace approximation* to the posterior.

This is a drop-in replacement for the SVI surrogate ``qz`` that MCLMC uses to
set its initial mass matrix and chain-initialisation scatter.  Unlike SVI, it
requires only a single forward pass per Hessian-vector product and is
deterministic: there is no optimisation loop and no ELBO to track.

When the Hessian is well-defined (i.e. the MAP is a true local maximum with a
negative-definite Hessian) the Laplace covariance matches the posterior
covariance to second order, which is all MCLMC needs for a good preconditioner.

Usage example::

    from gigalens_research.inference.hessian_surrogate import HessianSurrogate
    qz = HessianSurrogate(model_seq, z_map)
    # qz.covariance() is the Laplace covariance; qz.mean() is z_map.
    # Feed directly into MCLMC_JIT or MCLMCStage as the qz argument.
"""

from __future__ import annotations

import warnings
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

# NOTE: this experimental surrogate was NOT migrated to the scene API; it built a legacy
# ``LensSimulator`` from ``model_seq.phys_model``, which was removed with the old gigalens
# API. The module stays IMPORT-SAFE (no module-level old-API import); ``HessianSurrogate``
# raises if actually CALLED. Restore from git b82397c to re-enable.
# The first parameter is named ``prob_model`` to match what live callers now pass (a scene
# ProbModel), but the UNREACHABLE body below is still written against the old
# ``model_seq`` shape — it must be rewritten, not just renamed, when this is restored.

_tfd = tfp.distributions


def HessianSurrogate(
    prob_model,
    z_best,
    *,
    fix_indefinite: bool = True,
    eigenvalue_floor: Optional[float] = None,
) -> _tfd.MultivariateNormalTriL:
    """Return a Laplace-approximation surrogate posterior as a
    ``MultivariateNormalTriL``.

    The covariance is ``C = inv(M)`` where ``M = -H(log p)`` evaluated at
    ``z_best``.  A well-converged MAP has a negative-definite ``H`` and hence
    a positive-definite ``M``.  When that is not the case (saddle points, flat
    directions, imperfect convergence) some eigenvalues of ``M`` will be
    non-positive; the function warns and optionally repairs the matrix.

    Parameters
    ----------
    model_seq : gigalens.jax.inference.ModellingSequence
        The model being inferred; provides ``prob_model``, ``phys_model``, and
        ``sim_config``.
    z_best : array-like, shape ``(n_params,)`` or ``(1, n_params)``
        Point in unconstrained parameter space at which to evaluate the
        Hessian (e.g. the MAP estimate).
    fix_indefinite : bool, default ``True``
        If ``True``, any non-positive eigenvalue of ``M`` is replaced by its
        absolute value before inversion.  This ensures the returned
        distribution is a valid Gaussian, at the cost of treating saddle
        directions as if they were the same curvature but flipped — which is
        the most conservative assumption available from local information
        alone.
    eigenvalue_floor : float or None
        Absolute lower bound applied to eigenvalues of ``M`` *after* any
        flipping, expressed as a fraction of the largest eigenvalue.  When
        ``None`` (default), ``1e-8 * max_eigenvalue`` is used.

    Returns
    -------
    qz : tfd.MultivariateNormalTriL
        Mean ``z_best``, covariance ``inv(M)`` with the above repairs applied.
        This object has the same interface as the SVI output and can be fed
        directly into ``MCLMC_JIT`` or ``MCLMCStage``.

    Warns
    -----
    UserWarning
        Emitted (once, with eigenvalue details) if any eigenvalues of ``M``
        are non-positive, regardless of the value of ``fix_indefinite``.
    """
    raise NotImplementedError(
        "HessianSurrogate was not migrated to the scene API; it used the legacy "
        "LensSimulator, which was removed with the old gigalens API. Restore from git "
        "b82397c if needed.")
    # --- build the scalar log-density callable ----------------------------
    # Create a throw-away LensSimulator; same pattern as MCLMC_JIT.
    lens_sim = _sim.LensSimulator(
        model_seq.phys_model, model_seq.sim_config, bs=1
    )

    def _logdensity(z_flat):
        return jnp.squeeze(
            model_seq.prob_model.log_prob(lens_sim, z_flat.reshape(1, -1))[0]
        )

    # --- prepare z_best as a flat 1-D array in the native dtype -----------
    z_np = np.asarray(z_best).reshape(-1)
    z0 = jnp.asarray(z_np)
    n = z0.shape[0]

    # --- build the Hessian column-by-column via HVPs ----------------------
    # Using jvp(grad, ...) avoids vmap-over-n_comp convolutions that OOM at
    # high n_max.
    grad_f = jax.grad(_logdensity)
    hvp = jax.jit(lambda v: jax.jvp(grad_f, (z0,), (v,))[1])

    cols = [
        np.asarray(hvp(jnp.zeros(n, z0.dtype).at[i].set(1.0)))
        for i in range(n)
    ]
    H = np.stack(cols, axis=1)              # Hessian of log p (symmetric)
    # Symmetrise numerically (HVPs have tiny asymmetric rounding errors).
    H = 0.5 * (H + H.T)

    # --- metric M = -H(log p); should be positive-definite at a MAP -------
    M = -H
    evals, evecs = np.linalg.eigh(M)

    # --- check for non-positive eigenvalues and warn ----------------------
    n_nonpos = int((evals <= 0).sum())
    if n_nonpos > 0:
        neg_vals = evals[evals <= 0]
        warnings.warn(
            f"HessianSurrogate: {n_nonpos}/{n} eigenvalue(s) of the negative "
            f"log-posterior Hessian are non-positive (min={neg_vals.min():.3e}, "
            f"max={neg_vals.max():.3e}).  This indicates the point estimate is "
            f"not at a strict local maximum (saddle point, flat direction, or "
            f"unconverged MAP).  "
            + (
                "Flipping their signs before inversion (fix_indefinite=True)."
                if fix_indefinite
                else "Covariance may not be positive-definite (fix_indefinite=False)."
            ),
            UserWarning,
            stacklevel=2,
        )

    # --- repair eigenvalues -----------------------------------------------
    if fix_indefinite:
        evals = np.abs(evals)

    floor = (evals.max() * 1e-10) if eigenvalue_floor is None else float(eigenvalue_floor)
    evals_safe = np.clip(evals, floor, None)

    # --- covariance C = V diag(1/evals) V^T -------------------------------
    cov = (evecs * (1.0 / evals_safe)) @ evecs.T
    cov = 0.5 * (cov + cov.T)              # enforce symmetry

    # Cholesky for MultivariateNormalTriL.
    scale_tril = np.linalg.cholesky(cov).astype(np.float32)

    return _tfd.MultivariateNormalTriL(
        loc=jnp.asarray(z_np, jnp.float32),
        scale_tril=jnp.asarray(scale_tril, jnp.float32),
    )
