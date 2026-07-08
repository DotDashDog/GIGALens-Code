"""Flow-preconditioned sampling (NeuTra-style): transformed-ProbModel wrapper.

Implements the u-space pullback of docs/plans/flow-preconditioned-mams.md (§2, §5.1):

    log_pi_tilde(u) = inner.log_prob(T(u))[0] + log|det J_T(u)|

so that MAMS (or NUTS) runs UNCHANGED on the wrapped model in the flow's latent
space, and final draws are decoded via z = T(u). The Metropolis correction of the
kernel keeps everything exactly unbiased regardless of flow quality.

Only the interface `MAMS_JIT` actually touches is exposed
(`model_seq.prob_model.log_prob`; `qz.sample/covariance/mean` via `make_latent_qz`).
Nothing else is passed through on purpose: u-space columns are NOT parameters, and
any per-parameter diagnostic (R-hat/ESS/traces/occupancy) must be computed on
z = T(u), never raw u. Decode with `u_to_z` and then the inner model's `bij`.
"""

import jax.numpy as jnp

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


class TransformedProbModel:
    """Pulled-back target: ProbModel seen through a flow T: u -> z.

    `flow` is a tfp-jax bijector whose `forward` maps latent u to the inner
    model's unconstrained z (event_ndims=1 over the last axis). With
    `flow = tfb.Identity()` this wrapper must reproduce the inner model bit-for-bit
    (GATE I of the plan).
    """

    def __init__(self, inner_prob_model, flow):
        self.inner = inner_prob_model
        self.flow = flow

    def log_prob(self, u):
        z = self.flow.forward(u)
        lp, red_chi2 = self.inner.log_prob(z)
        fldj = self.flow.forward_log_det_jacobian(u, event_ndims=1)
        # Jacobian enters the log-density only; red_chi2 stays a pure data-fit
        # readout in z-space.
        return lp + fldj, red_chi2

    def u_to_z(self, u):
        return self.flow.forward(u)

    def z_to_u(self, z):
        return self.flow.inverse(z)

    def bij_forward(self, u):
        """Physical decoding of latent draws: inner.bij.forward(T(u)).

        Follows the inner model's `bij.forward` calling convention exactly
        (e.g. list-of-columns in the scene API) after mapping u through the flow.
        """
        return self.inner.bij.forward(self.flow.forward(u))


class FlowModelSeq:
    """Minimal model_seq shim for MAMS_JIT, which touches only `.prob_model`.

    Deliberately does NOT forward MAP/SVI/etc. -- those run on the inner model
    in z-space before the flow exists.
    """

    def __init__(self, inner_model_seq, transformed_prob_model):
        self.inner = inner_model_seq
        self.prob_model = transformed_prob_model


def make_latent_qz(dim, dtype=jnp.float64):
    """Latent-space stand-in for `qz`: the flow's base distribution N(0, I).

    MAMS_JIT consumes qz three ways (mams.py:73,79,91): chain init draws,
    `covariance()` as the initial inverse mass matrix, and `mean()` as the
    Welford/SVI reference. With N(0, I) all three are simultaneously right in
    u-space: init = base draws (pushed through T they are flow draws), initial
    inverse mass = identity (the flow whitens by construction), reference
    mean = 0 (the whitened SVI mean). NOTE: this deviates from the plan's
    "MultivariateNormalDiag(0, 1e-2)" sketch, which would silently install
    1e-4 * I as the initial inverse mass via `qz.covariance()`.
    """
    return tfd.MultivariateNormalDiag(
        loc=jnp.zeros(dim, dtype=dtype), scale_diag=jnp.ones(dim, dtype=dtype)
    )
