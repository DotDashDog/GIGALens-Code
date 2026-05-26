from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from tensorflow_probability.substrates.jax import bijectors as tfb

from gigalens.model import ProbabilisticModel

from .pixelized_regularization import build_regularization_matrix


class PixelizedSourceProbModel(ProbabilisticModel):
    """
    Probabilistic model for pixelised source reconstruction with analytic
    marginal likelihood (evidence) following WD03 / Suyu+06 form.

    This mirrors `gigalens.jax.model.BackwardProbModel` in that it:
    - uses the prior's default event-space bijector and pack/unpack bijector
    - expects `log_prob(simulator, z)` and returns (log_post, chi_sq)

    The simulator must expose `basis_and_lens_light(x)` returning:
      - basis_images: (I, H, W) basis images already PSF-convolved + downsampled
      - lens_light:   (H, W) lens light model already PSF-convolved + downsampled
    """

    def __init__(
        self,
        prior,
        observed_image,
        background_rms,
        exp_time,
        edges,
        *,
        reg_kind: str = "constant_gradient",
        reg_vertex_positions: np.ndarray | None = None,
        reg_vertex_lam: np.ndarray | None = None,
    ):
        super().__init__(prior)
        self.reg_kind = str(reg_kind)
        self.reg_vertex_positions = (
            None
            if reg_vertex_positions is None
            else jnp.asarray(reg_vertex_positions, dtype=jnp.float32)
        )
        self.reg_vertex_lam = (
            None if reg_vertex_lam is None else jnp.asarray(reg_vertex_lam, dtype=jnp.float32)
        )
        self.observed_image = jnp.array(observed_image, dtype=jnp.float32)

        err_map = jnp.sqrt(
            jnp.float32(background_rms) ** 2
            + jnp.clip(self.observed_image, 0, np.inf).astype(jnp.float32) / jnp.float32(exp_time)
        )
        self.err_map = err_map
        self.inv_var = (1.0 / (err_map**2)).astype(jnp.float32)

        # Constant term for absolute evidence (optional but helpful for debugging).
        self.const_term = jnp.sum(jnp.log(2.0 * jnp.pi * (err_map**2))).astype(jnp.float32)

        # Static edges for regularization
        self.edges = jnp.array(edges, dtype=jnp.int32)

        example = prior.sample(seed=jax.random.PRNGKey(0))
        self.pack_bij = tfb.pack_sequence_as(example)
        self.bij = tfb.Chain(
            [
                prior.experimental_default_event_space_bijector(),
                self.pack_bij,
            ]
        )

    @functools.partial(jit, static_argnums=(0, 1))
    def log_prob(self, simulator, z):
        with jax.named_scope("pixelized_source_prob_model"):
            # MAP / SVI / HMC pass z as (batch, n_params). Do NOT broadcast the
            # source basis across the batch dimension (it is huge). Instead,
            # vmap over samples and build / solve per-sample.
            if z.ndim == 1:
                z = z[jnp.newaxis, :]

            def one(z1):
                z1 = z1[jnp.newaxis, ...]  # add batch dim of 1 for bijector
                z_list = list(z1.T)
                x_batched = self.bij.forward(z_list)
                x = jax.tree.map(lambda a: jnp.squeeze(a, axis=0), x_batched)

                out = simulator.basis_and_lens_light((x[0], x[1]))
                basis = out.basis_images  # (I,H,W)
                lens_light = out.lens_light  # (H,W)
                I = basis.shape[0]

                resid = (self.observed_image - lens_light).astype(jnp.float32)
                w_resid = resid * self.inv_var

                D = jnp.einsum("ihw,hw->i", basis, w_resid, precision="highest")
                F = jnp.einsum("ihw,khw,hw->ik", basis, basis, self.inv_var, precision="highest")

                lam = x[2][0]["lambda"].astype(jnp.float32)
                H = build_regularization_matrix(
                    self.reg_kind,
                    num_vertices=I,
                    edges=self.edges,
                    lam=lam,
                    vertex_positions=self.reg_vertex_positions,
                    vertex_lam=self.reg_vertex_lam,
                )
                A = F + H
                # Numerical jitter for Cholesky stability.
                trA = jnp.trace(A)
                jitter = jnp.float32(1e-6) * jnp.maximum(trA / I, 1.0)
                A = A + jitter * jnp.eye(I, dtype=A.dtype)
                chol = jnp.linalg.cholesky(A)
                y = jax.lax.linalg.triangular_solve(chol, D, left_side=True, lower=True)
                s = jax.lax.linalg.triangular_solve(chol.T, y, left_side=True, lower=False)

                src_img = jnp.einsum("i,ihw->hw", s, basis, precision="highest")
                model = lens_light + src_img

                chi2 = jnp.sum(((self.observed_image - model) ** 2) * self.inv_var)
                sHs = s @ (H @ s)
                logdetA = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
                logdetH = jnp.linalg.slogdet(H)[1]

                minus2logZ = chi2 + sHs + logdetA - logdetH + self.const_term
                logZ = -0.5 * minus2logZ

                log_prior = self.prior.log_prob(x) + self.bij.forward_log_det_jacobian(z_list)
                return logZ + log_prior, chi2 / (self.observed_image.size)

            return jax.vmap(one)(z)

    def debug_terms(self, simulator, z1):
        """
        Non-jitted single-sample evaluation that returns intermediate scalars.
        Intended for diagnosing non-finite log targets.
        """
        if z1.ndim == 2:
            z1 = jnp.squeeze(z1, axis=0)
        z1b = z1[jnp.newaxis, ...]
        z_list = list(z1b.T)
        x_batched = self.bij.forward(z_list)
        x = jax.tree.map(lambda a: jnp.squeeze(a, axis=0), x_batched)

        out = simulator.basis_and_lens_light((x[0], x[1]))
        basis = out.basis_images
        lens_light = out.lens_light
        I = basis.shape[0]

        resid = (self.observed_image - lens_light).astype(jnp.float32)
        w_resid = resid * self.inv_var
        D = jnp.einsum("ihw,hw->i", basis, w_resid, precision="highest")
        F = jnp.einsum("ihw,khw,hw->ik", basis, basis, self.inv_var, precision="highest")

        lam = x[2][0]["lambda"].astype(jnp.float32)
        H = build_regularization_matrix(
            self.reg_kind,
            num_vertices=I,
            edges=self.edges,
            lam=lam,
            vertex_positions=self.reg_vertex_positions,
            vertex_lam=self.reg_vertex_lam,
        )
        A = F + H
        trA = jnp.trace(A)
        jitter = jnp.float32(1e-6) * jnp.maximum(trA / I, 1.0)
        A = A + jitter * jnp.eye(I, dtype=A.dtype)
        chol = jnp.linalg.cholesky(A)
        y = jax.lax.linalg.triangular_solve(chol, D, left_side=True, lower=True)
        s = jax.lax.linalg.triangular_solve(chol.T, y, left_side=True, lower=False)

        src_img = jnp.einsum("i,ihw->hw", s, basis, precision="highest")
        model = lens_light + src_img
        chi2 = jnp.sum(((self.observed_image - model) ** 2) * self.inv_var)
        sHs = s @ (H @ s)
        logdetA = 2.0 * jnp.sum(jnp.log(jnp.diag(chol)))
        logdetH = jnp.linalg.slogdet(H)[1]
        minus2logZ = chi2 + sHs + logdetA - logdetH + self.const_term
        logZ = -0.5 * minus2logZ
        prior_lp = self.prior.log_prob(x)
        fldj = self.bij.forward_log_det_jacobian(z_list)
        log_prior = prior_lp + fldj
        return {
            "lambda": lam,
            "chi2": chi2,
            "chi2_mean": chi2 / self.observed_image.size,
            "sHs": sHs,
            "logdetA": logdetA,
            "logdetH": logdetH,
            "const": self.const_term,
            "logZ": logZ,
            "prior_lp": prior_lp,
            "fldj": fldj,
            "log_prior": log_prior,
            "log_target": logZ + log_prior,
            "degenerate_subpix": out.degenerate_subpix,
            "z_any_nonfinite": jnp.any(~jnp.isfinite(z1)),
        }

