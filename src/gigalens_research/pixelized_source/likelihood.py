"""Research-side evidence term for regularized lstsq sources (Option A prototype).

This is the *no-gigalens-change* way to get the analytically marginalized likelihood
(Suyu et al. 2006 / Warren & Dye 2003) for a :class:`~.profiles.MeshSource` with a
quadratic prior: a dataset subclass whose likelihood term renders the ordinary lstsq
basis stack through the scene simulator and then does its own solve, adding
``s^T H s + log det A - log det H`` to the Gaussian chi-square. See
``docs/plans/pixelized-source-regularizer-options.md`` for the alternatives and why
this one is a prototype, not the recommended permanent home:

* it reaches into ``SceneSimulator._light`` to find which columns of the basis stack
  belong to which component;
* it re-implements the normal-equation tail of ``lstsq_simulate``;
* it marginalizes ALL linear coefficients (flat prior on unregularized components), so
  its log-likelihood is the evidence, not the profile likelihood ``lstsq`` mode
  reports -- the two differ by ``-1/2 log det A``, which depends on the mass.

Use :class:`RegularizedImageData` exactly like ``ImageData``; every seen light
component is rendered by the scene simulator as usual.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from gigalens.jax.scene_prob_model import ImageData, ImageLikelihoodTerm
from gigalens.jax.simulator import _regularize_gram  # same jitter policy as lstsq_simulate

from .profiles import MeshSource

__all__ = ["RegularizedImageData", "RegularizedImageLikelihoodTerm"]


class RegularizedImageData(ImageData):
    """``ImageData`` whose likelihood is the evidence over the linear coefficients,
    with the quadratic priors declared by the seen :class:`MeshSource` components."""

    def build_term(self, model, seen: List, *, default_mode: str = "lstsq"):
        mode = default_mode if self.mode is None else self.mode
        if mode != "lstsq":
            raise ValueError(
                "RegularizedImageData marginalizes lstsq coefficients; it has no meaning "
                f"in mode {mode!r}. Use plain ImageData for forward-mode datasets.")
        return RegularizedImageLikelihoodTerm(self, model, seen, mode)


class RegularizedImageLikelihoodTerm(ImageLikelihoodTerm):
    """Evidence over the linear coefficients of one imaging dataset.

    ``log_like(params) = -1/2 [chi2(s*) + s*^T H s* + log det A - log det H + norm]``
    with ``A = F + H``, ``F = X^T W X`` the weighted Gram of the rendered basis stack,
    ``s* = A^-1 X^T W y`` the MAP coefficients, ``H`` block-diagonal over components
    (zero for unregularized ones) and ``norm`` the Gaussian normalization over the
    unmasked pixels. ``log det H`` is summed over the regularized blocks only (the flat
    prior on the other blocks contributes a constant, dropped).
    """

    def __init__(self, dataset, model, seen, mode):
        super().__init__(dataset, model, seen, mode)
        self._regs = self._find_regularized(self.simulator)
        if not self._regs:
            raise ValueError(
                "RegularizedImageData: none of the seen light components is a MeshSource "
                "with a regularizer; use plain ImageData (nothing to marginalize with a prior).")

    @staticmethod
    def _find_regularized(sim) -> List[Dict]:
        """``[{plane, light, profile, offset, depth}]`` for each regularized MeshSource in
        this simulator's basis-stack order (``sim._light`` order, offsets cumulative)."""
        out = []
        offset = 0
        for i, j, comp, d in sim._light:
            prof = comp.profile
            if isinstance(prof, MeshSource) and prof.regularizer is not None:
                out.append(dict(plane=i, light=j, profile=prof, offset=offset, depth=d))
            offset += d
        return out

    # -- core --------------------------------------------------------------------------
    def _design(self, params, sim):
        """Weighted, masked design ``X (bs, npix, ncomp)`` and target ``y (npix,)``."""
        ds = self.dataset
        stacked = sim.lstsq_simulate(params, ds.image, ds.error_map, ds.mask,
                                     return_stacked=True)  # (bs, h, w, ncomp)
        if stacked.ndim == 3:
            stacked = stacked[jnp.newaxis]
        mask = ds.mask.astype(stacked.dtype)
        w = (mask / ds.error_map)
        X = (stacked * w[jnp.newaxis, ..., jnp.newaxis]).reshape(stacked.shape[0], -1, stacked.shape[-1])
        y = (ds.image * w).reshape(-1)
        return X, y

    def _prior_matrix(self, params, sim, ncomp: int, b: int):
        """Block-diagonal ``H (b, ncomp, ncomp)`` and ``log det H_reg (b,)`` for a batch."""
        H = jnp.zeros((b, ncomp, ncomp))
        logdet = jnp.zeros((b,))
        for r in self._regs:
            hyper = sim.model.component_params(params, r["plane"], "light", r["light"])
            hyper = {k: jnp.broadcast_to(jnp.asarray(hyper[k]), (b,))
                     for k in r["profile"].regularizer.hyperparams}
            reg = r["profile"].regularizer
            Hk = jax.vmap(lambda **h: reg.matrix(**h))(**hyper)      # (b, d, d)
            ld = jax.vmap(lambda **h: reg.logdet(**h))(**hyper)      # (b,)
            o, d = r["offset"], r["depth"]
            H = H.at[:, o:o + d, o:o + d].add(Hk.astype(H.dtype))
            logdet = logdet + ld
        return H, logdet

    def evidence_terms(self, params, *, simulator=None) -> Dict[str, jnp.ndarray]:
        """All the pieces, each ``(bs,)``: ``chi2, sHs, logdetA, logdetH, norm, log_like``,
        plus ``coeffs (bs, ncomp)``."""
        sim = self.simulator if simulator is None else simulator
        ds = self.dataset
        X, y = self._design(params, sim)
        b, _, ncomp = X.shape
        H, logdetH = self._prior_matrix(params, sim, ncomp, b)

        def one(Xi, Hi):
            F = Xi.T @ Xi
            D = Xi.T @ y
            A = _regularize_gram(F + Hi)
            s = jnp.linalg.solve(A, D)
            r = y - Xi @ s
            chi2 = jnp.sum(r ** 2)
            sHs = s @ (Hi @ s)
            logdetA = jnp.linalg.slogdet(A)[1]
            return chi2, sHs, logdetA, s

        chi2, sHs, logdetA, s = jax.vmap(one)(X, H)
        maskb = ds.mask.astype(bool)
        norm = jnp.sum(jnp.where(maskb, jnp.log(2 * jnp.pi * ds.error_map ** 2), 0.0))
        log_like = -0.5 * (chi2 + sHs + logdetA - logdetH + norm)
        return dict(chi2=chi2, sHs=sHs, logdetA=logdetA, logdetH=logdetH,
                    norm=jnp.broadcast_to(norm, chi2.shape), log_like=log_like, coeffs=s)

    def log_like(self, params, *, simulator=None):
        t = self.evidence_terms(params, simulator=simulator)
        return t["log_like"], t["chi2"]

    # -- conveniences ------------------------------------------------------------------
    def coefficients(self, params, *, simulator=None) -> jnp.ndarray:
        """MAP linear coefficients ``(bs, ncomp)`` (all components, stack order)."""
        return self.evidence_terms(params, simulator=simulator)["coeffs"]

    def component_coefficients(self, params, component, *, simulator=None) -> jnp.ndarray:
        """``(bs, I)`` vertex values of one regularized MeshSource ``Component``."""
        sim = self.simulator if simulator is None else simulator
        for r in self._regs:
            if r["profile"] is component.profile:
                s = self.coefficients(params, simulator=sim)
                return s[:, r["offset"]:r["offset"] + r["depth"]]
        raise KeyError("component is not a regularized MeshSource seen by this dataset")

    def model_image(self, params, *, simulator=None) -> jnp.ndarray:
        """Regularized model image ``(bs, h, w)`` (mask not applied)."""
        sim = self.simulator if simulator is None else simulator
        ds = self.dataset
        stacked = sim.lstsq_simulate(params, ds.image, ds.error_map, ds.mask, return_stacked=True)
        if stacked.ndim == 3:
            stacked = stacked[jnp.newaxis]
        s = self.coefficients(params, simulator=sim)
        return jnp.einsum("bhwc,bc->bhw", stacked, s)
