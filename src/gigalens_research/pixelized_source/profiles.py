"""Pixelized-source light profiles for the gigalens scene API.

Both profiles evaluate a frozen :class:`~.domain.SourceDomain` at the ray-traced
positions the scene simulator hands every light profile (``light(x, y, **params)``
with the sample batch on the trailing axis). Nothing about the lens, the grid, the PSF
or the datasets lives here -- that is the point of putting them on the scene API.

* :class:`MeshSource` -- an lstsq basis with one function per vertex
  (``use_lstsq=True``, ``depth = I``). Its only parameters are the hyperparameters of
  the optional :class:`~.regularizers.QuadraticRegularizer` attached to it (e.g.
  ``lam``); the basis does not depend on them. With a regularizer the profile declares
  a ``linear_prior`` (gigalens' ``LightProfile`` hook) and every imaging dataset that
  sees it marginalizes the coefficients analytically; without one it is an ordinary
  flat-prior lstsq component.
* :class:`CorrelatedFieldSource` -- a forward-mode Gaussian random field on a
  :class:`~.domain.RegularGridDomain`: ``s = exp(mean_log + exp(log_amp) * f)`` with
  ``f = IFFT(A(k; slope) FFT(xi))``, unit-variance normalized, ``xi ~ N(0, 1)`` per
  grid node. The excitations and the spectral hyperparameters are ordinary sampled
  parameters, so the prior is expressed as a prior and nothing is marginalized.
"""
from __future__ import annotations

from typing import Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

import gigalens.profile
from gigalens.physicality import Domain

from .domain import RegularGridDomain, SourceDomain
from .regularizers import QuadraticRegularizer

tfd = tfp.distributions

__all__ = ["MeshSource", "CorrelatedFieldSource"]


class MeshSource(gigalens.profile.LightProfile):
    """One lstsq basis function per domain vertex, evaluated at the deflected grid."""

    _name = "MESH_SOURCE"
    _params: list = []
    _domains = {
        "lam": Domain(lo=0.0, lo_open=True, rationale=(
            "regularization strength multiplies an SPD matrix; lam <= 0 makes the "
            "coefficient prior improper / indefinite and log det H undefined.")),
    }
    _domain_fallback = Domain(rationale=(
        "regularizer hyperparameters are declared by the attached QuadraticRegularizer; "
        "unlisted ones are unbounded here and constrained by their priors."))

    def __init__(self, domain: SourceDomain, regularizer: Optional[QuadraticRegularizer] = None):
        super().__init__(use_lstsq=True)
        self.domain = domain
        self.regularizer = regularizer
        self.params = list(regularizer.hyperparams) if regularizer is not None else []
        self.depth = int(domain.n_basis)

    # -- coefficient prior (gigalens LightProfile.linear_prior hook) --------------------
    @property
    def has_linear_prior(self) -> bool:
        return self.regularizer is not None

    def linear_prior(self, **hyper):
        """``(H, log det H)`` from the attached regularizer, or ``None`` (flat prior).

        Hyperparameter leaves may carry a trailing sample-batch axis; the regularizer
        broadcasts them to ``(batch, I, I)`` / ``(batch,)`` as the hook requires.
        """
        if self.regularizer is None:
            return None
        return self.regularizer.matrix(**hyper), self.regularizer.logdet(**hyper)

    @property
    def use_lstsq(self):
        return True

    @use_lstsq.setter
    def use_lstsq(self, value: bool):
        if not value:
            raise ValueError(
                "MeshSource is an lstsq basis; explicit vertex values belong to a forward-mode "
                "profile such as CorrelatedFieldSource.")
        self._use_lstsq = True

    def light(self, x, y, **hyper):
        # hyper (regularizer hyperparameters) do not enter the basis; gigalens reads them
        # through linear_prior() when it marginalizes the coefficients.
        return self.domain.basis(x, y)

    def __str__(self):
        reg = "" if self.regularizer is None else f", {self.regularizer}"
        return f"MeshSource({self.domain}{reg})"


class CorrelatedFieldSource(gigalens.profile.LightProfile):
    """Log-normal Gaussian random field on a regular source grid (forward mode).

    Parameters (all sampled): ``mean_log`` (log of the geometric-mean surface
    brightness), ``log_amp`` (log of the field's standard deviation in log-space),
    ``slope`` (power-law index of the amplitude spectrum, ``A(k) ∝ (1 + (k/k0)^2)^(-slope/2)``
    with ``k0`` the grid's fundamental wavenumber; 0 is white noise), and
    ``xi_<i>`` for every grid node (standard-normal excitations, row-major order).

    Use :meth:`xi_prior` for the excitation prior: a single grouped (tuple-key)
    ``tfd.Sample(Normal(0, 1), n)`` site, which the scene packs as one ``n``-vector
    (much cheaper than ``n`` scalar sites). The three hyperparameter priors are yours
    to supply -- no defaults.
    """

    _name = "CORRELATED_FIELD_SOURCE"
    _params = ["mean_log", "log_amp", "slope"]
    _domains = {
        "mean_log": Domain(rationale="log of a positive brightness; any finite value."),
        "log_amp": Domain(rationale="log of a standard deviation; any finite value."),
        "slope": Domain(lo=0.0, rationale=(
            "spectral index of the amplitude spectrum; negative values put power at high "
            "k (rougher than white noise), which is allowed numerically but is never the "
            "intent for a galaxy -- kept hard at 0 so a sign slip is loud.")),
    }
    _domain_fallback = Domain(rationale="standard-normal excitations; unbounded by design.")

    def __init__(self, domain: RegularGridDomain, use_lstsq: bool = False):
        if use_lstsq:
            raise ValueError("CorrelatedFieldSource is a forward-mode profile (use_lstsq=False)")
        if not isinstance(domain, RegularGridDomain):
            raise TypeError("CorrelatedFieldSource needs a RegularGridDomain (FFT-based covariance)")
        super().__init__(use_lstsq=False)
        del self.params[self.params.index(self._amp)]  # no scalar amplitude
        self.domain = domain
        ny, nx = domain.shape
        self.n_nodes = ny * nx
        width = len(str(self.n_nodes))
        self.xi_names = [f"xi_{str(i).zfill(width)}" for i in range(self.n_nodes)]
        self.params = list(self._params) + list(self.xi_names)
        self.depth = 1
        # Wavenumber magnitude on the rfft grid, in units of the fundamental k0.
        ky = np.fft.fftfreq(ny, d=domain.dy)
        kx = np.fft.rfftfreq(nx, d=domain.dx)
        k0 = 1.0 / max(ny * domain.dy, nx * domain.dx)
        self._k_over_k0 = jnp.asarray(np.sqrt(kx[None, :] ** 2 + ky[:, None] ** 2) / k0)
        # irfft2 normalization: a white real field has E|rfft2(xi)_k|^2 ∝ N with the
        # Hermitian half-plane double-counted except on the self-conjugate columns.
        mult = np.full((ny, kx.size), 2.0)
        mult[:, 0] = 1.0
        if nx % 2 == 0:
            mult[:, -1] = 1.0
        self._parseval_mult = jnp.asarray(mult)

    # -- priors ----------------------------------------------------------------------
    def xi_prior(self, dtype=np.float64) -> Dict[tuple, tfd.Distribution]:
        """``{tuple(xi_names): Sample(Normal(0, 1), n)}`` -- drop into the Component's
        priors dict alongside the three hyperparameter priors."""
        one = np.asarray(1.0, dtype=dtype)
        return {tuple(self.xi_names): tfd.Sample(tfd.Normal(0.0 * one, one), self.n_nodes)}

    # -- field -----------------------------------------------------------------------
    def _amplitude_spectrum(self, slope):
        A = (1.0 + self._k_over_k0 ** 2) ** (-0.5 * slope)
        # Unit variance: Var[f] = mean over the full plane of A^2 = sum(mult * A^2) / N
        norm = jnp.sqrt(jnp.sum(self._parseval_mult * A ** 2) / self.n_nodes)
        return A / norm

    def field(self, xi, slope):
        """Unit-variance correlated field ``(ny, nx)`` from excitations ``(ny, nx)``."""
        ny, nx = self.domain.shape
        A = self._amplitude_spectrum(slope)
        return jnp.fft.irfft2(jnp.fft.rfft2(xi) * A, s=(ny, nx))

    def values(self, mean_log, log_amp, slope, xi):
        """Vertex values ``(I,)`` (or ``(I, B)`` for batched inputs)."""
        ny, nx = self.domain.shape

        def one(m, la, sl, x):
            f = self.field(x.reshape(ny, nx), sl)
            return jnp.exp(m + jnp.exp(la) * f).reshape(-1)

        if jnp.ndim(xi) == 1:
            return one(mean_log, log_amp, slope, xi)
        b = xi.shape[-1]
        bc = lambda a: jnp.broadcast_to(jnp.asarray(a), (b,))
        return jax.vmap(one, in_axes=(0, 0, 0, 1), out_axes=1)(
            bc(mean_log), bc(log_amp), bc(slope), xi)

    def light(self, x, y, mean_log, log_amp, slope, **xi):
        xi_arr = jnp.stack([xi[n] for n in self.xi_names], axis=0)  # (I,) or (I, B)
        vals = self.values(mean_log, log_amp, slope, xi_arr)
        return self.domain.interpolate(vals, x, y)

    def __str__(self):
        return f"CorrelatedFieldSource({self.domain.shape[0]}x{self.domain.shape[1]})"
