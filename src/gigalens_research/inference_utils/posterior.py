"""Posterior view layer.

A :class:`Posterior` wraps the outputs of one inference stage and exposes:

- raw arrays in unconstrained (``z``) space,
- physical-space (``x``) parameters via the bijector, lazily,
- representative points (``median``, ``mean``, ``best``),
- model-aware methods that use the lens simulator: predicted-image rendering,
  source-plane rendering, etc.,
- (samplers only) chain diagnostics: ``rhat``, ``ess``, and running variants.

Three concrete subclasses share the abstract base:

- :class:`SamplerPosterior` — chains from HMC, MCLMC, NUTS, ...
- :class:`SurrogatePosterior` — parametric (e.g. SVI Gaussian),
- :class:`PointEstimate` — MAP / single best fit.

Construction goes through :meth:`Pipeline.posterior` for in-memory results and
:func:`gigalens_research.inference_utils.pipeline.posterior_from_disk` for
loading saved runs without an active pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp

import gigalens.jax.simulator as _sim

_tfd = tfp.distributions

#: Default cap on the number of (chain-flattened) samples returned by
#: :attr:`SamplerPosterior.flat_z`. Set via ``subsample_n=`` on construction.
DEFAULT_SUBSAMPLE_N = 5000


def _n_basis(light_model) -> int:
    """Number of linear-amplitude basis functions a light profile contributes
    when used with ``use_lstsq=True``.

    Reads the profile's ``depth`` (the number of basis rows its ``light()``
    returns in lstsq mode), which is set on every profile:

    - single-component profiles (Sersic, SersicEllipse, ...) -> 1,
    - Shapelets -> ``n_layers``,
    - composite profiles (e.g. SersicShapelets) -> sum of their parts
      (so a single ``n_layers`` attribute, which composites lack, would
      undercount them).

    Used by :meth:`Posterior.source_plane` to slice the solved coefficient
    vector returned by :meth:`LensSimulator.lstsq_simulate`, which is laid out
    as ``[lens_light_bases..., source_light_bases...]`` in profile order.
    """
    return int(getattr(light_model, "depth", getattr(light_model, "n_layers", 1)))


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Posterior(ABC):
    """Base view over one inference stage's outputs."""

    def __init__(self, ctx):
        self.ctx = ctx
        self._sim_cache: dict = {}

    # -- subclass contract ---------------------------------------------------

    @property
    @abstractmethod
    def n_params(self) -> int: ...

    @abstractmethod
    def _point_z(self, name: str) -> np.ndarray:
        """Return a single (n_params,) z-vector for a representative point.

        Supported ``name`` values vary by subclass; ``'median'`` is universal.
        """

    # -- shared utilities ----------------------------------------------------

    def _lens_sim(self, bs: int = 1) -> _sim.LensSimulator:
        if bs not in self._sim_cache:
            self._sim_cache[bs] = _sim.LensSimulator(
                self.ctx.phys_model, self.ctx.sim_config, bs=bs,
            )
        return self._sim_cache[bs]

    def z_to_x(self, z) -> List:
        """Apply the bijector forward to a ``z`` of shape ``(n_params,)`` or
        ``(n, n_params)``. Returns the nested-list-of-dicts structure the
        :class:`~gigalens.jax.simulator.LensSimulator` expects."""
        z = jnp.atleast_2d(jnp.asarray(z))
        return self.ctx.prob_model.bij.forward(list(z.T))

    @property
    def is_backward(self) -> bool:
        """True iff the prob_model uses ``BackwardProbModel`` (lstsq amplitudes).

        Duck-typed on the ``err_map`` attribute; matches both gigalens'
        ``BackwardProbModel`` and any drop-in replacement that publishes the
        same field.
        """
        return hasattr(self.ctx.prob_model, "err_map")

    # -- model-aware rendering ----------------------------------------------

    def simulate(self, point: str = "median", *, return_coeffs: bool = False):
        """Render the predicted PSF-convolved image at a representative point.

        Auto-selects between :meth:`LensSimulator.simulate` (forward model)
        and :meth:`LensSimulator.lstsq_simulate` (backward / linear-amplitude
        model) by checking the prob_model for an ``err_map`` attribute.

        With ``return_coeffs=True``, also returns the solved linear amplitudes
        (or ``None`` for forward models).
        """
        x = self.z_to_x(self._point_z(point))
        sim = self._lens_sim(bs=1)
        if self.is_backward:
            obs = self.ctx.prob_model.observed_image
            err_map = self.ctx.prob_model.err_map
            img, coeffs = sim.lstsq_simulate(x, obs, err_map)
            img = np.asarray(img); coeffs = np.asarray(coeffs)
            return (img, coeffs) if return_coeffs else img
        img = np.asarray(sim.simulate(x))
        return (img, None) if return_coeffs else img

    def err_map_at(self, predicted) -> np.ndarray:
        """Per-pixel noise σ for a given predicted image, using the
        prob_model's noise convention.

        - :class:`BackwardProbModel` (anything that publishes an ``err_map``
          attribute): returns that ``err_map`` directly — it was frozen at
          init from the *observed* image, so the ``predicted`` argument is
          ignored.
        - :class:`ForwardProbModel` (anything that publishes
          ``background_rms`` and ``exp_time``): returns
          ``sqrt(max(predicted, 0) / exp_time + background_rms**2)``.

        Raises ``TypeError`` if the prob_model doesn't match either pattern.
        """
        pm = self.ctx.prob_model
        if hasattr(pm, "err_map"):
            return np.asarray(pm.err_map)
        if hasattr(pm, "background_rms") and hasattr(pm, "exp_time"):
            bg = float(np.asarray(pm.background_rms))
            et = float(np.asarray(pm.exp_time))
            pred = np.asarray(predicted)
            return np.sqrt(np.clip(pred, 0.0, np.inf) / et + bg ** 2)
        raise TypeError(
            f"prob_model {type(pm).__name__} exposes neither err_map nor "
            f"(background_rms, exp_time); cannot derive a per-pixel noise σ."
        )

    def normalized_residual(self, observed, point: str = "median") -> np.ndarray:
        """Convenience: ``(observed - predicted) / σ`` at a representative point,
        with σ from :meth:`err_map_at`. Returns a plain ``np.ndarray``.
        """
        predicted = np.asarray(self.simulate(point=point))
        sigma = self.err_map_at(predicted)
        return (np.asarray(observed) - predicted) / sigma

    def source_plane(
        self,
        point: str = "median",
        *,
        grid_pix: int = 400,
        fov_arcsec: Optional[float] = None,
        center: Optional[Tuple[float, float]] = None,
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Render the intrinsic source brightness on a fine source-plane grid.

        The grid is centered on the first source-light component's
        ``(center_x, center_y)`` (or ``center`` if supplied) and spans
        ``fov_arcsec`` on a side. No lensing, no PSF.

        For backward (lstsq) models, each profile's basis stack is contracted
        against the corresponding slice of the solved coefficients from
        :meth:`simulate`. This handles both single-component profiles
        (Sersic with ``use_lstsq=True``, contributing 1 basis function) and
        multi-component ones (Shapelets, contributing ``(n_max+1)(n_max+2)/2``
        basis functions). For forward models, each profile's light is rendered
        directly with its modeled amplitude.

        Returns
        -------
        img : ndarray, shape ``(grid_pix, grid_pix)``
        extent : tuple ``(x_min, x_max, y_min, y_max)`` for ``ax.imshow``.
        """
        x = self.z_to_x(self._point_z(point))
        src_params_list = x[-1]  # source light is the last group

        first = src_params_list[0]
        if center is None:
            cx = float(np.squeeze(np.asarray(first.get("center_x", 0.0))))
            cy = float(np.squeeze(np.asarray(first.get("center_y", 0.0))))
        else:
            cx, cy = float(center[0]), float(center[1])
        if fov_arcsec is None:
            fov_arcsec = self.ctx.sim_config.num_pix * self.ctx.sim_config.delta_pix
        half = fov_arcsec / 2.0
        gx = jnp.linspace(-half, half, grid_pix) + cx
        gy = jnp.linspace(-half, half, grid_pix) + cy
        # Light models expect arrays with a trailing "depth" axis for batching.
        X = jnp.broadcast_to(gx[None, :, None], (grid_pix, grid_pix, 1))
        Y = jnp.broadcast_to(gy[:, None, None], (grid_pix, grid_pix, 1))

        if self.is_backward:
            # lstsq path: each profile's light() returns a basis stack of shape
            # (n_layers_i, h, w, depth); contract with the matching coeff slice.
            _, coeffs = self.simulate(point=point, return_coeffs=True)
            # lstsq_simulate squeezes a single-basis solve down to a scalar;
            # keep it indexable as a length-1 coefficient vector.
            coeffs = np.atleast_1d(np.asarray(coeffs))
            offset = sum(_n_basis(lm) for lm in self.ctx.phys_model.lens_light)
            src_offset = 0
            img = jnp.zeros((grid_pix, grid_pix))
            for light_model, p in zip(self.ctx.phys_model.source_light, src_params_list):
                stack = light_model.light(X, Y, **p)  # (n_layers, h, w, depth)
                n_layers = _n_basis(light_model)
                c = jnp.asarray(coeffs[offset + src_offset : offset + src_offset + n_layers])
                # contract basis axis: sum_i c_i * stack[i]
                contribution = jnp.tensordot(c, stack, axes=([0], [0]))  # (h, w, depth)
                img = img + jnp.squeeze(contribution)
                src_offset += n_layers
            # The lstsq design matrix omits the simulator's pixel-area
            # conversion_factor (= det(transform_pix2angle) = delta_pix^2),
            # which LensSimulator.simulate applies to convert surface
            # brightness -> flux/pixel. The solved coefficients therefore
            # absorb that factor, so the contracted source is in flux/pixel
            # units. Divide it back out to express the recovered source as a
            # surface brightness, matching both the forward path below and an
            # intrinsic-SB truth (e.g. ImageBasedLight in cps/arcsec^2).
            img = img / self._lens_sim(bs=1).conversion_factor
        else:
            # Forward path: each profile's light() returns (h, w, depth) with
            # its modeled amplitude baked in; sum directly.
            img = jnp.zeros((grid_pix, grid_pix))
            for light_model, p in zip(self.ctx.phys_model.source_light, src_params_list):
                contribution = jnp.squeeze(light_model.light(X, Y, **p))
                img = img + contribution

        extent = (cx - half, cx + half, cy - half, cy + half)
        return np.asarray(img), extent

    # -- common conveniences -------------------------------------------------

    @cached_property
    def median_x(self):
        return self.z_to_x(self._point_z("median"))


# ---------------------------------------------------------------------------
# SamplerPosterior: HMC / MCLMC / NUTS / ...
# ---------------------------------------------------------------------------


class SamplerPosterior(Posterior):
    """View over chained MCMC samples.

    Parameters
    ----------
    ctx : InferenceContext
    samples_z : ndarray, shape ``(n_chains, n_steps, n_params)``
        Canonical sample layout. HMCStage / MCLMCStage already reshape to this.
    subsample_n : int or None
        Cap on the number of chain-flattened samples returned by ``flat_z`` /
        ``flat_x``. Statistics (median/mean/quantiles/rhat/ess) always use
        the full sample set; this only affects plotting density. Set to
        ``None`` to disable subsampling.
    seed : int
        RNG seed for the subsampling choice.
    """

    def __init__(self, ctx, samples_z, *, subsample_n: Optional[int] = DEFAULT_SUBSAMPLE_N, seed: int = 0):
        super().__init__(ctx)
        sz = np.asarray(samples_z)
        if sz.ndim != 3:
            raise ValueError(
                f"samples_z must be (n_chains, n_steps, n_params); got shape {sz.shape}"
            )
        self._samples_z = sz
        self.subsample_n = subsample_n
        self.seed = int(seed)

    # -- shape / raw ---------------------------------------------------------

    @property
    def n_params(self) -> int:
        return int(self._samples_z.shape[-1])

    @property
    def n_chains(self) -> int:
        return int(self._samples_z.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self._samples_z.shape[1])

    @property
    def samples_z(self) -> np.ndarray:
        """Full chains, shape ``(n_chains, n_steps, n_params)``."""
        return self._samples_z

    @cached_property
    def _all_flat_z(self) -> np.ndarray:
        return self._samples_z.reshape(-1, self.n_params)

    @cached_property
    def flat_z(self) -> np.ndarray:
        """Chain-flattened, possibly subsampled, samples for plotting."""
        flat = self._all_flat_z
        n = self.subsample_n
        if n is None or flat.shape[0] <= n:
            return flat
        rng = np.random.default_rng(self.seed)
        return flat[rng.choice(flat.shape[0], size=n, replace=False)]

    @cached_property
    def flat_x(self):
        """Physical-space view of ``flat_z`` (nested dicts of arrays)."""
        return self.z_to_x(self.flat_z)

    def subsample(self, n: Optional[int]) -> "SamplerPosterior":
        return SamplerPosterior(self.ctx, self._samples_z, subsample_n=n, seed=self.seed)

    # -- summary statistics --------------------------------------------------

    @cached_property
    def median_z(self) -> np.ndarray:
        return np.median(self._all_flat_z, axis=0)

    @cached_property
    def mean_z(self) -> np.ndarray:
        return np.mean(self._all_flat_z, axis=0)

    def quantiles_z(self, q) -> np.ndarray:
        return np.quantile(self._all_flat_z, q, axis=0)

    # -- convergence diagnostics --------------------------------------------

    @cached_property
    def rhat(self) -> np.ndarray:
        """Gelman-Rubin R-hat per parameter (shape ``(n_params,)``)."""
        return self._rhat(self._samples_z)

    @cached_property
    def ess(self) -> np.ndarray:
        """Effective sample size per parameter (shape ``(n_params,)``)."""
        return self._ess(self._samples_z)

    def running_rhat(self, *, schedule=None) -> Tuple[np.ndarray, np.ndarray]:
        """R-hat computed on a prefix of the chains, swept across ``schedule``.

        Returns ``(schedule, rhat)`` where ``rhat[i]`` is the R-hat from
        ``samples_z[:, :schedule[i], :]``.

        With ``schedule=None`` (default), uses a log-spaced grid of ~20
        window-ends from ~n_steps/50 up to n_steps.
        """
        schedule = self._default_schedule() if schedule is None else np.asarray(schedule)
        rh = np.stack([self._rhat(self._samples_z[:, :int(N), :]) for N in schedule])
        return schedule, rh

    def running_ess(self, *, schedule=None) -> Tuple[np.ndarray, np.ndarray]:
        """ESS computed on a prefix of the chains, swept across ``schedule``."""
        schedule = self._default_schedule() if schedule is None else np.asarray(schedule)
        es = np.stack([self._ess(self._samples_z[:, :int(N), :]) for N in schedule])
        return schedule, es

    def _default_schedule(self) -> np.ndarray:
        lo = max(50, self.n_steps // 50)
        return np.unique(np.geomspace(lo, self.n_steps, 20).astype(int))

    @staticmethod
    def _rhat(sz: np.ndarray) -> np.ndarray:
        # tfp wants (num_results, num_chains, ...) so we transpose from our
        # canonical (chains, steps, params) layout.
        sz_t = jnp.transpose(jnp.asarray(sz), (1, 0, 2))
        return np.asarray(tfp.mcmc.potential_scale_reduction(sz_t))

    @staticmethod
    def _ess(sz: np.ndarray) -> np.ndarray:
        sz_t = jnp.transpose(jnp.asarray(sz), (1, 0, 2))
        return np.asarray(tfp.mcmc.effective_sample_size(sz_t))

    # -- representative point ----------------------------------------------

    def _point_z(self, name: str) -> np.ndarray:
        if name == "median":
            return self.median_z
        if name == "mean":
            return self.mean_z
        raise ValueError(f"SamplerPosterior: unknown point name {name!r}; expected "
                         f"'median' or 'mean'.")


# ---------------------------------------------------------------------------
# SurrogatePosterior: SVI / parametric q(z)
# ---------------------------------------------------------------------------


class SurrogatePosterior(Posterior):
    """View over a parametric surrogate posterior ``q(z)`` (e.g. from SVI).

    For most purposes this behaves like a sampler: ``draw(n)`` yields a
    :class:`SamplerPosterior` constructed from ``n`` samples of ``q``. The
    surrogate's analytic mean / covariance are also exposed directly.
    """

    def __init__(self, ctx, qz, *, n_samples: int = DEFAULT_SUBSAMPLE_N, seed: int = 0,
                 loss_hist: Optional[np.ndarray] = None):
        super().__init__(ctx)
        self.qz = qz
        self.n_samples = int(n_samples)
        self.seed = int(seed)
        self.loss_hist = None if loss_hist is None else np.asarray(loss_hist)

    @property
    def n_params(self) -> int:
        return int(self.qz.loc.shape[-1])

    @cached_property
    def median_z(self) -> np.ndarray:
        return np.asarray(self.qz.mean())

    @property
    def mean_z(self) -> np.ndarray:
        return self.median_z

    @cached_property
    def covariance(self) -> np.ndarray:
        return np.asarray(self.qz.covariance())

    def draw(self, n: Optional[int] = None, seed: Optional[int] = None) -> SamplerPosterior:
        """Sample from ``q(z)`` and wrap as a single-chain SamplerPosterior."""
        n = self.n_samples if n is None else int(n)
        seed = self.seed if seed is None else int(seed)
        z = np.asarray(self.qz.sample(n, seed=jax.random.PRNGKey(seed)))
        return SamplerPosterior(self.ctx, z[None, ...], subsample_n=None, seed=seed)

    @cached_property
    def flat_z(self) -> np.ndarray:
        return self.draw().flat_z

    @cached_property
    def flat_x(self):
        return self.z_to_x(self.flat_z)

    def quantiles_z(self, q) -> np.ndarray:
        # No closed form for arbitrary q on a multivariate normal in general;
        # sample once and cache via flat_z.
        return np.quantile(self.flat_z, q, axis=0)

    def _point_z(self, name: str) -> np.ndarray:
        if name in ("median", "mean", "best"):
            return self.median_z
        raise ValueError(f"SurrogatePosterior: unknown point name {name!r}.")


# ---------------------------------------------------------------------------
# PointEstimate: MAP / single best fit
# ---------------------------------------------------------------------------


class PointEstimate(Posterior):
    """View over a single point estimate (e.g. MAP).

    Carries optional loss / chi-squared histories for diagnostic plotting.
    """

    def __init__(self, ctx, z_best, *, lp_hist: Optional[np.ndarray] = None,
                 chisq_hist: Optional[np.ndarray] = None):
        super().__init__(ctx)
        z = np.asarray(z_best)
        if z.ndim > 1:
            z = np.squeeze(z)
        if z.ndim != 1:
            raise ValueError(f"z_best must be 1-D; got shape {z_best.shape}")
        self.z_best = z
        self.lp_hist = None if lp_hist is None else np.asarray(lp_hist)
        self.chisq_hist = None if chisq_hist is None else np.asarray(chisq_hist)

    @property
    def n_params(self) -> int:
        return int(self.z_best.shape[-1])

    @property
    def median_z(self) -> np.ndarray:
        return self.z_best

    @property
    def mean_z(self) -> np.ndarray:
        return self.z_best

    @property
    def best_z(self) -> np.ndarray:
        return self.z_best

    @cached_property
    def x(self):
        """Physical-space parameters at the best point (batched, batch size 1)."""
        return self.z_to_x(self.z_best)

    def _point_z(self, name: str) -> np.ndarray:
        if name in ("best", "median", "mean"):
            return self.z_best
        raise ValueError(f"PointEstimate: unknown point name {name!r}.")
