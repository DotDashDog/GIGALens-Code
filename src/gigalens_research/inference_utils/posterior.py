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

_tfd = tfp.distributions

#: Default cap on the number of (chain-flattened) samples returned by
#: :attr:`SamplerPosterior.flat_z`. Set via ``subsample_n=`` on construction.
DEFAULT_SUBSAMPLE_N = 5000


# ---------------------------------------------------------------------------
# Rank-normalized split convergence diagnostics (Vehtari et al. 2021,
# "Rank-Normalization, Folding, and Localization: An Improved R̂").
#
# Why this and not plain Gelman-Rubin / tfp.effective_sample_size: the sampler
# works in unconstrained (z) space, where a prior bound induces a z->inf
# bijector stretch -> heavy-tailed / non-normal marginals. The classic PSRF
# badly over-reports R̂ there (we measured z-PSRF ~12 where the *physical*
# posterior is essentially one mode; rank-R̂ ~1.7). Rank normalization is
# invariant to any monotone reparameterization, so it neither invents nor
# hides convergence because of the prior's parameterization.
#
# We defer to ArviZ's reference implementation rather than re-derive the
# rank/fold/Geyer machinery here (a hand-rolled ESS is an easy way to fool
# yourself). ArviZ is part of the canonical env (docs/env_setup.md); runtime
# deps are env-managed, not declared in pyproject.
# ---------------------------------------------------------------------------


def _require_arviz():
    try:
        import arviz as az
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Rank-normalized R̂/ESS diagnostics require ArviZ. It is part of the "
            "canonical gigalens env (docs/env_setup.md); install with `pip install arviz`."
        ) from e
    return az


def _az_diag(sz: np.ndarray, fn) -> np.ndarray:
    """Apply an ArviZ diagnostic ``fn`` (e.g. ``az.rhat``) per parameter.

    ArviZ's array API only takes a uni-dimensional ``(chain, draw)`` variable,
    so we wrap our canonical ``(chains, steps, params)`` array as a dataset
    (chain, draw, param) and read the per-param result back.
    """
    az = _require_arviz()
    ds = az.convert_to_dataset(np.asarray(sz))
    var = list(ds.data_vars)[0]
    return np.atleast_1d(np.asarray(fn(ds)[var].values))


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

    @property
    def _scene_model(self):
        """The scene LensModel for this (scene-only) ctx."""
        model = getattr(getattr(self.ctx, "model_seq", None), "scene_model", None)
        if model is None:
            raise TypeError(
                "Posterior requires a scene-backed InferenceContext; the legacy "
                "PhysicalModel path was removed with the old gigalens API.")
        return model

    def _lens_sim(self):
        """Legacy fallback simulator over the whole scene (all light, single PSF).

        Only used when the prob_model does not expose per-dataset ``simulators``
        (non-scene / older prob_models). Scene-backed multi-dataset models render
        through :meth:`_sim_for`, which returns the prob_model's per-dataset
        simulators (each with the correct ``sees`` filter and PSF). The new
        SceneSimulator is batch-flexible, so no ``bs`` is threaded here."""
        if "all" not in self._sim_cache:
            from gigalens.jax.scene_simulator import SceneSimulator
            self._sim_cache["all"] = SceneSimulator(
                self._scene_model, self.ctx.sim_config)
        return self._sim_cache["all"]

    # -- per-dataset accessors (multi-band aware; single-dataset by default) ----

    @property
    def _prob_datasets(self):
        """The prob_model's Dataset list if it is dataset-aware, else ``None``."""
        return getattr(self.ctx.prob_model, "datasets", None)

    def n_datasets(self) -> int:
        """Number of observed datasets/bands this posterior was fit against."""
        ds = self._prob_datasets
        return len(ds) if ds is not None else 1

    def observed_for(self, dataset: int = 0) -> np.ndarray:
        """The observed image for band ``dataset`` (avoids the single-dataset-only
        ``observed_image`` property, which raises for multi-dataset models)."""
        ds = self._prob_datasets
        if ds is not None:
            return np.asarray(ds[dataset].image)
        return np.asarray(self.ctx.prob_model.observed_image)

    def _error_for(self, dataset: int = 0) -> Optional[np.ndarray]:
        ds = self._prob_datasets
        if ds is not None:
            return np.asarray(ds[dataset].error_map)
        return np.asarray(getattr(self.ctx.prob_model, "error_map"))

    def mask_for(self, dataset: int = 0):
        """Per-dataset fit mask (or ``None`` for legacy prob_models without one).

        Returns a boolean array where ``True`` = valid/kept pixel and
        ``False`` = excluded (hot/bad) pixel, matching the convention used by
        :meth:`~gigalens.jax.scene_prob_model.ProbModel._dataset_chi2`.

        Passed to ``lstsq_simulate`` so the plotted model matches the likelihood,
        which solves amplitudes over the masked pixels (see ProbModel._model_image).
        Returns ``None`` for legacy prob_models that have no per-dataset ``mask``
        (treat as all-True, i.e. no masking)."""
        ds = self._prob_datasets
        if ds is not None:
            return ds[dataset].mask
        return None

    # Keep the private alias so internal callers and any external code that
    # used the underscore name continue to work during the transition.
    _mask_for = mask_for

    def _sim_for(self, dataset: int = 0):
        """The simulator for band ``dataset``: the prob_model's per-dataset
        simulator when available (correct ``sees`` + PSF), else the legacy
        whole-scene fallback."""
        sims = getattr(self.ctx.prob_model, "simulators", None)
        if sims is not None:
            return sims[dataset]
        return self._lens_sim()

    def z_to_x(self, z) -> List:
        """Apply the bijector forward to a ``z`` of shape ``(n_params,)`` or
        ``(n, n_params)``. Returns the nested-list-of-dicts structure the
        :class:`~gigalens.jax.simulator.LensSimulator` expects."""
        z = jnp.atleast_2d(jnp.asarray(z))
        return self.ctx.prob_model.bij.forward(list(z.T))

    def x_grouped(self, point: str = "median"):
        """Physical params at ``point`` in the legacy 3-group nested form
        ``{'lens_mass': {...}, 'lens_light': {...}, 'source_light': {...}}`` (keys
        are stringified profile indices), for BOTH legacy and scene-backed ctx.

        The scene bijector returns a flat unique-key dict (``planes/i/role/j/param``),
        so plotting / source-plane consumers that index the old group keys would
        ``KeyError`` on a scene-backed posterior. This funnels them through one place:
        a legacy ctx returns its already-grouped ``x`` unchanged; a scene ctx scatters
        to the structured ``planes`` params and regroups by role — mass -> lens_mass,
        light on a lensed plane -> source_light, other light -> lens_light — in the SAME
        order as :class:`_ScenePhysModelView`, so the regrouped param dicts line up
        index-for-index with ``ctx.phys_model.{lenses,lens_light,source_light}``.
        """
        x = self.z_to_x(self._point_z(point))
        scene = self._scene_model
        params = scene.to_params(dict(x))
        lens_mass, lens_light, source_light = {}, {}, {}
        mi = li = si = 0
        for i, plane in enumerate(scene.planes):
            lensed = any(scene.planes[j].has_mass for j in range(i))
            for j in range(len(plane.mass)):
                lens_mass[str(mi)] = params["planes"][i]["mass"][j]; mi += 1
            for j in range(len(plane.light)):
                if lensed:
                    source_light[str(si)] = params["planes"][i]["light"][j]; si += 1
                else:
                    lens_light[str(li)] = params["planes"][i]["light"][j]; li += 1
        return {"lens_mass": lens_mass, "lens_light": lens_light,
                "source_light": source_light}

    def grouped_free_x(self, x_flat):
        """Regroup the FREE params of a scene bijector output (flat
        ``{unique_key: array}`` dict) into the legacy 3-group nested form, so the
        flatten / latex-label / truth-overlay machinery (corner plots, z-scores)
        applies unchanged and a 3-group truth aligns label-for-label.

        Differs from :meth:`x_grouped` (which scatters ALL params incl. constants for
        rendering): this places ONLY the sampled (free) params — constants are not
        corner columns. A ``shared()`` param is fanned out to each of its sites (matching
        how a 3-group truth carries it per site). Legacy (already-grouped) ``x_flat``
        passes through unchanged. Accepts a single point or a batched bijector output
        (values are carried as-is).
        """
        scene = self._scene_model
        loc = {}  # (plane_i, role, local_j) -> (group_name, global_idx)
        mi = li = si = 0
        for i, plane in enumerate(scene.planes):
            lensed = any(scene.planes[k].has_mass for k in range(i))
            for j in range(len(plane.mass)):
                loc[(i, "mass", j)] = ("lens_mass", mi); mi += 1
            for j in range(len(plane.light)):
                if lensed:
                    loc[(i, "light", j)] = ("source_light", si); si += 1
                else:
                    loc[(i, "light", j)] = ("lens_light", li); li += 1
        groups = {"lens_mass": {}, "lens_light": {}, "source_light": {}}
        for path, ukey in scene._site_to_unique:
            if not path or path[0] != "planes":
                continue  # cosmo etc. are not corner groups
            _, i, role, j, param = path
            info = loc.get((i, role, j))
            if info is None:
                continue
            grp, idx = info
            groups[grp].setdefault(str(idx), {})[param] = x_flat[ukey]
        return groups

    @property
    def is_backward(self) -> bool:
        """True iff the model solves linear amplitudes by least squares (lstsq mode).

        Scene-only (Q4): the scene ProbModel carries the amplitude mode explicitly
        (``mode`` in {"lstsq", "forward"}); ``"lstsq"`` is the backward (linear-amplitude)
        path that recovers amplitudes via ``lstsq_simulate``."""
        return self.ctx.prob_model.mode == "lstsq"

    # -- model-aware rendering ----------------------------------------------

    def simulate(self, point: str = "median", *, dataset: int = 0,
                 return_coeffs: bool = False):
        """Render the predicted PSF-convolved image for one band at a representative point.

        ``dataset`` selects the band (0 by default; single-dataset models ignore it).
        The simulator, observed image, noise map and fit mask are all taken from that
        band, so for a multi-dataset model each band is rendered through its own
        ``sees`` filter and PSF — matching the per-dataset likelihood.

        Auto-selects between :meth:`SceneSimulator.simulate` (forward model) and
        :meth:`SceneSimulator.lstsq_simulate` (backward / linear-amplitude model) via
        :attr:`is_backward`. With ``return_coeffs=True``, also returns the solved linear
        amplitudes (or ``None`` for forward models).
        """
        x = self.z_to_x(self._point_z(point))
        sim = self._sim_for(dataset)
        # Scene-only: the bijector returns the scene unique-key dict; the SceneSimulator
        # consumes the structured (planes/cosmo) params, so scatter via to_params.
        x = self._scene_model.to_params(dict(x))
        # Cast params to the simulator's working dtype. Under jax_enable_x64 a
        # float64 ``z`` (e.g. an MCLMC/bootstrap qz built at x64) yields float64
        # model arrays, which clash with the float32 PSF kernel inside
        # ``lax.conv`` ("requires arguments to have the same dtypes"). The PSF
        # kernel defines the convolution dtype; fall back to the observed image.
        kernel = getattr(sim, "flat_kernel", None)
        target_dtype = (
            kernel.dtype if kernel is not None
            else jnp.asarray(self.observed_for(dataset)).dtype
        )
        x = jax.tree_util.tree_map(
            lambda a: a.astype(target_dtype)
            if jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating) else a,
            x,
        )
        if self.is_backward:
            obs = self.observed_for(dataset)
            err_map = self._error_for(dataset)
            mask = self.mask_for(dataset)
            # lstsq_simulate returns the reconstructed image by default; the
            # linear coeffs are a separate return_coeffs=True call (new gigalens
            # convention). Only solve for coeffs when they're actually requested.
            img = np.asarray(sim.lstsq_simulate(x, obs, err_map, mask))
            if return_coeffs:
                # New gigalens returns coeffs shaped (bs, depth); this single-point
                # API renders at bs=1, so flatten the batch axis to a 1-D (depth,)
                # amplitude vector that downstream consumers (source_plane) index.
                coeffs = np.asarray(
                    sim.lstsq_simulate(x, obs, err_map, mask,
                                       return_coeffs=True)).reshape(-1)
                return img, coeffs
            return img
        img = np.asarray(sim.simulate(x))
        return (img, None) if return_coeffs else img

    def err_map_at(self, predicted, *, dataset: int = 0) -> np.ndarray:
        """Per-pixel noise σ for a given predicted image, using the
        prob_model's noise convention. ``dataset`` selects the band.

        - Backward (lstsq) models: returns band ``dataset``'s ``error_map``
          directly — it was frozen at init from the *observed* image, so the
          ``predicted`` argument is ignored.
        - :class:`ForwardProbModel` (anything that publishes
          ``background_rms`` and ``exp_time``): returns
          ``sqrt(max(predicted, 0) / exp_time + background_rms**2)``.

        Raises ``TypeError`` if the prob_model matches neither pattern.

        Note: the gigalens base ``ProbModel`` now builds ``error_map`` for
        every model and most carry ``background_rms``/``exp_time`` too, so this
        branches on :attr:`is_backward` (the lstsq behaviour) rather than on
        attribute presence.
        """
        pm = self.ctx.prob_model
        # Multi-dataset-aware: read the band's frozen error map directly.
        if self.is_backward and self._prob_datasets is not None:
            return self._error_for(dataset)
        if self.is_backward and hasattr(pm, "error_map"):
            return np.asarray(pm.error_map)
        if hasattr(pm, "background_rms") and hasattr(pm, "exp_time"):
            bg = float(np.asarray(pm.background_rms))
            et = float(np.asarray(pm.exp_time))
            pred = np.asarray(predicted)
            return np.sqrt(np.clip(pred, 0.0, np.inf) / et + bg ** 2)
        if hasattr(pm, "error_map"):
            return np.asarray(pm.error_map)
        raise TypeError(
            f"prob_model {type(pm).__name__} exposes neither error_map nor "
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
        dataset: int = 0,
        plane_index: Optional[int] = None,
    ) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Render the intrinsic source brightness on a fine source-plane grid,
        reconstructed from band ``dataset``.

        Multi-dataset aware: each band sees only the light Components in its
        ``sees`` view and solves its OWN linear amplitudes, so the source is
        reconstructed from band ``dataset``'s coefficients over exactly the
        (lensed) source Components that band sees. Iterating *all* of the model's
        sources against a single band's coefficient vector would mis-align the
        slices (and raise once a source the band doesn't see is reached).
        ``dataset`` selects the band (default 0).

        ``plane_index`` optionally restricts the rendered source light to the
        Components on a single (lensed) plane — used to render one source plane at
        a time, since a model can carry several source planes at different
        redshifts (hence different caustics). ``None`` (default) renders every
        source Component the band sees. The coefficient offset still walks over
        *all* seen light so the per-band amplitude slices stay aligned; only the
        accumulation into the image is filtered.

        The grid is centered on the first seen source Component's
        ``(center_x, center_y)`` (or ``center`` if supplied) and spans
        ``fov_arcsec`` on a side. No lensing, no PSF.

        For backward (lstsq) models, each Component's basis stack is contracted
        against the matching slice of band ``dataset``'s solved coefficients
        (single-component Sersic -> 1 basis function; Shapelets ->
        ``(n_max+1)(n_max+2)/2``). For forward models, each Component's light is
        rendered directly with its modeled amplitude.

        Returns
        -------
        img : ndarray, shape ``(grid_pix, grid_pix)``
        extent : tuple ``(x_min, x_max, y_min, y_max)`` for ``ax.imshow``.
        """
        sim = self._sim_for(dataset)
        # Structured (planes/cosmo) params at the point -- the same layout the simulator
        # consumes; each leaf carries a singleton batch axis.
        params = self._scene_model.to_params(dict(self.z_to_x(self._point_z(point))))
        # Lensed source Components, by identity (lens-plane light is excluded).
        source_ids = {id(c) for c in self._scene_model.source_plane_light()}
        # The band's seen light in the simulator's basis/coefficient order -- this is the
        # order lstsq_simulate concatenates (and solves) amplitudes in, so it is the
        # authoritative layout for slicing the per-band coefficient vector.
        seen_light = sim._light  # list of (plane_idx, light_idx, Component, depth)

        if center is None:
            cx = cy = 0.0
            for i, j, comp, _ in seen_light:
                if id(comp) in source_ids and (plane_index is None or i == plane_index):
                    lp = params["planes"][i]["light"][j]
                    cx = float(np.squeeze(np.asarray(lp.get("center_x", 0.0))))
                    cy = float(np.squeeze(np.asarray(lp.get("center_y", 0.0))))
                    break
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

        img = jnp.zeros((grid_pix, grid_pix))
        if self.is_backward:
            # lstsq path: walk the band's seen light in coefficient order, contracting
            # each source Component's basis stack with its slice of THIS band's coeffs.
            # ``offset`` advances over every seen Component (lens light included) to stay
            # aligned with the coefficient vector; only source Components are summed.
            _, coeffs = self.simulate(point=point, dataset=dataset, return_coeffs=True)
            coeffs = np.atleast_1d(np.asarray(coeffs))
            offset = 0
            for i, j, comp, depth in seen_light:
                if id(comp) in source_ids and (plane_index is None or i == plane_index):
                    lp = params["planes"][i]["light"][j]
                    stack = comp.profile.light(X, Y, **lp)  # (depth, h, w, 1)
                    c = jnp.asarray(coeffs[offset:offset + depth])
                    img = img + jnp.squeeze(jnp.tensordot(c, stack, axes=([0], [0])))
                offset += depth
            # The lstsq design matrix omits the simulator's pixel-area
            # conversion_factor (= det(transform_pix2angle) = delta_pix^2); the solved
            # coefficients absorb it, so divide it back out to express the source as a
            # surface brightness (matching the forward path and intrinsic-SB truths).
            img = img / sim.conversion_factor
        else:
            # Forward path: each Component's light() returns (h, w, 1) with its modeled
            # amplitude baked in; sum the band's seen sources directly.
            for i, j, comp, depth in seen_light:
                if id(comp) in source_ids and (plane_index is None or i == plane_index):
                    lp = params["planes"][i]["light"][j]
                    img = img + jnp.squeeze(comp.profile.light(X, Y, **lp))

        extent = (cx - half, cx + half, cy - half, cy + half)
        return np.asarray(img), extent

    def _deflection_ratio_at(self, plane_index: int, params) -> float:
        """The deflection ratio of (lensed) plane ``plane_index`` at a structured
        params dict, mirroring :meth:`SceneSimulator._trace_deflection_ratio`.

        Either read directly from ``deflection_ratio`` geometry (no cosmology) or
        derived from the cosmology as ``deflection_ratio(z_source)`` (the D_ls/D_s
        ratio normalized to the cosmology's ``z_source_ref``). Raises if the plane
        carries neither — it is then not a well-defined source plane.
        """
        geom = params["planes"][plane_index].get("geometry", {})
        if "deflection_ratio" in geom:
            return float(np.squeeze(np.asarray(geom["deflection_ratio"])))
        cosmo = self._scene_model.cosmo
        if cosmo is not None and "redshift" in geom:
            dr = cosmo.profile.deflection_ratio(geom["redshift"], **params["cosmo"])
            return float(np.squeeze(np.asarray(dr)))
        raise ValueError(
            f"plane {plane_index} has no deflection_ratio and no cosmology+redshift "
            "to derive one from; it is not a well-defined source plane.")

    def source_plane_views(self, point: str = "median"):
        """Enumerate the distinct source planes this posterior can reconstruct.

        Returns a list of ``(dataset, plane_index, deflection_ratio)`` tuples, one
        per distinct lensed plane that carries source light, in plane-index order.
        ``dataset`` is the first band whose ``sees`` view includes a source on that
        plane (the band whose solved coefficients reconstruct it).

        A model can carry several source planes at different redshifts — each with
        its own deflection ratio and therefore its own caustic/critical curve — so
        this is what the source-plane report iterates to render one panel per plane.
        Deduplicated by plane index: a plane seen by more than one band is rendered
        once, from the first band that sees it.
        """
        params = self._scene_model.to_params(dict(self.z_to_x(self._point_z(point))))
        source_ids = {id(c) for c in self._scene_model.source_plane_light()}
        views = []
        seen_planes = set()
        for d in range(self.n_datasets()):
            sim = self._sim_for(d)
            for i, j, comp, depth in sim._light:
                if id(comp) in source_ids and i not in seen_planes:
                    seen_planes.add(i)
                    views.append((d, int(i), self._deflection_ratio_at(i, params)))
        views.sort(key=lambda t: t[1])
        return views

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
        """Physical-space view of ``flat_z`` (nested dicts of arrays), in the legacy
        3-group label space (scene-backed samples are regrouped so corner plots and
        truth overlays align; see :meth:`grouped_free_x`)."""
        return self.grouped_free_x(self.z_to_x(self.flat_z))

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
        """Rank-normalized split-R-hat per parameter (shape ``(n_params,)``)."""
        return self._rhat(self._samples_z)

    @cached_property
    def ess(self) -> np.ndarray:
        """Rank-normalized bulk-ESS per parameter (shape ``(n_params,)``)."""
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
        """Rank-normalized split-R-hat (Vehtari et al. 2021), per parameter.

        Reported as ``max(rank, folded)``: the *rank* term catches location
        non-convergence; the *folded* term (ranks of ``|theta - median|``)
        catches scale/variance non-convergence. ArviZ expects ``(chain, draw,
        ...)``, which is our canonical layout, so the trailing param axis maps
        straight through. Standard (sqrt-form) R̂ -> thresholds 1.01/1.1 apply.
        """
        az = _require_arviz()
        rank = _az_diag(sz, lambda ds: az.rhat(ds, method="rank"))
        folded = _az_diag(sz, lambda ds: az.rhat(ds, method="folded"))
        return np.maximum(rank, folded)

    @staticmethod
    def _ess(sz: np.ndarray) -> np.ndarray:
        """Rank-normalized bulk-ESS (Vehtari et al. 2021), per parameter."""
        az = _require_arviz()
        return _az_diag(sz, lambda ds: az.ess(ds, method="bulk"))

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
        return self.grouped_free_x(self.z_to_x(self.flat_z))

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
