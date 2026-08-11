"""Compound multi-panel reports.

Two convenience builders, both thin orchestrators over the primitive plotters
in this package. They produce matplotlib ``Figure`` objects; callers save
with ``fig.savefig(...)``.

- :class:`PosteriorReport` — single-posterior panels (image diagnostics,
  convergence, source plane, corner plot).
- :class:`PipelineReport` — multi-stage overlays (compound corner, loss
  histories side by side, image comparison across stages).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from ..param_index import ParamSite
from .convergence import (
    plot_chain_traces,
    plot_loss_history,
    plot_running_ess,
    plot_running_rhat,
)
from .corner import plot_corner, plot_corner_overlay
from .image import normalized_residual, plot_image, plot_residual_histogram
from .source_plane import plot_critical_curves, plot_source_plane
from .truth import plot_source_comparison, plot_z_scores


# ---------------------------------------------------------------------------
# PosteriorReport
# ---------------------------------------------------------------------------


def _per_view(spec, plane: int, dataset: int):
    """Resolve a per-view framing argument for one ``(dataset, plane)`` row.

    ``spec`` is either a plain value applied to every row, or a dict keyed by plane
    index or by ``(dataset, plane)``; a row the dict does not mention resolves to
    ``None``, which means "let the plotter auto-frame this one". The ``(dataset,
    plane)`` key wins over the bare plane key when both are present.

    A tuple ``spec`` is treated as a plain value, not a mapping, so ``center=(x, y)``
    keeps working as the apply-to-all form.
    """
    if not isinstance(spec, dict):
        return spec
    if (dataset, plane) in spec:
        return spec[(dataset, plane)]
    return spec.get(plane)


class PosteriorReport:
    """A bundle of plotting methods for a single :class:`Posterior`.

    Every method returns the figure it built and does not call ``plt.show``;
    callers control saving and display.

    Optional truth inputs enable the simulated-system diagnostics (truth
    overlays on corners, z-score bar plots, source-plane comparisons). They
    are silently ignored on methods that don't need them, and methods that
    *do* need them raise a clear error if they were omitted.

    Parameters
    ----------
    posterior : Posterior
    prefix : str
        Optional prefix injected into figure titles (e.g. a system name).
    truth_x : nested params, optional
        Truth in physical-space, structured like ``posterior.median_x``.
        Used by :meth:`corner` (as the default ``truth=``), by
        :meth:`z_score_panel`, and by :meth:`full_report`.
    truth_source_image : ndarray, optional
        Square 2-D image of the *intrinsic* (un-lensed, un-PSF'd) source on
        a known grid. Used by :meth:`source_comparison_panel`. Must come
        with ``truth_source_extent``.
    truth_source_extent : (xmin, xmax, ymin, ymax), optional
        Field-of-view of the truth source image, in arcseconds, matching
        the convention used by ``ax.imshow``.
    truth_source_fn : callable, optional
        ``f(X, Y) -> image`` that produces the truth source on arbitrary
        source-plane coordinates. Mutually exclusive with
        ``truth_source_image``. Use this when the truth source is a continuous
        function (e.g. an interpolated image like
        ``ImageBasedLight``); ergonomically:
        ``truth_source_fn=lambda X, Y: vela_light.light(X, Y, **truth_x[2][0])``,
        or via :func:`truth_source_from_light_model`.
    """

    def __init__(
        self,
        posterior,
        *,
        prefix: str = "",
        truth_x=None,
        truth_source_image: Optional[np.ndarray] = None,
        truth_source_extent: Optional[tuple] = None,
        truth_source_fn=None,
    ):
        self.posterior = posterior
        self.prefix = prefix
        self.truth_x = truth_x
        self.truth_source_image = (
            None if truth_source_image is None else np.asarray(truth_source_image)
        )
        self.truth_source_extent = truth_source_extent
        self.truth_source_fn = truth_source_fn
        if (self.truth_source_image is None) != (self.truth_source_extent is None):
            raise ValueError(
                "truth_source_image and truth_source_extent must be supplied together."
            )
        if self.truth_source_image is not None and self.truth_source_fn is not None:
            raise ValueError(
                "Pass truth_source_image OR truth_source_fn, not both."
            )

    # -- image-plane panel ---------------------------------------------------

    def _finalize(self, fig: Figure) -> Figure:
        """Lay out ``fig`` with the prefix as a single figure-level title.

        The prefix (e.g. a system/sweep id) goes in one ``suptitle`` instead of
        being repeated into every subplot title — repeating a long prefix across
        a 4-up row makes the titles (and the χ²) overlap and clip. ``rect``
        leaves headroom so the suptitle never overlaps the top row of axes.
        """
        if self.prefix and self.prefix.strip():
            fig.suptitle(self.prefix.strip(), fontsize=11)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
        else:
            fig.tight_layout()
        return fig

    def image_panel(
        self,
        observed: Optional[np.ndarray] = None,
        *,
        point: str = "median",
        with_critical_curves: bool = False,
        scale: str = "asinh",
        linear_width: Optional[float] = None,
        log_vmin: float = 1e-3,
    ) -> Figure:
        """Per-band image panel: one row of (observed, model, normalized residual,
        residual histogram) for **each** dataset the posterior was fit against.

        A single-dataset model gives the classic 1×4 figure; an N-dataset model gives
        an N×4 grid, one row per band, each rendered through its own ``sees`` filter,
        PSF, noise map and mask (see :meth:`Posterior.simulate`).

        ``observed`` overrides the observed image for *every* row (e.g. a noise-free
        truth image); leave it ``None`` to use each band's own observed image. Noise σ
        comes from :meth:`Posterior.err_map_at`, so this works for both
        :class:`ForwardProbModel` and :class:`BackwardProbModel`.

        ``with_critical_curves`` overlays the **critical** curves (image/lens plane
        only — caustics belong on the source plane, see :meth:`source_panel`) on the
        observed and model panels, one per source plane the model carries (each at
        its own deflection ratio).
        """
        n_ds = self.posterior.n_datasets()
        views = (self.posterior.source_plane_views(point=point)
                 if with_critical_curves else [])
        obs_override = None if observed is None else np.asarray(observed)

        fig, axs = plt.subplots(n_ds, 4, figsize=(13, 3.2 * n_ds), squeeze=False)
        for k, row_ax in enumerate(axs):
            # Per band: rows may come from grids of different size/scale.
            extent = self._band_extent(k)
            observed_k = (obs_override if obs_override is not None
                          else np.asarray(self.posterior.observed_for(k)))
            predicted = np.asarray(self.posterior.simulate(point=point, dataset=k))
            err_map = self.posterior.err_map_at(predicted, dataset=k)
            residual = normalized_residual(observed_k, predicted, err_map)

            # Apply per-band fit mask so chi², color scales, and residual panels
            # are not dominated by excluded (hot/bad) pixels.
            m = self.posterior.mask_for(k)
            mbool = None if m is None else np.asarray(m).astype(bool)

            resid_fit = residual if mbool is None else residual[mbool]
            chisq = float(np.sum(resid_fit ** 2))
            n_pix = residual.size if mbool is None else int(mbool.sum())
            ndof = max(n_pix - self.posterior.n_params, 1)
            red_chisq = chisq / ndof

            # NaN-mask display arrays so excluded pixels appear blank (not hot).
            # plot_image derives its vmax from finite values only.
            obs_disp = observed_k if mbool is None else np.where(mbool, observed_k, np.nan)
            res_disp = residual if mbool is None else np.where(mbool, residual, np.nan)

            band = f" band {k}" if n_ds > 1 else ""
            plot_image(row_ax[0], obs_disp, extent=extent, title=f"Observed{band}",
                       scale=scale, linear_width=linear_width, log_vmin=log_vmin)
            plot_image(row_ax[1], predicted, extent=extent,
                       title=f"Model{band} ({point}, χ²/ν={red_chisq:.3f})",
                       scale=scale, linear_width=linear_width, log_vmin=log_vmin)
            plot_image(row_ax[2], res_disp, extent=extent,
                       title=f"Normalized residual{band}", residual=True)
            # Histogram gets a 1-D finite array (NaNs break the Gaussianity fit).
            plot_residual_histogram(row_ax[3], resid_fit.ravel(), title="Gaussianity test")
            for _d, plane_i, dr in views:
                for col in (0, 1):  # observed + model (both image plane)
                    plot_critical_curves(row_ax[col], self.posterior, point=point,
                                         plane=plane_i, deflection_ratio=dr,
                                         label=f"crit z-plane {plane_i}")
        return self._finalize(fig)

    # -- convergence panel ---------------------------------------------------

    def convergence_panel(
        self, *, trace_param: Optional[int] = None, n_worst: int = 3,
    ) -> Figure:
        """1×3 panel: chain traces, running R-hat, running ESS. Only applicable
        to :class:`SamplerPosterior`.

        The R-hat and ESS panels are annotated with the ``n_worst`` worst
        parameters — by R-hat and by bulk-ESS respectively — using the labeled
        :attr:`~SamplerPosterior.convergence` report, so each panel names the
        parameter driving it. The trace panel defaults to the single
        worst-R-hat **z-column** (the space those diagnostics live in).

        ``trace_param`` overrides the traced parameter with an explicit
        **x-space** (corner-plot) index instead; ``None`` keeps the
        worst-R-hat default.
        """
        if not hasattr(self.posterior, "samples_z"):
            raise TypeError(
                f"convergence_panel needs a SamplerPosterior; got "
                f"{type(self.posterior).__name__}."
            )
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.5))
        if trace_param is None:
            # nanargmax would raise on an all-NaN report; map NaN->-inf first.
            rhat = np.asarray(self.posterior.convergence.rhat, dtype=float)
            worst_z = int(np.argmax(np.where(np.isnan(rhat), -np.inf, rhat)))
            plot_chain_traces(axs[0], self.posterior, z_param=worst_z)
        else:
            plot_chain_traces(axs[0], self.posterior, param=trace_param)
        plot_running_rhat(axs[1], self.posterior, aggregate="max",
                          annotate_worst=n_worst)
        plot_running_ess(axs[2], self.posterior, aggregate="min",
                         annotate_worst=n_worst)
        return self._finalize(fig)

    # -- source-plane panel --------------------------------------------------

    def model_panel(
        self,
        *,
        point: str = "median",
        with_critical_curves: bool = True,
        scale: str = "asinh",
        linear_width: Optional[float] = None,
        log_vmin: float = 1e-2,
    ) -> Figure:
        """One row per observation: the rendered model image, with critical curves.

        The forward counterpart of :meth:`image_panel`, which pairs model against
        observed and residual. This one needs **no observed data**, so it is what a
        scene being built (rather than fit) can show — though it is equally useful on a
        fitted posterior when you want the model alone.

        Critical curves are drawn for the source plane(s) that band sees, each at its
        own deflection ratio, since the Jacobian ``I - r * dalpha/dtheta`` depends on
        ``r`` and every source redshift therefore has its own curve.
        """
        n_ds = self.posterior.n_datasets()
        views = (self.posterior.source_plane_views(point=point)
                 if with_critical_curves else [])
        by_band: Dict[int, List[Tuple[int, float]]] = {}
        for d, plane_i, dr in views:
            by_band.setdefault(d, []).append((plane_i, dr))

        fig, axs = plt.subplots(n_ds, 1, figsize=(5.5, 4.6 * n_ds), squeeze=False)
        for k in range(n_ds):
            ax = axs[k][0]
            predicted = np.asarray(self.posterior.simulate(point=point, dataset=k))
            band = f" band {k}" if n_ds > 1 else ""
            plot_image(ax, predicted, extent=self._band_extent(k),
                       title=f"Model{band} ({self.posterior.point_label(point)})",
                       scale=scale,
                       linear_width=linear_width, log_vmin=log_vmin)
            for plane_i, dr in by_band.get(k, []):
                plot_critical_curves(ax, self.posterior, point=point,
                                     plane=plane_i, deflection_ratio=dr)
        return self._finalize(fig)

    def _band_extent(self, dataset: int) -> Tuple[float, float, float, float]:
        """``imshow`` extent in arcsec for one band's grid.

        Read from that band's OWN simulator config where available, falling back to
        the context-wide one. Bands can differ in ``num_pix``/``delta_pix`` (different
        instruments, or IFU cutouts trimmed per wavelength), and a figure-wide extent
        would place the image on the wrong axes scale — silently, and with a critical
        curve drawn over it in true sky coordinates, so the mismatch reads as a
        physical offset rather than a plotting bug.
        """
        sc = self.posterior._sim_config_for(dataset)
        nx = sc.num_pix if isinstance(sc.num_pix, int) else sc.num_pix[1]
        ny = sc.num_pix if isinstance(sc.num_pix, int) else sc.num_pix[0]
        return (-nx / 2 * sc.delta_pix, nx / 2 * sc.delta_pix,
                -ny / 2 * sc.delta_pix, ny / 2 * sc.delta_pix)

    def source_panel(
        self,
        *,
        point: str = "median",
        grid_pix: int = 400,
        fov_arcsec: Optional[Any] = None,
        center: Optional[Any] = None,
        observed: Optional[np.ndarray] = None,
        with_observed: bool = True,
        scale: str = "asinh",
        linear_width: Optional[float] = None,
        log_vmin: float = 1e-2,
        frame_frac: float = 0.99,
    ) -> Figure:
        """One row **per source plane**: the intrinsic source-plane image (with that
        plane's *caustic*) and, optionally, the observed image of the band that
        reconstructs it (with that plane's *critical* curve).

        A model can carry several source planes at different redshifts; each has its
        own deflection ratio and therefore its own caustic/critical pair. The rows
        come from :meth:`Posterior.source_plane_views` — one per distinct lensed
        plane carrying source light. Each row renders only that plane's source light
        (``plane_index=``) from the band that sees it (``dataset=``).

        Plane convention (no mixing): the left/source panel carries only the
        **caustic**; the right/observed panel carries only the **critical** curve.

        ``observed`` overrides the observed image for every row (e.g. a noise-free
        truth); leave it ``None`` to use each band's own image. ``with_observed=False``
        drops the right column — and needs no observed data at all, which is the
        forward-mode path (see
        :class:`~gigalens_research.inference_utils.posterior.FixedParams`). Masked
        pixels (band fit mask False) are shown blank so hot/bad pixels do not hijack
        the color scale.

        Framing (``fov_arcsec``, ``center``) is **per view**: source planes sit at
        different places and have wildly different angular sizes, so one global window
        either crops the compact ones or buries the extended ones. Pass a scalar to
        apply it everywhere, or a dict keyed by plane index (``{2: 4.0, 7: 12.0}``) or
        by ``(dataset, plane)`` to set individual rows; unlisted rows fall back to the
        auto-framing, which sizes each window to the plane's own light (see
        :meth:`Posterior.source_plane`). ``frame_frac`` tunes how much absolute flux
        that window encloses; ``fov_arcsec="full"`` opts a row back out to the whole
        cutout.
        """
        views = self.posterior.source_plane_views(point=point)
        if not views:
            raise ValueError(
                "no source planes to plot: this posterior's model carries no lensed "
                "source light.")
        obs_override = None if observed is None else np.asarray(observed)

        ncols = 2 if with_observed else 1
        n = len(views)
        fig, axs = plt.subplots(n, ncols, figsize=(5 * ncols, 4 * n), squeeze=False)
        for row, (d, plane_i, dr) in enumerate(views):
            band = f", band {d}" if self.posterior.n_datasets() > 1 else ""
            plot_source_plane(
                axs[row][0], self.posterior, point=point,
                grid_pix=grid_pix, fov_arcsec=_per_view(fov_arcsec, plane_i, d),
                center=_per_view(center, plane_i, d),
                dataset=d, plane_index=plane_i, deflection_ratio=dr,
                scale=scale, linear_width=linear_width, log_vmin=log_vmin,
                frame_frac=frame_frac,
                title=f"Source plane {plane_i} (dr={dr:.3f}{band})",
            )
            if not with_observed:
                continue
            observed_k = (obs_override if obs_override is not None
                          else np.asarray(self.posterior.observed_for(d)))
            m = self.posterior.mask_for(d)
            mbool = None if m is None else np.asarray(m).astype(bool)
            obs_disp = (observed_k if mbool is None
                        else np.where(mbool, observed_k, np.nan))
            # Extent per BAND, not once per figure: bands may differ in num_pix /
            # delta_pix (different instruments, different IFU cutouts), and a shared
            # extent would silently mis-place the critical curve overlaid below it.
            plot_image(axs[row][1], obs_disp, extent=self._band_extent(d),
                       title=f"Observed{band}", scale=scale,
                       linear_width=linear_width, log_vmin=log_vmin)
            plot_critical_curves(axs[row][1], self.posterior, point=point,
                                 plane=plane_i, deflection_ratio=dr)
        return self._finalize(fig)

    def scene_panel(
        self,
        *,
        point: str = "median",
        grid_pix: int = 400,
        fov_arcsec: Optional[Any] = None,
        center: Optional[Any] = None,
        with_curves: bool = True,
        with_image_border: bool = False,
        scale: str = "asinh",
        linear_width: Optional[float] = None,
        log_vmin: float = 1e-2,
        frame_frac: float = 0.99,
    ) -> Figure:
        """One row per source plane, image plane and source plane side by side.

        The single-figure view of a scene: each row pairs the **lensed model image**
        of the band that sees a plane (with that plane's critical curve) against the
        **intrinsic source** on that plane (with its caustic). It answers the question
        you actually ask while iterating on a mock — "what did this source turn
        into?" — without leaving you to match up two separate figures by index.

        Rows come from :meth:`Posterior.source_plane_views`, so this is exactly one
        row per source plane. When bands and planes are 1:1 (the IFU case: one cutout
        per source redshift) that is also one row per observation. When they are not,
        a band seeing several planes repeats down the rows it contributes to — its
        image is rendered once and reused — and a band with no lensed source light at
        all still gets a row, with the source cell blanked, so no observation silently
        vanishes from the figure.

        **The two columns are not on the same scale**, and neither axis pretends
        otherwise: the image column spans that band's whole cutout, while the source
        column is auto-framed to its own light (typically far smaller). Both carry
        arcsec ticks so the difference is visible rather than assumed. ``fov_arcsec``
        / ``center`` / ``frame_frac`` control only the source column and are per view
        (see :meth:`source_panel`); pass ``fov_arcsec="full"`` to put a row's source
        panel back on the cutout window.

        Needs no observed data — for the data-bearing counterparts see
        :meth:`image_panel` (observed/model/residual) and
        :meth:`source_panel` (source/observed).
        """
        views = self.posterior.source_plane_views(point=point)
        n_ds = self.posterior.n_datasets()
        label = self.posterior.point_label(point)
        covered = {d for d, _, _ in views}
        rows: List[Tuple[int, Optional[int], Optional[float]]] = list(views)
        rows += [(d, None, None) for d in range(n_ds) if d not in covered]
        if not rows:
            raise ValueError("nothing to plot: no observations and no source planes.")

        # A band seen by several planes renders once, not once per row.
        images: Dict[int, np.ndarray] = {}
        fig, axs = plt.subplots(len(rows), 2, figsize=(11, 4.4 * len(rows)),
                                squeeze=False)
        for r, (d, plane_i, dr) in enumerate(rows):
            if d not in images:
                images[d] = np.asarray(self.posterior.simulate(point=point, dataset=d))
            band = f" band {d}" if n_ds > 1 else ""
            plane_lbl = "" if plane_i is None else f", plane {plane_i}"
            plot_image(axs[r][0], images[d], extent=self._band_extent(d),
                       title=f"Model{band}{plane_lbl} ({label})", scale=scale,
                       linear_width=linear_width, log_vmin=log_vmin,
                       remove_axis=False)
            axs[r][0].set_xlabel("x [arcsec]")
            axs[r][0].set_ylabel("y [arcsec]")
            if plane_i is None:
                axs[r][1].set_axis_off()
                axs[r][1].set_title(f"(no lensed source light in band {d})")
                continue
            if with_curves:
                plot_critical_curves(axs[r][0], self.posterior, point=point,
                                     plane=plane_i, deflection_ratio=dr)
            plot_source_plane(
                axs[r][1], self.posterior, point=point, grid_pix=grid_pix,
                fov_arcsec=_per_view(fov_arcsec, plane_i, d),
                center=_per_view(center, plane_i, d),
                dataset=d, plane_index=plane_i, deflection_ratio=dr,
                with_caustics=with_curves, with_image_border=with_image_border,
                scale=scale, linear_width=linear_width, log_vmin=log_vmin,
                frame_frac=frame_frac,
                title=f"Source plane {plane_i} (dr={dr:.3f})",
            )
        return self._finalize(fig)

    # -- corner --------------------------------------------------------------

    def corner(
        self,
        *,
        truth=None,
        overplots: Optional[Dict[str, Any]] = None,
        kind: Any = None,
        plane: Any = None,
        component: Any = None,
        select: Optional[Callable[[ParamSite], bool]] = None,
        plot_params: Optional[List[str]] = None,
        latex: bool = True,
    ) -> Figure:
        """Corner plot of this posterior. ``truth`` defaults to the
        ``truth_x`` supplied at init (if any); pass it explicitly to override
        or pass ``truth=False`` to skip the overlay even when ``truth_x`` was
        set.

        ``kind``/``plane``/``component``/``select`` narrow which parameters are
        drawn, and ``plot_params`` names columns explicitly; both work exactly as
        in :func:`~gigalens_research.plotting.plot_corner`.
        """
        if truth is None:
            truth = self.truth_x
        elif truth is False:
            truth = None
        return plot_corner(
            self.posterior,
            truth=truth, overplots=overplots,
            kind=kind, plane=plane, component=component, select=select,
            plot_params=plot_params, latex=latex,
        )

    # -- truth-aware panels --------------------------------------------------

    def z_score_panel(
        self,
        *,
        kind: Optional[str] = "mass",
        truth_x=None,
        threshold: float = 2.0,
        sort_by_abs: bool = False,
    ) -> Figure:
        """Bar plot of per-parameter z-scores against truth.

        Requires ``truth_x`` to have been supplied at init or passed in here.
        ``kind`` is one of ``'mass'`` (default), ``'cosmology'``,
        ``'geometry'``, ``'light'``, ``'all'``, or ``None``.
        """
        truth_x = truth_x if truth_x is not None else self.truth_x
        if truth_x is None:
            raise ValueError(
                "z_score_panel needs truth_x; pass it to PosteriorReport(...) "
                "or to z_score_panel(truth_x=...)."
            )
        fig, ax = plt.subplots(figsize=(max(6, 0.45 * self.posterior.n_params), 3.5))
        plot_z_scores(ax, self.posterior, truth_x,
                      kind=kind, threshold=threshold, sort_by_abs=sort_by_abs)
        return self._finalize(fig)

    def source_comparison_panel(
        self,
        *,
        truth_source_image: Optional[np.ndarray] = None,
        truth_source_extent: Optional[tuple] = None,
        truth_source_fn=None,
        point: str = "median",
        scale: str = "asinh",
        linear_width: Optional[float] = None,
        log_vmin: float = 1e-2,
        grid_pix: Optional[int] = None,
        fov_arcsec: Optional[float] = None,
        center: Optional[Tuple[float, float]] = None,
    ) -> Figure:
        """1×3 panel: truth source, recovered source, residual.

        The truth source can be supplied as a pre-rendered ``(image, extent)``
        pair or as a callable ``f(X, Y) -> image``. Both forms accept the
        equivalent inputs from ``PosteriorReport(...)`` as defaults. The
        recovered source is rendered on a matching grid via
        :meth:`Posterior.source_plane`.

        With a callable truth, ``grid_pix``/``fov_arcsec``/``center`` set the
        source-plane grid; they default to the recovered source's defaults
        (400 pixels, the sim grid's FoV, centered on the first source-light
        component).
        """
        # Resolve which mode the caller wants, preferring explicit args over
        # init-time defaults.
        img = truth_source_image if truth_source_image is not None else self.truth_source_image
        ext = truth_source_extent if truth_source_extent is not None else self.truth_source_extent
        fn = truth_source_fn if truth_source_fn is not None else self.truth_source_fn
        if fn is not None and (img is not None or ext is not None) \
                and (truth_source_image is None and truth_source_extent is None):
            # Init had both? __init__ would already have rejected. So this
            # branch only fires when call-time passes fn while init had img;
            # explicit fn wins.
            img = ext = None
        if (img is None or ext is None) and fn is None:
            raise ValueError(
                "source_comparison_panel needs a truth source: either "
                "(truth_source_image + truth_source_extent) or "
                "truth_source_fn. Supply at PosteriorReport init or here."
            )

        fig = plt.figure(figsize=(13, 4))
        truth_source = fn if fn is not None else img
        plot_source_comparison(
            fig, self.posterior, truth_source,
            extent=ext if fn is None else None,
            point=point, scale=scale, linear_width=linear_width,
            log_vmin=log_vmin,
            grid_pix=grid_pix, fov_arcsec=fov_arcsec, center=center,
        )
        return self._finalize(fig)

    # -- full report --------------------------------------------------------

    def full_report(
        self,
        *,
        observed: Optional[np.ndarray] = None,
        truth=None,
        save_dir: Optional[str] = None,
        z_score_kind: Optional[str] = "mass",
    ) -> Dict[str, Figure]:
        """Build all panels and (optionally) save each as a PNG to ``save_dir``.

        ``observed`` defaults to each band's own observed image (via
        :meth:`Posterior.observed_for`). Noise model is read from the prob_model
        in all cases.

        If the report was constructed with truth inputs (``truth_x`` or
        ``truth_source_image``), this also generates a z-score bar plot
        (``z_scores``) and a source-plane comparison (``source_comparison``).
        Use ``z_score_kind`` to switch which parameter class is shown
        (default ``'mass'``).

        Returns a dict ``{name: fig}`` so callers can post-process before
        showing or saving manually.
        """
        import os
        figs: Dict[str, Figure] = {}
        figs["image"] = self.image_panel(observed)
        if hasattr(self.posterior, "samples_z"):
            figs["convergence"] = self.convergence_panel()
        figs["source"] = self.source_panel(observed=observed)
        # Corner needs a sample distribution; a point estimate (MAP) has a single
        # point with no dynamic range, so skip it there (mirrors the convergence guard).
        if hasattr(self.posterior, "flat_z"):
            figs["corner"] = self.corner(truth=truth)

        # Truth-aware panels: included automatically if the inputs are there.
        if (truth if truth is not None else self.truth_x) is not None \
                and hasattr(self.posterior, "quantiles_z"):
            figs["z_scores"] = self.z_score_panel(
                truth_x=truth, kind=z_score_kind,
            )
        if self.truth_source_image is not None or self.truth_source_fn is not None:
            figs["source_comparison"] = self.source_comparison_panel()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            for name, fig in figs.items():
                fig.savefig(os.path.join(save_dir, f"{name}.png"),
                            bbox_inches="tight", dpi=150)
        return figs


# ---------------------------------------------------------------------------
# PipelineReport
# ---------------------------------------------------------------------------


class PipelineReport:
    """A bundle of multi-stage plotting methods.

    Construct directly from a :class:`Pipeline` *after* it has run, or via
    :meth:`from_disk` to load saved stage results without an active pipeline.
    """

    def __init__(self, pipeline=None, *, ctx=None, stages: Optional[Dict[str, Any]] = None):
        # Source for debug diagnostics: either the live pipeline or an out_dir
        # (set by from_disk). Resolved lazily in :meth:`diagnostics`.
        self._pipeline = pipeline
        self._out_dir: Optional[str] = None
        if pipeline is not None:
            self.ctx = pipeline.ctx
            self.stages = {}
            for s in pipeline.stages:
                if s.instance_name not in pipeline.results:
                    continue
                try:
                    self.stages[s.instance_name] = pipeline.posterior(s.instance_name)
                except TypeError:
                    # Stage has no posterior view (e.g. BridgeStage); skip.
                    continue
        else:
            if ctx is None or stages is None:
                raise ValueError("Provide either a pipeline or both ctx and stages.")
            self.ctx = ctx
            self.stages = dict(stages)

    @classmethod
    def from_disk(cls, out_dir: str, ctx, *, stage_names: Optional[Iterable[str]] = None) -> "PipelineReport":
        """Load every (or a selected subset of) stage's posterior from disk."""
        from ..inference_utils.pipeline import posterior_from_disk
        import os
        if stage_names is None:
            stage_names = [n for n in os.listdir(out_dir)
                           if os.path.isdir(os.path.join(out_dir, n))
                           and not n.endswith(".stale") and ".stale-" not in n]
        stages: Dict[str, Any] = {}
        for name in stage_names:
            try:
                stages[name] = posterior_from_disk(out_dir, name, ctx)
            except TypeError:
                # stage with no posterior view (e.g. bridge); skip silently
                continue
        report = cls(ctx=ctx, stages=stages)
        report._out_dir = out_dir
        return report

    # -- loss histories ------------------------------------------------------

    def loss_histories(self) -> Figure:
        """Side-by-side loss curves for any stage that carries one. Currently
        knows about :class:`PointEstimate.chisq_hist` and
        :class:`SurrogatePosterior.loss_hist`."""
        plotters = []
        for name, p in self.stages.items():
            if getattr(p, "chisq_hist", None) is not None:
                plotters.append((f"{name} χ²", p.chisq_hist, "χ²", False))
            if getattr(p, "loss_hist", None) is not None:
                plotters.append((f"{name} -ELBO", p.loss_hist, "-ELBO", False))
        if not plotters:
            raise RuntimeError("No stage in this pipeline carries a loss history.")
        fig, axs = plt.subplots(1, len(plotters),
                                figsize=(4.5 * len(plotters), 3.2),
                                squeeze=False)
        for ax, (title, hist, ylabel, log_y) in zip(axs[0], plotters):
            plot_loss_history(ax, hist, title=title, ylabel=ylabel, log_y=log_y)
        fig.tight_layout()
        return fig

    # -- debug diagnostics ---------------------------------------------------

    def diagnostics(self, stage: str, **kwargs) -> Figure:
        """Render a stage's captured debug diagnostics (e.g. an MCLMC tuning
        history). Requires that the stage was run with ``debug=True``.

        Works whether this report wraps a live pipeline or was built via
        :meth:`from_disk`. Extra kwargs are forwarded to the stage's plotter
        (e.g. ``chain=`` for MCLMC). See
        :func:`gigalens_research.plotting.plot_stage_diagnostics`."""
        from .diagnostics import plot_stage_diagnostics

        if self._pipeline is not None:
            diag = self._pipeline.diagnostics(stage)
        elif self._out_dir is not None:
            from ..inference_utils.pipeline import diagnostics_from_disk
            diag = diagnostics_from_disk(self._out_dir, stage, self.ctx)
        else:
            raise RuntimeError(
                "This report has no pipeline or out_dir to read diagnostics "
                "from; build it from a Pipeline or via PipelineReport.from_disk."
            )
        return plot_stage_diagnostics(diag, **kwargs)

    def diagnostics_surrogate_corner(self, stage: str, **kwargs) -> Figure:
        """Corner plot of an MCLMC stage's final draws vs. a Gaussian surrogate
        (mean = sample mean, covariance = final inverse mass matrix), in the
        unconstrained z-space. Requires the stage was run with ``debug=True``.

        Extra kwargs are forwarded to
        :func:`gigalens_research.plotting.plot_mclmc_surrogate_corner` (e.g.
        ``max_samples=``, ``seed=``)."""
        from .diagnostics import plot_mclmc_surrogate_corner

        if self._pipeline is not None:
            diag = self._pipeline.diagnostics(stage)
        elif self._out_dir is not None:
            from ..inference_utils.pipeline import diagnostics_from_disk
            diag = diagnostics_from_disk(self._out_dir, stage, self.ctx)
        else:
            raise RuntimeError(
                "This report has no pipeline or out_dir to read diagnostics "
                "from; build it from a Pipeline or via PipelineReport.from_disk."
            )
        return plot_mclmc_surrogate_corner(diag, **kwargs)

    # -- compound corner -----------------------------------------------------

    def compound_corner(
        self,
        *,
        stages: Optional[Iterable[str]] = None,
        truth=None,
        overplots_stage: Optional[str] = None,
        overplot_label: str = "MAP",
        kind: Any = None,
        plane: Any = None,
        component: Any = None,
        select: Optional[Callable[[ParamSite], bool]] = None,
        plot_params: Optional[List[str]] = None,
        colors: Optional[Dict[str, str]] = None,
        latex: bool = True,
    ) -> Figure:
        """Overlay multiple stages' posteriors on one corner figure.

        ``stages`` defaults to all stages that have sample-like posteriors
        (i.e. excluding pure point estimates). Pass ``overplots_stage`` to mark
        e.g. the MAP optimum as stars on top of the contours.

        ``kind``/``plane``/``component``/``select`` and ``plot_params`` select
        the columns exactly as in
        :func:`~gigalens_research.plotting.plot_corner_overlay`.
        """
        if stages is None:
            stages = [n for n, p in self.stages.items() if hasattr(p, "flat_x")]
        posteriors = {n: self.stages[n] for n in stages}

        overplots = None
        if overplots_stage is not None:
            point = self.stages[overplots_stage]
            if hasattr(point, "x"):
                overplots = {overplot_label: point.x}
            else:
                overplots = {overplot_label: point.z_to_x(point.median_z)}

        return plot_corner_overlay(
            posteriors, kind=kind, plane=plane, component=component,
            select=select, plot_params=plot_params, truth=truth,
            overplots=overplots, colors=colors, latex=latex,
        )

    # -- image comparison ----------------------------------------------------

    def image_comparison(
        self,
        observed: Optional[np.ndarray] = None,
        *,
        stages: Optional[Iterable[str]] = None,
        point: str = "median",
    ) -> Figure:
        """One row of (observed, model, residual, hist) per (stage, dataset).

        Single-dataset stages give one row each (the classic layout); multi-dataset
        stages expand to one row per band, each rendered through its own ``sees``/PSF/
        noise/mask. ``observed`` overrides the observed image for every row; leave it
        ``None`` to use each band's own observed image. Each stage's noise σ is sourced
        from its prob_model (works for both Forward and Backward variants)."""
        obs_override = None if observed is None else np.asarray(observed)
        if stages is None:
            stages = list(self.stages.keys())
        stages = list(stages)
        # rows = flattened (stage, band); multi-dataset stages contribute n_datasets rows.
        rows = [(name, k) for name in stages
                for k in range(self.stages[name].n_datasets())]
        fig, axs = plt.subplots(len(rows), 4, figsize=(13, 3.2 * len(rows)),
                                squeeze=False)
        for row_ax, (name, k) in zip(axs, rows):
            p = self.stages[name]
            band = f"/band {k}" if p.n_datasets() > 1 else ""
            observed_k = (obs_override if obs_override is not None
                          else np.asarray(p.observed_for(k)))
            predicted = np.asarray(p.simulate(point=point, dataset=k))
            residual = normalized_residual(observed_k, predicted,
                                           p.err_map_at(predicted, dataset=k))

            # Apply per-band fit mask so chi², color scales, and residual panels
            # are not dominated by excluded (hot/bad) pixels.
            m = p.mask_for(k)
            mbool = None if m is None else np.asarray(m).astype(bool)

            resid_fit = residual if mbool is None else residual[mbool]
            chisq = float(np.sum(resid_fit ** 2))
            n_pix = residual.size if mbool is None else int(mbool.sum())
            ndof = max(n_pix - p.n_params, 1)

            obs_disp = observed_k if mbool is None else np.where(mbool, observed_k, np.nan)
            res_disp = residual if mbool is None else np.where(mbool, residual, np.nan)

            plot_image(row_ax[0], obs_disp, title=f"Observed{band}", scale="asinh")
            plot_image(row_ax[1], predicted,
                       title=f"{name}{band} model (χ²/ν={chisq / ndof:.3f})",
                       scale="asinh")
            plot_image(row_ax[2], res_disp, title=f"{name}{band} residual", residual=True)
            plot_residual_histogram(row_ax[3], resid_fit.ravel(), title=f"{name}{band} hist")
        fig.tight_layout()
        return fig


# ---------------------------------------------------------------------------
# Forward-mode convenience
# ---------------------------------------------------------------------------


def plot_scene(
    model,
    simulators,
    params,
    *,
    save_dir: Optional[str] = None,
    grid_pix: int = 400,
    fov_arcsec: Optional[Any] = None,
    center: Optional[Any] = None,
    combined: bool = True,
    scale: str = "asinh",
    linear_width: Optional[float] = None,
    log_vmin: float = 1e-2,
    **kw,
) -> Dict[str, Figure]:
    """Render a scene at explicit parameters: model images and source planes.

    The forward-mode entry point — give it the scene, one
    :class:`~gigalens.jax.scene_simulator.SceneSimulator` per observation, and a
    structured params dict, and get back the figures without an inference run behind
    them::

        src_planes = [i for i, p in enumerate(model.planes) if p.has_light]
        sims = [SceneSimulator(model, cfg, sees=model.planes[i].light)
                for i in src_planes]
        figs = plot_scene(model, sims, model.to_params(truth))

    Returns ``{"scene": fig}``: one figure, one row per source plane, the lensed
    model image beside the intrinsic source it came from (see
    :meth:`PosteriorReport.scene_panel`). ``combined=False`` instead returns the two
    panels separately as ``{"model": fig, "source": fig}`` — the image per
    observation, and the source planes — which is the better shape when many bands
    see the same plane, or when you want to save them at different sizes.

    Neither form needs observed data; for residuals, add noise to the render, wrap it
    in ``ImageData``/``ProbModel`` and use :class:`PosteriorReport` on the result.

    The source panels are auto-framed on their own light, so they are generally much
    tighter than the image panels beside them. ``fov_arcsec`` / ``center`` override
    that and accept a scalar or a per-view dict — see
    :meth:`PosteriorReport.source_panel`. Extra keywords reach the panel builder.

    ``scale`` sets the color scale on **both** columns and in both ``combined`` modes:
    ``"asinh"`` (default), ``"sqrt"`` (:class:`~matplotlib.colors.PowerNorm` with
    ``gamma=0.5``), ``"linear"``, or ``"log"``. ``linear_width`` tunes the asinh knee
    and ``log_vmin`` the log floor; each is ignored by the scales it does not apply to.
    Reach for ``"sqrt"`` or ``"linear"`` when the compressive default is flattening
    real structure — a low-Sersic-index source rendered in asinh reads as a flat disc
    whether or not it actually has a core.

    This is deliberately thin: the work is in
    :class:`~gigalens_research.inference_utils.posterior.FixedParams`, which makes
    every existing plotter work on a forward scene. Reach for the class directly when
    you want individual panels or to drive the overlays yourself.
    """
    import os

    from ..inference_utils.posterior import FixedParams

    scene = FixedParams(model, simulators, params)
    report = PosteriorReport(scene)
    if combined:
        figs: Dict[str, Figure] = {
            "scene": report.scene_panel(
                grid_pix=grid_pix, fov_arcsec=fov_arcsec, center=center,
                scale=scale, linear_width=linear_width, log_vmin=log_vmin, **kw),
        }
    else:
        figs = {
            "model": report.model_panel(
                scale=scale, linear_width=linear_width, log_vmin=log_vmin),
            "source": report.source_panel(
                with_observed=False, grid_pix=grid_pix,
                fov_arcsec=fov_arcsec, center=center,
                scale=scale, linear_width=linear_width, log_vmin=log_vmin, **kw),
        }
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        for name, fig in figs.items():
            fig.savefig(os.path.join(save_dir, f"{name}.png"),
                        dpi=140, bbox_inches="tight")
    return figs
