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

from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .convergence import (
    plot_chain_traces,
    plot_loss_history,
    plot_running_ess,
    plot_running_rhat,
)
from .corner import plot_corner, plot_corner_overlay
from .image import normalized_residual, plot_image, plot_residual_histogram
from .source_plane import plot_caustics_critical, plot_source_plane
from .truth import plot_source_comparison, plot_z_scores


# ---------------------------------------------------------------------------
# PosteriorReport
# ---------------------------------------------------------------------------


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

    def image_panel(
        self,
        observed: Optional[np.ndarray] = None,
        *,
        point: str = "median",
        with_caustics: bool = False,
        log_vmin: float = 1e-3,
    ) -> Figure:
        """1×4 panel: observed, model, normalized residual, residual histogram.

        ``observed`` defaults to ``posterior.ctx.prob_model.observed_image``;
        pass it only if you want to compare against something else (e.g. a
        noise-free truth image). Noise σ comes from
        :meth:`Posterior.err_map_at`, so this works for both
        :class:`ForwardProbModel` and :class:`BackwardProbModel`.
        """
        if observed is None:
            observed = np.asarray(self.posterior.ctx.prob_model.observed_image)
        else:
            observed = np.asarray(observed)
        predicted = np.asarray(self.posterior.simulate(point=point))
        err_map = self.posterior.err_map_at(predicted)
        residual = normalized_residual(observed, predicted, err_map)
        chisq = float(np.sum(residual ** 2))
        ndof = max(observed.size - self.posterior.n_params, 1)
        red_chisq = chisq / ndof

        sc = self.posterior.ctx.sim_config
        extent = (-sc.num_pix / 2 * sc.delta_pix, sc.num_pix / 2 * sc.delta_pix,
                  -sc.num_pix / 2 * sc.delta_pix, sc.num_pix / 2 * sc.delta_pix)

        fig, axs = plt.subplots(1, 4, figsize=(13, 3.2))
        plot_image(axs[0], observed, extent=extent, title="Observed", log_vmin=log_vmin)
        plot_image(axs[1], predicted, extent=extent,
                   title=f"{self.prefix}Model ({point}, χ²/ν={red_chisq:.3f})",
                   log_vmin=log_vmin)
        plot_image(axs[2], residual, extent=extent,
                   title=f"{self.prefix}Normalized residual", residual=True)
        plot_residual_histogram(axs[3], residual,
                                title=f"{self.prefix}Gaussianity test")
        if with_caustics:
            plot_caustics_critical(axs[0], self.posterior, point=point)
            plot_caustics_critical(axs[1], self.posterior, point=point)
        fig.tight_layout()
        return fig

    # -- convergence panel ---------------------------------------------------

    def convergence_panel(self, *, trace_param: int = 0) -> Figure:
        """1×3 panel: chain traces (for one parameter), running R-hat,
        running ESS. Only applicable to :class:`SamplerPosterior`."""
        if not hasattr(self.posterior, "samples_z"):
            raise TypeError(
                f"convergence_panel needs a SamplerPosterior; got "
                f"{type(self.posterior).__name__}."
            )
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.5))
        plot_chain_traces(axs[0], self.posterior, param=trace_param)
        plot_running_rhat(axs[1], self.posterior, aggregate="max")
        plot_running_ess(axs[2], self.posterior, aggregate="min")
        fig.tight_layout()
        return fig

    # -- source-plane panel --------------------------------------------------

    def source_panel(
        self,
        *,
        point: str = "median",
        grid_pix: int = 400,
        fov_arcsec: Optional[float] = None,
        with_caustics_on_image: bool = True,
        observed: Optional[np.ndarray] = None,
        with_observed: bool = True,
    ) -> Figure:
        """1×2 panel: intrinsic source-plane image, plus the observed image
        with caustic/critical overlay.

        ``observed`` defaults to ``posterior.ctx.prob_model.observed_image``;
        pass it explicitly to use something else, or set ``with_observed=False``
        to drop the second panel and show only the source plane.
        """
        if not with_observed:
            fig, ax = plt.subplots(1, 1, figsize=(5, 4))
            plot_source_plane(ax, self.posterior, point=point,
                              grid_pix=grid_pix, fov_arcsec=fov_arcsec)
            fig.tight_layout()
            return fig
        if observed is None:
            observed = np.asarray(self.posterior.ctx.prob_model.observed_image)

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        plot_source_plane(axs[0], self.posterior, point=point,
                          grid_pix=grid_pix, fov_arcsec=fov_arcsec)
        sc = self.posterior.ctx.sim_config
        extent = (-sc.num_pix / 2 * sc.delta_pix, sc.num_pix / 2 * sc.delta_pix,
                  -sc.num_pix / 2 * sc.delta_pix, sc.num_pix / 2 * sc.delta_pix)
        plot_image(axs[1], np.asarray(observed), extent=extent, title="Observed")
        if with_caustics_on_image:
            plot_caustics_critical(axs[1], self.posterior, point=point)
        fig.tight_layout()
        return fig

    # -- corner --------------------------------------------------------------

    def corner(
        self,
        *,
        truth=None,
        overplots: Optional[Dict[str, Any]] = None,
        plot_params: Optional[List[str]] = None,
        latex: bool = True,
    ) -> Figure:
        """Corner plot of this posterior. ``truth`` defaults to the
        ``truth_x`` supplied at init (if any); pass it explicitly to override
        or pass ``truth=False`` to skip the overlay even when ``truth_x`` was
        set."""
        if truth is None:
            truth = self.truth_x
        elif truth is False:
            truth = None
        return plot_corner(
            self.posterior,
            truth=truth, overplots=overplots,
            plot_params=plot_params, latex=latex,
        )

    # -- truth-aware panels --------------------------------------------------

    def z_score_panel(
        self,
        *,
        group: str = "mass",
        truth_x=None,
        threshold: float = 2.0,
        sort_by_abs: bool = False,
    ) -> Figure:
        """Bar plot of per-parameter z-scores against truth.

        Requires ``truth_x`` to have been supplied at init or passed in here.
        ``group`` is one of ``'mass'`` (default), ``'lens_light'``,
        ``'src_light'``, ``'all'``, or ``None``.
        """
        truth_x = truth_x if truth_x is not None else self.truth_x
        if truth_x is None:
            raise ValueError(
                "z_score_panel needs truth_x; pass it to PosteriorReport(...) "
                "or to z_score_panel(truth_x=...)."
            )
        fig, ax = plt.subplots(figsize=(max(6, 0.45 * self.posterior.n_params), 3.5))
        plot_z_scores(ax, self.posterior, truth_x,
                      group=group, threshold=threshold, sort_by_abs=sort_by_abs)
        fig.tight_layout()
        return fig

    def source_comparison_panel(
        self,
        *,
        truth_source_image: Optional[np.ndarray] = None,
        truth_source_extent: Optional[tuple] = None,
        truth_source_fn=None,
        point: str = "median",
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
            point=point, log_vmin=log_vmin,
            grid_pix=grid_pix, fov_arcsec=fov_arcsec, center=center,
        )
        fig.tight_layout()
        return fig

    # -- full report --------------------------------------------------------

    def full_report(
        self,
        *,
        observed: Optional[np.ndarray] = None,
        truth=None,
        save_dir: Optional[str] = None,
        z_score_group: str = "mass",
    ) -> Dict[str, Figure]:
        """Build all panels and (optionally) save each as a PNG to ``save_dir``.

        ``observed`` defaults to ``posterior.ctx.prob_model.observed_image``.
        Noise model is read from the prob_model in all cases.

        If the report was constructed with truth inputs (``truth_x`` or
        ``truth_source_image``), this also generates a z-score bar plot
        (``z_scores``) and a source-plane comparison (``source_comparison``).
        Use ``z_score_group`` to switch which parameter group is shown
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
        figs["corner"] = self.corner(truth=truth)

        # Truth-aware panels: included automatically if the inputs are there.
        if (truth if truth is not None else self.truth_x) is not None \
                and hasattr(self.posterior, "quantiles_z"):
            figs["z_scores"] = self.z_score_panel(
                truth_x=truth, group=z_score_group,
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
        return cls(ctx=ctx, stages=stages)

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

    # -- compound corner -----------------------------------------------------

    def compound_corner(
        self,
        *,
        stages: Optional[Iterable[str]] = None,
        truth=None,
        overplots_stage: Optional[str] = None,
        overplot_label: str = "MAP",
        plot_params: Optional[List[str]] = None,
        colors: Optional[Dict[str, str]] = None,
        latex: bool = True,
    ) -> Figure:
        """Overlay multiple stages' posteriors on one corner figure.

        ``stages`` defaults to all stages that have sample-like posteriors
        (i.e. excluding pure point estimates). Pass ``overplots_stage`` to mark
        e.g. the MAP optimum as stars on top of the contours.
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
            posteriors, plot_params=plot_params, truth=truth,
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
        """One row of (observed, model, residual, hist) per requested stage.

        ``observed`` defaults to ``ctx.prob_model.observed_image``. Each
        stage's noise σ is sourced from its prob_model (works for both
        Forward and Backward variants)."""
        if observed is None:
            observed = np.asarray(self.ctx.prob_model.observed_image)
        else:
            observed = np.asarray(observed)
        if stages is None:
            stages = list(self.stages.keys())
        n_rows = len(list(stages))
        fig, axs = plt.subplots(n_rows, 4, figsize=(13, 3.2 * n_rows),
                                squeeze=False)
        for row_ax, name in zip(axs, stages):
            p = self.stages[name]
            predicted = np.asarray(p.simulate(point=point))
            residual = normalized_residual(observed, predicted, p.err_map_at(predicted))
            chisq = float(np.sum(residual ** 2))
            ndof = max(observed.size - p.n_params, 1)
            plot_image(row_ax[0], observed, title="Observed")
            plot_image(row_ax[1], predicted,
                       title=f"{name} model (χ²/ν={chisq / ndof:.3f})")
            plot_image(row_ax[2], residual, title=f"{name} residual", residual=True)
            plot_residual_histogram(row_ax[3], residual, title=f"{name} hist")
        fig.tight_layout()
        return fig
