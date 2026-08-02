# `plotting` — GIGALens Plotting Utilities

matplotlib-based plotters for GIGALens inference outputs. All functions
return `Figure` / `Axes` objects without calling `plt.show`; callers
control display and saving.

---

## Contents

- [Module overview](#module-overview)
- [Quick start](#quick-start)
- [PosteriorReport — single-posterior compound panels](#posteriorReport)
  - [image_panel](#image_panel)
  - [convergence_panel](#convergence_panel)
  - [source_panel](#source_panel)
  - [corner](#corner)
  - [full_report](#full_report)
  - [Truth-aware panels](#truth-aware-panels)
- [PipelineReport — multi-stage comparison](#pipelinereport)
  - [loss_histories](#loss_histories)
  - [compound_corner](#compound_corner)
  - [image_comparison](#image_comparison)
  - [diagnostics](#diagnostics)
- [Primitive plotters reference](#primitive-plotters-reference)
  - [plot_image](#plot_image)
  - [Source plane and caustics](#source-plane-and-caustics)
  - [Corner plots](#corner-plots)
  - [Convergence plots](#convergence-plots)
  - [Truth diagnostics](#truth-diagnostics)
  - [Debug diagnostics](#debug-diagnostics)
- [Parameter names and labels](#parameter-names-and-labels)

---

## Module overview

| Module | What it provides |
|---|---|
| `image.py` | `plot_image`, `normalized_residual`, `plot_residual_histogram` |
| `source_plane.py` | `plot_source_plane`, `plot_caustics`, `plot_critical_curves`, `plot_caustics_critical` |
| `convergence.py` | `plot_chain_traces`, `plot_running_rhat`, `plot_running_ess`, `plot_loss_history` |
| `corner.py` | `plot_corner`, `plot_corner_overlay` |
| `diagnostics.py` | `plot_stage_diagnostics`, `plot_mclmc_diagnostics`, `register_diagnostic_plotter` |
| `labels.py` | `LATEX_LABELS`, `latex_label`, `z_column_labels` |
| `truth.py` | `plot_z_scores`, `plot_source_comparison` |
| `reports.py` | `PosteriorReport`, `PipelineReport` |

Everything is importable from the top-level package:

```python
from gigalens_research.plotting import (
    PosteriorReport, PipelineReport,
    plot_image, plot_corner, plot_source_plane,
    plot_running_rhat, plot_z_scores,
    # etc.
)
```

Parameters are named by their **scene path** — `planes/0/mass/0/theta_E`,
`planes/1/geometry/redshift`, `cosmo/H0`. The records behind that naming live in
`gigalens_research.param_index`; `ParamSite`, `param_sites` and `select_sites`
are re-exported here for building the plotters' `plot_params=` / `select=`
arguments. See [Parameter names and labels](#parameter-names-and-labels).

---

## Quick start

```python
from gigalens_research.plotting import PosteriorReport, PipelineReport

# Single-stage report (HMC posterior, no truth):
report = PosteriorReport(pipeline.posterior("hmc"), prefix="System 07 ")
figs = report.full_report(save_dir="plots/system_07")

# Multi-stage comparison:
pr = PipelineReport(pipeline)
fig = pr.compound_corner()
fig.savefig("plots/corner_all_stages.pdf", bbox_inches="tight")
```

---

## PosteriorReport

Wraps a single `Posterior` and exposes one method per panel type. Every
method returns its `Figure`; call `.savefig(...)` yourself or let
`full_report` do it in bulk.

```python
report = PosteriorReport(
    posterior,
    prefix="",               # prepended to figure titles
    truth_x=None,            # physical-space truth: scene-nested
                             # ({"planes": {0: {"mass": {0: {...}}}}, "cosmo": {...}})
                             # or path-keyed ({"planes/0/mass/0/theta_E": ...})
    truth_source_image=None, # pre-rendered 2-D truth source image
    truth_source_extent=None,# (xmin, xmax, ymin, ymax) in arcsec — required
                             # when truth_source_image is provided
    truth_source_fn=None,    # OR a callable f(X, Y) -> image
                             # (mutually exclusive with truth_source_image)
)
```

If `truth_x` is supplied, corners automatically overlay the truth and
`full_report` adds a z-score panel. If a truth source is supplied (either
form), `full_report` adds a source-comparison panel.

---

### image_panel

A 1×4 row: **observed | model | normalized residual | Gaussianity
histogram**. Reduced χ²/ν is shown in the model panel title. Noise σ is
read automatically from the prob_model, so this works for both
`ForwardProbModel` and `BackwardProbModel`.

```python
fig = report.image_panel(
    observed=None,  # defaults to prob_model.observed_image
    point="median", # "median", "mean", or "best" (PointEstimate only)
    log_vmin=1e-3,
)
fig.savefig("image.pdf", bbox_inches="tight")
```

---

### convergence_panel

A 1×3 row: **chain traces | running R̂−1 (log scale) | running ESS**.
Only valid for `SamplerPosterior`.

```python
fig = report.convergence_panel()                 # trace the worst-R̂ z-column
fig = report.convergence_panel(n_worst=5)         # list the 5 worst per panel
fig = report.convergence_panel(trace_param=0)     # trace an explicit x-space param
```

R̂ is plotted as R̂ − 1 on a log-y axis so near-convergence (values close
to 0) is visible. The aggregate curves shown are `max(R̂)` and `min(ESS)`
across all parameters — the conservative worst-case statistics — and each
panel is now annotated, from the labeled `posterior.convergence` report, with
the `n_worst` (default 3) worst parameters **by name**: the R̂ panel by largest
R̂, the ESS panel by lowest bulk-ESS (shown as `bulk/tail`). The chain-trace
panel defaults to the single worst-R̂ **z-column** (the space R̂/ESS live in);
pass `trace_param=` to trace an explicit x-space / corner-plot parameter instead.

---

### source_panel

A 1×2 panel: **intrinsic source plane** (no PSF, no lensing) | **observed
image with caustic/critical curve overlay**.

```python
fig = report.source_panel(
    point="median",
    grid_pix=400,             # source plane resolution in pixels
    fov_arcsec=None,          # field of view; defaults to sim_config extent
    with_caustics_on_image=True,
    with_observed=True,       # False → source plane only, no second panel
)
```

Caustics are computed via `lenstronomy` at the mass-model parameters
corresponding to `point`.

---

### corner

```python
fig = report.corner(
    truth=None,        # defaults to self.truth_x; pass False to suppress
    overplots=None,    # dict {label: point} drawn as star markers
    kind=None,         # ┐ the plot_corner filters, ANDed together;
    plane=None,        # │ omitting them all plots every parameter
    component=None,    # │
    select=None,       # ┘
    plot_params=None,  # explicit scene path keys; excludes the filters above
    latex=True,        # use LaTeX labels from the LATEX_LABELS registry
)

report.corner(kind="mass", plane=0)      # just plane 0's mass parameters
report.corner(kind="cosmology")
```

Parameters present in the posterior but missing from `truth_x` (e.g. when
approximating an `ImageBasedLight` truth with a shapelet source) get `NaN`
in the truth row — `corner` silently skips those crosshairs. A
`UserWarning` lists the unmatched parameters.

---

### full_report

Generates all applicable panels in one call and optionally saves each to
`<save_dir>/<name>.png`.

```python
figs = report.full_report(
    save_dir="plots/system_07",  # None → don't save automatically
    z_score_kind="mass",         # parameter class shown in the z-score bar plot
)
# Returns a dict. Keys depend on the posterior type and truth inputs:
# "image"              — always included
# "convergence"        — SamplerPosterior only
# "source"             — always included
# "corner"             — always included
# "z_scores"           — only if truth_x is provided
# "source_comparison"  — only if truth_source_* is provided
```

---

### Truth-aware panels

These require truth inputs provided at `PosteriorReport` construction time
(or passed directly to the method).

**z_score_panel** — bar plot of `z = (truth − median) / σ` for each
parameter of the chosen kind. Bars exceeding ±2σ are accented to
highlight biased fits.

```python
fig = report.z_score_panel(
    kind="mass",         # "cosmology", "geometry", "mass", "light",
                         # or "all" / None for everything
    threshold=2.0,       # horizontal reference lines at ±threshold
    sort_by_abs=False,   # True → sort bars by |z|, most biased on the left
)
```

**source_comparison_panel** — 1×3 panel: truth source | recovered source |
residual. Both image panels share a common colorbar so brightnesses are
directly comparable. The recovered source is expressed in surface-brightness
units to match the truth.

```python
# Pre-rendered truth array:
fig = report.source_comparison_panel(
    truth_source_image=arr,                          # shape (N, N)
    truth_source_extent=(-1.5, 1.5, -1.5, 1.5),     # arcsec
    point="median",
    log_vmin=1e-2,
)

# Continuous truth function (e.g. Vela ImageBasedLight):
from gigalens_research.inference_utils import truth_source_from_light_model
# The truth params of the source's light component, read out of the
# scene-nested truth at its own path (here: plane 1, light component 0).
truth_fn = truth_source_from_light_model(
    vela.light, truth_x["planes"][1]["light"][0])
fig = report.source_comparison_panel(
    truth_source_fn=truth_fn,
    grid_pix=400,
    fov_arcsec=2.0,
)
```

**Full Vela simulated-system example:**

```python
from gigalens_research.simulations import load_vela_source
from gigalens_research.inference_utils import truth_source_from_light_model
from gigalens_research.plotting import PosteriorReport

vela = load_vela_source("vela_sources/vela07_cam0_a0.500_f814w")
post = pipeline.posterior("hmc")

report = PosteriorReport(
    post,
    prefix="Vela07 ",
    truth_x=truth_x,
    truth_source_fn=truth_source_from_light_model(
        vela.light, truth_x["planes"][1]["light"][0]),
)
figs = report.full_report(save_dir="plots/vela07")
# Saves: image.png, convergence.png, source.png, corner.png,
#        z_scores.png, source_comparison.png
```

---

## PipelineReport

Multi-stage compound plots. Build from a live `Pipeline` or load from
disk without re-running.

```python
# From a live pipeline (after pipeline.run()):
pr = PipelineReport(pipeline)

# From disk:
pr = PipelineReport.from_disk(
    "results/system_07", ctx,
    stage_names=["svi", "hmc"],  # None → load all non-stale stage dirs
)
```

---

### loss_histories

Side-by-side MAP χ² and SVI −ELBO curves for all stages that carry one.

```python
fig = pr.loss_histories()
```

---

### compound_corner

Overlay multiple stages' posteriors on one corner figure. Useful for
comparing SVI contours against HMC chains or two MCLMC runs with
different settings.

```python
fig = pr.compound_corner(
    stages=None,            # list of stage names; default: all with flat_x
    truth=truth_x,          # physical-space truth overlay (optional)
    overplots_stage="map",  # mark the MAP point as star markers
    overplot_label="MAP",
    kind=None,              # ┐ the plot_corner_overlay filters; default: all
    plane=None,             # │
    component=None,         # │
    select=None,            # ┘
    plot_params=None,       # explicit scene path keys; excludes the filters
    colors=None,            # dict {stage_name: color}; auto-assigned if None
)
```

All posteriors share a single axis range (computed from the joint quantile
range across all stages) to prevent the `corner` library from creating
orphaned duplicate figures on overlay.

---

### image_comparison

One row of (observed | model | residual | histogram) per stage.

```python
fig = pr.image_comparison(
    observed=None,  # defaults to ctx.prob_model.observed_image
    stages=None,    # list of stage names; default: all
    point="median",
)
```

---

### diagnostics

Render a stage's captured *debug* run history (e.g. an MCLMC tuning trace).
Requires that the stage was run with `debug=True` (see the inference_utils
README). Works on both live-pipeline and `from_disk` reports.

```python
fig = pr.diagnostics("mclmc", chain=3)  # extra kwargs forwarded to the plotter
```

For MCLMC this produces five stacked panels vs. step: per-chain step size,
trajectory length `L`, inverse-mass-matrix eigenvalue spread, the energy-error
ratio `xi` for one chain, and a finite-step (NaN) heatmap — with dashed lines
marking the three tuning-stage boundaries. The eigenvalue panel also overlays
the final output samples' covariance eigenvalue spread (min/mean/max) as
horizontal dashed lines, so you can see whether the inverse mass matrix settled
toward the posterior covariance it targets. It's the quickest way to see *where*
a sampling run blew up.

An optional companion plot corner-plots the final draws against a Gaussian
surrogate — a multivariate normal with mean equal to the sample mean and
covariance equal to the **final inverse mass matrix** used during sampling.
Everything is in the unconstrained (z) space, where both the positions and the
inverse mass matrix live. It shows how Gaussian the posterior is and whether the
preconditioner the sampler used matches the realized draw covariance:

```python
fig = pr.diagnostics_surrogate_corner("mclmc")  # max_samples=, seed= forwarded
```

---

## Primitive plotters reference

### plot_image

The workhorse for all 2-D surface-brightness and residual display.

```python
from gigalens_research.plotting import plot_image

im = plot_image(
    ax, image,
    fig=None,           # needed for colorbar; inferred from ax if None
    extent=None,        # (xmin, xmax, ymin, ymax) in arcsec
    title=None,
    residual=False,     # True → bwr + CenteredNorm; False → inferno + LogNorm
    colorbar=True,
    remove_axis=True,
    log_vmin=1e-2,      # lower floor for LogNorm
    log_norm=True,
    vmin=None,          # explicit color limits; override auto-scale
    vmax=None,          # (used by plot_source_comparison for shared colorbar)
)
```

---

### Source plane and caustics

```python
from gigalens_research.plotting import (
    plot_source_plane,
    plot_caustics,           # source-plane caustics only
    plot_critical_curves,    # image-plane critical curves only
    plot_caustics_critical,  # both on the same axes
)

# Intrinsic source plane (no lensing, no PSF):
plot_source_plane(
    ax, posterior,
    point="median",
    grid_pix=400,
    fov_arcsec=None,    # defaults to sim_config field of view
    center=None,        # (cx, cy) override; defaults to first src component
    with_caustics=True, # overlay caustics; default True
    log_vmin=1e-2,
)

# Caustic / critical-curve overlays. Both accept deflection_ratio= to select
# which source plane (a model can carry several at different redshifts):
plot_caustics(ax, posterior, point="median", deflection_ratio=0.7)         # source plane only
plot_critical_curves(ax, posterior, point="median", deflection_ratio=0.7)  # image plane only
plot_caustics_critical(ax, posterior, point="median")                      # both on one axes
```

Caustics and critical curves are computed **natively** from the gigalens lens
model — the deflection comes straight from each `profile.deriv`, its Jacobian
from `jax.jacfwd`, and the critical-curve contours from
`skimage.measure.find_contours`. There is no lenstronomy translation layer (no
name map): adding a new mass profile needs nothing here.

For a source plane with deflection ratio `r` the lens map is
`beta = theta - r * alpha(theta)`, so both the **critical** curve
(`det(I - r * dalpha/dtheta) = 0`, image plane) and its **caustic** (the lens
map of that curve, source plane) depend on `r`. Pass `deflection_ratio=` to draw
the pair for a given source plane; the per-plane ratios come from
`Posterior.source_plane_views()`, which `PosteriorReport.source_panel` iterates
to build one row (source plane + observed image) per source plane.

> Validated against lenstronomy's `critical_curve_caustics` at `r=1`
> (EPL + SHEAR + NFW_ELLIPSE): the physical critical/caustic curves agree to
> <5e-3 arcsec (≪ the contour grid spacing); lenstronomy's only extra output is
> spurious few-point loops on the compute-window edge, which the native path
> does not produce.

---

### Corner plots

```python
from gigalens_research.plotting import plot_corner, plot_corner_overlay

# Single posterior — every free parameter, by default:
fig = plot_corner(
    posterior,
    fig=None,           # existing Figure to draw into (for manual overlays)
    kind=None,          # "cosmology" | "geometry" | "mass" | "light", or a list
    plane=None,         # plane index, or a list of them
    component=None,     # component index, ("mass", 0), or a list
    select=None,        # predicate on a ParamSite — the escape hatch
    plot_params=None,   # explicit scene path keys, in the order given
    truth=None,         # scene-nested or path-keyed; NaN for params it omits
    overplots=None,     # dict {label: point}, same forms → star markers
    color="black",
    truth_color="black",
    overplot_color="red",
    latex=True,
    **corner_kwargs,    # passed through to corner.corner
)
```

The `kind` / `plane` / `component` / `select` filters AND together, and panels
come out in a fixed order: cosmology → geometry → mass → light, then by plane,
then by component.

```python
plot_corner(post)                                   # everything
plot_corner(post, kind="cosmology")
plot_corner(post, kind=["geometry", "mass"])
plot_corner(post, plane=1)                          # one plane's parameters
plot_corner(post, kind="mass", plane=1)
plot_corner(post, component=("mass", 0))            # pin the role
plot_corner(post, select=lambda s: s.param.startswith("e"))
plot_corner(post, plot_params=["cosmo/H0", "planes/0/mass/0/theta_E"])
```

`plot_params` and the filters are mutually exclusive — otherwise it is ambiguous
whether `plot_params` is a selection or an ordering. A filter that matches
nothing raises, rather than quietly drawing a blank figure.

```python
# Multiple posteriors with a consistent shared axis range:
fig = plot_corner_overlay(
    {"SVI": svi_post, "HMC": hmc_post},
    kind="mass",           # filters work as in plot_corner; resolved once
                           # against the first posterior, then every overlay is
                           # pinned to exactly those columns
    truth=truth_x,
    overplots={"MAP": map_post.x},
    colors={"SVI": "blue", "HMC": "black"},
    range_quantile=0.999,  # symmetric quantile bound for shared axis range
    range_pad=0.05,        # fractional padding beyond that range
)
```

---

### Convergence plots

```python
from gigalens_research.plotting import (
    plot_chain_traces,
    plot_running_rhat,
    plot_running_ess,
    plot_loss_history,
)

# Chain values for one parameter, in physical (x) space:
plot_chain_traces(ax, sampler_posterior, param=0)

# Running R̂ − 1 on a log-y axis (aggregate="max" = worst-case parameter):
plot_running_rhat(ax, sampler_posterior, aggregate="max")

# Running ESS (aggregate="min" = most constrained parameter):
plot_running_ess(ax, sampler_posterior, aggregate="min")

# Optimization loss curve:
plot_loss_history(ax, loss_array, title="MAP χ²", ylabel="χ²", log_y=False)
```

Two indexings, never mixed. `plot_chain_traces(param=i)` is in **x space**: `i`
indexes `param_sites` order, which is the corner plot's column order, so column
`i` is the same parameter in both figures. `plot_running_rhat(params=[...])` and
`plot_running_ess(params=[...])` select **sampler (z) columns**, because that is
what R̂ and ESS are computed per; those are named from
`prob_model.z_param_names`, the model's own column→name map. The bijector
reorders, so the same integer means different parameters in the two spaces.

---

### Truth diagnostics

```python
from gigalens_research.plotting import plot_z_scores, plot_source_comparison

# Z-score bar plot (requires SamplerPosterior or SurrogatePosterior):
plot_z_scores(
    ax, posterior, truth_x,
    kind="mass",         # "cosmology", "geometry", "mass", "light",
                         # or "all" / None for everything
    params=None,         # explicit scene path keys; overrides kind
    threshold=2.0,       # horizontal reference lines at ±threshold
    sort_by_abs=False,
    latex=True,
)

# 1×3 source comparison panel (truth and recovered share one colorbar):
ax_truth, ax_rec, ax_res = plot_source_comparison(
    fig, posterior, truth_source,
    extent=None,         # required when truth_source is a pre-rendered array
    point="median",
    log_vmin=1e-2,
    grid_pix=None,       # for callable truth (default 400)
    fov_arcsec=None,     # for callable truth (default sim_config FoV)
    center=None,         # for callable truth (default first src component)
)
# truth_source can be:
#   - a (N, N) ndarray  →  provide extent=(xmin, xmax, ymin, ymax) in arcsec
#   - a callable f(X, Y) -> image  →  provide grid_pix / fov_arcsec
```

---

### Debug diagnostics

Stage-specific run histories (distinct from the convergence plots above,
which work off a finished posterior). Dispatched by stage class through a
registry, so each algorithm gets its own plotter.

```python
from gigalens_research.plotting import (
    plot_stage_diagnostics,        # dispatch on diag.stage_class
    plot_mclmc_diagnostics,        # the MCLMCStage plotter
    register_diagnostic_plotter,   # decorator to add a new stage's plotter
    has_diagnostic_plotter,
)

# `diag` is a StageDiagnostics from pipeline.diagnostics(stage) or
# diagnostics_from_disk(out_dir, stage, ctx). Requires the stage ran with
# debug=True.
fig = plot_stage_diagnostics(diag, chain=0)

# Add diagnostics for a new stage:
@register_diagnostic_plotter("NutsStage")
def plot_nuts_diagnostics(diag, **kwargs):
    arr, cfg = diag.arrays, diag.config
    ...
    return fig
```

`plot_stage_diagnostics` raises a clear error if the stage wasn't run with
`debug=True` (no captured arrays) or if no plotter is registered for its class.

---

## Parameter names and labels

A parameter is named by **where it lives in the scene**:

```
cosmo/H0                      # cosmology
planes/1/geometry/redshift    # plane 1's geometry
planes/0/mass/0/theta_E       # plane 0, mass component 0
planes/1/light/0/R_sersic     # plane 1, light component 0
```

This is the scene's own naming, not a derived one, so nothing has to be
translated back to ask which plane or component a parameter belongs to. The
records are `ParamSite`s from `gigalens_research.param_index`; the plotters build
their columns from them, and so can you:

```python
from gigalens_research.plotting import param_sites, select_sites

sites = param_sites(posterior)              # one record per free parameter
[s.key for s in sites]                      # "planes/0/mass/0/theta_E", …
[(s.kind, s.plane, s.component) for s in sites]
select_sites(sites, kind="mass", plane=0)   # the same filters plot_corner takes
```

Each record carries `kind` (one of `param_index.KINDS`: `"cosmology"`,
`"geometry"`, `"mass"`, `"light"`), `param` (the bare name, `theta_E`), `paths`
(every site it feeds — longer than one only for a `shared()` parameter, which
stays a single column), and `key`, the canonical path string `plot_params=`
matches on.

**Labels.** Identity and display are separate. `param_index.site_labels(sites)`
renders a set of columns: the bare symbol (`$\theta_E$`) where it is unique on
the figure, with a `(plane, component)` superscript (`$\theta_E^{(1,0)}$`) only
where two columns would otherwise collide. `latex_label` renders one name and
accepts either a path key or a bare parameter:

```python
from gigalens_research.plotting import latex_label, LATEX_LABELS

latex_label("planes/0/mass/0/theta_E")   # r"$\theta_E$"
latex_label("theta_E")                   # same — the registry is keyed by the
                                         # bare name, the last path segment
```

**Adding custom LaTeX labels** (until profile classes carry them natively):

```python
from gigalens_research.plotting import LATEX_LABELS

LATEX_LABELS.update({
    "my_custom_param": r"$\alpha_{\rm custom}$",
    "n_max":           r"$n_{\rm max}$",
})
```

Unregistered parameter names fall back to a readable plain-text form, so an
unregistered profile still plots.

For the sampler's unconstrained columns there is a separate helper,
`plotting.labels.z_column_labels(names)`, since a z column need not correspond
to a single site. Feed it `prob_model.z_param_names` and nothing else — that is
the model's own column→name map, and any other name order disagrees with the
sampler's.

> **Why paths?** The retired label space flattened the scene into three groups
> and named parameters `theta_E__1`, `lens_R_sersic`, `src_Ie`, `cosmo_H0`. Its
> `__<i>` suffix was a *global running index* across planes, so `theta_E__1`
> could not say which plane it was on — filtering by plane was impossible. And
> its `lens_`/`src_` prefixes encoded a lens-light/source-light distinction the
> model does not have: a `Plane` carries only `mass` and `light`, and the split
> was synthesized from "light is source light iff an earlier plane has mass".
> The plane index is the real distinction, and it is what these records carry.
> The old space also silently dropped geometry parameters (a free `redshift` or
> `deflection_ratio`), which now plot as `kind="geometry"`. `flatten_params` and
> `flatten_param_names` are gone with it.
