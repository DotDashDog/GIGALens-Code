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
- [Primitive plotters reference](#primitive-plotters-reference)
  - [plot_image](#plot_image)
  - [Source plane and caustics](#source-plane-and-caustics)
  - [Corner plots](#corner-plots)
  - [Convergence plots](#convergence-plots)
  - [Truth diagnostics](#truth-diagnostics)
- [Labels and parameter flattening](#labels-and-parameter-flattening)

---

## Module overview

| Module | What it provides |
|---|---|
| `image.py` | `plot_image`, `normalized_residual`, `plot_residual_histogram` |
| `source_plane.py` | `plot_source_plane`, `plot_caustics`, `plot_critical_curves`, `plot_caustics_critical` |
| `convergence.py` | `plot_chain_traces`, `plot_running_rhat`, `plot_running_ess`, `plot_loss_history` |
| `corner.py` | `plot_corner`, `plot_corner_overlay` |
| `labels.py` | `LATEX_LABELS`, `latex_label`, `flatten_params`, `flatten_param_names` |
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
    truth_x=None,            # physical-space truth (nested list-of-dicts)
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
fig = report.convergence_panel(trace_param=0)  # 0-indexed parameter to trace
```

R̂ is plotted as R̂ − 1 on a log-y axis so near-convergence (values close
to 0) is visible. The aggregates shown are `max(R̂)` and `min(ESS)` across
all parameters — the conservative worst-case statistics.

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
    overplots=None,    # dict {label: physical_params} drawn as star markers
    plot_params=None,  # subset of flat parameter names; default: all
    latex=True,        # use LaTeX labels from the LATEX_LABELS registry
)
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
    z_score_group="mass",        # parameter group shown in z-score bar plot
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
parameter in the chosen group. Bars exceeding ±2σ are accented to
highlight biased fits.

```python
fig = report.z_score_panel(
    group="mass",        # "mass", "lens_light", "src_light", "all"
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
truth_fn = truth_source_from_light_model(vela.light, truth_x[2][0])
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
    truth_source_fn=truth_source_from_light_model(vela.light, truth_x[2][0]),
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
    plot_params=None,       # subset of flat parameter names
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

# Caustic / critical-curve overlays:
plot_caustics_critical(ax, posterior, point="median")  # both on one axes
plot_caustics(ax, posterior, point="median")            # source plane only
plot_critical_curves(ax, posterior, point="median")     # image plane only
```

Caustics and critical curves are computed via `lenstronomy` using the
mass-profile name map in `LENS_MODEL_NAME_MAP`. Add entries there for
profiles not yet covered.

---

### Corner plots

```python
from gigalens_research.plotting import plot_corner, plot_corner_overlay

# Single posterior:
fig = plot_corner(
    posterior,
    fig=None,           # existing Figure to draw into (for manual overlays)
    plot_params=None,   # subset of flat param names; default: all
    truth=None,         # NaN used for params absent from truth_x
    overplots=None,     # dict {label: physical_params} → star markers
    color="black",
    truth_color="black",
    overplot_color="red",
    latex=True,
    **corner_kwargs,    # passed through to corner.corner
)

# Multiple posteriors with a consistent shared axis range:
fig = plot_corner_overlay(
    {"SVI": svi_post, "HMC": hmc_post},
    truth=truth_x,
    overplots={"MAP": map_x},
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

# Raw chain values for one parameter index:
plot_chain_traces(ax, sampler_posterior, param=0)

# Running R̂ − 1 on a log-y axis (aggregate="max" = worst-case parameter):
plot_running_rhat(ax, sampler_posterior, aggregate="max")

# Running ESS (aggregate="min" = most constrained parameter):
plot_running_ess(ax, sampler_posterior, aggregate="min")

# Optimization loss curve:
plot_loss_history(ax, loss_array, title="MAP χ²", ylabel="χ²", log_y=False)
```

---

### Truth diagnostics

```python
from gigalens_research.plotting import plot_z_scores, plot_source_comparison

# Z-score bar plot (requires SamplerPosterior or SurrogatePosterior):
plot_z_scores(
    ax, posterior, truth_x,
    group="mass",        # "mass", "lens_light", "src_light", "all"
    params=None,         # explicit list overrides group
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

## Labels and parameter flattening

```python
from gigalens_research.plotting import (
    LATEX_LABELS,          # dict: short_name → LaTeX string
    latex_label,           # e.g. latex_label("theta_E") → r"$\theta_E$"
    flatten_params,        # nested list-of-dicts → {flat_name: array}
    flatten_param_names,   # nested list-of-dicts → [flat_name, ...]
)
```

`flatten_params` applies the gigalens group-prefix convention:
- **mass** params: no prefix (`theta_E`, `gamma`, `e1`, …)
- **lens light** params: `lens_` prefix (`lens_R_sersic`, `lens_Ie`, …)
- **source light** params: `src_` prefix (`src_beta`, `src_center_x`, …)

This flat naming is used consistently by all plotters, the label registry,
and the truth-diagnostics functions.

**Adding custom LaTeX labels** (until profile classes carry them natively):

```python
from gigalens_research.plotting.labels import LATEX_LABELS

LATEX_LABELS.update({
    "my_custom_param": r"$\alpha_{\rm custom}$",
    "src_n_max":        r"$n_{\rm max}$",
})
```

Unregistered parameter names fall back to a readable plain-text form.
