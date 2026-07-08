#!/usr/bin/env python3
"""Deflection-ratio-only grid-search posterior for the DSPL cosmology system,
regenerated for the REAL system (z_lens=0.5), overlaid with the completed
MCLMC run's cosmology samples.

This is a diagnostic/plotting script (method-discipline: diagnosing, not
fixing; report numbers with plots; no fabrication). It does NOT run any
sampler, MAP, or simulator -- it only:

  1. Ports `GIGALens-Code-paper/experiments/sample_cosmology/def_ratio_likelihood.py`
     (which hardcoded z_lens=0.4) onto the NEW gigalens API's
     `gigalens.jax.cosmo.w0waCDM_Cosmo.deflection_ratio`, at the real system's
     z_lens=0.5 / z_source1=1.0 / z_source2=1.5.
  2. Rebuilds the SAME `LensModel` as
     `experiments/sample_cosmology/dspl_cosmology_newapi.ipynb` (its cell-5
     model-construction code, copied verbatim below) -- needed ONLY for its
     derived `z_param_names` + `bijector` (the prior's unconstrained-space
     parameterization), not for simulating or fitting anything.
  3. Loads the completed MCLMC run's `arrays.npz` (unconstrained z-space
     samples) from `results/sample_cosmology/dspl_cosmology_newapi/mclmc/`,
     maps the two cosmology columns to physical (Om0, w0) via that bijector,
     and overlays them on the grid-search posterior and the r2 level
     contours.

Everything here is 1-D quadrature over a 400x400 grid (or a bijector.forward
on ~80k points) -- cheap on CPU, no GPU, no simulator.
"""
from __future__ import annotations

import argparse
import os

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import tensorflow_probability.substrates.jax as tfp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

tfd = tfp.distributions
tfb = tfp.bijectors

from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.light.sersic import SersicEllipse
from gigalens.jax.cosmo import w0waCDM_Cosmo

from gigalens_research.inference_utils.posterior import SamplerPosterior


# ---------------------------------------------------------------------------
# System parameters -- MUST match dspl_cosmology_newapi.ipynb exactly.
# ---------------------------------------------------------------------------
Z_LENS = 0.5
Z_SOURCE1 = Z_LENS * 2  # == 1.0; the cosmology's z_source_ref (deflection_ratio == 1 here)
Z_SOURCE2 = Z_LENS * 3  # == 1.5; the plane whose deflection ratio r2 the data constrain

TRUTH = dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0, wa=0.0)

RESULTS_DIR = os.path.join(
    os.path.expanduser("~"), "GIGALens-Code", "results",
    "sample_cosmology", "dspl_cosmology_newapi",
)
MCLMC_DIR = os.path.join(RESULTS_DIR, "mclmc")

OUT_OVERLAY_PNG = os.path.join(RESULTS_DIR, "def_ratio_grid_overlay.png")
OUT_BARE_PNG = os.path.join(RESULTS_DIR, "def_ratio_grid.png")
OUT_NPZ = os.path.join(RESULTS_DIR, "def_ratio_grid.npz")


# ---------------------------------------------------------------------------
# Model construction -- verbatim from dspl_cosmology_newapi.ipynb cell 5.
#
# Only needed for `model.z_param_names` (column->name map for the MCLMC
# sampler's flat unconstrained z-vector) and `model.bijector` (the SAME
# object `ProbModel.bij` in the notebook's pipeline would carry --
# `gigalens/src/gigalens/jax/scene_prob_model.py:135` sets `self.bij =
# model.bijector`). Does NOT touch data/simulator/prob_model/pipeline, so no
# image rendering or sampling happens here.
# ---------------------------------------------------------------------------
def build_model() -> LensModel:
    z_lens = Z_LENS
    z_source1 = z_lens * 2
    z_source2 = z_lens * 3

    lens = Component(
        EPL(),
        dict(
            theta_E=tfd.LogNormal(jnp.log(1.25), 0.25),
            gamma=2.0,
            e1=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
            e2=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
            center_x=tfd.Normal(0, 0.05),
            center_y=tfd.Normal(0, 0.05),
        ),
    )

    source1 = Component(
        SersicEllipse(use_lstsq=False),
        dict(
            center_x=tfd.Normal(0, 2),
            center_y=tfd.Normal(0, 2),
            e1=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
            e2=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
            n_sersic=tfd.Uniform(1, 10),
            R_sersic=tfd.LogNormal(jnp.log(1.), 0.15),
            Ie=tfd.LogNormal(jnp.log(150), 1),
        ),
    )
    source2 = Component(
        SersicEllipse(use_lstsq=False),
        dict(
            center_x=tfd.Normal(0, 2),
            center_y=tfd.Normal(0, 2),
            e1=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
            e2=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
            n_sersic=tfd.Uniform(1, 10),
            R_sersic=tfd.LogNormal(jnp.log(1.), 0.15),
            Ie=tfd.LogNormal(jnp.log(150), 1),
        ),
    )

    def tNCDF_bij(low, high):
        return tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.NormalCDF()])

    # Verbatim from the notebook: a tfd.Uniform with a Shift/Scale/NormalCDF
    # event-space bijector in place of TFP's default (sigmoid-based) one for Uniform.
    class UniformBij(tfd.Uniform):
        def __init__(self, *args, event_space_bijector_class=tNCDF_bij, **kwargs):
            self._esb = event_space_bijector_class(*args)
            super().__init__(*args, **kwargs)

        def _default_event_space_bijector(self):
            return self._esb

    cosmo = Component(
        w0waCDM_Cosmo(z_lens=z_lens, z_source_ref=z_source1),
        dict(
            H0=70.,
            Om0=UniformBij(jnp.float64(0.0), jnp.float64(1.0)),
            w0=UniformBij(jnp.float64(-2.0), jnp.float64(-1 / 3)),
            wa=0.0,
            k=0.0,
        ),
    )

    model = LensModel(
        [
            Plane(redshift=z_lens, mass=[lens]),
            Plane(redshift=z_source1, light=[source1]),
            Plane(redshift=z_source2, light=[source2]),
        ],
        cosmo=cosmo,
    )
    return model


# ---------------------------------------------------------------------------
# Grid-search likelihood (port of def_ratio_likelihood.py onto the new API).
# ---------------------------------------------------------------------------
def logsumexp_norm(log_prob: np.ndarray) -> np.ndarray:
    c = np.max(log_prob)
    lse = c + np.log(np.nansum(np.exp(log_prob - c)))
    return log_prob - lse


def get_mass_levels(prob_grid: np.ndarray, levels=(0.68, 0.955, 0.997)):
    """Probability values at which the cumulative mass (sorted descending)
    first reaches each requested fraction -- i.e. the 68/95.5/99.7% highest-
    density contour levels. Verbatim port of the old script's
    `get_ll_levels`."""
    flat = prob_grid.flatten()
    idx = np.argsort(flat)[::-1]
    flat_sorted = flat[idx]
    cum = np.cumsum(flat_sorted)
    cum /= cum[-1]
    out = []
    for lvl in levels:
        out.append(flat_sorted[cum <= lvl][-1])
    return sorted(out)


def make_prob_grid(cosmo_model, r2_obs, sigma_r, n_grid=400):
    """Gaussian likelihood on the observed deflection ratio r2 = deflection_ratio
    at z_source2, uniform priors on (Om0, w0) over exactly the grid box, evaluated
    on a plain vectorized (Om0, w0) grid -- no jax.pmap, no vmap needed:
    `CosmoBase.comoving_distance_z1z2` already broadcasts the 1-D quadrature over
    a batch of (Om0, w0) via its own batch_ndim handling (verified empirically:
    a flat (n_grid**2,) Om0/w0 pair runs through deflection_ratio unchanged).
    """
    Om0_grid = jnp.linspace(0.0, 1.0, n_grid)
    w0_grid = jnp.linspace(-2.0, -1.0 / 3.0, n_grid)
    Om0_mesh, w0_mesh = jnp.meshgrid(Om0_grid, w0_grid, indexing="ij")
    Om0_flat, w0_flat = Om0_mesh.flatten(), w0_mesh.flatten()

    r2_flat = cosmo_model.deflection_ratio(
        Z_SOURCE2, H0=TRUTH["H0"], Om0=Om0_flat, k=TRUTH["k"], w0=w0_flat, wa=TRUTH["wa"]
    )
    # cosmo_model.z_lens is stored as a shape-(1,) array (jax/cosmo.py), so every
    # deflection_ratio call carries a leading size-1 axis; squeeze it back off.
    r2_flat = jnp.squeeze(jnp.asarray(r2_flat), axis=0)
    r2_grid = np.asarray(r2_flat, dtype=np.float64).reshape(Om0_mesh.shape)

    chi2 = ((r2_grid - r2_obs) / sigma_r) ** 2
    log_like = -0.5 * (chi2 + np.log(2 * np.pi * sigma_r ** 2))
    # Uniform priors on Om0 in (0,1) and w0 in (-2,-1/3), evaluated over EXACTLY
    # this box: log p(Om0)+log p(w0) is a single constant added to every grid
    # cell, so it cancels exactly under the logsumexp normalization below and is
    # omitted (not silently dropped -- it is provably a no-op here).
    log_prob = logsumexp_norm(log_like)
    prob = np.exp(log_prob)
    return prob, r2_grid, np.asarray(Om0_mesh), np.asarray(w0_mesh)


# ---------------------------------------------------------------------------
# MCLMC sample loading + z -> physical (Om0, w0) via the model's own bijector.
# ---------------------------------------------------------------------------
def load_mclmc_cosmo_samples(model: LensModel):
    npz_path = os.path.join(MCLMC_DIR, "arrays.npz")
    with np.load(npz_path) as f:
        samples_z = np.asarray(f["samples_z"])  # (n_chains, n_steps, n_params)
    n_chains, n_steps, n_params = samples_z.shape
    if n_params != model.num_free_params:
        raise ValueError(
            f"arrays.npz samples_z has {n_params} params but the rebuilt LensModel "
            f"has {model.num_free_params}; the model-construction cell above no "
            "longer matches the run that produced this file."
        )
    idx_om0 = model.z_param_names.index("cosmo/Om0")
    idx_w0 = model.z_param_names.index("cosmo/w0")

    z_flat = jnp.asarray(samples_z.reshape(-1, n_params))
    x = model.bijector.forward(list(z_flat.T))  # same op as SamplerPosterior.z_to_x
    Om0_samples = np.asarray(x["cosmo/Om0"]).reshape(n_chains, n_steps)
    w0_samples = np.asarray(x["cosmo/w0"]).reshape(n_chains, n_steps)
    return Om0_samples, w0_samples, samples_z, idx_om0, idx_w0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sigma-frac", type=float, default=0.001,
                         help="sigma_r as a fraction of r2(truth) (default 0.1%%).")
    parser.add_argument("--n-grid", type=int, default=400)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    cosmo_model = w0waCDM_Cosmo(z_lens=Z_LENS, z_source_ref=Z_SOURCE1)
    r2_truth = float(np.squeeze(np.asarray(
        cosmo_model.deflection_ratio(
            Z_SOURCE2, H0=TRUTH["H0"], Om0=TRUTH["Om0"], k=TRUTH["k"],
            w0=TRUTH["w0"], wa=TRUTH["wa"],
        )
    )))
    sigma_r = args.sigma_frac * r2_truth

    print(f"[def_ratio_grid] z_lens={Z_LENS} z_source1(ref)={Z_SOURCE1} z_source2={Z_SOURCE2}")
    print(f"[def_ratio_grid] r2(truth) = {r2_truth:.10f}")
    print(f"[def_ratio_grid] sigma_frac = {args.sigma_frac}, sigma_r = {sigma_r:.3e}")

    prob, r2_grid, Om0_mesh, w0_mesh = make_prob_grid(
        cosmo_model, r2_truth, sigma_r, n_grid=args.n_grid
    )
    mass_levels = get_mass_levels(prob, levels=(0.68, 0.955, 0.997))
    print(f"[def_ratio_grid] r2 over grid box: min={r2_grid.min():.6f} max={r2_grid.max():.6f}")
    print(f"[def_ratio_grid] mass levels (68/95.5/99.7%): {mass_levels}")

    # A handful of r2 level contours, including EXACTLY r2(truth).
    r2_min, r2_max = float(r2_grid.min()), float(r2_grid.max())
    pad = 0.1 * (r2_max - r2_min)
    other_r2_levels = np.linspace(r2_min + pad, r2_max - pad, 4)
    r2_levels = np.sort(np.unique(np.concatenate([other_r2_levels, [r2_truth]])))

    # --- rebuild the notebook's LensModel (bijector + z_param_names only) ---
    model = build_model()
    print(f"[def_ratio_grid] rebuilt LensModel: num_free_params={model.num_free_params}")

    Om0_samples, w0_samples, samples_z, idx_om0, idx_w0 = load_mclmc_cosmo_samples(model)
    n_chains, n_steps = Om0_samples.shape

    # r2 at each MCLMC sample's (Om0, w0) -- vectorized, cheap.
    r2_samples_flat = cosmo_model.deflection_ratio(
        Z_SOURCE2, H0=TRUTH["H0"], Om0=Om0_samples.flatten(), k=TRUTH["k"],
        w0=w0_samples.flatten(), wa=TRUTH["wa"],
    )
    r2_samples = np.asarray(jnp.squeeze(jnp.asarray(r2_samples_flat), axis=0)).reshape(
        n_chains, n_steps
    )

    # rank-Rhat / bulk-ESS via the project's own SamplerPosterior (arviz-backed;
    # does not need a real ctx -- rhat/ess never touch it).
    sp = SamplerPosterior(None, samples_z)
    rhat_all, ess_all = sp.rhat, sp.ess
    rhat_om0, rhat_w0 = float(rhat_all[idx_om0]), float(rhat_all[idx_w0])
    ess_om0, ess_w0 = float(ess_all[idx_om0]), float(ess_all[idx_w0])

    # --- numeric summary (printed; also folded into the npz) ---
    print("\n[def_ratio_grid] per-chain Om0 summary (physical space):")
    for c in range(n_chains):
        print(f"  chain {c}: min={Om0_samples[c].min():.4f} "
              f"median={np.median(Om0_samples[c]):.4f} max={Om0_samples[c].max():.4f}")
    print(f"[def_ratio_grid] Om0 overall: min={Om0_samples.min():.4f} "
          f"max={Om0_samples.max():.4f}")
    print(f"[def_ratio_grid] r2(samples) range: min={r2_samples.min():.6f} "
          f"max={r2_samples.max():.6f}  (r2(truth)={r2_truth:.6f})")
    print(f"[def_ratio_grid] rank-Rhat: Om0={rhat_om0:.4f} w0={rhat_w0:.4f}")
    print(f"[def_ratio_grid] bulk-ESS:  Om0={ess_om0:.1f} w0={ess_w0:.1f} "
          f"(n_chains={n_chains}, n_steps={n_steps}, total draws={n_chains * n_steps})")

    # ---------------------------------------------------------------------
    # Figure 1: bare regenerated grid-search posterior (no MCLMC overlay).
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contour(Om0_mesh, w0_mesh, prob, levels=mass_levels, colors="k",
                     linestyles=[":", "--", "-"])
    ax.clabel(cs, fmt={mass_levels[0]: "99.7%", mass_levels[1]: "95.5%",
                        mass_levels[2]: "68%"}, fontsize=7)
    ax.scatter([TRUTH["Om0"]], [TRUTH["w0"]], marker="*", s=160, color="red",
               edgecolor="k", zorder=5, label="truth")
    ax.set_xlim(0, 1)
    ax.set_ylim(-2, -1 / 3)
    ax.set_xlabel(r"$\Omega_{m,0}$")
    ax.set_ylabel(r"$w_0$")
    ax.set_title(
        f"Deflection-ratio grid-search posterior (z_lens={Z_LENS}, "
        f"$\\sigma_r$={args.sigma_frac * 100:.2g}% of r2)"
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_BARE_PNG, dpi=150)
    plt.close(fig)
    print(f"[def_ratio_grid] wrote {OUT_BARE_PNG}")

    # ---------------------------------------------------------------------
    # Figure 2: overlay -- grid-search contours + r2 level contours + truth
    # + MCLMC cosmology samples colored by chain.
    # ---------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6.5))

    cs_r2 = ax.contour(Om0_mesh, w0_mesh, r2_grid, levels=r2_levels,
                        colors="gray", linewidths=0.8, linestyles="-", zorder=1)
    ax.clabel(cs_r2, fmt=lambda v: f"r2={v:.4f}", fontsize=6)

    cmap = plt.get_cmap("tab10")
    for c in range(n_chains):
        ax.scatter(Om0_samples[c], w0_samples[c], s=3, alpha=0.03,
                   color=cmap(c % 10), label=f"chain {c}", zorder=2,
                   rasterized=True)

    cs = ax.contour(Om0_mesh, w0_mesh, prob, levels=mass_levels, colors="k",
                     linestyles=[":", "--", "-"], linewidths=1.3, zorder=3)
    ax.clabel(cs, fmt={mass_levels[0]: "99.7%", mass_levels[1]: "95.5%",
                        mass_levels[2]: "68%"}, fontsize=7)

    ax.scatter([TRUTH["Om0"]], [TRUTH["w0"]], marker="*", s=200, color="red",
               edgecolor="k", zorder=5, label="truth")

    ax.set_xlim(0, 1)
    ax.set_ylim(-2, -1 / 3)
    ax.set_xlabel(r"$\Omega_{m,0}$")
    ax.set_ylabel(r"$w_0$")
    ax.set_title(
        "Grid-search posterior (black) + r2 contours (gray)\n"
        "+ MCLMC samples (by chain)",
        fontsize=11,
    )
    leg = ax.legend(loc="upper right", fontsize=6, ncol=2, markerscale=4, framealpha=0.85)
    for lh in leg.legend_handles:
        try:
            lh.set_alpha(1.0)
        except AttributeError:
            pass
    fig.tight_layout()
    fig.savefig(OUT_OVERLAY_PNG, dpi=170)
    plt.close(fig)
    print(f"[def_ratio_grid] wrote {OUT_OVERLAY_PNG}")

    # ---------------------------------------------------------------------
    # Save the probability grid + settings.
    # ---------------------------------------------------------------------
    np.savez(
        OUT_NPZ,
        Om0_grid=Om0_mesh[:, 0],
        w0_grid=w0_mesh[0, :],
        Om0_mesh=Om0_mesh,
        w0_mesh=w0_mesh,
        prob=prob,
        r2_grid=r2_grid,
        mass_levels=np.asarray(mass_levels),
        r2_levels=r2_levels,
        r2_truth=r2_truth,
        sigma_r=sigma_r,
        sigma_frac=args.sigma_frac,
        z_lens=Z_LENS,
        z_source1=Z_SOURCE1,
        z_source2=Z_SOURCE2,
        truth_H0=TRUTH["H0"],
        truth_Om0=TRUTH["Om0"],
        truth_w0=TRUTH["w0"],
        truth_k=TRUTH["k"],
        truth_wa=TRUTH["wa"],
        n_grid=args.n_grid,
        Om0_samples=Om0_samples,
        w0_samples=w0_samples,
        r2_samples=r2_samples,
        rhat_om0=rhat_om0,
        rhat_w0=rhat_w0,
        ess_om0=ess_om0,
        ess_w0=ess_w0,
    )
    print(f"[def_ratio_grid] wrote {OUT_NPZ}")


if __name__ == "__main__":
    main()
