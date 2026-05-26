"""
Reusable building blocks for Vela-based simulated-system modeling.

These functions are lifted (with light refactoring to remove notebook
closures) from ``sim_system_complex.ipynb`` so that batch scripts and
notebooks can share the same modeling code. Use them together with
``run_vela_modeling.py`` for non-interactive runs.
"""

from __future__ import annotations

import json
import os
import pickle
from os.path import expanduser

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp
from matplotlib import pyplot as plt

from gigalens.jax.inference import ModellingSequence
from gigalens.jax.model import BackwardProbModel
from gigalens.jax.profiles.light import sersic, shapelets
from gigalens.jax.profiles.mass import epl, shear
from gigalens.jax.simulator import LensSimulator
from gigalens.model import PhysicalModel
from gigalens.simulator import SimulatorConfig

from gigalens_research.plotting import (
    cornerplot_labels,
    cornerplot_posterior,
)

tfd = tfp.distributions


# ---------------------------------------------------------------------------
# Configuration defaults (match the notebook).
# ---------------------------------------------------------------------------
HOME = expanduser("~/")
DEFAULT_BACKGROUND_RMS = 0.002
DEFAULT_EXP_TIME = 2000.0
DEFAULT_NUM_PIX = 200
DEFAULT_SUPERSAMPLE = 1
DEFAULT_CAM = "12"
DEFAULT_FILTER_TAG = "a0.500_f814w"
DEFAULT_SOURCE_DIR_ROOT = os.path.join(
    HOME, "GIGALens-Code", "data", "vela_sources"
)
DEFAULT_SYSTEM_DIR_ROOT = os.path.join(
    HOME, "GIGALens-Code", "data", "vela_sim_systems"
)
DEFAULT_RESULT_DIR_ROOT = os.path.join(
    HOME, "GIGALens-Code", "results", "shapelets_systematics"
)
# Backwards-compatible name for callers that still pass a data root.
DEFAULT_SAVE_DIR_ROOT = DEFAULT_SYSTEM_DIR_ROOT


# ---------------------------------------------------------------------------
# Path helpers.
# ---------------------------------------------------------------------------
def system_save_dir(
    sim_num: str,
    rep: int,
    cam: str = DEFAULT_CAM,
    filter_tag: str = DEFAULT_FILTER_TAG,
    base_dir: str = DEFAULT_SAVE_DIR_ROOT,
) -> str:
    """Return the per-system input folder under ``data/vela_sim_systems``."""
    name = f"vela{sim_num}_cam{cam}_rep{int(rep):02d}_{filter_tag}"
    return os.path.join(base_dir, name)


def run_save_dir(
    system_dir: str,
    n_max: int,
    result_dir_root: str = DEFAULT_RESULT_DIR_ROOT,
) -> str:
    """Return the result folder for a given system and shapelet order."""
    return os.path.join(
        result_dir_root,
        os.path.basename(os.path.normpath(system_dir)),
        f"n_max{int(n_max):02d}",
    )


def source_plane_dir(
    sim_num: str,
    cam: str = DEFAULT_CAM,
    filter_tag: str = DEFAULT_FILTER_TAG,
    base_dir: str = DEFAULT_SOURCE_DIR_ROOT,
) -> str:
    name = f"vela{sim_num}_cam{cam}_{filter_tag}"
    return os.path.join(base_dir, name)


# ---------------------------------------------------------------------------
# I/O.
# ---------------------------------------------------------------------------
def load_vela_sim_system(
    sim_num: str,
    rep: int,
    cam: str = DEFAULT_CAM,
    num_pix: int = DEFAULT_NUM_PIX,
    supersample: int = DEFAULT_SUPERSAMPLE,
    filter_tag: str = DEFAULT_FILTER_TAG,
    source_dir_root: str = DEFAULT_SOURCE_DIR_ROOT,
    save_dir_root: str = DEFAULT_SAVE_DIR_ROOT,
):
    """Load the observed image, truth parameters, and the SimulatorConfig.

    This is the multi-rep version of the notebook's ``load_vela_sim_system``,
    parameterised by ``num_pix`` and ``supersample`` so that batch scripts
    can sweep over them.
    """
    src_dir = source_plane_dir(sim_num, cam=cam, filter_tag=filter_tag,
                               base_dir=source_dir_root)
    sys_dir = system_save_dir(sim_num, rep, cam=cam, filter_tag=filter_tag,
                              base_dir=save_dir_root)

    psf = np.load(os.path.join(src_dir, "psf.npy"))
    with open(os.path.join(src_dir, "metadata.json")) as f:
        meta = json.load(f)
    delta_pix = meta["instrument_pixel_scale_arcsec"]

    sim_config = SimulatorConfig(
        delta_pix=delta_pix, num_pix=num_pix, supersample=supersample, kernel=psf,
    )

    observed_img = jnp.load(os.path.join(sys_dir, "lens_img.npy"))
    with open(os.path.join(sys_dir, "true_params"), "rb") as f:
        true_params = pickle.load(f)

    return observed_img, true_params, sim_config, meta


# ---------------------------------------------------------------------------
# Priors / forward model construction.
# ---------------------------------------------------------------------------
def fixed_prior(profile_params):
    """Tight uniform prior pinned around `profile_params` (used to fix
    the lens parameters while leaving the source free)."""
    prior_dists = {}
    for key in profile_params:
        val = jnp.squeeze(profile_params[key])
        if len(val.shape) != 0:
            raise ValueError(
                f"Must have single parameter as truth (leaf shape is "
                f"{profile_params[key].shape})"
            )
        prior_dists[key] = tfd.Uniform(val - 1e-6, val + 1e-6)
    return tfd.JointDistributionNamed(prior_dists)


def vela_priors(use_shapelets: bool = True):
    """Return the (lens, lens_light, source_light) priors used in the
    notebook's ``vela_system_model``."""
    lens_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    theta_E=tfd.LogNormal(jnp.log(1.25), 0.4),
                    gamma=tfd.TruncatedNormal(2, 0.5, 1, 3),
                    e1=tfd.TruncatedNormal(0, 0.2, -0.5, 0.5),
                    e2=tfd.TruncatedNormal(0, 0.2, -0.5, 0.5),
                    center_x=tfd.Normal(0, 0.06),
                    center_y=tfd.Normal(0, 0.06),
                )
            ),
            tfd.JointDistributionNamed(
                dict(
                    gamma1=tfd.TruncatedNormal(0, 0.1, -0.5, 0.5),
                    gamma2=tfd.Normal(0, 0.1, -0.5, 0.5),
                )
            ),
        ]
    )
    lens_light_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    R_sersic=tfd.LogNormal(jnp.log(1.6), 0.25),
                    n_sersic=tfd.Uniform(0.5, 8),
                    e1=tfd.TruncatedNormal(0, 0.1, -0.2, 0.2),
                    e2=tfd.TruncatedNormal(0, 0.1, -0.2, 0.2),
                    center_x=tfd.Normal(0, 0.02),
                    center_y=tfd.Normal(0, 0.02),
                )
            )
        ]
    )

    if use_shapelets:
        source_light_prior = tfd.JointDistributionSequential(
            [
                tfd.JointDistributionNamed(
                    dict(
                        beta=tfd.LogNormal(jnp.log(0.7), 0.4),
                        center_x=tfd.Normal(0, 0.5),
                        center_y=tfd.Normal(0, 0.5),
                    )
                ),
            ]
        )
    else:
        # Notebook's "no shapelets" branch overrides lens_light_prior; keep
        # the same behaviour here for parity.
        lens_light_prior = tfd.JointDistributionSequential(
            [
                tfd.JointDistributionNamed(
                    dict(
                        R_sersic=tfd.LogNormal(jnp.log(1.6), 0.4),
                        n_sersic=tfd.Uniform(0.5, 8),
                        e1=tfd.TruncatedNormal(0, 0.1, -0.2, 0.2),
                        e2=tfd.TruncatedNormal(0, 0.1, -0.2, 0.2),
                        center_x=tfd.Normal(0, 0.5),
                        center_y=tfd.Normal(0, 0.5),
                    )
                )
            ]
        )
        source_light_prior = tfd.JointDistributionSequential(
            [
                tfd.JointDistributionNamed(
                    dict(
                        R_sersic=tfd.LogNormal(jnp.log(0.25), 0.25),
                        n_sersic=tfd.Uniform(0.5, 8),
                        e1=tfd.TruncatedNormal(0, 0.3, -0.5, 0.5),
                        e2=tfd.TruncatedNormal(0, 0.3, -0.5, 0.5),
                        center_x=tfd.Normal(0, 0.5),
                        center_y=tfd.Normal(0, 0.5),
                    )
                )
            ]
        )

    return tfd.JointDistributionSequential(
        [lens_prior, lens_light_prior, source_light_prior]
    )


def _src_model(use_shapelets: bool, n_max: int):
    return (
        shapelets.ShapeletsFast(n_max=n_max, use_lstsq=True, interpolate=False)
        if use_shapelets
        else sersic.SersicEllipse(use_lstsq=True)
    )


def vela_system_model(
    sim_config: SimulatorConfig,
    observed_img,
    background_rms: float = DEFAULT_BACKGROUND_RMS,
    exp_time: float = DEFAULT_EXP_TIME,
    use_shapelets: bool = True,
    n_max: int = 10,
):
    """Notebook's free-lens / free-source Vela model."""
    prior = vela_priors(use_shapelets=use_shapelets)
    src_model = _src_model(use_shapelets, n_max)
    phys_model = PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [sersic.SersicEllipse(use_lstsq=True)],
        [src_model],
    )
    lens_sim = LensSimulator(phys_model, sim_config, bs=1)
    prob_model = BackwardProbModel(
        prior, observed_img, background_rms=background_rms, exp_time=exp_time
    )
    model_seq = ModellingSequence(phys_model, prob_model, sim_config)
    return model_seq, lens_sim


def free_source_fixed_lens_model(
    sim_config: SimulatorConfig,
    observed_img,
    fixed_params,
    background_rms: float = DEFAULT_BACKGROUND_RMS,
    exp_time: float = DEFAULT_EXP_TIME,
    use_shapelets: bool = True,
    n_max: int = 10,
):
    """Lens-light + source free; lens parameters pinned to truth. Used to
    bootstrap MCLMC initialisation (notebook cell that produces
    ``shp_true``)."""
    lens_prior = tfd.JointDistributionSequential(
        [fixed_prior(p) for p in fixed_params[0]]
    )

    ie_less = fixed_params[1][0].copy()
    del ie_less["Ie"]
    lens_light_prior = tfd.JointDistributionSequential([fixed_prior(ie_less)])

    source_light_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    beta=tfd.LogNormal(jnp.log(0.7), 0.4),
                    center_x=tfd.Normal(0, 0.01),
                    center_y=tfd.Normal(0, 0.01),
                )
            ),
        ]
    )

    prior = tfd.JointDistributionSequential(
        [lens_prior, lens_light_prior, source_light_prior]
    )

    src_model = _src_model(use_shapelets, n_max)
    phys_model = PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [sersic.SersicEllipse(use_lstsq=True)],
        [src_model],
    )
    lens_sim = LensSimulator(phys_model, sim_config, bs=1)
    prob_model = BackwardProbModel(
        prior, observed_img, background_rms=background_rms, exp_time=exp_time
    )
    model_seq = ModellingSequence(phys_model, prob_model, sim_config)
    return model_seq, lens_sim


# ---------------------------------------------------------------------------
# Plotting / diagnostics (verbatim from the notebook, save-only).
# ---------------------------------------------------------------------------
def plot_diagnostics(
    debug_hist,
    num_burnin_steps: int,
    frac_tune1: float,
    frac_tune2: float,
    frac_tune3: float,
    save_dir: str,
):
    stage1 = int(frac_tune1 * num_burnin_steps)
    stage2 = int((frac_tune1 + frac_tune2) * num_burnin_steps)
    stage3 = int((frac_tune1 + frac_tune2 + frac_tune3) * num_burnin_steps)

    fig, axs = plt.subplots(5, 1, sharex=True)
    ax1, ax2, ax3, ax4, ax_last = axs
    fig.set_size_inches(10, 8)

    ax1.plot(debug_hist.step_size.T)
    ax1.set_title("Chain-Wise Step Size")
    ax1.set_ylabel("Step Size")

    ax2.plot(debug_hist.L.T)
    ax2.set_title("Chain-Wise L")
    ax2.set_ylabel("L")

    inverse_mass_matrix_hist = debug_hist.inverse_mass_matrix[0]
    vmapped_eigval = jax.vmap(lambda x: jnp.linalg.eig(x)[0])
    mass_mat_eigval = vmapped_eigval(inverse_mass_matrix_hist)
    min_eigval = jnp.min(mass_mat_eigval, axis=1)
    max_eigval = jnp.max(mass_mat_eigval, axis=1)
    mean_eigval = jnp.mean(mass_mat_eigval, axis=1)

    ax3.plot(min_eigval, label="Min", color="blue")
    ax3.plot(max_eigval, label="Max", color="red")
    ax3.plot(mean_eigval, label="Mean", color="black")
    ax3.legend()
    ax3.set_title("Covariance Eigenvalues")
    ax3.set_yscale("log")
    ax3.set_ylabel("Eigenvalue")

    smooth_kernel_size = 30
    kernel = np.ones(smooth_kernel_size) / smooth_kernel_size
    xi_chain = debug_hist.xi[8 % debug_hist.xi.shape[0]]
    xi_smoothed = np.convolve(xi_chain, kernel, mode="same")
    ax4.plot(xi_chain, alpha=0.5, color="blue")
    ax4.plot(xi_smoothed, alpha=1.0, color="blue")
    ax4.set_yscale("log")
    ax4.set_ylabel("xi for chain 0")
    ax4.axhline(1.0, color="black", linestyle="--")

    ax_last.set_xlabel("Step")
    ax_last.set_title("Nans?")
    ax_last.imshow(
        debug_hist.nonan[:, :stage2], aspect="auto", interpolation="none",
        cmap="RdYlGn",
    )

    for ax in axs:
        ax.axvline(stage1, color="red", linestyle="--")
        ax.axvline(stage2, color="blue", linestyle="--")
        ax.axvline(stage3, color="green", linestyle="--")

    plt.savefig(os.path.join(save_dir, "mclmc_diagnostics.png"))
    plt.close(fig)


def cornerplot_all(mclmc_samples, qz, prob_model, save_dir, true_params_shp):
    dim = mclmc_samples.shape[-1]
    run_key = jax.random.key(0)
    samples = mclmc_samples[:, :, :].reshape(-1, dim)
    MCMC_x = prob_model.bij.forward(list(samples.T))

    SVI_samples = qz.sample((1000,), run_key)
    SVI_x = prob_model.bij.forward(list(SVI_samples.T))

    plot_params = cornerplot_labels(MCMC_x)

    n_samp_MCMC = MCMC_x[0][0]["e1"].shape[0]
    rand_idx_MCMC = np.random.choice(
        np.arange(n_samp_MCMC), size=(20000,), replace=True
    )
    fig = cornerplot_posterior(
        jax.tree.map(lambda x: x[rand_idx_MCMC], MCMC_x),
        color="black", plot_params=plot_params, truth=true_params_shp,
    )
    cornerplot_posterior(SVI_x, fig=fig, color="blue", plot_params=plot_params)
    plt.savefig(os.path.join(save_dir, "cornerplot.png"))
    plt.close(fig)


# ---------------------------------------------------------------------------
# Z-score helper, identical to the notebook's stdev_calc.
# ---------------------------------------------------------------------------
def stdev_calc(truth, med, sig_low, sig_up):
    sigma = jnp.where(truth > med, sig_up - med, med - sig_low)
    return (truth - med) / sigma


def zscore_summary(mclmc_samples, prob_model, true_params_shp, true_params):
    """Return a (printable) dict mapping mass-parameter key to a
    'predicted | true | z-score' string, mirroring the notebook print."""
    med_x = prob_model.bij.forward(
        list(jnp.median(mclmc_samples, axis=(0, 1)).T)
    )
    sigma_low = prob_model.bij.forward(
        list(jnp.quantile(mclmc_samples, q=0.159, axis=(0, 1)).T)
    )
    sigma_up = prob_model.bij.forward(
        list(jnp.quantile(mclmc_samples, q=1 - 0.159, axis=(0, 1)).T)
    )
    sigma = jax.tree.map(stdev_calc, true_params_shp, med_x, sigma_low, sigma_up)

    summary = jax.tree.map(
        lambda x, y, z: (
            f"{float(jnp.squeeze(x)):.4f} | {float(jnp.squeeze(y)):.4f} "
            f"| {float(jnp.squeeze(z)):.4f}"
        ),
        med_x[0], true_params[0], sigma[0],
    )
    return summary


def build_true_params_shp(true_params, shp_true):
    """Combine the original truth with the MAP-fit shapelet centre/beta to
    produce ``true_params_shp`` (the same structure the MCLMC samples live
    in)."""
    lens_light_no_Ie = true_params[1][0].copy()
    del lens_light_no_Ie["Ie"]
    return [true_params[0], [lens_light_no_Ie], [shp_true]]


__all__ = [
    "DEFAULT_BACKGROUND_RMS",
    "DEFAULT_EXP_TIME",
    "DEFAULT_NUM_PIX",
    "DEFAULT_SUPERSAMPLE",
    "DEFAULT_CAM",
    "DEFAULT_FILTER_TAG",
    "DEFAULT_SOURCE_DIR_ROOT",
    "DEFAULT_SYSTEM_DIR_ROOT",
    "DEFAULT_RESULT_DIR_ROOT",
    "DEFAULT_SAVE_DIR_ROOT",
    "system_save_dir",
    "run_save_dir",
    "source_plane_dir",
    "load_vela_sim_system",
    "vela_priors",
    "vela_system_model",
    "free_source_fixed_lens_model",
    "fixed_prior",
    "plot_diagnostics",
    "cornerplot_all",
    "stdev_calc",
    "zscore_summary",
    "build_true_params_shp",
]
