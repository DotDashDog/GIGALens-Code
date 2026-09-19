"""Vela shapelets experiment: EPL + Shear + SersicEllipse lens + ShapeletsFast source.

This module registers:

- ``"vela_existing"`` generator — adapts the existing
  ``data/vela_sim_systems/`` directories into the framework's
  :class:`~system.System` format (reads ``lens_img.npy`` + pickled
  ``true_params``).  The generation notebook
  (``experiments/vela_sim_systems/lens_vela_system.ipynb``) remains the
  canonical simulation tool; this adapter brings existing results into the
  framework without re-simulating.
- ``"epl_shear_sersic_shapelets"`` inference builder — builds the Vela
  inference scene :class:`~gigalens.jax.scene_prob_model.ProbModel` using
  ``BackwardProbModel`` (lstsq amplitudes) and ``ShapeletsFast`` source.
- ``"map_bootstrap_mclmc"`` pipeline builder (registered in ``pipelines.py``).

The sweep axis for this experiment is typically ``n_max`` (shapelet order).
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from gigalens_research.simtests.registry import (
    register_generator,
    register_inference_builder,
)


# ---------------------------------------------------------------------------
# Default Vela paths
# ---------------------------------------------------------------------------

_HOME = os.path.expanduser("~")
_DEFAULT_SYSTEM_DIR_ROOT = os.path.join(_HOME, "GIGALens-Code", "data", "vela_sim_systems")
_DEFAULT_SOURCE_DIR_ROOT = os.path.join(_HOME, "GIGALens-Code", "data", "vela_sources")

_DEFAULT_VELA_IDS = ["01", "03", "04", "07", "08", "10", "15", "21", "22", "23", "25", "26"]
_DEFAULT_CAM = "12"
_DEFAULT_FILTER_TAG = "a0.500_f814w"


# ---------------------------------------------------------------------------
# Vela inference prior
# ---------------------------------------------------------------------------


def vela_inference_prior(use_shapelets: bool = True):
    """Inference prior for the Vela shapelets experiment.

    Mirrors ``vela_utilities.vela_priors()``.
    """
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions

    lens_prior = tfd.JointDistributionNamed({
        '0': tfd.JointDistributionNamed(dict(
            theta_E=tfd.LogNormal(jnp.log(1.25), 0.4),
            gamma=tfd.TruncatedNormal(2.0, 0.5, 1.0, 3.0),
            e1=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
            e2=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
            center_x=tfd.Normal(0.0, 0.06),
            center_y=tfd.Normal(0.0, 0.06),
        )),
        '1': tfd.JointDistributionNamed(dict(
            gamma1=tfd.TruncatedNormal(0.0, 0.1, -0.5, 0.5),
            gamma2=tfd.Normal(0.0, 0.1),
        )),
    })
    lens_light_prior = tfd.JointDistributionNamed({
        '0': tfd.JointDistributionNamed(dict(
            R_sersic=tfd.LogNormal(jnp.log(1.6), 0.25),
            n_sersic=tfd.Uniform(0.5, 8.0),
            e1=tfd.TruncatedNormal(0.0, 0.1, -0.2, 0.2),
            e2=tfd.TruncatedNormal(0.0, 0.1, -0.2, 0.2),
            center_x=tfd.Normal(0.0, 0.02),
            center_y=tfd.Normal(0.0, 0.02),
        )),
    })
    if use_shapelets:
        source_prior = tfd.JointDistributionNamed({
            '0': tfd.JointDistributionNamed(dict(
                beta=tfd.LogNormal(jnp.log(SHAPELET_BETA_PRIOR["median_arcsec"]), SHAPELET_BETA_PRIOR["log_sigma"]),
                center_x=tfd.Normal(0.0, 0.5),
                center_y=tfd.Normal(0.0, 0.5),
            )),
        })
    else:
        source_prior = tfd.JointDistributionNamed({
            '0': tfd.JointDistributionNamed(dict(
                R_sersic=tfd.LogNormal(jnp.log(0.25), 0.4),
                n_sersic=tfd.Uniform(0.5, 8.0),
                e1=tfd.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
                e2=tfd.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
                center_x=tfd.Normal(0.0, 0.5),
                center_y=tfd.Normal(0.0, 0.5),
            )),
        })
    return tfd.JointDistributionNamed({
        'lens_mass': lens_prior,
        'lens_light': lens_light_prior,
        'source_light': source_prior,
    })


# ---------------------------------------------------------------------------
# Inference builder
# ---------------------------------------------------------------------------


# Fit-side lens-light family (user decision 2026-09-18, DC-5): the same core-Sersic family as
# the generator's truth, with NO truth knowledge — R_b free under a broad log prior (median
# 1% of the R_e prior median = 0.016", 1 dex), gamma U(0, 0.5), alpha fixed at 5 (as in the
# truth; unresolvable). The pure Sersic is the R_b -> 0 limit, so low-n lenses return an
# upper bound on R_b. "sersic" keeps the pre-2026-09-18 model for the recorded runs.
LENS_LIGHT_PROFILES = ("sersic", "core_sersic")
CORE_SERSIC_FIT_PRIOR = {"Rb_median_arcsec": 0.016, "Rb_log_sigma": 2.302585, "gamma_high": 0.5, "alpha": 5.0}


def _vela_scene_lens_priors(lens_light_profile: str = "sersic"):
    """Shared scene priors for EPL + Shear mass and the lens light (per-param dicts;
    fresh objects) and the lens-light PROFILE object. Mirrors ``vela_inference_prior``'s
    lens/lens-light blocks, which are identical across the vela shapelets/sersiclets
    builders. ``lens_light_profile``: "sersic" (SersicEllipse) or "core_sersic"
    (CoreSersic with :data:`CORE_SERSIC_FIT_PRIOR`)."""
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    from gigalens.jax.profiles.light import sersic
    tfd = tfp.distributions
    if lens_light_profile not in LENS_LIGHT_PROFILES:
        raise ValueError(f"lens_light_profile must be one of {LENS_LIGHT_PROFILES}; got {lens_light_profile!r}.")
    epl_p = dict(
        theta_E=tfd.LogNormal(jnp.log(1.25), 0.4),
        gamma=tfd.TruncatedNormal(2.0, 0.5, 1.0, 3.0),
        e1=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
        center_x=tfd.Normal(0.0, 0.06),
        center_y=tfd.Normal(0.0, 0.06),
    )
    shear_p = dict(
        gamma1=tfd.TruncatedNormal(0.0, 0.1, -0.5, 0.5),
        gamma2=tfd.Normal(0.0, 0.1),
    )
    lens_light_p = dict(
        R_sersic=tfd.LogNormal(jnp.log(1.6), 0.25),
        n_sersic=tfd.Uniform(0.5, 8.0),
        e1=tfd.TruncatedNormal(0.0, 0.1, -0.2, 0.2),
        e2=tfd.TruncatedNormal(0.0, 0.1, -0.2, 0.2),
        center_x=tfd.Normal(0.0, 0.02),
        center_y=tfd.Normal(0.0, 0.02),
    )
    if lens_light_profile == "core_sersic":
        c = CORE_SERSIC_FIT_PRIOR
        lens_light_p.update(
            Rb=tfd.LogNormal(jnp.log(c["Rb_median_arcsec"]), c["Rb_log_sigma"]),
            gamma=tfd.Uniform(0.0, c["gamma_high"]),
            alpha=float(c["alpha"]),   # constant
        )
        profile = sersic.CoreSersic(use_lstsq=True)
    else:
        profile = sersic.SersicEllipse(use_lstsq=True)
    return epl_p, shear_p, lens_light_p, profile


def make_image_data(system: Any, adaptive: Any = None, mask_disk: Any = None, **common):
    """The inference dataset for a Vela system: a plain ``ImageData`` on the dataset's
    uniform ``inference_supersample``, or — when ``adaptive`` (dict) is given — an
    ``AdaptiveImageData`` whose factor map is derived from the observed image
    (``driver`` "curvature" or "snr", remaining keys forwarded to that driver) with the
    config's uniform supersample forced to 1 (the factor map IS the quadrature).
    Curvature needs ``psf_sigma`` [native px] explicitly: there is no honest default
    (the flux-moment estimate of a wide empirical kernel over-estimates the core width
    and under-corrects the finest LoG scale). Shared by every Vela builder so one
    campaign key (``adaptive:``) means the same thing for every source model.
    ``mask_disk`` (dict, optional): ``{"radius_pix": r, "centre": "peak" | [row, col]}``
    drops the pixels within ``r`` native px of the centre (``"peak"`` = the brightest
    observed pixel — data-driven, no truth) from the likelihood (``mask`` False there).
    Used by the lens-cusp quadrature falsifier (DC-3, 2026-09-18); the excluded pixels are
    counted in the printout."""
    import dataclasses
    import numpy as np
    import jax.numpy as jnp
    from gigalens.jax.scene_prob_model import ImageData
    img = jnp.asarray(system.observed_image)
    if mask_disk:
        md = dict(mask_disk)
        r = float(md.pop("radius_pix")); centre = md.pop("centre", "peak")
        if md:
            raise ValueError(f"make_image_data: unknown mask_disk keys {sorted(md)}.")
        obs = np.asarray(system.observed_image)
        if centre == "peak":
            row, col = np.unravel_index(int(np.argmax(obs)), obs.shape)
        else:
            row, col = float(centre[0]), float(centre[1])
        yy, xx = np.indices(obs.shape)
        m = np.hypot(yy - row, xx - col) > r
        if "mask" in common and common["mask"] is not None:
            m = m & np.asarray(common["mask"], bool)
        common["mask"] = jnp.asarray(m)
        print(f"[{system.system_id}] mask_disk: {int((~m).sum())} px within {r} px of ({row}, {col}) dropped from the likelihood")
    common.setdefault("background_rms", system.background_rms)
    common.setdefault("exp_time", system.exp_time)
    common.setdefault("sees", "all")
    if not adaptive:
        return ImageData(img, system.sim_config, **common)
    from gigalens.jax.experimental.adaptive_supersample import AdaptiveImageData
    a = dict(adaptive)
    driver = a.pop("driver", None)
    if driver not in ("curvature", "snr"):
        raise ValueError(f"make_image_data: adaptive.driver must be 'curvature' or 'snr'; got {driver!r}.")
    cfg1 = dataclasses.replace(system.sim_config, supersample=1)
    if driver == "curvature":
        if "psf_sigma" not in a:
            raise ValueError("make_image_data: adaptive.psf_sigma [native px] is required for the "
                             "curvature driver (no honest default).")
        ds = AdaptiveImageData(img, cfg1, driver="curvature", curvature_kwargs=a, **common)
    else:
        ds = AdaptiveImageData(img, cfg1, driver="snr", **a, **common)
    print(f"[{system.system_id}] adaptive quadrature ({driver}): {ds.adaptive_grid!r}")
    return ds


# Shapelet scale prior, LogNormal(log median, log_sigma) in arcsec. Re-centred 2026-09-18
# (user request): the old LogNormal(0.7", 0.4) sat 4 prior-sigma above the beta the vela22
# fits wanted (0.13-0.15" at every n_max; beta ~ 0.55 R50) and pulled beta up by 0.1-0.35
# posterior sigma. The 10 kept VELA sources have R50 0.16-0.72" (median 0.33"), so beta is
# expected in 0.09-0.4"; median 0.2" with log-sigma 0.7 puts the central 98% at 0.04-1.0".
SHAPELET_BETA_PRIOR = {"median_arcsec": 0.2, "log_sigma": 0.7}


def _beta_prior(kwargs):
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    bp = {**SHAPELET_BETA_PRIOR, **dict(kwargs.get("beta_prior") or {})}
    unknown = set(bp) - set(SHAPELET_BETA_PRIOR)
    if unknown:
        raise ValueError(f"beta_prior: unknown keys {sorted(unknown)}; allowed {sorted(SHAPELET_BETA_PRIOR)}.")
    return tfp.distributions.LogNormal(jnp.log(float(bp["median_arcsec"])), float(bp["log_sigma"]))


@register_inference_builder("epl_shear_sersic_shapelets")
def build_epl_shear_sersic_shapelets(system: Any, **kwargs) -> Any:
    """Build the SCENE ``ProbModel`` for the Vela shapelets fit (G1b).

    Scene ``LensModel`` (EPL+Shear mass + Sérsic lens light on plane 0; a Shapelets
    source — or a Sérsic source when ``use_shapelets=False`` — on plane 1, lstsq amps)
    + ``Dataset`` + ``ProbModel(mode="lstsq")``, returned directly. Public
    signature unchanged.

    Kwargs: ``n_max`` (REQUIRED when ``use_shapelets=True``; no default — it sets
    the source model complexity), ``use_shapelets`` (default True), ``adaptive``
    (optional dict, see :func:`make_image_data`; default: uniform quadrature at the
    dataset's ``inference_supersample``), ``beta_prior`` (optional dict overriding
    :data:`SHAPELET_BETA_PRIOR`: ``median_arcsec``, ``log_sigma``), ``mask_disk`` (see
    :func:`make_image_data`).
    """
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    from gigalens.jax.profiles.light import sersic, shapelets
    from gigalens.jax.profiles.mass import epl, shear
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_prob_model import ImageData, ProbModel
    tfd = tfp.distributions

    use_shapelets = bool(kwargs.get("use_shapelets", True))
    if use_shapelets and "n_max" not in kwargs:
        raise TypeError(
            "build_epl_shear_sersic_shapelets: 'n_max' is required when "
            "use_shapelets=True (no default; it sets the source model complexity)."
        )
    n_max = int(kwargs["n_max"]) if use_shapelets else None  # physics-default-ok: n_max unused when use_shapelets=False; required-check above

    epl_p, shear_p, lens_light_p, lens_light_profile = _vela_scene_lens_priors(kwargs.get("lens_light_profile", "sersic"))
    if use_shapelets:
        src_profile = shapelets.Shapelets(n_max=n_max, use_lstsq=True, interpolate=False)
        source_p = dict(
            beta=_beta_prior(kwargs),
            center_x=tfd.Normal(0.0, 0.5),
            center_y=tfd.Normal(0.0, 0.5),
        )
    else:
        src_profile = sersic.SersicEllipse(use_lstsq=True)
        source_p = dict(
            R_sersic=tfd.LogNormal(jnp.log(0.25), 0.4),
            n_sersic=tfd.Uniform(0.5, 8.0),
            e1=tfd.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
            e2=tfd.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
            center_x=tfd.Normal(0.0, 0.5),
            center_y=tfd.Normal(0.0, 0.5),
        )

    model = LensModel([
        Plane(mass=[Component(epl.EPL(50), epl_p), Component(shear.Shear(), shear_p)],
              light=[Component(lens_light_profile, lens_light_p)]),
        Plane(deflection_ratio=1.0,
              light=[Component(src_profile, source_p)]),
    ])
    ds = make_image_data(system, kwargs.get("adaptive"), mask_disk=kwargs.get("mask_disk"))
    prob_model = ProbModel(model, ds, mode="lstsq")
    return prob_model


# ---------------------------------------------------------------------------
# Generator: vela_existing
# ---------------------------------------------------------------------------


@register_generator("vela_existing")
def generate_vela_existing(spec: Any, dataset_dir: str, seed: int) -> None:
    """Adapt existing ``data/vela_sim_systems/`` directories into System format.

    Reads each ``vela{id}_cam{cam}_rep{rep:02d}_{filter_tag}/`` directory under
    ``system_dir_root`` and the corresponding source dir under
    ``source_dir_root``.  Does NOT re-simulate; the Vela images are unchanged.

    Required / optional extra keys in ``DatasetSpec.extra``:

    ``vela_ids`` (default: all 12 standard IDs),
    ``reps`` (default: 5, so reps 0–4),
    ``cam`` (default: ``"12"``),
    ``filter_tag`` (default: ``"a0.500_f814w"``),
    ``system_dir_root`` (default: ``~/GIGALens-Code/data/vela_sim_systems``),
    ``source_dir_root`` (default: ``~/GIGALens-Code/data/vela_sources``),
    ``num_pix`` (default: 200), ``supersample`` (default: 1),
    ``background_rms`` (default: 0.002), ``exp_time`` (default: 2000).
    """
    from gigalens_research.simtests.system import from_vela_dir, write_manifest
    from gigalens_research.simtests.generate import _hash_dataset

    extra = dict(spec.extra)
    vela_ids = list(extra.get("vela_ids", _DEFAULT_VELA_IDS))
    n_reps = int(extra.get("reps", 5))
    cam = str(extra.get("cam", _DEFAULT_CAM))
    filter_tag = str(extra.get("filter_tag", _DEFAULT_FILTER_TAG))
    sys_root = os.path.expanduser(str(extra.get("system_dir_root", _DEFAULT_SYSTEM_DIR_ROOT)))
    src_root = os.path.expanduser(str(extra.get("source_dir_root", _DEFAULT_SOURCE_DIR_ROOT)))
    num_pix = int(extra.get("num_pix", 200))  # physics-default-ok: documented vela_existing generation default, persisted to meta.json
    supersample = int(extra.get("supersample", 1))  # physics-default-ok: documented vela_existing generation default, persisted to meta.json
    background_rms = float(extra.get("background_rms", 0.002))  # physics-default-ok: documented vela_existing generation default, persisted to meta.json
    exp_time = float(extra.get("exp_time", 2000.0))  # physics-default-ok: documented vela_existing generation default, persisted to meta.json
    # Numerics: gigalens defaults to float64 going forward (see docs/project-standards.md).
    # Persisted to meta.json so run/plot honour it; requires jax_enable_x64 (set by the
    # simtests package import). Override in the dataset YAML if you need float32.
    likelihood_precision = extra.get("likelihood_precision", "float64")
    conv_precision = extra.get("conv_precision", None)

    system_ids = []
    n_adapted = 0
    n_missing = 0

    for vela_id in vela_ids:
        source_dir = os.path.join(src_root, f"vela{vela_id}_cam{cam}_{filter_tag}")
        # Read delta_pix from source metadata
        delta_pix = _read_delta_pix(source_dir)

        for rep in range(n_reps):
            system_dir = os.path.join(
                sys_root, f"vela{vela_id}_cam{cam}_rep{rep:02d}_{filter_tag}"
            )
            if not os.path.isdir(system_dir):
                print(f"[vela_existing] missing: {system_dir} — skipping")
                n_missing += 1
                continue

            system_id = f"vela{vela_id}_cam{cam}_rep{rep:02d}"
            try:
                sys = from_vela_dir(
                    system_dir=system_dir,
                    source_dir=source_dir,
                    system_id=system_id,
                    delta_pix=delta_pix,
                    num_pix=num_pix,
                    supersample=supersample,
                    background_rms=background_rms,
                    exp_time=exp_time,
                    likelihood_precision=likelihood_precision,
                    conv_precision=conv_precision,
                )
                sys.save(dataset_dir)
                system_ids.append(system_id)
                n_adapted += 1
            except Exception as exc:
                print(f"[vela_existing] failed for {system_id}: {exc}")
                n_missing += 1

    print(f"[vela_existing] adapted {n_adapted} systems, {n_missing} missing/failed.")

    dataset_hash = _hash_dataset(dataset_dir, system_ids)
    write_manifest(
        dataset_dir,
        generator="vela_existing",
        seed=seed,
        system_ids=system_ids,
        dataset_hash=dataset_hash,
        extra={"system_dir_root": sys_root, "source_dir_root": src_root},
    )


def _read_delta_pix(source_dir: str) -> float:
    """Read ``instrument_pixel_scale_arcsec`` from the Vela source metadata.json."""
    meta_path = os.path.join(source_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return float(meta["instrument_pixel_scale_arcsec"])
    return 0.03  # fallback: HST ACS F814W
