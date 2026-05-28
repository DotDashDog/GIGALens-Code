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
  inference :class:`~gigalens.jax.inference.ModellingSequence` using
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

    lens_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            theta_E=tfd.LogNormal(jnp.log(1.25), 0.4),
            gamma=tfd.TruncatedNormal(2.0, 0.5, 1.0, 3.0),
            e1=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
            e2=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
            center_x=tfd.Normal(0.0, 0.06),
            center_y=tfd.Normal(0.0, 0.06),
        )),
        tfd.JointDistributionNamed(dict(
            gamma1=tfd.TruncatedNormal(0.0, 0.1, -0.5, 0.5),
            gamma2=tfd.Normal(0.0, 0.1),
        )),
    ])
    lens_light_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            R_sersic=tfd.LogNormal(jnp.log(1.6), 0.25),
            n_sersic=tfd.Uniform(0.5, 8.0),
            e1=tfd.TruncatedNormal(0.0, 0.1, -0.2, 0.2),
            e2=tfd.TruncatedNormal(0.0, 0.1, -0.2, 0.2),
            center_x=tfd.Normal(0.0, 0.02),
            center_y=tfd.Normal(0.0, 0.02),
        )),
    ])
    if use_shapelets:
        source_prior = tfd.JointDistributionSequential([
            tfd.JointDistributionNamed(dict(
                beta=tfd.LogNormal(jnp.log(0.7), 0.4),
                center_x=tfd.Normal(0.0, 0.5),
                center_y=tfd.Normal(0.0, 0.5),
            )),
        ])
    else:
        source_prior = tfd.JointDistributionSequential([
            tfd.JointDistributionNamed(dict(
                R_sersic=tfd.LogNormal(jnp.log(0.25), 0.25),
                n_sersic=tfd.Uniform(0.5, 8.0),
                e1=tfd.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
                e2=tfd.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
                center_x=tfd.Normal(0.0, 0.5),
                center_y=tfd.Normal(0.0, 0.5),
            )),
        ])
    return tfd.JointDistributionSequential([lens_prior, lens_light_prior, source_prior])


# ---------------------------------------------------------------------------
# Inference builder
# ---------------------------------------------------------------------------


@register_inference_builder("epl_shear_sersic_shapelets")
def build_epl_shear_sersic_shapelets(system: Any, **kwargs) -> Any:
    """Build the Vela BackwardProbModel + ModellingSequence.

    Uses ``ShapeletsFast(n_max=n_max, use_lstsq=True)`` for the source and
    ``BackwardProbModel`` (fixed error map from the observed image).

    Kwargs: ``n_max`` (default 10), ``use_shapelets`` (default True).
    """
    import jax.numpy as jnp
    from gigalens.jax.inference import ModellingSequence
    from gigalens.jax.model import BackwardProbModel
    from gigalens.jax.profiles.light import sersic, shapelets
    from gigalens.jax.profiles.mass import epl, shear
    from gigalens.model import PhysicalModel

    n_max = int(kwargs.get("n_max", 10))
    use_shapelets = bool(kwargs.get("use_shapelets", True))

    prior = vela_inference_prior(use_shapelets=use_shapelets)

    if use_shapelets:
        src_model = shapelets.ShapeletsFast(n_max=n_max, use_lstsq=True, interpolate=False)
    else:
        src_model = sersic.SersicEllipse(use_lstsq=True)

    phys_model = PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [sersic.SersicEllipse(use_lstsq=True)],
        [src_model],
    )
    prob_model = BackwardProbModel(
        prior,
        jnp.asarray(system.observed_image),
        background_rms=system.background_rms,
        exp_time=system.exp_time,
    )
    return ModellingSequence(phys_model, prob_model, system.sim_config)


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
    num_pix = int(extra.get("num_pix", 200))
    supersample = int(extra.get("supersample", 1))
    background_rms = float(extra.get("background_rms", 0.002))
    exp_time = float(extra.get("exp_time", 2000.0))

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
