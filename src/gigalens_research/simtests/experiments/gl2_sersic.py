"""GL2 Sérsic experiment: EPL + Shear + SersicEllipse lens + SersicEllipse source.

This module registers:

- ``"parametric"`` generator — generic truth-model + sim-prior generation
  (used when generating new GL2-style systems from scratch).
- ``"gl2_existing"`` generator — adapts the existing ``SystemSaves/``
  ``100SystemsStandard80px.npz`` + ``100SystemsStandardParams.yaml`` on disk
  into the framework's :class:`~system.System` format without re-simulating.
- ``"epl_shear_sersic_sersic"`` inference builder — builds the standard GL2
  :class:`~gigalens.jax.inference.ModellingSequence` with the GIGALens paper
  inference prior (:func:`gl2_inference_prior`).
- ``"map_svi_hmc"`` pipeline builder (already registered in ``pipelines.py``).

Priors
------
``gl2_simulation_prior()`` — the tighter draw distribution used in
``attic/Linus-FourSim.ipynb`` (the Simulation column of paper Eq. 8).

``gl2_inference_prior()`` — the broader inference prior from the GIGALens paper
(the Prior column of paper Eq. 8); identical to ``make_default_prior()`` in
``experiments/hundred_systems_GL2/setup.py``.
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np

from gigalens_research.simtests.registry import (
    register_generator,
    register_inference_builder,
)


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------


def gl2_simulation_prior():
    """Tight simulation prior (paper Eq. 8, Simulation column).

    Mirrors the prior defined inline in ``attic/Linus-FourSim.ipynb``.
    """
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions

    lens_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            theta_E=tfd.LogNormal(jnp.log(1.25), 0.25),
            gamma=tfd.TruncatedNormal(2.0, 0.25, 1.0, 3.0),
            e1=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
            e2=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
            center_x=tfd.Normal(0.0, 0.03),
            center_y=tfd.Normal(0.0, 0.03),
        )),
        tfd.JointDistributionNamed(dict(
            gamma1=tfd.Normal(0.0, 0.05),
            gamma2=tfd.Normal(0.0, 0.05),
        )),
    ])
    lens_light_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            R_sersic=tfd.LogNormal(jnp.log(1.6), 0.15),
            n_sersic=tfd.Uniform(2.0, 6.0),
            e1=tfd.TruncatedNormal(0.0, 0.05, -0.15, 0.15),
            e2=tfd.TruncatedNormal(0.0, 0.05, -0.15, 0.15),
            center_x=tfd.Normal(0.0, 0.01),
            center_y=tfd.Normal(0.0, 0.01),
            Ie=tfd.LogNormal(jnp.log(300.0), 0.3),
        )),
    ])
    source_light_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            R_sersic=tfd.LogNormal(jnp.log(0.25), 0.25),
            n_sersic=tfd.Uniform(0.5, 4.0),
            e1=tfd.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
            e2=tfd.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
            center_x=tfd.Normal(0.0, 0.25),
            center_y=tfd.Normal(0.0, 0.25),
            Ie=tfd.LogNormal(jnp.log(150.0), 0.5),
        )),
    ])
    return tfd.JointDistributionSequential([lens_prior, lens_light_prior, source_light_prior])


def gl2_inference_prior():
    """Broad inference prior from GIGALens paper Eq. 8 (Prior / b column).

    This is the exact ``make_default_prior()`` from
    ``experiments/hundred_systems_GL2/setup.py``, moved here as a registered
    builder so experiment scripts no longer need to import the broken
    ``helpers.py``.
    """
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions

    lens_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            theta_E=tfd.LogNormal(jnp.log(1.25), 0.4),
            gamma=tfd.TruncatedNormal(2.0, 0.5, 1.0, 3.0),
            e1=tfd.Normal(0.0, 0.2),
            e2=tfd.Normal(0.0, 0.2),
            center_x=tfd.Normal(0.0, 0.06),
            center_y=tfd.Normal(0.0, 0.06),
        )),
        tfd.JointDistributionNamed(dict(
            gamma1=tfd.Normal(0.0, 0.1),
            gamma2=tfd.Normal(0.0, 0.1),
        )),
    ])
    lens_light_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            R_sersic=tfd.LogNormal(jnp.log(1.6), 0.25),
            n_sersic=tfd.Uniform(0.5, 8.0),
            e1=tfd.TruncatedNormal(0.0, 0.1, -0.15, 0.15),
            e2=tfd.TruncatedNormal(0.0, 0.1, -0.15, 0.15),
            center_x=tfd.Normal(0.0, 0.02),
            center_y=tfd.Normal(0.0, 0.02),
            Ie=tfd.LogNormal(jnp.log(300.0), 0.5),
        )),
    ])
    source_light_prior = tfd.JointDistributionSequential([
        tfd.JointDistributionNamed(dict(
            R_sersic=tfd.LogNormal(jnp.log(0.25), 0.25),
            n_sersic=tfd.Uniform(0.5, 8.0),
            e1=tfd.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
            e2=tfd.TruncatedNormal(0.0, 0.3, -0.5, 0.5),
            center_x=tfd.Normal(0.0, 0.5),
            center_y=tfd.Normal(0.0, 0.5),
            Ie=tfd.LogNormal(jnp.log(150.0), 0.9),
        )),
    ])
    return tfd.JointDistributionSequential([lens_prior, lens_light_prior, source_light_prior])


def _gl2_phys_model():
    from gigalens.jax.profiles.light import sersic
    from gigalens.jax.profiles.mass import epl, shear
    from gigalens.model import PhysicalModel
    return PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [sersic.SersicEllipse(use_lstsq=False)],
        [sersic.SersicEllipse(use_lstsq=False)],
    )


def _gl2_psf(srcdir: str | None = None) -> np.ndarray:
    """Load the gigalens bundled HST PSF."""
    if srcdir is None:
        srcdir = os.path.join(os.path.expanduser("~"), "gigalens", "src")
    psf_path = os.path.join(srcdir, "gigalens", "assets", "psf.npy")
    return np.load(psf_path).astype(np.float32)


# ---------------------------------------------------------------------------
# Inference builder
# ---------------------------------------------------------------------------


@register_inference_builder("epl_shear_sersic_sersic")
def build_epl_shear_sersic_sersic(system: Any, **kwargs) -> Any:
    """Build the standard GL2 ForwardProbModel + ModellingSequence.

    Uses ``gl2_inference_prior()`` (the broad paper prior) and
    ``ForwardProbModel`` (model-based Poisson noise).
    """
    from gigalens.jax.inference import ModellingSequence
    from gigalens.jax.model import ForwardProbModel
    import jax.numpy as jnp

    prior = gl2_inference_prior()
    phys_model = _gl2_phys_model()
    prob_model = ForwardProbModel(
        prior,
        jnp.asarray(system.observed_image),
        background_rms=system.background_rms,
        exp_time=system.exp_time,
    )
    return ModellingSequence(phys_model, prob_model, system.sim_config)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


@register_generator("parametric")
def generate_parametric_gl2(spec: Any, dataset_dir: str, seed: int) -> None:
    """Generic parametric generator: draws truth from gl2_simulation_prior.

    Recognised DatasetSpec.extra keys:
    ``n_systems`` (100), ``gen_chunk`` (8), ``background_rms`` (0.2),
    ``exp_time`` (100), ``delta_pix`` (0.065), ``num_pix`` (80),
    ``supersample`` (2), ``psf_srcdir`` (path to gigalens/src).
    """
    from gigalens.simulator import SimulatorConfig
    from gigalens_research.simtests.generate import generate_parametric

    extra = dict(spec.extra)
    n_systems = int(extra.get("n_systems", 100))
    gen_chunk = int(extra.get("gen_chunk", 8))
    background_rms = float(extra.get("background_rms", 0.2))  # physics-default-ok: documented GL2 generation default, persisted to meta.json
    exp_time = float(extra.get("exp_time", 100.0))  # physics-default-ok: documented GL2 generation default, persisted to meta.json
    delta_pix = float(extra.get("delta_pix", 0.065))  # physics-default-ok: documented GL2 generation default, persisted to meta.json
    num_pix = int(extra.get("num_pix", 80))  # physics-default-ok: documented GL2 generation default, persisted to meta.json
    supersample = int(extra.get("supersample", 2))  # physics-default-ok: documented GL2 generation default, persisted to meta.json
    psf_srcdir = extra.get("psf_srcdir", None)  # physics-default-ok: defaults to bundled gigalens PSF; _gl2_psf raises if missing

    psf = _gl2_psf(psf_srcdir)
    sim_config = SimulatorConfig(
        delta_pix=delta_pix,
        num_pix=num_pix,
        supersample=supersample,
        kernel=psf,
    )

    generate_parametric(
        spec=spec,
        dataset_dir=dataset_dir,
        seed=seed,
        build_phys_model_fn=_gl2_phys_model,
        build_sim_prior_fn=gl2_simulation_prior,
        sim_config=sim_config,
        noise_kind="forward",
        background_rms=background_rms,
        exp_time=exp_time,
        n_systems=n_systems,
        gen_chunk=gen_chunk,
    )


@register_generator("gl2_existing")
def generate_gl2_existing(spec: Any, dataset_dir: str, seed: int) -> None:
    """Adapt the existing GL2 SystemSaves dataset in-place.

    Reads ``100SystemsStandard80px.npz`` and (optionally)
    ``100SystemsStandardParams.yaml`` from the paths given in
    ``DatasetSpec.extra``.  No re-simulation occurs.

    Required extra keys:
    ``npz_path`` — path to the images npz.

    Optional extra keys:
    ``yaml_path`` — path to the truth params YAML.
    ``delta_pix``, ``num_pix``, ``supersample``, ``background_rms``,
    ``exp_time``, ``psf_srcdir``.
    """
    import yaml as _yaml

    from gigalens_research.simtests.system import (
        System, write_manifest, from_gl2_npz_entry,
    )
    from gigalens_research.simtests.generate import _hash_dataset

    extra = dict(spec.extra)
    npz_path = os.path.expanduser(str(extra["npz_path"]))
    yaml_path = extra.get("yaml_path")
    delta_pix = float(extra.get("delta_pix", 0.065))  # physics-default-ok: documented GL2 generation default, persisted to meta.json
    num_pix = int(extra.get("num_pix", 80))  # physics-default-ok: documented GL2 generation default, persisted to meta.json
    supersample = int(extra.get("supersample", 2))  # physics-default-ok: documented GL2 generation default, persisted to meta.json
    background_rms = float(extra.get("background_rms", 0.2))  # physics-default-ok: documented GL2 generation default, persisted to meta.json
    exp_time = float(extra.get("exp_time", 100.0))  # physics-default-ok: documented GL2 generation default, persisted to meta.json
    psf_srcdir = extra.get("psf_srcdir", None)  # physics-default-ok: defaults to bundled gigalens PSF; _gl2_psf raises if missing

    psf = _gl2_psf(psf_srcdir)

    # Load images
    f = np.load(npz_path)
    keys = list(f.files)
    n_systems = len(keys)

    # Load truth params if available.
    # The YAML format stores ALL systems in vectorized form:
    #   raw = [mass_groups, lens_light_groups, source_groups]
    # where each group is a list of profile-dicts, and each dict value is
    # a list/array of length n_systems.
    raw_params = None
    if yaml_path:
        yaml_path = os.path.expanduser(str(yaml_path))
        if os.path.exists(yaml_path):
            with open(yaml_path) as fh:
                raw_params = _yaml.safe_load(fh)

    n_digits = len(str(n_systems - 1))
    system_ids = []
    for i, key in enumerate(keys):
        system_id = f"sys_{i:0{n_digits}d}"
        system_ids.append(system_id)

        # Extract per-system truth params from the vectorized YAML.
        truth_x = None
        if raw_params is not None:
            truth_x = _extract_system_i(raw_params, i)

        sys = from_gl2_npz_entry(
            observed_image=f[key],
            truth_x=truth_x,
            system_id=system_id,
            delta_pix=delta_pix,
            num_pix=num_pix,
            supersample=supersample,
            psf=psf,
            background_rms=background_rms,
            exp_time=exp_time,
        )
        sys.save(dataset_dir)
        if (i + 1) % 10 == 0:
            print(f"[gl2_existing] adapted {i + 1}/{n_systems}")

    dataset_hash = _hash_dataset(dataset_dir, system_ids)
    write_manifest(
        dataset_dir,
        generator="gl2_existing",
        seed=seed,
        system_ids=system_ids,
        dataset_hash=dataset_hash,
        extra={"source_npz": npz_path},
    )


def _extract_system_i(raw_params: Any, i: int) -> Any:
    """Extract per-system truth params for system index ``i`` from the vectorized YAML.

    The GL2 YAML stores parameters in a vectorized format::

        raw_params = [
            # mass group (2 profiles: EPL + Shear)
            [{"theta_E": [v_0, v_1, ...100...], ...}, {"gamma1": [...], ...}],
            # lens light group (1 profile: Sérsic)
            [{"R_sersic": [...100...], ...}],
            # source light group (1 profile: Sérsic)
            [{"R_sersic": [...100...], ...}],
        ]

    Returns the nested list-of-dicts structure for a single system::

        [
            [{"theta_E": float, ...}, {"gamma1": float, ...}],
            [{"R_sersic": float, ...}],
            [{"R_sersic": float, ...}],
        ]
    """
    import numpy as np
    if raw_params is None:
        return None
    result = []
    for group in raw_params:
        group_result = []
        for profile_dict in group:
            single = {k: np.float32(v[i]) if hasattr(v, "__len__") else np.float32(v)
                      for k, v in profile_dict.items()}
            group_result.append(single)
        result.append(group_result)
    return result
