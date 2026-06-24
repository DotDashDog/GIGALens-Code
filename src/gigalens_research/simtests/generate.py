"""Generation driver for simulated system datasets.

:func:`generate_campaign` is the top-level entry point.  It reads the
``DatasetSpec``, looks up the generator in the registry, and calls it with the
spec and the dataset directory.

The built-in ``"parametric"`` generator covers any campaign where the truth
model and simulation prior are expressed as gigalens
:class:`~gigalens.model.PhysicalModel` and a TFP joint distribution.  It
simulates and noises systems in **chunks** to bound GPU memory usage.

Custom generators (e.g. ``"gl2_existing"`` or ``"vela_existing"``) are
registered by the experiment modules in ``experiments/`` and simply adapt
existing on-disk data into the standard :class:`~system.System` format.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np


def generate_campaign(
    campaign_spec: Any,
    base_dir: str,
    *,
    force: bool = False,
    verbose: bool = True,
) -> str:
    """Generate (or load) the system dataset for a campaign.

    If a valid ``manifest.json`` already exists at ``<base_dir>/dataset/``
    and ``force=False``, generation is skipped.

    Returns the path to the dataset directory.
    """
    from .config import CampaignSpec
    from .registry import get_generator
    from .system import load_manifest

    assert isinstance(campaign_spec, CampaignSpec)
    dataset_dir = os.path.join(base_dir, "dataset")

    if not force:
        manifest_path = os.path.join(dataset_dir, "manifest.json")
        if os.path.exists(manifest_path):
            m = load_manifest(dataset_dir)
            if verbose:
                print(
                    f"[generate] dataset already exists: {m['n_systems']} systems "
                    f"(generator={m['generator']}, seed={m['seed']}). "
                    f"Pass --force to regenerate."
                )
            return dataset_dir

    gen_fn = get_generator(campaign_spec.dataset.generator)
    os.makedirs(dataset_dir, exist_ok=True)

    if verbose:
        print(
            f"[generate] running generator {campaign_spec.dataset.generator!r} "
            f"(seed={campaign_spec.seed}) → {dataset_dir}"
        )

    gen_fn(
        spec=campaign_spec.dataset,
        dataset_dir=dataset_dir,
        seed=campaign_spec.seed,
    )

    if verbose:
        from .system import load_manifest as _lm
        m = _lm(dataset_dir)
        print(f"[generate] wrote {m['n_systems']} systems.")

    return dataset_dir


def generate_parametric(
    spec: Any,
    dataset_dir: str,
    seed: int,
    *,
    build_scene_model_fn: Any,
    build_sim_prior_fn: Any,
    sim_config: Any,
    noise_kind: str = "forward",
    background_rms: float = 0.2,
    exp_time: float = 100.0,
    n_systems: int = 100,
    gen_chunk: int = 10,
    noise_seed_offset: int = 1000,
    id_prefix: str = "sys",
) -> None:
    """Shared core for parametric generators (truth model = inference model).

    Generates ``n_systems`` simulated lensed images by:
    1. Drawing truth params from ``build_sim_prior_fn()``.
    2. Simulating noiseless images in batches of ``gen_chunk`` systems.
    3. Adding Poisson + Gaussian background noise (lenstronomy-style).
    4. Saving each system with :meth:`~system.System.save`.
    5. Writing a ``manifest.json``.

    Parameters
    ----------
    build_scene_model_fn : callable() -> gigalens.jax.scene.LensModel
        Returns the *truth* scene model (used for simulation via SceneSimulator).
    build_sim_prior_fn : callable() -> tfd.Distribution
        Returns the simulation prior (typically tighter than inference prior). Emits the
        legacy 3-group truth structure; the persisted ``truth_x`` is unchanged.
    sim_config : SimulatorConfig
    noise_kind : str
    background_rms : float
    exp_time : float
    n_systems : int
    gen_chunk : int
        Number of systems simulated per JAX batch (controls GPU memory use).
    id_prefix : str
        Prefix for system IDs (``sys_000``, ``sys_001``, …).
    """
    import jax
    import jax.numpy as jnp
    import jax.tree_util as jtu
    from jax import random

    from gigalens.jax.scene_simulator import SceneSimulator

    from gigalens_research.inference_utils.params import truth_x_to_scene_params
    from .system import System, write_manifest

    # G1 scene migration: the truth model is now a scene LensModel; the simulation prior
    # still emits the legacy 3-group truth (``{lens_mass, lens_light, source_light}``) so
    # the PERSISTED ``truth_x`` is unchanged (downstream plotting / bootstrap read it).
    # Each 3-group truth is adapted to scene structured params (truth_x_to_scene_params)
    # and rendered with SceneSimulator. Verified byte-identical to the old LensSimulator
    # render (single + batched) at the float64 floor.
    scene_model = build_scene_model_fn()
    prior = build_sim_prior_fn()

    n_digits = len(str(n_systems - 1))
    system_ids: List[str] = [
        f"{id_prefix}_{i:0{n_digits}d}" for i in range(n_systems)
    ]

    key = random.PRNGKey(seed)
    params_key, key = random.split(key)

    # Draw ALL truth parameters at once and immediately convert to numpy to
    # free device memory before the chunked simulation loop.
    all_truth_x = prior.sample(n_systems, seed=params_key)
    all_truth_np = jax.tree.map(lambda a: np.asarray(a), all_truth_x)

    psf = sim_config.kernel

    def _stack_scene_params(truth_dicts):
        """Stack per-system scene structured params (from truth_x_to_scene_params) into a
        single batched params dict (leading batch axis on every leaf), as SceneSimulator
        expects for ``bs>1`` (verified equivalent to per-system renders)."""
        per_sys = [truth_x_to_scene_params(t, scene_model) for t in truth_dicts]
        leaves0, treedef = jtu.tree_flatten(per_sys[0])
        stacked = [jnp.stack([jtu.tree_flatten(p)[0][i] for p in per_sys])
                   for i in range(len(leaves0))]
        return jtu.tree_unflatten(treedef, stacked)

    # Simulate in chunks to bound GPU memory.
    for chunk_start in range(0, n_systems, gen_chunk):
        chunk_end = min(chunk_start + gen_chunk, n_systems)
        bs = chunk_end - chunk_start

        # Slice the truth params for this chunk (kept in 3-group form for persistence).
        chunk_truth = jax.tree.map(lambda a: jnp.asarray(a[chunk_start:chunk_end]), all_truth_np)
        # Per-system 3-group truth dicts -> stacked scene params for a batched render.
        chunk_truth_list = [
            jax.tree.map(lambda a: np.asarray(a[j]), chunk_truth) for j in range(bs)
        ]
        chunk_truth_list = [_unbatch_truth(t) for t in chunk_truth_list]
        scene_params = _stack_scene_params(chunk_truth_list)

        chunk_key, key = random.split(key)
        lens_sim = SceneSimulator(scene_model, sim_config, bs=bs)
        noiseless = lens_sim.simulate(scene_params)  # (bs, H, W)

        noisy_batch = _add_noise(noiseless, background_rms, exp_time, chunk_key)
        noisy_np = np.asarray(noisy_batch)

        for j in range(bs):
            idx = chunk_start + j
            system_id = system_ids[idx]
            # 3-group truth (already unbatched above) — persisted unchanged.
            truth_x_nested = chunk_truth_list[j]

            sys = System(
                system_id=system_id,
                observed_image=noisy_np[j],
                truth_x=truth_x_nested,
                delta_pix=sim_config.delta_pix,
                num_pix=sim_config.num_pix,
                supersample=sim_config.supersample,
                psf=psf,
                noise_kind=noise_kind,
                background_rms=background_rms,
                exp_time=exp_time,
            )
            sys.save(dataset_dir)

        print(
            f"[generate_parametric] chunk {chunk_start}–{chunk_end - 1} done "
            f"({chunk_end}/{n_systems})"
        )

    dataset_hash = _hash_dataset(dataset_dir, system_ids)
    write_manifest(
        dataset_dir,
        generator="parametric",
        seed=seed,
        system_ids=system_ids,
        dataset_hash=dataset_hash,
    )


def _add_noise(
    images: Any,
    background_rms: float,
    exp_time: float,
    key: Any,
) -> Any:
    """Add Poisson + Gaussian background noise (lenstronomy-compatible).

    Mirrors ``noise_images`` from ``Linus-FourSim.ipynb`` and
    ``lens_vela_system.ipynb``.
    """
    import jax.numpy as jnp
    from jax import random

    imgs = jnp.asarray(images)
    key_p, key_g = random.split(key)

    # Poisson: σ² = I / exp_time
    poisson_std = jnp.sqrt(jnp.clip(imgs / exp_time, 0.0, None))
    poisson_noise = random.normal(key_p, shape=imgs.shape) * poisson_std

    # Gaussian background
    bkg_noise = random.normal(key_g, shape=imgs.shape) * background_rms

    return imgs + poisson_noise + bkg_noise


def _unbatch_truth(truth_j: Any) -> Any:
    """Given a slice of batched truth (scalars), reconstruct nested list-of-dicts.

    When prior.sample(N) produces arrays of shape (N, ...) and we slice row j,
    leaves are shape () scalars. This wraps them back in numpy float scalars so
    the nested structure is plain-python serializable.
    """
    import numpy as np
    return _map_leaves(truth_j, lambda v: float(np.asarray(v).squeeze())
                       if np.asarray(v).ndim == 0 or np.asarray(v).size == 1
                       else np.asarray(v).squeeze())


def _map_leaves(obj: Any, fn: Any) -> Any:
    import numpy as np
    if isinstance(obj, dict):
        return {k: _map_leaves(v, fn) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        mapped = [_map_leaves(v, fn) for v in obj]
        return type(obj)(mapped)
    try:
        return fn(obj)
    except Exception:
        return obj


def _hash_dataset(dataset_dir: str, system_ids: List[str]) -> str:
    h = hashlib.sha256()
    for sid in system_ids:
        img_path = os.path.join(dataset_dir, "systems", sid, "observed_image.npy")
        if os.path.exists(img_path):
            data = np.load(img_path)
            h.update(data.tobytes())
    return h.hexdigest()[:16]
