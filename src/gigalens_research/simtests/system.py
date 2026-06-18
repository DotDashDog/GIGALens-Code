"""System dataclass: a single simulated lensing observation with known truth.

On-disk layout under ``<dataset_dir>/systems/<system_id>/``::

    observed_image.npy    (num_pix, num_pix) float32
    truth_x.pkl           nested list-of-dicts of JAX arrays (physical truth)
    psf.npy               (optional) PSF kernel
    meta.json             {delta_pix, num_pix, supersample, noise_kind,
                           background_rms, exp_time, truth_assets}

:func:`load_manifest` / :func:`write_manifest` manage the dataset-level
``manifest.json`` that records the generator, seed, count, dataset hash, and
ordered list of system IDs.

Legacy adapters (:func:`from_gl2_npz_entry`, :func:`from_vela_dir`) convert
the two existing data formats into :class:`System` objects without copying or
re-simulating, allowing the framework to be run over existing data immediately.
"""
from __future__ import annotations

import dataclasses
import json
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np


@dataclasses.dataclass
class System:
    """A single simulated lensing system with known truth parameters.

    Parameters
    ----------
    system_id : str
        Unique identifier within the dataset (e.g. ``"sys_000"``).
    observed_image : ndarray, shape (num_pix, num_pix)
        Noisy observed image in count-per-second (or equivalent) units.
    truth_x : nested list-of-dicts
        Ground-truth physical parameters in the same nested structure as a
        gigalens ``prior.sample()`` result:
        ``[[mass_dicts...], [lens_light_dicts...], [source_light_dicts...]]``.
    delta_pix : float
        Pixel scale in arcseconds.
    num_pix : int
        Image width/height in pixels.
    supersample : int
        Supersampling factor used during generation.
    psf : ndarray or None
        PSF kernel, or None for no PSF.
    noise_kind : str
        ``"forward"`` (model-based Poisson + Gaussian background) or
        ``"backward"`` (fixed err_map).
    background_rms : float
        Background Gaussian noise σ.
    exp_time : float
        Effective exposure time in seconds (used for the Poisson term).
    truth_assets : dict
        Extra information needed for diagnostics, e.g.
        ``{"vela_source_dir": "/path/to/vela01_cam12_..."}`` for Vela systems.
    """

    system_id: str
    observed_image: np.ndarray
    truth_x: Any
    delta_pix: float
    num_pix: int
    supersample: int
    psf: Optional[np.ndarray]
    noise_kind: str
    background_rms: float
    exp_time: float
    truth_assets: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # Likelihood precision, propagated through sim_config to BOTH the bootstrap and the
    # sampler models. "float32" (default), "mixed" (float32 forward + float64 reduction),
    # or "float64" (full float64 forward + reduction; required for high-n_max shapelets).
    # "mixed"/"float64" require jax_enable_x64. See gigalens SimulatorConfig.likelihood_precision.
    likelihood_precision: Optional[str] = None
    # Deprecated: True is treated as likelihood_precision="mixed".
    high_precision_likelihood: bool = False
    # PSF convolution precision override, independent of likelihood_precision.
    # None keeps the convolution in the basis dtype (default). "float32" runs ONLY the
    # PSF convolution arithmetic in float32 (basis generation, gram/solve and the
    # reduction stay float64), which moves the FLOP-dominant per-component conv off the
    # slow Ampere FP64 path. See gigalens SimulatorConfig.conv_precision.
    conv_precision: Optional[str] = None

    @property
    def sim_config(self):
        """Build a :class:`gigalens.simulator.SimulatorConfig` from stored params."""
        from gigalens.simulator import SimulatorConfig
        return SimulatorConfig(
            delta_pix=self.delta_pix,
            num_pix=self.num_pix,
            supersample=self.supersample,
            kernel=self.psf,
            likelihood_precision=self.likelihood_precision,
            high_precision_likelihood=self.high_precision_likelihood,
            conv_precision=self.conv_precision,
        )

    # ------------------------------------------------------------------
    # On-disk I/O
    # ------------------------------------------------------------------

    def save(self, dataset_dir: str) -> None:
        """Write this system to ``<dataset_dir>/systems/<system_id>/``."""
        sys_dir = _system_dir(dataset_dir, self.system_id)
        os.makedirs(sys_dir, exist_ok=True)

        np.save(os.path.join(sys_dir, "observed_image.npy"),
                np.asarray(self.observed_image, dtype=np.float32))

        with open(os.path.join(sys_dir, "truth_x.pkl"), "wb") as f:
            pickle.dump(
                # Convert leaves to plain numpy for portability
                _to_numpy_leaves(self.truth_x),
                f, protocol=4,
            )

        if self.psf is not None:
            np.save(os.path.join(sys_dir, "psf.npy"),
                    np.asarray(self.psf, dtype=np.float32))

        meta = {
            "delta_pix": float(self.delta_pix),
            "num_pix": int(self.num_pix),
            "supersample": int(self.supersample),
            "noise_kind": str(self.noise_kind),
            "background_rms": float(self.background_rms),
            "exp_time": float(self.exp_time),
            # Numerics: persisted so the precision survives the dataset round-trip
            # (run/plot load via System.load). Dropping these silently degrades a
            # float64 dataset to float32 at inference time.
            "likelihood_precision": self.likelihood_precision,
            "high_precision_likelihood": bool(self.high_precision_likelihood),
            "conv_precision": self.conv_precision,
            "truth_assets": {k: str(v) for k, v in self.truth_assets.items()},
        }
        with open(os.path.join(sys_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, dataset_dir: str, system_id: str) -> "System":
        """Load a system from ``<dataset_dir>/systems/<system_id>/``."""
        sys_dir = _system_dir(dataset_dir, system_id)

        observed_image = np.load(os.path.join(sys_dir, "observed_image.npy"))

        with open(os.path.join(sys_dir, "truth_x.pkl"), "rb") as f:
            truth_x = pickle.load(f)

        psf_path = os.path.join(sys_dir, "psf.npy")
        if not os.path.exists(psf_path):
            raise FileNotFoundError(
                f"No PSF at {psf_path!r} for system {system_id!r}. A PSF is mandatory "
                "(the data are PSF-convolved); refusing to load a system that would be "
                "modeled without one. Re-save the system with its PSF."
            )
        psf = np.load(psf_path)

        with open(os.path.join(sys_dir, "meta.json")) as f:
            meta = json.load(f)

        return cls(
            system_id=system_id,
            observed_image=observed_image,
            truth_x=truth_x,
            delta_pix=float(meta["delta_pix"]),
            num_pix=int(meta["num_pix"]),
            supersample=int(meta["supersample"]),
            psf=psf,
            noise_kind=str(meta["noise_kind"]),
            background_rms=float(meta["background_rms"]),
            exp_time=float(meta["exp_time"]),
            # Numerics: default to None/False for datasets generated before these
            # were persisted (they ran float32); fresh datasets carry the value.
            likelihood_precision=meta.get("likelihood_precision"),
            high_precision_likelihood=bool(meta.get("high_precision_likelihood", False)),
            conv_precision=meta.get("conv_precision"),
            truth_assets=dict(meta.get("truth_assets", {})),
        )


# ------------------------------------------------------------------
# Dataset manifest
# ------------------------------------------------------------------

def write_manifest(
    dataset_dir: str,
    *,
    generator: str,
    seed: int,
    system_ids: List[str],
    dataset_hash: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Write ``<dataset_dir>/manifest.json``."""
    os.makedirs(dataset_dir, exist_ok=True)
    manifest = {
        "generator": generator,
        "seed": seed,
        "n_systems": len(system_ids),
        "system_ids": system_ids,
        "dataset_hash": dataset_hash,
        "extra": extra or {},
    }
    with open(os.path.join(dataset_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(dataset_dir: str) -> Dict[str, Any]:
    """Load ``<dataset_dir>/manifest.json``."""
    path = os.path.join(dataset_dir, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No manifest.json found at {path}. "
            f"Run 'python -m gigalens_research.simtests generate <campaign.yaml>' first."
        )
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------
# Legacy format adapters (read existing data in-place)
# ------------------------------------------------------------------

def from_gl2_npz_entry(
    observed_image: np.ndarray,
    truth_x: Any,
    system_id: str,
    *,
    delta_pix: float,
    num_pix: int,
    supersample: int,
    psf: Optional[np.ndarray],
    background_rms: float,
    exp_time: float,
) -> "System":
    """Construct a System from one entry of the GL2 npz+yaml dataset."""
    return System(
        system_id=system_id,
        observed_image=np.asarray(observed_image, dtype=np.float32),
        truth_x=truth_x,
        delta_pix=delta_pix,
        num_pix=num_pix,
        supersample=supersample,
        psf=psf,
        noise_kind="forward",
        background_rms=background_rms,
        exp_time=exp_time,
        truth_assets={"source": "gl2_npz"},
    )


def from_vela_dir(
    system_dir: str,
    source_dir: str,
    system_id: str,
    *,
    delta_pix: float,
    num_pix: int,
    supersample: int,
    background_rms: float,
    exp_time: float,
    likelihood_precision: Optional[str] = None,
    high_precision_likelihood: bool = False,
    conv_precision: Optional[str] = None,
    allow_no_psf: bool = False,
) -> "System":
    """Construct a System from an existing Vela sim-system directory.

    Loads ``lens_img.npy`` and ``true_params`` from ``system_dir``, and
    ``psf.npy`` from ``source_dir``.  The source directory path is stored in
    ``truth_assets["vela_source_dir"]`` for downstream truth-source rendering.

    The Vela observations are PSF-convolved, so a model without a PSF is
    misspecified. Rather than silently fall back to ``psf=None`` (which makes a
    wrong model look like a successful run), this raises if ``psf.npy`` cannot
    be found at ``source_dir``. Pass ``allow_no_psf=True`` only for the rare
    deliberate no-PSF test.
    """
    observed_image = np.load(os.path.join(system_dir, "lens_img.npy"))
    with open(os.path.join(system_dir, "true_params"), "rb") as f:
        truth_x = pickle.load(f)

    psf_path = os.path.join(source_dir, "psf.npy")
    if os.path.exists(psf_path):
        psf = np.load(psf_path)
    elif allow_no_psf:
        psf = None
    else:
        raise FileNotFoundError(
            f"No PSF found at {psf_path!r} (source_dir={source_dir!r}). The Vela "
            "data are PSF-convolved, so modeling without a PSF is misspecified. "
            "Check the source_dir path (note Vela source dirs are rep-less, e.g. "
            "'vela01_cam12_a0.500_f814w', not '..._rep03_...'), or pass "
            "allow_no_psf=True to deliberately model without a PSF."
        )

    return System(
        system_id=system_id,
        observed_image=np.asarray(observed_image, dtype=np.float32),
        truth_x=truth_x,
        delta_pix=delta_pix,
        num_pix=num_pix,
        supersample=supersample,
        psf=psf,
        noise_kind="forward",
        background_rms=background_rms,
        exp_time=exp_time,
        likelihood_precision=likelihood_precision,
        high_precision_likelihood=high_precision_likelihood,
        conv_precision=conv_precision,
        truth_assets={
            "vela_source_dir": source_dir,
            "vela_system_dir": system_dir,
        },
    )


def _system_dir(dataset_dir: str, system_id: str) -> str:
    return os.path.join(dataset_dir, "systems", system_id)


def _to_numpy_leaves(obj: Any) -> Any:
    """Recursively convert JAX/numpy arrays in a nested structure to plain numpy."""
    try:
        import jax
        import jax.numpy as jnp
        return jax.tree.map(lambda x: np.asarray(x), obj)
    except Exception:
        return obj
