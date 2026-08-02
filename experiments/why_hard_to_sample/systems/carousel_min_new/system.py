"""Carousel minimal case -- NEW arm (NFW_ELLIPSE_EINSTEIN, theta_E ~ N(13,1)).

Rebuilds EXACTLY the model behind the reference MCLMC run at
  experiments/sim_carousel/messy_tests/minimal_case_newbij/
(pipeline MAPStage -> BridgeStage "diag_qz" v1 -> MCLMCStage, 8 chains,
10000/10000, desired_energy_variance=5e-4, regularize_mass_matrix=True, seed 42;
n_params=14). This is the notebook's CURRENT active state.

This arm's mass profile is NFW_ELLIPSE_EINSTEIN parameterised by the Einstein
radius: Rs ~ Uniform(20,100), theta_E ~ Normal(13,1). The rest of the model
(shear, two lensed shapelet sources n_max=8/6, cosmology, data, grid) is shared
with the OLD arm and lives in ../carousel_min_common.py.

Interface: `load_target()` returns the harness 5-tuple
  (prob_model, qz, z_center, dim, param_names)
exactly like systems/sys60/system.py (common.load_target dispatches here).

Provenance / staleness (see carousel_min_common docstring): qz is the diag_qz v1
Gaussian rebuilt FROM the stored MAP z_best (never re-optimized); load raises if
stable_hash(qz) != the pinned manifest value.
"""
from __future__ import annotations

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_COMMON_PY = os.path.join(os.path.dirname(_HERE), "carousel_min_common.py")

# --- pinned reference run + qz hash (module constants; see task facts) --------
REF_DIR = ("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/"
           "messy_tests/minimal_case_newbij")
PINNED_QZ_HASH = "b0e1b247f9051cf4"  # newbij mclmc/manifest.json upstream_hashes.qz


def _common():
    spec = importlib.util.spec_from_file_location(
        "carousel_min_common", _COMMON_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _nfw0():
    """NEW arm NFW mass Component. Built lazily (imports need x64 context)."""
    import jax.numpy as jnp  # noqa: F401  (kept for parity / future log-priors)
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions
    from gigalens.jax.scene import Component
    from gigalens.jax.profiles.mass.nfw import NFW_ELLIPSE_EINSTEIN
    return Component(NFW_ELLIPSE_EINSTEIN(), dict(
        Rs=tfd.Uniform(20, 100),
        theta_E=tfd.Normal(13, 1),
        e1=tfd.TruncatedNormal(0, 0.05, -0.2, 0.2),
        e2=tfd.TruncatedNormal(0, 0.05, -0.2, 0.2),
        center_x=tfd.Normal(5.344, 0.05),
        center_y=tfd.Normal(3.805, 0.05),
    ))


def load_target(supersample=None):
    return _common().load_target(_nfw0, REF_DIR, PINNED_QZ_HASH,
                                 supersample=supersample)


if __name__ == "__main__":
    if "--verify" in sys.argv:
        sys.exit(_common().verify(_nfw0, REF_DIR, PINNED_QZ_HASH))
    print("carousel_min_new: pass --verify to run the GPU self-check.")
