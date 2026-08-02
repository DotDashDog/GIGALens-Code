"""Per-system target for the simulated Vela system
vela01_cam12_rep03_a0.500_f814w (T0/T1 harness).

Replicates EXACTLY the model built in the human's modeling notebook
`experiments/TestNewAPI/TestSersiclets.ipynb` (cells 4, 7): a scene-API lens
(EPL+Shear mass + elliptical Sersic lens light on plane 0; EllipticalSersiclets
source on plane 1) built through the registered inference builder
`epl_shear_sersic_elliptical_sersiclets` with n_max=5. Source amplitudes are
lstsq-marginalized (mode="lstsq"), so the 20 sampled parameters are all
non-linear. dim=20.

Reference run (MAIN checkout, read-only):
  results/test_new_api_elliptical_sersiclets/vela01_cam12_rep03_a0.500_f814w/
  -- pipeline PartialTruthBootstrapQzStage -> MCLMC (8 chains, 5000/5000,
  seed 0, n_params=20, debug=True). pipeline.json shows BOTH stages status
  "ran"; mclmc/manifest.json upstream_hashes.qz == the bootstrap stage's qz.

qz PROVENANCE (single most correctness-critical item)
-----------------------------------------------------
The MCLMC reference run's `qz` is the truth-bootstrap MAP ball produced by
PartialTruthBootstrapQzStage (free=("source",), diag_scale=1e-6 => diagonal
scale 1e-3, centered on truth for pinned params and on a short truth-pinned MAP
for the free source params). We do NOT re-run the bootstrap (it runs a random
MAP; a different seed/order would give a different qz and break the T0/T1
design). Instead we LOAD the exact float64 arrays the stage persisted and
rebuild the identical TFP object the pipeline builds
(pipeline.py:1624-1629 / PartialTruthBootstrapQzStage derive):

    file:  results/test_new_api_elliptical_sersiclets/
             vela01_cam12_rep03_a0.500_f814w/bootstrap_map/arrays.npz
    keys:  qz_loc (20,) float64, qz_scale_tril (20,20) float64
           (also per-free-param scalars, unused here)
    obj:   tfd.MultivariateNormalTriL(loc=qz_loc, scale_tril=qz_scale_tril)

Because we load the same bytes and construct the same distribution class, the
qz is byte-identical to the one MCLMCStage consumed. (Verified: stable_hash of
the rebuilt qz == 1d80d19891ea0749 == mclmc/manifest.json upstream_hashes.qz.)

TRUTH parameters (for T4 freeze-at-truth, later): the simulated truth lives in
  data/vela_sim_systems/vela01_cam12_rep03_a0.500_f814w/true_params  (pickle),
loaded by from_vela_dir into `system.truth_x` (3-group nested params). The
bootstrap stage already used it (config truth_hash be69fea07af651fb).
"""
from __future__ import annotations

import os

import numpy as np

# Absolute paths into the MAIN checkout (this worktree lacks untracked data).
_MAIN = "/global/homes/l/linusu/GIGALens-Code"

_SYSTEM_NAME = "vela01_cam12_rep03_a0.500_f814w"
_SOURCE_NAME = "vela01_cam12_a0.500_f814w"  # Vela source dirs are rep-less
_SYSTEM_DIR = os.path.join(_MAIN, "data/vela_sim_systems", _SYSTEM_NAME)
_SOURCE_DIR = os.path.join(_MAIN, "data/vela_sources", _SOURCE_NAME)
_BOOTSTRAP_ARRAYS = os.path.join(
    _MAIN, "results/test_new_api_elliptical_sersiclets", _SYSTEM_NAME,
    "bootstrap_map/arrays.npz")

_N_MAX = 5
_EXPECTED_DIM = 20
_QZ_UPSTREAM_HASH = "1d80d19891ea0749"  # mclmc/manifest.json upstream_hashes.qz


def _require(path: str, what: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"[vela] missing {what}: {path}")
    return path


def _assert_qz_not_stale() -> None:
    """Guard against silent qz drift: if this Vela pipeline is ever re-run,
    bootstrap_map/arrays.npz is REPLACED (old stage rotated to *.stale-*) and
    the rebuilt qz would no longer be the one the reference MCLMC run consumed.
    Cheap check: the mclmc manifest's recorded upstream qz hash must still equal
    the pinned constant. (Residual blind spot: a bootstrap-only re-run without a
    new MCLMC run changes arrays.npz but not the manifest -- re-verify
    stable_hash(qz) after any pipeline re-run and re-pin.)"""
    import json
    man = os.path.join(_MAIN, "results/test_new_api_elliptical_sersiclets",
                       _SYSTEM_NAME, "mclmc/manifest.json")
    with open(_require(man, "reference mclmc manifest")) as f:
        got = json.load(f).get("upstream_hashes", {}).get("qz")
    if got != _QZ_UPSTREAM_HASH:
        raise RuntimeError(
            f"[vela] reference mclmc manifest upstream_hashes.qz={got!r} != "
            f"pinned {_QZ_UPSTREAM_HASH!r}: the pipeline was re-run and the "
            "persisted qz no longer matches the reference run this module was "
            "verified against. Re-verify and re-pin before sampling.")


def _recover_param_names(prob_model, dim):
    """C-8-safe param names: zero-probe the bijector and SORT the output dict's
    keys (validated route; see t2_zspace_diagnostics.recover_param_names)."""
    probe = np.zeros((1, dim))
    out = prob_model.bij.forward(list(probe.T))
    if not isinstance(out, dict) or not out or any(
            isinstance(v, (dict, list)) for v in out.values()):
        raise TypeError(
            f"[vela] bij.forward did not return a flat scalar-key dict: {type(out)}")
    names = sorted(str(k) for k in out.keys())
    if len(names) != dim or len(set(names)) != len(names):
        raise ValueError(
            f"[vela] recovered {len(names)} unique names != dim {dim}: {names}")
    return names


def load_target():
    """Return (prob_model, qz, z_center, dim, param_names) for the Vela system.

    - prob_model.log_prob(z) -> (scalar_logp, chisq); MCLMC_JIT uses
      index [0].
    - qz: truth-bootstrap MAP ball rebuilt byte-identically from the reference
      bootstrap_map stage.
    - z_center: the bootstrap loc (qz.loc == truth-bootstrapped center).
    """
    import jax
    jax.config.update("jax_enable_x64", True)  # notebook cell 0; sampling needs float64
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions

    from gigalens_research.simtests.system import from_vela_dir
    from gigalens_research.simtests.registry import get_inference_builder
    # Importing this module registers the "epl_shear_sersic_elliptical_sersiclets"
    # inference builder (notebook cell 1 does the same import for the side effect).
    import gigalens_research.simtests.experiments.vela_elliptical_sersiclets  # noqa: F401

    # --- system + model (notebook cell 4) ---------------------------------
    _require(_SYSTEM_DIR, "Vela system dir")
    _require(_SOURCE_DIR, "Vela source dir")
    system = from_vela_dir(
        system_dir=_SYSTEM_DIR,
        source_dir=_SOURCE_DIR,
        system_id="vela01_cam12_rep00",   # verbatim from the notebook (id label only)
        delta_pix=0.03,
        num_pix=200,
        supersample=1,
        background_rms=0.002,
        exp_time=2000.0,
    )
    system.likelihood_precision = "float64"
    system.conv_precision = "float32"

    prob_model = get_inference_builder(
        "epl_shear_sersic_elliptical_sersiclets")(system, n_max=_N_MAX)

    # --- qz: byte-identical rebuild of the reference bootstrap Gaussian ----
    a = np.load(_require(_BOOTSTRAP_ARRAYS, "reference bootstrap_map arrays.npz"))
    qz_loc = np.asarray(a["qz_loc"], dtype=np.float64)
    qz_scale_tril = np.asarray(a["qz_scale_tril"], dtype=np.float64)
    dim = int(qz_loc.shape[0])
    if qz_loc.shape != (_EXPECTED_DIM,):
        raise ValueError(f"[vela] qz_loc shape {qz_loc.shape} != ({_EXPECTED_DIM},)")
    if qz_scale_tril.shape != (_EXPECTED_DIM, _EXPECTED_DIM):
        raise ValueError(
            f"[vela] qz_scale_tril shape {qz_scale_tril.shape} != "
            f"({_EXPECTED_DIM},{_EXPECTED_DIM})")
    qz = tfd.MultivariateNormalTriL(
        loc=jnp.asarray(qz_loc), scale_tril=jnp.asarray(qz_scale_tril))

    param_names = _recover_param_names(prob_model=prob_model, dim=dim)
    z_center = np.asarray(qz_loc)
    return prob_model, qz, z_center, dim, param_names
