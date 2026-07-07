"""Per-system target for simulated Sersic "system 60" (T0/T1 harness).

Replicates EXACTLY the model built in the human's modeling notebook
`experiments/TestNewAPI/TestSersic60.ipynb` (cells 2-5). This is a scene-API
lens with SAMPLED (forward-mode) Sersic amplitudes, so all 22 non-linear +
amplitude parameters are sampled by MCLMC.

Reference run (MAIN checkout, read-only):
  results/testsys60/  -- pipeline MAP -> SVI -> MCLMC (8 chains, 20000/20000,
  seed 0, n_params=22, debug=True). pipeline.json shows MCLMCStage.status="ran"
  with upstream qz hash e59573bb9e715b71 == the SVI stage's produced qz.

qz PROVENANCE (single most correctness-critical item)
-----------------------------------------------------
The MCLMC reference run's `qz` is the SVI stage's Gaussian. We do NOT re-run SVI
(a different seed would give a different qz and silently break the T0/T1 design,
which requires all arms to differ ONLY in the MCLMC seed). Instead we LOAD the
exact float64 arrays the SVI stage persisted and rebuild the identical TFP
object the pipeline's SVIStage.derive_artifacts builds (pipeline.py:1478-1483):

    file:  results/testsys60/svi/arrays.npz
    keys:  qz_loc (22,) float64, qz_scale_tril (22,22) float64
    obj:   tfd.MultivariateNormalTriL(loc=qz_loc, scale_tril=qz_scale_tril)

Because we load the same bytes and construct the same distribution class, the
qz is byte-identical to the one MCLMCStage consumed. (Verified: stable_hash of
the rebuilt qz == e59573bb9e715b71 == mclmc/manifest.json upstream_hashes.qz.)

TRUTH parameters (for T4 freeze-at-truth, later): the simulated truth for
system 60 lives in
  data/simulated_systems/100SystemsStandardParams.yaml  (index i=60),
loaded via the notebook's `params_lists_to_jax(...)` then `jax.tree.map(lambda
a: a[60], ...)`. NOT loaded here (not needed to build the target).
"""
from __future__ import annotations

import os

import numpy as np

# Absolute paths into the MAIN checkout (this worktree lacks untracked data).
_MAIN = "/global/homes/l/linusu/GIGALens-Code"
_GIGALENS_SRC = "/global/homes/l/linusu/gigalens/src"

_DATA_NPZ = os.path.join(_MAIN, "data/simulated_systems/100SystemsStandard80px.npz")
_PSF = os.path.join(_GIGALENS_SRC, "gigalens/assets/psf.npy")
# RE-PINNED 2026-07-03: the user's own re-runs rotated the original reference
# pipeline stages to .stale-* dirs (the qz-staleness guard caught it). The
# ORIGINAL stages (qz hash e59573bb..., mclmc arrays byte-identical to our
# archived clone_source.npz) now live at these FROZEN paths:
_SVI_ARRAYS = os.path.join(_MAIN, "results/testsys60/svi.stale-20260703T110909/arrays.npz")
_SYSTEM_INDEX = 60
_EXPECTED_DIM = 22
_QZ_UPSTREAM_HASH = "e59573bb9e715b71"  # mclmc/manifest.json upstream_hashes.qz (verified)


def _require(path: str, what: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[sys60] missing {what}: {path}")
    return path


def _assert_qz_not_stale() -> None:
    """Guard against silent qz drift: if the testsys60 pipeline is ever re-run,
    svi/arrays.npz is REPLACED (old stage rotated to *.stale-*) and the rebuilt
    qz would no longer be the one the reference MCLMC run consumed. Cheap check:
    the mclmc manifest's recorded upstream qz hash must still equal the pinned
    constant. (Residual blind spot: an SVI-only re-run without a new MCLMC run
    changes arrays.npz but not the manifest -- re-verify stable_hash(qz) after
    any pipeline re-run and re-pin both the hash and this comment.)"""
    import json
    man = os.path.join(_MAIN, "results/testsys60/mclmc.stale-20260703T111618/manifest.json")
    with open(_require(man, "reference mclmc manifest")) as f:
        got = json.load(f).get("upstream_hashes", {}).get("qz")
    if got != _QZ_UPSTREAM_HASH:
        raise RuntimeError(
            f"[sys60] reference mclmc manifest upstream_hashes.qz={got!r} != "
            f"pinned {_QZ_UPSTREAM_HASH!r}: the pipeline was re-run and the "
            "persisted qz no longer matches the reference run this module was "
            "verified against. Re-verify and re-pin before sampling.")


def _recover_param_names(prob_model, dim):
    """C-8-safe param names: zero-probe the bijector and SORT the output dict's
    keys (validated route; see t2_zspace_diagnostics.recover_param_names and
    common.load_param_names). Never positional names from unsorted output."""
    probe = np.zeros((1, dim))
    out = prob_model.bij.forward(list(probe.T))
    if not isinstance(out, dict) or not out or any(
            isinstance(v, (dict, list)) for v in out.values()):
        raise TypeError(
            f"[sys60] bij.forward did not return a flat scalar-key dict: {type(out)}")
    names = sorted(str(k) for k in out.keys())
    if len(names) != dim or len(set(names)) != len(names):
        raise ValueError(
            f"[sys60] recovered {len(names)} unique names != dim {dim}: {names}")
    return names


def load_target(supersample=None):
    """Return (model_seq, qz, z_center, dim, param_names) for system 60.

    - model_seq.prob_model.log_prob(z) -> (scalar_logp, chisq); MCLMC_JIT uses
      index [0]. Verified: scene ProbModel.log_prob returns a tuple.
    - qz: SVI Gaussian rebuilt byte-identically from the reference SVI stage.
    - z_center: the SVI mean (qz.loc), the natural typical-set center.

    supersample : None (default) reproduces the notebook's SimulatorConfig value
      (supersample=2) EXACTLY, byte-for-byte -- the reference-run behavior. An int
      overrides ONLY the SimulatorConfig supersample factor (e.g. 4 for the T12
      pixelation control); everything else is unchanged. Nothing else about the
      build depends on this argument.
    """
    import jax
    if os.environ.get("WHTS_FLOAT32_CONTROL") == "1":
        # T3 float32 positive-control arm ONLY (slurm/run_t3.sh): leave x64 OFF
        # so the model is built natively float32, constants included (a clean
        # O5-style noise-floor control). NEVER set for real sampling runs --
        # run_standard_mclmc's assert_x64 will refuse anyway.
        pass
    else:
        jax.config.update("jax_enable_x64", True)  # notebook cell 0; sampling needs float64
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions

    from gigalens.simulator import SimulatorConfig
    from gigalens.jax.profiles.light import sersic
    from gigalens.jax.profiles.mass import epl, shear
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_prob_model import Dataset, ProbModel
    from gigalens.jax.inference import ModellingSequence

    # --- data (notebook cell 3) -------------------------------------------
    _require(_DATA_NPZ, "simulated-systems npz")
    f = np.load(_DATA_NPZ)
    observed_imgs = jnp.array([f[key] for key in f.files])
    observed_img = observed_imgs[_SYSTEM_INDEX]

    # --- model (notebook cell 4) ------------------------------------------
    epl0 = Component(epl.EPL(50), dict(
        theta_E=tfd.LogNormal(jnp.log(1.25), 0.4),
        gamma=tfd.TruncatedNormal(2, 0.5, 1, 3),
        e1=tfd.Normal(0, 0.2),
        e2=tfd.Normal(0, 0.2),
        center_x=tfd.Normal(0, 0.06),
        center_y=tfd.Normal(0, 0.06),
    ))
    shear0 = Component(shear.Shear(), dict(
        gamma1=tfd.Normal(0, 0.1), gamma2=tfd.Normal(0, 0.1)))
    lens_light = Component(sersic.SersicEllipse(use_lstsq=False), dict(
        R_sersic=tfd.LogNormal(jnp.log(1.6), 0.25),
        n_sersic=tfd.Uniform(0.5, 8),
        e1=tfd.TruncatedNormal(0, 0.1, -0.15, 0.15),
        e2=tfd.TruncatedNormal(0, 0.1, -0.15, 0.15),
        center_x=tfd.Normal(0, 0.02),
        center_y=tfd.Normal(0, 0.02),
        Ie=tfd.LogNormal(jnp.log(300.0), 0.5),
    ))
    source_light = Component(sersic.SersicEllipse(use_lstsq=False), dict(
        R_sersic=tfd.LogNormal(jnp.log(0.25), 0.25),
        n_sersic=tfd.Uniform(0.5, 8),
        e1=tfd.TruncatedNormal(0, 0.3, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0, 0.3, -0.5, 0.5),
        center_x=tfd.Normal(0, 0.5),
        center_y=tfd.Normal(0, 0.5),
        Ie=tfd.LogNormal(jnp.log(150.0), 0.9),
    ))

    model = LensModel([
        Plane(mass=[epl0, shear0], light=[lens_light]),
        Plane(light=[source_light]),
    ])

    kernel = np.load(_require(_PSF, "PSF")).astype(np.float64)
    _ss = 2 if supersample is None else int(supersample)  # None => notebook value (exact)
    sim_config = SimulatorConfig(delta_pix=0.065, num_pix=80, supersample=_ss, kernel=kernel)
    ds = Dataset(observed_img, sim_config,
                 sees=[lens_light, source_light],
                 background_rms=0.2, exp_time=100)
    prob_model = ProbModel(model, ds, mode="forward")
    model_seq = ModellingSequence(prob_model)

    # --- qz: byte-identical rebuild of the reference SVI Gaussian ----------
    _assert_qz_not_stale()
    a = np.load(_require(_SVI_ARRAYS, "reference SVI stage arrays.npz"))
    qz_loc = np.asarray(a["qz_loc"], dtype=np.float64)
    qz_scale_tril = np.asarray(a["qz_scale_tril"], dtype=np.float64)
    dim = int(qz_loc.shape[0])
    if qz_loc.shape != (_EXPECTED_DIM,):
        raise ValueError(f"[sys60] qz_loc shape {qz_loc.shape} != ({_EXPECTED_DIM},)")
    if qz_scale_tril.shape != (_EXPECTED_DIM, _EXPECTED_DIM):
        raise ValueError(
            f"[sys60] qz_scale_tril shape {qz_scale_tril.shape} != "
            f"({_EXPECTED_DIM},{_EXPECTED_DIM})")
    qz = tfd.MultivariateNormalTriL(
        loc=jnp.asarray(qz_loc), scale_tril=jnp.asarray(qz_scale_tril))

    param_names = _recover_param_names(prob_model, dim)
    z_center = np.asarray(qz_loc)
    return model_seq, qz, z_center, dim, param_names
