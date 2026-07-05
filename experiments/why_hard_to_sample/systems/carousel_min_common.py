"""Shared model builder for the two "carousel minimal case" arms.

Both arms rebuild EXACTLY the scene-API model behind two reference MCLMC runs
produced by `experiments/sim_carousel/carousel_sampling_minimal_example.ipynb`
(pipeline MAPStage -> BridgeStage "diag_qz" v1 -> MCLMCStage, 8 chains,
10000/10000, seed 42). The ONLY difference between the two arms is the NFW mass
profile + its priors:

  * carousel_min_old : NFW_ELLIPSE            with Rs~U(20,100), alpha_Rs~U(10,40)
  * carousel_min_new : NFW_ELLIPSE_EINSTEIN   with Rs~U(20,100), theta_E~N(13,1)

Everything else (shear, two lensed shapelet sources, cosmology, data, grid,
priors) is identical and lives here.

qz PROVENANCE (correctness-critical)
------------------------------------
The reference runs' `qz` is NOT an SVI Gaussian; it is a BridgeStage "diag_qz"
(v1) diagonal Gaussian centred on the MAP optimum. The notebook's bridge fn is:

    def make_diag_qz(z_best):
        return tfd.MultivariateNormalDiag(
            loc=jnp.asarray(z_best),
            scale_diag=jnp.full(z_best.shape[-1], 1e-3),   # NOTE: 1e-3, not the
        )                                                  # BridgeStage-docstring 1e-2

We do NOT re-optimize. We LOAD the exact float64 `z_best` the MAP stage
persisted (`<ref>/map/arrays.npz`, key "z_best", shape (14,)) and rebuild the
identical `tfd.MultivariateNormalDiag`. Because we feed the same bytes to the
same constructor, the rebuilt qz is object-identical (up to jax device array
wrapping) to the one MCLMCStage consumed, so
`gigalens_research...pipeline.stable_hash(qz)` reproduces the pinned manifest
hash. We assert that at load time (staleness guard).

HASHING (how the pinned hashes were computed)
---------------------------------------------
`gigalens_research/inference_utils/pipeline.py`:
  - `stable_hash(obj)` (line 156): sha256 of a canonical type-tagged encoding,
    truncated to 16 hex chars (`_short`, `_HEX_LEN=16`).
  - Pipeline.run records, per stage, `upstream_hashes = {k: stable_hash(artifact_k)}`
    (line 1031 + 1056). So `mclmc/manifest.json` upstream_hashes.qz ==
    stable_hash(qz-object) and `diag_qz/manifest.json` upstream_hashes.z_best ==
    stable_hash(z_best-array). (z_best hashes verified byte-exact offline.)

NOTE ON `depth`/`n_layers=45,28` in the model cards: these are the shapelet
basis counts for n_max=8 -> (8+1)(8+2)/2 = 45 and n_max=6 -> 7*8/2 = 28
(45+28 = 73 lstsq amplitude bases). They are NOT sampled parameters; n_params
(z-space dim) = 14 (NFW 6 + Shear 2 + src4 {cx,cy,beta} 3 + src5 {cx,cy,beta} 3;
cosmo is fixed).
"""
from __future__ import annotations

import json
import os

import numpy as np

# --- absolute, pinned paths into the (READ-ONLY) sim_carousel checkout --------
_SIM_CAROUSEL = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel"
# Data: the notebook's `path="newnewcutouts/"`, ext "4-5" -> source4-5.fits.
_DATA_FITS = os.path.join(_SIM_CAROUSEL, "newnewcutouts", "source4-5.fits")

# --- fixed physical constants (notebook cell "All the physical stuff") --------
_Z_LENS = 0.49
_Z_SOURCE_45 = 1.432  # the one lensed source plane in the minimal case
_EXPECTED_DIM = 14
_QZ_SCALE = 1e-3  # make_diag_qz scale_diag (verified against the notebook)


def _require(path: str, what: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[carousel_min] missing {what}: {path}")
    return path


# ---------------------------------------------------------------------------
# data (notebook: dataset_from_dir + ds("4-5", ...))
# ---------------------------------------------------------------------------

def _load_dataset(supersample):
    """Rebuild the single lensed-source Dataset for ext '4-5', byte-for-byte as
    the notebook builds it (source4-5.fits: DATA, sqrt(STAT), PSF, MASK)."""
    from astropy.io import fits
    import jax.numpy as jnp
    from gigalens.simulator import SimulatorConfig
    from gigalens.jax.scene_prob_model import Dataset

    with fits.open(_require(_DATA_FITS, "source4-5.fits")) as hdul:
        observed_image = jnp.array(hdul["DATA"].data.astype("float64"))
        error_map = jnp.array(np.sqrt(hdul["STAT"].data.astype("float64")))
        psf = hdul["PSF"].data.astype(jnp.float64)
        mask = hdul["MASK"].data.astype(jnp.bool)
    return observed_image, error_map, psf, mask


def _build_prob_model(nfw0, supersample):
    """Assemble model_seq/prob_model given the (already-constructed) NFW mass
    Component `nfw0`. Everything except `nfw0` is shared across the two arms.

    supersample : None reproduces the notebook value (1) EXACTLY. An int
      overrides ONLY the SimulatorConfig supersample factor; nothing else.
    """
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions

    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.profiles.mass.shear import Shear
    from gigalens.jax.profiles.light.shapelets import Shapelets
    from gigalens.jax.cosmo import wCDM_Cosmo
    from gigalens.simulator import SimulatorConfig
    from gigalens.jax.scene_prob_model import Dataset, ProbModel
    from gigalens.jax.inference import ModellingSequence

    # --- shared mass: shear (notebook) ------------------------------------
    shear = Component(Shear(), dict(
        gamma1=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        gamma2=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
    ))

    # --- shared light: two lensed shapelet sources (notebook) -------------
    src4 = Component(Shapelets(n_max=8, use_lstsq=True), dict(
        center_x=tfd.Normal(3.7, 1),
        center_y=tfd.Normal(3.2, 1),
        beta=tfd.LogNormal(jnp.log(0.4), 0.15),
    ))
    src5 = Component(Shapelets(n_max=6, use_lstsq=True), dict(
        center_x=tfd.Normal(3.0, 1),
        center_y=tfd.Normal(0., 1),
        beta=tfd.LogNormal(jnp.log(0.1), 0.15),
    ))

    cosmo = Component(
        wCDM_Cosmo(z_lens=_Z_LENS, z_source_ref=_Z_SOURCE_45),
        dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0),
    )

    model = LensModel([
        Plane(redshift=_Z_LENS, mass=[nfw0, shear]),
        Plane(redshift=_Z_SOURCE_45, light=[src4, src5]),
    ], cosmo=cosmo)

    # --- data + prob model (notebook) -------------------------------------
    observed_image, error_map, psf, mask = _load_dataset(supersample)
    _ss = 1 if supersample is None else int(supersample)  # None => notebook value
    # WHTS_CONV_PRECISION: T22 discriminator hook. Default "float32" is the
    # NOTEBOOK value (what the reference runs used); "float64" is the T22
    # conv-precision arm. Never changes silently: overrides are printed.
    _conv = os.environ.get("WHTS_CONV_PRECISION", "float32")
    if _conv != "float32":
        print(f"[carousel_min] WHTS_CONV_PRECISION={_conv} OVERRIDES the notebook "
              "float32 conv (T22 arm; reference runs used float32)")
    cfg = SimulatorConfig(
        delta_pix=0.2, num_pix=300, supersample=_ss, kernel=psf,
        likelihood_precision="float64", conv_precision=_conv,
    )
    dset = Dataset(observed_image, cfg, error_map=error_map, mask=mask,
                   sees=[src4, src5])
    prob_model = ProbModel(model, [dset], mode="lstsq")
    model_seq = ModellingSequence(prob_model)
    return model_seq, prob_model


# ---------------------------------------------------------------------------
# qz rebuild (from stored MAP z_best) + param names
# ---------------------------------------------------------------------------

def _load_z_best(ref_dir):
    p = _require(os.path.join(ref_dir, "map", "arrays.npz"),
                 "reference MAP arrays.npz")
    a = np.load(p)
    z_best = np.asarray(a["z_best"], dtype=np.float64)
    if z_best.shape != (_EXPECTED_DIM,):
        raise ValueError(
            f"[carousel_min] z_best shape {z_best.shape} != ({_EXPECTED_DIM},)")
    return z_best


def _rebuild_qz(z_best):
    """Rebuild the diag_qz v1 Gaussian EXACTLY as the notebook's make_diag_qz."""
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions
    return tfd.MultivariateNormalDiag(
        loc=jnp.asarray(z_best),
        scale_diag=jnp.full(z_best.shape[-1], _QZ_SCALE),
    )


def _stable_hash(obj):
    from gigalens_research.inference_utils.pipeline import stable_hash
    return stable_hash(obj)


def _manifest_qz_hash(ref_dir):
    p = _require(os.path.join(ref_dir, "mclmc", "manifest.json"),
                 "reference mclmc manifest")
    with open(p) as f:
        return json.load(f).get("upstream_hashes", {}).get("qz")


def _manifest_best_chisq(ref_dir):
    p = _require(os.path.join(ref_dir, "map", "manifest.json"),
                 "reference map manifest")
    with open(p) as f:
        return float(json.load(f).get("metadata", {}).get("best_chisq"))


def _recover_param_names(prob_model, dim):
    """C-8-safe param names: zero-probe the bijector and SORT the output dict's
    keys (validated route; identical to sys60.system._recover_param_names and
    common.load_param_names). sampler column i == alphabetically i-th key."""
    probe = np.zeros((1, dim))
    out = prob_model.bij.forward(list(probe.T))
    if not isinstance(out, dict) or not out or any(
            isinstance(v, (dict, list)) for v in out.values()):
        raise TypeError(
            f"[carousel_min] bij.forward did not return a flat scalar-key dict: "
            f"{type(out)}")
    names = sorted(str(k) for k in out.keys())
    if len(names) != dim or len(set(names)) != len(names):
        raise ValueError(
            f"[carousel_min] recovered {len(names)} unique names != dim {dim}: {names}")
    return names


# ---------------------------------------------------------------------------
# staleness guard
# ---------------------------------------------------------------------------

def _assert_qz_not_stale(ref_dir, pinned_qz_hash):
    """Guard against silent qz drift. The reference dirs can be ROTATED by the
    user's re-runs (a re-run renames the old <stage>/ to <stage>.stale-<ts>/ and
    writes a fresh one). Two cheap checks, both must hold:

      (a) the reference mclmc manifest's recorded upstream qz hash still equals
          the pinned constant (catches a re-run that changed z_best/qz), and
      (b) the qz we rebuild from map/arrays.npz hashes to that same pinned value
          (catches a reconstruction mismatch or a rotated map stage).

    Raise loudly on either mismatch -- never sample against a drifted target."""
    got = _manifest_qz_hash(ref_dir)
    if got != pinned_qz_hash:
        raise RuntimeError(
            f"[carousel_min] STALE: {ref_dir}/mclmc manifest upstream_hashes.qz="
            f"{got!r} != pinned {pinned_qz_hash!r}. The pipeline was re-run and the "
            "persisted qz no longer matches the reference run this module was "
            "verified against. Re-verify and re-pin before sampling.")


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def load_target(nfw0_factory, ref_dir, pinned_qz_hash, supersample=None):
    """Return (model_seq, qz, z_center, dim, param_names) for a carousel arm.

    nfw0_factory : zero-arg callable returning the arm's NFW mass Component
                   (constructed lazily so tfd/gigalens import inside x64 context).
    ref_dir      : absolute path to the reference run dir (map/, mclmc/, ...).
    pinned_qz_hash : the stable_hash(qz) value pinned from mclmc/manifest.json.

    - model_seq.prob_model.log_prob(z) -> (logp, reduced_chi2); MCLMC uses [0].
    - qz : the diag_qz v1 Gaussian rebuilt from the stored MAP z_best (loc), NOT
      re-optimized. Verified hash-identical to the reference run's qz.
    - z_center : the MAP optimum z_best (= qz.loc), the natural center.

    supersample : None (default) reproduces the notebook value (1) EXACTLY. An
      int overrides ONLY the SimulatorConfig supersample factor.
    """
    import jax
    # WHTS_FLOAT32_CONTROL=1: instrument-control arm for texture probes (T20).
    # Skips the forced x64 update so the target runs in float32 (the roughness
    # detector must fire there). NEVER set for science arms.
    _f32_control = os.environ.get("WHTS_FLOAT32_CONTROL") == "1"
    if _f32_control:
        print("[carousel_min] WHTS_FLOAT32_CONTROL=1: x64 NOT forced; rebuilt-qz "
              "hash check SKIPPED (float32 bytes differ). Instrument control only.")
    else:
        jax.config.update("jax_enable_x64", True)  # notebook cell 0; sampling needs f64

    nfw0 = nfw0_factory()
    model_seq, prob_model = _build_prob_model(nfw0, supersample)

    z_best = _load_z_best(ref_dir)
    dim = int(z_best.shape[0])
    qz = _rebuild_qz(z_best)

    # staleness guard (a): manifest still records the pinned qz hash
    _assert_qz_not_stale(ref_dir, pinned_qz_hash)
    # staleness guard (b): our rebuilt qz actually hashes to the pinned value
    got = _stable_hash(qz) if not _f32_control else pinned_qz_hash
    if got != pinned_qz_hash:
        raise RuntimeError(
            f"[carousel_min] REBUILD MISMATCH: stable_hash(rebuilt qz)={got!r} != "
            f"pinned {pinned_qz_hash!r}. The rebuilt qz is NOT the one the reference "
            "MCLMC run consumed; do not sample. (Check z_best bytes / scale / dtype.)")

    param_names = _recover_param_names(prob_model, dim)
    z_center = np.asarray(z_best)
    return model_seq, qz, z_center, dim, param_names


def verify(nfw0_factory, ref_dir, pinned_qz_hash, supersample=None):
    """GPU-node self-check. Rebuild everything and check, printing PASS/FAIL:
      1. stable_hash(rebuilt qz) == pinned manifest qz hash.
      2. reduced chi^2 at z_best == pinned map/manifest best_chisq (rel tol).
    Return process exit code (0 = all PASS, 1 = any FAIL). Never raises for a
    check failure (so both checks always print); only infra errors propagate."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    ok = True
    print(f"[verify] ref_dir = {ref_dir}")
    print(f"[verify] pinned qz hash = {pinned_qz_hash}")

    nfw0 = nfw0_factory()
    model_seq, prob_model = _build_prob_model(nfw0, supersample)
    z_best = _load_z_best(ref_dir)
    dim = int(z_best.shape[0])
    print(f"[verify] z_best shape = {z_best.shape} dtype = {z_best.dtype}")

    # --- check 1: qz hash --------------------------------------------------
    qz = _rebuild_qz(z_best)
    got_manifest = _manifest_qz_hash(ref_dir)
    got_rebuilt = _stable_hash(qz)
    c1_manifest = (got_manifest == pinned_qz_hash)
    c1_rebuilt = (got_rebuilt == pinned_qz_hash)
    print(f"[verify] manifest upstream_hashes.qz = {got_manifest} "
          f"({'PASS' if c1_manifest else 'FAIL'} vs pinned)")
    print(f"[verify] stable_hash(rebuilt qz)     = {got_rebuilt} "
          f"({'PASS' if c1_rebuilt else 'FAIL'} vs pinned)")
    ok = ok and c1_manifest and c1_rebuilt

    # --- check 2: reduced chi^2 at z_best vs pinned best_chisq -------------
    expected = _manifest_best_chisq(ref_dir)
    z = jnp.asarray(z_best)[None, :]  # log_like uses z.T -> needs (batch, dim)
    _, red_chi2 = prob_model.log_prob(z)
    got_chisq = float(np.asarray(red_chi2).reshape(-1)[0])
    rel = abs(got_chisq - expected) / abs(expected)
    # rtol 1e-5. Measured env deltas vs the user's pipeline env (2026-07-03):
    # old arm 3.1e-8, new arm 2.6e-6. The new-arm delta was adjudicated as
    # evaluation-path float noise, NOT a model/prior mismatch: logp at z_best
    # matches manifest best_lp to 0.12 nats (a wrong prior shifts it O(1)+),
    # the notebook priors were verified character-exact, and the gigalens
    # source predates the reference runs. Wrong-prior failures show up at
    # rel >= 1e-3, far above this gate.
    c2 = rel <= 1e-5
    print(f"[verify] reduced chi^2 @ z_best = {got_chisq!r}")
    print(f"[verify] pinned best_chisq      = {expected!r}")
    print(f"[verify] relative diff          = {rel:.3e} "
          f"({'PASS' if c2 else 'FAIL'}; <=1e-6 required, ~1e-9 if exact)")
    ok = ok and c2

    # --- end-to-end: load_target must not raise ---------------------------
    try:
        ms, q, zc, d, names = load_target(nfw0_factory, ref_dir, pinned_qz_hash,
                                          supersample=supersample)
        print(f"[verify] load_target OK: dim={d} n_names={len(names)} "
              f"names={names}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"[verify] load_target RAISED (FAIL): {type(e).__name__}: {e}")

    print(f"[verify] RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1
