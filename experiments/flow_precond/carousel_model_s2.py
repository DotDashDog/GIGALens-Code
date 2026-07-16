"""Plain-Python builder for the SECOND carousel lens system ("1_2_3_4_5_9").

Extracted verbatim (model/prior/dataset construction only — no inference/pipeline
code) from ``experiments/sim_carousel/debug_carousel_1_3_4_5_9.ipynb``. That
notebook's cells define this model; this script reproduces exactly that subset
as importable Python, mirroring the conventions of ``carousel_model.py``
(the first, dPIE-only, 2-dataset carousel system) but for a DIFFERENT,
richer 5-mass-component / 4-dataset system:

  - cell 0 : jax x64 config
  - cell 1 : stdlib/astropy/jax/tfp/matplotlib imports (matplotlib omitted here —
             this module is a builder, not a plotting notebook)
  - cell 2 : gigalens scene/profile/cosmo/prob-model imports
  - cell 3 : DiskEllipticity grouped-prior import + AdaptiveImageData/
             plot_factor_map (the notebook uses ``AdaptiveImageData``, NOT the
             plain ``Dataset``/``ImageData`` that ``carousel_model.py`` uses,
             for its datasets — plot_factor_map is plotting-only, not imported
             here)
  - cell 6 : mass + light Components and their priors (5 mass components:
             NFW_ELLIPSE_SLOPE, DPIE, EPL(18), DPIE, Shear; 5 light components:
             src1, src3, src4, src5, src9 — src2 is commented out in the
             notebook and is NOT part of this model)
  - cell 8 : cosmology + LensModel (mass plane z=0.49, four source planes at
             z=0.962, 1.166, 1.432, 1.506)
  - cell 9 : dataset loader (``dataset_from_dir``) + ``ds()`` helper building
             ``AdaptiveImageData`` + ``ProbModel(model, [d1, d3, d4_5, d9],
             mode="lstsq")`` construction
  - cell 12: ``model_seq = ModellingSequence(prob_model)`` (only the
             model-building line — the rest of that cell builds an
             InferenceContext/Pipeline and calls ``pipeline.run(...)``, which
             actually executes MAP/Bridge/MCLMC stages and is inference, not
             model construction, so it is deliberately NOT reproduced here)

Data-path note: the notebook's cwd is ``experiments/sim_carousel/`` and it loads
FITS cutouts from a relative ``newnewcutouts/`` directory (the SAME directory
``carousel_model.py`` reads, since sources 1/3/4-5/9 are shared raw cutouts
across both carousel lens systems — only the *priors* and *plane layout*
differ). See ``DATA_DIR`` below, which points read-only at the main checkout's
copy, exactly as ``carousel_model.py`` does.

Results-archive note: the notebook writes MAP/Bridge/MCLMC pipeline artifacts
to ``debug_carousel/1_2_3_4_5_9`` (relative to ``experiments/sim_carousel/``);
see ``S2_ROOT`` / ``MAP_NPZ_S2`` / ``MCLMC_NPZ_S2`` below.
"""

import os
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

from astropy.io import fits
import numpy as np
import jax.numpy as jnp

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors

from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.mass.shear import Shear
from gigalens.jax.profiles.mass.nfw import NFW_ELLIPSE, NFW_ELLIPSE_EINSTEIN
from gigalens.jax.profiles.mass.nfw_ellipse_slope import NFW_ELLIPSE_SLOPE
from gigalens.jax.profiles.mass.piemd import DPIE

from gigalens.jax.profiles.light.sersic import SersicEllipse
from gigalens.jax.profiles.light.shapelets import Shapelets

from gigalens.jax.cosmo import wCDM_Cosmo

from gigalens.jax.scene_prob_model import ProbModel
# 2026-07-10: upstream gigalens (linusu-dev-merge @698b990, PRs #32/#33) refactored
# Dataset into an ABC; the concrete imaging class is now ImageData with an identical
# constructor signature (image, sim_config, *, error_map=, mask=, sees=, ...). This
# module's ``AdaptiveImageData`` (cell 3) subclasses whichever concrete class the
# installed gigalens ships (post-refactor: ImageData), so no separate fallback is
# needed for it; the try/except below mirrors carousel_model.py only for parity /
# in case some other code path in this module ever needs the bare name.
try:
    from gigalens.jax.scene_prob_model import ImageData as Dataset
except ImportError:
    from gigalens.jax.scene_prob_model import Dataset
from gigalens.simulator import SimulatorConfig
from gigalens.jax.inference import ModellingSequence

from gigalens.jax.grouped_priors import DiskEllipticity
from gigalens.jax.experimental.adaptive_supersample import AdaptiveImageData


# --------------------------------------------------------------------------- paths
# This script lives in experiments/flow_precond/; the notebook it was extracted
# from lives in experiments/sim_carousel/ and resolves data paths relative to that
# directory. Resolve explicitly off __file__ rather than relying on cwd.
_THIS_DIR = Path(__file__).resolve().parent
SIM_CAROUSEL_DIR = (_THIS_DIR.parent / "sim_carousel").resolve()

# READ-ONLY reference into the main checkout: this worktree does not have
# experiments/sim_carousel/newnewcutouts/. Do not write here. Same directory
# carousel_model.py reads (the raw source cutouts are shared across both
# carousel lens systems).
DATA_DIR = Path(
    "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/newnewcutouts"
)

# Second-system results archive (MAP/Bridge/MCLMC pipeline artifacts), written by
# the notebook's ``pipeline.run(out_dir="debug_carousel/1_2_3_4_5_9", ...)``.
S2_ROOT = Path(
    "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/debug_carousel"
    "/1_2_3_4_5_9"
)
MAP_NPZ_S2 = S2_ROOT / "map" / "arrays.npz"
MCLMC_NPZ_S2 = S2_ROOT / "mclmc" / "arrays.npz"
DIM_S2 = 46


# ----------------------------------------------------------------- cell 5 verbatim
# (carried over unchanged from carousel_model.py / the notebook's cell 5 — not
# referenced by the priors actually used below, kept only for verbatim parity.)
def tNCDF_bij(low, high):
    return tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.NormalCDF()])


class UniformBij(tfd.Uniform):
    def __init__(self, *args, event_space_bijector_class=tNCDF_bij, **kwargs):
        self._esb = event_space_bijector_class(*args)
        super().__init__(*args, **kwargs)

    def _default_event_space_bijector(self):
        return self._esb


class TruncatedNormalBij(tfd.TruncatedNormal):
    def __init__(self, *args, event_space_bijector_class=tNCDF_bij, **kwargs):
        args = [jnp.float64(arg) for arg in args]
        low, high = args[2], args[3]
        self._esb = event_space_bijector_class(low, high)
        super().__init__(*args, **kwargs)

    def _default_event_space_bijector(self):
        return self._esb


def _build_components():
    """Cell 6 verbatim: mass + light Components with their priors."""

    #* MASS PRIORS
    NFW0 = Component(NFW_ELLIPSE_SLOPE(), dict(
        theta_E=tfd.Normal(13, 1),  # alpha_Rs = tfd.Uniform(10,40),
        s_E=tfd.Uniform(0, 0.75),
        e1=tfd.TruncatedNormal(0, 0.05, -0.2, 0.2),
        e2=tfd.TruncatedNormal(0, 0.05, -0.2, 0.2),
        center_x=tfd.Normal(5.344, 0.05),
        center_y=tfd.Normal(3.805, 0.05)
    ))

    EPL_Le = Component(EPL(18), {
        'theta_E': tfd.TruncatedNormal(2.4, 0.1, 1, 3),
        'gamma': tfd.TruncatedNormal(2.2, 0.5, 1, 3),
        ('e1', 'e2'): DiskEllipticity(e_max=0.3, scale=0.1),
        # e1 = TruncatedNormalBij(0, 0.1, -0.5, 0.5),
        # e2 = TruncatedNormalBij(0, 0.1, -0.5, 0.5),
        'center_x': tfd.Normal(-22.1, 0.1),
        'center_y': tfd.Normal(-24.7, 0.1)
    })

    DPIE_Ld = Component(DPIE(), {
        'theta_E': tfd.TruncatedNormal(1.6730331, 0.1, 1, 2.5),
        'r_core': tfd.Uniform(0, 1),
        'r_cut': tfd.Uniform(1, 20),
        'center_x': tfd.Normal(11.80977389, 0.1),
        'center_y': tfd.Normal(23.0283886, 0.1),
        ('e1', 'e2'): DiskEllipticity(e_max=0.3, scale=0.05),
        # e1 = tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
        # e2 = tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
    })

    DPIE_Lf = Component(DPIE(), {
        'theta_E': tfd.TruncatedNormal(0.8151327, 0.05, 0.2, 1.5),
        'r_core': tfd.Uniform(0, 1),
        'r_cut': tfd.Uniform(1, 20),
        'center_x': tfd.Normal(-15.10088063, 0.1),
        'center_y': tfd.Normal(-4.66657821, 0.1),
        ('e1', 'e2'): DiskEllipticity(e_max=0.3, scale=0.1),
        # e1 = tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
        # e2 = tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
    })

    shear = Component(Shear(), dict(
        gamma1=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        gamma2=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
    ))

    #* LIGHT PRIORS
    src1 = Component(Shapelets(n_max=8, use_lstsq=True), dict(
        center_x=tfd.Normal(7.67187389, 2),
        center_y=tfd.Normal(3.31911655, 2),
        beta=tfd.LogNormal(jnp.log(0.4), 0.15),
    ))

    # src2 commented out in the notebook: NOT part of this model.
    # src2 = Component(SersicEllipse(use_lstsq=True), dict(
    #     center_x = tfd.Normal(10, 2),
    #     center_y = tfd.Normal(3., 2),
    #     R_sersic = tfd.LogNormal(jnp.log(0.4), 0.15),
    #     n_sersic = tfd.Uniform(0.5,15),
    #     e1 = tfd.Normal(0,0.1),
    #     e2 = tfd.Normal(0,0.1),
    # ))

    src3 = Component(Shapelets(n_max=8, use_lstsq=True), {  # default n_max=12
        'center_x': tfd.Normal(5., 1),
        'center_y': tfd.Normal(5., 1),
        'beta': tfd.LogNormal(jnp.log(0.4), 0.15)
    })

    src4 = Component(Shapelets(n_max=8, use_lstsq=True), dict(  # default n_max=12
        center_x=tfd.Normal(3.7, 1),
        center_y=tfd.Normal(3.2, 1),
        beta=tfd.LogNormal(jnp.log(0.4), 0.15),
    ))

    src5 = Component(Shapelets(n_max=6, use_lstsq=True), dict(
        center_x=tfd.Normal(3.0, 1),
        center_y=tfd.Normal(0., 1),
        beta=tfd.LogNormal(jnp.log(0.1), 0.15),
    ))

    src9 = Component(SersicEllipse(use_lstsq=True), {  # default n_max=12
        'center_x': tfd.Normal(-10, 1),
        'center_y': tfd.Normal(-16, 1),
        'n_sersic': tfd.Uniform(0.1, 10),
        'R_sersic': tfd.LogNormal(jnp.log(0.4), 0.15),
        ('e1', 'e2'): DiskEllipticity(e_max=0.3, scale=0.1),
        # e1 = tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
        # e2 = tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
    })

    return NFW0, EPL_Le, DPIE_Ld, DPIE_Lf, shear, src1, src3, src4, src5, src9


def _build_lens_model(NFW0, EPL_Le, DPIE_Ld, DPIE_Lf, shear,
                       src1, src3, src4, src5, src9):
    """Cell 8 verbatim: cosmology + LensModel."""

    #* All the physical stuff
    z1_2 = 0.962
    z3 = 1.166
    z4_5 = 1.432
    z9 = 1.506
    z12_13 = 3.086
    z8 = 3.549
    z11 = 4.090

    z_lens = 0.49

    # Om0_prior = tfd.Uniform(0, 1)
    # w0_prior = tfd.Uniform(-2, -1/3)
    cosmo = Component(wCDM_Cosmo(z_lens=z_lens, z_source_ref=z4_5),
                       dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0))

    model = LensModel([
        Plane(redshift=z_lens, mass=[NFW0, DPIE_Ld, EPL_Le, DPIE_Lf, shear]),  # the one deflector
        Plane(redshift=z1_2, light=[src1]),    # nearer source plane
        Plane(redshift=z3, light=[src3]),
        Plane(redshift=z4_5, light=[src4, src5]),
        Plane(redshift=z9, light=[src9]),
        # Plane(redshift=z12_13, light=[src12, src13]),
        # Plane(redshift=z8, light=[src8]),
        # Plane(redshift=z11, light=[src11]),
    ], cosmo=cosmo)

    return model


def _dataset_from_dir(path, ext):
    """Cell 9 verbatim (``dataset_from_dir``)."""

    img_path = os.path.join(path, f"source{ext}.fits")
    with fits.open(img_path) as hdul:
        observed_image = jnp.array(hdul['DATA'].data.astype("float64"))

        error_map = jnp.array(np.sqrt(hdul['STAT'].data.astype("float64")))
        # background_rms = hdul['DATA'].header['BKG_RMS']
        # exp_time = hdul['PRIMARY'].header['EXPTIME']
        # if centroids is None: (self.centroids_x, self.centroids_y) = Table(hdul['CENTROIDS'].data)['centroid'].data.T
        # if centroids_error is None: self.centroids_error = Table(hdul['CENTROIDS'].data)['sky_covariance'].data
        psf = hdul['PSF'].data.astype(jnp.float64)
        mask = hdul['MASK'].data.astype(jnp.bool)
        # hot_pix = jnp.load(os.path.join(path, f"hot_pix.npy"))

    # mask = jnp.logical_and(mask, hot_pix)

    return observed_image, error_map, psf, mask


def build():
    """Reproduce debug_carousel_1_3_4_5_9.ipynb cells 0,1,2,3,5,6,8,9,12 verbatim.

    Returns:
        (model_seq, prob_model): a ``gigalens.jax.inference.ModellingSequence``
        wrapping the ``gigalens.jax.scene_prob_model.ProbModel`` built from the
        second carousel LensModel (5 mass components on one deflector plane;
        four source planes carrying src1, src3, src4+src5, src9) and the four
        (1, 3, 4-5, 9) source-plane ``AdaptiveImageData`` datasets.
    """

    NFW0, EPL_Le, DPIE_Ld, DPIE_Lf, shear, src1, src3, src4, src5, src9 = (
        _build_components())
    model = _build_lens_model(
        NFW0, EPL_Le, DPIE_Ld, DPIE_Lf, shear, src1, src3, src4, src5, src9)

    # cell 9: dataset loader + Dataset/ProbModel construction, verbatim except that
    # ``path`` is repointed from the notebook's relative "newnewcutouts/" (which is
    # cwd-relative to experiments/sim_carousel/ and absent from this worktree) to
    # the main checkout's copy of the same directory, read-only (see DATA_DIR above).
    path = str(DATA_DIR) + "/"

    def ds(ext, sees):
        observed_image, error_map, psf, mask = _dataset_from_dir(path, ext)

        cfg = SimulatorConfig(delta_pix=0.2, num_pix=300, supersample=1, kernel=psf,
                               likelihood_precision="float64", conv_precision="float32")
        # NOTE: the notebook uses AdaptiveImageData here (NOT the plain Dataset/
        # ImageData that carousel_model.py's ``ds()`` uses) — see cell 3/9.
        dset = AdaptiveImageData(observed_image, cfg, error_map=error_map, mask=mask,
                                  sees=sees)
        return dset

    d1 = ds("1", sees=[src1])
    d3 = ds("3", sees=[src3])
    d4_5 = ds("4-5", sees=[src4, src5])
    d9 = ds("9", sees=[src9])
    # d12_13 = ds("12-13", sees=[src12_13])
    # d8 = ds("8", sees=[src8])
    # d11 = ds("11", sees=[src11])

    prob_model = ProbModel(model, [d1, d3, d4_5, d9], mode="lstsq")

    # cell 12 (model-building line only; the rest of that cell builds and runs an
    # inference Pipeline, which is out of scope for a model builder).
    model_seq = ModellingSequence(prob_model)

    return model_seq, prob_model


# --------------------------------------------------------------------------- verify
def verify():
    model_seq, prob_model = build()
    model = prob_model.model

    print("=" * 78)
    print("carousel S2 (1_2_3_4_5_9) model card")
    print("=" * 78)
    print(f"SIM_CAROUSEL_DIR : {SIM_CAROUSEL_DIR}")
    print(f"DATA_DIR (RO)    : {DATA_DIR}")
    print(f"S2_ROOT          : {S2_ROOT}")
    print(f"MAP_NPZ_S2       : {MAP_NPZ_S2}")
    print(f"MCLMC_NPZ_S2     : {MCLMC_NPZ_S2}")
    print(f"num_free_params  : {model.num_free_params}")
    print(f"num datasets     : {len(prob_model.datasets)}")
    for i, d in enumerate(prob_model.datasets):
        cfg = d.sim_config
        print(f"  dataset[{i}] image.shape={tuple(d.image.shape)} "
              f"event_size={d.event_size} delta_pix={cfg.delta_pix} "
              f"num_pix={cfg.num_pix} supersample={cfg.supersample} "
              f"likelihood_precision={cfg.likelihood_precision} "
              f"conv_precision={cfg.conv_precision} kernel.shape={cfg.kernel.shape}")
    print(f"mode             : {prob_model.mode}")
    print(f"planes           : {[p.redshift for p in model.planes]}")

    # (a) 46 free params. NOTE: ``num_free_params`` lives on the LensModel
    # (``prob_model.model.num_free_params``), NOT on ProbModel itself — ProbModel
    # has no such attribute/property (verified against
    # gigalens/src/gigalens/jax/scene_prob_model.py). This mirrors exactly how
    # carousel_model.py's verify() reads it (via ``model.num_free_params``).
    assert model.num_free_params == DIM_S2, (
        f"expected {DIM_S2} free params, got {model.num_free_params}")
    print(f"\n[a] OK: model.num_free_params == {DIM_S2}")

    # (b) full z_param_names index->name mapping (not stored anywhere on disk).
    print(f"\n[b] z_param_names (n={len(prob_model.z_param_names)}):")
    for i, name in enumerate(prob_model.z_param_names):
        print(f"    [{i:2d}] {name}")
    print(f"    z_param_names[37] = {prob_model.z_param_names[37]!r}")
    print(f"    z_param_names[27] = {prob_model.z_param_names[27]!r}")

    # (c) HARD GATE: log_prob at the stored MAP z_best vs. the recorded MAP
    # manifest (S2_ROOT/map/manifest.json) best_lp / best_chisq. Print the
    # comparison; do NOT loosen a tolerance to force a pass or adjust the model.
    assert MAP_NPZ_S2.exists(), f"missing {MAP_NPZ_S2}"
    map_arrays = np.load(MAP_NPZ_S2)
    z_best = map_arrays["z_best"]
    assert z_best.shape == (DIM_S2,), (
        f"expected z_best shape ({DIM_S2},), got {z_best.shape}")

    z_best = jnp.asarray(z_best)
    lp, red_chi2 = prob_model.log_prob(z_best)
    lp = float(lp)
    red_chi2 = float(red_chi2)

    best_lp = -521149.3527310094
    best_chisq = 1.148182863187924

    abs_diff_lp = abs(lp - best_lp)
    rel_diff_lp = abs_diff_lp / abs(best_lp)
    abs_diff_chisq = abs(red_chi2 - best_chisq)
    rel_diff_chisq = abs_diff_chisq / abs(best_chisq)

    print("\n[c] HARD GATE: log_prob(z_best) vs. MAP manifest "
          f"({S2_ROOT / 'map' / 'manifest.json'}):")
    print(f"    computed lp        = {lp!r}")
    print(f"    manifest best_lp   = {best_lp!r}")
    print(f"    abs diff (lp)      = {abs_diff_lp!r}")
    print(f"    rel diff (lp)      = {rel_diff_lp!r}")
    print(f"    computed red_chi2  = {red_chi2!r}")
    print(f"    manifest best_chisq= {best_chisq!r}")
    print(f"    abs diff (chi2)    = {abs_diff_chisq!r}")
    print(f"    rel diff (chi2)    = {rel_diff_chisq!r}")
    print("    (no pass/fail tolerance asserted here by design — this is a "
          "FINDING to report, not a gate to paper over.)")

    print("\nVERIFY DONE (report the [c] comparison above; do not claim "
          "certification from this printout alone)")
    return model_seq, prob_model


if __name__ == "__main__":
    verify()
