"""Plain-Python builder for the carousel dPIE model.

Extracted verbatim (model/prior/dataset construction only — no inference/pipeline
code) from ``experiments/sim_carousel/prelim_sim_carousel.ipynb``. A previous agent
verified that cells 0, 1, 2, 3, 5, 6, 8, 9, 10 of that notebook define the model;
this script reproduces exactly that subset as importable Python:

  - cell 0 : jax x64 config
  - cell 1 : stdlib/astropy/jax/tfp imports
  - cell 2 : gigalens scene/profile/cosmo/prob-model imports
  - cell 3 : DiskEllipticity grouped-prior import
  - cell 5 : tNCDF_bij / UniformBij / TruncatedNormalBij helpers (defined in the
             notebook; not referenced by the priors actually used in cell 6, but
             kept for verbatim reproduction)
  - cell 6 : mass + light Components and their priors
  - cell 8 : cosmology + LensModel (mass plane z=0.49, two source planes)
  - cell 9 : dataset loader + Dataset/ProbModel construction
  - cell 10: ``model_seq = ModellingSequence(prob_model)`` (only the model-building
             line — the rest of that cell builds an InferenceContext/Pipeline and
             calls ``pipeline.run(...)``, which actually executes MAP/SVI/MAMS
             sampling and is inference, not model construction, so it is
             deliberately NOT reproduced here)

Data-path note: the notebook's cwd is ``experiments/sim_carousel/`` and it loads
FITS cutouts from a relative ``newnewcutouts/`` directory. That directory is
missing from this worktree (only the ``messy_tests/dpie`` MAMS npz payloads are
symlinked in) — see ``DATA_DIR`` below, which points read-only at the main
checkout's copy.
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
# constructor signature (image, sim_config, *, error_map=, mask=, sees=, ...).
# Fall back to Dataset on pre-refactor checkouts so old-env reproduction still works.
try:
    from gigalens.jax.scene_prob_model import ImageData as Dataset
except ImportError:
    from gigalens.jax.scene_prob_model import Dataset
from gigalens.simulator import SimulatorConfig
from gigalens.jax.inference import ModellingSequence

from gigalens.jax.grouped_priors import DiskEllipticity


# --------------------------------------------------------------------------- paths
# This script lives in experiments/flow_precond/; the notebook it was extracted
# from lives in experiments/sim_carousel/ and resolves data paths relative to that
# directory. Resolve explicitly off __file__ rather than relying on cwd.
_THIS_DIR = Path(__file__).resolve().parent
SIM_CAROUSEL_DIR = (_THIS_DIR.parent / "sim_carousel").resolve()

# READ-ONLY reference into the main checkout: this worktree does not have
# experiments/sim_carousel/newnewcutouts/ (only the messy_tests/dpie MAMS npz
# payloads are symlinked into this worktree). Do not write here.
DATA_DIR = Path(
    "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/newnewcutouts"
)

# In-worktree (symlinked) MAMS sample archive used by verify().
ARRAYS_NPZ = SIM_CAROUSEL_DIR / "messy_tests" / "dpie" / "mams" / "arrays.npz"

# Basin-slice diagnostic archive (scratch, not part of the repo).
BASIN_SLICE_NPZ = Path(
    "/pscratch/sd/l/linusu/carousel_diag/basin_slice/basin_slice.npz"
)


# ----------------------------------------------------------------- cell 5 verbatim
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

    #* Comes along with source 3
    # EPL_Ld = dict(
    #     theta_E = tfd.TruncatedNormal(1.6730331, 0.1, 1, 2.5),
    #     gamma = tfd.Uniform(1,3),
    #     e1 = tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
    #     e2 = tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
    #     center_x = tfd.Normal(11.80977389, 0.1),
    #     center_y = tfd.Normal(23.0283886, 0.1)
    # )

    # EPL_Lf = Component(EPL(50), {
    #    'center_x' : tfd.Normal(-15.10088063, 0.1),
    #    'center_y' : tfd.Normal(-4.66657821, 0.1),
    #     ('e1', 'e2') : DiskEllipticity(e_max=0.3, scale=0.1),
    #    # e1 = TruncatedNormalBij(0, 0.1, -0.5, 0.5),
    #    # e2 = TruncatedNormalBij(0, 0.1, -0.5, 0.5),
    #    'theta_E' : tfd.TruncatedNormal(0.8151327, 0.05, 0.2, 1.5),
    #    'gamma' : tfd.TruncatedNormal(2.2266, 0.5, 1, 3)
    # })

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

    return NFW0, EPL_Le, DPIE_Lf, shear, src4, src5, src9


def _build_lens_model(NFW0, EPL_Le, DPIE_Lf, shear, src4, src5, src9):
    """Cell 8 verbatim: cosmology + LensModel."""

    #* All the physical stuff
    z1 = 0.962
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
        Plane(redshift=z_lens, mass=[NFW0, EPL_Le, DPIE_Lf, shear]),  # the one deflector
        # Plane(redshift=z1, light=[src1]),    # nearer source plane
        # Plane(redshift=z3, light=[src3]),
        Plane(redshift=z4_5, light=[src4, src5]),
        Plane(redshift=z9, light=[src9]),
        # Plane(redshift=z12, light=[src12, src13]),
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
    """Reproduce prelim_sim_carousel.ipynb cells 0,1,2,3,5,6,8,9,10 verbatim.

    Returns:
        (model_seq, prob_model): a ``gigalens.jax.inference.ModellingSequence``
        wrapping the ``gigalens.jax.scene_prob_model.ProbModel`` built from the
        carousel dPIE LensModel and the two (4-5, 9) source-plane Datasets.
    """

    NFW0, EPL_Le, DPIE_Lf, shear, src4, src5, src9 = _build_components()
    model = _build_lens_model(NFW0, EPL_Le, DPIE_Lf, shear, src4, src5, src9)

    # cell 9: dataset loader + Dataset/ProbModel construction, verbatim except that
    # ``path`` is repointed from the notebook's relative "newnewcutouts/" (which is
    # cwd-relative to experiments/sim_carousel/ and absent from this worktree) to
    # the main checkout's copy of the same directory, read-only (see DATA_DIR above).
    path = str(DATA_DIR) + "/"

    def ds(ext, sees):
        observed_image, error_map, psf, mask = _dataset_from_dir(path, ext)

        cfg = SimulatorConfig(delta_pix=0.2, num_pix=300, supersample=1, kernel=psf,
                               likelihood_precision="float64", conv_precision="float32")
        dset = Dataset(observed_image, cfg, error_map=error_map, mask=mask, sees=sees)
        return dset

    # d1 = ds("1", sees=[src1])
    # d3 = ds("3", sees=[src3])
    d4_5 = ds("4-5", sees=[src4, src5])
    d9 = ds("9", sees=[src9])
    # d12_13 = ds("12-13", sees=[src12_13])
    # d8 = ds("8", sees=[src8])
    # d11 = ds("11", sees=[src11])

    prob_model = ProbModel(model, [d4_5, d9], mode="lstsq")

    # cell 10 (model-building line only; the rest of that cell builds and runs an
    # inference Pipeline, which is out of scope for a model builder).
    model_seq = ModellingSequence(prob_model)

    return model_seq, prob_model


# --------------------------------------------------------------------------- verify
def verify():
    model_seq, prob_model = build()
    model = prob_model.model

    print("=" * 78)
    print("carousel dPIE model card")
    print("=" * 78)
    print(f"SIM_CAROUSEL_DIR : {SIM_CAROUSEL_DIR}")
    print(f"DATA_DIR (RO)    : {DATA_DIR}")
    print(f"ARRAYS_NPZ       : {ARRAYS_NPZ}")
    print(f"BASIN_SLICE_NPZ  : {BASIN_SLICE_NPZ}")
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

    # (a) 33 free params.
    assert model.num_free_params == 33, (
        f"expected 33 free params, got {model.num_free_params}")
    print(f"\n[a] OK: model.num_free_params == 33")

    # (b) z_param_names[6] labels the second mass Component's center_x.
    name6 = prob_model.z_param_names[6]
    assert name6 == 'planes/0/mass/1/center_x', (
        f"expected z_param_names[6] == 'planes/0/mass/1/center_x', got {name6!r}")
    print(f"[b] OK: z_param_names[6] == {name6!r}")

    # (c) MAMS posterior samples: log_prob finite, red_chi2 in [1.15, 1.17].
    assert ARRAYS_NPZ.exists(), f"missing {ARRAYS_NPZ}"
    arrays = np.load(ARRAYS_NPZ)
    samples_z = arrays["samples_z"]
    assert samples_z.shape == (64, 1000, 33), (
        f"expected samples_z shape (64, 1000, 33), got {samples_z.shape}")

    sample_idxs = [(0, 0), (10, 500), (63, 999)]
    print(f"\n[c] MAMS posterior samples ({ARRAYS_NPZ}):")
    for (chain, step) in sample_idxs:
        z = jnp.asarray(samples_z[chain, step])
        lp, red_chi2 = prob_model.log_prob(z)
        lp = float(lp)
        red_chi2 = float(red_chi2)
        print(f"    chain={chain:2d} step={step:4d}  lp={lp:.6f}  red_chi2={red_chi2:.6f}")
        assert np.isfinite(lp), f"non-finite lp at chain={chain} step={step}: {lp}"
        assert np.isfinite(red_chi2), (
            f"non-finite red_chi2 at chain={chain} step={step}: {red_chi2}")
        assert 1.15 <= red_chi2 <= 1.17, (
            f"red_chi2 out of [1.15, 1.17] at chain={chain} step={step}: {red_chi2}")
    print("[c] OK: all 3 samples finite with red_chi2 in [1.15, 1.17]")

    # (d) basin-slice zP/zM: lp within +/-1 nat of recorded values.
    assert BASIN_SLICE_NPZ.exists(), f"missing {BASIN_SLICE_NPZ}"
    basin = np.load(BASIN_SLICE_NPZ)
    zP = jnp.asarray(basin["zP"])
    zM = jnp.asarray(basin["zM"])
    assert zP.shape == (33,) and zM.shape == (33,), (
        f"expected zP/zM shape (33,), got {zP.shape}/{zM.shape}")

    lpP, red_chi2P = prob_model.log_prob(zP)
    lpM, red_chi2M = prob_model.log_prob(zM)
    lpP, red_chi2P = float(lpP), float(red_chi2P)
    lpM, red_chi2M = float(lpM), float(red_chi2M)

    print(f"\n[d] basin-slice ({BASIN_SLICE_NPZ}):")
    print(f"    zP: lp={lpP:.6f}  red_chi2={red_chi2P:.6f}  (target -291319.81 +/-1)")
    print(f"    zM: lp={lpM:.6f}  red_chi2={red_chi2M:.6f}  (target -291325.24 +/-1)")

    assert np.isfinite(lpP) and np.isfinite(lpM)
    assert abs(lpP - (-291319.81)) <= 1.0, (
        f"zP lp {lpP} not within +/-1 nat of -291319.81")
    assert abs(lpM - (-291325.24)) <= 1.0, (
        f"zM lp {lpM} not within +/-1 nat of -291325.24")
    print("[d] OK: zP/zM lp within +/-1 nat of recorded values")

    print("\nALL CHECKS PASSED")
    return model_seq, prob_model


if __name__ == "__main__":
    verify()
