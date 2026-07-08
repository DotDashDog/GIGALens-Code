#!/usr/bin/env python3
"""Run B: arm-initialized frozen-metric MCLMC (mechanism falsification).

Pre-registered design checkpoint: `docs/logs/sample-cosmology-dspl.md`, section
"Run B: arm-initialized frozen-metric MCLMC (mechanism falsification)".

Hypothesis under test (from the checkpoint / T2 mechanism analysis): the
baseline MCLMC run's truncation of the DSPL cosmology posterior at
Om0 ~= 0.15-0.163 is a soft reflection barrier caused by the FROZEN global
metric mis-tracking the rotating (Om0, w0) ridge past the crest (tangent-metric
mismatch measured at 2.9 deg in the bulk vs 28-34.5 deg at the crest/turnaround
in `docs/logs/sample-cosmology-dspl.md` T2). This script starts chains INSIDE
the unvisited arm (Om0=0.05, on the r2(truth) contour) and samples them under
the BASELINE run's own frozen metric (inverse mass matrix / step_size / L, no
further adaptation), to see whether they can cross back to the bulk.

Falsifier (pre-registered): mean >= 3 crest crossings per chain in 10000 steps
would be incompatible with a hard barrier (the baseline's ~64 bulk-side
approaches made ~8 excursions/chain with 0 crossings). See
`dspl_arm_init_analysis.py` for the crossing-counting analysis (run AFTER this
script, on its output).

Stages (mirrors `dspl_cosmology_newapi.ipynb` construction verbatim; all heavy
compute is gated behind `if __name__ == "__main__":` + `--run`):

  1. profile-MAP: rebuild the notebook's LensModel but with the cosmology
     Component's Om0/w0 FIXED as constants at the arm point (Om0=0.05,
     w0=w0_arm, the latter read off the r2(truth) contour in
     `results/sample_cosmology/dspl_cosmology_newapi/def_ratio_grid.npz`).
     MAP-fit the 19 remaining (nuisance) free parameters against the SAME
     simulated dataset as the baseline notebook (rendered at the notebook's
     TRUE cosmology, Om0=0.3/w0=-1.0 -- the arm point only fixes the *model*
     used for this profile fit, not the observed data).
  2. init assembly: build the FULL model (free cosmology, exactly the
     notebook's `model`) and assemble a 21-dim unconstrained z: the profile-MAP
     nuisance z's (19) in their slots + z(Om0=0.05, w0=w0_arm) via the model's
     own bijector, with the column order verified at runtime (never assumed).
  3. frozen-metric sampling: 8 chains started in a 1e-3-radius ball around that
     init z, 10000 steps, using the baseline run's final (inverse_mass_matrix,
     step_size, L) with NO further adaptation. `MCLMC_JIT` /
     `full_mclmc_with_adapt_sharded` (`src/gigalens_research/inference/mclmc.py`)
     CANNOT safely do this with frac_tune1=frac_tune2=frac_tune3=0 -- see the
     `frozen_metric_mclmc` docstring below for the exact code path and an
     empirical (not just code-read) confirmation. This script instead drives
     the same low-level kernel factory (`_build_kernel_shardmap` from
     `blackjax_updated_utils.py`) directly in a bare `jax.lax.scan`, matching
     MCLMC_JIT's own per-step mechanics (integrator, NaN handling) but with
     step_size/L/inverse_mass_matrix held fixed for every step.

A MANDATORY cheap CPU toy validation (3-D standard normal, 4 chains, 200 steps)
exercises this exact `frozen_metric_mclmc` function before it is pointed at the
real likelihood; run it with `--run toy` (safe on a login node, no GPU, no
`gigalens`/`photutils` heavy deps needed beyond jax/blackjax).

HARD SAFETY GATE: stages 1-3 ("map"/"assemble"/"sample"/"all") are real,
non-trivial compute (profile-MAP ~2 min per the notebook's timing; the 8-chain
10k-step frozen scan is roughly half the baseline run's cost) and require
`--confirm-run-b-approved`, which must ONLY be set after a grader has approved
the Run B checkpoint in `docs/logs/sample-cosmology-dspl.md`. This script was
authored WITHOUT ever being run in these real modes.

Environment: see `docs/env_setup.md` (Shifter `ghcr.io/nvidia/jax:jax-2026-04-13`
+ `gigalens_multinode_env` conda + sidecar PYTHONPATH) and the header of
`experiments/sample_cosmology/def_ratio_grid.py`. Run with the container's
`/usr/bin/python3`, not the conda env's own `python`.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions
tfb = tfp.bijectors


# ---------------------------------------------------------------------------
# System constants -- verbatim from dspl_cosmology_newapi.ipynb / def_ratio_grid.py.
# ---------------------------------------------------------------------------
Z_LENS = 0.5
Z_SOURCE1 = Z_LENS * 2  # == 1.0, the cosmology's z_source_ref
Z_SOURCE2 = Z_LENS * 3  # == 1.5, the plane whose deflection ratio r2 constrains cosmology

NUM_PIX, DELTA_PIX, EXP_TIME, BACKGROUND_RMS = 100, 0.065, 1000, 0.1

# The notebook's TRUE truth (used to render the observed data -- NOT the arm point;
# the arm point only fixes the *model* used for the Stage-1 profile-MAP fit).
TRUE_TRUTH_SCENE = {
    "planes": {
        0: {"geometry": {"redshift": Z_LENS},
            "mass": {0: {"theta_E": 1.1, "gamma": 2.0, "e1": 0.05, "e2": 0.02,
                         "center_x": 0.0, "center_y": 0.0}}},
        1: {"geometry": {"redshift": Z_SOURCE1},
            "light": {0: {"R_sersic": 0.25, "n_sersic": 2., "e1": 0.05, "e2": 0.,
                          "center_x": 0.05, "center_y": 0., "Ie": 50.}}},
        2: {"geometry": {"redshift": Z_SOURCE2},
            "light": {0: {"R_sersic": 1., "n_sersic": 6., "e1": 0.0, "e2": 0.05,
                          "center_x": 0., "center_y": 0.05, "Ie": 15.}}},
    },
    "cosmo": dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0, wa=0.0),
}

OM0_ARM = 0.05  # Stage-1/2/3 fixed/init Om0 (design checkpoint value)

HOME = os.path.expanduser("~")
BASELINE_DIR = os.path.join(HOME, "GIGALens-Code", "results", "sample_cosmology",
                            "dspl_cosmology_newapi")
BASELINE_DIAG_NPZ = os.path.join(BASELINE_DIR, "mclmc", "diagnostics.npz")
GRID_NPZ = os.path.join(BASELINE_DIR, "def_ratio_grid.npz")

ARM_DIR = os.path.join(HOME, "GIGALens-Code", "results", "sample_cosmology", "dspl_arm_init")
ARM_PROFILE_MAP_DIR = os.path.join(ARM_DIR, "profile_map")
ARM_SAMPLES_NPZ = os.path.join(ARM_DIR, "samples_z.npz")
# Candidate cached-dataset paths to check before regenerating (see make_observed_images).
_CACHED_DATASET_CANDIDATES = [
    os.path.join(BASELINE_DIR, "dataset.npz"),
    os.path.join(BASELINE_DIR, "observed_images.npz"),
]


# ---------------------------------------------------------------------------
# Model construction -- verbatim from dspl_cosmology_newapi.ipynb cell 5.
# ---------------------------------------------------------------------------
def build_components():
    """Returns (lens, source1, source2, cosmo) Components, verbatim from the notebook."""
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.cosmo import w0waCDM_Cosmo
    from gigalens.jax.scene import Component

    lens = Component(
        EPL(),
        dict(
            theta_E=tfd.LogNormal(jnp.log(1.25), 0.25),
            gamma=2.0,
            e1=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
            e2=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
            center_x=tfd.Normal(0, 0.05),
            center_y=tfd.Normal(0, 0.05),
        ),
    )

    def _make_source():
        return Component(
            SersicEllipse(use_lstsq=False),
            dict(
                center_x=tfd.Normal(0, 2),
                center_y=tfd.Normal(0, 2),
                e1=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
                e2=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
                n_sersic=tfd.Uniform(1, 10),
                R_sersic=tfd.LogNormal(jnp.log(1.), 0.15),
                Ie=tfd.LogNormal(jnp.log(150), 1),
            ),
        )

    source1 = _make_source()
    source2 = _make_source()

    def tNCDF_bij(low, high):
        return tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.NormalCDF()])

    class UniformBij(tfd.Uniform):
        def __init__(self, *args, event_space_bijector_class=tNCDF_bij, **kwargs):
            self._esb = event_space_bijector_class(*args)
            super().__init__(*args, **kwargs)

        def _default_event_space_bijector(self):
            return self._esb

    cosmo = Component(
        w0waCDM_Cosmo(z_lens=Z_LENS, z_source_ref=Z_SOURCE1),
        dict(
            H0=70.,
            Om0=UniformBij(jnp.float64(0.0), jnp.float64(1.0)),
            w0=UniformBij(jnp.float64(-2.0), jnp.float64(-1 / 3)),
            wa=0.0,
            k=0.0,
        ),
    )
    return lens, source1, source2, cosmo


def build_full_model():
    """The notebook's FULL model (free cosmology). Returns (model, lens, source1, source2)."""
    from gigalens.jax.scene import Plane, LensModel

    lens, source1, source2, cosmo = build_components()
    model = LensModel(
        [
            Plane(redshift=Z_LENS, mass=[lens]),
            Plane(redshift=Z_SOURCE1, light=[source1]),
            Plane(redshift=Z_SOURCE2, light=[source2]),
        ],
        cosmo=cosmo,
    )
    return model, lens, source1, source2


# ---------------------------------------------------------------------------
# w0_arm: interpolate the r2(truth) contour at Om0=OM0_ARM off def_ratio_grid.npz.
# ---------------------------------------------------------------------------
def compute_w0_arm(grid_npz=GRID_NPZ, om0_arm=OM0_ARM, verbose=True):
    """On-contour w0 at Om0=om0_arm: bilinear-interpolate r2_grid(Om0, w0) from the
    400x400 grid in `def_ratio_grid.npz` and bisect in w0 for r2 == r2_truth.

    Sanity check (per the design checkpoint): the resulting (Om0, w0) point must
    sit within the grid's own 99.7% highest-density region (`mass_levels[0]`) --
    it should, by construction, since the ENTIRE r2(truth) contour is
    (structurally, per T1) at essentially uniform density. Prints the probability
    at the arm point vs. all three mass_levels and raises if it fails the 99.7%
    check (a failure would mean the grid orientation / contour identification is
    wrong, not that "the arm is just low density").
    """
    with np.load(grid_npz) as d:
        Om0_grid = np.asarray(d["Om0_grid"])
        w0_grid = np.asarray(d["w0_grid"])
        r2_grid = np.asarray(d["r2_grid"])       # r2_grid[i, j] <-> (Om0_grid[i], w0_grid[j])
        prob = np.asarray(d["prob"])
        r2_truth = float(d["r2_truth"])
        mass_levels = np.asarray(d["mass_levels"])  # ascending: [99.7%, 95.5%, 68%] thresholds

    if not (Om0_grid.min() <= om0_arm <= Om0_grid.max()):
        raise ValueError(f"om0_arm={om0_arm} outside grid Om0 range "
                          f"[{Om0_grid.min()}, {Om0_grid.max()}]")

    def _bilinear(om0_val, w0_val, grid):
        i1 = int(np.searchsorted(Om0_grid, om0_val)); i0 = i1 - 1
        j1 = int(np.searchsorted(w0_grid, w0_val)); j0 = j1 - 1
        fx = (om0_val - Om0_grid[i0]) / (Om0_grid[i1] - Om0_grid[i0])
        fy = (w0_val - w0_grid[j0]) / (w0_grid[j1] - w0_grid[j0])
        v00, v01, v10, v11 = grid[i0, j0], grid[i0, j1], grid[i1, j0], grid[i1, j1]
        return (v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy)
                + v01 * (1 - fx) * fy + v11 * fx * fy)

    def f(w0_val):
        return _bilinear(om0_arm, w0_val, r2_grid) - r2_truth

    # Bracket: the r2(truth) contour at Om0 in [0, ~0.2] is single-valued in w0 (verified
    # by scanning several Om0 rows during development -- see docs/logs entry for Run B).
    lo, hi = w0_grid.min() + 1e-6, -0.5
    flo, fhi = f(lo), f(hi)
    if np.sign(flo) == np.sign(fhi):
        raise RuntimeError(
            f"compute_w0_arm: no sign change for r2-r2_truth over w0 in [{lo},{hi}] at "
            f"Om0={om0_arm} (f(lo)={flo}, f(hi)={fhi}) -- contour bracketing assumption "
            "violated; inspect def_ratio_grid_overlay.png before trusting this value."
        )
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if np.sign(fm) == np.sign(flo):
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    w0_arm = 0.5 * (lo + hi)
    prob_at_arm = float(_bilinear(om0_arm, w0_arm, prob))

    if verbose:
        print(f"[dspl_arm_init] w0_arm at Om0={om0_arm}: w0_arm={w0_arm:.6f} "
              f"(r2 residual={f(w0_arm):.3e})")
        print(f"[dspl_arm_init] prob(Om0={om0_arm}, w0={w0_arm:.4f}) = {prob_at_arm:.4e}; "
              f"mass_levels (99.7/95.5/68%% thresholds, ascending) = {mass_levels}")

    if not (prob_at_arm > mass_levels[0]):
        raise RuntimeError(
            f"compute_w0_arm: arm point prob={prob_at_arm:.3e} is BELOW the grid's own "
            f"99.7%% threshold ({mass_levels[0]:.3e}) -- this contradicts the T1 constancy "
            "argument (the whole r2(truth) contour should read as near-maximal density); "
            "check the Om0_mesh/w0_mesh orientation before trusting w0_arm."
        )
    if not (0.0 <= w0_arm <= 1.0) and not (-1.5 < w0_arm < -0.8):
        # Loose sanity band from the design checkpoint ("w0 ~= -1.0 to -1.3"); not a hard
        # gate (T2's own crest/edge values already span -0.9 to -1.27), just a loud flag.
        print(f"[dspl_arm_init] WARNING: w0_arm={w0_arm:.4f} is outside the checkpoint's "
              "expected -1.0..-1.3 ballpark; proceeding, but inspect def_ratio_grid.npz.")
    return w0_arm, prob_at_arm, mass_levels


# ---------------------------------------------------------------------------
# Baseline frozen metric (inverse_mass_matrix / step_size / L).
# ---------------------------------------------------------------------------
def load_baseline_frozen_metric(diag_npz=BASELINE_DIAG_NPZ, verbose=True):
    """Final shared (inverse_mass_matrix, step_size, L) from the baseline MCLMC run's
    debug diagnostics (`MCLMCStage(debug=True)` -> `<out_dir>/mclmc/diagnostics.npz`).

    `inverse_mass_matrix` has shape (1, num_burnin+num_results, dim, dim) -- ONE shared
    dense metric broadcast to all chains and logged only once per step (see
    `MCLMCStage.run`'s `diagnostics["inverse_mass_matrix"] = hist.inverse_mass_matrix[:1]`
    in `src/gigalens_research/inference_utils/pipeline.py`); `step_size`/`L` are logged
    per-chain but converge to an identical shared value after burn-in (step-size sync +
    shared L per `full_mclmc_with_adapt_sharded`). Asserts both facts rather than
    assuming them.
    """
    with np.load(diag_npz) as d:
        imm_hist = np.asarray(d["inverse_mass_matrix"])   # (1, T, dim, dim)
        step_size_hist = np.asarray(d["step_size"])        # (n_chains, T)
        L_hist = np.asarray(d["L"])                         # (n_chains, T)

    if imm_hist.shape[0] != 1:
        raise ValueError(f"expected inverse_mass_matrix leading dim 1, got {imm_hist.shape}")
    inverse_mass_matrix = imm_hist[0, -1]                    # (dim, dim)
    dim = inverse_mass_matrix.shape[-1]
    if inverse_mass_matrix.shape != (dim, dim):
        raise ValueError(f"expected square inverse_mass_matrix, got {inverse_mass_matrix.shape}")

    ss_final_per_chain = step_size_hist[:, -1]
    if not np.allclose(ss_final_per_chain, ss_final_per_chain[0]):
        raise ValueError(
            f"baseline step_size is NOT identical across chains at the final step "
            f"({ss_final_per_chain}); the 'single frozen shared step_size' assumption "
            "behind Run B is violated -- stop and re-check the baseline run."
        )
    step_size = float(ss_final_per_chain[0])

    L_final_per_chain = L_hist[:, -1]
    if not np.allclose(L_final_per_chain, L_final_per_chain[0]):
        raise ValueError(
            f"baseline L is NOT identical across chains at the final step "
            f"({L_final_per_chain}); the 'single frozen shared L' assumption behind "
            "Run B is violated -- stop and re-check the baseline run."
        )
    L = float(L_final_per_chain[0])

    evals = np.linalg.eigvalsh(0.5 * (inverse_mass_matrix + inverse_mass_matrix.T))
    if evals.min() <= 0:
        raise ValueError(f"baseline inverse_mass_matrix is not PSD (min eigval={evals.min()})")

    if verbose:
        print(f"[dspl_arm_init] baseline frozen metric: dim={dim} step_size={step_size:.6f} "
              f"L={L:.6f} imm_eig=[{evals.min():.3e}, {evals.max():.3e}]")
        # Cross-check against the values quoted in the design checkpoint (informational,
        # not a hard gate -- the checkpoint values were rounded).
        if abs(step_size - 0.152826) > 1e-4 or abs(L - 41.440) > 1e-2:
            print(f"[dspl_arm_init] NOTE: loaded step_size/L differ from the checkpoint's "
                  f"quoted 0.152826/41.440 by more than the expected rounding; double-check "
                  f"{diag_npz} is the same completed baseline run.")

    return inverse_mass_matrix, step_size, L


# ---------------------------------------------------------------------------
# Observed data (rendered at the TRUE cosmology, shared by Stage 1 and Stage 3).
# ---------------------------------------------------------------------------
def make_observed_images(full_model, lens, source1, source2, data_seed=0):
    """Render the two noisy bands EXACTLY as
    `dspl_cosmology_newapi.ipynb` ("Rendering the two 'bands'" cell), at the
    notebook's TRUE truth (Om0=0.3, w0=-1.0) -- NOT the arm point (the arm
    point only fixes the *sampling model* used in Stage 1, never the observed
    data). Returns (observed_image1, observed_image2, sim_config).

    First checks a couple of plausible on-disk cache paths (in case a previous
    run of this experiment saved the dataset); `ls` of
    `results/sample_cosmology/dspl_cosmology_newapi/` as of this writing shows
    NO such file, and the notebook itself never calls `np.random.seed(...)`
    before `lenstronomy.Util.image_util.add_poisson`/`add_background`, so the
    baseline run's own noise realization is NOT reproducible from code. This
    function therefore falls back to regenerating a FRESH realization with
    `np.random.seed(data_seed)` (default 0) -- a genuine, flagged deviation
    from bit-for-bit reproducing the completed baseline run's data. The frozen
    metric's *direction* (the degeneracy geometry we are testing against) is
    set by the smooth likelihood curvature, not by one noise draw, so this is
    not expected to change the mechanism-falsification result -- but that is
    an assumption, not a proof, and is logged as such.
    """
    for cand in _CACHED_DATASET_CANDIDATES:
        if os.path.exists(cand):
            print(f"[dspl_arm_init] found cached dataset at {cand}; loading it "
                  "instead of regenerating.")
            with np.load(cand) as d:
                img1, img2 = np.asarray(d["observed_image1"]), np.asarray(d["observed_image2"])
            break
    else:
        print(f"[dspl_arm_init] no cached dataset found in {_CACHED_DATASET_CANDIDATES}; "
              f"regenerating with np.random.seed({data_seed}) -- see docstring caveat: "
              "this will NOT bit-for-bit match the baseline notebook's (unseeded) "
              "noise realization.")
        import photutils.psf as psf
        from lenstronomy.Util import image_util
        from gigalens.jax.scene_simulator import SceneSimulator
        from gigalens.simulator import SimulatorConfig

        kernel = psf.GaussianPSF(x_fwhm=2, y_fwhm=2)
        yy, xx = np.mgrid[-7:8, -7:8]
        kernel = kernel(xx, yy)
        sim_config = SimulatorConfig(
            delta_pix=DELTA_PIX, num_pix=NUM_PIX, kernel=kernel, supersample=1,
            likelihood_precision="float64",
        )

        sim1 = SceneSimulator(full_model, sim_config, sees=[source1])
        sim2 = SceneSimulator(full_model, sim_config, sees=[source2])

        np.random.seed(data_seed)

        def add_noise(img, exp_time, sigma_bkd):
            poisson = image_util.add_poisson(img, exp_time=exp_time)
            bkg = image_util.add_background(img, sigma_bkd=sigma_bkd)
            return img + poisson + bkg

        img1 = np.asarray(sim1.simulate(TRUE_TRUTH_SCENE))
        img2 = np.asarray(sim2.simulate(TRUE_TRUTH_SCENE))
        img1 = add_noise(img1, exp_time=EXP_TIME, sigma_bkd=BACKGROUND_RMS)
        img2 = add_noise(img2, exp_time=EXP_TIME, sigma_bkd=BACKGROUND_RMS)
        return img1, img2, sim_config

    # cached-load path: still need a sim_config to build datasets/simulators.
    import photutils.psf as psf
    from gigalens.simulator import SimulatorConfig
    kernel = psf.GaussianPSF(x_fwhm=2, y_fwhm=2)
    yy, xx = np.mgrid[-7:8, -7:8]
    kernel = kernel(xx, yy)
    sim_config = SimulatorConfig(delta_pix=DELTA_PIX, num_pix=NUM_PIX, kernel=kernel,
                                  supersample=1, likelihood_precision="float64")
    return img1, img2, sim_config


def make_prob_model(model, comp1, comp2, img1, img2, sim_config):
    """Build a `forward`-mode ProbModel for `model`, seeing `comp1`/`comp2` -- these
    MUST be the light Component instances actually held by `model` (identity-matched
    by `ProbModel._resolve_sees`); `LensModel.fix_to` returns NEW Component instances,
    so the profile-MAP model's own `.planes[i].light[0]` must be passed, not the full
    model's original `source1`/`source2`."""
    from gigalens.jax.scene_prob_model import Dataset, ProbModel

    dataset1 = Dataset(img1, sim_config, background_rms=BACKGROUND_RMS, exp_time=EXP_TIME,
                        sees=[comp1])
    dataset2 = Dataset(img2, sim_config, background_rms=BACKGROUND_RMS, exp_time=EXP_TIME,
                        sees=[comp2])
    return ProbModel(model, [dataset1, dataset2], mode="forward")


# ---------------------------------------------------------------------------
# Stage 1: profile-MAP (cosmology fixed at the arm point).
# ---------------------------------------------------------------------------
def run_profile_map(w0_arm, img1, img2, sim_config, seed=1, out_dir=ARM_PROFILE_MAP_DIR):
    """Fix cosmology at (Om0=OM0_ARM, w0=w0_arm) via `LensModel.fix_to` (keeps the
    lens/source1/source2 priors verbatim, fixes ONLY the cosmo Component + plane
    geometry), then MAP-fit the 19 remaining nuisance params with the notebook's
    exact MAPStage settings (seed=1, num_steps=4000, n_samples=1000,
    pbar_interval=100, adabelief 1e-2 b1=0.95 b2=0.99, NO nesterov).

    Returns (profile_map_model, z_best (19,)).
    """
    import optax
    from gigalens.jax.inference import ModellingSequence
    from gigalens_research.inference_utils import InferenceContext, Pipeline, MAPStage

    full_model, lens, source1, source2 = build_full_model()

    arm_truth_scene = {
        "planes": {
            0: {"geometry": {"redshift": Z_LENS}},
            1: {"geometry": {"redshift": Z_SOURCE1}},
            2: {"geometry": {"redshift": Z_SOURCE2}},
        },
        "cosmo": dict(H0=70.0, Om0=OM0_ARM, k=0.0, w0=float(w0_arm), wa=0.0),
    }
    # LensModel.fix_to (gigalens/src/gigalens/jax/scene.py): fixes every param NOT
    # belonging to a `free` Component to its truth_scene value; lens/source1/source2 ARE
    # in `free`, so their (already-verbatim) priors pass through untouched, while cosmo
    # (not in `free`) is fixed entirely -- Om0/w0 to the arm point, H0/k/wa to their
    # existing constants (round-tripped through the same mechanism, no special-casing).
    profile_map_model = full_model.fix_to(arm_truth_scene, free=[lens, source1, source2])
    expected_free = full_model.num_free_params - 2
    if profile_map_model.num_free_params != expected_free:
        raise RuntimeError(
            f"profile_map_model.num_free_params={profile_map_model.num_free_params}, "
            f"expected {expected_free} (full model's {full_model.num_free_params} minus "
            "cosmo/Om0, cosmo/w0)."
        )
    if set(profile_map_model.z_param_names) != (set(full_model.z_param_names)
                                                 - {"cosmo/Om0", "cosmo/w0"}):
        raise RuntimeError(
            "profile_map_model.z_param_names does not equal full_model's minus "
            "{'cosmo/Om0','cosmo/w0'} -- fix_to changed something unexpected."
        )

    comp1 = profile_map_model.planes[1].light[0]
    comp2 = profile_map_model.planes[2].light[0]
    prob_model = make_prob_model(profile_map_model, comp1, comp2, img1, img2, sim_config)
    model_seq = ModellingSequence(prob_model)
    ctx = InferenceContext.from_modelling_sequence(model_seq)
    pipeline = Pipeline(ctx, seed=seed)

    def _old_map_optimizer():
        return optax.adabelief(1e-2, b1=0.95, b2=0.99)

    pipeline.add(MAPStage(
        num_steps=4000, n_samples=1000, pbar_interval=100, seed=1,
        optimizer_factory=_old_map_optimizer,
        optimizer_id="adabelief_1e-2_b1_0.95_b2_0.99_no_nesterov",
    ))
    artifacts = pipeline.run(out_dir=out_dir, resume=True)
    return profile_map_model, np.asarray(artifacts["z_best"])


# ---------------------------------------------------------------------------
# Stage 2: init assembly (19 profile-MAP nuisance z's + 2 cosmo z's -> 21-dim z).
# ---------------------------------------------------------------------------
def assemble_init_z(full_model, profile_map_model, z_best19, w0_arm):
    """Combine the profile-MAP's 19-dim unconstrained z with z(Om0=OM0_ARM, w0=w0_arm)
    (via the FULL model's own bijector) into the full 21-dim init z, verifying the
    column order/keys at runtime rather than assuming it (never reconstruct
    `z_param_names` order by hand -- see `gigalens/jax/scene.py::_z_param_names`)."""
    nuisance_phys = profile_map_model.bijector.forward(list(jnp.asarray(z_best19)))
    if set(nuisance_phys) != set(profile_map_model.z_param_names):
        raise RuntimeError("profile_map_model.bijector.forward key set mismatch")

    full_phys = dict(nuisance_phys)
    full_phys["cosmo/Om0"] = jnp.asarray(OM0_ARM, dtype=jnp.float64)
    full_phys["cosmo/w0"] = jnp.asarray(w0_arm, dtype=jnp.float64)

    missing = set(full_model.z_param_names) - set(full_phys)
    extra = set(full_phys) - set(full_model.z_param_names)
    if missing or extra:
        raise RuntimeError(f"init-z key mismatch: missing={missing} extra={extra}")

    z_init_list = full_model.bijector.inverse(full_phys)
    z_init = jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in z_init_list])
    if z_init.shape != (full_model.num_free_params,):
        raise RuntimeError(f"z_init shape {z_init.shape} != ({full_model.num_free_params},)")

    # Round-trip check: forward(z_init) must reproduce the intended physical values.
    round_trip = full_model.bijector.forward(list(z_init))
    om0_rt, w0_rt = float(round_trip["cosmo/Om0"]), float(round_trip["cosmo/w0"])
    if abs(om0_rt - OM0_ARM) > 1e-6 or abs(w0_rt - float(w0_arm)) > 1e-6:
        raise RuntimeError(
            f"init-z round-trip failed: got (Om0,w0)=({om0_rt},{w0_rt}), "
            f"expected ({OM0_ARM},{w0_arm})"
        )
    for name in profile_map_model.z_param_names:
        rt_val = float(np.asarray(round_trip[name]))
        want_val = float(np.asarray(nuisance_phys[name]))
        if abs(rt_val - want_val) > 1e-6 * max(1.0, abs(want_val)):
            raise RuntimeError(f"init-z round-trip failed for nuisance param {name!r}: "
                               f"got {rt_val}, expected {want_val}")

    idx_om0 = full_model.z_param_names.index("cosmo/Om0")
    idx_w0 = full_model.z_param_names.index("cosmo/w0")
    print(f"[dspl_arm_init] assembled init z: shape={z_init.shape}, "
          f"idx(cosmo/Om0)={idx_om0}, idx(cosmo/w0)={idx_w0}, "
          f"z[idx_om0]={float(z_init[idx_om0]):.6f}, z[idx_w0]={float(z_init[idx_w0]):.6f}")
    return z_init, idx_om0, idx_w0


# ---------------------------------------------------------------------------
# Stage 3: frozen-metric MCLMC (no adaptation) -- and its mandatory toy validation.
# ---------------------------------------------------------------------------
def frozen_metric_mclmc(log_prob, z_init, inverse_mass_matrix, step_size, L,
                         n_chains, num_results, ball_radius, seed):
    """Minimal frozen-metric MCLMC: NO step-size/L/mass-matrix adaptation, ever.

    WHY NOT `MCLMC_JIT` / `full_mclmc_with_adapt_sharded` with
    frac_tune1=frac_tune2=frac_tune3=0
    -----------------------------------------------------------------------
    `src/gigalens_research/inference/mclmc.py::full_mclmc_with_adapt_sharded`:
      - `num_steps1 = num_steps2 = num_steps3 = round(0 * num_burnin_steps) = 0`
        and `tuning_steps = 0` (line ~191-192).
      - `L_adaptation_step = tuning_steps = 0` (line ~195): `calc_new_L` is
        wired to fire via `jax.lax.cond(i == L_adaptation_step, calc_new_L,
        lambda _: params.L, ...)` (line ~439-444) -- i.e. AT STEP i=0, always,
        regardless of tuning fractions.
      - `l_stage_bufs_init = jnp.zeros((num_chains, num_steps3, dim))` (line
        ~484) -- with `num_steps3=0` this is an EMPTY buffer along the sample
        axis, and `l_buffer_start = L_adaptation_step - num_steps3 = 0` (line
        ~279), so `write_l_stage_buffer` (line ~357-359, itself gated on
        `mode==3`, which never occurs when all tuning fractions are 0 --
        `mode` is all-zeros, see line ~462-468) never populates it either.
      - `calc_new_L` (line ~429-437) unconditionally reads
        `l_stage_bufs[:, :, :]` via `_ess_shardmap(buf[None], ...)`
        (`blackjax_updated_utils.py`, ESS over an FFT-based autocovariance that
        divides by `num_samples`) on this EMPTY (chains, 0, dim) buffer.

      EMPIRICALLY CONFIRMED (not just code-read): calling
      `full_mclmc_with_adapt_sharded(..., frac_tune1=0, frac_tune2=0,
      frac_tune3=0)` on a 3-D toy raises, at TRACE time, inside
      `write_l_stage_buffer`'s `buf.at[:, buf_index].set(...)`:
      `IndexError: index is out of bounds for axis 1 with size 0` (the
      `l_stage_bufs` write path is reached even though `mode` never equals 3,
      because `jax.lax.cond`'s both branches are still TRACED against the same
      buffer shape). This is a hard, unconditional failure, not merely a risk
      of NaN propagation -- confirmed on 2026-07-07 CPU toy run (see
      `docs/logs/sample-cosmology-dspl.md`).

    A duck-typed `qz` wrapper around `MCLMC_JIT` would not help: the failure is
    inside `full_mclmc_with_adapt_sharded`'s tracing, upstream of anything `qz`
    controls. This function therefore drives the SAME low-level kernel factory
    (`_build_kernel_shardmap`, `blackjax_updated_utils.py`) directly in a bare
    `jax.lax.scan`, with NO adaptation machinery at all: step_size, L, and the
    (dense) inverse_mass_matrix are baked into the kernel/call and never
    change across the whole scan. Chain initialization reuses `init_multi`
    (same helper `MCLMC_JIT` itself uses). No `shard_map`/cross-device
    collectives are needed here (there is no adaptation state to synchronize
    across devices), so chains are simply `jax.vmap`-ed; this is a deliberate
    simplification relative to the baseline's multi-GPU sharded driver (Run B
    is a much smaller, diagnostic-only run) and is flagged here rather than
    silently assumed equivalent in cost/parallelism.

    Returns (positions, energy_changes, nonans), each with chain axis 0 and
    step axis 1 (canonical `(n_chains, num_results, dim)` / `(n_chains,
    num_results)` layout, matching `SamplerPosterior`'s convention).
    """
    from gigalens_research.inference.blackjax_updated_utils import (
        _build_kernel_shardmap, init_multi, isokinetic_mclachlan_smart,
    )

    dtype = jnp.asarray(z_init).dtype
    dim = z_init.shape[0]
    kernel = _build_kernel_shardmap(
        logdensity_fn=log_prob,
        integrator=isokinetic_mclachlan_smart,
        inverse_mass_matrix=jnp.asarray(inverse_mass_matrix, dtype=dtype),
    )

    rng_key = jax.random.key(seed)
    init_key, ball_key, run_key = jax.random.split(rng_key, 3)

    ball_keys = jax.random.split(ball_key, n_chains)
    starts = jax.vmap(
        lambda k: z_init + ball_radius * jax.random.normal(k, (dim,), dtype=dtype)
    )(ball_keys)

    # init_multi auto-splits a single key across chains when rng_keys.ndim == 0
    # (blackjax_updated_utils.py::init_multi) -- same call pattern MCLMC_JIT uses.
    state0 = init_multi(starts, init_key, log_prob)

    step_size_f = jnp.asarray(step_size, dtype=dtype)
    L_f = jnp.asarray(L, dtype=dtype)

    # Same key layout as full_mclmc_with_adapt_sharded: split(key, (n_chains, steps))
    # then moveaxis so axis 0 is the scan (step) axis and axis 1 is the chain axis.
    keys = jax.random.split(run_key, (n_chains, num_results))
    keys = jnp.moveaxis(keys, 0, 1)

    def step_one_chain(state, key):
        new_state, info = kernel(rng_key=key, state=state, L=L_f, step_size=step_size_f)
        return new_state, (new_state.position, info.energy_change, info.nonans)

    def step_all_chains(states, keys_t):
        return jax.vmap(step_one_chain, in_axes=(0, 0))(states, keys_t)

    _final_state, (positions, energy_changes, nonans) = jax.lax.scan(
        step_all_chains, state0, keys
    )
    positions = jnp.moveaxis(positions, 0, 1)          # (n_chains, num_results, dim)
    energy_changes = jnp.moveaxis(energy_changes, 0, 1)  # (n_chains, num_results)
    nonans = jnp.moveaxis(nonans, 0, 1)
    return positions, energy_changes, nonans


def run_toy_validation():
    """MANDATORY cheap CPU validation of `frozen_metric_mclmc`'s calling mechanics:
    a 3-D standard-normal logdensity, 4 chains, 200 steps. Asserts no NaNs and that
    every chain actually moves. Does NOT touch the real likelihood/model."""
    print("=== Stage 0: toy validation of frozen-metric MCLMC mechanics "
          "(CPU, 3-D standard normal, 4 chains, 200 steps) ===")
    dim = 3

    def toy_log_prob(z):
        return -0.5 * jnp.sum(jnp.square(z))

    t0 = time.perf_counter()
    positions, energy_changes, nonans = frozen_metric_mclmc(
        toy_log_prob,
        z_init=jnp.zeros(dim, dtype=jnp.float64),
        inverse_mass_matrix=jnp.eye(dim, dtype=jnp.float64),
        step_size=0.2,
        L=float(jnp.sqrt(dim)),
        n_chains=4,
        num_results=200,
        ball_radius=1e-3,
        seed=0,
    )
    positions = np.asarray(positions)
    energy_changes = np.asarray(energy_changes)
    nonans = np.asarray(nonans)
    dt = time.perf_counter() - t0

    assert positions.shape == (4, 200, dim), positions.shape
    assert np.all(np.isfinite(positions)), "toy validation FAILED: NaN/Inf in positions"
    assert np.all(np.isfinite(energy_changes)), "toy validation FAILED: NaN/Inf in energy_change"
    disp = np.linalg.norm(positions[:, -1, :] - positions[:, 0, :], axis=-1)
    assert np.all(disp > 1e-6), f"toy validation FAILED: a chain did not move (disp={disp})"

    print(f"  wall time: {dt:.2f}s")
    print(f"  per-chain displacement (step0 -> step200): {disp}")
    print(f"  nonan fraction: {nonans.mean():.6f}")
    print(f"  energy_change: mean={energy_changes.mean():.3e} std={energy_changes.std():.3e} "
          f"max|.|={np.abs(energy_changes).max():.3e}")
    print("=== toy validation PASSED: no NaNs, all chains moved ===")


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------
def _prepare_stage3_inputs(data_seed, seed_map):
    """Runs Stages 1-2 (real compute) and returns everything Stage 3 needs."""
    w0_arm, prob_at_arm, mass_levels = compute_w0_arm()
    inverse_mass_matrix, step_size, L = load_baseline_frozen_metric()

    full_model, lens, source1, source2 = build_full_model()
    img1, img2, sim_config = make_observed_images(full_model, lens, source1, source2,
                                                   data_seed=data_seed)

    profile_map_model, z_best19 = run_profile_map(
        w0_arm, img1, img2, sim_config, seed=seed_map, out_dir=ARM_PROFILE_MAP_DIR)

    z_init, idx_om0, idx_w0 = assemble_init_z(full_model, profile_map_model, z_best19, w0_arm)

    comp1_full, comp2_full = full_model.planes[1].light[0], full_model.planes[2].light[0]
    full_prob_model = make_prob_model(full_model, comp1_full, comp2_full, img1, img2, sim_config)

    def log_prob(z):
        return full_prob_model.log_prob(z)[0]

    return dict(
        w0_arm=w0_arm, inverse_mass_matrix=inverse_mass_matrix, step_size=step_size, L=L,
        full_model=full_model, z_init=z_init, idx_om0=idx_om0, idx_w0=idx_w0,
        log_prob=log_prob,
    )


def run_stage3(n_chains, num_results, ball_radius, seed_mclmc, data_seed, seed_map,
               out_npz=ARM_SAMPLES_NPZ):
    prep = _prepare_stage3_inputs(data_seed=data_seed, seed_map=seed_map)

    print(f"[dspl_arm_init] Stage 3: {n_chains} chains x {num_results} steps, "
          f"frozen step_size={prep['step_size']:.6f} L={prep['L']:.4f}, "
          f"ball_radius={ball_radius}, seed={seed_mclmc}")
    t0 = time.perf_counter()
    positions, energy_changes, nonans = frozen_metric_mclmc(
        prep["log_prob"], prep["z_init"], prep["inverse_mass_matrix"], prep["step_size"],
        prep["L"], n_chains=n_chains, num_results=num_results, ball_radius=ball_radius,
        seed=seed_mclmc,
    )
    dt = time.perf_counter() - t0
    print(f"[dspl_arm_init] Stage 3 done in {dt:.1f}s; nonan fraction={float(jnp.mean(nonans)):.6f}")

    os.makedirs(os.path.dirname(out_npz), exist_ok=True)
    np.savez(
        out_npz,
        samples_z=np.asarray(positions),
        energy_change=np.asarray(energy_changes),
        nonan=np.asarray(nonans),
        z_init=np.asarray(prep["z_init"]),
        inverse_mass_matrix=np.asarray(prep["inverse_mass_matrix"]),
        step_size=prep["step_size"],
        L=prep["L"],
        w0_arm=prep["w0_arm"],
        om0_arm=OM0_ARM,
        idx_om0=prep["idx_om0"],
        idx_w0=prep["idx_w0"],
        n_chains=n_chains,
        num_results=num_results,
        ball_radius=ball_radius,
        seed_mclmc=seed_mclmc,
        data_seed=data_seed,
        seed_map=seed_map,
        z_param_names=np.asarray(prep["full_model"].z_param_names),
        wall_time_s=dt,
    )
    print(f"[dspl_arm_init] wrote {out_npz}")
    return out_npz


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", choices=["toy", "map", "all"], default=None,
                         help="'toy': CPU-only mechanics validation (safe, no GPU/gigalens "
                              "heavy deps beyond jax/blackjax). 'map': run Stage 1+2 only "
                              "(profile-MAP + init assembly; prints z_init, does not sample). "
                              "'all': run Stages 1-3 end to end and save samples_z.npz. "
                              "Omit to do nothing (import/construction check only).")
    parser.add_argument("--confirm-run-b-approved", action="store_true",
                         help="Required for --run map/all. Set ONLY after a grader has "
                              "approved the Run B design checkpoint in "
                              "docs/logs/sample-cosmology-dspl.md.")
    parser.add_argument("--n-chains", type=int, default=8)
    parser.add_argument("--num-results", type=int, default=10000)
    parser.add_argument("--ball-radius", type=float, default=1e-3)
    parser.add_argument("--seed-map", type=int, default=1)
    parser.add_argument("--seed-mclmc", type=int, default=10)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--out-npz", default=ARM_SAMPLES_NPZ)
    args = parser.parse_args()

    if args.run is None:
        print("No --run given; nothing executed (guards against accidental heavy compute "
              "on a login node). Choose --run {toy,map,all}.")
        raise SystemExit(0)

    if args.run == "toy":
        run_toy_validation()
        raise SystemExit(0)

    if not args.confirm_run_b_approved:
        raise SystemExit(
            "Refusing to run a real stage ('map'/'all') without "
            "--confirm-run-b-approved. Run B (arm-initialized frozen-metric MCLMC) "
            "is a pre-registered design checkpoint awaiting grader approval -- see "
            "docs/logs/sample-cosmology-dspl.md. Do not set this flag yourself unless "
            "you ARE the grader approving the run."
        )

    print("!!! Running REAL compute for Run B (profile-MAP / MCLMC) -- confirmed via "
          "--confirm-run-b-approved !!!")

    if args.run == "map":
        w0_arm, _, _ = compute_w0_arm()
        _, step_size, L = load_baseline_frozen_metric()
        full_model, lens, source1, source2 = build_full_model()
        img1, img2, sim_config = make_observed_images(
            full_model, lens, source1, source2, data_seed=args.data_seed)
        profile_map_model, z_best19 = run_profile_map(
            w0_arm, img1, img2, sim_config, seed=args.seed_map, out_dir=ARM_PROFILE_MAP_DIR)
        z_init, idx_om0, idx_w0 = assemble_init_z(
            full_model, profile_map_model, z_best19, w0_arm)
        print("[dspl_arm_init] Stage 1+2 complete (no sampling run). z_init:")
        print(np.asarray(z_init))
    elif args.run == "all":
        run_stage3(
            n_chains=args.n_chains, num_results=args.num_results,
            ball_radius=args.ball_radius, seed_mclmc=args.seed_mclmc,
            data_seed=args.data_seed, seed_map=args.seed_map, out_npz=args.out_npz,
        )
