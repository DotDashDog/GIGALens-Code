"""Lensed point-source SBC campaign: lenstronomy truth sim -> gigalens fit.

This module registers everything ``experiments/hundred_point_sources/campaign.yaml``
references:

- ``"lenstronomy_point_source"`` generator — draws EPL+Shear truths (and a
  per-system H0) from the campaign's truth prior, solves the lens equation with
  lenstronomy (independent of the gigalens solver under test), simulates noisy
  image positions / fluxes / time delays, and persists every quantity the
  likelihood must agree with (observables, sigmas, redshifts, the exact prior
  spec) into each system's ``truth_assets``.
- ``"epl_shear_point_source_obs"`` inference builder — reconstructs the SAME
  priors from the persisted spec (``priors: from_dataset``) and builds a
  :class:`~gigalens.jax.scene_prob_model.ProbModel` over
  :class:`~gigalens.jax.point_source_position.PointSourceObsData` (or the
  position-only dataset when both channels are off).
- ``"map_svi_mclmc"`` pipeline — MAPStage -> SVIStage -> MCLMCStage.
- ``"sbc_ranks"`` / ``"loglik_rank"`` / ``"ps_solver_health"`` per-run metrics.
- ``"sbc_uniformity"`` campaign metric — rank-ECDF uniformity analysis over
  converged runs, with attrition accounting.

SBC validity (see the campaign yaml header for the full discussion): truths are
drawn from the *inference* prior by default; the source is parameterized
absolutely so the generator draw and the fitted prior coincide; selection
(multiplicity/min-separation redraws) is recorded per system and in the
manifest.

Precision: the point-source likelihood requires float64. x64 is enabled at
module import so every downstream jax array (generation, fit, metrics) is
double precision.
"""
from __future__ import annotations

import json
import math
import os
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# The point-source lens-equation solve is float64-only (the Levenberg trust-region
# iteration underflows in float32). Set it at import: plugins are imported by the
# CLI before any jax array exists, so this governs generate, run, and metrics.
import jax
jax.config.update("jax_enable_x64", True)

from gigalens_research.simtests.registry import (
    register_generator,
    register_inference_builder,
    register_pipeline_builder,
    register_metric,
)
from gigalens_research.simtests.aggregate import register_campaign_metric


# ---------------------------------------------------------------------------
# Campaign physics configuration
# ---------------------------------------------------------------------------

# Every knob the generator understands, with its default. Unknown keys in the
# campaign yaml are a hard error (a typo'd knob must never silently fall back
# to a default in a calibration campaign).
_GENERATOR_DEFAULTS: Dict[str, Any] = {
    "n_systems": 100,
    # truth draw
    "truth_prior": "inference",     # "inference" (strict SBC) | "simulation"
    "source_prior_scale": 0.25,     # arcsec; absolute source pos ~ Normal(0, this)
    "amp_median": 2.0,              # amp ~ LogNormal(ln median, sigma_log)
    "amp_sigma_log": 0.5,
    # cosmology (fixed truth<->inference except H0, which is drawn per system)
    "H0_min": 50.0,
    "H0_max": 100.0,
    "Om0": 0.3,
    "z_lens": 0.5,
    "z_source": 1.5,
    # image selection
    "multiplicity": "quad",         # "quad" | "triple" | "double" | "any2plus"
    "max_redraws": 2000,
    "min_image_sep": 0.05,          # arcsec, on noiseless true images
    # observational noise (also the likelihood's assumed noise, verbatim)
    "sigma_ast": 0.004,             # arcsec per image coordinate
    "frac_flux": 0.05,              # sigma_F = frac_flux * F_true (per image)
    "sigma_td": 1.0,                # days, per non-reference image
    # lenstronomy truth solver
    "lt_min_distance": 0.01,
    "lt_search_window": 6.0,
    "lt_precision_limit": 1e-10,
    # gigalens EPL quadrature order (used for the offset-mode truth here and as
    # the builder's default fit profile, so the two stay consistent)
    "epl_niter": 18,
}

_TRUTH_PRIORS = ("inference", "simulation")
_MULTIPLICITIES = ("quad", "triple", "double", "any2plus")


def _resolve_generator_cfg(extra: Dict[str, Any]) -> Dict[str, Any]:
    """Merge DatasetSpec.extra over the defaults; reject unknown keys."""
    unknown = set(extra) - set(_GENERATOR_DEFAULTS)
    if unknown:
        raise ValueError(
            f"[lenstronomy_point_source] unknown dataset keys {sorted(unknown)}. "
            f"Known keys: {sorted(_GENERATOR_DEFAULTS)}. A typo'd physics knob "
            f"must not silently fall back to a default in a calibration campaign."
        )
    cfg = dict(_GENERATOR_DEFAULTS)
    cfg.update(extra)
    # normalize types (yaml can deliver ints where floats are meant, etc.)
    for k, default in _GENERATOR_DEFAULTS.items():
        if isinstance(default, bool):
            cfg[k] = bool(cfg[k])
        elif isinstance(default, int):
            cfg[k] = int(cfg[k])
        elif isinstance(default, float):
            cfg[k] = float(cfg[k])
        else:
            cfg[k] = str(cfg[k])
    if cfg["truth_prior"] not in _TRUTH_PRIORS:
        raise ValueError(f"truth_prior must be one of {_TRUTH_PRIORS}; "
                         f"got {cfg['truth_prior']!r}")
    if cfg["multiplicity"] not in _MULTIPLICITIES:
        raise ValueError(f"multiplicity must be one of {_MULTIPLICITIES}; "
                         f"got {cfg['multiplicity']!r}")
    if not (cfg["H0_min"] < cfg["H0_max"]):
        raise ValueError("H0_min must be < H0_max")
    return cfg


def _truth_dists(cfg: Dict[str, Any]):
    """The truth-draw distributions, keyed by role.

    This is the SINGLE definition of the campaign's priors: the generator
    samples from these and the builder (``priors: from_dataset``) fits with the
    very same objects, so simulation prior == inference prior by construction —
    the load-bearing requirement for SBC rank uniformity.

    ``"inference"`` is the GL2 inference-prior mass block (paper Eq. 8, Prior
    column, as in :func:`~.gl2_sersic.gl2_inference_prior`); ``"simulation"``
    is the tighter GL2 simulation block (robustness mode — score such runs with
    z-scores/coverage, not ranks).
    """
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions

    if cfg["truth_prior"] == "inference":
        epl = dict(
            theta_E=tfd.LogNormal(jnp.log(1.25), 0.4),
            gamma=tfd.TruncatedNormal(2.0, 0.5, 1.0, 3.0),
            e1=tfd.Normal(0.0, 0.2),
            e2=tfd.Normal(0.0, 0.2),
            center_x=tfd.Normal(0.0, 0.06),
            center_y=tfd.Normal(0.0, 0.06),
        )
        shear = dict(gamma1=tfd.Normal(0.0, 0.1), gamma2=tfd.Normal(0.0, 0.1))
    else:  # "simulation" (GL2 Eq. 8 Simulation column)
        epl = dict(
            theta_E=tfd.LogNormal(jnp.log(1.25), 0.25),
            gamma=tfd.TruncatedNormal(2.0, 0.25, 1.0, 3.0),
            e1=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
            e2=tfd.TruncatedNormal(0.0, 0.2, -0.5, 0.5),
            center_x=tfd.Normal(0.0, 0.03),
            center_y=tfd.Normal(0.0, 0.03),
        )
        shear = dict(gamma1=tfd.Normal(0.0, 0.05), gamma2=tfd.Normal(0.0, 0.05))

    s = cfg["source_prior_scale"]
    src = dict(center_x=tfd.Normal(0.0, s), center_y=tfd.Normal(0.0, s))
    amp = tfd.LogNormal(jnp.log(cfg["amp_median"]), cfg["amp_sigma_log"])
    H0 = tfd.Uniform(cfg["H0_min"], cfg["H0_max"])
    return {"epl": epl, "shear": shear, "src": src, "amp": amp, "H0": H0}


_EPL_KEYS = ("theta_E", "gamma", "e1", "e2", "center_x", "center_y")
_SHEAR_KEYS = ("gamma1", "gamma2")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@register_generator("lenstronomy_point_source")
def generate_lenstronomy_point_source(spec: Any, dataset_dir: str, seed: int) -> None:
    """Simulate ``n_systems`` lensed point sources with lenstronomy.

    Per system: rejection-sample a truth (mass + source position + amp + H0)
    until the lenstronomy image count matches ``multiplicity`` and the
    noiseless images are separated by at least ``min_image_sep`` — a pure
    function of the candidate truth, scanned in draw order, so the accepted
    truths are exact draws from the prior conditioned on acceptance. Then:

    - images sorted by arrival time (image 0 = first arriving; the time-delay
      reference, ``td_obs[0] == 0`` by the PointSourceObsData convention);
    - positions/fluxes/delays noised with sigmas computed from TRUE quantities
      (so the stored sigmas describe the injected noise law exactly);
    - offset-mode dx/dy truths computed from the NOISY observed positions
      delensed under the truth mass with the gigalens EPL (matching the
      likelihood's centroid definition);
    - everything persisted per system: ``truth_assets["ps_obs"]`` (observables
      + sigmas + per-image truth diagnostics, JSON) and
      ``truth_assets["ps_config"]`` (the resolved physics config, JSON), plus a
      scene-nested ``truth_x`` (``{"planes": ..., "cosmo": {"H0": ...}}``).

    The ``observed_image`` slot stores the stacked observed positions (n, 2) —
    the actual observation of this modality — which also gives the dataset
    manifest hash real content; the mandatory PSF slot gets a 1x1 placeholder
    (nothing on the point-source path reads it).
    """
    import jax.numpy as jnp
    from jax import random
    import tensorflow_probability.substrates.jax as tfp
    from astropy.cosmology import FlatLambdaCDM
    from lenstronomy.LensModel.lens_model import LensModel as LtLensModel
    from lenstronomy.LensModel.Solver.lens_equation_solver import LensEquationSolver

    from gigalens.jax.point_source import _total_deflection
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.mass.shear import Shear

    from gigalens_research.simtests.system import System, write_manifest
    from gigalens_research.simtests.generate import _hash_dataset

    tfd = tfp.distributions
    cfg = _resolve_generator_cfg(dict(spec.extra))
    dists = _truth_dists(cfg)

    # One joint over every truth parameter: a single vectorized sample of
    # max_redraws candidates per system, scanned in order (rejection sampling).
    joint = tfd.JointDistributionNamed({
        **{f"epl_{k}": v for k, v in dists["epl"].items()},
        **{f"shear_{k}": v for k, v in dists["shear"].items()},
        "src_x": dists["src"]["center_x"],
        "src_y": dists["src"]["center_y"],
        "amp": dists["amp"],
        "H0": dists["H0"],
    })

    n_systems = cfg["n_systems"]
    n_digits = len(str(n_systems - 1))
    system_ids: List[str] = []
    key = random.PRNGKey(seed)
    epl_prof, shear_prof = EPL(niter=cfg["epl_niter"]), Shear()
    reject_counts = {"multiplicity": 0, "min_sep": 0, "solver_error": 0}
    attempts_total = 0

    for i in range(n_systems):
        sys_key = random.fold_in(key, i)
        cand = joint.sample(cfg["max_redraws"], seed=sys_key)
        cand = {k: np.asarray(v, dtype=np.float64) for k, v in cand.items()}

        accepted = None
        for a in range(cfg["max_redraws"]):
            epl_t = {k: float(cand[f"epl_{k}"][a]) for k in _EPL_KEYS}
            shr_t = {k: float(cand[f"shear_{k}"][a]) for k in _SHEAR_KEYS}
            sx, sy = float(cand["src_x"][a]), float(cand["src_y"][a])
            amp_t, H0_t = float(cand["amp"][a]), float(cand["H0"][a])

            cosmo = FlatLambdaCDM(H0=H0_t, Om0=cfg["Om0"], Ob0=0.0)
            lm = LtLensModel(["EPL", "SHEAR"], z_lens=cfg["z_lens"],
                             z_source=cfg["z_source"], cosmo=cosmo)
            kw = [dict(epl_t), dict(shr_t, ra_0=0.0, dec_0=0.0)]
            try:
                cx, cy = LensEquationSolver(lm).image_position_from_source(
                    sx, sy, kw,
                    min_distance=cfg["lt_min_distance"],
                    search_window=cfg["lt_search_window"],
                    precision_limit=cfg["lt_precision_limit"],
                )
            except Exception:
                # e.g. |e| >= 1 tail draws where EPL is undefined; measure ~0
                # under both priors, so counting it as a rejection is harmless.
                reject_counts["solver_error"] += 1
                continue
            x_t, y_t = np.asarray(cx, float).reshape(-1), np.asarray(cy, float).reshape(-1)
            n_img = x_t.size
            ok_mult = {"quad": n_img == 4,
                       # 3-image = naked-cusp region (high |e|+shear tails);
                       # selection is on the SOLVER-FOUND count, like the others
                       "triple": n_img == 3,
                       "double": n_img == 2,
                       "any2plus": n_img >= 2}[cfg["multiplicity"]]
            if not ok_mult:
                reject_counts["multiplicity"] += 1
                continue
            if n_img >= 2:
                dx = x_t[:, None] - x_t[None, :]
                dy = y_t[:, None] - y_t[None, :]
                sep = np.sqrt(dx ** 2 + dy ** 2)
                min_sep = float(sep[np.triu_indices(n_img, k=1)].min())
                if min_sep < cfg["min_image_sep"]:
                    reject_counts["min_sep"] += 1
                    continue
            accepted = (epl_t, shr_t, sx, sy, amp_t, H0_t, lm, kw, x_t, y_t, a)
            break

        if accepted is None:
            raise RuntimeError(
                f"[lenstronomy_point_source] system {i}: no acceptable truth in "
                f"{cfg['max_redraws']} draws (multiplicity={cfg['multiplicity']}). "
                f"Rejections so far: {reject_counts}. Widen max_redraws or relax "
                f"the selection."
            )
        epl_t, shr_t, sx, sy, amp_t, H0_t, lm, kw, x_t, y_t, n_attempts = accepted
        attempts_total += n_attempts + 1
        n_img = x_t.size

        # Truth observables, images sorted by arrival time (image 0 = reference).
        mu_t = np.asarray(lm.magnification(x_t, y_t, kw), float).reshape(-1)
        t_days = np.asarray(lm.arrival_time(x_t, y_t, kw), float).reshape(-1)
        order = np.argsort(t_days)
        x_t, y_t, mu_t, t_days = x_t[order], y_t[order], mu_t[order], t_days[order]
        td_t = t_days - t_days[0]
        flux_t = amp_t * np.abs(mu_t)
        sigma_flux = cfg["frac_flux"] * flux_t

        # Noise: a numpy stream independent of the truth-draw PRNG, so changing
        # the selection never perturbs the noise of an accepted system.
        rng = np.random.default_rng([seed, 20260723, i])
        x_obs = x_t + rng.normal(0.0, cfg["sigma_ast"], n_img)
        y_obs = y_t + rng.normal(0.0, cfg["sigma_ast"], n_img)
        flux_obs = flux_t + rng.normal(0.0, 1.0, n_img) * sigma_flux
        if np.any(flux_obs <= 0):
            # ~20 sigma at frac_flux=0.05; a clip would silently break the
            # noise model, so fail loudly instead.
            raise RuntimeError(
                f"[lenstronomy_point_source] system {i}: non-positive observed "
                f"flux (frac_flux={cfg['frac_flux']}) — the Gaussian flux noise "
                f"model is inconsistent at this noise level."
            )
        td_obs = td_t.copy()
        td_obs[1:] += rng.normal(0.0, cfg["sigma_td"], n_img - 1)

        # Offset-mode truth: the likelihood's centroid is the mean of the NOISY
        # observed positions delensed under the (truth) mass, with the gigalens
        # EPL quadrature the fit uses.
        mp_t = {0: {k: jnp.asarray(v) for k, v in epl_t.items()},
                1: {k: jnp.asarray(v) for k, v in shr_t.items()}}
        axd, ayd = _total_deflection([epl_prof, shear_prof], mp_t,
                                     jnp.asarray(x_obs), jnp.asarray(y_obs))
        dx_t = sx - float(jnp.mean(jnp.asarray(x_obs) - axd))
        dy_t = sy - float(jnp.mean(jnp.asarray(y_obs) - ayd))

        truth_x = {
            "planes": {
                0: {"mass": {0: dict(epl_t), 1: dict(shr_t)}},
                1: {"light": {0: {"center_x": sx, "center_y": sy,
                                  "dx": dx_t, "dy": dy_t, "amp": amp_t}}},
            },
            "cosmo": {"H0": H0_t},
        }
        obs = {
            "x_obs": x_obs.tolist(), "y_obs": y_obs.tolist(),
            "sigma_ast": cfg["sigma_ast"],
            "flux_obs": flux_obs.tolist(), "sigma_flux": sigma_flux.tolist(),
            "td_obs": td_obs.tolist(), "sigma_td": cfg["sigma_td"],
            # truth-side diagnostics (never fed to the likelihood)
            "x_true": x_t.tolist(), "y_true": y_t.tolist(),
            "mu_true": mu_t.tolist(), "td_true": td_t.tolist(),
            "flux_true": flux_t.tolist(),
            "n_images": int(n_img), "n_redraws": int(n_attempts),
        }

        system_id = f"sys_{i:0{n_digits}d}"
        system_ids.append(system_id)
        System(
            system_id=system_id,
            # The stacked observed positions ARE the observation of this
            # modality; storing them here also gives the dataset hash content.
            # (float32 slot — the full-precision observables live in ps_obs.)
            observed_image=np.stack([x_obs, y_obs], axis=1).astype(np.float32),
            truth_x=truth_x,
            delta_pix=0.0,          # no imaging grid
            num_pix=int(n_img),
            supersample=1,
            psf=np.ones((1, 1), dtype=np.float32),  # mandatory slot; unused
            noise_kind="pointsource",
            background_rms=0.0,
            exp_time=0.0,
            truth_assets={"ps_obs": json.dumps(obs),
                          "ps_config": json.dumps(cfg)},
        ).save(dataset_dir)

        if (i + 1) % 10 == 0 or i == n_systems - 1:
            print(f"[lenstronomy_point_source] simulated {i + 1}/{n_systems} "
                  f"(attempts so far: {attempts_total}, rejects: {reject_counts})")

    write_manifest(
        dataset_dir,
        generator="lenstronomy_point_source",
        seed=seed,
        system_ids=system_ids,
        dataset_hash=_hash_dataset(dataset_dir, system_ids),
        extra={
            "config": cfg,
            "selection": {
                "attempts_total": attempts_total,
                "acceptance_rate": n_systems / max(attempts_total, 1),
                "rejects": reject_counts,
            },
        },
    )


def _load_ps_system(system: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Parse the persisted observables + physics config off a System."""
    assets = getattr(system, "truth_assets", {}) or {}
    if "ps_obs" not in assets or "ps_config" not in assets:
        raise ValueError(
            f"system {system.system_id!r} carries no point-source observables "
            f"(truth_assets keys: {sorted(assets)}). This builder only works on "
            f"datasets written by the 'lenstronomy_point_source' generator."
        )
    return json.loads(assets["ps_obs"]), json.loads(assets["ps_config"])


# ---------------------------------------------------------------------------
# Inference builder
# ---------------------------------------------------------------------------


@register_inference_builder("epl_shear_point_source_obs")
def build_epl_shear_point_source_obs(system: Any, **kwargs) -> Any:
    """EPL+Shear point-source fit over the persisted observables.

    Recognised kwargs (via ``inference.pipeline_kwargs`` / sweep points):

    ``fit_flux`` (True), ``fit_td`` (True) — likelihood channels. ``fit_td``
    also decides whether cosmology exists in the model (delays are the only
    H0-sensitive observable, so ``fit_td: false`` builds a cosmology-free
    scene). Both off falls back to the position-only dataset.
    ``priors`` ("from_dataset") — the only supported value: priors are rebuilt
    from the persisted truth-prior spec, guaranteeing the sim/inference match
    SBC requires. Anything else raises.
    ``src_parameterization`` ("absolute" | "offset"),
    ``offset_prior_scale`` (0.05, offset mode only),
    ``newton_steps`` (10), ``trust_region_frac`` (0.4),
    ``epl_niter`` (from the dataset config),
    ``src_anchor_sigma`` (None = off; arcsec) — the source-plane anchor of
    gigalens ``PointSourcePositionData``: tilts the saturated honesty charge's
    shelf so MAP/chains cannot stall on unconverged-solve plateaus (the sys_76
    failure, docs/logs/point-source-sbc.md P-4). Natural scale: ``sigma_ast``.
    ``multiplicity_constraint`` (None = off) — opt-in smoothed-image-count
    penalty of gigalens ``PointSourcePositionData``: a dict
    ``{n_obs, mu_min, eps, lam, grid_n, n_tiles, window|window_scale}``
    (``n_obs`` defaults to the system's observed image count) adding
    ``-lam (N_eff - n_obs)^2`` to log_like so phantom-image posterior modes
    are suppressed IN the likelihood rather than by post-hoc draw filtering
    (docs/logs/point-source-sbc.md P-8/P-9). The batched pipeline anneals its
    eps during a dedicated MAP refinement phase and keeps it fixed through
    SVI scoring and MCLMC (batched_pipeline.batched_map_anneal).
    """
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp

    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_prob_model import ProbModel
    from gigalens.jax.cosmo import wCDM_Cosmo
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.mass.shear import Shear
    from gigalens.jax.profiles.light.point_source import PointSourcePosition
    from gigalens.jax.point_source_position import (
        PointSourceObsData, PointSourcePositionData,
    )

    tfd = tfp.distributions
    obs, cfg = _load_ps_system(system)

    fit_flux = bool(kwargs.get("fit_flux", True))
    fit_td = bool(kwargs.get("fit_td", True))
    priors_mode = str(kwargs.get("priors", "from_dataset"))
    if priors_mode != "from_dataset":
        raise ValueError(
            f"priors={priors_mode!r} is not supported: this campaign's priors "
            f"come from the persisted dataset spec ('from_dataset') so the "
            f"sim/inference prior match SBC requires cannot silently drift. "
            f"Deliberate-mismatch experiments need their own builder."
        )
    src_par = str(kwargs.get("src_parameterization", "absolute"))
    if src_par not in ("absolute", "offset"):
        raise ValueError(f"src_parameterization must be 'absolute' or 'offset'; "
                         f"got {src_par!r}")
    newton_steps = int(kwargs.get("newton_steps", 10))
    trust_region_frac = float(kwargs.get("trust_region_frac", 0.4))
    offset_prior_scale = float(kwargs.get("offset_prior_scale", 0.05))
    epl_niter = int(kwargs.get("epl_niter", cfg["epl_niter"]))
    src_anchor_sigma = kwargs.get("src_anchor_sigma", None)
    if src_anchor_sigma is not None:
        src_anchor_sigma = float(src_anchor_sigma)
    multiplicity_constraint = kwargs.get("multiplicity_constraint", None)
    if multiplicity_constraint is not None:
        multiplicity_constraint = dict(multiplicity_constraint)

    # Metric knobs: metrics only receive (posterior, system), but the builder
    # runs first in the same process (run._run_one), so the yaml's sbc knobs are
    # threaded to the rank/health metrics via these module-level targets rather
    # than being silently ignored.
    global _SBC_RANK_DRAWS, _LOGLIK_RANK_DRAWS, _SOLVER_HEALTH_DRAWS
    _SBC_RANK_DRAWS = int(kwargs.get("sbc_rank_draws", _SBC_RANK_DRAWS))
    _LOGLIK_RANK_DRAWS = int(kwargs.get("loglik_rank_draws", _LOGLIK_RANK_DRAWS))
    _SOLVER_HEALTH_DRAWS = int(kwargs.get("solver_health_draws", _SOLVER_HEALTH_DRAWS))

    dists = _truth_dists(cfg)

    # Point source: sampled position (+ amp iff the flux channel is on).
    if src_par == "absolute":
        ps_prior = dict(center_x=dists["src"]["center_x"],
                        center_y=dists["src"]["center_y"])
    else:
        # Offset mode's implied absolute-position prior is data-dependent, so
        # it cannot exactly match the generator draw — approximate SBC for
        # dx/dy (see the campaign yaml header, note 2).
        ps_prior = dict(dx=tfd.Normal(0.0, offset_prior_scale),
                        dy=tfd.Normal(0.0, offset_prior_scale))
    if fit_flux:
        ps_prior["amp"] = dists["amp"]
    ps_comp = Component(
        PointSourcePosition(absolute=(src_par == "absolute"), with_amp=fit_flux),
        ps_prior,
    )
    epl_comp = Component(EPL(niter=epl_niter), dict(dists["epl"]))
    shear_comp = Component(Shear(), dict(dists["shear"]))

    if fit_td:
        cosmo_comp = Component(
            wCDM_Cosmo(z_lens=cfg["z_lens"], z_source_ref=cfg["z_source"]),
            dict(H0=dists["H0"], Om0=cfg["Om0"], k=0.0, w0=-1.0),
        )
        model = LensModel(
            [Plane(redshift=cfg["z_lens"], mass=[epl_comp, shear_comp]),
             Plane(redshift=cfg["z_source"], light=[ps_comp])],
            cosmo=cosmo_comp,
        )
    else:
        # No delays -> no H0 sensitivity -> cosmology-free scene (a lone lensed
        # plane defaults to deflection_ratio 1).
        model = LensModel(
            [Plane(mass=[epl_comp, shear_comp]),
             Plane(deflection_ratio=1.0, light=[ps_comp])],
        )

    x = np.asarray(obs["x_obs"], float)
    y = np.asarray(obs["y_obs"], float)
    sigma_ast = float(obs["sigma_ast"])
    common = dict(newton_steps=newton_steps, trust_region_frac=trust_region_frac,
                  src_anchor_sigma=src_anchor_sigma,
                  multiplicity_constraint=multiplicity_constraint)
    if fit_flux or fit_td:
        data = PointSourceObsData(
            ps_comp, x, y, sigma_ast,
            flux_obs=np.asarray(obs["flux_obs"], float) if fit_flux else None,
            sigma_flux=np.asarray(obs["sigma_flux"], float) if fit_flux else None,
            td_obs=np.asarray(obs["td_obs"], float) if fit_td else None,
            sigma_td=float(obs["sigma_td"]) if fit_td else None,
            **common,
        )
    else:
        data = PointSourcePositionData(ps_comp, x, y, sigma_ast, **common)

    return ProbModel(model, data)


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------


@register_pipeline_builder("map_svi_mclmc")
def build_map_svi_mclmc(system: Any, **kwargs) -> List[Any]:
    """MAP -> SVI -> MCLMC. Kwargs consumed (defaults):

    ``map_num_steps`` (1500), ``map_n_samples`` (1000), ``map_lr`` (3e-3),
    ``map_clip_norm`` (1.0) — MAP runs Adam behind zero_nans + global-norm
    clipping (the EPL |e|>=1 prior tail is a NaN region; zero_nans keeps those
    particles alive during wide exploration).
    ``svi_num_steps`` (1500), ``svi_n_vi`` (500), ``svi_lr`` (1e-4),
    ``svi_init_scale`` (1e-3).
    ``n_chains`` (8), ``num_burnin_steps`` (5000), ``num_results`` (10000),
    ``desired_energy_variance`` (5e-4), ``frac_tune1/2/3`` (0.2/0.6/0.2),
    ``mclmc_debug`` (False).
    """
    import optax
    from gigalens_research.inference_utils.pipeline import (
        MAPStage, SVIStage, MCLMCStage,
    )

    map_lr = float(kwargs.get("map_lr", 3e-3))
    map_clip = float(kwargs.get("map_clip_norm", 1.0))
    svi_lr = float(kwargs.get("svi_lr", 1e-4))

    def _map_optimizer():
        return optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(map_clip),
            optax.adam(map_lr),
        )

    def _svi_optimizer():
        return optax.adabelief(svi_lr, b1=0.95, b2=0.99)

    return [
        MAPStage(
            num_steps=int(kwargs.get("map_num_steps", 1500)),
            n_samples=int(kwargs.get("map_n_samples", 1000)),
            optimizer_factory=_map_optimizer,
            # The factory is a closure the cache can't hash; the id must encode
            # everything that changes the optimizer's behavior.
            optimizer_id=f"zero_nans_clip{map_clip}_adam{map_lr}",
        ),
        SVIStage(
            num_steps=int(kwargs.get("svi_num_steps", 1500)),
            n_vi=int(kwargs.get("svi_n_vi", 500)),
            init_scales=float(kwargs.get("svi_init_scale", 1e-3)),
            optimizer_factory=_svi_optimizer,
            optimizer_id=f"adabelief_{svi_lr}_b1_0.95_b2_0.99",
        ),
        MCLMCStage(
            n_chains=int(kwargs.get("n_chains", 8)),
            num_burnin_steps=int(kwargs.get("num_burnin_steps", 5000)),
            num_results=int(kwargs.get("num_results", 10000)),
            desired_energy_variance=float(kwargs.get("desired_energy_variance", 5e-4)),
            frac_tune1=float(kwargs.get("frac_tune1", 0.2)),
            frac_tune2=float(kwargs.get("frac_tune2", 0.6)),
            frac_tune3=float(kwargs.get("frac_tune3", 0.2)),
            debug=bool(kwargs.get("mclmc_debug", False)),
        ),
    ]


# ---------------------------------------------------------------------------
# Per-run metrics
# ---------------------------------------------------------------------------

# Thinning targets. Metrics receive only (posterior, system), so these are
# module constants rather than yaml knobs; the yaml's sbc_* keys configure the
# GATING in the campaign metric, not the per-run rank computation.
_SBC_RANK_DRAWS = 1024
_LOGLIK_RANK_DRAWS = 512
_SOLVER_HEALTH_DRAWS = 2000
_SOLVER_CONVERGED_ARCSEC = 1e-4


def _thin_samples(samples_z: np.ndarray, target: int) -> np.ndarray:
    """Evenly thin (C, N, P) chains to ~``target`` pooled draws.

    Even in-chain strides approximate independence far better than a random
    subsample of the pooled array (which preserves autocorrelation clumps);
    residual autocorrelation inflates rank noise at the extremes, which the
    uniformity test would misread as miscalibration.
    """
    C, N, P = samples_z.shape
    per_chain = min(N, max(1, int(math.ceil(target / C))))
    idx = np.unique(np.linspace(0, N - 1, num=per_chain).round().astype(int))
    thin = samples_z[:, idx, :].reshape(-1, P)
    finite = np.isfinite(thin).all(axis=1)
    return thin[finite]


@register_metric("sbc_ranks")
def sbc_ranks(posterior: Any, system: Any) -> Dict[str, Any]:
    """Per-parameter SBC rank: #{thinned posterior draws < truth}, in {0..L}.

    Returns ``{scene_path: rank, "_L": L}``. Uniform over {0..L} across systems
    iff the pipeline is calibrated AND truths were drawn from the fitted prior.
    """
    if not hasattr(posterior, "samples_z"):
        return {}
    try:
        from gigalens_research.param_index import (
            param_sites, sites_to_matrix, truth_row,
        )
        from gigalens_research.simtests.metrics import _scene_truth

        thin = _thin_samples(np.asarray(posterior.samples_z), _SBC_RANK_DRAWS)
        if thin.shape[0] == 0:
            return {}
        sites = param_sites(posterior)
        mat = sites_to_matrix(sites, posterior.z_to_x(thin))
        truth = truth_row(sites, _scene_truth(posterior, system),
                          what="sbc_ranks truth", warn=False)
        out: Dict[str, Any] = {}
        for i, site in enumerate(sites):
            if np.isfinite(truth[i]):
                out[site.key] = int(np.sum(mat[:, i] < truth[i]))
        out["_L"] = int(thin.shape[0])
        return out
    except Exception as exc:
        warnings.warn(f"[metrics.sbc_ranks] failed: {exc}", stacklevel=2)
        return {}


def _truth_unique(scene: Any, truth: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The truth as a flat unique dict keyed by ``scene.z_param_names``.

    Returns None if the truth is silent on any sampled parameter (the log-lik
    rank is then undefined for this run).
    """
    from gigalens_research.param_index import _nested_get
    import jax.numpy as jnp

    unique: Dict[str, Any] = {}
    for name in scene.z_param_names:
        path = tuple(int(seg) if seg.isdigit() else seg for seg in name.split("/"))
        v = _nested_get(truth, path)
        if v is None:
            return None
        unique[name] = jnp.asarray(float(np.squeeze(np.asarray(v))))
    return unique


@register_metric("loglik_rank")
def loglik_rank(posterior: Any, system: Any) -> Dict[str, Any]:
    """Rank of the truth's log-likelihood among thinned posterior draws'.

    A joint-calibration check: marginal ranks can all pass while the joint
    posterior is rotated or jointly mis-scaled; the log-likelihood is a scalar
    functional of the full parameter vector and catches that. Also uniform on
    {0..L} under correct calibration (SBC holds for any f(theta, y)).
    """
    if not hasattr(posterior, "samples_z"):
        return {}
    try:
        import jax.numpy as jnp
        from gigalens_research.simtests.metrics import _scene_truth

        prob = posterior.ctx.prob_model
        scene = posterior.scene
        unique = _truth_unique(scene, _scene_truth(posterior, system))
        if unique is None:
            return {}
        z_truth = np.asarray(scene.bijector.inverse(unique)).reshape(-1)

        thin = _thin_samples(np.asarray(posterior.samples_z), _LOGLIK_RANK_DRAWS)
        if thin.shape[0] == 0:
            return {}
        lls: List[np.ndarray] = []
        for start in range(0, thin.shape[0], 512):
            ll, _ = prob.log_like(jnp.asarray(thin[start:start + 512]))
            lls.append(np.asarray(ll).reshape(-1))
        ll_draws = np.concatenate(lls)
        ll_truth = float(np.asarray(
            prob.log_like(jnp.asarray(z_truth)[None, :])[0]).reshape(-1)[0])
        finite = np.isfinite(ll_draws)
        return {"rank": int(np.sum(ll_draws[finite] < ll_truth)),
                "_L": int(finite.sum()),
                "ll_truth": float(ll_truth)}
    except Exception as exc:
        warnings.warn(f"[metrics.loglik_rank] failed: {exc}", stacklevel=2)
        return {}


@register_metric("ps_solver_health")
def ps_solver_health(posterior: Any, system: Any) -> Dict[str, float]:
    """Lens-equation solver health over posterior draws.

    Chains can mix beautifully (R-hat ~1) while the solve silently fails on a
    fraction of draws, so this is an independent gate: the per-draw reduced
    chi2 of the point-source term and the fraction of draws whose solved
    positions do not delens back to the source (src residual > 1e-4").
    """
    prob = getattr(posterior.ctx, "prob_model", None)
    terms = getattr(prob, "terms", None) or []
    if not hasattr(posterior, "samples_z") or not terms or not hasattr(terms[0], "solve"):
        return {}
    try:
        import jax
        import jax.numpy as jnp
        from gigalens.jax.point_source_position import _delens_and_jacobian

        term = terms[0]
        model = prob.model
        event_size = float(term.event_size)

        @jax.jit
        def diag(z):
            params = model.to_params(prob.bij.forward(z))
            tx, ty, bsx, bsy = term.solve(params)
            mp = params["planes"][term.lens_i]["mass"]
            (bx, by), _ = _delens_and_jacobian(term.mass_profiles, mp, tx, ty)
            src_res = jnp.max(jnp.hypot(bx - bsx, by - bsy), axis=0)
            _, chi2 = term.log_like(params)
            return src_res, chi2

        thin = _thin_samples(np.asarray(posterior.samples_z), _SOLVER_HEALTH_DRAWS)
        if thin.shape[0] == 0:
            return {}
        res_parts, chi2_parts = [], []
        for start in range(0, thin.shape[0], 1000):
            r, c = diag(jnp.asarray(thin[start:start + 1000]))
            res_parts.append(np.asarray(r).reshape(-1))
            chi2_parts.append(np.asarray(c).reshape(-1))
        src_res = np.concatenate(res_parts)
        red = np.concatenate(chi2_parts) / event_size
        return {
            "red_chi2_median": float(np.median(red)),
            "red_chi2_p90": float(np.percentile(red, 90)),
            "red_chi2_p99": float(np.percentile(red, 99)),
            "red_chi2_max": float(np.max(red)),
            "frac_red_gt10": float(np.mean(red > 10)),
            "frac_unconverged": float(np.mean(src_res > _SOLVER_CONVERGED_ARCSEC)),
        }
    except Exception as exc:
        warnings.warn(f"[metrics.ps_solver_health] failed: {exc}", stacklevel=2)
        return {}


# ---------------------------------------------------------------------------
# Campaign metric: SBC uniformity
# ---------------------------------------------------------------------------


@register_campaign_metric("sbc_uniformity")
def sbc_uniformity(index_df: Any, campaign_spec: Any, agg_dir: str) -> None:
    """Rank-uniformity analysis over converged runs, per sweep point.

    Gates (read from ``inference.pipeline_kwargs``): ``sbc_rhat_max`` (1.05),
    ``sbc_min_ess`` (100), ``sbc_solver_unconverged_max`` (0.05). Excluded runs
    are REPORTED, not hidden — attrition above ~5% taints the SBC set because
    failures rarely land uniformly in parameter space.

    Per parameter: the exact PIT ``u = (rank + eta) / (L + 1)`` (eta ~ U[0,1),
    fixed seed — exactly Uniform(0,1) under H0 for discrete ranks), a KS
    p-value with a Holm correction across parameters, and an ECDF-difference
    plot with a Monte-Carlo simultaneous 95% band. Outputs per sweep:
    ``sbc_ecdf_<sweep>.png``, ``sbc_hist_<sweep>.png``, and a combined
    ``sbc_report.json``.
    """
    import pandas as pd

    if index_df is None or not hasattr(index_df, "columns") or index_df.empty:
        return
    if "sbc_ranks_json" not in index_df.columns:
        print("[sbc_uniformity] no sbc_ranks_json column; is the 'sbc_ranks' "
              "metric in the campaign's metric list?")
        return

    pk = dict(campaign_spec.inference.pipeline_kwargs)
    rhat_max = float(pk.get("sbc_rhat_max", 1.05))
    ess_min = float(pk.get("sbc_min_ess", 100.0))
    unconv_max = float(pk.get("sbc_solver_unconverged_max", 0.05))

    report: Dict[str, Any] = {
        "gates": {"sbc_rhat_max": rhat_max, "sbc_min_ess": ess_min,
                  "sbc_solver_unconverged_max": unconv_max},
        "sweeps": {},
    }

    def _num(series):
        return pd.to_numeric(series, errors="coerce")

    for sweep_name, g in index_df.groupby("sweep_name"):
        ok = g[g["status"] == "ok"] if "status" in g.columns else g
        n_total = len(g)
        gate = pd.Series(True, index=ok.index)
        if "max_rhat" in ok.columns:
            gate &= _num(ok["max_rhat"]) <= rhat_max
        if "min_ess" in ok.columns:
            gate &= _num(ok["min_ess"]) >= ess_min
        unconv_col = "ps_solver_health.frac_unconverged"
        if unconv_col in ok.columns:
            gate &= _num(ok[unconv_col]).fillna(1.0) <= unconv_max
        passed = ok[gate.fillna(False)]
        excluded = sorted(set(g["system_id"]) - set(passed["system_id"]))

        # Collect ranks: {param: [(rank, L), ...]}
        ranks: Dict[str, List[Tuple[int, int]]] = {}
        for _, row in passed.iterrows():
            try:
                d = json.loads(row["sbc_ranks_json"])
            except Exception:
                continue
            L = int(float(d.get("_L", 0)))
            if L <= 0:
                continue
            for param, r in d.items():
                if param == "_L":
                    continue
                try:
                    ranks.setdefault(param, []).append((int(float(r)), L))
                except (TypeError, ValueError):
                    continue
        # loglik ranks ride along as one more "parameter"
        if "loglik_rank_json" in passed.columns:
            for _, row in passed.iterrows():
                try:
                    d = json.loads(row["loglik_rank_json"])
                    ranks.setdefault("loglik", []).append(
                        (int(float(d["rank"])), int(float(d["_L"]))))
                except Exception:
                    continue

        sweep_rep: Dict[str, Any] = {
            "n_runs": n_total,
            "n_status_ok": int(len(ok)),
            "n_passed_gates": int(len(passed)),
            "attrition_frac": float(1.0 - len(passed) / n_total) if n_total else None,
            "excluded_system_ids": excluded,
            "params": {},
        }
        if excluded and n_total:
            frac = 1.0 - len(passed) / n_total
            if frac > 0.05:
                sweep_rep["warning"] = (
                    f"{frac:.1%} of runs excluded by the convergence/solver "
                    f"gates; check whether the excluded truths cluster in "
                    f"parameter space before trusting the rank analysis."
                )

        pvals: Dict[str, float] = {}
        pit: Dict[str, np.ndarray] = {}
        for param, rl in sorted(ranks.items()):
            if len(rl) < 5:
                continue
            r = np.array([x[0] for x in rl], float)
            L = np.array([x[1] for x in rl], float)
            # Exact PIT for discrete ranks: (rank + U[0,1)) / (L + 1) is
            # exactly Uniform(0,1) under H0. Deterministic jitter stream.
            import hashlib
            jitter_seed = int(hashlib.sha256(
                f"{campaign_spec.name}|{sweep_name}|{param}".encode()
            ).hexdigest()[:8], 16)
            eta = np.random.default_rng(jitter_seed).uniform(size=r.size)
            u = (r + eta) / (L + 1.0)
            pit[param] = u
            try:
                from scipy import stats
                pvals[param] = float(stats.kstest(u, "uniform").pvalue)
            except ImportError:
                pvals[param] = float("nan")

        # Holm correction across parameters
        names = [p for p in pvals if np.isfinite(pvals[p])]
        m = len(names)
        holm_reject: Dict[str, bool] = {}
        for rank_i, p in enumerate(sorted(names, key=lambda q: pvals[q])):
            threshold = 0.05 / (m - rank_i)
            if pvals[p] > threshold:
                break
            holm_reject[p] = True
        for param in pvals:
            sweep_rep["params"][param] = {
                "n": int(pit[param].size),
                "ks_pvalue": pvals[param],
                "holm_rejected_at_0.05": bool(holm_reject.get(param, False)),
                "mean_pit": float(pit[param].mean()),
            }

        _plot_sbc(pit, pvals, holm_reject, sweep_name, campaign_spec, agg_dir)
        report["sweeps"][str(sweep_name)] = sweep_rep
        n_bad = sum(holm_reject.values())
        print(f"[sbc_uniformity] sweep {sweep_name!r}: "
              f"{len(passed)}/{n_total} runs pass gates "
              f"({len(excluded)} excluded); "
              f"{n_bad}/{m} parameters rejected at Holm-0.05"
              + (f" ({sorted(holm_reject)})" if holm_reject else ""))

    with open(os.path.join(agg_dir, "sbc_report.json"), "w") as f:
        json.dump(report, f, indent=2)


def _plot_sbc(pit: Dict[str, np.ndarray], pvals: Dict[str, float],
              holm_reject: Dict[str, bool], sweep_name: Any,
              campaign_spec: Any, agg_dir: str) -> None:
    """ECDF-difference panels with a simultaneous 95% MC band + rank histograms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not pit:
        return
    params = sorted(pit)
    n = len(params)
    ncols = min(4, n)
    nrows = int(math.ceil(n / ncols))
    tag = str(sweep_name).replace("/", "_")

    # Simultaneous band: 95th percentile of sup_t |ECDF_sim(t) - t| over MC
    # replicates of N exact uniforms (KS-style constant band; conservative
    # near the edges relative to the Saeilynoja pointwise-adjusted band, which
    # is fine for a first-line diagnostic).
    grid = np.linspace(0.0, 1.0, 201)
    n_runs = min(len(v) for v in pit.values())
    rng = np.random.default_rng(1)
    sups = np.empty(2000)
    for s in range(2000):
        sim = np.sort(rng.uniform(size=n_runs))
        ecdf = np.searchsorted(sim, grid, side="right") / n_runs
        sups[s] = np.abs(ecdf - grid).max()
    band = float(np.percentile(sups, 95))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows),
                             squeeze=False)
    fig.suptitle(f"{campaign_spec.name} [{sweep_name}] — SBC rank ECDF - uniform "
                 f"(95% simultaneous band, N={n_runs})", fontsize=10)
    for k, param in enumerate(params):
        ax = axes[k // ncols][k % ncols]
        u = np.sort(pit[param])
        ecdf = np.searchsorted(u, grid, side="right") / u.size
        bad = holm_reject.get(param, False)
        ax.fill_between(grid, -band, band, color="0.9")
        ax.axhline(0.0, color="0.5", lw=0.6)
        ax.plot(grid, ecdf - grid, color=("crimson" if bad else "steelblue"), lw=1.4)
        ax.set_title(f"{param}  (p={pvals.get(param, float('nan')):.3f})",
                     fontsize=7, color=("crimson" if bad else "black"))
        ax.set_ylim(-max(3 * band, 0.2), max(3 * band, 0.2))
        ax.tick_params(labelsize=6)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(agg_dir, f"sbc_ecdf_{tag}.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.2 * nrows),
                             squeeze=False)
    fig.suptitle(f"{campaign_spec.name} [{sweep_name}] — SBC rank histograms",
                 fontsize=10)
    n_bins = 20
    for k, param in enumerate(params):
        ax = axes[k // ncols][k % ncols]
        u = pit[param]
        ax.hist(u, bins=n_bins, range=(0, 1), edgecolor="white")
        expect = u.size / n_bins
        ax.axhline(expect, color="0.4", lw=0.8, ls="--")
        # ~99% pointwise binomial band for a quick visual reference
        sd = math.sqrt(expect * (1 - 1 / n_bins))
        ax.axhspan(max(expect - 2.58 * sd, 0), expect + 2.58 * sd,
                   color="0.92", zorder=0)
        ax.set_title(param, fontsize=7)
        ax.tick_params(labelsize=6)
    for k in range(n, nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(agg_dir, f"sbc_hist_{tag}.png"), dpi=150)
    plt.close(fig)
