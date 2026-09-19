"""Built-in pipeline builders and custom inference stages.

Registered pipeline builders
----------------------------
- ``map_svi_hmc``: standard MAP → SVI → HMC pipeline (used for GL2 Sérsic test).
- ``map_bootstrap_mclmc``: fixed-lens MAP bootstrap → MCLMC (used for Vela
  shapelets systematics test).
- ``map_mclmc``: truth-free multi-start MAP → diagonal qz → MCLMC (used for the
  vela_f140w_v3 Sersic-source fits).

Custom stages
-------------
- :class:`PartialTruthBootstrapQzStage`: profile-agnostic bootstrap.  Given a
  truth that constrains some parameters but leaves others free, it runs a short
  MAP with the constrained parameters pinned to truth (the rest optimised),
  then constructs a tight diagonal ``qz`` around the full true-parameter vector.
  This ``qz`` initialises the subsequent MCLMC chains at truth, isolating model
  misspecification as the only source of posterior bias.  Profiles and free-
  parameter priors are read from the ``InferenceContext`` rather than re-
  specified, so the same stage serves every source/lens model.
"""
from __future__ import annotations

import hashlib
import pickle
import time
from typing import Any, Dict, List, Optional

import numpy as np

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

from gigalens_research.inference_utils.pipeline import (
    BridgeStage,
    InferenceStage,
    MAPStage,
    MCLMCStage,
    StageResult,
    SVIStage,
    HMCStage,
    register_stage,
)

from .registry import register_pipeline_builder
from gigalens_research.inference_utils.params import to_dict_params


# ---------------------------------------------------------------------------
# Standard pipeline builders
# ---------------------------------------------------------------------------


@register_pipeline_builder("map_svi_hmc")
def build_map_svi_hmc(system: Any, **kwargs) -> List[InferenceStage]:
    """MAP → SVI → HMC pipeline.  Kwargs consumed:

    ``map_num_steps`` (1000), ``map_n_samples`` (2000),
    ``svi_num_steps`` (5000), ``svi_n_vi`` (1000),
    ``hmc_n_hmc`` (64), ``hmc_num_results`` (1500), ``hmc_num_burnin`` (500),
    ``hmc_init_eps`` (0.3), ``hmc_init_l`` (3),
    ``hmc_max_leapfrog_steps`` (30).
    """
    return [
        MAPStage(
            num_steps=int(kwargs.get("map_num_steps", 1000)),
            n_samples=int(kwargs.get("map_n_samples", 2000)),
        ),
        SVIStage(
            num_steps=int(kwargs.get("svi_num_steps", 5000)),
            n_vi=int(kwargs.get("svi_n_vi", 1000)),
        ),
        HMCStage(
            n_hmc=int(kwargs.get("hmc_n_hmc", 64)),
            num_results=int(kwargs.get("hmc_num_results", 1500)),
            num_burnin_steps=int(kwargs.get("hmc_num_burnin", 500)),
            init_eps=float(kwargs.get("hmc_init_eps", 0.3)),
            init_l=int(kwargs.get("hmc_init_l", 3)),
            max_leapfrog_steps=int(kwargs.get("hmc_max_leapfrog_steps", 30)),
        ),
    ]


@register_pipeline_builder("map_bootstrap_mclmc")
def build_map_bootstrap_mclmc(system: Any, **kwargs) -> List[InferenceStage]:
    """Fixed-lens MAP bootstrap → MCLMC pipeline.

    Runs :class:`PartialTruthBootstrapQzStage` with the lens (mass + lens light)
    pinned to truth and the source left free, recovering the source geometry via
    a short MAP, then runs MCLMC starting from a tight diagonal ``qz`` centred at
    the full truth in unconstrained space.  The source profile is whatever the
    inference model uses (read from ``ctx``), so this builder is source-agnostic.

    Kwargs consumed:

    ``bootstrap_map_steps`` (200), ``bootstrap_map_n_samples`` (100),
    ``bootstrap_diag_scale`` (1e-6), ``bootstrap_pin_eps`` (1e-6),
    ``n_chains`` (8), ``num_burnin_steps`` (4000), ``num_results`` (4000),
    ``desired_energy_variance`` (5e-4),
    ``frac_tune1`` (0.2), ``frac_tune2`` (0.6), ``frac_tune3`` (0.2),
    ``undersampling_check`` (False; ``True`` or a dict of
    :class:`UndersamplingCheckStage` kwargs inserts a quadrature certification at the
    bootstrapped start point that raises before sampling if it fails).

    ``bootstrap_diag_scale`` is the variance of the tight diagonal ``qz`` the
    chains are initialised from (``scale = sqrt(diag_scale)``); ``bootstrap_pin_eps``
    is the half-width of the ``Uniform`` used to pin the truth-constrained
    parameters during the bootstrap MAP.
    """
    stages: List[InferenceStage] = [
        PartialTruthBootstrapQzStage(
            system=system,
            free=("source",),
            map_num_steps=int(kwargs.get("bootstrap_map_steps", 200)),
            map_n_samples=int(kwargs.get("bootstrap_map_n_samples", 100)),
            diag_scale=float(kwargs.get("bootstrap_diag_scale", 1e-6)),
            pin_eps=float(kwargs.get("bootstrap_pin_eps", 1e-6)),
        ),
    ]
    uc = kwargs.get("undersampling_check")
    if uc:
        uc = dict(uc) if isinstance(uc, dict) else {}
        stages.append(UndersamplingCheckStage(**uc))
    stages.append(
        MCLMCStage(
            n_chains=int(kwargs.get("n_chains", 8)),
            num_burnin_steps=int(kwargs.get("num_burnin_steps", 4000)),
            num_results=int(kwargs.get("num_results", 4000)),
            desired_energy_variance=float(kwargs.get("desired_energy_variance", 5e-4)),
            frac_tune1=float(kwargs.get("frac_tune1", 0.2)),
            frac_tune2=float(kwargs.get("frac_tune2", 0.6)),
            frac_tune3=float(kwargs.get("frac_tune3", 0.2)),
            debug=bool(kwargs.get("mclmc_debug", False)),
        ))
    return stages


@register_pipeline_builder("map_mclmc")
def build_map_mclmc(system: Any, **kwargs) -> List[InferenceStage]:
    """Truth-free MAP → diagonal-qz bridge → MCLMC.

    Unlike ``map_bootstrap_mclmc`` nothing here reads ``system.truth_x``: the MAP is
    a multi-start optimisation from prior draws, the chains start from a tight
    diagonal Gaussian around the MAP optimum in unconstrained space (its scale is
    also the initial mass-matrix guess that MCLMC's tuning refines), and the
    sampler runs from there. This is the pipeline for fits that must not use the
    simulation truth (2026-09-17, vela_f140w_v3 first Sersic-source fit).

    Kwargs consumed:

    ``map_num_steps`` (500), ``map_n_samples`` (500), ``qz_diag_scale`` (1e-2;
    the std of the diagonal qz around ``z_best``), ``n_chains`` (8),
    ``num_burnin_steps`` (4000), ``num_results`` (4000),
    ``desired_energy_variance`` (5e-4), ``frac_tune1`` (0.2), ``frac_tune2`` (0.6),
    ``frac_tune3`` (0.2), ``mclmc_debug`` (False).
    """
    import jax.numpy as jnp

    qz_scale = float(kwargs.get("qz_diag_scale", 1e-2))

    def _diag_qz(z_best):
        z = jnp.asarray(z_best)
        return tfd.MultivariateNormalDiag(
            loc=z, scale_diag=jnp.full(z.shape[-1], qz_scale, dtype=z.dtype))

    return [
        MAPStage(
            num_steps=int(kwargs.get("map_num_steps", 500)),
            n_samples=int(kwargs.get("map_n_samples", 500)),
        ),
        BridgeStage(
            name="diag_qz_from_map",
            version=f"v1_scale{qz_scale:g}",
            requires=("z_best",),
            produces=("qz",),
            fn=_diag_qz,
        ),
        MCLMCStage(
            n_chains=int(kwargs.get("n_chains", 8)),
            num_burnin_steps=int(kwargs.get("num_burnin_steps", 4000)),
            num_results=int(kwargs.get("num_results", 4000)),
            desired_energy_variance=float(kwargs.get("desired_energy_variance", 5e-4)),
            frac_tune1=float(kwargs.get("frac_tune1", 0.2)),
            frac_tune2=float(kwargs.get("frac_tune2", 0.6)),
            frac_tune3=float(kwargs.get("frac_tune3", 0.2)),
            debug=bool(kwargs.get("mclmc_debug", False)),
        ),
    ]


# ---------------------------------------------------------------------------
# Custom stage: UndersamplingCheckStage (2026-09-17)
# ---------------------------------------------------------------------------


@register_stage
class UndersamplingCheckStage(InferenceStage):
    """Certify the inference quadrature at the sampler's starting point BEFORE
    sampling (user request 2026-09-17: "check the undersampling for the
    bootstrapped truth before running sampling").

    Runs :func:`gigalens.jax.analysis.diagnose_undersampling` on the inference
    ``ProbModel`` at ``qz.mean()`` — the point the chains start from (the truth
    with the bootstrapped source, or a MAP) — through the likelihood path (lstsq
    amplitudes re-solved per quadrature), against a converged uniform reference
    with its own half-supersample self-check. The configured quadrature (uniform
    or the adaptive grid) is the rung that is gated; the uniform ladder is
    reported for context.

    Gate (derived 2026-09-17 on vela22, `docs/logs/vela-f140w-modelling.md`):
    smooth-source renders on the factor-8 adaptive map sit within 0.05 sigma of
    a ss=32 reference on every pixel except the lens-light centre, whose cusp
    quadrature is non-monotone (0.5 sigma at factor 8, 0.15 at 16). So the stage
    FAILS (raises, no sampling) when the configured rung's worst pixel exceeds
    ``max_delta_sigma`` (default 1.0: a pixel-level bias no known cusp residue
    reaches) or when more than ``max_frac_above`` of the pixels exceed the
    per-pixel ``tolerance`` (default 1e-3 = 25 px of 25,600: the known residue
    is one pixel). ``reference_supersample`` (16) must be even; its self-check
    is reported and a non-converged reference is a failure too. The stage
    persists the configured rung's ``delta_over_sigma`` map and the report
    summary; ``qz`` passes through untouched.
    """

    name = "undersampling_check"
    schema_version: int = 1
    requires = ("qz",)
    produces = ("undersampling",)

    def __init__(self, *, reference_supersample: int = 16,
                 supersample_ladder=(1, 2, 4), tolerance: float = 0.1,
                 max_delta_sigma: float = 1.0, max_frac_above: float = 1e-3,
                 exclude_cusp_radius_pix: float = 4.0,
                 reference_self_check_frac: float = 0.5,
                 name: Optional[str] = None, seed: Optional[int] = None):
        super().__init__(name=name or "undersampling_check", seed=seed)
        self.reference_supersample = int(reference_supersample)
        self.supersample_ladder = tuple(int(s) for s in supersample_ladder)
        self.tolerance = float(tolerance)
        self.max_delta_sigma = float(max_delta_sigma)
        self.max_frac_above = float(max_frac_above)
        # Pixels within this radius of every LENS-plane light centre (at the start
        # point) are excluded from the certification statistics and from the reference
        # self-check: a Sersic cusp there converges non-monotonically under midpoint
        # quadrature (vela22 lens light, n = 5.2: 0.19 / 0.50 / 0.15 / 0.035 sigma at
        # ss 4 / 8 / 16 / 32 vs 64), and its wings keep the ss=8-vs-16 self-check above
        # 0.05 sigma out to ~4 px (measured 2026-09-17, lens light only: max outside
        # r <= 2 / 3 / 4 / 5 px = 0.108 / 0.068 / 0.042 / 0.015 sigma; the ss=16
        # reference's own error there, 16 vs 32: 0.035 / 0.021 / 0.013 / 0.005). The
        # default 4 px excludes 50 of 25,600 pixels. The excluded region's worst
        # residual is still measured and recorded (``configured_max_in_excluded``).
        # 0 disables.
        self.exclude_cusp_radius_pix = float(exclude_cusp_radius_pix)
        # Reference certification: the module flags a reference "converged" only if its
        # half-supersample self-check is within tolerance/4 (0.025 sigma), which the
        # ss=16 reference misses on these data by a hair (0.08 sigma at a smooth start
        # point) and ss=32 is unaffordable for a 231-basis shapelet stack (48 GB).
        # Under the midpoint rule's second-order convergence, err(16) ~= self_err / 3,
        # so self_err <= tolerance * frac (default 0.5 -> 0.05 sigma) bounds the
        # reference's own error to ~0.017 sigma, well inside the 0.1 gate. Both the
        # module's verdict and this one are recorded.
        self.reference_self_check_frac = float(reference_self_check_frac)

    def config_hash_data(self) -> Dict[str, Any]:
        return {"reference_supersample": self.reference_supersample,
                "supersample_ladder": list(self.supersample_ladder),
                "tolerance": self.tolerance, "max_delta_sigma": self.max_delta_sigma,
                "max_frac_above": self.max_frac_above,
                "exclude_cusp_radius_pix": self.exclude_cusp_radius_pix,
                "reference_self_check_frac": self.reference_self_check_frac}

    def _cusp_mask(self, ctx, z0, dataset):
        """Boolean (H, W) mask, True = certify this pixel; False inside the excluded
        disks around the lens-plane light centres at ``z0``. Returns (mask, centres)."""
        import jax.numpy as jnp
        model = ctx.prob_model.model
        ny, nx = tuple(int(v) for v in dataset.image.shape[-2:])
        mask = np.ones((ny, nx), dtype=bool)
        centres = []
        if self.exclude_cusp_radius_pix <= 0:
            return mask, centres
        lab = ctx.prob_model.labeled_samples(jnp.asarray(z0)[None])
        lab = {k: float(np.asarray(v).ravel()[0]) for k, v in lab.items()}
        src_ids = {id(c) for c in model.source_plane_light()}
        dp = float(dataset.sim_config.delta_pix)
        yy, xx = np.indices((ny, nx))
        for i, plane in enumerate(model.planes):
            for j, comp in enumerate(plane.light):
                if id(comp) in src_ids:
                    continue
                base = f"planes/{model.plane_key(i)}/light/{model.component_key(i, 'light', j)}/"
                cx, cy = lab.get(base + "center_x"), lab.get(base + "center_y")
                if cx is None or cy is None:
                    consts = getattr(model, "constants", {}) or {}
                    node = (consts.get("planes", {}).get(model.plane_key(i), {})
                            .get("light", {}).get(model.component_key(i, "light", j), {}))
                    cx, cy = node.get("center_x"), node.get("center_y")
                if cx is None or cy is None:
                    continue
                col = (nx - 1) / 2.0 + float(cx) / dp
                row = (ny - 1) / 2.0 + float(cy) / dp
                mask &= np.hypot(yy - row, xx - col) > self.exclude_cusp_radius_pix
                centres.append([float(row), float(col)])
        return mask, centres

    @staticmethod
    def _masked_copy(ds, mask):
        """Same observation with ``mask`` ANDed in (ImageData or AdaptiveImageData)."""
        import jax.numpy as jnp
        from gigalens.jax.scene_prob_model import ImageData
        common = dict(error_map=ds.error_map, mask=jnp.asarray(mask) & ds.mask,
                      sees=getattr(ds, "_sees_spec", None) or getattr(ds, "sees", "all"),
                      mode=getattr(ds, "mode", None))
        try:
            from gigalens.jax.experimental.adaptive_supersample import AdaptiveImageData
        except ImportError:
            AdaptiveImageData = ()
        if AdaptiveImageData and isinstance(ds, AdaptiveImageData):
            return AdaptiveImageData(ds.image, ds.sim_config, adaptive_grid=ds.adaptive_grid, **common)
        return ImageData(ds.image, ds.sim_config, **common)

    def run(self, ctx, artifacts, seed):
        import jax.numpy as jnp
        from gigalens.jax.analysis import diagnose_undersampling
        t0 = time.perf_counter()
        from gigalens.jax.scene_prob_model import ProbModel
        z0 = jnp.asarray(artifacts["qz"].mean())
        # Certify on masked copies of the datasets (cusp disks excluded, see __init__);
        # the model and amplitude mode are the inference ones.
        masks, centres_all, masked_ds = [], [], []
        for ds in ctx.prob_model.datasets:
            m, centres = self._cusp_mask(ctx, z0, ds)
            masks.append(m); centres_all.append(centres); masked_ds.append(self._masked_copy(ds, m))
        check_prob = ProbModel(ctx.prob_model.model, masked_ds,
                               mode=getattr(ctx.prob_model, "mode", "lstsq"))
        reports = diagnose_undersampling(
            check_prob, z0, supersample_ladder=self.supersample_ladder,
            reference_supersample=self.reference_supersample,
            tolerance=self.tolerance, check_reference=True)
        arrays: Dict[str, np.ndarray] = {"z0": np.asarray(z0)}
        meta: Dict[str, Any] = {"wall_time_s": None, "reports": [],
                                "exclude_cusp_radius_pix": self.exclude_cusp_radius_pix}
        failures = []
        for i, rep in enumerate(reports):
            summary = rep.summary()
            print(f"[undersampling_check] dataset {i} at the sampler start point "
                  f"({int((~masks[i]).sum())} px excluded around lens-light centres "
                  f"{centres_all[i]}, radius {self.exclude_cusp_radius_pix} px):\n{summary}")
            rung_meta = []
            for r in rep.rungs:
                rung_meta.append({
                    "label": r.label, "supersample": r.supersample,
                    "convention": r.convention,
                    "max_abs_delta_over_sigma": float(r.max_abs_delta_over_sigma),
                    "argmax": [int(v) for v in r.argmax],
                    "frac_above_tolerance": float(r.frac_above_tolerance),
                    "delta_chi2": float(np.asarray(r.delta_chi2).ravel()[0]),
                    "passes": bool(r.passes)})
            configured = [r for r in rep.rungs if r.supersample is None]
            if not configured:   # uniform quadrature: the configured supersample is a ladder rung
                ss = int(ctx.prob_model.datasets[i].sim_config.supersample)
                configured = [r for r in rep.rungs if r.supersample == ss]
            r = configured[0]
            dos = np.asarray(r.delta_over_sigma)[0]
            arrays[f"delta_over_sigma_{i}"] = dos
            arrays[f"certified_mask_{i}"] = masks[i]
            excl = ~masks[i]
            max_in_excluded = float(np.abs(dos[excl]).max()) if excl.any() else None
            self_err = getattr(rep, "reference_self_error", None)
            converged = getattr(rep, "reference_converged", True)
            meta["reports"].append({
                "summary": summary, "rungs": rung_meta,
                "configured_label": r.label,
                "configured_max_abs_delta_over_sigma": float(r.max_abs_delta_over_sigma),
                "configured_frac_above_tolerance": float(r.frac_above_tolerance),
                "configured_argmax_yx": [int(r.argmax[1]), int(r.argmax[2])],
                "configured_delta_chi2": float(np.asarray(r.delta_chi2).ravel()[0]),
                "excluded_centres_rowcol": centres_all[i],
                "excluded_pixels": int(excl.sum()),
                "configured_max_in_excluded": max_in_excluded,
                "reference_supersample": self.reference_supersample,
                "reference_self_error": (None if self_err is None else
                                         {k: float(v) for k, v in dict(self_err).items()}
                                         if isinstance(self_err, dict) else float(self_err)),
                })
            self_err_max = (max(float(v) for v in dict(self_err).values())
                            if isinstance(self_err, dict) and self_err else
                            (None if self_err is None else float(self_err)))
            ref_ok = (self_err_max is None) or (self_err_max <= self.tolerance * self.reference_self_check_frac)
            meta["reports"][-1]["reference_self_error_max"] = self_err_max
            meta["reports"][-1]["reference_converged_module"] = bool(converged) if converged is not None else None
            meta["reports"][-1]["reference_converged"] = bool(ref_ok)
            print(f"[undersampling_check] reference ss={self.reference_supersample}: self-check "
                  f"{self_err_max} sigma vs stage criterion {self.tolerance * self.reference_self_check_frac:.3f} "
                  f"-> {'OK' if ref_ok else 'FAIL'} (module margin tolerance/4: "
                  f"{'converged' if converged else 'not converged'}); configured rung worst "
                  f"{r.max_abs_delta_over_sigma:.3f} sigma, {r.frac_above_tolerance:.2%} px above "
                  f"{self.tolerance}; excluded-disk worst {max_in_excluded}")
            if not ref_ok:
                failures.append(f"dataset {i}: reference ss={self.reference_supersample} self-check "
                                f"{self_err_max:.3f} sigma > {self.tolerance * self.reference_self_check_frac:.3f}")
            if r.max_abs_delta_over_sigma > self.max_delta_sigma:
                failures.append(f"dataset {i}: configured quadrature worst pixel "
                                f"{r.max_abs_delta_over_sigma:.3f} sigma > {self.max_delta_sigma} at (y,x)={tuple(r.argmax[1:])}")
            if r.frac_above_tolerance > self.max_frac_above:
                failures.append(f"dataset {i}: {r.frac_above_tolerance:.2%} of pixels above "
                                f"{self.tolerance} sigma > {self.max_frac_above:.2%}")
        meta["wall_time_s"] = time.perf_counter() - t0
        meta["passed"] = not failures
        meta["failures"] = failures
        if failures:
            raise RuntimeError("UndersamplingCheckStage FAILED at the sampler start point — "
                               "not sampling on an uncertified quadrature: " + "; ".join(failures))
        return StageResult(arrays=arrays, metadata=meta)

    def derive_artifacts(self, arrays):
        return {"undersampling": {k: v for k, v in arrays.items()}}

    @classmethod
    def to_posterior(cls, arrays, ctx):
        raise TypeError("UndersamplingCheckStage produces no posterior.")


# ---------------------------------------------------------------------------
# Custom stage: PartialTruthBootstrapQzStage
# ---------------------------------------------------------------------------


# Component index → name, matching the canonical
# ``PhysicalModel(lens_mass, lens_light, source_light)`` / prior ordering.
_COMPONENT_NAMES: tuple = ("lens", "lens_light", "source")

# New gigalens (dev refactor) keys priors/params by component name, not position.
# These are the dict keys emitted by the prior and consumed by the simulator,
# in the canonical [lens, lens_light, source] order.
# Map the simulator/prior component key onto the ``free`` predicate's vocabulary
# (the ``("lens", "lens_light", "source")`` names a ``free=`` spec uses).
_KEY_TO_FREE_NAME: Dict[str, str] = {
    "lens_mass": "lens",
    "lens_light": "lens_light",
    "source_light": "source",
}


def _truth_to_dict(truth_x: Any) -> Dict[str, Dict[str, Any]]:
    """Normalise a truth to the dict-keyed structure the new gigalens API uses.

    Thin wrapper over :func:`gigalens_research.inference_utils.params.to_dict_params`
    (shared with the plotting code).  Accepts either the dict form (the new
    ``prior.sample`` output, as produced by freshly generated systems) or the
    legacy 3-list form ``[lens_list, lens_light_list, source_list]`` (vela
    ``true_params`` pickles, older ``truth_x.pkl``, and hand-built fixtures).
    """
    return to_dict_params(truth_x)


@register_stage
class PartialTruthBootstrapQzStage(InferenceStage):
    """Bootstrap ``qz`` from a *partial* truth, profile-agnostically.

    Given a truth that constrains *some* parameters but leaves others free, this
    stage runs a short MAP with the constrained parameters pinned to truth and
    the free ones optimised, recovering the free-parameter values that best
    represent the truth.  Those are combined with the full simulation truth to
    build a tight diagonal ``qz`` in the inference model's unconstrained space,
    which then initialises MCLMC at truth — isolating model misspecification as
    the sole mechanism driving posterior bias (vs. initialisation failure).

    Unlike the old ``VelaBootstrapQzStage`` this class does **not** name any
    profile or re-specify any prior.  It reads the physical model and the prior
    from the :class:`InferenceContext` (``ctx``), so the free parameters keep
    *exactly* the inference prior — there is one source of truth for the prior,
    and swapping the source (or lens) profile requires no change here.

    Parameters
    ----------
    system : System
        The simulated system (provides ``truth_x``, image, noise params).
    free : Sequence[str] | Callable[[str, int, str], bool]
        Which parameters truth does *not* constrain.  Either a collection of
        component names (any of ``{"lens", "lens_light", "source"}`` — every
        parameter of those components is left free) or a predicate
        ``is_free(component_name, profile_idx, param_name) -> bool`` for
        finer-grained control.  Everything not selected is pinned to truth.
        Defaults to ``("source",)``.
    free_tag : str, optional
        Stable label for ``free`` used in the config hash.  Required (or
        derived) when ``free`` is a callable, since callables have no stable
        repr.  For a collection of names it defaults to the sorted names.
    map_num_steps, map_n_samples : int
        MAP optimisation steps / random starts for the fixed bootstrap.
    diag_scale : float
        Variance of the diagonal ``qz`` (``scale = sqrt(diag_scale)``).
    pin_eps : float
        Half-width of the ``Uniform`` used to pin a constrained parameter.
    """

    name = "bootstrap_map"
    schema_version: int = 2
    requires = ()
    produces = ("qz",)

    def __init__(
        self,
        system: Any,
        *,
        free: Any = ("source",),
        free_tag: Optional[str] = None,
        map_num_steps: int = 200,
        map_n_samples: int = 100,
        diag_scale: float = 1e-6,
        pin_eps: float = 1e-6,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(name=name or "bootstrap_map", seed=seed)
        self.system = system
        self.free = free
        self._is_free, self._free_tag = _normalise_free(free, free_tag)
        self.map_num_steps = int(map_num_steps)
        self.map_n_samples = int(map_n_samples)
        self.diag_scale = float(diag_scale)
        self.pin_eps = float(pin_eps)

    def config_hash_data(self) -> Dict[str, Any]:
        try:
            truth_bytes = pickle.dumps(
                _to_numpy_leaves(self.system.truth_x), protocol=4
            )
            truth_hash = hashlib.sha256(truth_bytes).hexdigest()[:16]
        except Exception:
            truth_hash = repr(self.system.system_id)
        return {
            "system_id": self.system.system_id,
            "truth_hash": truth_hash,
            "free": self._free_tag,
            "map_num_steps": self.map_num_steps,
            "map_n_samples": self.map_n_samples,
            "diag_scale": self.diag_scale,
            "pin_eps": self.pin_eps,
        }

    def _scene_free_components(self, model):
        """Map this stage's ``free`` spec onto scene Components by ROLE (G1 E).

        A light Component on a lensed plane has role ``"source"``; other light is
        ``"lens_light"``; mass is ``"lens"``. A Component is free iff ``self._is_free``
        returns True for it (component-level — checked with a representative param). Only
        light Components participate in the bootstrap free set (lstsq amplitudes aside,
        masses are pinned in the documented bootstrap use). Raises if the resolved free
        set is empty (a no-op bootstrap is a wiring error, not a silent pass)."""
        src_ids = {id(c) for c in model.source_plane_light()}
        free = []
        for comp in model.light_components:
            role = "source" if id(comp) in src_ids else "lens_light"
            pnames = list(comp.profile.params)
            rep = pnames[0] if pnames else "_"
            if self._is_free(role, 0, rep):
                free.append(comp)
        if not free:
            raise ValueError(
                "PartialTruthBootstrapQzStage (scene path): free set is empty after "
                f"resolving {self._free_tag!r} against the scene model's light "
                "Components; nothing would be optimised. Check the `free` spec.")
        return free

    def _run_scene(self, ctx, seed, t0, truth_x, observed_img):
        """Scene-backed bootstrap (G1 E): fix_to(truth, free=source) + short MAP + qz."""
        import jax.numpy as jnp
        import optax

        from gigalens.jax.inference import MAP
        from gigalens.jax.scene_prob_model import ImageData, ProbModel
        from gigalens_research.inference_utils.params import truth_x_to_scene_params

        model = ctx.prob_model.model
        sim_config = self.system.sim_config

        # D2 adapter: persisted 3-group truth -> scene structured params.
        truth_scene = truth_x_to_scene_params(truth_x, model)
        # D1 role mapping + partial fix: free the source-plane light, pin the rest.
        free_components = self._scene_free_components(model)
        fixed_model = model.fix_to(truth_scene, free=free_components)

        # Scene prob model on the partially-fixed model, on the SAME dataset objects and
        # amplitude MODE as the inference model (lstsq vs forward). Reusing
        # ``ctx.prob_model.datasets`` (2026-09-17) keeps the bootstrap on the inference
        # quadrature — an ``AdaptiveImageData`` builder would otherwise be bootstrapped on
        # the meta.json uniform supersample. Hardcoding "lstsq" would render a forward
        # (sampled-amplitude) model through the lstsq solver and crash; the bootstrap
        # must mirror the inference mode so gl2 (forward) works too.
        datasets = getattr(ctx.prob_model, "datasets", None)
        if not datasets:
            datasets = ImageData(observed_img, sim_config,
                                 background_rms=self.system.background_rms,
                                 exp_time=self.system.exp_time, sees="all")
        mode = getattr(ctx.prob_model, "mode", "lstsq")
        fixed_prob = ProbModel(fixed_model, datasets, mode=mode)

        optimizer = optax.adabelief(1e-2, b1=0.95, b2=0.99)
        map_samples, lps, _ = MAP(
            fixed_prob,
            optimizer=optimizer,
            n_samples=self.map_n_samples,
            num_steps=self.map_num_steps,
            seed=seed,
            output_type="best_step",
            pbar_interval=0,
        )
        lps_np = np.asarray(lps)
        map_samples_np = np.asarray(map_samples)
        best = int(np.nanargmax(lps_np))
        map_z = jnp.asarray(map_samples_np[best])  # (n_params,) in fixed_model space

        # Recovered free (source) params, keyed by the fixed_model's unique keys (which
        # are a SUBSET of the inference model's unique keys, same site->key strings).
        recovered_unique = fixed_model.bijector.forward(jnp.atleast_2d(map_z))
        recovered_unique = {k: jnp.squeeze(jnp.asarray(v))
                            for k, v in recovered_unique.items()}

        # Compose the full inference-model unique-key dict: recovered value for the free
        # (source) sites, truth value for every other site. The inference model has ALL
        # params free, so we fill each of its unique keys from its site->unique map.
        full_unique = {}
        free_vals = {}
        # A grouped (tuple-key) prior maps several sites to one ukey via cidx; its truth
        # scalars are assembled back into a vector. group_size[ukey] = #components.
        group_size = {}
        for _p, uk, ci in model._site_to_unique:
            if ci is not None:
                group_size[uk] = group_size.get(uk, 0) + 1
        pending_group = {}   # ukey -> {cidx: truth value}
        for path, ukey, cidx in model._site_to_unique:
            if ukey in recovered_unique:
                val = recovered_unique[ukey]
                if cidx is None:                       # scalar site -> labelled free val
                    free_vals[ukey.replace("/", "_")] = float(np.asarray(val))
                full_unique[ukey] = val
                continue
            # truth at this site (structured truth_scene), squeezed scalar.
            cur = truth_scene
            for key in path:
                cur = cur[key]
            tval = jnp.squeeze(jnp.asarray(cur))
            if cidx is None:
                full_unique[ukey] = tval
            else:
                pending_group.setdefault(ukey, {})[cidx] = tval
        for ukey, comps in pending_group.items():
            full_unique[ukey] = jnp.stack([comps[i] for i in range(group_size[ukey])])

        true_z = model.bijector.inverse(full_unique)
        d_dim = true_z.shape[-1]
        scale_tril = jnp.diag(jnp.ones(d_dim) * jnp.sqrt(self.diag_scale))

        return StageResult(
            arrays={
                "qz_loc": np.asarray(true_z),
                "qz_scale_tril": np.asarray(scale_tril),
                **{f"free_{k}": np.array([v]) for k, v in free_vals.items()},
            },
            metadata={
                "wall_time_s": time.perf_counter() - t0,
                "free": self._free_tag,
                "n_free": len(free_vals),
                "scene_backed": True,
                **{f"free_{k}": v for k, v in free_vals.items()},
            },
        )

    def run(
        self,
        ctx: Any,
        artifacts: Dict[str, Any],
        seed: int,
    ) -> StageResult:
        # Scene-only (old gigalens API dropped): the bootstrap uses the scene fix_to /
        # source_plane_light helpers + the D2 truth adapter (see ``_run_scene``). The
        # legacy 3-group prior-walk path was removed.
        import jax.numpy as jnp

        if getattr(ctx.prob_model, "model", None) is None:
            raise TypeError(
                "PartialTruthBootstrapQzStage requires a scene-backed InferenceContext; "
                "the legacy 3-group prior-walk path was removed with the old gigalens API.")
        return self._run_scene(
            ctx, seed, time.perf_counter(),
            self.system.truth_x, jnp.asarray(self.system.observed_image))

    def derive_artifacts(self, arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
        import jax.numpy as jnp
        loc = jnp.asarray(arrays["qz_loc"])
        # Keep loc / scale_tril dtype-consistent: under jax_enable_x64 the MAP loc is
        # float64 while the diag_scale-built scale_tril may stay float32, which trips
        # tfd's common-dtype check.
        scale_tril = jnp.asarray(arrays["qz_scale_tril"]).astype(loc.dtype)
        qz = tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)
        return {"qz": qz}

    @classmethod
    def to_posterior(cls, arrays: Dict[str, np.ndarray], ctx: Any):
        """Expose the bootstrap ``qz`` as a viewable posterior.

        Without this, ``pipeline.posterior("bootstrap_map")`` raises and the
        stage contributes nothing to reports.  Returning a ``SurrogatePosterior``
        over the (tight, truth-centred) ``qz`` lets the stage appear as an
        image-comparison / residual row in :class:`PipelineReport` and as a
        single-posterior :class:`PosteriorReport` — visualising the model image
        at the recovered-truth parameters that initialise MCLMC.
        """
        import jax.numpy as jnp
        from gigalens_research.inference_utils.posterior import SurrogatePosterior

        loc = jnp.asarray(arrays["qz_loc"])
        # Match derive_artifacts' dtype handling (loc may be x64, scale_tril x32).
        scale_tril = jnp.asarray(arrays["qz_scale_tril"]).astype(loc.dtype)
        qz = tfd.MultivariateNormalTriL(loc=loc, scale_tril=scale_tril)
        return SurrogatePosterior(ctx, qz=qz)

    def _component_name(self, ci: int) -> str:
        if ci < len(_COMPONENT_NAMES):
            return _COMPONENT_NAMES[ci]
        return f"component_{ci}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_free(free: Any, free_tag: Optional[str]):
    """Return ``(is_free_predicate, stable_tag)`` for a ``free`` spec.

    ``free`` is either a predicate ``(component, profile_idx, param) -> bool``
    or a collection of component names.  ``free_tag`` is a stable label for the
    config hash; for a collection it defaults to the sorted names, for a
    callable it falls back to the function ``__name__`` (override via
    ``free_tag`` if that is not unique enough).
    """
    if callable(free):
        tag = free_tag or getattr(free, "__name__", repr(free))
        return free, tag
    names = frozenset(free)
    tag = free_tag or ",".join(sorted(names))
    return (lambda comp, _pi, _p: comp in names), tag


def _to_numpy_leaves(obj: Any) -> Any:
    try:
        import jax
        return jax.tree.map(lambda x: np.asarray(x), obj)
    except Exception:
        return obj
