"""Built-in pipeline builders and custom inference stages.

Registered pipeline builders
----------------------------
- ``map_svi_hmc``: standard MAP → SVI → HMC pipeline (used for GL2 Sérsic test).
- ``map_bootstrap_mclmc``: fixed-lens MAP bootstrap → MCLMC (used for Vela
  shapelets systematics test).

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
    ``frac_tune1`` (0.2), ``frac_tune2`` (0.6), ``frac_tune3`` (0.2).

    ``bootstrap_diag_scale`` is the variance of the tight diagonal ``qz`` the
    chains are initialised from (``scale = sqrt(diag_scale)``); ``bootstrap_pin_eps``
    is the half-width of the ``Uniform`` used to pin the truth-constrained
    parameters during the bootstrap MAP.
    """
    return [
        PartialTruthBootstrapQzStage(
            system=system,
            free=("source",),
            map_num_steps=int(kwargs.get("bootstrap_map_steps", 200)),
            map_n_samples=int(kwargs.get("bootstrap_map_n_samples", 100)),
            diag_scale=float(kwargs.get("bootstrap_diag_scale", 1e-6)),
            pin_eps=float(kwargs.get("bootstrap_pin_eps", 1e-6)),
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

        from gigalens.jax.inference import ModellingSequence
        from gigalens.jax.scene_prob_model import Dataset, ProbModel
        from gigalens_research.inference_utils.params import truth_x_to_scene_params

        model = ctx.model_seq.scene_model
        sim_config = self.system.sim_config

        # D2 adapter: persisted 3-group truth -> scene structured params.
        truth_scene = truth_x_to_scene_params(truth_x, model)
        # D1 role mapping + partial fix: free the source-plane light, pin the rest.
        free_components = self._scene_free_components(model)
        fixed_model = model.fix_to(truth_scene, free=free_components)

        # Scene prob model on the partially-fixed model, same dataset/noise AND amplitude
        # MODE as the inference model (lstsq vs forward). Hardcoding "lstsq" would render a
        # forward (sampled-amplitude) model through the lstsq solver and crash; the
        # bootstrap must mirror the inference mode so gl2 (forward) works too.
        ds = Dataset(observed_img, sim_config,
                     background_rms=self.system.background_rms,
                     exp_time=self.system.exp_time, sees="all")
        mode = getattr(ctx.prob_model, "mode", "lstsq")
        fixed_prob = ProbModel(fixed_model, ds, mode=mode)
        fixed_seq = ModellingSequence.from_scene(fixed_model, fixed_prob, sim_config)

        optimizer = optax.adabelief(1e-2, b1=0.95, b2=0.99)
        map_samples, lps, _ = fixed_seq.MAP(
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
        recovered_unique = fixed_model.bijector.forward(list(jnp.atleast_2d(map_z).T))
        recovered_unique = {k: jnp.squeeze(jnp.asarray(v))
                            for k, v in recovered_unique.items()}

        # Compose the full inference-model unique-key dict: recovered value for the free
        # (source) sites, truth value for every other site. The inference model has ALL
        # params free, so we fill each of its unique keys from its site->unique map.
        full_unique = {}
        free_vals = {}
        for path, ukey in model._site_to_unique:
            if ukey in recovered_unique:
                val = recovered_unique[ukey]
                free_vals[ukey.replace("/", "_")] = float(np.asarray(val))
            else:
                # truth at this site (structured truth_scene), squeezed scalar.
                cur = truth_scene
                for key in path:
                    cur = cur[key]
                val = jnp.squeeze(jnp.asarray(cur))
            full_unique[ukey] = val

        true_z = jnp.stack(model.bijector.inverse(full_unique))
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
        import jax.numpy as jnp
        import jax.tree_util as jtu
        import optax

        from gigalens.jax.inference import ModellingSequence

        t0 = time.perf_counter()

        truth_x = self.system.truth_x
        sim_config = self.system.sim_config
        observed_img = jnp.asarray(self.system.observed_image)

        # G1 dual-path: a scene-backed ModellingSequence uses the scene fix_to /
        # source_plane_light helpers + the D2 truth adapter, NOT the legacy 3-group
        # prior walk below. Branch early; the legacy path is unchanged for the other
        # (not-yet-migrated) builders.
        if getattr(ctx.model_seq, "scene_model", None) is not None:
            return self._run_scene(ctx, seed, t0, truth_x, observed_img)

        # The inference model is the single source of truth for both the
        # profiles (physical model) and the free-parameter priors.
        inf_prior = ctx.prob_model.prior

        # Build the fixed prior by walking the inference prior's nested
        # JointDistributionNamed({component: {profile_idx: {param: dist}}})
        # structure: keep each free param's inference distribution, replace each
        # pinned param with a near-delta Uniform at its truth value.  Parameters
        # solved by lstsq (e.g. Sersic ``Ie``, shapelet amplitudes) are absent
        # from the prior and so are correctly never pinned or freed.
        truth_dict = _truth_to_dict(truth_x)
        fixed_components: Dict[str, Any] = {}
        for ckey, comp in inf_prior.model.items():
            cname = _KEY_TO_FREE_NAME.get(ckey, ckey)
            prof_blocks: Dict[str, Any] = {}
            for pkey, named in comp.model.items():
                pi = int(pkey)
                dists: Dict[str, Any] = {}
                for pname, dist in named.model.items():
                    if self._is_free(cname, pi, pname):
                        dists[pname] = dist
                    else:
                        v = float(jnp.squeeze(jnp.asarray(truth_dict[ckey][pkey][pname])))
                        dists[pname] = tfd.Uniform(v - self.pin_eps, v + self.pin_eps)
                prof_blocks[pkey] = tfd.JointDistributionNamed(dists)
            fixed_components[ckey] = tfd.JointDistributionNamed(prof_blocks)
        fixed_prior = tfd.JointDistributionNamed(fixed_components)

        # Reuse the inference physical model (identical profiles) and rebuild the
        # prob model with the same class + the system's noise; only the prior
        # changes.  Noise comes from ``system`` (not from ``ctx.prob_model``
        # attributes) because ``BackwardProbModel`` stores a precomputed
        # ``err_map`` rather than ``background_rms`` / ``exp_time``.
        fixed_prob = type(ctx.prob_model)(
            fixed_prior,
            observed_img,
            background_rms=self.system.background_rms,
            exp_time=self.system.exp_time,
        )
        fixed_seq = ModellingSequence(ctx.phys_model, fixed_prob, sim_config)

        optimizer = optax.adabelief(1e-2, b1=0.95, b2=0.99)
        map_samples, lps, _ = fixed_seq.MAP(
            optimizer=optimizer,
            n_samples=self.map_n_samples,
            num_steps=self.map_num_steps,
            seed=seed,
            output_type="best_step",
            pbar_interval=0,
        )
        # output_type="best_step" → samples shape (num_steps, n_params).
        # Pick the globally best step.
        lps_np = np.asarray(lps)
        map_samples_np = np.asarray(map_samples)
        best = int(np.nanargmax(lps_np))
        map_z = jnp.asarray(map_samples_np[best])  # (n_params,)
        # Constrained-space params recovered by the MAP, same nested structure
        # as the inference model.
        recovered = fixed_prob.bij.forward(list(jnp.atleast_2d(map_z).T))

        # Compose the full constrained truth: free leaves take their recovered
        # MAP value, pinned leaves take the exact truth value.  Same dict-keyed
        # structure as the inference prior.
        full_params: Dict[str, Any] = {}
        free_vals: Dict[str, float] = {}
        for ckey, comp in inf_prior.model.items():
            cname = _KEY_TO_FREE_NAME.get(ckey, ckey)
            profs: Dict[str, Any] = {}
            for pkey, named in comp.model.items():
                pi = int(pkey)
                d: Dict[str, Any] = {}
                for pname in named.model.keys():
                    if self._is_free(cname, pi, pname):
                        val = float(np.asarray(jnp.squeeze(recovered[ckey][pkey][pname])))
                        free_vals[f"{cname}{pi}_{pname}"] = val
                        d[pname] = jnp.asarray(val)
                    else:
                        d[pname] = jnp.asarray(truth_dict[ckey][pkey][pname])
                profs[pkey] = d
            full_params[ckey] = profs

        # Map to unconstrained space via the FULL inference model's bijector.
        # Truth leaves may have shape (1,) while recovered leaves are scalars
        # (), so squeeze everything to () for a consistent stackable list.
        full_params = jtu.tree_map(
            lambda x: jnp.squeeze(jnp.asarray(x)), full_params
        )
        true_z = jnp.stack(ctx.prob_model.bij.inverse(full_params))
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
                **{f"free_{k}": v for k, v in free_vals.items()},
            },
        )

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
