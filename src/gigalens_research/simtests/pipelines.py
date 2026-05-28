"""Built-in pipeline builders and custom inference stages.

Registered pipeline builders
----------------------------
- ``map_svi_hmc``: standard MAP → SVI → HMC pipeline (used for GL2 Sérsic test).
- ``map_bootstrap_mclmc``: fixed-lens MAP bootstrap → MCLMC (used for Vela
  shapelets systematics test).

Custom stages
-------------
- :class:`VelaBootstrapQzStage`: runs a short MAP with lens parameters pinned
  to truth to recover the shapelet geometry, then constructs a tight diagonal
  ``qz`` around the full true-parameter vector.  This ``qz`` initialises the
  subsequent MCLMC chains at truth, isolating source-model misspecification as
  the only source of posterior bias.
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

    Runs :class:`VelaBootstrapQzStage` to recover the shapelet truth geometry
    via a short MAP (lens pinned to truth), then runs MCLMC starting from a
    tight diagonal ``qz`` centred at the full truth in unconstrained space.

    Kwargs consumed:

    ``n_max`` (10),
    ``bootstrap_map_steps`` (200), ``bootstrap_map_n_samples`` (100),
    ``n_chains`` (8), ``num_burnin_steps`` (4000), ``num_results`` (4000),
    ``desired_energy_variance`` (5e-4),
    ``frac_tune1`` (0.2), ``frac_tune2`` (0.6), ``frac_tune3`` (0.2).
    """
    return [
        VelaBootstrapQzStage(
            system=system,
            n_max=int(kwargs.get("n_max", 10)),
            map_num_steps=int(kwargs.get("bootstrap_map_steps", 200)),
            map_n_samples=int(kwargs.get("bootstrap_map_n_samples", 100)),
        ),
        MCLMCStage(
            n_chains=int(kwargs.get("n_chains", 8)),
            num_burnin_steps=int(kwargs.get("num_burnin_steps", 4000)),
            num_results=int(kwargs.get("num_results", 4000)),
            desired_energy_variance=float(kwargs.get("desired_energy_variance", 5e-4)),
            frac_tune1=float(kwargs.get("frac_tune1", 0.2)),
            frac_tune2=float(kwargs.get("frac_tune2", 0.6)),
            frac_tune3=float(kwargs.get("frac_tune3", 0.2)),
        ),
    ]


# ---------------------------------------------------------------------------
# Custom stage: VelaBootstrapQzStage
# ---------------------------------------------------------------------------


@register_stage
class VelaBootstrapQzStage(InferenceStage):
    """Bootstrap ``qz`` for Vela MCLMC initialisation.

    Runs a short MAP with the lens parameters pinned to the simulation truth
    and the source shapelets free.  This recovers the shapelet geometry
    (``beta``, ``center_x``, ``center_y``) that best represents the true
    source at the true lens position.  The result is combined with the full
    simulation truth to produce a tight diagonal ``qz`` in the inference
    model's unconstrained space.

    Starting MCLMC from truth isolates source-model misspecification as the
    sole mechanism driving posterior bias (vs. initialisation failure).

    Parameters
    ----------
    system : System
        The simulated system (provides truth params, image, noise params).
    n_max : int
        Shapelet order — must match the inference model's ``n_max``.
    map_num_steps : int
        MAP optimisation steps for the fixed-lens bootstrap.
    map_n_samples : int
        Number of MAP random starts for the fixed-lens bootstrap.
    """

    name = "bootstrap_map"
    schema_version: int = 1
    requires = ()
    produces = ("qz",)

    def __init__(
        self,
        system: Any,
        *,
        n_max: int = 10,
        map_num_steps: int = 200,
        map_n_samples: int = 100,
        diag_scale: float = 1e-6,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(name=name or "bootstrap_map", seed=seed)
        self.system = system
        self.n_max = int(n_max)
        self.map_num_steps = int(map_num_steps)
        self.map_n_samples = int(map_n_samples)
        self.diag_scale = float(diag_scale)

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
            "n_max": self.n_max,
            "map_num_steps": self.map_num_steps,
            "map_n_samples": self.map_n_samples,
        }

    def run(
        self,
        ctx: Any,
        artifacts: Dict[str, Any],
        seed: int,
    ) -> StageResult:
        import jax.numpy as jnp
        import jax
        import optax

        from gigalens.jax.inference import ModellingSequence
        from gigalens.jax.model import BackwardProbModel
        from gigalens.jax.profiles.light import sersic, shapelets
        from gigalens.jax.profiles.mass import epl, shear
        from gigalens.model import PhysicalModel

        t0 = time.perf_counter()

        truth_x = self.system.truth_x
        sim_config = self.system.sim_config
        observed_img = jnp.asarray(self.system.observed_image)

        # Build fixed-lens prior: mass + lens_light pinned to truth, source free.
        def _fixed(profile_params: Dict[str, Any]) -> Any:
            dists: Dict[str, Any] = {}
            for key, val in profile_params.items():
                v = float(jnp.squeeze(jnp.asarray(val)))
                dists[key] = tfd.Uniform(v - 1e-6, v + 1e-6)
            return tfd.JointDistributionNamed(dists)

        lens_fixed_prior = tfd.JointDistributionSequential(
            [_fixed(p) for p in truth_x[0]]
        )
        # Lens light: exclude Ie (solved by lstsq).
        lens_light_no_ie = {k: v for k, v in truth_x[1][0].items() if k != "Ie"}
        lens_light_fixed_prior = tfd.JointDistributionSequential(
            [_fixed(lens_light_no_ie)]
        )
        src_free_prior = tfd.JointDistributionSequential([
            tfd.JointDistributionNamed(dict(
                beta=tfd.LogNormal(jnp.log(0.7), 0.4),
                center_x=tfd.Normal(0.0, 0.01),
                center_y=tfd.Normal(0.0, 0.01),
            ))
        ])
        fixed_prior = tfd.JointDistributionSequential([
            lens_fixed_prior, lens_light_fixed_prior, src_free_prior,
        ])

        src_model = shapelets.ShapeletsFast(
            n_max=self.n_max, use_lstsq=True, interpolate=False
        )
        fixed_phys = PhysicalModel(
            [epl.EPL(50), shear.Shear()],
            [sersic.SersicEllipse(use_lstsq=True)],
            [src_model],
        )
        fixed_prob = BackwardProbModel(
            fixed_prior, observed_img,
            background_rms=self.system.background_rms,
            exp_time=self.system.exp_time,
        )
        fixed_seq = ModellingSequence(fixed_phys, fixed_prob, sim_config)

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
        shp_true = fixed_prob.bij.forward(list(jnp.atleast_2d(map_z).T))[2][0]
        shp_true_np = {k: float(np.asarray(jnp.squeeze(v)))
                       for k, v in shp_true.items()}

        # Build full-model truth params (mass + lens_light without Ie + shapelets).
        true_params_shp = [
            truth_x[0],
            [lens_light_no_ie],
            [{k: jnp.asarray(v) for k, v in shp_true_np.items()}],
        ]

        # Map to unconstrained space via the FULL inference model's bijector.
        true_z = jnp.squeeze(jnp.stack(ctx.prob_model.bij.inverse(true_params_shp)).T)
        d = true_z.shape[-1]
        scale_tril = jnp.diag(jnp.ones(d) * jnp.sqrt(self.diag_scale))

        return StageResult(
            arrays={
                "qz_loc": np.asarray(true_z),
                "qz_scale_tril": np.asarray(scale_tril),
                **{f"shp_{k}": np.array([v]) for k, v in shp_true_np.items()},
            },
            metadata={
                "wall_time_s": time.perf_counter() - t0,
                "n_max": self.n_max,
                "shp_beta": shp_true_np.get("beta", float("nan")),
                "shp_center_x": shp_true_np.get("center_x", 0.0),
                "shp_center_y": shp_true_np.get("center_y", 0.0),
            },
        )

    def derive_artifacts(self, arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
        import jax.numpy as jnp
        qz = tfd.MultivariateNormalTriL(
            loc=jnp.asarray(arrays["qz_loc"]),
            scale_tril=jnp.asarray(arrays["qz_scale_tril"]),
        )
        return {"qz": qz}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_numpy_leaves(obj: Any) -> Any:
    try:
        import jax
        return jax.tree.map(lambda x: np.asarray(x), obj)
    except Exception:
        return obj
