"""CPU tests for :class:`LAPSStage` (``gigalens_research.inference_utils.pipeline``),
the pipeline-stage wrapper around ``LAPS_late_adjusted_JIT``.

Two things are tested at two different costs:

1. Preset resolution / override / validation logic (``LAPSStage.__init__`` and
   ``.config``/``.requires``) -- pure Python, no sampler call, effectively
   free. This is where the certified cold/warm hyperparameter dicts (see
   ``experiments/laps_validation/handoff/diag_resample.py`` [512 chains] and
   ``diag_resample128.py`` [128 chains, the default operating point] and
   ``docs/logs/laps_prior_init_investigation.md`` DC-7.3/DC-7.4) are pinned.
2. An end-to-end smoke run of ``LAPSStage.run`` on a tiny toy Gaussian
   "model", exercising the actual stage -> ``LAPS_late_adjusted_JIT`` ->
   ``StageResult`` plumbing (samples shape, metadata, diagnostics arrays).

The smoke run does NOT build a real gigalens lens model / scene / simulator
(that's heavyweight -- MAP+SVI+a rendered image). Instead it fakes the
handful of ``prob_model`` attributes ``LAPS_late_adjusted_JIT`` actually
touches: ``log_prob(z)`` (cold+warm), ``prior.sample`` + ``bij.inverse``
(cold/"prior" init only). This is a legitimate exercise of the STAGE's
wiring (the real ``LAPS_late_adjusted_JIT``/``LAPS_late_adjusted`` run for
real, unmocked) without paying for a full lens pipeline -- judgment call,
noted for review. The resample lever itself (``p2_resample_at_chunk``) is
already unit-tested at the sampler level in ``test_laps_resample.py``; the
smoke runs here disable it (tiny budgets can't reach chunk 13) to stay cheap
and instead just check the stage packages a `None` ``resample_info`` (and,
implicitly, that passing the kwarg through at all doesn't break anything).
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp

from gigalens_research.inference_utils.pipeline import InferenceContext, LAPSStage

tfd = tfp.distributions

DIM = 4


# --------------------------------------------------------------------------- #
# Fakes: the minimal prob_model surface LAPS_late_adjusted_JIT touches        #
# --------------------------------------------------------------------------- #
class _FakeBij:
    """Mimics gigalens' bijector INVERSE map: constrained (n, dim) array ->
    list of dim (n,) arrays (the same shape LAPS_late_adjusted_JIT expects
    from ``prob_model.bij.inverse``, per its "prior" init_mode docstring)."""

    def inverse(self, x):
        x = jnp.asarray(x)
        return [x[:, i] for i in range(x.shape[-1])]


class _FakeProbModel:
    """A toy diagonal-Gaussian "posterior" with a prior + bijector, standing
    in for a real gigalens ``ProbModel`` so ``LAPS_late_adjusted_JIT`` can run
    unmodified end-to-end without a rendered lens image."""

    def __init__(self, dim):
        self.dim = dim
        self._sigma = jnp.linspace(1.0, 2.0, dim)
        self.prior = tfd.MultivariateNormalDiag(
            loc=jnp.zeros(dim), scale_diag=jnp.ones(dim))
        self.bij = _FakeBij()

    def log_prob(self, z):
        # LAPS_late_adjusted_JIT does `prob_model.log_prob(z)[0]`.
        return (-0.5 * jnp.sum((z / self._sigma) ** 2), None)


class _FakeModelSeq:
    def __init__(self, dim):
        self.prob_model = _FakeProbModel(dim)


def _fake_ctx(dim=DIM):
    model_seq = _FakeModelSeq(dim)
    return InferenceContext(
        phys_model=None, prob_model=model_seq.prob_model,
        sim_config=None, model_seq=model_seq,
    )


# --------------------------------------------------------------------------- #
# 1 -- cold preset resolves to the certified defaults                        #
# --------------------------------------------------------------------------- #
def test_cold_preset_matches_certified_defaults():
    stage = LAPSStage(init="cold")
    assert stage.config == dict(
        init_mode="prior",
        num_chains=128,
        num_unadjusted_steps=300,
        num_adjusted_steps=248,
        early_stop=False,
        track_chains=True,
        p2_resample_at_chunk=13,
        p2_resample_min_survivors=24,
        p2_resample_mode="replace",
    )
    assert stage.requires == ()
    assert stage.extra_kwargs == {}


# --------------------------------------------------------------------------- #
# 2 -- warm preset resolves to the certified warm defaults, resampler OFF    #
# --------------------------------------------------------------------------- #
def test_warm_preset_matches_certified_defaults_and_resampler_off():
    stage = LAPSStage(init="warm")
    assert stage.config == dict(
        init_mode="warm",
        num_chains=128,
        num_unadjusted_steps=300,
        num_adjusted_steps=248,
        early_stop=False,
        track_chains=True,
        p2_resample_at_chunk=None,
        p2_resample_min_survivors=32,
        p2_resample_mode="replace",
    )
    assert stage.config["p2_resample_at_chunk"] is None
    assert stage.requires == ("qz",)


# --------------------------------------------------------------------------- #
# 3 -- user overrides win over the preset                                    #
# --------------------------------------------------------------------------- #
def test_user_overrides_win_over_preset():
    stage = LAPSStage(
        init="cold",
        num_chains=512,
        p2_resample_min_survivors=32,   # reproduce the certified 512-chain arm
        p2_resample_at_chunk=None,      # explicit None must stick (not _UNSET)
        extra_kwargs={"velocity_init": "gradient"},
    )
    assert stage.config["num_chains"] == 512
    assert stage.config["p2_resample_min_survivors"] == 32
    assert stage.config["p2_resample_at_chunk"] is None
    # Untouched fields still come from the cold preset.
    assert stage.config["num_adjusted_steps"] == 248
    assert stage.config["init_mode"] == "prior"
    assert stage.extra_kwargs == {"velocity_init": "gradient"}


# --------------------------------------------------------------------------- #
# 4 -- bad init value raises ValueError                                      #
# --------------------------------------------------------------------------- #
def test_bad_init_raises():
    with pytest.raises(ValueError):
        LAPSStage(init="bogus")


# --------------------------------------------------------------------------- #
# 5 -- end-to-end smoke run: cold (prior-init) on the toy Gaussian           #
# --------------------------------------------------------------------------- #
def test_cold_stage_smoke_run():
    ctx = _fake_ctx(DIM)
    stage = LAPSStage(
        init="cold",
        num_chains=8,
        num_unadjusted_steps=6,
        num_adjusted_steps=8,
        p2_resample_at_chunk=None,   # tiny budget can't reach chunk 13; see module docstring
        name="laps_cold_smoke",
    )
    result = stage.run(ctx, artifacts={}, seed=0)

    samples = result.arrays["samples_z"]
    assert samples.shape == (8, 1, DIM)   # (num_chains, p2_keep_per_chain=1, dim)
    assert np.all(np.isfinite(samples))

    assert result.metadata["num_chains"] == 8
    assert result.metadata["n_params"] == DIM
    assert result.metadata["init"] == "cold"
    assert result.metadata["init_mode"] == "prior"
    assert result.metadata["resample_info"] is None

    for key in ("p1_step_size", "p2_step_size", "p2_accept", "precond_var"):
        assert key in result.diagnostics
        assert np.all(np.isfinite(np.asarray(result.diagnostics[key])))


# --------------------------------------------------------------------------- #
# 6 -- end-to-end smoke run: warm (qz-init) on the toy Gaussian              #
# --------------------------------------------------------------------------- #
def test_warm_stage_smoke_run():
    ctx = _fake_ctx(DIM)
    qz = tfd.MultivariateNormalDiag(
        loc=jnp.zeros(DIM), scale_diag=0.5 * jnp.ones(DIM))
    stage = LAPSStage(
        init="warm",
        num_chains=8,
        num_unadjusted_steps=6,
        num_adjusted_steps=8,
        name="laps_warm_smoke",
    )
    assert stage.requires == ("qz",)
    result = stage.run(ctx, artifacts={"qz": qz}, seed=0)

    samples = result.arrays["samples_z"]
    assert samples.shape == (8, 1, DIM)
    assert np.all(np.isfinite(samples))
    assert result.metadata["init_mode"] == "warm"
    assert result.metadata["resample_info"] is None  # warm preset: resampler off


# --------------------------------------------------------------------------- #
# 7 -- registered in the stage registry (posterior_from_disk / disk-reload)  #
# --------------------------------------------------------------------------- #
def test_registered_and_has_to_posterior():
    from gigalens_research.inference_utils.pipeline import _STAGE_REGISTRY
    assert _STAGE_REGISTRY["LAPSStage"] is LAPSStage

    ctx = _fake_ctx(DIM)
    arrays = {"samples_z": np.zeros((8, 1, DIM))}
    posterior = LAPSStage.to_posterior(arrays, ctx)
    assert posterior is not None
