"""Tests for gigalens_research.inference_utils.pipeline.PTMCLMCStage, the
pipeline wrapper around gigalens.jax.experimental.pt_mclmc.sample_pt_mclmc.

Degrades gracefully: if gigalens.jax.experimental.pt_mclmc is not importable
in this environment (it lives on gigalens branch pt-mclmc-experimental, PR
seanxuseanxu/gigalens#66, not yet merged), the sampling-dependent tests are
skipped -- this repo's CI won't have the module until that PR merges. The
config-hash-discipline, num_burnin_rounds-validation, and ImportError-message
tests do not need the sampler and always run.

Runnable both via pytest and directly: `python3 test_pt_mclmc_stage.py`.

NOTE: PTMCLMCStage.to_posterior is NOT exercised here -- it needs a real
scene InferenceContext (bijector, prob_model, ...), not this file's
duck-typed fake ctx. Its implementation is byte-identical in form to the
already-exercised MCLMCStage.to_posterior / MAMSStage.to_posterior
(`SamplerPosterior(ctx, samples_z=arrays["samples_z"])`).
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Select a headless backend BEFORE anything (incl. gigalens_research.plotting)
# has a chance to import pyplot with a different one.
import matplotlib

matplotlib.use("Agg")
from matplotlib.figure import Figure

import contextlib
import sys
import types

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

try:
    import pytest
except ImportError:  # pytest not installed in this interpreter; __main__ still works
    pytest = None

_PT_MCLMC_MOD = "gigalens.jax.experimental.pt_mclmc"
try:
    import gigalens.jax.experimental.pt_mclmc  # noqa: F401

    _HAVE_PT_MCLMC = True
except ImportError:
    _HAVE_PT_MCLMC = False

_SKIP_REASON = (
    "gigalens.jax.experimental.pt_mclmc is not importable in this environment "
    "(needs gigalens.jax.experimental.pt_mclmc: branch pt-mclmc-experimental, on linusu-dev-merge since PR #66) -- "
    "skipping PT-MCLMC sampling tests."
)

from gigalens_research.inference_utils.pipeline import (
    PTMCLMCStage,
    StageDiagnostics,
    stable_hash,
)
from gigalens_research.plotting.diagnostics import (
    has_diagnostic_plotter,
    plot_stage_diagnostics,
)


@contextlib.contextmanager
def _expect_raises(exc_type):
    """pytest.raises if pytest is importable, else a manual try/except -- so
    this file works both under `pytest` and via `python3 test_pt_mclmc_stage.py`."""
    if pytest is not None:
        with pytest.raises(exc_type):
            yield
        return
    try:
        yield
    except exc_type:
        return
    else:
        raise AssertionError(f"expected {exc_type} to be raised, nothing was")


def _skip(msg):
    if pytest is not None:
        pytest.skip(msg)
    else:
        print(f"[SKIP] {msg}")


# ---------------------------------------------------------------------------
# Toy target: 6-D bimodal Gaussian mixture, modes +/-mu on axis 0 only, sigma
# 1, weights 0.8 (major, +mu) / 0.2 (minor, -mu) -- copied from the gigalens
# repo's tests/test_pt_mclmc.py toy, wrapped to return a (log_prob, red_chi2)
# TUPLE so PTMCLMCStage.run's `[0]` seam is actually exercised.
# ---------------------------------------------------------------------------
_D = 6
_MU = 4.0
_W = (0.8, 0.2)


def _bimodal_log_density(z):
    logw = jnp.log(jnp.asarray(_W, dtype=jnp.float64))
    shift = jnp.zeros(_D, dtype=jnp.float64).at[0].set(_MU)
    d_pos = jnp.sum(jnp.square(z - shift))
    d_neg = jnp.sum(jnp.square(z + shift))
    lp = jnp.stack([logw[0] - 0.5 * d_pos, logw[1] - 0.5 * d_neg])
    return jax.scipy.special.logsumexp(lp)


def _toy_log_prob(z):
    # (log_prob, reduced_chi2)-shaped return, mirroring the real
    # ProbModel.log_prob seam that MCLMC_JIT/PTMCLMCStage.run rely on
    # (`ctx.model_seq.prob_model.log_prob(z)[0]`). The chi2-like companion
    # value is a placeholder never used by the sampler.
    lp = _bimodal_log_density(z)
    return lp, -2.0 * lp / _D


def _make_ctx():
    return types.SimpleNamespace(
        model_seq=types.SimpleNamespace(
            prob_model=types.SimpleNamespace(log_prob=_toy_log_prob)
        )
    )


_INIT = jnp.zeros(_D, dtype=jnp.float64).at[0].set(_MU)  # major-mode "MAP"


# --------------------------------------------------------------------- (a)
def test_duck_typed_ctx_integration():
    if not _HAVE_PT_MCLMC:
        _skip(_SKIP_REASON)
        return
    ctx = _make_ctx()
    stage = PTMCLMCStage(
        beta_min=0.0625,
        n_walkers=8,
        n_rounds=400,
        num_burnin_rounds=200,
        metric_windows=(50, 100, 150),
        indicator=lambda z: z[..., 0] > 0,
        indicator_id="axis0_v1",
        debug=True,
        progress_every=0,
    )
    result = stage.run(ctx, {"z_best": _INIT}, seed=0)

    samples_z = np.asarray(result.arrays["samples_z"])
    assert samples_z.shape == (8, 200, 6), samples_z.shape

    md = result.metadata
    for key in (
        "wall_time_s", "num_chains", "num_steps", "n_params", "betas",
        "n_rungs", "num_burnin_rounds", "swap_acceptance", "round_trips_total",
        "n_nan_reverts", "u0_identity_rel", "metric_frozen", "transport_warning",
        "debug",
    ):
        assert key in md, f"missing metadata key {key!r}"

    assert md["round_trips_total"] > 0, (
        "zero round trips: the ladder never transported a discovery")
    assert md["n_nan_reverts"] == 0
    assert md["transport_warning"] is False
    for rate in md["swap_acceptance"]:
        assert 0.15 < rate < 0.95, (
            f"swap acceptance {rate} outside (0.15, 0.95); "
            f"full list={md['swap_acceptance']}")

    assert result.diagnostics, "expected non-empty diagnostics with debug=True"
    assert "cold_indicator" in result.diagnostics

    # Minor-mode occupancy over the PUBLISHED samples (post-burn-in only) --
    # same 0.12 tolerance as the gigalens repo's own toy test; not tuned here.
    major = samples_z[..., 0] > 0
    occ_minor = 1.0 - float(major.mean())
    assert abs(occ_minor - _W[1]) <= 0.12, (
        f"minor-mode occupancy {occ_minor:.3f} not within 0.12 of true weight "
        f"{_W[1]}")

    print(f"[test a] samples_z.shape={samples_z.shape} "
          f"round_trips_total={md['round_trips_total']} "
          f"swap_acceptance={np.round(md['swap_acceptance'], 3).tolist()} "
          f"n_nan_reverts={md['n_nan_reverts']} occ_minor={occ_minor:.3f}")


# --------------------------------------------------------------------- (b)
def test_config_hash_discipline():
    stage = PTMCLMCStage(beta_min=0.4)
    data = stage.config_hash_data()
    assert not any(callable(v) for v in data.values()), (
        f"config_hash_data leaked a callable: {data}")
    stable_hash(data)  # must not raise (no unhashable/unhandled types)

    with _expect_raises(ValueError):
        PTMCLMCStage(beta_min=0.4, indicator=lambda z: z[..., 0] > 0)


# --------------------------------------------------------------------- (c)
def test_num_burnin_rounds_must_be_less_than_n_rounds():
    with _expect_raises(ValueError):
        PTMCLMCStage(beta_min=0.4, n_rounds=100, num_burnin_rounds=100)
    with _expect_raises(ValueError):
        PTMCLMCStage(beta_min=0.4, n_rounds=100, num_burnin_rounds=150)


# --------------------------------------------------------------------- (d)
def test_plotter_smoke():
    if not _HAVE_PT_MCLMC:
        _skip(_SKIP_REASON)
        return

    assert has_diagnostic_plotter("PTMCLMCStage")

    ctx = _make_ctx()
    stage = PTMCLMCStage(
        beta_min=0.0625,
        n_walkers=8,
        n_rounds=400,
        num_burnin_rounds=200,
        metric_windows=(50, 100, 150),
        indicator=lambda z: z[..., 0] > 0,
        indicator_id="axis0_v1",
        debug=True,
        progress_every=0,
    )
    result = stage.run(ctx, {"z_best": _INIT}, seed=0)

    diag = StageDiagnostics(
        stage_name="pt_mclmc",
        stage_class="PTMCLMCStage",
        arrays=result.diagnostics,
        config=stage.diagnostics_config(),
        ctx=ctx,
    )
    fig = plot_stage_diagnostics(diag)
    assert isinstance(fig, Figure)
    print(f"[test d] plot_stage_diagnostics returned a {type(fig).__name__}")


# --------------------------------------------------------------------- (e)
def test_import_error_names_branch():
    """Simulate gigalens lacking the experimental pt_mclmc module (the
    pre-merge state) by nulling its sys.modules entry, regardless of whether
    it's actually installed here -- exercises PTMCLMCStage.run's lazy-import
    error path either way."""
    ctx = _make_ctx()
    stage = PTMCLMCStage(beta_min=0.4, n_rounds=2, num_burnin_rounds=1)

    old = sys.modules.pop(_PT_MCLMC_MOD, None)
    sys.modules[_PT_MCLMC_MOD] = None  # forces ImportError on the next `import`
    try:
        caught = None
        try:
            stage.run(ctx, {"z_best": jnp.zeros(2, dtype=jnp.float64)}, seed=0)
        except ImportError as e:
            caught = e
        assert caught is not None, "expected ImportError, none was raised"
        msg = str(caught)
        assert "pt-mclmc-experimental" in msg, msg
        assert "#66" in msg, msg
        print(f"[test e] ImportError message OK: {msg}")
    finally:
        sys.modules.pop(_PT_MCLMC_MOD, None)
        if old is not None:
            sys.modules[_PT_MCLMC_MOD] = old


if __name__ == "__main__":
    test_config_hash_discipline()
    print("[test b] config-hash discipline: OK")
    test_num_burnin_rounds_must_be_less_than_n_rounds()
    print("[test c] num_burnin_rounds >= n_rounds raises: OK")
    test_import_error_names_branch()

    if not _HAVE_PT_MCLMC:
        print(f"[SKIP] {_SKIP_REASON}")
    else:
        test_duck_typed_ctx_integration()
        test_plotter_smoke()
        print("ALL TESTS PASSED")
