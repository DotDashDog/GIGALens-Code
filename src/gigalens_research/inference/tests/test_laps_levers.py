"""CPU tests for the stocktake F1/F2/F3 cold-start levers in laps_late_adjusted.

F1  velocity_init="gradient"   (bj-faithful grad-aligned + equipartition signs)
F2  nan_eps_halving=True       (bj-faithful eps*0.5 when any chain NaN-rejects)
F3  precond_source="final"     (bj/paper end-state ensemble Var)

All levers default OFF; the default path must be unchanged (regression checks
compare against quantities reconstructable from the returned histories).
Cheap: dim=4 Gaussian, 16 chains, tens of steps, single CPU device.
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from gigalens_research.inference.laps_late_adjusted import (
    LAPS_late_adjusted,
    _gradient_equipartition_momentum,
)

DIM = 4
SIGMA = jnp.asarray([1.0, 2.0, 0.5, 1.5])


def gauss_logp(z):
    return -0.5 * jnp.sum((z / SIGMA) ** 2)


def run(**kw):
    base = dict(
        logdensity_fn=gauss_logp, dim=DIM, num_chains=16,
        num_unadjusted_steps=40, num_adjusted_steps=16,
        init_mode="cold", cold_scale=10.0, early_stop=False, seed=3,
    )
    base.update(kw)
    return LAPS_late_adjusted(**base)


# --------------------------------------------------------------------------- #
# F1 — gradient/equipartition velocity init                                    #
# --------------------------------------------------------------------------- #
def test_f1_momentum_math_overdispersed():
    # Overdispersed Gaussian ensemble: g = -x/sigma^2, E_ii = E[x_i^2]/sigma_i^2
    # >> 1 -> signs all +1 -> momentum = +g/|g| (toward the mode), unit norm.
    key = jax.random.key(0)
    x = 10.0 * jax.random.normal(key, (64, DIM))
    g = -x / SIGMA**2
    rand_mom = jax.random.normal(jax.random.key(1), (64, DIM))
    mom = _gradient_equipartition_momentum(x, rand_mom, g, jnp.float64)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(mom), axis=-1), 1.0, rtol=1e-12)
    expected = np.asarray(g / jnp.linalg.norm(g, axis=-1, keepdims=True))
    np.testing.assert_allclose(np.asarray(mom), expected, rtol=1e-12)


def test_f1_momentum_math_underdispersed_sign_flip():
    # Underdispersed ensemble (scale 0.01): E_ii << 1 -> signs all -1 ->
    # momentum = -g/|g| (away from the mode = expand), still unit norm.
    key = jax.random.key(0)
    x = 0.01 * jax.random.normal(key, (64, DIM))
    g = -x / SIGMA**2
    rand_mom = jax.random.normal(jax.random.key(1), (64, DIM))
    mom = _gradient_equipartition_momentum(x, rand_mom, g, jnp.float64)
    expected = -np.asarray(g / jnp.linalg.norm(g, axis=-1, keepdims=True))
    np.testing.assert_allclose(np.asarray(mom), expected, rtol=1e-12)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(mom), axis=-1), 1.0, rtol=1e-12)


def test_f1_momentum_math_per_coordinate_mixed_signs():
    # Hand-built ensemble where coordinate 0 is overdispersed and coordinate 1
    # underdispersed: sign vector must be per-coordinate (+1, -1, ...).
    x = jnp.zeros((32, DIM)).at[:, 0].set(5.0).at[:, 1].set(0.01)
    x = x.at[:16, 0].multiply(-1.0).at[:16, 1].multiply(-1.0)  # zero-mean
    g = -x / SIGMA**2
    rand_mom = jax.random.normal(jax.random.key(1), (32, DIM))
    E = np.mean(np.asarray(-x * g), axis=0)
    assert E[0] > 1.0 and E[1] < 1.0
    mom = np.asarray(
        _gradient_equipartition_momentum(x, rand_mom, g, jnp.float64))
    gn = np.asarray(g / jnp.linalg.norm(g, axis=-1, keepdims=True))
    np.testing.assert_allclose(mom[:, 0], gn[:, 0], rtol=1e-12)   # +1
    np.testing.assert_allclose(mom[:, 1], -gn[:, 1], rtol=1e-12)  # -1


def test_f1_nonfinite_rows_keep_random_momentum():
    x = jnp.ones((8, DIM))
    g = (-x / SIGMA**2).at[0, 0].set(jnp.nan)
    rand_mom = jax.random.normal(jax.random.key(1), (8, DIM))
    mom = np.asarray(
        _gradient_equipartition_momentum(x, rand_mom, g, jnp.float64))
    np.testing.assert_allclose(mom[0], np.asarray(rand_mom)[0])   # rail
    assert np.all(np.isfinite(mom))


def test_f1_end_to_end_runs():
    res = run(velocity_init="gradient")
    assert res.phase1_len == 40
    assert np.all(np.isfinite(np.asarray(res.samples)))


def test_f1_changes_trajectory():
    # wiring check: the lever must actually alter the dynamics from step 0
    res_off = run(phase2_enabled=False)
    res_on = run(phase2_enabled=False, velocity_init="gradient")
    assert not np.array_equal(np.asarray(res_on.p1_obs_sq),
                              np.asarray(res_off.p1_obs_sq))


# --------------------------------------------------------------------------- #
# F1-companion — first-step L = inf (bj laps_burn_in.py:261)                   #
# --------------------------------------------------------------------------- #
def test_L0_inf_runs_finite_and_differs():
    res_off = run(phase2_enabled=False, velocity_init="gradient")
    res_on = run(phase2_enabled=False, velocity_init="gradient", L0_inf=True)
    # L=inf on step 1 => no Maruyama noise (nu=1); everything stays finite and
    # the recorded per-step L history is the ensemble-derived (finite) value.
    assert np.all(np.isfinite(np.asarray(res_on.samples)))
    assert np.all(np.isfinite(np.asarray(res_on.p1_L)))
    # step 1 differs (refresh suppressed) -> trajectories diverge
    assert not np.array_equal(np.asarray(res_on.p1_obs_sq),
                              np.asarray(res_off.p1_obs_sq))


# --------------------------------------------------------------------------- #
# F2 — NaN -> eps halving                                                      #
# --------------------------------------------------------------------------- #
def nanny_logp(z):
    # sqrt(z_0) poisons any chain with z_0 < 0: its logp/energy goes NaN, the
    # kernel NaN-rejects it every step -> nan_frac > 0 deterministically.
    return -0.5 * jnp.sum(z**2) + 0.0 * jnp.sqrt(z[0])


def test_f2_nan_stream_and_halving():
    kw = dict(logdensity_fn=nanny_logp, num_unadjusted_steps=12,
              num_adjusted_steps=4, phase2_enabled=False)
    res_off = run(**kw)
    res_on = run(nan_eps_halving=True, **kw)
    # identical seeds -> identical init; about half the cold chains sit at
    # z_0 < 0 and NaN-reject: the diagnostic stream must see them in BOTH runs
    assert np.all(res_off.p1_nan_frac > 0.0)
    assert np.all(res_on.p1_nan_frac > 0.0)
    # ON: every step with nans multiplies eps by exactly 0.5
    # (bj laps_burn_in.py:333-335).
    ss_on = np.asarray(res_on.p1_step_size)
    np.testing.assert_allclose(ss_on[1:] / ss_on[:-1], 0.5, rtol=1e-12)
    assert np.all(np.isfinite(ss_on))
    # OFF: the baseline controller has no brake; zeroed-dE chains bias EEVPD_obs
    # low so eps GROWS until the ensemble statistics (and eps itself) go
    # non-finite -- the exact robustness gap F2 closes. Either outcome (runaway
    # NaN eps, or merely a larger eps) demonstrates the gap.
    ss_off_last = np.asarray(res_off.p1_step_size)[-1]
    assert (not np.isfinite(ss_off_last)) or (ss_on[-1] < ss_off_last)


def test_f2_no_op_on_clean_target():
    res_off = run(phase2_enabled=False)
    res_on = run(phase2_enabled=False, nan_eps_halving=True)
    assert np.all(res_off.p1_nan_frac == 0.0)
    # no NaNs -> the brake never fires -> the step-size trajectory is unchanged.
    # NOT bitwise: computing nan_frac changes XLA fusion on GPU, perturbing the
    # last ~2 ULP (measured 2e-15 rel.); assert to near-machine tolerance.
    # (no samples assertion: the ULP perturbation amplifies chaotically through
    # the dynamics; the brake's target quantity is the step size, so that is
    # the semantic no-op claim.)
    np.testing.assert_allclose(
        np.asarray(res_on.p1_step_size), np.asarray(res_off.p1_step_size),
        rtol=1e-12)


# --------------------------------------------------------------------------- #
# F3 — end-state preconditioner                                                #
# --------------------------------------------------------------------------- #
def test_f3_final_vs_pooled():
    res_pool = run()
    res_fin = run(precond_source="final")
    # Same seed + levers off elsewhere -> identical Phase-1 histories
    np.testing.assert_array_equal(
        np.asarray(res_fin.p1_obs_sq), np.asarray(res_pool.p1_obs_sq))
    sq, mn = np.asarray(res_pool.p1_obs_sq), np.asarray(res_pool.p1_obs_mean)
    # each mode reproduces its formula from the returned histories
    half = res_pool.phase1_len // 2
    want_pool = np.maximum(
        sq[half:].mean(0) - mn[half:].mean(0) ** 2, 1e-12)
    want_fin = np.maximum(sq[-1] - mn[-1] ** 2, 1e-12)
    np.testing.assert_allclose(np.asarray(res_pool.precond_var), want_pool,
                               rtol=1e-12)
    np.testing.assert_allclose(np.asarray(res_fin.precond_var), want_fin,
                               rtol=1e-12)
    assert not np.allclose(want_pool, want_fin)   # generically different


def test_track_chains_snapshots():
    res_off = run()
    assert res_off.chain_traj is None
    res = run(track_chains=True)
    # pure host-side instrumentation: bitwise no-op on the dynamics
    np.testing.assert_array_equal(np.asarray(res.samples),
                                  np.asarray(res_off.samples))
    tr = res.chain_traj
    # Phase 1: 40 steps / chunk 25 -> 2 chunks + initial snapshot = 3
    assert tr["p1_pos"].shape == (3, 16, DIM)
    assert tr["p1_logp"].shape == (3, 16)
    # Phase 2: 16 traj / chunk 8 -> 2 chunks
    assert tr["p2_pos"].shape == (2, 16, DIM)
    assert np.all(np.isfinite(tr["p1_logp"]))
    # snapshot 0 is the initial ensemble: cold_scale=10 -> broad; final is tighter
    assert tr["p1_pos"][0].std() > tr["p2_pos"][-1].std()


def test_bad_flag_values_raise():
    with pytest.raises(ValueError):
        run(velocity_init="bogus")
    with pytest.raises(ValueError):
        run(precond_source="bogus")
