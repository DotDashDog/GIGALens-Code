"""CPU tests for the DC-7.3 straggler-resample lever in laps_late_adjusted.

``p2_resample_at_chunk`` / ``p2_resample_dlogp`` / ``p2_resample_min_survivors``
gate a flag-off-by-default host-side step inside the Phase-2 adaptation loop
(the "DC-7.3 RESAMPLE step" block): after a chosen Phase-2 adaptation chunk,
straggler chains (logp <= ensemble max - dlogp) have their STATE replaced by a
uniform-with-replacement draw from survivor states, the diagonal preconditioner
and eps bisection are rebuilt from the resampled ensemble, and the run
continues. ``p2_resample_mode="retune_only"`` (v2) keeps ALL positions and
rebuilds metric+eps from the SURVIVOR subset only (isolates the poisoned-metric
fix from position replacement). Design:
``docs/logs/laps_prior_init_investigation.md`` (DC-7.3 section). Cheap: dim=4
Gaussian, 16-32 chains, tens of steps, single CPU device.
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from gigalens_research.inference.laps_late_adjusted import LAPS_late_adjusted

DIM = 4
SIGMA = jnp.asarray([1.0, 2.0, 0.5, 1.5])
SIGMA_NP = np.asarray([1.0, 2.0, 0.5, 1.5])


def gauss_logp(z):
    return -0.5 * jnp.sum((z / SIGMA) ** 2)


def run(**kw):
    base = dict(
        logdensity_fn=gauss_logp, dim=DIM, num_chains=16,
        num_unadjusted_steps=40, num_adjusted_steps=32,
        init_mode="cold", cold_scale=1.0, early_stop=False, seed=3,
    )
    base.update(kw)
    return LAPS_late_adjusted(**base)


# --------------------------------------------------------------------------- #
# 1 — off (default) is a bitwise no-op                                        #
# --------------------------------------------------------------------------- #
def test_off_is_noop():
    res_default = run()
    res_explicit_none = run(p2_resample_at_chunk=None)
    assert res_default.resample_info is None
    assert res_explicit_none.resample_info is None
    np.testing.assert_array_equal(np.asarray(res_default.samples),
                                  np.asarray(res_explicit_none.samples))
    np.testing.assert_array_equal(np.asarray(res_default.p2_step_size),
                                  np.asarray(res_explicit_none.p2_step_size))
    np.testing.assert_array_equal(np.asarray(res_default.p2_accept),
                                  np.asarray(res_explicit_none.p2_accept))
    np.testing.assert_array_equal(np.asarray(res_default.p2_frozen),
                                  np.asarray(res_explicit_none.p2_frozen))
    np.testing.assert_array_equal(np.asarray(res_default.precond_var),
                                  np.asarray(res_explicit_none.precond_var))


# --------------------------------------------------------------------------- #
# 2 — converged ensemble at resample time -> guard skips (0 stragglers)       #
# --------------------------------------------------------------------------- #
def test_skip_guard():
    # cold_scale=1 (same order as the target scale) + the FULL default
    # num_adjusted_steps=200/p2_chunk_size=8 budget (25 chunks -- ample room
    # to bracket, refine and freeze eps, per the module docstring) lets the
    # ensemble settle well before the LATE resample chunk (the very last of
    # the full budget): all chains sit within the default dlogp cut of the
    # ensemble max, so the resample guard should find zero stragglers and
    # skip.
    res = run(num_adjusted_steps=200, p2_resample_at_chunk=25)
    info = res.resample_info
    assert info is not None
    assert info["skipped"] is True
    assert info["n_stragglers"] == 0
    assert np.all(np.isfinite(np.asarray(res.samples)))


# --------------------------------------------------------------------------- #
# 3/4 — FORCE the active resample path (shared expensive run, module-scoped)   #
# --------------------------------------------------------------------------- #
NUM_CHAINS_ACTIVE = 32


@pytest.fixture(scope="module")
def active_resample_result():
    # Tiny Phase 1 (2 steps) + broad init (cold_scale=6, still >>1 relative to
    # SIGMA) leaves the Phase-2 ensemble overdispersed at the FIRST Phase-2
    # chunk boundary; a tight dlogp=5.0 cut (< the default ~7.66) then
    # virtually surely carves off nonzero stragglers while leaving a non-
    # degenerate survivor pool (~7/32 measured). p2_resample_min_survivors=2
    # (the validated floor) keeps the guard from skipping in any case.
    return run(
        num_chains=NUM_CHAINS_ACTIVE,
        num_unadjusted_steps=2, chunk_size=2,
        cold_scale=6.0,
        num_adjusted_steps=40, p2_chunk_size=8,
        p2_resample_at_chunk=1,
        p2_resample_dlogp=5.0,
        p2_resample_min_survivors=2,
        track_chains=True,
    )


def test_active_resample(active_resample_result):
    res = active_resample_result
    info = res.resample_info
    assert info is not None
    assert info["skipped"] is False

    n_surv, n_strag = info["n_survivors"], info["n_stragglers"]
    assert n_surv + n_strag == NUM_CHAINS_ACTIVE
    assert n_strag > 0                     # tight cut: stragglers actually cut

    strag = np.asarray(info["stragglers"])
    donors = np.asarray(info["donors"])
    assert strag.shape == donors.shape == (n_strag,)

    # donors are drawn only from survivors: no donor index may itself be a
    # straggler index (stragglers/survivors partition the full chain set).
    strag_set = set(strag.tolist())
    assert strag_set.isdisjoint(set(donors.tolist()))
    assert np.all((donors >= 0) & (donors < NUM_CHAINS_ACTIVE))

    # Immediately after the resample every chain must clear the cut.
    assert res.chain_traj is not None
    snap = info["snap_index"]
    lp_at_snap = np.asarray(res.chain_traj["p2_logp"][snap])
    assert lp_at_snap.shape == (NUM_CHAINS_ACTIVE,)
    assert np.all(lp_at_snap > info["cut"])

    samples = np.asarray(res.samples).reshape((-1, DIM))
    assert np.all(np.isfinite(samples))
    ratio = samples.std(axis=0) / SIGMA_NP
    assert np.all(ratio > 0.5) and np.all(ratio < 2.0)


def test_duplicates_decorrelate(active_resample_result):
    res = active_resample_result
    info = res.resample_info
    strag = np.asarray(info["stragglers"])
    donors = np.asarray(info["donors"])
    n_check = min(5, strag.size)
    assert n_check > 0

    pos = np.asarray(res.chain_traj["p2_pos"])          # (n_ck2, M, dim)
    snap = info["snap_index"]
    pos_snap = pos[snap]
    pos_last = pos[-1]

    for i in range(n_check):
        s, d = int(strag[i]), int(donors[i])
        # duplicated at the resample snapshot (straggler state overwritten by
        # its donor's state)
        np.testing.assert_array_equal(pos_snap[s], pos_snap[d])
        # MH evolution with independent per-chain keys must separate them by
        # the final Phase-2 snapshot
        assert not np.array_equal(pos_last[s], pos_last[d])


# --------------------------------------------------------------------------- #
# 5 — retune_only mode (v2): keep ALL positions, rebuild metric+eps only      #
# --------------------------------------------------------------------------- #
def test_retune_only():
    # Same forced-active config as the fixture, but mode="retune_only": the
    # metric/eps rebuild + bisection restart fires (resample_info present, not
    # skipped) while every chain KEEPS its position -- the post-resample
    # snapshot must be identical to the immediately-preceding Phase-2 snapshot
    # (no replacement), and donors is None (no straggler -> donor map).
    res = run(
        num_chains=NUM_CHAINS_ACTIVE,
        num_unadjusted_steps=2, chunk_size=2,
        cold_scale=6.0,
        num_adjusted_steps=40, p2_chunk_size=8,
        p2_resample_at_chunk=1,
        p2_resample_dlogp=5.0,
        p2_resample_min_survivors=2,
        p2_resample_mode="retune_only",
        track_chains=True,
    )
    info = res.resample_info
    assert info is not None
    assert info["skipped"] is False
    assert info["mode"] == "retune_only"
    assert info["donors"] is None
    assert info["n_survivors"] + info["n_stragglers"] == NUM_CHAINS_ACTIVE

    pos = np.asarray(res.chain_traj["p2_pos"])          # (n_ck2, M, dim)
    snap = info["snap_index"]
    assert snap >= 1
    # positions untouched: post-resample snapshot == end-of-chunk snapshot
    np.testing.assert_array_equal(pos[snap], pos[snap - 1])

    assert np.all(np.isfinite(np.asarray(res.samples)))


# --------------------------------------------------------------------------- #
# 6 — validation errors                                                       #
# --------------------------------------------------------------------------- #
def test_validation_errors():
    with pytest.raises(ValueError):
        run(p2_resample_at_chunk=0)
    with pytest.raises(ValueError):
        run(p2_resample_min_survivors=1)
    with pytest.raises(ValueError):
        run(p2_resample_mode="bogus")
