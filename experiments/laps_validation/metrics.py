r"""LAPS success metrics (spec D) + the finite-ensemble noise floors.

PRIMARY metric (pre-registered, fixed in advance): the paper's second-moment
bias on the observable f = x_i^2 (spec D, Eqs. 12-13, p.6):

    b^2_i   = ( Ehat[x_i^2] - E_true[x_i^2] )^2 / Var_true[x_i^2]
    b^2_max = max_i b^2_i ,   b^2_avg = (1/d) sum_i b^2_i
    SUCCESS  <=> b^2 < 0.01   (both max and avg), i.e. ~>=100 effective samples.

NOISE FLOORS (so "passing" is read against the irreducible finite-M level):
  * b^2 floor.  Ehat[x_i^2] is a mean of M iid draws of x_i^2, so its sampling
    variance is Var_true[x_i^2]/M.  Hence the EXPECTED b^2_i from perfect
    sampling (zero algorithmic bias) is

        E[b^2_i] = (Var_true[x_i^2]/M) / Var_true[x_i^2] = 1/M.

    So ``b2_floor = 1/M`` for every dim. At M=512 that is 1.95e-3 -- below the
    0.01 success bar, so the bar is reachable, but b^2 ~ 1/M is the ENTITLEMENT,
    not a defect; only b^2 >> 1/M signals algorithmic (equilibration) bias.
  * moment-error floor.  ``moment_floor_i = sqrt(Var_true[x_i^2]/M)`` is the
    irreducible std of the E[x_i^2] estimate, in moment units.
  * switch floor (the delta_i gate, spec switch-resolution).  For paper obs
    f=x_i^2: ``switch_floor_i = sqrt(Var_true[x_i^2]/M) / E_true[x_i^2]``; the
    self_calibrated switch fires when max_i delta_i/floor_i < k. This is the
    KNOWN-ANSWER reference floor used in the delta plot (the sampler itself gates
    on the ONLINE ensemble estimate of the same quantity).

METRIC BLIND SPOT (spec method-discipline 4): b^2 is a SECOND-moment bias only.
It is blind to (a) errors that preserve E[x_i^2] (e.g. a sign-symmetric shift of
mass, or first-moment bias on a zero-mean coord), and (b) cross-coordinate
correlation error. Mean-recovery and Var-recovery are reported alongside so a
b^2 "pass" with a broken mean/correlation is still visible.
"""

import numpy as np


def b2_metrics(samples, target):
    """Per-dim b^2 + b2_max/b2_avg from the final ensemble ``samples`` (M, d)."""
    samples = np.asarray(samples, dtype=np.float64)
    ex2_hat = np.mean(samples ** 2, axis=0)                 # (d,)
    b2_i = (ex2_hat - target.ex2_true) ** 2 / target.var_x2_true
    return {
        "ex2_hat": ex2_hat,
        "b2_per_dim": b2_i,
        "b2_max": float(np.max(b2_i)),
        "b2_avg": float(np.mean(b2_i)),
        "b2_success": bool(np.max(b2_i) < 0.01 and np.mean(b2_i) < 0.01),
    }


def mean_var_recovery(samples, target):
    """Per-dim first-moment and variance recovery (the b^2 blind-spot guards)."""
    samples = np.asarray(samples, dtype=np.float64)
    mean_hat = np.mean(samples, axis=0)
    var_hat = np.var(samples, axis=0, ddof=0)
    mean_err = mean_hat - target.mean_true
    var_rel_err = (var_hat - target.var_true) / target.var_true
    return {
        "mean_hat": mean_hat,
        "var_hat": var_hat,
        "mean_err": mean_err,
        "mean_abs_err_max": float(np.max(np.abs(mean_err))),
        "var_rel_err": var_rel_err,
        "max_var_rel_err": float(np.max(np.abs(var_rel_err))),
    }


def b2_trajectory(p1_obs_sq, target, switch_index, M):
    r"""b^2-trajectory from the Phase-1 ensemble-mean second-moment history.

    ``p1_obs_sq`` is ``(phase1_len, dim)`` = per-Phase-1-step ensemble mean of
    ``x_i^2``.  Per the spec, the SAME b^2 second-moment-bias metric is applied to
    each step of the Phase-1 trajectory (using the KNOWN-ANSWER true moments), so
    we can read off the residual bias *while Phase 1 is still running*:

        b2_i_traj[t]  = (p1_obs_sq[t] - E_true[x_i^2])^2 / Var_true[x_i^2]
        b2_avg_traj[t]= mean_i b2_i_traj[t]
        b2_max_traj[t]= max_i  b2_i_traj[t]

    Exposed scalars:
      * ``b2_avg_at_switch`` / ``b2_max_at_switch`` -- the trajectory value AT
        ``switch_index`` (clipped to ``phase1_len-1``). This is the residual
        second-moment bias at the instant the Phase-1->2 switch fired: the direct
        switch-ripeness test (small => switched ripe; large => premature switch).
      * ``steps_to_floor`` -- first Phase-1 step at which ``b2_avg_traj`` drops
        below ``2/M`` (else ``phase1_len`` if it never does). The equilibration-cost
        metric (less seed-noisy than the final b^2). The ``2/M`` level (not ``1/M``)
        is used because the per-step ensemble-mean E[x_i^2] is itself a finite-M
        estimate, so its own sampling floor is ~1/M; ``2/M`` is a 2x-floor "ripe"
        band that is robustly reachable without tripping on a single lucky step.

    BLIND SPOT (same as the final b^2): second-moment only -- blind to first-moment
    / correlation error that preserves E[x_i^2]; and the trajectory is the ENSEMBLE
    mean of x_i^2 at each step, so it cannot see within-step chain autocorrelation.
    """
    obs = np.asarray(p1_obs_sq, dtype=np.float64)
    T1 = obs.shape[0] if obs.ndim == 2 else 0
    if T1 == 0:
        return {
            "b2_avg_traj": np.zeros(0), "b2_max_traj": np.zeros(0),
            "b2_avg_at_switch": float("nan"), "b2_max_at_switch": float("nan"),
            "steps_to_floor": 0, "b2_floor2": 2.0 / float(M),
        }
    ex2 = np.asarray(target.ex2_true, dtype=np.float64)[None, :]
    vx2 = np.asarray(target.var_x2_true, dtype=np.float64)[None, :]
    b2_i_traj = (obs - ex2) ** 2 / vx2                      # (T1, dim)
    b2_avg_traj = b2_i_traj.mean(axis=1)                    # (T1,)
    b2_max_traj = b2_i_traj.max(axis=1)                     # (T1,)

    si = int(np.clip(int(switch_index), 0, T1 - 1))
    floor2 = 2.0 / float(M)
    below = np.where(b2_avg_traj < floor2)[0]
    steps_to_floor = int(below[0]) if below.size else int(T1)
    return {
        "b2_avg_traj": b2_avg_traj,
        "b2_max_traj": b2_max_traj,
        "b2_avg_at_switch": float(b2_avg_traj[si]),
        "b2_max_at_switch": float(b2_max_traj[si]),
        "steps_to_floor": steps_to_floor,
        "b2_floor2": floor2,
    }


def joint_shape_metrics(samples, target):
    r"""JOINT-shape bias the marginal b^2 is BLIND to (spec method-discipline 4).

    The marginal second-moment b^2 only constrains per-coordinate ``E[x_i^2]``; it
    cannot see (a) cross-coordinate coupling ``E[x_i x_j]`` (i!=j) or (b) the
    banana's CURVATURE, which lives in the third cross-moment ``E[x_0^2 x_1]`` and
    is INVISIBLE to every second moment (the banana's linear cross-covariance
    ``E[x_0 x_1]`` is ~0, so even a full covariance check is blind to the twist).

    Two diagnostics, both normalized into the b^2 units (bias^2 / Var_true, floor
    1/M) so they read directly against the marginal b^2:

      cross-moment bias (off-diagonal correlation error)
        c2_cross_ij = (Ehat[x_i x_j] - E_true[x_i x_j])^2 / Var_true[x_i x_j],
        reported as c2_cross_max / c2_cross_avg over i<j.

      curvature residual (the banana twist; the marginal-b^2 blind spot)
        c2_curv = (Ehat[x_i^2 x_j] - E_true[x_i^2 x_j])^2 / Var_true[x_i^2 x_j],
        with (i,j) the target's curvature monomial (banana: (0,1), truth 2b).

    SUCCESS reads identically to b^2: < 0.01 (~>=100 effective samples); the
    irreducible finite-M entitlement is 1/M. A b^2 "pass" with c2_curv >> 1/M is a
    sampler that nailed the marginals but missed the joint banana shape.

    BLIND SPOT of THIS metric: c2_cross is second-order (linear correlation only);
    c2_curv probes ONE chosen monomial. Together they still cannot see higher-order
    / multi-coordinate structure beyond the defining twist.
    """
    samples = np.asarray(samples, dtype=np.float64)
    M, d = samples.shape
    jt = target.joint_truth()
    cross_hat = (samples.T @ samples) / M                    # Ehat[x_i x_j] (d,d)
    c2_cross = (cross_hat - jt["cross_true"]) ** 2 / jt["var_cross_true"]
    iu = np.triu_indices(d, k=1)                             # off-diagonal i<j
    c2_cross_off = c2_cross[iu]
    i, j = jt["curv_idx"]
    curv_hat = float(np.mean(samples[:, i] ** 2 * samples[:, j]))
    c2_curv = (curv_hat - jt["curv_true"]) ** 2 / jt["var_curv_true"]
    return {
        "cross_hat": cross_hat,
        "c2_cross_per_pair": c2_cross_off,
        "c2_cross_max": float(np.max(c2_cross_off)) if c2_cross_off.size else 0.0,
        "c2_cross_avg": float(np.mean(c2_cross_off)) if c2_cross_off.size else 0.0,
        "curv_idx": (i, j),
        "curv_true": jt["curv_true"],
        "curv_hat": curv_hat,
        "curv_abs_err": float(abs(curv_hat - jt["curv_true"])),
        "c2_curv": float(c2_curv),
        "joint_floor": 1.0 / float(M),
        "joint_success": bool(np.max(c2_cross_off, initial=0.0) < 0.01 and c2_curv < 0.01),
    }


def noise_floors(target, M):
    """Finite-M floors: b^2 floor (1/M), moment-error floor, switch (delta) floor."""
    M = float(M)
    return {
        "b2_floor": 1.0 / M,                                          # scalar
        "moment_floor_i": np.sqrt(target.var_x2_true / M),            # (d,)
        "switch_floor_i": np.sqrt(target.var_x2_true / M)
        / np.maximum(np.abs(target.ex2_true), 1e-30),                # (d,) paper obs
    }
