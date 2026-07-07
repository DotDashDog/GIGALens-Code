r"""run_one: drive LAPS once on a known-answer target, return a full record.

The record carries TWO tiers (equal weight; spec method-discipline 5, plots
before metrics):

* INTERNALS (validate the machinery): D_tilde(t), EEVPD_obs(t)/EEVPD_wanted(t),
  step_size(t), L(t), the reconstructed per-dim delta_i(t)/floor_i and delta_max(t)
  with the switch step, switch_index, Phase-2 acceptance(t), p2_frozen(t).
* RESULTS (require ground truth): b2_max/b2_avg, per-dim mean & Var recovery,
  max var rel err.

The delta_i/floor_i internal is RECONSTRUCTED on the host from the returned
``p1_obs_sq`` (paper obs E[x_i^2]) / ``p1_obs_mean`` (emaus obs E[x_i]) histories,
using the SAME windowed sigma/mu the sampler's switch uses (laps_core math), and
divided by the KNOWN-ANSWER equilibrium floor from ``metrics.switch_floor_i``.
HONEST NOTE: the sampler's live gate used the ONLINE ensemble floor; we compare
against the true floor as the known-answer reference (they coincide near
equilibrium). The returned ``p1_delta_max`` (the sampler's own per-step window
delta_max, NaN before the window fills) is carried through unchanged.
"""

import json
import os

import numpy as np

from gigalens_research.inference.laps_late_adjusted import LAPS_late_adjusted

try:  # package import (python -m ...) or script import (python grid.py)
    from . import metrics as _metrics
except ImportError:
    import metrics as _metrics


def _windowed_delta(obs_hist, T, switch):
    """Per-step windowed delta_i (T, then trailing-T window), NaN before full window.

    Mirrors ``laps_core.phase1_switch``: delta = |sigma/mu| (paper, ddof=1) or
    (sigma/mu)^2 (emaus). Returns (T1, d) with NaN rows until the window fills.
    """
    obs_hist = np.asarray(obs_hist, dtype=np.float64)
    T1, d = obs_hist.shape
    out = np.full((T1, d), np.nan)
    for t in range(T - 1, T1):
        w = obs_hist[t - T + 1: t + 1]
        mu = w.mean(axis=0)
        sigma = w.std(axis=0, ddof=1)
        ratio = sigma / mu
        out[t] = np.abs(ratio) if switch == "paper" else ratio ** 2
    return out


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def run_one(target, *, run_id=None, qz_mode="warm", num_chains=512,
            json_path=None, **laps_kwargs):
    """Run LAPS once on ``target`` and return a full diagnostics+results record.

    ``qz_mode`` ("warm"|"off") selects the qz surrogate; ``laps_kwargs`` are
    forwarded to ``LAPS_late_adjusted`` (schedule, switch, switch_mode, switch_k,
    init_mode, num_unadjusted_steps, num_adjusted_steps, chunk_size, C, ...).
    """
    qz = target.make_qz(qz_mode)
    switch = laps_kwargs.get("switch", "paper")
    save_frac = laps_kwargs.get("save_frac", 0.2)
    num_unadj = laps_kwargs.get("num_unadjusted_steps", 300)

    res = LAPS_late_adjusted(
        target.log_prob, qz, dim=target.dim, num_chains=num_chains, **laps_kwargs)

    samples = np.asarray(res.samples, dtype=np.float64)
    M = samples.shape[0]

    b2 = _metrics.b2_metrics(samples, target)
    rec = _metrics.mean_var_recovery(samples, target)
    floors = _metrics.noise_floors(target, M)

    # b^2-trajectory (ALWAYS from p1_obs_sq = ensemble-mean E[x_i^2] history):
    # residual 2nd-moment bias AT the switch step + steps-to-2/M-floor.
    b2traj = _metrics.b2_trajectory(
        res.p1_obs_sq, target, int(res.switch_index), M)

    # reconstruct per-dim delta_i(t) and the delta/floor ratio against the
    # KNOWN-ANSWER equilibrium floor (see module docstring).
    obs_hist = res.p1_obs_sq if switch == "paper" else res.p1_obs_mean
    T_window = max(2, round(save_frac * num_unadj))
    delta_i_hist = _windowed_delta(obs_hist, T_window, switch)
    switch_floor_i = floors["switch_floor_i"]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio_hist = delta_i_hist / switch_floor_i[None, :]      # (T1, d)
    # rows before the window fills are all-NaN by design (no switch eval yet);
    # nanmax of an all-NaN row -> NaN (expected gap in the plot), warning silenced.
    all_nan_rows = np.all(np.isnan(ratio_hist), axis=1)
    ratio_max_hist = np.full(ratio_hist.shape[0], np.nan)
    if np.any(~all_nan_rows):
        ratio_max_hist[~all_nan_rows] = np.nanmax(ratio_hist[~all_nan_rows], axis=1)

    config = dict(
        target=target.name, dim=int(target.dim), num_chains=int(M),
        qz_mode=qz_mode, run_id=run_id, **laps_kwargs)

    record = {
        "config": config,
        # ---- INTERNALS ----
        "internals": {
            "D_tilde": np.asarray(res.p1_D_tilde),
            "eevpd_obs": np.asarray(res.p1_eevpd_obs),
            "eevpd_wanted": np.asarray(res.p1_eevpd_wanted),
            "step_size": np.asarray(res.p1_step_size),
            "L": np.asarray(res.p1_L),
            "delta_i_hist": delta_i_hist,
            "delta_max": np.asarray(res.p1_delta_max),
            "delta_over_floor_max": ratio_max_hist,
            "switch_floor_i": switch_floor_i,
            "b2_avg_traj": b2traj["b2_avg_traj"],
            "b2_max_traj": b2traj["b2_max_traj"],
            "switch_k": float(laps_kwargs.get("switch_k", 1.5)),
            "switch_index": int(res.switch_index),
            "switch_index_paper": int(res.switch_index_paper),
            "switch_index_emaus": int(res.switch_index_emaus),
            "switched": bool(res.switched),
            "phase1_len": int(res.phase1_len),
            "p2_accept": np.asarray(res.p2_accept),
            "p2_step_size": np.asarray(res.p2_step_size),
            "p2_frozen": np.asarray(res.p2_frozen),
            "target_accept": float(res.target_accept),
            "integrator_order": int(res.integrator_order),
            "precond_var": np.asarray(res.precond_var),
        },
        # ---- RESULTS ----
        "results": {
            "ex2_true": np.asarray(target.ex2_true),
            "ex2_hat": b2["ex2_hat"],
            "b2_per_dim": b2["b2_per_dim"],
            "b2_max": b2["b2_max"],
            "b2_avg": b2["b2_avg"],
            "b2_success": b2["b2_success"],
            "b2_floor": floors["b2_floor"],
            # ---- b^2-trajectory (switch-ripeness + equilibration cost) ----
            "b2_avg_at_switch": b2traj["b2_avg_at_switch"],
            "b2_max_at_switch": b2traj["b2_max_at_switch"],
            "steps_to_floor": b2traj["steps_to_floor"],
            "b2_floor2": b2traj["b2_floor2"],
            "mean_true": np.asarray(target.mean_true),
            "mean_hat": rec["mean_hat"],
            "mean_err": rec["mean_err"],
            "mean_abs_err_max": rec["mean_abs_err_max"],
            "var_true": np.asarray(target.var_true),
            "var_hat": rec["var_hat"],
            "var_rel_err": rec["var_rel_err"],
            "max_var_rel_err": rec["max_var_rel_err"],
            "moment_se": np.asarray(getattr(target, "moment_se", np.zeros(target.dim))),
        },
        # final-ensemble samples kept out of JSON (large); plotting uses moments.
    }

    # health flags
    record["nan_free"] = bool(
        np.all(np.isfinite(samples))
        and np.all(np.isfinite(record["internals"]["D_tilde"]))
        and np.all(np.isfinite(record["internals"]["eevpd_obs"])))

    if json_path is not None:
        save_record_json(record, json_path)
    return record


def save_record_json(record, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_to_serializable(record), fh, indent=2)
    return path
