r"""LAPS notebook handoff — clean, importable helpers for a REAL lens posterior.

Call these three from a notebook once you have a gigalens ``model_seq`` (the
ProbModel-bearing scene sequence) and a ``qz`` SVI surrogate (exposing
``.sample((n,), seed)`` / ``.mean()`` / ``.covariance()``):

    from laps_handoff import run_laps, diagnose, compare_warm_cold

    res = run_laps(model_seq, qz, init_mode="cold")     # validated defaults baked
    health = diagnose(res, out_png="laps_diag.png")     # GROUND-TRUTH-FREE health
    cmp = compare_warm_cold(model_seq, qz, out_png="laps_warm_cold.png")

Everything here is ground-truth-FREE: a real lens posterior has no known moments,
so the only convergence evidence is internal health (the Phase-1 controller and
switch ripeness) plus TWO INDEPENDENT INITS AGREEING. See ``laps_lensing_checklist.md``.

The VALIDATED DEFAULTS (from the CPU known-answer validation, P6a) are baked into
``run_laps`` and can be overridden by keyword:
    schedule="paper"  switch="paper"  switch_mode="self_calibrated"
    switch_k=1.5  C=0.025  alpha=2  N=15  target_accept 0.7(MN2)/0.9(MN4)
"""
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gigalens_research.inference.laps_late_adjusted import LAPS_late_adjusted_JIT


# Frozen validated defaults (P6a / laps_validation_report.md §6).
VALIDATED_DEFAULTS = dict(
    schedule="paper",
    switch="paper",
    switch_mode="self_calibrated",
    switch_k=1.5,
    C=0.025,
    alpha=2.0,
    num_unadjusted_steps=300,
    num_adjusted_steps=200,
    chunk_size=25,
    steps_per_trajectory=15,
)


# --------------------------------------------------------------------------- #
# 1. run_laps                                                                  #
# --------------------------------------------------------------------------- #
def run_laps(model_seq, qz, init_mode="cold", **overrides):
    r"""Run LAPS on a gigalens posterior with the validated defaults baked in.

    Parameters
    ----------
    model_seq : gigalens scene sequence; ``model_seq.prob_model.log_prob(z)[0]``
                is the unconstrained-``z`` log density (same entry MCLMC uses).
    qz        : SVI surrogate (``.sample/.mean/.covariance``). Used to seed chains
                in warm mode and to infer ``dim`` in BOTH modes.
    init_mode : "cold" (DEFAULT; robust, no early-switch risk) or "warm" (faster
                for a GOOD qz). See P6a guidance in the checklist.
    overrides : any LAPS hyperparameter (num_chains, num_unadjusted_steps,
                num_adjusted_steps, seed, switch_k, C, ...). Overrides the defaults.

    Returns the ``LAPSResults`` namedtuple (``.samples`` is (num_chains, dim), one
    sample per chain = the posterior ensemble; plus the full Phase-1/2 diagnostics).
    """
    kw = dict(VALIDATED_DEFAULTS)
    kw.update(overrides)
    kw["init_mode"] = init_mode
    return LAPS_late_adjusted_JIT(model_seq, qz, **kw)


# --------------------------------------------------------------------------- #
# cross-chain split-Rhat / ESS on the final ensemble                          #
# --------------------------------------------------------------------------- #
def _split_rhat_ess(samples, n_groups=8):
    r"""Between-subensemble split-Rhat + an efficiency ESS on the (M, d) ensemble.

    LAPS returns ONE sample per chain, so there is no within-chain time series for
    a classical autocorrelation R-hat/ESS. Instead we PARTITION the M independent
    ensemble members into ``n_groups`` sub-ensembles and run the standard
    between/within Gelman split-Rhat: if the independent sub-ensembles AGREE
    (Rhat ~ 1), the ensemble is homogeneous. ESS is reported as
    ``ESS = M / Rhat^2`` -- the between-group efficiency (= M when groups agree,
    < M when they disagree), NOT an autocorrelation ESS. Rank-normalized (robust
    to heavy tails) when scipy is available, else raw.
    """
    x = np.asarray(samples, np.float64)
    Mtot, d = x.shape
    G = int(n_groups)
    while G > 1 and Mtot % G != 0:
        G -= 1
    G = max(2, G)
    n = Mtot // G
    x = x[: G * n]

    # robust rank normalization per dim (folded-rank Rhat backbone)
    try:
        from scipy.stats import rankdata, norm
        def _rn(col):
            r = rankdata(col)
            return norm.ppf((r - 0.5) / col.size)
        z = np.stack([_rn(x[:, j]) for j in range(d)], axis=1)
    except Exception:
        z = (x - x.mean(0)) / (x.std(0) + 1e-30)

    zg = z[: G * n].reshape(G, n, d)
    cm = zg.mean(axis=1)                      # (G, d) group means
    cv = zg.var(axis=1, ddof=1)               # (G, d) group vars
    B = n * cm.var(axis=0, ddof=1)            # (d,)
    W = cv.mean(axis=0)                        # (d,)
    var_plus = (n - 1) / n * W + B / n
    with np.errstate(invalid="ignore", divide="ignore"):
        rhat = np.sqrt(np.where(W > 0, var_plus / W, np.nan))
        ess = float(Mtot) / np.maximum(rhat, 1.0) ** 2
    return {
        "rhat_per_dim": rhat,
        "rhat_max": float(np.nanmax(rhat)),
        "ess_per_dim": ess,
        "ess_min": float(np.nanmin(ess)),
        "n_groups": G,
        "group_size": n,
        "M": Mtot,
    }


def _ess_dim(chains, var_plus, eps=1e-30):
    r"""Multi-chain bulk-ESS for one dim from rank-normalized (C, n) draws.

    Geyer initial-positive-sequence on the cross-chain-averaged autocorrelation
    ``rho_t`` (normalized by the pooled ``var_plus``). With short within-chain
    series (small ``n``) this is a coarse but honest ESS; capped at the raw draw
    count ``C*n`` and floored at the chain count.
    """
    C, n = chains.shape
    if n < 2 or var_plus <= eps:
        return float(C)
    dev = chains - chains.mean(axis=1, keepdims=True)
    tau = 1.0
    t = 1
    while t + 1 < n:
        r1 = float(np.mean(dev[:, : n - t] * dev[:, t:])) / var_plus
        r2 = float(np.mean(dev[:, : n - t - 1] * dev[:, t + 1:])) / var_plus
        if r1 + r2 < 0.0:                  # Geyer: stop at first negative pair sum
            break
        tau += 2.0 * (r1 + r2)
        t += 2
    return float(min(C * n / max(tau, eps), C * n))


def _rhat_ess_chains(samples3d):
    r"""Rank-normalized split-Rhat + ESS using the (chain, draw) structure.

    ``samples3d`` is ``(M chains, K draws, d)`` with ``K >= 2`` (CHANGE B: the
    post-freeze thinned within-chain series). This is the PROPER Gelman-Rubin: the
    M chains are the groups and the K thinned draws are the within-chain time
    series, so R-hat/ESS are now meaningful (a single sample per chain, K=1, has
    no within-chain series -- the caller routes that to the between-subensemble
    fallback). Each chain is split in half when ``K >= 4`` (more, shorter chains,
    the standard split-Rhat). Returns the same dict keys as ``_split_rhat_ess``.
    """
    x = np.asarray(samples3d, np.float64)
    M, K, d = x.shape
    if K >= 4:
        h = K // 2
        x = np.concatenate([x[:, :h, :], x[:, h:2 * h, :]], axis=0)  # (2M, h, d)
    C, n, _ = x.shape
    try:
        from scipy.stats import rankdata, norm
        z = np.empty_like(x)
        for j in range(d):
            r = rankdata(x[:, :, j].reshape(-1)).reshape(C, n)
            z[:, :, j] = norm.ppf((r - 0.5) / (C * n))
    except Exception:
        z = (x - x.mean(axis=(0, 1), keepdims=True)) / (
            x.std(axis=(0, 1), keepdims=True) + 1e-30)
    chain_means = z.mean(axis=1)               # (C, d)
    chain_vars = z.var(axis=1, ddof=1)         # (C, d)
    W = chain_vars.mean(axis=0)                # (d,)
    B = n * chain_means.var(axis=0, ddof=1)    # (d,)
    var_plus = (n - 1) / n * W + B / n
    with np.errstate(invalid="ignore", divide="ignore"):
        rhat = np.sqrt(np.where(W > 0, var_plus / W, np.nan))
    ess = np.array([_ess_dim(z[:, :, j], float(var_plus[j])) for j in range(d)])
    return {
        "rhat_per_dim": rhat,
        "rhat_max": float(np.nanmax(rhat)),
        "ess_per_dim": ess,
        "ess_min": float(np.nanmin(ess)),
        "n_groups": C,
        "group_size": n,
        "M": M * K,
    }


def _rhat_ess(samples, n_groups=8):
    r"""Route to the proper (chain, draw) Rhat/ESS when a within-chain series exists.

    ``samples`` may be ``(M, d)`` (legacy, one sample/chain) or ``(M, K, d)``
    (CHANGE B). ``K >= 2`` -> within-chain Gelman-Rubin; ``K == 1`` (or 2-D) ->
    the between-subensemble fallback ``_split_rhat_ess`` (no within-chain series).
    """
    x = np.asarray(samples, np.float64)
    if x.ndim == 2:
        x = x[:, None, :]
    if x.shape[1] >= 2:
        return _rhat_ess_chains(x)
    return _split_rhat_ess(x[:, 0, :], n_groups=n_groups)


def _steps_to_floor(D_tilde, tol=2.0):
    r"""Host estimate: first Phase-1 step where D-tilde reaches its late-time floor.

    Ground-truth-free proxy for 'when Phase 1 equilibrated': the late-time median
    of D-tilde (last 20%) is the empirical floor; return the first step at which
    D-tilde stays within ``tol`` x of that floor. Used for the SWITCH MARGIN
    (switch_index - steps_to_floor): a small/negative margin = early-switch risk.
    """
    D = np.asarray(D_tilde, np.float64)
    T = D.shape[0]
    if T < 4:
        return T
    late = np.median(D[max(0, T - max(2, T // 5)):])
    below = np.where(D <= tol * max(late, 1e-30))[0]
    return int(below[0]) if below.size else T


# --------------------------------------------------------------------------- #
# 2. diagnose                                                                  #
# --------------------------------------------------------------------------- #
def diagnose(results, param_names=None, out_png="laps_diagnose.png", verbose=True):
    r"""GROUND-TRUTH-FREE internal-health report + plots for one LAPS run.

    Checks (PASS criteria in ``laps_lensing_checklist.md``):
      * D-tilde(t) -> floor (equilibration).
      * EEVPD_obs/EEVPD_wanted -> 1 (Phase-1 controller tracking).
      * SWITCH MARGIN = switch_index - steps_to_floor(D-tilde); flag if small/<=0
        (early-switch risk).
      * Phase-2 acceptance -> target, and freeze LATCHED (sticky).
      * cross-chain split-Rhat / ESS on the final ensemble (sub-ensembles agree?).

    Returns a dict of the scalar health numbers and writes ``out_png``.
    """
    res = results
    D = np.asarray(res.p1_D_tilde, np.float64)
    eo = np.asarray(res.p1_eevpd_obs, np.float64)
    ew = np.asarray(res.p1_eevpd_wanted, np.float64)
    acc = np.asarray(res.p2_accept, np.float64)
    frozen = np.asarray(res.p2_frozen, dtype=bool)
    # CHANGE B: res.samples is (M, K, d) (K = p2_keep_per_chain). Keep the (chain,
    # draw) structure for R-hat/ESS; flatten where a (N, d) view is needed.
    samples3d = np.asarray(res.samples, np.float64)
    if samples3d.ndim == 2:
        samples3d = samples3d[:, None, :]
    M, K, d = samples3d.shape
    samples = samples3d.reshape(-1, d)               # (M*K, d) flat view
    n_total = int(getattr(res, "n_samples_total", M * K))
    a_target = float(res.target_accept)
    phase1_len = int(res.phase1_len)
    switch_idx = int(res.switch_index)

    s2f = _steps_to_floor(D)
    switch_margin = switch_idx - s2f
    D_late = float(np.median(D[max(0, len(D) - max(2, len(D) // 5)):])) if len(D) else float("nan")
    eevpd_ratio_late = (float(np.median((eo / np.where(ew > 0, ew, np.nan))[
        max(0, len(eo) - max(2, len(eo) // 5)):])) if len(eo) else float("nan"))
    p2_ran = bool(np.all(np.isfinite(acc))) and acc.size > 1
    acc_final = float(acc[-1]) if p2_ran else float("nan")
    frozen_latched = bool(frozen[-1]) if p2_ran else False
    conv = _rhat_ess(samples3d)

    # ---- flags ----
    flags = {
        "D_reached_floor": bool(np.isfinite(D_late) and (len(D) == 0 or D[-1] <= 3.0 * D_late)),
        "eevpd_tracking": bool(np.isfinite(eevpd_ratio_late) and 0.3 <= eevpd_ratio_late <= 3.0),
        "switch_fired": bool(res.switched),
        "switch_margin_ok": bool(switch_margin >= max(2, phase1_len // 20)),
        "p2_accept_on_target": bool(p2_ran and abs(acc_final - a_target) <= 0.05),
        "p2_freeze_latched": frozen_latched,
        "rhat_ok": bool(conv["rhat_max"] < 1.01),
    }
    health = {
        "M": M, "dim": d, "keep_per_chain": K, "n_samples_total": n_total,
        "phase1_len": phase1_len, "switch_index": switch_idx,
        "switched": bool(res.switched), "steps_to_floor_est": s2f,
        "switch_margin": int(switch_margin), "D_tilde_late": D_late,
        "D_tilde_final": float(D[-1]) if len(D) else float("nan"),
        "eevpd_ratio_late": eevpd_ratio_late,
        "integrator_order": int(res.integrator_order), "target_accept": a_target,
        "p2_accept_final": acc_final, "p2_freeze_latched": frozen_latched,
        "rhat_max": conv["rhat_max"], "ess_min": conv["ess_min"],
        "n_groups": conv["n_groups"], "flags": flags,
    }

    # ---- plots ----
    names = param_names or [f"z{j}" for j in range(d)]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"LAPS diagnose (ground-truth-free)  M={M} K={K} N={n_total} d={d}  "
                 f"order={res.integrator_order}", fontsize=12)
    t1 = np.arange(len(D))

    a = ax[0, 0]
    a.semilogy(t1, np.maximum(D, 1e-30), color="C0", label=r"$\tilde D(t)$")
    a.axhline(max(D_late, 1e-30), color="orange", ls="--", lw=1,
              label=f"late floor={D_late:.2e}")
    a.axvline(s2f, color="green", ls=":", lw=1.2, label=f"steps-to-floor~{s2f}")
    if res.switched:
        a.axvline(switch_idx, color="red", ls="--", lw=1.2, label=f"switch@{switch_idx}")
    a.set_title("(a) D-tilde -> floor"); a.set_xlabel("Phase-1 step"); a.legend(fontsize=8)

    a = ax[0, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        a.plot(t1, eo / ew, color="C1")
    a.axhline(1.0, color="k", ls=":", lw=1)
    a.set_yscale("log"); a.set_title("(b) EEVPD obs/wanted -> 1")
    a.set_xlabel("Phase-1 step"); a.set_ylabel("obs/wanted")

    a = ax[0, 2]
    if p2_ran:
        t2 = np.arange(len(acc))
        a.plot(t2, acc, color="C3", label="accept(t)")
        a.axhline(a_target, color="k", ls=":", lw=1, label=f"target {a_target}")
        a.axhspan(a_target - 0.03, a_target + 0.03, color="green", alpha=0.15)
        if frozen.any():
            a.axvline(int(np.argmax(frozen)), color="purple", ls="--", lw=1.2,
                      label=f"freeze@{int(np.argmax(frozen))}")
        a.set_ylim(0, 1); a.legend(fontsize=8)
    else:
        a.text(0.5, 0.5, "Phase-2 disabled", ha="center", va="center")
    a.set_title("(c) Phase-2 acceptance"); a.set_xlabel("Phase-2 step")

    a = ax[1, 0]
    rh = conv["rhat_per_dim"]
    a.bar(np.arange(d), rh, color="C4")
    a.axhline(1.01, color="red", ls="--", lw=1, label="1.01 flag")
    a.set_title(f"(d) split-Rhat ({conv['n_groups']} groups)  max={conv['rhat_max']:.3f}")
    a.set_xlabel("dim"); a.set_ylabel("Rhat"); a.legend(fontsize=8)
    a.set_xticks(np.arange(d)); a.set_xticklabels(names, rotation=90, fontsize=6)

    a = ax[1, 1]
    a.bar(np.arange(d), conv["ess_per_dim"], color="C2")
    a.axhline(n_total, color="k", ls=":", lw=1, label=f"N={n_total}")
    a.set_title(f"(e) cross-chain ESS (min={conv['ess_min']:.0f})")
    a.set_xlabel("dim"); a.legend(fontsize=8)
    a.set_xticks(np.arange(d)); a.set_xticklabels(names, rotation=90, fontsize=6)

    a = ax[1, 2]; a.axis("off")
    lines = [
        f"switched         : {res.switched}  (switch@{switch_idx}/{phase1_len})",
        f"steps-to-floor   : {s2f}",
        f"SWITCH MARGIN    : {switch_margin}   {'OK' if flags['switch_margin_ok'] else 'SMALL -> early-switch risk'}",
        f"D-tilde final    : {health['D_tilde_final']:.2e}  (late {D_late:.2e})",
        f"EEVPD ratio late : {eevpd_ratio_late:.2f}  (-> 1)",
        f"P2 accept final  : {acc_final:.3f}  (target {a_target})",
        f"P2 freeze latched: {frozen_latched}",
        f"Rhat_max         : {conv['rhat_max']:.4f}   ESS_min: {conv['ess_min']:.0f} / N={n_total} (M={M} x K={K})",
        "",
        "FLAGS (True = pass):",
    ] + [f"  {k:<20}: {v}" for k, v in flags.items()]
    a.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace",
           fontsize=9, transform=a.transAxes)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=110); plt.close(fig)
    health["png"] = out_png
    if verbose:
        print(f"[diagnose] wrote {out_png}")
        for k, v in flags.items():
            print(f"  {'PASS' if v else 'FLAG'}  {k}={v}")
    return health


# --------------------------------------------------------------------------- #
# gradient-cost proxy                                                          #
# --------------------------------------------------------------------------- #
def _grad_cost(res):
    """Approx gradient evaluations per chain: Phase-1 (1 force/step, velocity
    Verlet) + Phase-2 (N integration steps/traj x ~order force evals/step). The
    Phase-2 factor uses the integrator stage count (2 for MN2, 4 for MN4) as a
    coarse force-eval count -- a documented approximation, not an exact tally."""
    p1 = int(res.phase1_len)
    n_p2 = int(np.asarray(res.p2_accept).size) if np.all(np.isfinite(np.asarray(res.p2_accept))) else 0
    N = 15
    stages = int(res.integrator_order)
    p2 = n_p2 * N * stages
    return {"phase1_grads": p1, "phase2_grads": p2, "total_grads_per_chain": p1 + p2}


# --------------------------------------------------------------------------- #
# 3. compare_warm_cold                                                         #
# --------------------------------------------------------------------------- #
def compare_warm_cold(model_seq, qz, param_names=None, out_png="laps_warm_cold.png",
                      n_pair_params=4, verbose=True, **kw):
    r"""Run BOTH init modes and CROSS-VALIDATE: two independent inits AGREEING is
    the key ground-truth-free convergence evidence on a real posterior.

    Overlays the two posteriors (1D marginals on the diagonal + pairwise 2D for the
    first ``n_pair_params`` params), compares per-parameter mean/std and the joint
    cross-moments E[x_i x_j], and reports switch margin + gradient cost for each.

    Returns a dict with per-mode results, the per-param mean/std table, the
    cross-moment agreement, and the path to ``out_png``.
    """
    res_cold = run_laps(model_seq, qz, init_mode="cold", **kw)
    res_warm = run_laps(model_seq, qz, init_mode="warm", **kw)
    # CHANGE B: samples are (M, K, d); flatten the (chain, sample) axes to (M*K, d)
    # for the marginal/cross-moment comparisons (legacy (M, d) passes through).
    s_cold = np.asarray(res_cold.samples, np.float64)
    s_warm = np.asarray(res_warm.samples, np.float64)
    d = s_cold.shape[-1]
    s_cold = s_cold.reshape(-1, d)
    s_warm = s_warm.reshape(-1, d)
    names = param_names or [f"z{j}" for j in range(d)]

    mean_c, mean_w = s_cold.mean(0), s_warm.mean(0)
    std_c, std_w = s_cold.std(0), s_warm.std(0)
    # standardized mean disagreement: |dmean| / (pooled std / sqrt(M)) ~ z-score
    M = s_cold.shape[0]
    se = np.sqrt((std_c ** 2 + std_w ** 2) / M)
    mean_z = np.abs(mean_c - mean_w) / np.maximum(se, 1e-30)
    std_rel = np.abs(std_c - std_w) / np.maximum(0.5 * (std_c + std_w), 1e-30)
    # cross-moment agreement E[x_i x_j] as a SAMPLING z-score (robust to the
    # cross-moment crossing zero, where a relative error is ill-conditioned --
    # opposite-sign noise gives ratio 2.0; the SE-normalized z-score does not).
    cm_c = (s_cold.T @ s_cold) / M
    cm_w = (s_warm.T @ s_warm) / M
    iu = np.triu_indices(d, k=1)
    prod_c = np.einsum("ni,nj->nij", s_cold, s_cold)
    prod_w = np.einsum("ni,nj->nij", s_warm, s_warm)
    se_cross = np.sqrt((prod_c.var(0, ddof=1) + prod_w.var(0, ddof=1)) / M)
    cross_z = np.abs(cm_c - cm_w)[iu] / np.maximum(se_cross[iu], 1e-30)

    def _margin(r):
        return int(r.switch_index) - _steps_to_floor(np.asarray(r.p1_D_tilde, np.float64))
    gc_c, gc_w = _grad_cost(res_cold), _grad_cost(res_warm)

    # ---- plots: pairwise overlay for first npp params ----
    npp = int(min(n_pair_params, d))
    fig, ax = plt.subplots(npp, npp, figsize=(3 * npp, 3 * npp))
    if npp == 1:
        ax = np.array([[ax]])
    fig.suptitle("LAPS warm (orange) vs cold (blue) — agreement = convergence evidence",
                 fontsize=12)
    for r in range(npp):
        for c in range(npp):
            a = ax[r, c]
            if r == c:
                a.hist(s_cold[:, r], bins=40, density=True, alpha=0.5, color="C0", label="cold")
                a.hist(s_warm[:, r], bins=40, density=True, alpha=0.5, color="C1", label="warm")
                if r == 0:
                    a.legend(fontsize=7)
                a.set_title(names[r], fontsize=8)
            elif r > c:
                a.scatter(s_cold[:, c], s_cold[:, r], s=3, alpha=0.25, color="C0")
                a.scatter(s_warm[:, c], s_warm[:, r], s=3, alpha=0.25, color="C1")
            else:
                a.axis("off")
            if r == npp - 1:
                a.set_xlabel(names[c], fontsize=7)
            if c == 0 and r > 0:
                a.set_ylabel(names[r], fontsize=7)
            a.tick_params(labelsize=6)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=110); plt.close(fig)

    summary = {
        "mean_z_max": float(np.max(mean_z)),
        "std_rel_max": float(np.max(std_rel)),
        "cross_z_max": float(np.max(cross_z)) if cross_z.size else 0.0,
        "margin_cold": _margin(res_cold), "margin_warm": _margin(res_warm),
        "grad_cost_cold": gc_c, "grad_cost_warm": gc_w,
        "agree": bool(np.max(mean_z) < 4.0 and np.max(std_rel) < 0.15
                      and (cross_z.size == 0 or np.max(cross_z) < 4.0)),
        "png": out_png,
        "res_cold": res_cold, "res_warm": res_warm,
    }
    if verbose:
        print(f"[compare_warm_cold] wrote {out_png}")
        print(f"  mean z-score max   : {summary['mean_z_max']:.2f}  (agree if < 4)")
        print(f"  std rel-diff max   : {summary['std_rel_max']:.3f} (agree if < 0.15)")
        print(f"  cross-moment z max : {summary['cross_z_max']:.2f}  (agree if < 4)")
        print(f"  switch margin cold/warm : {summary['margin_cold']} / {summary['margin_warm']}")
        print(f"  grad/chain cold/warm    : {gc_c['total_grads_per_chain']} / "
              f"{gc_w['total_grads_per_chain']}")
        print(f"  WARM-COLD AGREE: {summary['agree']}")
    return summary
