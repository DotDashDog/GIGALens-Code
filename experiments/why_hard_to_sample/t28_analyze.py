"""T28 analyzer -- login-node adjudication of the observable-slope-prior run (NO jax).

Pre-registered 2026-07-04 (docs/logs/why-hard-to-sample.md, "T28"). Reads the T28
run npz's + the existing Route-A (M200,c) pushforward (results_carousel/phaseC/t27/
t27_m200c.npz) and computes the registered diagnostics + verdicts, all "proposed
(UNCERTIFIED)". numpy / scipy / arviz / matplotlib only.

Outputs (results_carousel/phaseC/t28/):
  t28_analysis.json          -- metrics + P-T28a / P-T28b verdict fields
  t28_s_Rs_marginals.png     -- s and Rs marginals: T28 vs Route A (+ old 20/100 edges)
  t28_m200c_overlay.png      -- (M200,c) overlay: Route A vs T28

Registered verdict fields:
  P-T28a (sampler health): eps_results_mean/seed, min_min_ess, max_rhat;
    F_T28a_fired if any seed has eps<0.4 OR minESS<50 OR max R-hat>1.2.
  P-T28b (science): Rs_p95, s_p95, s_edge_pileup (s_p95>0.73), branch
    (likelihood-limited / prior-limited / ambiguous), delta_median_log10M200 and
    delta_median_c vs Route A, too_much (|dM|>0.15 dex or |dc|>0.5).
  F-T28b (comparison validity): max R-hat>1.2 => overlay is NOT a posterior comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from t23_t24_common import load_run  # noqa: E402 (numpy only)
from t25_transforms import slope_s_of_Rs  # noqa: E402 (numpy g_ mirror)
from t28_common import S_HI, S_LO, THETA_E_STAR, load_leaf  # noqa: E402 (numpy paths)
# t27 physics (login numpy/scipy) -- reuse, do NOT re-derive cosmology
from t27_pushforward import (  # noqa: E402
    forward_physics, hard_gate, kpc_per_arcsec, rho_cr_Msun_kpc3,
    sigma_cr_Msun_kpc2, Z_LENS, Z_SOURCE,
)

T28_OUT = os.path.join(HERE, "results_carousel", "phaseC", "t28")
T27_NPZ = os.path.join(HERE, "results_carousel", "phaseC", "t27", "t27_m200c.npz")
RUNMETA = os.path.join(T28_OUT, "t28_runmeta.json")

NR_DEFAULT = 2000
XI_THRESH = 10.0
SEEDS_DEFAULT = [1, 2, 3]
OLD_RS_EDGES = (20.0, 100.0)     # the old hard prior box (for reference lines)

# registered P-T28a thresholds (falsifier)
EPS_FAIL = 0.4
MINESS_FAIL = 50.0
RHAT_FAIL = 1.2
# registered P-T28b thresholds
S_EDGE = 0.73                    # s_edge_pileup if s_p95 > this
S_PRIOR_LIMITED = 0.02           # prior-limited if s_p95 within this of S_HI=0.75
RS_LIKELIHOOD_LIMITED = 200.0    # likelihood-limited branch if Rs_p95 <~ this
TOO_MUCH_DLOGM = 0.15            # |delta median log10 M200| threshold
TOO_MUCH_DC = 0.5                # |delta median c| threshold


def _u_to_s(u):
    return S_LO + (S_HI - S_LO) / (1.0 + np.exp(-np.asarray(u, np.float64)))


# ---------------------------------------------------------------------------
# arviz diagnostics: rank R-hat, bulk + tail ESS
# ---------------------------------------------------------------------------
def _diag(post, names):
    """post: (chains, draws, dim). Returns per-param rank-Rhat, bulk-ESS, tail-ESS."""
    import arviz as az
    ds = az.convert_to_dataset(np.asarray(post))
    var = list(ds.data_vars)[0]
    rhat = np.atleast_1d(np.asarray(az.rhat(ds, method="rank")[var].values))
    ess_b = np.atleast_1d(np.asarray(az.ess(ds, method="bulk")[var].values))
    ess_t = np.atleast_1d(np.asarray(az.ess(ds, method="tail")[var].values))
    return rhat, ess_b, ess_t


def _load_seed(seed, smoke):
    suffix = "_smoke" if smoke else ""
    p = os.path.join(T28_OUT, f"t28_seed{seed}{suffix}.npz")
    if not os.path.isfile(p):
        return None
    return load_run(p)


# ---------------------------------------------------------------------------
# main analysis
# ---------------------------------------------------------------------------
def analyze(smoke=False):
    if not os.path.isfile(RUNMETA):
        return {"present": False, "reason": f"missing {RUNMETA}"}
    with open(RUNMETA) as f:
        meta = json.load(f)
    names = meta["param_names"]
    rs_col = int(meta["rs_col"])
    te_col = next(i for i, n in enumerate(names) if str(n).endswith("theta_E"))
    seeds = meta.get("seeds", SEEDS_DEFAULT)
    nr = int(meta.get("config", {}).get("num_results", NR_DEFAULT))

    leaf = load_leaf()

    # --- per-seed sampler-health metrics ----------------------------------
    per_seed = []
    pooled_u_s, pooled_te = [], []
    pooled_post = []
    for seed in seeds:
        run = _load_seed(seed, smoke)
        if run is None:
            per_seed.append({"seed": seed, "present": False})
            continue
        pos = run["position"]
        post = pos[:, -nr:, :]                       # (chains, nr, dim)
        rhat, ess_b, ess_t = _diag(post, names)
        eps_mean = float(np.mean(run["step_size"][:, -nr:]))
        xi = run["xi"][:, -nr:].reshape(-1)
        frac_xi = float(np.mean(xi > XI_THRESH))
        imin_b = int(np.argmin(ess_b)); imin_t = int(np.argmin(ess_t))
        imax_r = int(np.argmax(rhat))
        per_seed.append({
            "seed": seed, "present": True,
            "eps_results_mean": eps_mean,
            "min_ess_bulk": float(ess_b[imin_b]), "min_ess_bulk_param": names[imin_b],
            "min_ess_tail": float(ess_t[imin_t]), "min_ess_tail_param": names[imin_t],
            "min_ess": float(min(ess_b[imin_b], ess_t[imin_t])),
            "max_rhat": float(rhat[imax_r]), "max_rhat_param": names[imax_r],
            "frac_xi_gt10": frac_xi,
            "xi_p99": float(np.percentile(xi, 99)), "xi_max": float(xi.max()),
        })
        pooled_u_s.append(post[..., rs_col].reshape(-1))
        pooled_te.append(post[..., te_col].reshape(-1))
        pooled_post.append(post)

    present = [s for s in per_seed if s.get("present")]
    if not present:
        return {"present": False, "reason": "no T28 seed npz found"}

    # --- pooled sampler health (R-hat across all seeds' chains) -----------
    pooled_post_arr = np.concatenate(pooled_post, axis=0)   # (C*nseed, nr, dim)
    p_rhat, p_ess_b, p_ess_t = _diag(pooled_post_arr, names)
    pooled_health = {
        "max_rhat": float(np.max(p_rhat)),
        "max_rhat_param": names[int(np.argmax(p_rhat))],
        "min_ess_bulk": float(np.min(p_ess_b)),
        "min_ess_tail": float(np.min(p_ess_t)),
        "min_ess": float(min(np.min(p_ess_b), np.min(p_ess_t))),
    }

    # --- physical marginals: T28 s, Rs ------------------------------------
    u_s = np.concatenate(pooled_u_s)
    te = np.concatenate(pooled_te)
    s_T28 = _u_to_s(u_s)
    Rs_T28 = leaf._forward_np(s_T28)

    # --- cosmology + pushforward (reuse t27; re-run its HARD GATE) ---------
    cosmo = {"Sigma_cr": sigma_cr_Msun_kpc2(Z_LENS, Z_SOURCE),
             "rho_cr_l": rho_cr_Msun_kpc3(Z_LENS),
             "kpc_arcsec_l": kpc_per_arcsec(Z_LENS)}
    gate_pass, gate_info = hard_gate(cosmo)
    if not gate_pass:
        return {"present": True, "hard_gate_failed": True, "gate": gate_info}

    A28 = forward_physics(Rs_T28, te, cosmo)
    lM_T28 = np.log10(A28["M200"]); c_T28 = A28["c"]

    # --- Route A reference (from t27) -------------------------------------
    d27 = np.load(T27_NPZ)
    Rs_A = np.asarray(d27["routeA_Rs_arcsec"], np.float64)
    lM_A = np.log10(np.asarray(d27["routeA_M200"], np.float64))
    c_A = np.asarray(d27["routeA_c"], np.float64)
    s_A = np.asarray(slope_s_of_Rs(Rs_A, THETA_E_STAR), np.float64)

    def q(a):
        return [float(v) for v in np.percentile(a, [5, 50, 95])]

    quant = {
        "T28": {"s": q(s_T28), "Rs_arcsec": q(Rs_T28), "theta_E": q(te),
                "log10M200": q(lM_T28), "c": q(c_T28), "n": int(s_T28.size)},
        "routeA": {"s": q(s_A), "Rs_arcsec": q(Rs_A),
                   "log10M200": q(lM_A), "c": q(c_A), "n": int(Rs_A.size)},
    }

    # --- P-T28a verdict ----------------------------------------------------
    eps_per_seed = {s["seed"]: s["eps_results_mean"] for s in present}
    min_min_ess = float(min(s["min_ess"] for s in present))
    max_rhat = float(max(s["max_rhat"] for s in present))
    f_eps = any(s["eps_results_mean"] < EPS_FAIL for s in present)
    f_ess = any(s["min_ess"] < MINESS_FAIL for s in present)
    f_rhat = any(s["max_rhat"] > RHAT_FAIL for s in present)
    P_T28a = {
        "eps_results_mean_per_seed": eps_per_seed,
        "min_min_ess": min_min_ess, "max_rhat": max_rhat,
        "thresholds": {"eps_fail": EPS_FAIL, "minESS_fail": MINESS_FAIL,
                       "rhat_fail": RHAT_FAIL},
        "fired_eps": bool(f_eps), "fired_minESS": bool(f_ess), "fired_rhat": bool(f_rhat),
        "F_T28a_fired": bool(f_eps or f_ess or f_rhat),
    }

    # --- P-T28b verdict ----------------------------------------------------
    Rs_p95 = quant["T28"]["Rs_arcsec"][2]
    s_p95 = quant["T28"]["s"][2]
    s_edge_pileup = bool(s_p95 > S_EDGE)
    d_logM = quant["T28"]["log10M200"][1] - quant["routeA"]["log10M200"][1]
    d_c = quant["T28"]["c"][1] - quant["routeA"]["c"][1]
    too_much = bool(abs(d_logM) > TOO_MUCH_DLOGM or abs(d_c) > TOO_MUCH_DC)
    prior_limited = bool(s_p95 > (S_HI - S_PRIOR_LIMITED))
    likelihood_limited = bool((Rs_p95 <= RS_LIKELIHOOD_LIMITED)
                              and (abs(d_logM) < TOO_MUCH_DLOGM) and (abs(d_c) < TOO_MUCH_DC))
    if prior_limited:
        branch = "prior-limited"
    elif likelihood_limited:
        branch = "likelihood-limited"
    else:
        branch = "ambiguous"
    P_T28b = {
        "Rs_p95": Rs_p95, "s_p95": s_p95, "s_edge_pileup": s_edge_pileup,
        "branch": branch,
        "delta_median_log10M200_vs_routeA": float(d_logM),
        "delta_median_c_vs_routeA": float(d_c),
        "too_much": too_much,
        "thresholds": {"s_edge": S_EDGE, "s_prior_limited_margin": S_PRIOR_LIMITED,
                       "Rs_likelihood_limited": RS_LIKELIHOOD_LIMITED,
                       "too_much_dlogM": TOO_MUCH_DLOGM, "too_much_dc": TOO_MUCH_DC},
    }
    F_T28b_fired = bool(pooled_health["max_rhat"] > RHAT_FAIL)

    # --- plots -------------------------------------------------------------
    marg_png = os.path.join(T28_OUT, f"t28_s_Rs_marginals{'_smoke' if smoke else ''}.png")
    overlay_png = os.path.join(T28_OUT, f"t28_m200c_overlay{'_smoke' if smoke else ''}.png")
    _plot_marginals(s_T28, Rs_T28, s_A, Rs_A, marg_png)
    _plot_overlay(lM_A, c_A, lM_T28, c_T28, quant, overlay_png)

    result = {
        "experiment": "T28 observable-slope prior -- analysis",
        "status": "proposed (UNCERTIFIED)",
        "present": True, "smoke": smoke, "seeds": [s["seed"] for s in present],
        "num_results_used": nr,
        "per_seed": per_seed, "pooled_health": pooled_health,
        "hard_gate": gate_info, "hard_gate_passed": gate_pass,
        "quantiles": quant,
        "P_T28a": P_T28a, "P_T28b": P_T28b, "F_T28b_fired": F_T28b_fired,
        "plots": {"marginals": marg_png, "overlay": overlay_png},
        "inputs": {"runmeta": RUNMETA, "routeA_t27": T27_NPZ},
        "prior_spec": meta.get("prior_spec"),
        "old_Rs_edges": list(OLD_RS_EDGES),
    }
    return result


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------
def _plot_marginals(s_T28, Rs_T28, s_A, Rs_A, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cT = "#2ca02c"; cA = "#1f77b4"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    # s marginal
    sb = np.linspace(min(s_T28.min(), s_A.min()) - 0.02,
                     max(s_T28.max(), s_A.max()) + 0.02, 90)
    ax[0].hist(s_T28, bins=sb, density=True, color=cT, alpha=0.55, label="T28 (s-prior)")
    ax[0].hist(s_A, bins=sb, density=True, histtype="step", color=cA, lw=1.6,
               label="Route A (U(20,100))")
    ax[0].axvline(S_LO, color="0.4", ls=":", lw=1)
    ax[0].axvline(S_HI, color="k", ls="--", lw=1.2, label="s prior edge (0.75)")
    for Rs_edge in OLD_RS_EDGES:
        se = float(slope_s_of_Rs(np.array([Rs_edge]), THETA_E_STAR)[0])
        ax[0].axvline(se, color="0.6", ls="-.", lw=1)
    ax[0].set_xlabel("s = dln(alpha)/dln(r) at theta_E*"); ax[0].set_ylabel("density")
    ax[0].set_title("s marginal"); ax[0].legend(fontsize=8, frameon=False)
    # Rs marginal (log x)
    rb = np.logspace(np.log10(min(Rs_T28.min(), Rs_A.min()) * 0.95),
                     np.log10(max(Rs_T28.max(), Rs_A.max()) * 1.05), 90)
    ax[1].hist(Rs_T28, bins=rb, density=True, color=cT, alpha=0.55, label="T28")
    ax[1].hist(Rs_A, bins=rb, density=True, histtype="step", color=cA, lw=1.6,
               label="Route A")
    for Rs_edge in OLD_RS_EDGES:
        ax[1].axvline(Rs_edge, color="0.6", ls="-.", lw=1)
    ax[1].axvline(614.4, color="k", ls="--", lw=1.0, label="Rs @ s=0.75 (614)")
    ax[1].set_xscale("log"); ax[1].set_xlabel("Rs [arcsec]")
    ax[1].set_title("Rs marginal (old hard edges 20/100 dash-dot)")
    ax[1].legend(fontsize=8, frameon=False)
    fig.suptitle("T28 s / Rs marginals -- T28 s-prior vs Route A  "
                 "[proposed / UNCERTIFIED]", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_png}", flush=True)


def _plot_overlay(lM_A, c_A, lM_T, c_T, quant, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    cA = "#1f77b4"; cT = "#2ca02c"
    x_lo = min(lM_A.min(), lM_T.min()); x_hi = max(lM_A.max(), lM_T.max())
    y_lo = min(c_A.min(), c_T.min()); y_hi = max(c_A.max(), c_T.max())
    xp = 0.03 * (x_hi - x_lo + 1e-9); yp = 0.03 * (y_hi - y_lo + 1e-9)
    x_lo -= xp; x_hi += xp; y_lo -= yp; y_hi += yp

    fig = plt.figure(figsize=(8.2, 8.2))
    gs = GridSpec(2, 2, width_ratios=[4, 1.3], height_ratios=[1.3, 4],
                  hspace=0.05, wspace=0.05)
    ax2d = fig.add_subplot(gs[1, 0])
    axtop = fig.add_subplot(gs[0, 0], sharex=ax2d)
    axright = fig.add_subplot(gs[1, 1], sharey=ax2d)

    # razor-thin near-1D degeneracy: scatter both + a thin connecting line
    order_A = np.argsort(lM_A); order_T = np.argsort(lM_T)
    ax2d.plot(lM_A[order_A], c_A[order_A], color=cA, lw=0.6, alpha=0.5)
    ax2d.plot(lM_T[order_T], c_T[order_T], color=cT, lw=0.6, alpha=0.5)
    ax2d.scatter(lM_A, c_A, s=3, c=cA, alpha=0.05, edgecolors="none",
                 rasterized=True, label="Route A (U(20,100))")
    ax2d.scatter(lM_T, c_T, s=3, c=cT, alpha=0.05, edgecolors="none",
                 rasterized=True, label="T28 (s-prior)")
    ax2d.set_xlim(x_lo, x_hi); ax2d.set_ylim(y_lo, y_hi)
    ax2d.set_xlabel(r"$\log_{10}\, M_{200}\ [\mathrm{M_\odot}]$")
    ax2d.set_ylabel(r"concentration $c_{200}$")

    xb = np.linspace(x_lo, x_hi, 90); yb = np.linspace(y_lo, y_hi, 90)
    axtop.hist(lM_A, bins=xb, density=True, histtype="step", color=cA, lw=1.6)
    axtop.hist(lM_T, bins=xb, density=True, color=cT, alpha=0.5)
    axright.hist(c_A, bins=yb, density=True, histtype="step", color=cA, lw=1.6,
                 orientation="horizontal")
    axright.hist(c_T, bins=yb, density=True, color=cT, alpha=0.5,
                 orientation="horizontal")
    axtop.tick_params(labelbottom=False); axright.tick_params(labelleft=False)
    axtop.set_yticks([]); axright.set_xticks([])

    qA = quant["routeA"]; qT = quant["T28"]
    txt = ("5/50/95%  (Route A | T28 s-prior)\n"
           r"$\log_{10}M_{200}$: "
           f"{qA['log10M200'][0]:.2f}/{qA['log10M200'][1]:.2f}/{qA['log10M200'][2]:.2f} | "
           f"{qT['log10M200'][0]:.2f}/{qT['log10M200'][1]:.2f}/{qT['log10M200'][2]:.2f}\n"
           r"$c_{200}$: "
           f"{qA['c'][0]:.1f}/{qA['c'][1]:.1f}/{qA['c'][2]:.1f} | "
           f"{qT['c'][0]:.1f}/{qT['c'][1]:.1f}/{qT['c'][2]:.1f}")
    fig.text(0.62, 0.80, txt, fontsize=8.5, va="top", ha="left",
             bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.95))
    ax2d.legend(loc="upper right", fontsize=8, frameon=False, markerscale=3)
    fig.suptitle("T28  (M200, c) overlay -- Route A (original U(20,100) prior) vs "
                 "T28 (s~U(0,0.75))\n[proposed / UNCERTIFIED]", fontsize=11)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] wrote {out_png}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="T28 login analysis")
    ap.add_argument("--smoke", action="store_true", help="analyze *_smoke outputs")
    args = ap.parse_args(argv)

    os.makedirs(T28_OUT, exist_ok=True)
    print("=" * 72, flush=True)
    print("T28 analysis (login node; numpy/scipy/arviz; UNCERTIFIED)", flush=True)
    print("=" * 72, flush=True)
    result = analyze(smoke=args.smoke)
    out = os.path.join(T28_OUT, f"t28_analysis{'_smoke' if args.smoke else ''}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"[t28] wrote {out}", flush=True)

    if result.get("present"):
        pa = result["P_T28a"]; pb = result["P_T28b"]
        print("\n--- P-T28a (sampler health) ---", flush=True)
        print(f"  eps/seed = {pa['eps_results_mean_per_seed']}", flush=True)
        print(f"  min-min ESS = {pa['min_min_ess']:.1f}  max R-hat = {pa['max_rhat']:.4f}",
              flush=True)
        print(f"  F_T28a_fired = {pa['F_T28a_fired']} "
              f"(eps<{EPS_FAIL}:{pa['fired_eps']} minESS<{MINESS_FAIL}:{pa['fired_minESS']} "
              f"rhat>{RHAT_FAIL}:{pa['fired_rhat']})", flush=True)
        print("--- P-T28b (science readout) ---", flush=True)
        print(f"  Rs_p95 = {pb['Rs_p95']:.1f}  s_p95 = {pb['s_p95']:.4f}  "
              f"edge_pileup = {pb['s_edge_pileup']}", flush=True)
        print(f"  branch = {pb['branch']}", flush=True)
        print(f"  delta median log10M200 = {pb['delta_median_log10M200_vs_routeA']:+.3f} dex; "
              f"delta median c = {pb['delta_median_c_vs_routeA']:+.3f}; "
              f"too_much = {pb['too_much']}", flush=True)
        print(f"  F_T28b_fired (R-hat>1.2) = {result['F_T28b_fired']}", flush=True)
    else:
        print(f"[t28] not present: {result.get('reason')}", flush=True)
    print("\n[t28] analysis DONE (proposed / UNCERTIFIED)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
