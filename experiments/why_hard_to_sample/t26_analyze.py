"""T26 analyzer -- login-node adjudication of the acceptance battery (NO jax).

Pre-registered 2026-07-03 (docs/logs/why-hard-to-sample.md, "T25 ... + T26 ...").
Reads the T26 route outputs + the T21 new-arm baselines and computes, per route,
the registered acceptance metrics and verdicts (all "proposed (UNCERTIFIED)" --
the producer never self-certifies). numpy / scipy / matplotlib / arviz only.

Registered acceptance thresholds (from the pre-registration):
  * tuned eps (results-phase mean) >= 0.8   [F: eps < 0.5]   (baseline 0.354, clone 1.22)
  * per-param min bulk-ESS >= 500
  * frac(xi>10) <= 0.04                       (baseline ~0.08, clone ~0.02)
  * per-param ESS max/min <= 3               (uniformity; secondary)
  * clone gap (clone min-ESS / real min-ESS) <= 2x  (secondary)
  Success = eps + min-ESS + frac all met; uniformity and clone gap graded but
  secondary. Below-floor OPENING (a shifted Rs marginal) is a FINDING, not a
  failure -- reported descriptively with R-hat.

Plots (per route): per-param ESS bars (route vs T21 baseline vs clone); xi/
stability ECDF overlay; xi-vs-u_Rs hexbin (funnel-reappearance check at the new
coordinate's low end); Rs theta-space marginal baseline-vs-route overlay
(below-floor opening check). Writes t26/t26_analysis.json + pngs.
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

from common import compute_diagnostics  # noqa: E402 (arviz/numpy only; no jax)
from t23_t24_common import load_run  # noqa: E402 (numpy only)

T26_OUT = os.path.join(HERE, "results_carousel", "phaseC", "t26")
T21_NEW = os.path.join(HERE, "results_carousel", "phaseC", "t21", "new")
T25_OUT = os.path.join(HERE, "results_carousel", "phaseC", "t25")

# registered constants
NR_DEFAULT = 2000
XI_THRESH = 10.0
EPS_BAR = 0.8
EPS_FAIL = 0.5
MINESS_BAR = 500.0
FRAC_BAR = 0.04
UNIF_BAR = 3.0
CLONEGAP_BAR = 2.0
BASELINE_EPS = 0.354
CLONE_TARGET_EPS = 1.22
SEEDS = [1, 2, 3]


# ---------------------------------------------------------------------------
# metric helpers (numpy / arviz)
# ---------------------------------------------------------------------------

def _eps_results_mean(run, nr):
    ss = run["step_size"]                       # (chains, steps)
    return float(np.mean(ss[:, -nr:]))


def _frac_xi(run, nr, thresh=XI_THRESH):
    xi = run["xi"][:, -nr:].reshape(-1)
    return float(np.mean(xi > thresh))


def _rs_from_zcol(zcol, rsmap_z=None, rsmap_Rs=None):
    """Map an unconstrained Rs column -> Rs (theta). Uses the exact real-leaf map
    from profile.npz if available; else the Uniform(20,100) sigmoid closed form."""
    zcol = np.asarray(zcol, np.float64)
    if rsmap_z is not None:
        return np.interp(zcol, rsmap_z, rsmap_Rs)
    return 20.0 + 80.0 / (1.0 + np.exp(-zcol))


def _route_zcol_to_Rs(ucol, leaf, rsmap_z, rsmap_Rs):
    """Route runs live in u; map u->z (leaf.forward) then z->Rs."""
    z = leaf._forward_np(np.asarray(ucol, np.float64))
    return _rs_from_zcol(z, rsmap_z, rsmap_Rs)


# ---------------------------------------------------------------------------
# per-route analysis
# ---------------------------------------------------------------------------

def analyze_route(route, smoke=False):
    suffix = "_smoke" if smoke else ""
    rdir = os.path.join(T26_OUT, f"route{route}")
    summ_path = os.path.join(rdir, f"summary{suffix}.json")
    if not os.path.isfile(summ_path):
        return {"route": route, "present": False, "reason": f"missing {summ_path}"}
    with open(summ_path) as f:
        summary = json.load(f)
    res = summary.get("result", {})
    nr = int(res.get("config", {}).get("num_results", NR_DEFAULT))
    param_names = res.get("param_names")
    rs_col = res.get("rs_col")

    out = {"route": route, "present": True, "nr": nr,
           "artifact_sha256": res.get("artifact_sha256")}

    # --- real seeds ----------------------------------------------------------
    real = {}
    eps_list, miness_list, frac_list, unif_list = [], [], [], []
    ess_by_param = None
    for seed in SEEDS:
        p = os.path.join(rdir, f"t0_seed{seed}{suffix}.npz")
        if not os.path.isfile(p):
            continue
        run = load_run(p)
        diag = compute_diagnostics(run["position"], nr, param_names)
        ess = np.asarray(diag["ess"], float)
        eps = _eps_results_mean(run, nr)
        frac = _frac_xi(run, nr)
        unif = float(np.max(ess) / max(np.min(ess), 1e-30))
        eps_list.append(eps); miness_list.append(diag["min_ess"])
        frac_list.append(frac); unif_list.append(unif)
        if ess_by_param is None:
            ess_by_param = {n: [] for n in diag["names"]}
        for n, e in zip(diag["names"], ess):
            ess_by_param[n].append(float(e))
        real[f"seed{seed}"] = {
            "eps_results_mean": eps, "min_ess": diag["min_ess"],
            "min_ess_param": diag["min_ess_param"], "max_rhat": diag["max_rhat"],
            "frac_xi_gt10": frac, "ess_max_min": unif,
        }
    out["real_per_seed"] = real
    if not eps_list:
        out["reason"] = "no real seed runs found"
        return out

    eps_mean = float(np.mean(eps_list))
    min_min_ess = float(np.min(miness_list))
    frac_mean = float(np.mean(frac_list))
    unif_max = float(np.max(unif_list))
    ess_param_mean = {n: float(np.mean(v)) for n, v in ess_by_param.items()}

    # --- clone ---------------------------------------------------------------
    clone_min_ess = None
    clone_eps = None
    cp = os.path.join(rdir, f"t1_clone_typical_seed1{suffix}.npz")
    if os.path.isfile(cp):
        crun = load_run(cp)
        cdiag = compute_diagnostics(crun["position"], nr, param_names)
        clone_min_ess = float(cdiag["min_ess"])
        clone_eps = _eps_results_mean(crun, nr)
    clone_gap = (clone_min_ess / min_min_ess) if (clone_min_ess and min_min_ess > 0) else None

    # --- registered verdicts (proposed / UNCERTIFIED) ------------------------
    verdict = {
        "eps_results_mean": eps_mean,
        "eps_meets_0p8": bool(eps_mean >= EPS_BAR),
        "eps_FAILS_below_0p5": bool(eps_mean < EPS_FAIL),
        "min_min_ess": min_min_ess,
        "miness_meets_500": bool(min_min_ess >= MINESS_BAR),
        "frac_xi_gt10_mean": frac_mean,
        "frac_meets_0p04": bool(frac_mean <= FRAC_BAR),
        "ess_uniformity_max": unif_max,
        "uniformity_meets_3": bool(unif_max <= UNIF_BAR),
        "clone_min_ess": clone_min_ess, "clone_eps": clone_eps,
        "clone_gap": clone_gap,
        "clone_gap_meets_2x": (bool(clone_gap <= CLONEGAP_BAR)
                               if clone_gap is not None else None),
    }
    # success = eps + min-ESS + frac all met (primary); others secondary
    verdict["SUCCESS_primary"] = bool(
        verdict["eps_meets_0p8"] and verdict["miness_meets_500"]
        and verdict["frac_meets_0p04"])
    verdict["fix_FAILED_eps"] = verdict["eps_FAILS_below_0p5"]
    out["verdict"] = verdict
    out["ess_per_param_mean"] = ess_param_mean

    # --- below-floor OPENING read (descriptive) ------------------------------
    # Rs marginal + max R-hat of the Rs column across route real runs.
    rsmap_z = rsmap_Rs = leaf = None
    prof = os.path.join(T25_OUT, "profile.npz")
    if os.path.isfile(prof):
        d = np.load(prof, allow_pickle=True)
        if "rsmap_z" in d.files:
            rsmap_z = np.asarray(d["rsmap_z"], np.float64)
            rsmap_Rs = np.asarray(d["rsmap_Rs"], np.float64)
    art = os.path.join(T25_OUT, f"transform_{route}.npz")
    if os.path.isfile(art):
        from reparam_bijector import MonotoneCubicBijector
        leaf = MonotoneCubicBijector.from_npz(art)
    if rs_col is not None and leaf is not None:
        rs_vals = []
        for seed in SEEDS:
            p = os.path.join(rdir, f"t0_seed{seed}{suffix}.npz")
            if os.path.isfile(p):
                run = load_run(p)
                u = run["position"][:, -nr:, rs_col].reshape(-1)
                rs_vals.append(_route_zcol_to_Rs(u, leaf, rsmap_z, rsmap_Rs))
        if rs_vals:
            rs_all = np.concatenate(rs_vals)
            out["Rs_marginal"] = {
                "min": float(rs_all.min()), "p01": float(np.percentile(rs_all, 1)),
                "median": float(np.median(rs_all)), "max": float(rs_all.max()),
                "below_24_frac": float(np.mean(rs_all < 24.0)),
            }
    return out


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def _baseline_arrays(nr):
    """T21 new-arm baseline: pooled real seeds + clone. Returns dict or None."""
    out = {"eps": [], "xi": [], "rs_z": [], "ess": None}
    ess_acc = None
    got = False
    for seed in SEEDS:
        p = os.path.join(T21_NEW, f"t0_seed{seed}.npz")
        if not os.path.isfile(p):
            continue
        got = True
        run = load_run(p)
        out["eps"].append(_eps_results_mean(run, nr))
        out["xi"].append(run["xi"][:, -nr:].reshape(-1))
    if not got:
        return None
    out["xi"] = np.concatenate(out["xi"]) if out["xi"] else np.array([])
    # per-param ess from seed1
    p1 = os.path.join(T21_NEW, "t0_seed1.npz")
    diag = compute_diagnostics(load_run(p1)["position"], nr, None)
    out["ess"] = np.asarray(diag["ess"], float)
    out["names"] = diag["names"]
    cp = os.path.join(T21_NEW, "t1_clone_typical_seed1.npz")
    if os.path.isfile(cp):
        crun = load_run(cp)
        out["clone_xi"] = crun["xi"][:, -nr:].reshape(-1)
        out["clone_ess"] = np.asarray(
            compute_diagnostics(crun["position"], nr, None)["ess"], float)
    return out


def make_plots(route, out, smoke=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    suffix = "_smoke" if smoke else ""
    rdir = os.path.join(T26_OUT, f"route{route}")
    nr = out.get("nr", NR_DEFAULT)
    base = _baseline_arrays(nr)

    # profile rs-map + leaf for Rs marginal
    rsmap_z = rsmap_Rs = leaf = None
    prof = os.path.join(T25_OUT, "profile.npz")
    if os.path.isfile(prof):
        d = np.load(prof, allow_pickle=True)
        if "rsmap_z" in d.files:
            rsmap_z = np.asarray(d["rsmap_z"]); rsmap_Rs = np.asarray(d["rsmap_Rs"])
    art = os.path.join(T25_OUT, f"transform_{route}.npz")
    if os.path.isfile(art):
        from reparam_bijector import MonotoneCubicBijector
        leaf = MonotoneCubicBijector.from_npz(art)
    with open(os.path.join(rdir, f"summary{suffix}.json")) as f:
        res = json.load(f)["result"]
    rs_col = res.get("rs_col")

    # gather route real arrays
    route_xi, route_u, route_ess = [], [], None
    for seed in SEEDS:
        p = os.path.join(rdir, f"t0_seed{seed}{suffix}.npz")
        if not os.path.isfile(p):
            continue
        run = load_run(p)
        route_xi.append(run["xi"][:, -nr:].reshape(-1))
        if rs_col is not None:
            route_u.append(run["position"][:, -nr:, rs_col].reshape(-1))
        if route_ess is None:
            route_ess = np.asarray(
                compute_diagnostics(run["position"], nr, None)["ess"], float)
    route_xi = np.concatenate(route_xi) if route_xi else np.array([])
    route_u = np.concatenate(route_u) if route_u else np.array([])

    # clone xi
    clone_xi = np.array([])
    cp = os.path.join(rdir, f"t1_clone_typical_seed1{suffix}.npz")
    if os.path.isfile(cp):
        clone_xi = load_run(cp)["xi"][:, -nr:].reshape(-1)

    # ---- (1) per-param ESS bars ----
    fig, ax = plt.subplots(figsize=(10, 4))
    if route_ess is not None:
        x = np.arange(len(route_ess))
        ax.bar(x - 0.25, route_ess, width=0.25, label=f"route {route}")
        if base is not None:
            ax.bar(x, base["ess"], width=0.25, label="T21 new baseline")
            if "clone_ess" in base:
                ax.bar(x + 0.25, base["clone_ess"], width=0.25, label="T21 clone")
        ax.axhline(MINESS_BAR, ls="--", c="k", lw=0.8, label=f"bar {MINESS_BAR:.0f}")
        ax.set_yscale("log"); ax.set_xlabel("param column"); ax.set_ylabel("bulk-ESS")
        ax.set_title(f"T26 route {route}: per-param ESS (route vs T21 baseline/clone)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(rdir, f"ess_bars{suffix}.png"), dpi=110)
    plt.close(fig)

    # ---- (2) xi / stability ECDF overlay ----
    fig, ax = plt.subplots(figsize=(7, 4))
    for arr, lab in [(route_xi, f"route {route}"), (clone_xi, f"route {route} clone"),
                     (base["xi"] if base is not None else np.array([]), "T21 baseline")]:
        if arr.size:
            s = np.sort(arr)
            ax.plot(s, np.linspace(0, 1, s.size), label=lab, lw=1.2)
    ax.axvline(XI_THRESH, ls="--", c="k", lw=0.8, label="xi=10")
    ax.set_xscale("log"); ax.set_xlabel("stability number xi"); ax.set_ylabel("ECDF")
    ax.set_title(f"T26 route {route}: xi ECDF overlay"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(rdir, f"xi_ecdf{suffix}.png"), dpi=110)
    plt.close(fig)

    # ---- (3) xi vs u_Rs hexbin (funnel reappearance at the new coord low end) ----
    if route_u.size and route_xi.size:
        fig, ax = plt.subplots(figsize=(7, 4))
        hb = ax.hexbin(route_u, np.log10(np.clip(route_xi, 1e-8, None)),
                       gridsize=40, cmap="viridis", bins="log")
        fig.colorbar(hb, ax=ax, label="log count")
        ax.set_xlabel("u_Rs (reparam sampler coord)")
        ax.set_ylabel("log10 xi"); ax.set_title(
            f"T26 route {route}: xi vs u_Rs (funnel-reappearance check)")
        fig.tight_layout()
        fig.savefig(os.path.join(rdir, f"xi_vs_u{suffix}.png"), dpi=110)
        plt.close(fig)

    # ---- (4) Rs theta-space marginal overlay (below-floor opening) ----
    if leaf is not None and rs_col is not None and route_u.size:
        fig, ax = plt.subplots(figsize=(7, 4))
        rs_route = _route_zcol_to_Rs(route_u, leaf, rsmap_z, rsmap_Rs)
        ax.hist(rs_route, bins=60, density=True, alpha=0.6, label=f"route {route}")
        if base is not None:
            # baseline lives in z; map its Rs column z->Rs directly
            base_rs = []
            for seed in SEEDS:
                p = os.path.join(T21_NEW, f"t0_seed{seed}.npz")
                if os.path.isfile(p):
                    zc = load_run(p)["position"][:, -nr:, rs_col].reshape(-1)
                    base_rs.append(_rs_from_zcol(zc, rsmap_z, rsmap_Rs))
            if base_rs:
                ax.hist(np.concatenate(base_rs), bins=60, density=True, alpha=0.6,
                        label="T21 new baseline")
        ax.axvline(24.0, ls="--", c="r", lw=0.8, label="Rs=24 (near floor)")
        ax.set_xlabel("Rs (arcsec, theta space)"); ax.set_ylabel("density")
        ax.set_title(f"T26 route {route}: Rs marginal (below-floor opening check)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(rdir, f"Rs_marginal{suffix}.png"), dpi=110)
        plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="T26 acceptance-battery analyzer (login node; no jax)")
    ap.add_argument("--routes", default="A,B", help="comma list of routes to analyze")
    ap.add_argument("--smoke", action="store_true", help="read *_smoke outputs")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    routes = [r.strip() for r in args.routes.split(",") if r.strip()]
    os.makedirs(T26_OUT, exist_ok=True)

    analysis = {"experiment": "T26 acceptance-battery analysis",
                "status": "proposed (UNCERTIFIED)", "smoke": bool(args.smoke),
                "thresholds": {"eps_bar": EPS_BAR, "eps_fail": EPS_FAIL,
                               "miness_bar": MINESS_BAR, "frac_bar": FRAC_BAR,
                               "uniformity_bar": UNIF_BAR, "clone_gap_bar": CLONEGAP_BAR},
                "routes": {}}
    for route in routes:
        r = analyze_route(route, smoke=args.smoke)
        analysis["routes"][route] = r
        if r.get("present") and "verdict" in r:
            v = r["verdict"]
            print(f"\n===== T26 route {route} (proposed / UNCERTIFIED) =====")
            print(f"  eps(results mean) = {v['eps_results_mean']:.3f}  "
                  f"[bar {EPS_BAR}: {'MET' if v['eps_meets_0p8'] else 'not met'}"
                  f"{'; FAILS<0.5' if v['eps_FAILS_below_0p5'] else ''}]")
            print(f"  min-ESS           = {v['min_min_ess']:.4g}  "
                  f"[bar {MINESS_BAR:.0f}: {'MET' if v['miness_meets_500'] else 'not met'}]")
            print(f"  frac(xi>10)       = {v['frac_xi_gt10_mean']:.4f}  "
                  f"[bar {FRAC_BAR}: {'MET' if v['frac_meets_0p04'] else 'not met'}]")
            print(f"  ESS uniformity    = {v['ess_uniformity_max']:.2f}  "
                  f"[bar {UNIF_BAR}: {'MET' if v['uniformity_meets_3'] else 'not met'}]")
            if v["clone_gap"] is not None:
                print(f"  clone gap         = {v['clone_gap']:.2f}x  "
                      f"[bar {CLONEGAP_BAR}: "
                      f"{'MET' if v['clone_gap_meets_2x'] else 'not met'}]")
            print(f"  ==> SUCCESS (primary eps+minESS+frac): {v['SUCCESS_primary']}")
            if "Rs_marginal" in r:
                rm = r["Rs_marginal"]
                print(f"  Rs marginal: min={rm['min']:.2f} p01={rm['p01']:.2f} "
                      f"median={rm['median']:.2f} (below-24 frac={rm['below_24_frac']:.3f})"
                      f"  [opening = FINDING, not failure]")
            if not args.no_plots:
                try:
                    make_plots(route, r, smoke=args.smoke)
                    print(f"  plots -> results_carousel/phaseC/t26/route{route}/*.png")
                except Exception as e:  # noqa: BLE001
                    print(f"  [plot warning] {type(e).__name__}: {e}")
        else:
            print(f"[T26 {route}] not present: {r.get('reason')}")

    ojson = os.path.join(T26_OUT, "t26_analysis.json")
    with open(ojson, "w") as f:
        json.dump(analysis, f, indent=2, default=float)
    print(f"\n[T26] wrote {ojson} (status: proposed (UNCERTIFIED))")


if __name__ == "__main__":
    main()
