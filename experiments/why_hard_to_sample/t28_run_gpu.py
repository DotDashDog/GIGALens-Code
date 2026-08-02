"""T28 -- GPU stage: implementation gates + 3-seed standard MCLMC under the
observable-slope prior (s ~ Uniform(0, 0.75)).

Pre-registered 2026-07-04 (docs/logs/why-hard-to-sample.md, "T28"). Runs the
registered gates FIRST and ABORTS (nonzero exit + gates json) on any hard-gate
failure, then samples 3 seeds of STANDARD MCLMC (8x2000/2000, dev 5e-4, conv f64)
from the T21 typical-set init mapped into the new s-chart, with qz' a 1e-3 ball in
the NEW z-space. Saves t21-schema npz per seed + a runmeta json.

HARD GATES (abort on failure):
  G4  s-consistency : jnp slope_s_of_Rs_jnp vs t25 numpy slope < 1e-10 on a grid.
  G1  round-trip    : max |s_jnp(leaf.forward(s)) - s| < 1e-8 on 512 pts in [0,0.75].
  G2  prior identity: sprior Rs-slot log_prob == -log(0.75) + log ds/dRs_jnp(Rs)
                      < 1e-10 on 64 prior draws; other 13 components byte-identical
                      to baseline; JointDistribution assembles as the component sum.
  G3  loglike identity: baseline vs sprior rendered log_like at IDENTICAL physical
                      params < 1e-8 (the prior swap must not touch the renderer).
  INIT identity     : theta(baseline z_init) == theta(sprior u_init) < 1e-8.
Reported DIAGNOSTIC (non-blocking): jnp chart-consistency |ds/dRs·dRs/ds - 1|.

Outputs (HARDCODED dir):
  results_carousel/phaseC/t28/t28_seed{1,2,3}.npz  (+ .manifest.json, .census.json)
  results_carousel/phaseC/t28/t28_gates_gpu.json
  results_carousel/phaseC/t28/t28_runmeta.json
  --limit N -> short chains (N/N), *_smoke outputs, seed 1 only (end-to-end wiring).

Run (inside the shifter image; see slurm/run_t28.sh):
  python t28_run_gpu.py --stage gates          # gates only
  python t28_run_gpu.py --limit 200            # gates + seed-1 smoke
  python t28_run_gpu.py                         # gates + full 3 seeds
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import (  # noqa: E402
    assert_x64, compute_diagnostics, git_commit, load_target,
    print_diagnostics, run_standard_mclmc,
)
from exp_config import STANDARD  # noqa: E402
from reparam_bijector import sha256_file  # noqa: E402
from t18_map_arm import _displaced_census, CENSUS_SIGMA  # noqa: E402
from t21_typical_init import (  # noqa: E402
    select_typical_init, xi_stats, Z_INIT_SEED, QZ_SCALE, ARMS as T21_ARMS,
)
from t25_transforms import slope_s_of_Rs  # noqa: E402  (numpy reference)
from t28_common import (  # noqa: E402
    ARTIFACT, S_HI, S_LO, THETA_E_STAR, ds_dRs_jnp, load_leaf, slope_s_of_Rs_jnp,
)

SEEDS = [1, 2, 3]
T28_OUT = os.path.join(HERE, "results_carousel", "phaseC", "t28")
BASE_SYS = os.path.join(HERE, "systems", "carousel_min_new")
SPRIOR_SYS = os.path.join(HERE, "systems", "carousel_min_sprior")
GATES_JSON = os.path.join(T28_OUT, "t28_gates_gpu.json")
RUNMETA_JSON = os.path.join(T28_OUT, "t28_runmeta.json")

# tolerances (re-derived 2026-07-04 after the first gate run ABORTED; see the
# log's T28 gate-correction entry -- the original G4/G2/G3/INIT numbers were
# mis-derived for comparisons that are NOT exact identities as implemented)
#
# G4: jnp-vs-numpy s. The FD-in-ln-r window is 2*FD_REL = 2e-4; XLA-GPU vs
#     numpy f64 transcendentals (log/arccosh) differ at ~1e-13 rel, amplified
#     by 1/window = 5e3 -> ~1e-9 expected (measured 4.1e-9). Tol = 1e-8.
G4_TOL = 1e-8
G1_TOL = 1e-8
# G2: prior identity against the IMPLEMENTED density. Measured (tfp numpy
#     substrate probe): TransformedDistribution.log_prob computes the Jacobian
#     as -fldj_spline(s_analytic(Rs)) via the bijector cache, NOT the analytic
#     ildj. The manual formula below now uses the SAME composition, so this is
#     a true plumbing identity again -> f64 roundoff on ~O(30) logp sums.
G2_TOL = 1e-10
# G3: loglike identity now routed through baseline's EXACT sigmoid inverse
#     (no spline round-trip inside the comparison) -> true identity.
G3_TOL = 1e-8
# INIT: the mapped init necessarily takes ONE spline traversal; theta error =
#     s-round-trip (<=8.6e-9, G1) * dRs/ds (~500) ~ 5e-8 -- an intrinsic chart
#     cost, NOT an identity. Requirement: << the qz init ball in physical units
#     (1e-3 in z ~ 0.02 in Rs). Tol = 1e-5 (2000x below the ball).
INIT_TOL = 1e-5
# Chart consistency |ds/dRs_analytic * dRs/ds_spline - 1| is BLOCKING now: it
# bounds the log-density gap between the implemented prior (uniform in
# spline-s) and the registered analytic statement. 1e-4 in log-density is
# orders below anything the posterior can resolve (loglike range O(1e3)).
CHART_TOL = 1e-4

# T21 new-arm baselines (context for the runmeta; NOT gates here)
T21_NEW_BASELINE = {"eps": 0.354, "min_ess": 143, "frac_xi_gt10": [0.077, 0.087]}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sprior_module():
    spec = importlib.util.spec_from_file_location(
        "_sprior_sys", os.path.join(SPRIOR_SYS, "system.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# gates
# ===========================================================================
def run_gates(sprior, baseline_pm, leaf, z_center):
    import jax
    import jax.numpy as jnp

    pm = sprior["prob_model"]
    dim = sprior["dim"]
    names = sprior["param_names"]
    rs_col = sprior["rs_col"]
    rs_key = names[rs_col]
    res = {"tols": {"G4": G4_TOL, "G1": G1_TOL, "G2": G2_TOL, "G3": G3_TOL,
                    "CHART": CHART_TOL,
                    "INIT": INIT_TOL}}
    print("\n===== T28 IMPLEMENTATION GATES =====", flush=True)

    # --- G4: jnp s(Rs) vs numpy slope --------------------------------------
    Rs_grid = np.logspace(np.log10(10.478), np.log10(614.4), 400)
    s_np = np.asarray(slope_s_of_Rs(Rs_grid, THETA_E_STAR), np.float64)
    s_jx = np.asarray(slope_s_of_Rs_jnp(jnp.asarray(Rs_grid), THETA_E_STAR), np.float64)
    g4 = float(np.max(np.abs(s_np - s_jx)))
    res["G4_s_consistency_max"] = g4
    res["G4_pass"] = bool(g4 < G4_TOL)
    print(f"[G4] jnp s(Rs) vs numpy slope max|diff| = {g4:.3e} "
          f"({'PASS' if res['G4_pass'] else 'FAIL'}; tol {G4_TOL})", flush=True)

    # --- G1 (jnp): round-trip s -> Rs -> s ---------------------------------
    s_test = np.linspace(S_LO, S_HI, 512)
    Rs_of_s = np.asarray(leaf.forward(jnp.asarray(s_test)), np.float64)   # spline (jnp)
    s_back = np.asarray(slope_s_of_Rs_jnp(jnp.asarray(Rs_of_s), THETA_E_STAR), np.float64)
    g1 = float(np.max(np.abs(s_back - s_test)))
    res["G1_jnp_roundtrip_max"] = g1
    res["G1_pass"] = bool(g1 < G1_TOL)
    print(f"[G1] jnp round-trip max|s_jnp(leaf.forward(s)) - s| = {g1:.3e} "
          f"({'PASS' if res['G1_pass'] else 'FAIL'}; tol {G1_TOL})", flush=True)

    # --- chart-consistency gate (jnp; exact jax.grad ds/dRs) ---------------
    # BLOCKING: bounds the log-density gap between the IMPLEMENTED prior
    # (uniform in spline-s; tfp cache uses the spline fldj for the density,
    # measured 2026-07-04) and the registered analytic s ~ U(0, 0.75).
    dRs_ds = np.asarray(leaf.derivative(jnp.asarray(s_test)), np.float64)
    ds_dRs = np.asarray(ds_dRs_jnp(jnp.asarray(Rs_of_s)), np.float64)
    chart = float(np.max(np.abs(ds_dRs * dRs_ds - 1.0)))
    res["chart_consistency_jnp_max"] = chart
    res["chart_consistency_is_blocking"] = True
    res["chart_pass"] = bool(chart < CHART_TOL)
    print(f"[chart] jnp max|ds/dRs·dRs/ds - 1| = {chart:.3e} "
          f"({'PASS' if res['chart_pass'] else 'FAIL'}; tol {CHART_TOL}; bounds "
          f"implemented-vs-analytic prior log-density gap)", flush=True)

    # --- G2: prior-density identity (robust; via .log_prob only) ----------
    # Draw 64 baseline-chart points z ~ N(z_center, 1) so Rs lands in (20,100),
    # where BOTH priors are finite. The registered identity: the sprior prior is
    # the baseline prior with ONLY the Rs component swapped, i.e.
    #   sprior.log_prob(theta) == baseline.log_prob(theta)
    #                             - Uniform(20,100).log_prob(Rs)          [remove old]
    #                             + (-log(0.75) + log ds/dRs_jnp(Rs))     [add new].
    # This uses only JointDistribution.log_prob (no .model internals), tests the
    # OTHER 13 components are unchanged (they cancel exactly), and pins the Rs slot
    # to the manual s-formula. Uniform(20,100).log_prob(Rs) = -log(80) on (20,100).
    prior = pm.prior
    base_prior = baseline_pm.prior
    rng2 = np.random.RandomState(777)
    Zg = z_center[None, :] + rng2.standard_normal((64, dim)) * 1.0
    theta_g = baseline_pm.bij.forward([jnp.asarray(Zg[:, j]) for j in range(dim)])
    Rs_g = np.asarray(theta_g[rs_key], np.float64)
    if not np.all((Rs_g > 20.0) & (Rs_g < 100.0)):
        raise RuntimeError("G2 setup: baseline draw gave Rs outside (20,100)")
    sprior_lp = np.asarray(prior.log_prob(theta_g), np.float64)
    base_lp = np.asarray(base_prior.log_prob(theta_g), np.float64)
    uni_lp = -np.log(100.0 - 20.0)                       # Uniform(20,100).log_prob
    # IMPLEMENTED Rs-slot density (what tfp actually computes, via the bijector
    # cache): -log(0.75) - fldj_spline(s_analytic(Rs)). Same composition as
    # TransformedDistribution.log_prob -> true plumbing identity. The gap to the
    # registered ANALYTIC density is bounded by the blocking chart gate above.
    s_g = slope_s_of_Rs_jnp(jnp.asarray(Rs_g), THETA_E_STAR)
    fldj_g = np.asarray(leaf.forward_log_det_jacobian(s_g), np.float64)
    new_rs_lp = -np.log(S_HI - S_LO) - fldj_g
    manual = base_lp - uni_lp + new_rs_lp
    g2 = float(np.max(np.abs(sprior_lp - manual)))
    res["G2_prior_identity_max"] = g2
    res["G2_pass"] = bool(g2 < G2_TOL)
    print(f"[G2] |sprior.log_prob - (baseline - U(20,100).lp + (-log0.75+log ds/dRs))| "
          f"= {g2:.3e} ({'PASS' if res['G2_pass'] else 'FAIL'}; tol {G2_TOL})", flush=True)

    # --- G3: rendered-loglike identity at identical physical params -------
    # Direction matters (re-derived after the first gate run): the original
    # comparison went baseline z -> theta -> sprior u (analytic inverse) and
    # rendered u through the SPLINE forward, so "identical theta" silently
    # included one spline round-trip (delta_Rs ~ 1e-7 -> delta_LL ~ 1e-6).
    # Now: start from sprior-chart points, take theta_s = sprior forward
    # (spline, ONCE -- these ARE the physical points being compared), and map
    # into baseline z via baseline's EXACT closed-form sigmoid inverse. Both
    # models then render theta identical to ~1e-15 rel; renderer identity is
    # the only thing left in the difference.
    rng = np.random.RandomState(20260704)
    z_center = np.asarray(z_center, np.float64)
    Zr = z_center[None, :] + rng.standard_normal((8, dim)) * 1.0   # baseline coords
    theta_b = baseline_pm.bij.forward([jnp.asarray(Zr[:, j]) for j in range(dim)])
    u_cols = pm.bij.inverse(theta_b)                              # analytic -> u
    U = np.stack([np.asarray(c, np.float64) for c in u_cols], axis=1)
    theta_s = pm.bij.forward([jnp.asarray(U[:, j]) for j in range(dim)])  # spline
    Rs_s = np.asarray(theta_s[rs_key], np.float64)
    if not np.all((Rs_s > 20.0) & (Rs_s < 100.0)):
        raise RuntimeError("G3 setup: round-tripped Rs left (20,100)")
    zb_cols = baseline_pm.bij.inverse(theta_s)                    # exact sigmoid
    Zb = np.stack([np.asarray(c, np.float64) for c in zb_cols], axis=1)
    ll_b = np.asarray(baseline_pm.log_like(jnp.asarray(Zb))[0], np.float64)
    ll_s = np.asarray(pm.log_like(jnp.asarray(U))[0], np.float64)
    g3 = float(np.max(np.abs(ll_b - ll_s)))
    res["G3_loglike_identity_max"] = g3
    res["G3_pass"] = bool(g3 < G3_TOL)
    print(f"[G3] baseline vs sprior log_like (identical theta) max|diff| = {g3:.3e} "
          f"({'PASS' if res['G3_pass'] else 'FAIL'}; tol {G3_TOL})", flush=True)

    # --- finite render check on prior draws (NaN guard; covers HIGH Rs) ---
    theta_p = pm.prior.sample(64, seed=jax.random.PRNGKey(20260704))
    u_prior = pm.bij.inverse(theta_p)
    Up = np.stack([np.asarray(c, np.float64) for c in u_prior], axis=1)
    lp_prior, _ = pm.log_prob(jnp.asarray(Up))
    Rs_hi = float(np.max(np.asarray(theta_p[rs_key], np.float64)))
    res["prior_draw_max_Rs"] = Rs_hi
    n_bad = int(np.sum(~np.isfinite(np.asarray(lp_prior, np.float64))))
    res["prior_draw_nonfinite_logp"] = n_bad
    res["render_finite_pass"] = bool(n_bad == 0)
    print(f"[render] non-finite log_prob over 64 prior draws (max Rs={Rs_hi:.1f}) = "
          f"{n_bad} ({'PASS' if res['render_finite_pass'] else 'FAIL'})", flush=True)

    res["all_pass"] = bool(res["chart_pass"] and
                           res["G4_pass"] and res["G1_pass"] and res["G2_pass"]
                           and res["G3_pass"] and res["render_finite_pass"])
    print(f"[gates] OVERALL {'PASS' if res['all_pass'] else 'FAIL'}", flush=True)
    return res


# ===========================================================================
# run
# ===========================================================================
def _map_init(baseline_pm, sprior, smod):
    """Recover the EXACT T21 typical-set z_init in baseline coords and map into the
    sprior chart. Returns (z_init, u_init, ledger, init_gate)."""
    import jax.numpy as jnp
    dim = sprior["dim"]
    ref_path = T21_ARMS["new"]["ref"]
    chains = T21_ARMS["new"]["chains"]
    z_init, ledger, sel_dim = select_typical_init(baseline_pm, ref_path, chains, Z_INIT_SEED)
    if sel_dim != dim:
        raise ValueError(f"selection dim {sel_dim} != dim {dim}")
    u_init, info = smod.map_baseline_z_to_u(z_init, baseline_pm, sprior["prob_model"], dim)

    # INIT identity gate: theta(baseline z_init) == theta(sprior u_init)
    theta_b = baseline_pm.bij.forward([jnp.asarray(z_init[j][None]) for j in range(dim)])
    theta_s = sprior["prob_model"].bij.forward(
        [jnp.asarray(u_init[j][None]) for j in range(dim)])
    keys = set(theta_b) | set(theta_s)
    m = 0.0
    for k in keys:
        m = max(m, float(np.max(np.abs(
            np.asarray(theta_b[k], np.float64) - np.asarray(theta_s[k], np.float64)))))
    init_gate = {"theta_identity_max": m, "pass": bool(m < INIT_TOL),
                 "z_Rs": info["z_Rs"], "u_Rs": info["u_Rs"],
                 "max_abs_other_coord_change": info["max_abs_other_coord_change"]}
    print(f"[init] mapped init: z_Rs {info['z_Rs']:.4f} -> u_Rs {info['u_Rs']:.4f}; "
          f"other-coord max|change| = {info['max_abs_other_coord_change']:.3g}; "
          f"theta identity = {m:.3e} ({'PASS' if init_gate['pass'] else 'FAIL'}; "
          f"tol {INIT_TOL})", flush=True)
    return z_init, u_init, ledger, init_gate


def run_seeds(sprior, u_init, ledger, limit):
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions

    cfg = STANDARD if not limit else dataclasses.replace(
        STANDARD, num_burnin_steps=int(limit), num_results=int(limit))
    nr = cfg.num_results
    suffix = "_smoke" if limit else ""
    seeds = [1] if limit else SEEDS      # smoke: seed 1 only (end-to-end wiring)
    dim = sprior["dim"]
    names = sprior["param_names"]
    pm = sprior["prob_model"]

    qz_prime = tfd.MultivariateNormalDiag(
        loc=jnp.asarray(u_init), scale_diag=jnp.full(dim, QZ_SCALE))

    per_seed = []
    for seed in seeds:
        out_npz = os.path.join(T28_OUT, f"t28_seed{seed}{suffix}.npz")
        pos = run_standard_mclmc(
            pm, qz_prime, cfg, seed, out_npz,
            target_desc="carousel_min NEW arm, T28 observable-slope prior "
                        f"(s~Uniform({S_LO},{S_HI})), typical init mapped to s-chart",
            provenance={"experiment": "T28", "artifact": sprior["artifact"],
                        "artifact_sha256": sprior["sha256"],
                        "qz": f"MVNDiag(loc=u_init, scale_diag={QZ_SCALE})",
                        "z_init_median_draw": ledger["median_draw_used_as_init"]},
        )
        diag = compute_diagnostics(pos, nr, names)
        print_diagnostics(diag, header=f"T28 seed {seed}")
        census = _displaced_census(pos, nr)
        xis = xi_stats(out_npz, num_results=nr)
        print(f"[T28] seed {seed} persistent chains={census['persistent_chains']}; "
              f"frac(xi>10)={xis['frac_gt10']:.4f} (p99={xis['p99']:.3g} "
              f"max={xis['max']:.3g})", flush=True)
        with open(os.path.splitext(out_npz)[0] + ".census.json", "w") as f:
            json.dump(census, f, indent=2)
        per_seed.append({
            "seed": seed, "npz": os.path.abspath(out_npz),
            "min_ess": diag["min_ess"], "min_ess_param": diag["min_ess_param"],
            "max_rhat": diag["max_rhat"], "max_rhat_param": diag["max_rhat_param"],
            "ess_per_param": {n: float(e) for n, e in zip(diag["names"], diag["ess"])},
            "n_persistent": census["n_persistent"],
            "persistent_chains": census["persistent_chains"], "xi": xis,
        })
    return cfg, per_seed


def main(argv=None):
    ap = argparse.ArgumentParser(description="T28 gates + s-prior MCLMC (GPU)")
    ap.add_argument("--stage", choices=["gates", "run", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke: short chains of N/N, seed 1 only; writes *_smoke")
    args = ap.parse_args(argv)

    os.makedirs(T28_OUT, exist_ok=True)
    assert_x64()
    print(f"[T28] float64 asserted; WHTS_CONV_PRECISION="
          f"{os.environ.get('WHTS_CONV_PRECISION')}; stage={args.stage} "
          f"limit={args.limit or 'none'}", flush=True)

    if not os.path.isfile(ARTIFACT):
        print(f"[T28] FATAL: transform artifact missing: {ARTIFACT}. Run "
              "t28_sprior_transform.py first.", flush=True)
        return 2

    # --- build targets -----------------------------------------------------
    smod = _sprior_module()
    sprior = smod.build_sprior_target()
    leaf = sprior["leaf"]
    baseline_pm, _qz_b, z_center_b, dim_b, names_b = load_target(BASE_SYS)
    if sprior["dim"] != dim_b or list(sprior["param_names"]) != list(names_b):
        raise ValueError("sprior dim/param_names mismatch vs baseline new arm")

    # --- gates -------------------------------------------------------------
    gates = run_gates(sprior, baseline_pm, leaf, z_center_b)
    z_init, u_init, ledger, init_gate = _map_init(baseline_pm, sprior, smod)
    gates["INIT_identity"] = init_gate
    all_gates = bool(gates["all_pass"] and init_gate["pass"])
    gates["all_gates_pass"] = all_gates

    payload = {
        "experiment": "T28 GPU gates", "status": "proposed (UNCERTIFIED)",
        "timestamp_utc": _now(), "artifact": os.path.abspath(ARTIFACT),
        "artifact_sha256": sprior["sha256"], "gates": gates,
        "git_commit": git_commit(),
    }
    with open(GATES_JSON, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"[T28] wrote {GATES_JSON}", flush=True)

    if not all_gates:
        print("[T28] *** GATE FAILURE -- ABORTING before any sampling. ***", flush=True)
        return 3
    if args.stage == "gates":
        print("[T28] gates-only stage complete (all PASS).", flush=True)
        return 0

    # --- sample ------------------------------------------------------------
    cfg, per_seed = run_seeds(sprior, u_init, ledger, args.limit)

    runmeta = {
        "experiment": "T28 observable-slope prior MCLMC",
        "status": "proposed (UNCERTIFIED)",
        "timestamp_utc": _now(),
        "prior_spec": {"observable": "s(Rs) = dln(alpha)/dln(r) at r=theta_E*",
                       "theta_E_star": THETA_E_STAR,
                       "s_prior": f"Uniform({S_LO}, {S_HI})",
                       "Rs_at_s0": 10.478, "Rs_at_s075": 614.4,
                       "other_priors": "UNCHANGED from carousel_min_new"},
        "artifact": os.path.abspath(ARTIFACT), "artifact_sha256": sprior["sha256"],
        "seeds": ([1] if args.limit else SEEDS), "smoke_limit": (args.limit or None),
        "config": cfg.to_dict(), "qz_scale": QZ_SCALE, "z_init_seed": Z_INIT_SEED,
        "rs_col": sprior["rs_col"], "param_names": list(sprior["param_names"]),
        "z_init_ledger": ledger, "u_init_Rs": float(u_init[sprior["rs_col"]]),
        "conv_precision_env": os.environ.get("WHTS_CONV_PRECISION", "float32(default)"),
        "T21_new_baseline": T21_NEW_BASELINE, "gates_all_pass": all_gates,
        "per_seed": per_seed, "git_commit": git_commit(),
    }
    mfile = os.path.join(T28_OUT, f"t28_runmeta{'_smoke' if args.limit else ''}.json")
    with open(mfile, "w") as f:
        json.dump(runmeta, f, indent=2, default=float)
    print(f"[T28] wrote {mfile} (status: proposed (UNCERTIFIED))", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
