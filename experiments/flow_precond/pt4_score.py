#!/usr/bin/env python
"""GATE PT-4 scorer: pinned formulas from the rd-2-certified checkpoint.

G1/G2/G3 (MAP-entry, seeds 50/51/52) + G4 (SVI-entry, seed 53), all
within-estimator arms, windows 100/250/500 + estimator == "within" (asserted,
A7), scoring window = post-freeze rounds 1000-1500 (PT-2 adapt lineage,
lo = max(500, rounds-500)). New this gate:
  * W-M mechanism clause: per window boundary the pooled estimator is
    reconstructed from the recorded W/B pair by the EXACT law-of-total-variance
    identity Sigma_pool = ((N-1)*T*W + N*(T-1)*B) / (T*N - 1); cold-rung max
    gen-eig of W-only vs reconstructed-pooled is measured against the FIXED
    pooled MAMS reference (np.cov of ALL draws — the harness's geneig
    diagnostic ref; the PT-2 D2 anchor 48.6 -> 126.4 -> 24.9 was measured
    against it, so comparability requires the same ref). F-M (W-only max > 10
    in ANY window) is evaluated on ALL FOUR arms including G4.
  * POOLED occupancy clause W-o (primary — replaces per-arm occ): 48
    per-system means across G1-G3, realized pooled se + between-arm component
    reported (rd-1 A4), A6-pinned near-edge corroboration (LOW-side needs a
    RISING pooled trace; HIGH-side needs flat-or-falling toward band).
  * rd-2 W-p assembly: candidate arms = G1-G3 PINNED (condition 1, G4 caveat
    string mandatory on MAP-scoped proposals); F-M/W-p mutual exclusion
    (condition 2); W-L is adjudicated by pt4_recipe_validate.py, NOT here,
    with L1b failure routed as port-bug class (condition 3).
F-S convention PINNED from pt3_score.py lines 90-101: offending low-side
eigendirections mapped to z-space x = L^-T u (B-Cholesky of Sigma_ref) before
the |cos| test vs the stored pt3_fs_reference vector; F-S SYSTEMATIC fires iff
>= 2 of the 3 MAP arms show an aligned (|cos| >= 0.8) low-side exit — folds
into the low-side zone reading, recorded NOT fatal. Login-node numpy only;
constants transcribed, never re-derived.

--selftest: verifies the reconstruction identity on synthetic gaussian data
(T=50 rounds x N=8 chains x dim=5) direct-vs-identity to 1e-12, then exits.
"""
import json
import sys

import numpy as np


def _selftest():
    rng = np.random.default_rng(20260713)
    T, N, dim = 50, 8, 5
    x = rng.standard_normal((T, N, dim))
    W = np.mean([np.cov(x[t].T, ddof=1) for t in range(T)], axis=0)
    B = np.cov(x.mean(axis=1).T, ddof=1)
    pool_direct = np.cov(x.reshape(-1, dim).T, ddof=1)
    pool_rec = ((N - 1) * T * W + N * (T - 1) * B) / (T * N - 1)
    err = float(np.max(np.abs(pool_rec - pool_direct)))
    assert err <= 1e-12, f"reconstruction identity FAILED: max|diff| = {err:.3e}"
    print(f"[selftest] pooled-cov reconstruction identity OK: "
          f"max|diff| = {err:.3e} <= 1e-12 (T=50, N=8, dim=5)")


if "--selftest" in sys.argv:
    _selftest()
    sys.exit(0)


def split_rhat_binary(ch):
    x = ch.astype(float); n2 = x.shape[1] // 2
    halves = np.concatenate([x[:, :n2], x[:, n2:2 * n2]], axis=0)
    W = halves.var(axis=1, ddof=1).mean()
    B = n2 * halves.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n2 - 1) / n2 * W + B / n2) / W)) if W > 0 else float("inf")


OUT = "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/experiments/flow_precond/carousel_gate_pt0_out"
MAMS = "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/experiments/sim_carousel/messy_tests/dpie/mams/arrays.npz"
BAND = (0.32, 0.49)
EEVPD_BAND = (1e-4, 2e-3)
GENEIG_BAND = (1.0 / 3.0, 3.0)
SOFT_HI = 10.0                 # soft-high zone (3, 10], max 4 axes; > 10 = F-R'
LOW_FLOOR = 0.1                # low-side zone [0.1, 1/3), any count; HEURISTIC
                               # floor (A5, carousel-derived, MUST NOT migrate
                               # into recipe v1); < 0.1 = F-U
DRIFT_ENV = 0.05
W_CENTER = 0.42
POCKET_COL, POCKET_THR = 6, -22.35
WINDOWS = [100, 250, 500]
T_WIN = [100, 150, 250]        # window LENGTHS = diffs of WINDOWS (pinned)
FLOORS = dict(G1=175, G2=175, G3=175, G4=175)   # op-7 scaled by rounds/1500
FS_COS = 0.8
NOMINAL = 1500
MAP_ARMS = ("G1", "G2", "G3")  # rd-2 condition 1: candidate arms PINNED
ALL_ARMS = ("G1", "G2", "G3", "G4")
TAGS = dict(G1="D2_G1pt4", G2="D2_G2pt4", G3="D2_G3pt4", G4="D1_G4pt4")

# per-basin pools (POSITIONS ONLY — standing directive) + FIXED pooled ref
draws = np.load(MAMS)["samples_z"].reshape(-1, 33).astype(np.float64)
pk = draws[:, POCKET_COL] > POCKET_THR
cov_P, cov_M = np.cov(draws[pk].T), np.cov(draws[~pk].T)
dmu = draws[pk].mean(0) - draws[~pk].mean(0)
cov_pool = np.cov(draws.T)     # harness geneig diagnostic ref (W-M clause)
L_POOL = np.linalg.cholesky(cov_pool)
fs_ref = np.load(f"{OUT}/pt3_fs_reference.npz")["eigvec_z"]


def sigma_ref(w):
    return w * cov_P + (1 - w) * cov_M + w * (1 - w) * np.outer(dmu, dmu)


def geneig(adapted, L):
    X = np.linalg.solve(L, np.linalg.solve(L, adapted).T)
    return np.sort(np.linalg.eigvalsh((X + X.T) / 2.0))


def occ_a6(m, recent, prev):
    """A6-pinned occupancy reading. recent = mean over post-lo rounds
    (1000-1500 nominal), prev = mean over the 500 rounds before lo (500-1000
    nominal). LOW-side near-edge is corroborated by a RISING trace only;
    HIGH-side near-edge requires flat-or-falling toward band (recent <= prev).
    Returns (ok, description)."""
    in_b = BAND[0] <= m <= BAND[1]
    near_lo = abs(m - BAND[0]) < DRIFT_ENV
    near_hi = abs(m - BAND[1]) < DRIFT_ENV
    rising = recent > prev
    if in_b and not (near_lo or near_hi):
        return True, "in band"
    if near_lo:
        return rising, (f"near LOW edge +-{DRIFT_ENV} (trace {prev:.3f}->{recent:.3f} "
                        f"{'RISING: corroborated' if rising else 'not rising: NOT corroborated'})")
    if near_hi:
        return (not rising), (f"near HIGH edge +-{DRIFT_ENV} (trace {prev:.3f}->{recent:.3f} "
                              f"{'flat-or-falling toward band: corroborated' if not rising else 'RISING: NOT corroborated (A6: rising corroborates LOW side only)'})")
    return False, "OUT of band beyond near-edge"


res = {}
for name in ALL_ARMS:
    tag = TAGS[name]
    try:
        d = np.load(f"{OUT}/arrays_{tag}.npz")
    except FileNotFoundError:
        print(f"\n[{name}] MISSING — skipped"); continue
    rounds = int(d["rounds_done"])
    assert float(d["betas"][-1]) == 1.0, "cold rung is not beta=1"
    assert list(d["metric_windows"]) == WINDOWS, "non-pinned windows!"
    assert str(d["metric_estimator"]) == "within", "estimator is not 'within' (A7)!"
    assert d["cold_ind"].shape[0] >= rounds, "truncated buffer (shape guard)"
    if rounds < NOMINAL:
        print(f"*** {name} INCOMPLETE {rounds}/{NOMINAL} — op-7 scaling ***")
    cold = d["cold_ind"][:rounds]
    lo = max(500, rounds - 500)
    if rounds <= lo:
        print(f"\n[{name}] no post-freeze rounds — skipped"); continue
    win = cold[lo:]
    nsys = cold.shape[1]
    sysm = win.mean(axis=0)
    m, sd = float(sysm.mean()), float(sysm.std(ddof=1))
    se = sd / np.sqrt(nsys)
    occ_recent = float(win.mean())
    occ_prev = float(cold[max(0, lo - 500):lo].mean())
    fl = FLOORS[name] * min(1.0, rounds / float(NOMINAL))
    rtp = int(d["round_trips_pocket"].sum())
    ev = d["eevpd"][:rounds]
    ev_med = np.median(ev[lo:], axis=0)          # post-lo only (pt3 A2 lineage)
    ev_ok = all(EEVPD_BAND[0] <= v <= EEVPD_BAND[1] for v in ev_med)
    tail_rung = (ev[lo:] > 2e-3).mean(axis=0)    # report
    att, acc = d["swap_attempts"].astype(float), d["swap_accepts"].astype(float)
    pacc = acc.sum(1) / np.maximum(att.sum(1), 1)
    pacc_ok = bool(np.all((pacc >= 0.25) & (pacc <= 0.65)))
    rhat = split_rhat_binary(cold[lo:].T)        # report-only diagnostic
    in_band = BAND[0] <= m <= BAND[1]
    near = min(abs(m - BAND[0]), abs(m - BAND[1])) < DRIFT_ENV
    # scored gen-eig vs Sigma_ref(w-hat)
    ref = sigma_ref(m)
    L = np.linalg.cholesky(ref)
    A = d["metric_frozen"][-1]                   # cold rung
    X = np.linalg.solve(L, np.linalg.solve(L, A).T); X = (X + X.T) / 2
    g, V = np.linalg.eigh(X)
    order = np.argsort(g); g, V = g[order], V[:, order]
    n_out = int(((g < GENEIG_BAND[0]) | (g > GENEIG_BAND[1])).sum())
    n_soft_hi = int(((g > GENEIG_BAND[1]) & (g <= SOFT_HI)).sum())
    n_soft_lo = int(((g >= LOW_FLOOR) & (g < GENEIG_BAND[0])).sum())
    n_hard_hi = int((g > SOFT_HI).sum())         # F-R'
    n_hard_lo = int((g < LOW_FLOOR).sum())       # F-U
    # F-S: low-side exits mapped to z-space x = L^-T u (PINNED pt3 convention)
    fs_aligned = False
    for i in np.where(g < GENEIG_BAND[0])[0]:
        x = np.linalg.solve(L.T, V[:, i]); x /= np.linalg.norm(x)
        cosr = abs(float(x @ fs_ref))
        is_map = name in MAP_ARMS
        print(f"  [F-S check {name}] low-side axis g={g[i]:.3f}: |cos(stored)| = {cosr:.3f}"
              + ((" -> aligned (counts toward F-S SYSTEMATIC 2-of-3 MAP tally)"
                  if is_map else " (matches stored dir; SVI arm — report only)")
                 if cosr >= FS_COS else " (distinct direction)"))
        if cosr >= FS_COS and is_map:
            fs_aligned = True
    # ---- W-M mechanism clause: W-only vs reconstructed-pooled, cold rung,
    #      vs the FIXED pooled MAMS diagnostic ref (PT-2 D2 anchor basis)
    Wc, Bc = d["metric_within_covs"], d["metric_between_covs"]
    have_wb = Wc.ndim == 4 and Wc.shape[0] > 0
    w_max, pool_max = [], []
    if not have_wb:
        # audit A3-1: a scored arm (rounds > 500) MUST carry all boundaries;
        # empty W/B means harness/schema drift — fail closed, never default
        raise RuntimeError(f"[{name}] metric_within_covs EMPTY on a scored arm "
                           f"(rounds={rounds}) — F-M unevaluable; refusing to "
                           f"score (fail-closed, audit A3-1)")
    else:
        n_b = Wc.shape[0]
        if n_b < len(WINDOWS):
            print(f"  [W-M {name}] PARTIAL: {n_b}/{len(WINDOWS)} boundaries "
                  f"recorded — evaluated on recorded windows only")
        for b in range(n_b):
            Tb, N = T_WIN[b], nsys
            Wcold, Bcold = Wc[b, -1], Bc[b, -1]
            pool_rec = ((N - 1) * Tb * Wcold + N * (Tb - 1) * Bcold) / (Tb * N - 1)
            gw, gp = geneig(Wcold, L_POOL), geneig(pool_rec, L_POOL)
            w_max.append(float(gw[-1])); pool_max.append(float(gp[-1]))
            print(f"  [W-M {name} b{b} T={Tb}] cold-rung max gen-eig vs FIXED pooled "
                  f"ref: W-only {gw[-1]:.1f}, reconstructed-pooled {gp[-1]:.1f}")
            if b == 1:   # boundary 2: identity-attributed per-axis shares (M prediction)
                infl = np.diag(pool_rec) / np.diag(cov_pool)
                wsh = (N - 1) * Tb * np.diag(Wcold)
                bsh = N * (Tb - 1) * np.diag(Bcold)
                top4 = np.argsort(infl)[::-1][:4]
                print(f"  [W-M {name} b1 attribution] top-4 pooled-inflated axes "
                      f"(prediction: B-share >= 10x W-share on the {{19,2,3,20}} family — report):")
                for j in top4:
                    ratio = float(bsh[j] / wsh[j]) if wsh[j] > 0 else float("inf")
                    print(f"    z[{int(j):2d}]: pooled-infl {infl[j]:8.1f}x  "
                          f"B/W share = {ratio:6.1f}"
                          + ("  (>= 10x, matches M prediction)" if ratio >= 10 else ""))
        wm_pass = (any(p > 10 for p in pool_max) and all(w <= 10 for w in w_max)
                   and n_b == len(WINDOWS))
        fm_fired = any(w > 10 for w in w_max)
    clauses = dict(rt=rtp >= fl, occ=in_band, eevpd=ev_ok, pair_acc=pacc_ok,
                   nan=int(d["n_revert"]) == 0, geneig=n_out == 0,
                   geneig_routed=(n_hard_hi == 0 and n_hard_lo == 0
                                  and n_soft_hi <= 4),
                   wm=wm_pass, fm=fm_fired)
    arm = dict(rounds=rounds, nsys=nsys, m=m, sd=sd, se=se,
               per_system_means=[float(v) for v in sysm],
               occ_recent=occ_recent, occ_prev=occ_prev,
               rt_pocket=rtp, floor_eff=fl, ev_med=ev_med.tolist(),
               tail_rung=tail_rung.tolist(), pair_acc=pacc.tolist(),
               split_rhat=rhat, n_revert=int(d["n_revert"]),
               geneig_full=[float(v) for v in g], n_out=n_out,
               n_soft_hi=n_soft_hi, n_soft_lo=n_soft_lo,
               n_hard_hi=n_hard_hi, n_hard_lo=n_hard_lo,
               fs_aligned_low=fs_aligned, wm_w_max=w_max,
               wm_pool_max=pool_max, wm_pass=wm_pass, fm_fired=fm_fired)
    print(f"\n[{name}] rounds={rounds} m={m:.4f}±{se:.4f} RTpocket={rtp} (floor {fl:.0f}) "
          f"geneig [{g[0]:.3f}, {g[-1]:.3f}] out={n_out} soft-hi={n_soft_hi} "
          f"low-side={n_soft_lo} split-Rhat={rhat:.4f} NaN={arm['n_revert']}")
    print(f"  eevpd_med={['%.1e' % v for v in ev_med]}; tail>2e-3 per rung "
          f"{[float(round(v, 3)) for v in tail_rung]} (report); pair acc "
          f"{np.round(pacc, 3).tolist()}")
    if near:
        print(f"  *** per-arm NEAR-EDGE (±{DRIFT_ENV}) — per-arm occ is REPORT-ONLY "
              f"this gate; pooled W-o is primary ***")
    if not in_band:
        L42 = np.linalg.cholesky(sigma_ref(W_CENTER))
        g42 = geneig(A, L42)
        arm["geneig_vs_042"] = [float(v) for v in g42]
        print(f"  blind-spot ix: gen-eig vs Sigma_ref(0.42) [{g42[0]:.3f}, {g42[-1]:.3f}]")
    if fm_fired:
        bad = [b for b, w in enumerate(w_max) if w > 10]
        print(f"  *** F-M: W-only cold-rung max gen-eig > 10 in window(s) {bad} — "
              f"within-round spread is itself transit-inflated; drift hypothesis "
              f"WRONG, drop-B family cannot fix adaptation -> report-to-human "
              f"(options: explicit shrink-to-prior bounding; or ACCEPT PT-2 "
              f"freeze-500 (3,10] inflation as product) ***")
    if n_hard_hi > 0:
        print(f"  *** F-R': {n_hard_hi} scored post-freeze axis(es) > 10 — "
              + ("WITH F-M: same report-to-human routing ***" if fm_fired else
                 "WITHOUT F-M (W clean): blend/regularize path implicated — "
                 "diagnostic finding, report ***"))
    if n_hard_lo > 0:
        print(f"  *** F-U: {n_hard_lo} scored axis(es) < {LOW_FLOOR} — estimator "
              f"pathology beyond finite-window under-read ***")
    if 0 < n_soft_hi <= 4 and n_hard_hi == 0:
        print(f"  ZONE (3,10] on {n_soft_hi} <= 4 axes: IMPROVED-PARTIAL routing "
              f"IF W-t/W-o(pooled)/W-h pass (B3 caveats attach)")
    if n_soft_lo > 0 and n_hard_lo == 0:
        print(f"  ZONE [0.1,1/3) on {n_soft_lo} axis(es): certifiable-with-caveat "
              f"IF W-t/W-o(pooled)/W-h pass (budget-limited ridge equilibration, "
              f"the predicted signature; 0.1 floor is HEURISTIC (A5) — carousel-"
              f"derived, MUST NOT migrate into recipe v1)")
    if not clauses["rt"]:
        rising = occ_recent > occ_prev
        print(f"  *** F-C component: RT {rtp} < floor {fl:.0f} — "
              + ("rising coldocc trace: A1-style budget-limited zone (routed to "
                 "extend-ROUNDS decision) ***" if rising else "trace not rising: report ***"))
    if 1.05 < rhat <= 1.2 and clauses["rt"]:
        print(f"  split-Rhat {rhat:.3f} in (1.05,1.2] with transport: budget-limited "
              f"reading (report-only diagnostic this gate)")
    print(f"  clauses: {clauses}")
    arm["clauses"] = {k: bool(v) for k, v in clauses.items()}
    res[name] = arm

# ---- F-S SYSTEMATIC tally (2 of 3 MAP arms; folds into low-side zone reading)
fs_count = sum(1 for n in MAP_ARMS if n in res and res[n]["fs_aligned_low"])
fs_systematic = fs_count >= 2
if fs_systematic:
    print(f"\n*** F-S SYSTEMATIC: aligned (|cos| >= {FS_COS}) low-side exit on "
          f"{fs_count}/3 MAP arms — the {{10,4,11,1}} under-inflation direction is "
          f"SYSTEMATIC; expected under the W mechanism, folds into the [0.1,1/3) "
          f"low-side zone reading (RECORDED, NOT FATAL) ***")
elif fs_count == 1:
    print(f"\nF-S: aligned low-side exit on 1/3 MAP arms — below the 2-of-3 "
          f"systematic threshold (report)")

# ---- POOLED occupancy clause W-o (primary; rd-1 A4 + A6)
pooled = {}
wo = None
if all(n in res for n in MAP_ARMS):
    pool_means = np.concatenate([np.asarray(res[n]["per_system_means"])
                                 for n in MAP_ARMS])
    n_pool = pool_means.size
    m_pool = float(pool_means.mean())
    se_pool = float(pool_means.std(ddof=1) / np.sqrt(n_pool))
    arm_ms = [res[n]["m"] for n in MAP_ARMS]
    var_between = float(np.var(arm_ms, ddof=1))
    var_within_scaled = float(np.mean([res[n]["sd"] ** 2 / res[n]["nsys"]
                                       for n in MAP_ARMS]))
    recent = float(np.mean([res[n]["occ_recent"] for n in MAP_ARMS]))
    prev = float(np.mean([res[n]["occ_prev"] for n in MAP_ARMS]))
    wo, desc = occ_a6(m_pool, recent, prev)
    pooled = dict(n=n_pool, m_pool=m_pool, se_pool=se_pool,
                  arm_means=arm_ms, var_between_arms=var_between,
                  var_within_over_nsys=var_within_scaled,
                  trace_prev=prev, trace_recent=recent, wo=bool(wo),
                  reading=desc)
    print(f"\nW-o (POOLED, primary): m_pool={m_pool:.4f}±{se_pool:.4f} (n={n_pool}) "
          f"in ({BAND[0]}, {BAND[1]}): {BAND[0] <= m_pool <= BAND[1]} — {desc}")
    print(f"  between-arm component (A4): var(arm means)={var_between:.3e} vs "
          f"within-arm/nsys={var_within_scaled:.3e} (both reported; the 0.022 "
          f"planning se assumed no between-arm component)")
    if not wo and desc == "OUT of band beyond near-edge":
        print("  *** F-C: pooled occ out of band beyond near-edge — "
              + ("rising coldocc trace: A1-style budget-limited zone (routed to "
                 "extend-ROUNDS decision) ***" if recent > prev else
                 "trace not rising: report ***"))
else:
    print("\nW-o (POOLED): requires G1+G2+G3 — unavailable "
          f"(missing: {[n for n in MAP_ARMS if n not in res]})")

# ---- W-s seed pairs (all three G1/G2/G3 pairs) + F-eq report
def pair(a, b):
    if a in res and b in res:
        da, db = res[a], res[b]
        s = float(np.hypot(da["se"], db["se"]))
        dm = abs(da["m"] - db["m"])
        ok = dm <= 2 * s
        print(f"\nW-s [{a}/{b}]: |{da['m']:.4f}-{db['m']:.4f}|={dm:.4f} "
              f"<= {2 * s:.4f}: {ok}")
        if dm > 3 * s:
            print(f"  *** F-eq: pair > 3 sigma ({3 * s:.4f}) — seed pooling "
                  f"invalid at this budget (report, no auto-lever) ***")
        return ok
    return None

ws_pairs = dict(G1G2=pair("G1", "G2"), G1G3=pair("G1", "G3"),
                G2G3=pair("G2", "G3"))

# ---- W-p assembly (rd-2 conditions, implemented EXACTLY)
def wg_ok(n):
    r = res[n]
    return r["n_hard_hi"] == 0 and r["n_hard_lo"] == 0 and r["n_soft_hi"] <= 4


def health_ok(n):
    c = res[n]["clauses"]
    return c["eevpd"] and c["pair_acc"] and c["nan"]


def g4_caveat(r):
    occ_ok, occ_desc = occ_a6(r["m"], r["occ_recent"], r["occ_prev"])
    g = r["geneig_full"]
    return (f"G4 (SVI entry): occ {r['m']:.3f} {occ_desc}; gen-eig "
            f"[{g[0]:.2f}, {g[-1]:.2f}] out={r['n_out']} (soft-hi {r['n_soft_hi']}, "
            f"low-side {r['n_soft_lo']}, F-R' {r['n_hard_hi']}, F-U {r['n_hard_lo']}); "
            f"RT {r['rt_pocket']} vs floor {r['floor_eff']:.0f}; "
            f"health {'OK' if health_ok('G4') else 'FAIL'}; F-M "
            f"{'fired' if r['fm_fired'] else 'clean'}")


fm_any = any(res[n]["fm_fired"] for n in res)
missing = [n for n in ALL_ARMS if n not in res]
if fm_any:
    verdict = "BLOCKED by F-M (mechanism falsified — report to human)"
elif missing:
    verdict = f"NOT assembled (W-p requires all four arms; missing: {missing})"
else:
    wt_all = all(res[n]["clauses"]["rt"] for n in ALL_ARMS)
    wh_all = all(health_ok(n) for n in ALL_ARMS)
    wg_all = all(wg_ok(n) for n in MAP_ARMS)   # F-R'/F-U on any candidate blocks
    ws_all = all(bool(v) for v in ws_pairs.values())
    wp = wt_all and bool(wo) and wh_all and wg_all and ws_all
    print(f"\nW-p components: W-t(all four)={wt_all}, W-o(pooled)={bool(wo)}, "
          f"W-h(all four)={wh_all}, W-g(G1-G3 band-or-routed, no F-R'/F-U)={wg_all}, "
          f"W-s(all three pairs)={ws_all}, F-M(none, incl. G4)={not fm_any}")
    if wp:
        g4_occ_ok, _ = occ_a6(res["G4"]["m"], res["G4"]["occ_recent"],
                              res["G4"]["occ_prev"])
        g4_full = g4_occ_ok and wg_ok("G4")
        if g4_full:
            verdict = ("PROPOSED DELIVERABLE, EITHER ENTRY MODE "
                       "(UNCERTIFIED; 1e-6*I ratification pending)")
        else:
            verdict = ("PROPOSED DELIVERABLE, MAP-ENTRY-SCOPED "
                       "(UNCERTIFIED; 1e-6*I ratification pending) — "
                       "G4 CAVEAT ATTACHED: " + g4_caveat(res["G4"]))
        # audit B3-1: routed-zone / F-S outcomes MUST reach the verdict string
        # (the one channel quoted into the claims register) — the checkpoint
        # names distinct product labels per zone
        zone_notes = []
        for n in list(MAP_ARMS) + (["G4"] if g4_full else []):
            r = res[n]
            if r["n_soft_hi"] > 0:
                zone_notes.append(f"{n}: {r['n_soft_hi']} axis(es) in (3,10] — "
                                  f"IMPROVED-PARTIAL, B3 caveats attach")
            if r["n_soft_lo"] > 0:
                zone_notes.append(f"{n}: {r['n_soft_lo']} axis(es) in [0.1,1/3) "
                                  f"— certifiable-with-caveat (budget-limited "
                                  f"ridge equilibration)")
        if fs_systematic:
            zone_notes.append("F-S SYSTEMATIC {10,4,11,1} low-side direction "
                              "(recorded)")
        if zone_notes:
            verdict += (" — ZONE-QUALIFIED (blind-spot (viii) residual "
                        "attaches): " + "; ".join(zone_notes))
    else:
        verdict = "NOT assembled (see zones)"

print(f"\nW-p verdict: {verdict}")
print("\nW-L (recipe) reminder: adjudicated by pt4_recipe_validate.py, NOT this "
      "scorer — attach its L1/L1b/L2/L3 output to the record before any W-p "
      "reading is final. L1b failure routes as PORT-BUG class (fix + re-audit; "
      "rd-2 condition 3). W-L failing alone does NOT block W-p (independent "
      "link; blocks PT-5 recipe use until fixed).")

out = dict(arms=res, pooled_occ=pooled, ws_pairs=ws_pairs,
           fs_systematic=bool(fs_systematic), fs_aligned_count=fs_count,
           fm_any=bool(fm_any), verdict=verdict)
json.dump(out, open(f"{OUT}/pt4_score.json", "w"), indent=1, default=str)
print(f"\n-> {OUT}/pt4_score.json")
