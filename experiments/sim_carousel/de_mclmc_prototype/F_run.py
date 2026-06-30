"""VALIDATION of the KERNEL-HOP composite (kernel_hop.py) on carousel-faithful
analytic mixtures with KNOWN weights. UNBIASEDNESS + the human's two worries are
the GATE. CPU, synchronous. ALL VERDICTS PROPOSED / UNCERTIFIED.

Geometry: carousel_testbed.fit_clusters() -> the two REAL carousel clusters
(mu,cov): index 0 = SECONDARY (~5x tighter, the hard/tight mode), index 1 =
GLOBAL (wide). We re-weight these same two modes to probe different regimes.

================================ PRE-REGISTRATION ================================
Driven by the REAL MCLMC kernel + honest within-mode mass matrix (global cov),
identical structure to the linear-DE composite; only the cross-chain mover changes.

T1 EFFICIENCY (carousel-faithful geometry, W=[0.30,0.70], balanced init):
  Cause hyp: the kernel-hop proposes near an ON-manifold chain, so (unlike the
    linear chord DE, measured ~0.6% accept / 0 round-trips) it should land
    on-manifold and ACCEPT.
  Prediction: hop acceptance >> 0.6% (order 10-50%) and >0 round-trips for some
    eps in {0.1,0.2,0.5}.
  Falsifier: best-eps hop acceptance <= 5% AND 0 round-trips -> hop is NOT better
    than linear DE here. (Baseline linear DE re-run head-to-head on this testbed.)
  Pick eps* = argmax round-trips (tie-break acceptance) for the unbiasedness battery.

T2 COMPARABLE-MASS (W=[0.60,0.40]):
  (a) weight recovery from SINGLE-MODE init: post-burn w(mode1) within max(3 SE,
      0.05) of 0.40 -> no draining of a significant mode.
  (b) invariance-from-truth: init exact 0.60/0.40; |w_2half - w_1half| < 3 SE_diff
      (no drift) AND |w - 0.40| < 3 SE.
  Falsifier: drift > 3 SE_diff, or recovered weight off by > max(3 SE,0.05).

T3 DOMINANT + TINY (tiny mode = the TIGHT secondary, index 0):
  regimes wt in {0.03, 0.001}.  16 chains.  OVER-REP stress: init ONE chain in the
  tiny mode (init occupancy 1/16=0.0625) + 15 in the dominant mode.
  Primary metric: post-burn TIME-AVERAGED tiny-mode occupancy f_tiny (+/- moving-
  block-bootstrap SE).  Truth = wt.
  Pre-registered PASS:
    NOT PINNED : f_tiny < 0.0625 - 3 SE   (the stuck chain DID leave)
    NOT DRAINED: f_tiny consistent with wt within 3 SE (for wt=0.03 where truth
                 deserves mass); for wt=0.001 the dominant requirement is NOT
                 PINNED and f_tiny within 3 SE of 0.001 (occupancy ~ empty).
  Falsifier / REJECT: f_tiny pinned ~0.0625 (over-representation) OR f_tiny==0
    when wt>0 deserves mass (draining). A fast mover that mis-recovers weight is
    REJECTED however fast.

T4 KS marginals vs analytic (W=[0.60,0.40], truth init, thinned by tau): min KS
  p-value over 14 axes > 0.01.
=================================================================================
"""
import os, sys, time, json
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import carousel_testbed as T
from kernel_hop import make_kernel_hop_composite
from de_mclmc import make_composite as make_linear_de
from validate_analytic import integrated_autocorr_time, block_bootstrap_se

NCH = 16; K = 20; SEED = 20260627
def pr(*a): print(*a, flush=True)

# ---- shared geometry (the two real carousel clusters) ----
mu_np, cov_np, sec, glob = T.fit_clusters()           # [0]=secondary(tight), [1]=global(wide)
Cg = jnp.asarray(cov_np[1])                            # honest GLOBAL-mode MM + kernel cov
chol_np = [np.linalg.cholesky(cov_np[k]) for k in range(2)]

def build_mix(w):
    return T.Mixture(mu_np, cov_np, np.asarray(w))

def draws_from_mode(mix, mode, n, rng):
    std = rng.standard_normal((n, T.D))
    return mu_np[mode] + std @ np.asarray(mix._chol_np[mode]).T

def init_balanced(mix, seed):
    rng = np.random.default_rng(seed)
    a = draws_from_mode(mix, 0, NCH//2, rng); b = draws_from_mode(mix, 1, NCH//2, rng)
    out = np.empty((NCH, T.D)); out[0::2] = a; out[1::2] = b; return out

def init_single(mix, mode, seed):
    return draws_from_mode(mix, mode, NCH, np.random.default_rng(seed))

def init_truth(mix, seed):
    z, _ = mix.exact_draws_balanced(NCH, np.random.default_rng(seed)); return z

def init_one_in_tiny(mix, tiny_mode, seed):
    """1 chain in tiny mode, NCH-1 in the dominant mode (over-representation test)."""
    rng = np.random.default_rng(seed)
    dom = 1 - tiny_mode
    out = draws_from_mode(mix, dom, NCH, rng)
    out[0] = draws_from_mode(mix, tiny_mode, 1, rng)[0]
    return out

# ---- run harness ----
def run(comp, init_pos, n_rounds, seed):
    st = comp["init_states"](jnp.asarray(init_pos), jax.random.key(seed))
    keys = jax.random.split(jax.random.key(seed+7), n_rounds)
    pos = np.empty((n_rounds, NCH, T.D)); acc = np.empty((n_rounds, NCH))
    for r in range(n_rounds):
        st, (p, ec, a) = comp["round"](st, keys[r]); pos[r] = np.asarray(p); acc[r] = np.asarray(a)
    return pos, acc

def round_trips(modes):
    R, C = modes.shape; nrt = np.zeros(C, int); ncr = np.zeros(C, int)
    for c in range(C):
        d = np.diff(modes[:, c]); ups = int((d == 1).sum()); dns = int((d == -1).sum())
        nrt[c] = min(ups, dns); ncr[c] = int(np.abs(d).sum())
    return nrt, ncr

def make_hop(mix, eps):
    return make_kernel_hop_composite(mix.logp, T.D, NCH, L=T.L0, step_size=T.SS0, K=K,
                                     eps=eps, kernel_cov=Cg, p_hop=1.0, inverse_mass_matrix=Cg)

# =============================================================================
def main():
    t0 = time.time(); boot = np.random.default_rng(SEED); store = {}; summ = {}

    # ----------------------------- T1: EFFICIENCY ----------------------------
    pr("\n===== T1 EFFICIENCY (W=[0.30,0.70], balanced init): kernel-hop eps sweep + linear-DE baseline =====")
    mix1 = build_mix([0.30, 0.70]); R1 = 1500
    # linear-DE baseline (Cholesky jitter b0=0.1, p_jump=0.3 -- the carousel config)
    Lg = jnp.linalg.cholesky(Cg)
    de = make_linear_de(mix1.logp, T.D, NCH, L=T.L0, step_size=T.SS0, K=K,
                        b0=0.1, p_jump=0.3, inverse_mass_matrix=Cg, eps_scale=Lg)
    pos, acc = run(de, init_balanced(mix1, SEED), R1, SEED)
    m = mix1.classify(pos); nrt, ncr = round_trips(m)
    de_acc = float(acc.mean()); de_rt = int(nrt.sum())
    pr(f"  linear-DE baseline : accept={de_acc*100:6.3f}%   round-trips={de_rt}")
    eps_grid = [0.5, 1.0, 2.0]; t1 = []   # diagnosed-best band (F_bandwidth_diag: overlap needs eps>~1)
    for eps in eps_grid:
        comp = make_hop(mix1, eps)
        pos, acc = run(comp, init_balanced(mix1, SEED), R1, SEED)
        m = mix1.classify(pos); nrt, ncr = round_trips(m)
        a = float(acc.mean()); rt = int(nrt.sum())
        wser = m.mean(1); wpb = float(wser[R1//2:].mean())
        t1.append(dict(eps=eps, acc=a, rt=rt, cross=int(ncr.sum()), w=wpb, wser=wser))
        pr(f"  hop eps={eps:<4}     : accept={a*100:6.3f}%   round-trips={rt:4d}  cross={int(ncr.sum()):4d}  w(glob)={wpb:.3f}")
    # pick eps*: max round-trips, tie-break acceptance
    eps_star = max(t1, key=lambda d: (d["rt"], d["acc"]))["eps"]
    # Pre-registered FALSIFIER: best-eps accept<=5% AND 0 round-trips. T1 passes unless BOTH.
    max_acc = max(d["acc"] for d in t1); max_rt = max(d["rt"] for d in t1)
    t1_pass = not (max_acc <= 0.05 and max_rt == 0)
    best_rt = max(t1, key=lambda d: d["rt"])
    pr(f"  -> eps* = {eps_star} | best round-trips={best_rt['rt']} (linear-DE {de_rt}) | "
       f"T1 {'PASS' if t1_pass else 'FAIL'} (registered falsifier: acc<=5% AND rt==0)")
    summ["T1"] = dict(linear_de_acc=de_acc, linear_de_rt=de_rt,
                      hop=[{k: d[k] for k in ('eps','acc','rt','cross','w')} for d in t1],
                      eps_star=eps_star, pass_=bool(t1_pass))
    store["T1_de_acc"] = de_acc; store["T1_de_rt"] = de_rt
    store["T1_eps"] = np.array([d["eps"] for d in t1]); store["T1_acc"] = np.array([d["acc"] for d in t1])
    store["T1_rt"] = np.array([d["rt"] for d in t1])
    for d in t1: store[f"T1_wser_eps{d['eps']}"] = d["wser"]

    # ----------------------- T2: COMPARABLE-MASS 0.6/0.4 ---------------------
    pr("\n===== T2 COMPARABLE-MASS (W=[0.60,0.40]) =====")
    mix2 = build_mix([0.60, 0.40]); R2 = 2000; RINV = 3000
    comp2 = make_hop(mix2, eps_star)
    # (a) weight recovery from single-mode (all-mode-0) init
    pos, acc = run(comp2, init_single(mix2, 0, SEED+11), R2, SEED+11)
    wser = mix2.classify(pos).mean(1); w_rec = float(wser[R2//2:].mean())
    se_rec, _ = block_bootstrap_se(wser[R2//2:], rng=boot)
    rec_acc = float(acc.mean())
    pr(f"  (a) recovery/all-mode0 init: w(mode1)={w_rec:.3f} +/- {se_rec:.3f} (target 0.40) | hop acc={rec_acc*100:.2f}%")
    # (b) invariance-from-truth
    pos, acc = run(comp2, init_truth(mix2, SEED+21), RINV, SEED+21)
    wser_i = mix2.classify(pos).mean(1); w_inv = float(wser_i.mean())
    se_inv, _ = block_bootstrap_se(wser_i, rng=boot)
    h = RINV//2; w1 = float(wser_i[:h].mean()); w2 = float(wser_i[h:].mean())
    se1, _ = block_bootstrap_se(wser_i[:h], rng=boot); se2, _ = block_bootstrap_se(wser_i[h:], rng=boot)
    se_diff = float(np.hypot(se1, se2))
    pr(f"  (b) invariance-from-truth : w={w_inv:.3f}+/-{se_inv:.3f} (target 0.40) | drift |w2-w1|={abs(w2-w1):.4f} (3 SE_diff={3*se_diff:.4f})")
    c2a = abs(w_rec - 0.40) < max(3*se_rec, 0.05)
    c2b_w = abs(w_inv - 0.40) < 3*se_inv; c2b_drift = abs(w2-w1) < 3*se_diff
    t2_pass = c2a and c2b_w and c2b_drift
    pr(f"  T2 {'PASS' if t2_pass else 'FAIL'}  [recover {c2a} | inv-w {c2b_w} | no-drift {c2b_drift}]")
    summ["T2"] = dict(w_rec=w_rec, se_rec=se_rec, rec_acc=rec_acc, w_inv=w_inv, se_inv=se_inv,
                      w1=w1, w2=w2, se_diff=se_diff, pass_=bool(t2_pass))
    store["T2_wser_rec"] = wser; store["T2_wser_inv"] = wser_i

    # ----------------------- T3: DOMINANT + TINY -----------------------------
    pr("\n===== T3 DOMINANT + TINY (tiny = tight secondary mode, index 0); over-rep stress: 1 chain in tiny =====")
    R3 = 4000; t3 = {}
    for wt in [0.03, 0.001]:
        mixT = build_mix([wt, 1.0 - wt])     # index0=tiny(secondary), index1=dominant(global)
        compT = make_hop(mixT, eps_star)
        pos, acc = run(compT, init_one_in_tiny(mixT, tiny_mode=0, seed=SEED+31), R3, SEED+31)
        modes = mixT.classify(pos)
        f_series = (modes == 0).mean(1)              # occupancy fraction of TINY mode per round
        burn = R3//2
        f_tiny = float(f_series[burn:].mean())
        se_f, _ = block_bootstrap_se(f_series[burn:], rng=boot)
        not_pinned = f_tiny < 0.0625 - 3*se_f
        within_truth = abs(f_tiny - wt) < 3*max(se_f, 1e-4)
        drained = (f_tiny == 0.0) and (wt > 0)
        # nrt for context
        nrt, _ = round_trips(modes)
        if wt == 0.03:
            ok = not_pinned and within_truth and not drained
        else:  # wt=0.001: dominant requirement = NOT pinned + occupancy ~ truth(empty)
            ok = not_pinned and within_truth and not drained
        pr(f"  wt={wt:<6} : f_tiny(post-burn)={f_tiny:.4f} +/- {se_f:.4f}  (truth {wt}) | "
           f"init 0.0625 | not_pinned={not_pinned} within_truth={within_truth} drained={drained} "
           f"round-trips={int(nrt.sum())} hopacc={float(acc.mean())*100:.2f}% -> {'PASS' if ok else 'FAIL'}")
        if float(acc.mean()) < 0.01 and f_tiny >= 0.0625 - 1e-9:
            pr(f"           NOTE: hop acceptance ~0 -> mover is INERT; pin at 0.0625 is the global "
               f"inefficiency, NOT a tiny-mode-specific drain/over-rep. T3 CONFOUNDED/INCONCLUSIVE here.")
        t3[wt] = dict(f_tiny=f_tiny, se_f=se_f, not_pinned=bool(not_pinned),
                      within_truth=bool(within_truth), drained=bool(drained),
                      rt=int(nrt.sum()), acc=float(acc.mean()), pass_=bool(ok))
        store[f"T3_fseries_wt{wt}"] = f_series
    t3_pass = all(v["pass_"] for v in t3.values())
    summ["T3"] = {str(k): v for k, v in t3.items()}; summ["T3"]["pass_"] = bool(t3_pass)
    pr(f"  T3 {'PASS' if t3_pass else 'FAIL'}")

    # ----------------------- T4: KS marginals --------------------------------
    pr("\n===== T4 KS marginals vs analytic (W=[0.60,0.40], truth init) =====")
    RKS = 3000
    pos, acc = run(comp2, init_truth(mix2, SEED+41), RKS, SEED+41)
    wser = mix2.classify(pos).mean(1); tau = integrated_autocorr_time(wser)
    spacing = max(1, int(round(tau))); half = RKS//2
    pool = pos[half::spacing].reshape(-1, T.D)
    rng = np.random.default_rng(SEED+1)
    if pool.shape[0] > 8000: pool = pool[rng.choice(pool.shape[0], 8000, replace=False)]
    exact, _ = mix2.exact_draws(max(pool.shape[0], 2000), rng)
    ps = np.empty(T.D)
    for j in range(T.D):
        _, ps[j] = stats.ks_2samp(pool[:, j], exact[:, j])
    min_p = float(ps.min()); argmin = int(ps.argmin())
    t4_pass = min_p > 0.01
    pr(f"  min KS p={min_p:.3f} on axis {argmin} (tau~{tau:.0f}, spacing {spacing}) -> {'PASS' if t4_pass else 'FAIL'}")
    summ["T4"] = dict(min_p=min_p, argmin=argmin, tau=float(tau), pass_=bool(t4_pass))
    store["T4_ks_p"] = ps; store["T4_pool_argmin"] = pool[:, argmin]; store["T4_exact_argmin"] = exact[:, argmin]

    # ----------------------------- plots -------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    ax[0,0].plot([d["eps"] for d in t1], [d["acc"]*100 for d in t1], "o-", label="kernel-hop")
    ax[0,0].axhline(de_acc*100, color="r", ls="--", label=f"linear-DE {de_acc*100:.2f}%")
    ax[0,0].set_xlabel("eps"); ax[0,0].set_ylabel("accept %"); ax[0,0].legend(fontsize=8); ax[0,0].set_title("T1 acceptance")
    ax[0,1].plot([d["eps"] for d in t1], [d["rt"] for d in t1], "s-", color="C2")
    ax[0,1].axhline(de_rt, color="r", ls="--", label=f"linear-DE {de_rt}"); ax[0,1].legend(fontsize=8)
    ax[0,1].set_xlabel("eps"); ax[0,1].set_ylabel("round-trips"); ax[0,1].set_title(f"T1 mixing (eps*={eps_star})")
    ax[0,2].plot(store["T2_wser_rec"], lw=0.5, label="all-mode0 init")
    ax[0,2].axhline(0.40, color="k", ls="--", label="truth 0.40"); ax[0,2].axhline(w_rec, color="C1", alpha=.6, label=f"post-burn {w_rec:.3f}")
    ax[0,2].set_xlabel("round"); ax[0,2].set_ylabel("w(mode1)"); ax[0,2].legend(fontsize=8); ax[0,2].set_title("T2a weight recovery")
    ax[1,0].plot(store["T2_wser_inv"], lw=0.4)
    ax[1,0].axhline(0.40, color="k", ls="--"); ax[1,0].axhline(w1, color="C0", alpha=.6, label=f"1st {w1:.3f}"); ax[1,0].axhline(w2, color="C1", alpha=.6, label=f"2nd {w2:.3f}")
    ax[1,0].set_xlabel("round"); ax[1,0].set_ylabel("w(mode1)"); ax[1,0].legend(fontsize=8); ax[1,0].set_title("T2b invariance-from-truth")
    for wt, col in [(0.03, "C3"), (0.001, "C4")]:
        fs = store[f"T3_fseries_wt{wt}"]
        # running mean for visibility
        rm = np.cumsum(fs)/np.arange(1, len(fs)+1)
        ax[1,1].plot(rm, color=col, lw=0.8, label=f"wt={wt} (run-mean)")
        ax[1,1].axhline(wt, color=col, ls=":", lw=1)
    ax[1,1].axhline(0.0625, color="gray", ls="--", label="1/16 init (pin level)")
    ax[1,1].set_xlabel("round"); ax[1,1].set_ylabel("tiny-mode occupancy"); ax[1,1].set_yscale("log")
    ax[1,1].legend(fontsize=7); ax[1,1].set_title("T3 tiny-mode occupancy (1-in-tiny init)")
    ax[1,2].hist(store["T4_pool_argmin"], bins=50, density=True, alpha=.5, label="hop")
    ax[1,2].hist(store["T4_exact_argmin"], bins=50, density=True, alpha=.5, label="exact")
    ax[1,2].set_xlabel(f"axis {argmin} (worst KS, p={min_p:.3f})"); ax[1,2].legend(fontsize=8); ax[1,2].set_title("T4 worst marginal")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "F_validation.png"), dpi=110); plt.close(fig)

    # ---- overall proposed verdict ----
    overall = all([t1_pass, t2_pass, t3_pass, t4_pass])
    pr("\n===== PROPOSED / UNCERTIFIED VERDICT =====")
    for name, ok in [("T1 efficiency", t1_pass), ("T2 comparable-mass", t2_pass),
                     ("T3 dominant+tiny", t3_pass), ("T4 KS marginals", t4_pass)]:
        pr(f"   {name:22s} -> {'PASS' if ok else 'FAIL'}")
    pr(f"   OVERALL -> {'PASS (PROPOSED UNBIASED+EFFICIENT)' if overall else 'FAIL / INCONCLUSIVE (see per-test)'}")
    summ["overall_pass"] = bool(overall); summ["eps_star"] = eps_star

    np.savez(os.path.join(HERE, "F_backing.npz"),
             **{k: np.asarray(v) for k, v in store.items()
                if isinstance(v, (np.ndarray, list, float, int, np.floating, np.integer))})
    with open(os.path.join(HERE, "F_summary.json"), "w") as f:
        json.dump(summ, f, indent=2, default=float)
    pr(f"\nDONE in {time.time()-t0:.1f}s -> F_validation.png, F_backing.npz, F_summary.json")


if __name__ == "__main__":
    main()
