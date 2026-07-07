"""Validate the SA-MCMC composite mover on the EASY analytic bimodal target
(identical target to validate_analytic.py: D=10, modes +/-5 on axis0, weights
[0.7,0.3], barrier m^2/2=12.5 so vanilla MCLMC is trapped).

Gates A (V1..V4) pre-registered in the orchestrator brief; thresholds derived.
Run:  python validate_sa_analytic.py <gaussian|mixture>
Saves V_sa_<prop>.npz and four V_sa_<prop>_*.png; prints a PROPOSED/UNCERTIFIED
verdict for the orchestrator to grade.  PLOTS are produced before the numbers.

PRE-REGISTERED (method discipline), single Gaussian vs mixture:
 cause: SA-MCMC preserves pi^{otimes N} by Prop.1 (self-inclusive deletion), so
   from a both-modes init the weight must hold at 0.70 with no drift (V2). From a
   single-mode init, discovery needs the empty mode SEEDED: a rare MCLMC barrier
   leak, then SA amplification. Prediction: the MIXTURE (KDE) amplifies from ONE
   seed (a kernel sits on the leaked point) so it should DISCOVER and recover 0.70;
   the single GAUSSIAN needs MANY seeds before its fitted cov spreads enough to
   propose across, so its V1 discovery may be SLOW/marginal even though its V2
   invariance is fine. Falsifier (core method): V2 weight off truth by >3 SE, or
   1st-vs-2nd-half drift >3 SE, or a mode drains/over-fills.
"""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from sa_move import make_sa_composite
from validate_analytic import (integrated_autocorr_time, block_bootstrap_se,
                               logdensity_fn, analytic_axis0_pdf,
                               analytic_axis0_samples, exact_mixture_draws,
                               D, m, W, MU)

# ----------------------------- config ----------------------------------------
PROP = sys.argv[1] if len(sys.argv) > 1 else "mixture"
N_CHAINS = 64
L, STEP, K = 2.0, 0.5, 20
N_SA = 64
SCALE = 1.0          # gaussian prop_scale
BW = 1.0             # mixture bandwidth (within-mode std is 1.0; modes +/-5)
R_V1, BURN_V1 = 2500, 800
R_V2 = 2500
SEED = 20260627

def make_mover():
    kw = dict(prop_scale=SCALE) if PROP == "gaussian" else dict(bandwidth=BW)
    return make_sa_composite(logdensity_fn, D, N_CHAINS, L=L, step_size=STEP, K=K,
                             n_sa=N_SA, proposal=PROP, **kw)

def run_composite(comp, init_pos, n_rounds, key):
    st = comp["init_states"](init_pos, jax.random.fold_in(key, 0))
    keys = jax.random.split(key, n_rounds)
    pos_out = np.empty((n_rounds, N_CHAINS, D)); acc_out = np.empty(n_rounds)
    for r in range(n_rounds):
        st, (p, ec, sub) = comp["round"](st, keys[r])
        pos_out[r] = np.asarray(p); acc_out[r] = float(np.asarray(sub).mean())
    return pos_out, acc_out

def main():
    t_start = time.time()
    comp = make_mover(); print("config:", comp["config"])
    boot_rng = np.random.default_rng(SEED)

    # ----- V1: weight recovery from single-mode init -------------------------
    print(f"\n[V1] SA({PROP}) from ALL-in-+mode init ...")
    init_single = jnp.zeros((N_CHAINS, D)).at[:, 0].set(m)
    t0 = time.time()
    pos1, acc1 = run_composite(comp, init_single, R_V1, jax.random.key(SEED + 1))
    print(f"   {R_V1} rounds in {time.time()-t0:.1f}s")
    keep1 = pos1[BURN_V1:]; frac_series = (keep1[:, :, 0] > 0).mean(axis=1)
    w_plus = float(frac_series.mean())
    se_w, blk = block_bootstrap_se(frac_series, rng=boot_rng)
    tau1 = integrated_autocorr_time(frac_series)
    print(f"   weight(+mode)={w_plus:.4f} +/- {se_w:.4f} (block={blk}, tau~{tau1:.1f})  "
          f"|est-0.70|={abs(w_plus-0.70):.4f} (3SE={3*se_w:.4f})")

    # vanilla MCLMC contrast
    print("[V1] vanilla MCLMC (no SA) from SAME init ...")
    total_steps = R_V1 * K
    st_v = comp["init_states"](init_single, jax.random.key(SEED + 2))
    chunk = 2000; vmaster = jax.random.key(SEED + 3); tail = []; done = 0
    while done < total_steps:
        nthis = min(chunk, total_steps - done)
        ck = jax.random.split(jax.random.fold_in(vmaster, done), nthis * N_CHAINS).reshape(nthis, N_CHAINS)
        st_v, posv = comp["mclmc_only"](st_v, ck)
        if done >= total_steps - 2 * chunk: tail.append(np.asarray(posv[:, :, 0]))
        done += nthis
    tail = np.concatenate(tail, axis=0); w_vanilla = float((tail > 0).mean())
    print(f"   vanilla weight(+mode) tail={w_vanilla:.4f} (should be >0.85)")

    # ----- V2: invariance from EXACT-truth init ------------------------------
    print(f"\n[V2] SA({PROP}) from EXACT mixture init (correct 0.7/0.3) ...")
    init_truth, _ = exact_mixture_draws(N_CHAINS, np.random.default_rng(SEED + 4))
    t0 = time.time()
    pos2, acc2 = run_composite(comp, jnp.asarray(init_truth), R_V2, jax.random.key(SEED + 5))
    print(f"   {R_V2} rounds in {time.time()-t0:.1f}s")
    frac2 = (pos2[:, :, 0] > 0).mean(axis=1); w2 = float(frac2.mean())
    se2, blk2 = block_bootstrap_se(frac2, rng=boot_rng)
    half = R_V2 // 2
    w_first, w_second = float(frac2[:half].mean()), float(frac2[half:].mean())
    se_first, _ = block_bootstrap_se(frac2[:half], rng=boot_rng)
    se_second, _ = block_bootstrap_se(frac2[half:], rng=boot_rng)
    se_diff = np.hypot(se_first, se_second)
    print(f"   weight(+mode)={w2:.4f} +/- {se2:.4f}  drift |{w_second-w_first:.4f}| (3SE_diff={3*se_diff:.4f})")
    thin = pos2[half::5].reshape(-1, D)
    plus = thin[thin[:, 0] > 0]; minus = thin[thin[:, 0] < 0]
    pm_mean_p, pm_mean_m = plus[:, 0].mean(), minus[:, 0].mean()
    pm_var_p, pm_var_m = plus[:, 0].var(), minus[:, 0].var(); within_var = thin[:, 1].var()
    print(f"   +mode mean={pm_mean_p:.3f} var={pm_var_p:.3f}; -mode mean={pm_mean_m:.3f} var={pm_var_m:.3f}; axis1 var={within_var:.3f}")
    tau2 = integrated_autocorr_time(frac2); ks_sp = max(1, int(round(tau2)))
    ks_pool = pos2[half::ks_sp].reshape(-1, D)
    rng_ks = np.random.default_rng(SEED + 6)
    ca0 = ks_pool[:, 0].copy(); rng_ks.shuffle(ca0); ca0 = ca0[:4000]
    ks0, ksp0 = stats.ks_2samp(ca0, analytic_axis0_samples(len(ca0), rng_ks))
    ca1 = ks_pool[:, 1].copy(); rng_ks.shuffle(ca1); ca1 = ca1[:4000]
    ks1, ksp1 = stats.ks_2samp(ca1, rng_ks.standard_normal(len(ca1)))
    print(f"   KS axis0 p={ksp0:.3f}  KS axis1 p={ksp1:.3f}  (spacing {ks_sp}, tau2~{tau2:.0f})")

    # ----- V4: acceptance + ergodicity ---------------------------------------
    sub_acc = float(np.concatenate([acc1, acc2]).mean())
    min_eig = float(np.linalg.eigvalsh(np.cov(thin.T)).min())
    print(f"\n[V4] SA substitution rate={sub_acc:.3f} (0.05..0.95); min eig pooled cov={min_eig:.4f}")

    # =============================== PLOTS ===================================
    pre = os.path.join(HERE, f"V_sa_{PROP}")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(R_V1), (pos1[:, :, 0] > 0).mean(axis=1), lw=0.7, label="composite +mode frac")
    ax.axhline(0.70, color="k", ls="--", label="truth 0.70"); ax.axvline(BURN_V1, color="r", ls=":")
    ax.axhline(w_vanilla, color="C3", label=f"vanilla tail {w_vanilla:.3f}")
    ax.axhline(w_plus, color="C1", alpha=.6, label=f"post-burn {w_plus:.3f}")
    ax.set_title(f"V1 weight recovery SA-{PROP}"); ax.set_xlabel("round"); ax.set_ylabel("+mode frac"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(pre + "_V1.png", dpi=110); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(np.arange(R_V2), frac2, lw=0.7); axes[0].axhline(0.70, color="k", ls="--")
    axes[0].axhline(w_first, color="C0", alpha=.6, label=f"1st {w_first:.3f}")
    axes[0].axhline(w_second, color="C1", alpha=.6, label=f"2nd {w_second:.3f}")
    axes[0].set_title(f"V2 invariance-from-truth SA-{PROP}"); axes[0].legend(fontsize=8)
    axes[1].hist(pos2[0, :, 0], bins=20, density=True, alpha=.4, label="t=0")
    axes[1].hist(pos2[-1, :, 0], bins=20, density=True, alpha=.4, label="t=end")
    xs = np.linspace(-9, 9, 400); axes[1].plot(xs, analytic_axis0_pdf(xs), "k-", label="analytic")
    axes[1].legend(fontsize=8); axes[1].set_title("axis0 t=0 vs end vs analytic")
    fig.tight_layout(); fig.savefig(pre + "_V2.png", dpi=110); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].hist(thin[:, 0], bins=60, density=True, alpha=.5, label="composite")
    axes[0].plot(xs, analytic_axis0_pdf(xs), "k-", lw=2, label="analytic"); axes[0].legend(fontsize=8)
    axes[0].set_title(f"axis0 marginal (KS p={ksp0:.3f})")
    axes[1].hist(thin[:, 1], bins=60, density=True, alpha=.5, label="composite")
    axes[1].plot(xs, stats.norm.pdf(xs), "k-", lw=2, label="N(0,1)"); axes[1].set_xlim(-5, 5)
    axes[1].legend(fontsize=8); axes[1].set_title(f"axis1 marginal (KS p={ksp1:.3f})")
    fig.tight_layout(); fig.savefig(pre + "_V3.png", dpi=110); plt.close(fig)

    np.savez(pre + ".npz", frac_series_v1=frac_series, w_plus=w_plus, se_w=se_w,
             w_vanilla=w_vanilla, frac2=frac2, w2=w2, se2=se2, w_first=w_first,
             w_second=w_second, se_diff=se_diff, pm_mean_p=pm_mean_p, pm_mean_m=pm_mean_m,
             pm_var_p=pm_var_p, pm_var_m=pm_var_m, within_var=within_var,
             ks_p0=ksp0, ks_p1=ksp1, sub_acc=sub_acc, min_eig=min_eig, config=str(comp["config"]))

    # ----------------------- PROPOSED verdict --------------------------------
    c_v1 = abs(w_plus - 0.70) < 3 * se_w
    c_v1b = w_vanilla > 0.85
    c_v2a = abs(w2 - 0.70) < 3 * se2
    c_v2b = abs(w_second - w_first) < 3 * se_diff
    c_v2c = (abs(pm_mean_p - m) < 0.10 and abs(pm_mean_m + m) < 0.10
             and abs(pm_var_p - 1.0) < 0.10 and abs(pm_var_m - 1.0) < 0.10)
    c_v2d = ksp0 > 0.05; c_v3 = ksp0 > 0.05 and ksp1 > 0.05
    c_v4 = (0.05 < sub_acc < 0.95) and (min_eig > 0)
    M = lambda b: "PASS" if b else "FAIL"
    print("\n========== PRE-REGISTERED CHECKS (PROPOSED / UNCERTIFIED) ==========")
    print(f" V1  weight recovery  {w_plus:.4f}+/-{se_w:.4f} vs .70 -> {M(c_v1)}")
    print(f" V1  vanilla trapped  {w_vanilla:.4f} > 0.85         -> {M(c_v1b)}")
    print(f" V2a weight@truth     {w2:.4f}+/-{se2:.4f} vs .70    -> {M(c_v2a)}")
    print(f" V2b no drift         |{w_second-w_first:.4f}|<{3*se_diff:.4f}    -> {M(c_v2b)}")
    print(f" V2c per-mode moments within 0.10                    -> {M(c_v2c)}")
    print(f" V2d KS axis0 p={ksp0:.3f}                            -> {M(c_v2d)}")
    print(f" V3  KS marginals a0={ksp0:.3f} a1={ksp1:.3f}          -> {M(c_v3)}")
    print(f" V4  sub-rate={sub_acc:.3f} min_eig={min_eig:.3f}      -> {M(c_v4)}")
    print(f" total wall {time.time()-t_start:.0f}s")

if __name__ == "__main__":
    main()
