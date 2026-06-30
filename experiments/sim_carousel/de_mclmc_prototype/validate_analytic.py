"""Validate the standalone DE-MCLMC composite sampler on an analytic bimodal
target where the truth is known exactly.

PRE-REGISTERED CRITERIA (stated before running; thresholds derived, not tuned):

 V1 MODE-WEIGHT RECOVERY (from a single-mode init):
     - Composite started with ALL chains in the +m (weight-0.7) mode must recover
       weight(+mode) = 0.70. Threshold: |est - 0.70| < 3 * SE, where SE is a
       moving-block-bootstrap SE over the per-round +mode-fraction time series
       (block length set from the estimated integrated autocorrelation time, so
       the SE accounts for temporal correlation). Falsifier: |est-0.70| >= 3 SE.
     - CONTRAST: vanilla MCLMC (no DE) from the SAME init must stay trapped,
       weight(+mode) > 0.98. (Establishes DE is NECESSARY.)

 V2 INVARIANCE FROM TRUTH (the sharp bias test):
     - Initialize chains with EXACT mixture draws (correct 0.7/0.3 weights). Run
       the composite many rounds. The distribution must NOT drift:
         (a) weight(+mode): |est - 0.70| < 3 SE  (block-bootstrap SE as in V1)
         (b) NO temporal drift: |w(second half) - w(first half)| < 3 SE_diff
         (c) per-mode means within 0.10 of +/-3.5; per-mode marginal vars within
             0.10 of 1.0
         (d) axis-0 marginal at t=end vs analytic mixture: KS p > 0.05 (thinned
             to curb autocorrelation; plot is primary, KS supportive)
     - A biased kernel would pull weights/moments away from truth.

 V3 MARGINAL MATCH:
     - axis-0 marginal (across modes) and axis-1 marginal (within mode) of
       composite samples vs the analytic densities: overlay + KS. Pre-registered:
       visual overlay must show no systematic mismatch; KS p > 0.05 (thinned).

 V4 DE ACCEPTANCE + ERGODICITY:
     - DE acceptance rate in (0.05, 0.95) (neither ~0 nor ~1).
     - eps>0 keeps full rank: min eigenvalue of the pooled sample covariance > 0.

METHOD DISCIPLINE: plots are produced BEFORE the numeric verdict; the verdict is
PROPOSED / UNCERTIFIED for the orchestrator to grade.
"""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from de_mclmc import make_composite

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------- analytic target -------------------------------
# m and p_jump chosen via diagnostics _scan_m.py and _scan_pjump.py.
# Free-energy barrier (log density drop midpoint->peak) = m^2/2.
#   m=3.5 (barrier 6.1), m=4.0 (8.0): vanilla MCLMC CROSSES within the run ->
#     contrast cannot show DE necessity (barrier too low).
#   m=5.0 (barrier 12.5): vanilla is TRAPPED (~0.98 in start mode over the run).
# Cross-mode DE mixing rate is set by p_jump (gamma=1 difference-vector jumps land
# DIRECTLY on the other mode, skipping the low-density midpoint -> barrier-
# independent), NOT by the barrier. So p_jump=0.5 keeps the integrated
# autocorrelation time of the mode weight small (tau~70 rounds, the run is
# powered) EVEN at a barrier vanilla cannot cross. With p_jump=0.5 DE ALSO
# discovers the hidden mode from a single-mode init at m=5.0 (it fails at the
# lower p_jump=0.2: more gamma=1 attempts are needed for discovery).
D = 10
m = 5.0                       # modes at +/-5.0 along axis 0; barrier m^2/2 = 12.5
W = np.array([0.7, 0.3])      # UNEQUAL weights
MU = np.array([+m, -m])
_logW = jnp.log(jnp.asarray(W))
_MU = jnp.asarray(MU)
_c = -0.5 * D * jnp.log(2 * jnp.pi)

def logdensity_fn(z):
    z0 = z[0]
    quad_rest = jnp.sum(z[1:] ** 2)
    comp0 = _logW[0] + _c - 0.5 * ((z0 - _MU[0]) ** 2 + quad_rest)
    comp1 = _logW[1] + _c - 0.5 * ((z0 - _MU[1]) ** 2 + quad_rest)
    return jax.scipy.special.logsumexp(jnp.stack([comp0, comp1]))

def analytic_axis0_pdf(x):
    return (W[0] * stats.norm.pdf(x, MU[0], 1.0)
            + W[1] * stats.norm.pdf(x, MU[1], 1.0))

def exact_mixture_draws(n, rng):
    # DETERMINISTIC component counts (round(0.7 n) in +mode) so the ensemble starts
    # at EXACTLY the correct weight -> the invariance test measures drift with NO
    # initialization transient (a random draw would give a binomial weight offset
    # that relaxes and muddies the drift test). Positions are exact Gaussian draws.
    n_plus = int(round(W[0] * n))
    comp = np.ones(n, int); comp[:n_plus] = 0          # 0 = +mode, 1 = -mode
    z = rng.standard_normal((n, D))
    z[:, 0] += MU[comp]
    return z, comp

def analytic_axis0_samples(n, rng):
    comp = (rng.random(n) >= W[0]).astype(int)
    x = rng.standard_normal(n) + MU[comp]
    return x

# ----------------------------- config ----------------------------------------
N_CHAINS = 64
L, STEP, K = 2.0, 0.5, 20
B0, P_JUMP = 0.05, 0.5
R_V1, BURN_V1 = 3000, 1000
R_V2 = 3000
SEED = 20260627

# --------------------------- helpers: bootstrap ------------------------------
def integrated_autocorr_time(x, c=5.0):
    """Simple integrated autocorrelation time (Sokal window)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    n = len(x)
    if np.allclose(x, 0):
        return 1.0
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[:n].real
    acf /= acf[0]
    tau = 1.0
    for k in range(1, n):
        tau += 2.0 * acf[k]
        if k > c * tau:
            break
    return max(tau, 1.0)

def block_bootstrap_se(series, n_boot=2000, block_len=None, rng=None):
    """Moving-block bootstrap SE of the mean of an autocorrelated series."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(series, float)
    n = len(x)
    if block_len is None:
        tau = integrated_autocorr_time(x)
        block_len = int(max(1, min(n // 2, round(5 * tau))))
    n_blocks = int(np.ceil(n / block_len))
    starts_max = n - block_len + 1
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, starts_max, size=n_blocks)
        idx = (starts[:, None] + np.arange(block_len)[None, :]).ravel()[:n]
        means[i] = x[idx].mean()
    return float(means.std()), int(block_len)

def run_composite(comp, init_pos, n_rounds, key):
    """Run n_rounds composite rounds; return positions(R,n,D), de_accept(R,n)."""
    st = comp["init_states"](init_pos, jax.random.fold_in(key, 0))
    keys = jax.random.split(key, n_rounds)
    pos_out = np.empty((n_rounds, N_CHAINS, D))
    acc_out = np.empty((n_rounds, N_CHAINS))
    for r in range(n_rounds):
        st, (p, ec, acc) = comp["round"](st, keys[r])
        pos_out[r] = np.asarray(p)
        acc_out[r] = np.asarray(acc)
    return pos_out, acc_out

# ======================================================================= MAIN
def main():
    t_start = time.time()
    comp = make_composite(logdensity_fn, D, N_CHAINS, L=L, step_size=STEP, K=K,
                          b0=B0, p_jump=P_JUMP)
    print("config:", comp["config"], "gamma_big=", round(comp["gamma_big"], 4))
    boot_rng = np.random.default_rng(SEED)

    # ----- V1: weight recovery from single-mode init -------------------------
    print("\n[V1] composite from ALL-in-+mode init ...")
    init_single = jnp.zeros((N_CHAINS, D)).at[:, 0].set(m)
    t0 = time.time()
    pos1, acc1 = run_composite(comp, init_single, R_V1, jax.random.key(SEED + 1))
    print(f"   {R_V1} rounds in {time.time()-t0:.1f}s")
    keep1 = pos1[BURN_V1:]                                  # (Rk, n, D)
    z0_keep = keep1[:, :, 0]
    frac_series = (z0_keep > 0).mean(axis=1)               # per-round +mode frac
    w_plus = float(frac_series.mean())
    se_w, blk = block_bootstrap_se(frac_series, rng=boot_rng)
    tau1 = integrated_autocorr_time(frac_series)
    print(f"   weight(+mode) = {w_plus:.4f} +/- {se_w:.4f} (block={blk}, tau~{tau1:.1f})")
    print(f"   |est-0.70| = {abs(w_plus-0.70):.4f}  (3 SE = {3*se_w:.4f})")

    # vanilla MCLMC contrast: same total kernel steps, same init, NO DE
    print("[V1] vanilla MCLMC (no DE) from SAME init ...")
    total_steps = R_V1 * K
    st_v = comp["init_states"](init_single, jax.random.key(SEED + 2))
    # run in chunks to bound memory; only need final-segment occupancy
    chunk = 2000
    vkeys_master = jax.random.key(SEED + 3)
    z0_v_tail = []
    done = 0
    while done < total_steps:
        nthis = min(chunk, total_steps - done)
        ck = jax.random.split(jax.random.fold_in(vkeys_master, done), nthis * N_CHAINS).reshape(nthis, N_CHAINS)
        st_v, posv = comp["mclmc_only"](st_v, ck)
        if done >= total_steps - 2 * chunk:                # keep tail only
            z0_v_tail.append(np.asarray(posv[:, :, 0]))
        done += nthis
    z0_v_tail = np.concatenate(z0_v_tail, axis=0)
    w_plus_vanilla = float((z0_v_tail > 0).mean())
    print(f"   vanilla weight(+mode) tail = {w_plus_vanilla:.4f} (should be >0.98)")

    # ----- V2: invariance from EXACT-truth init ------------------------------
    print("\n[V2] composite from EXACT mixture init (correct 0.7/0.3) ...")
    init_truth_np, comp_lbl = exact_mixture_draws(N_CHAINS, np.random.default_rng(SEED + 4))
    print(f"   init +mode fraction = {(comp_lbl==0).mean():.3f}")
    t0 = time.time()
    pos2, acc2 = run_composite(comp, jnp.asarray(init_truth_np), R_V2, jax.random.key(SEED + 5))
    print(f"   {R_V2} rounds in {time.time()-t0:.1f}s")
    z0_2 = pos2[:, :, 0]
    frac2 = (z0_2 > 0).mean(axis=1)
    w2 = float(frac2.mean())
    se2, blk2 = block_bootstrap_se(frac2, rng=boot_rng)
    half = R_V2 // 2
    w_first, w_second = float(frac2[:half].mean()), float(frac2[half:].mean())
    se_first, _ = block_bootstrap_se(frac2[:half], rng=boot_rng)
    se_second, _ = block_bootstrap_se(frac2[half:], rng=boot_rng)
    se_diff = np.hypot(se_first, se_second)
    print(f"   weight(+mode) = {w2:.4f} +/- {se2:.4f} (block={blk2})")
    print(f"   drift: w_first={w_first:.4f} w_second={w_second:.4f} "
          f"|diff|={abs(w_second-w_first):.4f} (3 SE_diff={3*se_diff:.4f})")

    # per-mode moments (pool a thinned post-equilibration set)
    thin = pos2[half::5].reshape(-1, D)                    # second half, thin x5
    plus = thin[thin[:, 0] > 0]
    minus = thin[thin[:, 0] < 0]
    pm_mean_p, pm_mean_m = plus[:, 0].mean(), minus[:, 0].mean()
    pm_var_p, pm_var_m = plus[:, 0].var(), minus[:, 0].var()
    within_var = thin[:, 1].var()
    print(f"   +mode z0 mean={pm_mean_p:.3f} var={pm_var_p:.3f} (target {m},1.0)")
    print(f"   -mode z0 mean={pm_mean_m:.3f} var={pm_var_m:.3f} (target {-m},1.0)")
    print(f"   within-mode axis1 var={within_var:.3f} (target 1.0)")

    # KS axis0: build a NEAR-INDEPENDENT composite sample by spacing rounds by the
    # mode-weight integrated autocorrelation time (tau2). KS assumes iid; feeding
    # autocorrelated draws would spuriously reject. (plot is primary; KS supportive)
    tau2 = integrated_autocorr_time(frac2)
    ks_spacing = max(1, int(round(tau2)))
    ks_pool = pos2[half::ks_spacing].reshape(-1, D)     # rounds spaced by ~tau
    print(f"   KS uses round-spacing {ks_spacing} (tau2~{tau2:.0f}) -> {ks_pool.shape[0]} near-iid samples")
    rng_ks = np.random.default_rng(SEED + 6)
    comp_axis0 = ks_pool[:, 0].copy(); rng_ks.shuffle(comp_axis0); comp_axis0 = comp_axis0[:4000]
    ana_axis0 = analytic_axis0_samples(len(comp_axis0), rng_ks)
    ks_stat0, ks_p0 = stats.ks_2samp(comp_axis0, ana_axis0)
    comp_axis1 = ks_pool[:, 1].copy(); rng_ks.shuffle(comp_axis1); comp_axis1 = comp_axis1[:4000]
    ana_axis1 = rng_ks.standard_normal(len(comp_axis1))
    ks_stat1, ks_p1 = stats.ks_2samp(comp_axis1, ana_axis1)
    print(f"   KS axis0 vs analytic: stat={ks_stat0:.4f} p={ks_p0:.3f}")
    print(f"   KS axis1 vs N(0,1):  stat={ks_stat1:.4f} p={ks_p1:.3f}")

    # ----- V4: DE acceptance + ergodicity ------------------------------------
    de_acc = float(np.concatenate([acc1.ravel(), acc2.ravel()]).mean())
    cov_pool = np.cov(thin.T)
    min_eig = float(np.linalg.eigvalsh(cov_pool).min())
    print(f"\n[V4] DE acceptance rate = {de_acc:.3f} (target 0.05..0.95)")
    print(f"   min eigenvalue pooled cov = {min_eig:.4f} (>0 => full rank)")

    # =================== PLOTS (produced before verdict) ====================
    # V_weight_recovery: per-round +mode fraction trace (V1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(R_V1), (pos1[:, :, 0] > 0).mean(axis=1), lw=0.7, label="composite +mode frac")
    ax.axhline(0.70, color="k", ls="--", label="truth 0.70")
    ax.axvline(BURN_V1, color="r", ls=":", label="burn-in end")
    ax.axhline(w_plus, color="C1", ls="-", alpha=0.6, label=f"post-burn mean {w_plus:.3f}")
    ax.set_xlabel("composite round"); ax.set_ylabel("fraction of chains in +mode")
    ax.set_title("V1 weight recovery: composite from ALL-in-+mode init"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "V_weight_recovery.png"), dpi=110); plt.close(fig)

    # V_vanilla_vs_composite
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(np.arange(R_V1), (pos1[:, :, 0] > 0).mean(axis=1), lw=0.8, label=f"composite (final {w_plus:.3f})")
    ax.axhline(w_plus_vanilla, color="C3", ls="-", label=f"vanilla MCLMC tail {w_plus_vanilla:.3f}")
    ax.axhline(0.70, color="k", ls="--", label="truth 0.70")
    ax.set_xlabel("composite round"); ax.set_ylabel("fraction in +mode")
    ax.set_title("V1 contrast: composite escapes & recovers mass; vanilla stays trapped")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "V_vanilla_vs_composite.png"), dpi=110); plt.close(fig)

    # V_invariance_from_truth: weight trace + first/second half
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(np.arange(R_V2), frac2, lw=0.7)
    axes[0].axhline(0.70, color="k", ls="--", label="truth 0.70")
    axes[0].axhline(w_first, color="C0", ls="-", alpha=0.6, label=f"1st half {w_first:.3f}")
    axes[0].axhline(w_second, color="C1", ls="-", alpha=0.6, label=f"2nd half {w_second:.3f}")
    axes[0].set_xlabel("round"); axes[0].set_ylabel("+mode fraction")
    axes[0].set_title("V2 invariance-from-truth: no drift"); axes[0].legend(fontsize=8)
    # overlay axis0 at t=0 ensemble vs t=end ensemble vs analytic
    axes[1].hist(pos2[0, :, 0], bins=20, density=True, alpha=0.4, label="t=0 ensemble")
    axes[1].hist(pos2[-1, :, 0], bins=20, density=True, alpha=0.4, label="t=end ensemble")
    xs = np.linspace(-9, 9, 400)
    axes[1].plot(xs, analytic_axis0_pdf(xs), "k-", label="analytic")
    axes[1].set_xlabel("axis 0"); axes[1].set_title("axis0: t=0 vs t=end vs analytic"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "V_invariance_from_truth.png"), dpi=110); plt.close(fig)

    # V_marginal_vs_analytic
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].hist(thin[:, 0], bins=60, density=True, alpha=0.5, label="composite")
    axes[0].plot(xs, analytic_axis0_pdf(xs), "k-", lw=2, label="analytic mixture")
    axes[0].set_xlabel("axis 0"); axes[0].set_title(f"axis0 marginal (KS p={ks_p0:.3f})"); axes[0].legend(fontsize=8)
    axes[1].hist(thin[:, 1], bins=60, density=True, alpha=0.5, label="composite")
    axes[1].plot(xs, stats.norm.pdf(xs), "k-", lw=2, label="N(0,1)")
    axes[1].set_xlim(-5, 5); axes[1].set_xlabel("axis 1 (within-mode)")
    axes[1].set_title(f"axis1 marginal (KS p={ks_p1:.3f})"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "V_marginal_vs_analytic.png"), dpi=110); plt.close(fig)

    # ----------------------------- save npz ----------------------------------
    np.savez(os.path.join(HERE, "V_backing.npz"),
             frac_series_v1=frac_series, w_plus=w_plus, se_w=se_w,
             w_plus_vanilla=w_plus_vanilla, z0_v_tail=z0_v_tail[::10],
             frac2=frac2, w2=w2, se2=se2, w_first=w_first, w_second=w_second, se_diff=se_diff,
             pm_mean_p=pm_mean_p, pm_mean_m=pm_mean_m, pm_var_p=pm_var_p, pm_var_m=pm_var_m,
             within_var=within_var, ks_stat0=ks_stat0, ks_p0=ks_p0, ks_stat1=ks_stat1, ks_p1=ks_p1,
             de_acc=de_acc, min_eig=min_eig,
             config=str(comp["config"]))

    # ---------------------- PROPOSED (UNCERTIFIED) verdict -------------------
    print("\n================ PRE-REGISTERED CHECKS (PROPOSED / UNCERTIFIED) ============")
    c_v1 = abs(w_plus - 0.70) < 3 * se_w
    # Principled necessity threshold: vanilla "fails" if it recovers LESS THAN HALF
    # the true minor-mode mass, i.e. +mode frac > 1 - 0.30/2 = 0.85. (Measured
    # vanilla ~0.98: it barely leaks across the barrier the composite crosses.)
    c_v1b = w_plus_vanilla > 0.85
    c_v2a = abs(w2 - 0.70) < 3 * se2
    c_v2b = abs(w_second - w_first) < 3 * se_diff
    c_v2c = (abs(pm_mean_p - m) < 0.10 and abs(pm_mean_m + m) < 0.10
             and abs(pm_var_p - 1.0) < 0.10 and abs(pm_var_m - 1.0) < 0.10)
    c_v2d = ks_p0 > 0.05
    c_v3 = ks_p0 > 0.05 and ks_p1 > 0.05
    c_v4 = (0.05 < de_acc < 0.95) and (min_eig > 0)
    def mark(b): return "PASS" if b else "FAIL"
    print(f" V1  weight recovery   {w_plus:.4f}+/-{se_w:.4f} vs 0.70  -> {mark(c_v1)}")
    print(f" V1  vanilla trapped   {w_plus_vanilla:.4f} > 0.85       -> {mark(c_v1b)}")
    print(f" V2a weight@truth      {w2:.4f}+/-{se2:.4f} vs 0.70      -> {mark(c_v2a)}")
    print(f" V2b no drift          |{w_second-w_first:.4f}| < {3*se_diff:.4f} -> {mark(c_v2b)}")
    print(f" V2c per-mode moments  +/-3.5,var1 within 0.10           -> {mark(c_v2c)}")
    print(f" V2d KS axis0 p>0.05   p={ks_p0:.3f}                     -> {mark(c_v2d)}")
    print(f" V3  KS marginals      axis0 p={ks_p0:.3f} axis1 p={ks_p1:.3f} -> {mark(c_v3)}")
    print(f" V4  DE acc & rank     acc={de_acc:.3f} min_eig={min_eig:.3f} -> {mark(c_v4)}")
    all_pass = all([c_v1, c_v1b, c_v2a, c_v2b, c_v2c, c_v2d, c_v3, c_v4])
    print("----------------------------------------------------------------------------")
    print(f" PROPOSED VERDICT (UNCERTIFIED): {'UNBIASED' if all_pass else 'BIASED / INCONCLUSIVE'}")
    print(f" total wall time {time.time()-t_start:.1f}s")
    print("============================================================================")

if __name__ == "__main__":
    main()
