"""GATE A (discovery) + GATE C (unbiased cold) for TEMPERED BURN-IN on the EASY
analytic bimodal mixture (D=10, modes +/-5 axis0, weights [0.7,0.3], barrier
m^2/2 = 12.5 -> vanilla MCLMC trapped, per validate_analytic).

PRE-REGISTRATION
================
CAUSE HYPOTHESIS: at small beta the tempered barrier beta*12.5 is crossable, so
MCLMC's gradient dynamics move chains between modes; cooling beta: small->1 lets
the ensemble track the tempered between-mode weight w0^beta/(w0^beta+w1^beta),
which sharpens toward the true 0.70 as beta->1. A chain started in the WRONG
(minor, -mode 0.30) basin will therefore DISCOVER the dominant (+mode 0.70)
basin -- a capability vanilla MCLMC and every ensemble mode-hop lack.

GATE A (discovery from wrong basin):  ALL chains init in the MINOR (-mode, 0.30)
basin. PREDICTION: cold +mode occupancy ~ 0.70 (direction: from 0.0 to ~0.70).
THRESHOLD: |occ_+ - 0.70| < 3*SE where SE is the BETWEEN-REALIZATION SE over
independent anneal seeds (the honest uncertainty of a one-shot annealed weight;
a block-bootstrap-over-cold-rounds SE is ~0 because the cold ensemble is FROZEN
-- chains cannot swap modes at beta=1 -- and would be an ill-posed test, so we
do NOT use it for the between-mode weight).
CONTRAST: vanilla MCLMC (beta=1) from the SAME init stays trapped, occ_+ < 0.02.
KNOWN CAVEAT (pre-stated): a one-shot anneal sets the weight near the FREEZE-OUT
temperature beta_f (where the barrier re-rises beyond crossing), so occ_+ may be
biased toward 0.5 (tempered weight at beta_f < 0.70). We MEASURE the gap; if it
exceeds the between-realization SE, the documented fix is replica-exchange PT
(the cold replica keeps receiving correctly-weighted configs -> no freeze-out
bias). Falsifier of discovery: occ_+ stays ~0 (no crossing) OR ~0.5 (annealed
but weight not re-sharpened at all).

GATE C (unbiased cold sampling): init from EXACT mixture draws (0.70/0.30) at
beta=1. WITHIN each mode the cold kernel must be unbiased: per-mode axis0
mean +/-5 & var 1, within-mode axis1 var 1, axis0 marginal KS vs analytic
mixture p>0.05. (Between-mode weight is preserved trivially because the cold
ensemble is frozen; this gate tests the WITHIN-mode invariance of the cold
kernel at the chosen step.)

METHOD: plots BEFORE the numeric verdict; verdict PROPOSED/UNCERTIFIED.
EEVPD note: easy mixture is benign-Gaussian; energy error does not bind in
step<=1 (EEVPD(step=0.5)=1.5e-9 << 5e-4 target, see tune_easy.log). step=0.5
matches the validated validate_analytic config and is deeply EEVPD-conservative.
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
from tempered_mclmc import make_tempered_sampler

D, m = 10, 5.0
W = np.array([0.7, 0.3]); MU = np.array([+m, -m])
_logW = jnp.log(jnp.asarray(W)); _MU = jnp.asarray(MU); _c = -0.5 * D * jnp.log(2 * jnp.pi)

def logdensity_fn(z):
    z0 = z[0]; qr = jnp.sum(z[1:] ** 2)
    c0 = _logW[0] + _c - 0.5 * ((z0 - _MU[0]) ** 2 + qr)
    c1 = _logW[1] + _c - 0.5 * ((z0 - _MU[1]) ** 2 + qr)
    return jax.scipy.special.logsumexp(jnp.stack([c0, c1]))

def analytic_axis0_pdf(x):
    return W[0]*stats.norm.pdf(x, MU[0], 1.0) + W[1]*stats.norm.pdf(x, MU[1], 1.0)

def tempered_weight_plus(beta):
    a = W[0]**beta; b = W[1]**beta
    return a/(a+b)

# config
N_CHAINS, L, STEP = 64, 2.0, 0.5
BETAS = np.geomspace(0.04, 1.0, 15)
STEPS_PER_BETA = 200
COLD_STEPS = 600
N_SEEDS = 6
SEED = 20260628

def main():
    t0 = time.time()
    samp = make_tempered_sampler(logdensity_fn, D, N_CHAINS, BETAS, L=L, step_size=STEP)
    print("config:", samp["config"], "n_betas:", len(BETAS), flush=True)

    # ---------------- GATE A: discovery from WRONG (minor -mode) basin --------
    print("\n[A] tempered burn-in from ALL-in-(-mode) init, N_SEEDS anneals ...", flush=True)
    init_wrong = np.zeros((N_CHAINS, D)); init_wrong[:, 0] = MU[1]   # all in -mode (0.30)
    occ_plus_seeds = []
    cold_traces = []
    one_trace = None
    for s in range(N_SEEDS):
        key = jax.random.key(SEED + 100 + s)
        ka, kc = jax.random.split(key)
        final_pos, trace = samp["anneal"](jnp.asarray(init_wrong), ka, STEPS_PER_BETA)
        if one_trace is None:
            one_trace = trace
        # cold sampling phase (confirm frozen + within-mode mixing)
        _, cold_pos, _ = samp["sample_cold"](final_pos, kc, COLD_STEPS)
        cold_pos = np.asarray(cold_pos)           # (T,n,D)
        occ = (cold_pos[:, :, 0] > 0).mean()       # cold +mode occupancy (time+chain)
        occ_plus_seeds.append(float(occ))
        cold_traces.append((cold_pos[:, :, 0] > 0).mean(axis=1))  # per-round frac
        print(f"   seed {s}: cold occ_+ = {occ:.4f}", flush=True)
    occ_plus_seeds = np.asarray(occ_plus_seeds)
    occA = float(occ_plus_seeds.mean())
    seA = float(occ_plus_seeds.std(ddof=1)/np.sqrt(N_SEEDS)) if N_SEEDS > 1 else float("nan")
    # also report between-chain binomial SE for a single anneal
    se_binom = float(np.sqrt(occA*(1-occA)/N_CHAINS))
    print(f"   GATE A: occ_+ = {occA:.4f} +/- {seA:.4f} (between-seed SE; "
          f"binomial SE/anneal {se_binom:.4f})")
    print(f"   |occ_+ - 0.70| = {abs(occA-0.70):.4f}  (3*SE_seed = {3*seA:.4f})")

    # ---------------- vanilla contrast (beta=1, same init) -------------------
    print("\n[A] vanilla MCLMC (beta=1) from SAME wrong init ...", flush=True)
    total_v = len(BETAS)*STEPS_PER_BETA + COLD_STEPS
    kv = jax.random.key(SEED + 7)
    _, vpos, _ = samp["sample_cold"](jnp.asarray(init_wrong), kv, total_v)
    vpos = np.asarray(vpos)
    occ_v = float((vpos[-COLD_STEPS:, :, 0] > 0).mean())
    print(f"   vanilla occ_+ tail = {occ_v:.4f} (should be < 0.02)")

    # ---------------- GATE C: invariance from EXACT-truth init ---------------
    print("\n[C] cold sampling from EXACT mixture init (0.70/0.30) ...", flush=True)
    rng = np.random.default_rng(SEED + 4)
    comp = (rng.random(N_CHAINS) >= W[0]).astype(int)
    z_truth = rng.standard_normal((N_CHAINS, D)); z_truth[:, 0] += MU[comp]
    kC = jax.random.key(SEED + 5)
    _, cpos, _ = samp["sample_cold"](jnp.asarray(z_truth), kC, 1200)
    cpos = np.asarray(cpos)                          # (T,n,D)
    fracC = (cpos[:, :, 0] > 0).mean(axis=1)
    half = cpos.shape[0]//2
    occC = float(fracC.mean()); occC1, occC2 = float(fracC[:half].mean()), float(fracC[half:].mean())
    thin = cpos[half::5].reshape(-1, D)
    plus = thin[thin[:, 0] > 0]; minus = thin[thin[:, 0] < 0]
    pmean_p, pvar_p = float(plus[:, 0].mean()), float(plus[:, 0].var())
    pmean_m, pvar_m = float(minus[:, 0].mean()), float(minus[:, 0].var())
    within_var = float(thin[:, 1].var())
    rng_ks = np.random.default_rng(SEED + 6)
    samp0 = thin[:, 0].copy(); rng_ks.shuffle(samp0); samp0 = samp0[:4000]
    cc = (rng_ks.random(len(samp0)) >= W[0]).astype(int); ana0 = rng_ks.standard_normal(len(samp0)) + MU[cc]
    ks0_s, ks0_p = stats.ks_2samp(samp0, ana0)
    samp1 = thin[:, 1].copy(); rng_ks.shuffle(samp1); samp1 = samp1[:4000]
    ks1_s, ks1_p = stats.ks_2samp(samp1, rng_ks.standard_normal(len(samp1)))
    print(f"   occ_+ = {occC:.4f}  (init {(comp==0).mean():.4f}); drift "
          f"|{occC2-occC1:.4f}|")
    print(f"   +mode mean {pmean_p:.3f} var {pvar_p:.3f}; -mode mean {pmean_m:.3f} var {pvar_m:.3f}")
    print(f"   within axis1 var {within_var:.3f}; KS axis0 p={ks0_p:.3f} axis1 p={ks1_p:.3f}")

    # ----------------------------- SAVE npz FIRST ----------------------------
    np.savez(os.path.join(HERE, "easy_discovery.npz"),
             betas=BETAS, occ_plus_seeds=occ_plus_seeds, occA=occA, seA=seA,
             se_binom=se_binom, occ_v=occ_v, stage_pos=one_trace["stage_pos"],
             stage_eevpd=one_trace["stage_eevpd"], vpos_tail=vpos[-COLD_STEPS::5, :, 0],
             fracC=fracC, occC=occC, occC1=occC1, occC2=occC2,
             pmean_p=pmean_p, pvar_p=pvar_p, pmean_m=pmean_m, pvar_m=pvar_m,
             within_var=within_var, ks0_p=ks0_p, ks1_p=ks1_p,
             cold_axis0=thin[::3, 0], cold_axis1=thin[::3, 1])

    # ============================ PLOTS (before verdict) =====================
    # 1. anneal trajectory: +frac per stage vs tempered-weight curve
    stage_fracplus = (one_trace["stage_pos"][:, :, 0] > 0).mean(axis=1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(BETAS, stage_fracplus, "o-", label="ensemble +mode frac (per stage)")
    ax.plot(BETAS, [tempered_weight_plus(b) for b in BETAS], "k--",
            label=r"tempered weight $w_+^\beta/(w_+^\beta+w_-^\beta)$")
    ax.axhline(0.70, color="C2", ls=":", label="true weight 0.70")
    ax.set_xlabel(r"$\beta$ (cooling stage)"); ax.set_ylabel("+mode fraction")
    ax.set_title("Gate A: anneal tracks tempered weight; discovers +mode from -mode init")
    ax.legend(fontsize=8); ax.set_xscale("log")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "A_anneal_trajectory.png"), dpi=110); plt.close(fig)

    # 2. discovery contrast
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(["tempered\n(cold occ_+)", "vanilla\n(cold occ_+)"], [occA, occ_v],
           yerr=[3*seA if np.isfinite(seA) else 0, 0], color=["C0", "C3"], alpha=0.8)
    ax.axhline(0.70, color="k", ls="--", label="truth 0.70")
    ax.set_ylabel("+mode occupancy"); ax.set_ylim(0, 1)
    ax.set_title("Gate A: tempered discovers dominant basin; vanilla stays trapped")
    ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(HERE, "A_discovery_contrast.png"), dpi=110); plt.close(fig)

    # 3. invariance-from-truth
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    axes[0].plot(fracC, lw=0.7); axes[0].axhline(0.70, color="k", ls="--", label="truth 0.70")
    axes[0].axhline(occC1, color="C0", alpha=.6, label=f"1st half {occC1:.3f}")
    axes[0].axhline(occC2, color="C1", alpha=.6, label=f"2nd half {occC2:.3f}")
    axes[0].set_xlabel("cold round"); axes[0].set_ylabel("+mode frac")
    axes[0].set_title("Gate C: invariance-from-truth (frozen, no drift)"); axes[0].legend(fontsize=8)
    xs = np.linspace(-9, 9, 400)
    axes[1].hist(thin[:, 0], bins=60, density=True, alpha=0.5, label="cold samples")
    axes[1].plot(xs, analytic_axis0_pdf(xs), "k-", lw=2, label="analytic mixture")
    axes[1].set_xlabel("axis 0"); axes[1].set_title(f"axis0 marginal (KS p={ks0_p:.3f})"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "C_invariance.png"), dpi=110); plt.close(fig)

    # ============================ PROPOSED verdict ===========================
    print("\n========== PRE-REGISTERED CHECKS (PROPOSED / UNCERTIFIED) ==========", flush=True)
    cA = abs(occA - 0.70) < 3*seA
    cA_contrast = occ_v < 0.02
    cA_discovered = occA > 0.5     # reached dominant basin at all (vs trapped ~0)
    cC_moments = (abs(pmean_p-m) < 0.10 and abs(pmean_m+m) < 0.10 and
                  abs(pvar_p-1) < 0.10 and abs(pvar_m-1) < 0.10 and abs(within_var-1) < 0.10)
    cC_drift = abs(occC2-occC1) < 0.05
    cC_ks = ks0_p > 0.05 and ks1_p > 0.05
    def mk(b): return "PASS" if b else "FAIL"
    print(f" A discovered dominant basin   occ_+={occA:.4f} > 0.5     -> {mk(cA_discovered)}")
    print(f" A weight within 3 SE          |{occA-0.70:.4f}| < {3*seA:.4f} -> {mk(cA)}")
    print(f" A vanilla trapped             occ_+={occ_v:.4f} < 0.02   -> {mk(cA_contrast)}")
    print(f" C within-mode moments         +/-5,var1                  -> {mk(cC_moments)}")
    print(f" C no drift                    |{occC2-occC1:.4f}| < 0.05  -> {mk(cC_drift)}")
    print(f" C KS marginals                axis0 {ks0_p:.3f} axis1 {ks1_p:.3f} -> {mk(cC_ks)}")
    print("--------------------------------------------------------------------")
    print(f" wall {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
