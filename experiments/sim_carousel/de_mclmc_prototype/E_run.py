"""DE-MCLMC composite on the CAROUSEL-FAITHFUL analytic testbed.

GATE 1 (pre-registration): reproduce the carousel's LOW DE acceptance (~0.6%) and
slow mode-mixing with the CURRENT settings (Cholesky jitter b0=0.1, p_jump=0.3,
16 chains, honest global-mode mass matrix, BALANCED init). If acceptance is NOT low
(<5%), the testbed is NOT faithful -> STOP (do not fix a non-problem).

SWEEP (only after Gate 1 passes):
  1. JITTER:  b0 in {0, 0.05, 0.5} at p_jump=0.3.
  2. P_JUMP:  {0.3, 1.0} at best b0.
For each: DE acceptance, mode-mixing (round-trips from balanced init; weight->0.3/0.7),
and UNBIASEDNESS (weight recovery, invariance-from-truth, KS marginals).

Pre-registered hypotheses (cause / prediction / falsifier):
  H-jitter: the Cholesky jitter eps~N(0,b0^2 Cg) is UNNECESSARY (MCLMC already supplies
    within-mode ergodicity) and HURTS the DE accept because it adds an O(b0) perturbation
    to an otherwise-exact difference-vector jump landing on the other mode.
    Prediction: b0=0 RAISES DE acceptance vs b0=0.1; b0=0.5 LOWERS it further.
    Falsifier: b0=0 gives acceptance <= the b0=0.1 baseline (jitter is not the killer).
  H-pjump: the gamma_big (=2.38/sqrt(2D)) LOCAL DE moves use cross-mode pairs and propose
    multi-sigma steps that MCLMC already covers, so they are rejected and DILUTE the
    average DE acceptance. p_jump=1.0 (jumps only) should RAISE average DE acceptance.
    Falsifier: p_jump=1.0 does not raise acceptance vs p_jump=0.3.

ALL VERDICTS ARE PROPOSED / UNCERTIFIED. CPU, synchronous.
"""
import os, sys, time, json
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import carousel_testbed as T
from de_mclmc import make_composite
from validate_analytic import integrated_autocorr_time, block_bootstrap_se

NCH = 16
K = 20
SEED = 20260627


def pr(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ build target
mix, info = T.build_target()
Cg = jnp.asarray(info["cov"][1])           # honest GLOBAL-mode covariance
Lg = jnp.linalg.cholesky(Cg)               # Cholesky factor (eps_scale for jitter)
mu_np = info["mu"]                          # (2,D) [sec, glob]


def make_comp(b0, p_jump, jitter="chol"):
    if jitter == "none":
        eps_scale = None       # ignored when b0=0 anyway
    elif jitter == "chol":
        eps_scale = Lg
    else:
        raise ValueError(jitter)
    return make_composite(mix.logp, T.D, NCH, L=T.L0, step_size=T.SS0, K=K,
                          b0=b0, p_jump=p_jump, inverse_mass_matrix=Cg,
                          eps_scale=eps_scale)


# ------------------------------------------------------------------ run + measure
def run_composite(comp, init_pos, n_rounds, seed):
    st = comp["init_states"](jnp.asarray(init_pos), jax.random.key(seed))
    keys = jax.random.split(jax.random.key(seed + 7), n_rounds)
    pos = np.empty((n_rounds, NCH, T.D))
    acc = np.empty((n_rounds, NCH))
    for r in range(n_rounds):
        st, (p, ec, a) = comp["round"](st, keys[r])
        pos[r] = np.asarray(p)
        acc[r] = np.asarray(a)
    return pos, acc


def round_trips(mode_series):
    """mode_series: (R, C) hard mode labels (0/1). per-chain min(up,down) transitions."""
    R, C = mode_series.shape
    nrt = np.zeros(C, int)
    ncross = np.zeros(C, int)
    for c in range(C):
        d = np.diff(mode_series[:, c])
        ups = int(np.sum(d == 1)); dns = int(np.sum(d == -1))
        nrt[c] = min(ups, dns)
        ncross[c] = int(np.sum(np.abs(d)))
    return nrt, ncross


def weight_series(pos):
    """fraction of chains classified as GLOBAL (mode 1) per round."""
    modes = mix.classify(pos)               # (R, C)
    return modes, modes.mean(axis=1)        # weight(global) per round


# initial-condition factories ---------------------------------------------------
def init_balanced(seed):
    """8 secondary + 8 global EXACT draws, interleaved (each two-group half straddles
    both modes -> difference vectors can span the gap)."""
    rng = np.random.default_rng(seed)
    # exact per-mode draws (lower-Cholesky: z = mu + L @ xi  ->  rows: mu + xi @ L.T)
    std_s = rng.standard_normal((NCH // 2, T.D))
    std_g = rng.standard_normal((NCH // 2, T.D))
    a = mu_np[0] + std_s @ np.asarray(mix._chol_np[0]).T
    b = mu_np[1] + std_g @ np.asarray(mix._chol_np[1]).T
    out = np.empty((NCH, T.D))
    out[0::2] = a
    out[1::2] = b
    return out


def init_single(mode, seed):
    """all NCH chains as exact draws from ONE mode (0=sec,1=glob)."""
    rng = np.random.default_rng(seed)
    std = rng.standard_normal((NCH, T.D))
    return mu_np[mode] + std @ np.asarray(mix._chol_np[mode]).T


def init_truth(seed):
    """exact mixture draws with deterministic 0.3/0.7 counts (no init transient)."""
    rng = np.random.default_rng(seed)
    z, comp = mix.exact_draws_balanced(NCH, rng)
    return z


# ------------------------------------------------------------------ measurements
def measure_mixing(comp, n_rounds, seed, init):
    pos, acc = run_composite(comp, init, n_rounds, seed)
    modes, w = weight_series(pos)
    nrt, ncross = round_trips(modes)
    de_acc = float(acc.mean())
    burn = n_rounds // 2
    w_mean = float(w[burn:].mean())
    return dict(de_acc=de_acc, w_series=w, w_mean=w_mean, nrt=nrt, ncross=ncross,
                total_rt=int(nrt.sum()), total_cross=int(ncross.sum()),
                pos=pos, acc=acc, modes=modes)


def measure_invariance(comp, n_rounds, seed, boot_rng):
    """init from EXACT truth (0.3/0.7); confirm NO drift in weight & per-mode moments."""
    pos, acc = run_composite(comp, init_truth(seed), n_rounds, seed)
    modes, w = weight_series(pos)
    w_mean = float(w.mean())
    se_w, blk = block_bootstrap_se(w, rng=boot_rng)
    half = n_rounds // 2
    w1, w2 = float(w[:half].mean()), float(w[half:].mean())
    se1, _ = block_bootstrap_se(w[:half], rng=boot_rng)
    se2, _ = block_bootstrap_se(w[half:], rng=boot_rng)
    se_diff = float(np.hypot(se1, se2))
    # per-mode moments (second half, pooled, classified)
    pos2 = pos[half:].reshape(-1, T.D)
    m2 = mix.classify(pos2)
    moments = {}
    for k, name in [(0, "sec"), (1, "glob")]:
        sub = pos2[m2 == k]
        if sub.shape[0] < 10:
            moments[name] = dict(n=int(sub.shape[0]), mean_err=np.nan, cov_relerr=np.nan)
            continue
        emu = sub.mean(0)
        ecov = np.cov(sub.T)
        # whiten by true cov: error metrics in stiff metric units
        Li = np.linalg.inv(np.asarray(mix._chol_np[k]))
        mean_err = float(np.sqrt(((Li @ (emu - mu_np[k])) ** 2).sum()))   # sigma units
        Mw = Li @ ecov @ Li.T                                             # ~I if correct
        cov_relerr = float(np.linalg.norm(Mw - np.eye(T.D)) / np.sqrt(T.D))
        moments[name] = dict(n=int(sub.shape[0]), mean_err=mean_err, cov_relerr=cov_relerr)
    return dict(w_mean=w_mean, se_w=se_w, w1=w1, w2=w2, se_diff=se_diff,
                w_series=w, moments=moments, pos=pos)


def measure_ks(comp, n_rounds, seed, boot_rng, n_exact=8000):
    """KS of composite marginals (from truth-init, post-burn, thinned by tau) vs exact
    mixture draws, on every axis. Returns min p, per-axis p, and pooled samples."""
    pos, acc = run_composite(comp, init_truth(seed), n_rounds, seed)
    modes, w = weight_series(pos)
    tau = integrated_autocorr_time(w)
    spacing = max(1, int(round(tau)))
    half = n_rounds // 2
    pool = pos[half::spacing].reshape(-1, T.D)
    rng = np.random.default_rng(seed + 1)
    if pool.shape[0] > n_exact:
        idx = rng.choice(pool.shape[0], n_exact, replace=False)
        pool = pool[idx]
    exact, _ = mix.exact_draws(max(pool.shape[0], 2000), rng)
    ps = np.empty(T.D); stats_ = np.empty(T.D)
    for j in range(T.D):
        s, p = stats.ks_2samp(pool[:, j], exact[:, j])
        stats_[j] = s; ps[j] = p
    return dict(ks_p=ps, ks_stat=stats_, min_p=float(ps.min()),
                argmin=int(ps.argmin()), tau=float(tau), spacing=spacing,
                pool=pool, exact=exact)


# =============================================================================
def main():
    t_all = time.time()
    boot_rng = np.random.default_rng(SEED)
    store = {}
    R_MIX = 2000          # mixing / Gate-1 run length
    R_INV = 4000          # invariance-from-truth (tight weight SE)

    # ---------------- GATE 1: reproduce low DE acceptance --------------------
    pr("\n================= GATE 1: reproduce the carousel failure =================")
    pr("config: Cholesky jitter b0=0.1, p_jump=0.3, 16 chains, honest global MM, balanced init")
    comp0 = make_comp(b0=0.1, p_jump=0.3, jitter="chol")
    g1 = measure_mixing(comp0, R_MIX, SEED, init_balanced(SEED))
    pr(f"  DE acceptance = {g1['de_acc']*100:.3f}%   (carousel ~0.6%; GATE: must be <5%)")
    pr(f"  round-trips/chain total = {g1['total_rt']}  (crossings total = {g1['total_cross']})")
    pr(f"  post-burn weight(global) = {g1['w_mean']:.3f}  (target 0.70)")
    gate1_pass = g1['de_acc'] < 0.05
    pr(f"  GATE 1 {'PASS (testbed reproduces low acceptance)' if gate1_pass else 'FAIL (testbed NOT faithful -> STOP)'}")
    store["gate1_de_acc"] = g1['de_acc']
    store["gate1_w_series"] = g1['w_series']
    store["gate1_acc_series"] = g1['acc'].mean(axis=1)
    store["gate1_total_rt"] = g1['total_rt']

    # E_gate1 plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
    ax[0].plot(g1['acc'].mean(axis=1), lw=0.6)
    ax[0].axhline(g1['de_acc'], color="C1", label=f"mean {g1['de_acc']*100:.2f}%")
    ax[0].axhline(0.05, color="r", ls="--", label="5% gate")
    ax[0].set_xlabel("round"); ax[0].set_ylabel("DE acceptance"); ax[0].legend(fontsize=8)
    ax[0].set_title("GATE 1: DE acceptance (b0=0.1 chol, p_jump=0.3)")
    ax[1].plot(g1['w_series'], lw=0.6, label="weight(global)")
    ax[1].axhline(0.70, color="k", ls="--", label="target 0.70")
    ax[1].set_xlabel("round"); ax[1].set_ylabel("fraction chains in global mode")
    ax[1].set_title(f"GATE 1 mixing: round-trips={g1['total_rt']}"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "E_gate1.png"), dpi=110); plt.close(fig)

    if not gate1_pass:
        pr("\nGate 1 did NOT reproduce low acceptance. The testbed is not faithful; STOPPING.")
        np.savez(os.path.join(HERE, "E_backing.npz"), **{k: v for k, v in store.items()
                 if isinstance(v, (np.ndarray, float, int))})
        return

    # ---------------- SWEEP 1: jitter scale at p_jump=0.3 --------------------
    pr("\n================= SWEEP 1: jitter scale (p_jump=0.3) =================")
    jitter_cfgs = [("b0=0(none)", 0.0, "none"),
                   ("b0=0.05(chol)", 0.05, "chol"),
                   ("b0=0.1(chol,Gate1)", 0.1, "chol"),
                   ("b0=0.5(chol)", 0.5, "chol")]
    sweep1 = []
    for name, b0, jit in jitter_cfgs:
        comp = make_comp(b0=b0, p_jump=0.3, jitter=jit)
        m = measure_mixing(comp, R_MIX, SEED, init_balanced(SEED))
        sweep1.append((name, b0, m))
        pr(f"  {name:22s} DE acc={m['de_acc']*100:6.3f}%  round-trips={m['total_rt']:4d}  "
           f"cross={m['total_cross']:4d}  w(glob)={m['w_mean']:.3f}")
    store["sweep1_b0"] = np.array([c[1] for c in sweep1])
    store["sweep1_acc"] = np.array([c[2]['de_acc'] for c in sweep1])
    store["sweep1_rt"] = np.array([c[2]['total_rt'] for c in sweep1])
    best_b0 = sweep1[int(np.argmax([c[2]['de_acc'] for c in sweep1]))][1]
    best_jit = "none" if best_b0 == 0.0 else "chol"
    pr(f"  -> best jitter by DE acc: b0={best_b0}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    bb = [c[1] for c in sweep1]; aa = [c[2]['de_acc']*100 for c in sweep1]; rr = [c[2]['total_rt'] for c in sweep1]
    ax[0].plot(bb, aa, "o-"); ax[0].set_xlabel("jitter b0"); ax[0].set_ylabel("DE acceptance %")
    ax[0].set_title("DE acceptance vs jitter (p_jump=0.3)")
    for x, y, n in zip(bb, aa, [c[0] for c in sweep1]): ax[0].annotate(f"{y:.2f}%", (x, y), fontsize=7)
    ax[1].plot(bb, rr, "s-", color="C2"); ax[1].set_xlabel("jitter b0"); ax[1].set_ylabel("round-trips (16 chains)")
    ax[1].set_title("mode-mixing vs jitter")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "E_acc_vs_jitter.png"), dpi=110); plt.close(fig)

    # ---------------- SWEEP 2: p_jump at best jitter ------------------------
    pr(f"\n================= SWEEP 2: p_jump (b0={best_b0}) =================")
    sweep2 = []
    for p_jump in [0.3, 1.0]:
        comp = make_comp(b0=best_b0, p_jump=p_jump, jitter=best_jit)
        m = measure_mixing(comp, R_MIX, SEED, init_balanced(SEED))
        sweep2.append((p_jump, m))
        pr(f"  p_jump={p_jump:.1f}  DE acc={m['de_acc']*100:6.3f}%  round-trips={m['total_rt']:4d}  "
           f"cross={m['total_cross']:4d}  w(glob)={m['w_mean']:.3f}")
    store["sweep2_pjump"] = np.array([c[0] for c in sweep2])
    store["sweep2_acc"] = np.array([c[1]['de_acc'] for c in sweep2])
    store["sweep2_rt"] = np.array([c[1]['total_rt'] for c in sweep2])
    best_pjump = sweep2[int(np.argmax([c[1]['de_acc'] for c in sweep2]))][0]
    pr(f"  -> best p_jump by DE acc: {best_pjump}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    pp = [c[0] for c in sweep2]; aa = [c[1]['de_acc']*100 for c in sweep2]; rr = [c[1]['total_rt'] for c in sweep2]
    ax[0].plot(pp, aa, "o-"); ax[0].set_xlabel("p_jump"); ax[0].set_ylabel("DE acceptance %")
    ax[0].set_title(f"DE acceptance vs p_jump (b0={best_b0})")
    ax[1].plot(pp, rr, "s-", color="C2"); ax[1].set_xlabel("p_jump"); ax[1].set_ylabel("round-trips")
    ax[1].set_title("mode-mixing vs p_jump")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "E_acc_vs_pjump.png"), dpi=110); plt.close(fig)

    # ---------------- RECOMMENDED CONFIG + unbiasedness ---------------------
    pr(f"\n================= RECOMMENDED CONFIG: b0={best_b0}, p_jump={best_pjump} =================")
    rec = make_comp(b0=best_b0, p_jump=best_pjump, jitter=best_jit)

    # mixing (balanced init) + weight recovery (single-mode init)
    rec_mix = measure_mixing(rec, R_MIX, SEED, init_balanced(SEED))
    pr(f"  [mixing/balanced] DE acc={rec_mix['de_acc']*100:.3f}%  round-trips={rec_mix['total_rt']}  "
       f"w(glob)={rec_mix['w_mean']:.3f}")
    rec_rec_sec = measure_mixing(rec, R_MIX, SEED + 11, init_single(0, SEED + 11))   # all-secondary init
    se_rec, _ = block_bootstrap_se(rec_rec_sec['w_series'][R_MIX//2:], rng=boot_rng)
    pr(f"  [weight recovery/all-sec init] post-burn w(glob)={rec_rec_sec['w_mean']:.3f} +/- {se_rec:.3f}  (target 0.70)")

    # invariance-from-truth
    inv = measure_invariance(rec, R_INV, SEED + 21, boot_rng)
    pr(f"  [invariance-from-truth] w={inv['w_mean']:.3f}+/-{inv['se_w']:.3f}  "
       f"drift |w2-w1|={abs(inv['w2']-inv['w1']):.4f} (3 SE_diff={3*inv['se_diff']:.4f})")
    for name in ["sec", "glob"]:
        mo = inv['moments'][name]
        pr(f"     {name}: mean_err={mo['mean_err']:.3f} sigma  cov_relerr={mo['cov_relerr']:.3f}  n={mo['n']}")

    # KS marginals
    ks = measure_ks(rec, R_INV, SEED + 31, boot_rng)
    pr(f"  [KS marginals vs exact] min p={ks['min_p']:.3f} on axis {ks['argmin']} (tau~{ks['tau']:.0f}, spacing {ks['spacing']})")

    # ---- unbiasedness verdict (PROPOSED / UNCERTIFIED) ----
    c_inv_w = abs(inv['w_mean'] - 0.70) < 3 * inv['se_w']
    c_inv_drift = abs(inv['w2'] - inv['w1']) < 3 * inv['se_diff']
    c_inv_mom = all(inv['moments'][n]['mean_err'] < 0.15 and inv['moments'][n]['cov_relerr'] < 0.15
                    for n in ["sec", "glob"])
    c_ks = ks['min_p'] > 0.01
    c_recov = abs(rec_rec_sec['w_mean'] - 0.70) < max(3 * se_rec, 0.05)
    pr("\n  --- UNBIASEDNESS GATE (PROPOSED / UNCERTIFIED) ---")
    def mk(b): return "PASS" if b else "FAIL"
    pr(f"   weight@truth   {inv['w_mean']:.3f} vs 0.70           -> {mk(c_inv_w)}")
    pr(f"   no drift       |{inv['w2']-inv['w1']:.4f}|<{3*inv['se_diff']:.4f}  -> {mk(c_inv_drift)}")
    pr(f"   per-mode moms  mean&cov err<0.15 sigma         -> {mk(c_inv_mom)}")
    pr(f"   KS marginals   min p={ks['min_p']:.3f}>0.01          -> {mk(c_ks)}")
    pr(f"   weight recovery {rec_rec_sec['w_mean']:.3f} vs 0.70 (all-sec init) -> {mk(c_recov)}")
    unbiased = all([c_inv_w, c_inv_drift, c_inv_mom, c_ks])
    pr(f"   PROPOSED: {'UNBIASED' if unbiased else 'BIASED / INCONCLUSIVE'}")

    # ---- plots: unbiasedness + mixing ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    ax[0].plot(rec_rec_sec['w_series'], lw=0.6, label="all-sec init")
    ax[0].axhline(0.70, color="k", ls="--", label="target 0.70")
    ax[0].axhline(rec_rec_sec['w_mean'], color="C1", alpha=0.6, label=f"post-burn {rec_rec_sec['w_mean']:.3f}")
    ax[0].set_xlabel("round"); ax[0].set_ylabel("w(global)"); ax[0].legend(fontsize=8)
    ax[0].set_title("weight recovery (single-mode init)")
    ax[1].plot(inv['w_series'], lw=0.5)
    ax[1].axhline(0.70, color="k", ls="--"); ax[1].axhline(inv['w1'], color="C0", alpha=.6, label=f"1st half {inv['w1']:.3f}")
    ax[1].axhline(inv['w2'], color="C1", alpha=.6, label=f"2nd half {inv['w2']:.3f}")
    ax[1].set_xlabel("round"); ax[1].set_ylabel("w(global)"); ax[1].legend(fontsize=8)
    ax[1].set_title("invariance-from-truth: no drift")
    # KS: worst axis overlay
    j = ks['argmin']
    ax[2].hist(ks['pool'][:, j], bins=50, density=True, alpha=0.5, label="composite")
    ax[2].hist(ks['exact'][:, j], bins=50, density=True, alpha=0.5, label="exact")
    ax[2].set_xlabel(f"axis {j} (worst KS, p={ks['min_p']:.3f})"); ax[2].legend(fontsize=8)
    ax[2].set_title("marginal vs analytic (worst axis)")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "E_unbiasedness.png"), dpi=110); plt.close(fig)

    # mixing trajectory plot (recommended config, balanced init): per-chain mode
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.plot(rec_mix['w_series'], lw=0.8, label="weight(global)")
    ax.axhline(0.70, color="k", ls="--", label="target 0.70")
    ax.set_xlabel("round"); ax.set_ylabel("fraction chains in global mode")
    ax.set_title(f"recommended config mixing (b0={best_b0}, p_jump={best_pjump}): "
                 f"round-trips={rec_mix['total_rt']}, DE acc={rec_mix['de_acc']*100:.2f}%")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "E_mixing_trajectory.png"), dpi=110); plt.close(fig)

    # ---- save backing ----
    store.update(dict(
        best_b0=best_b0, best_pjump=best_pjump,
        rec_mix_acc=rec_mix['de_acc'], rec_mix_rt=rec_mix['total_rt'], rec_mix_w=rec_mix['w_mean'],
        rec_w_series=rec_mix['w_series'],
        recov_w_series=rec_rec_sec['w_series'], recov_w=rec_rec_sec['w_mean'], recov_se=se_rec,
        inv_w=inv['w_mean'], inv_se=inv['se_w'], inv_w1=inv['w1'], inv_w2=inv['w2'], inv_se_diff=inv['se_diff'],
        inv_w_series=inv['w_series'],
        ks_p=ks['ks_p'], ks_stat=ks['ks_stat'], ks_min_p=ks['min_p'],
        unbiased=int(unbiased),
    ))
    np.savez(os.path.join(HERE, "E_backing.npz"),
             **{k: np.asarray(v) for k, v in store.items()
                if isinstance(v, (np.ndarray, float, int, np.floating, np.integer))})

    # ---- summary table to json for the report ----
    summary = dict(
        gate1=dict(de_acc_pct=g1['de_acc']*100, round_trips=g1['total_rt'], w_glob=g1['w_mean'],
                   pass_=gate1_pass),
        sweep1_jitter=[dict(name=c[0], b0=c[1], de_acc_pct=c[2]['de_acc']*100,
                            round_trips=c[2]['total_rt'], w_glob=c[2]['w_mean']) for c in sweep1],
        sweep2_pjump=[dict(p_jump=c[0], de_acc_pct=c[1]['de_acc']*100,
                           round_trips=c[1]['total_rt'], w_glob=c[1]['w_mean']) for c in sweep2],
        recommended=dict(b0=best_b0, p_jump=best_pjump,
                         mix_de_acc_pct=rec_mix['de_acc']*100, round_trips=rec_mix['total_rt'],
                         mix_w_glob=rec_mix['w_mean'],
                         recovery_w_glob=rec_rec_sec['w_mean'], recovery_se=se_rec,
                         inv_w=inv['w_mean'], inv_se=inv['se_w'],
                         inv_drift=abs(inv['w2']-inv['w1']), inv_3se_diff=3*inv['se_diff'],
                         ks_min_p=ks['min_p'], unbiased=unbiased))
    with open(os.path.join(HERE, "E_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    pr(f"\nDONE in {time.time()-t_all:.1f}s. Plots E_*.png, backing E_backing.npz, summary E_summary.json")


if __name__ == "__main__":
    main()
