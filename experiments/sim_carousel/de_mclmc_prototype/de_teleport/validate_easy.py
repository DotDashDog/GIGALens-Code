"""GATE A (unbiasedness + basic mode-hopping) on the EASY analytic mixture for all
three moves, PLUS the snooker with-vs-without-Jacobian bias demonstration.

Easy target (separated, NON-curved -> chords ARE on-manifold within a mode): D=10,
modes +-5 on axis 0, weights [0.7,0.3], barrier 12.5 (vanilla MCLMC trapped).

Pre-registration: each move must (V1) recover weight 0.70 from a single-mode init
while vanilla stays >0.85; (V2) show NO drift from a truth init (|est-0.70|<3SE,
|half diff|<3SE_diff, per-mode moments within 0.10, KS axis0/axis1 p>0.05);
(V4) acceptance in (0.05,0.95), pooled min-eig>0. SNOOKER bias demo: with the
(d-1) Jacobian V2 must PASS; DROPPING it must visibly BIAS (weight and/or moments
move off truth by >>3 SE). Plots before metrics; verdicts PROPOSED/UNCERTIFIED.

Usage: validate_easy.py [Rscale]   (Rscale<1 -> quick smoke)
"""
import os, sys, time, json
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))            # parent: de_mclmc, validate_analytic
from de_teleport import make_teleport_composite
from validate_analytic import integrated_autocorr_time, block_bootstrap_se

# ----------------------------- easy analytic target --------------------------
D = 10; m = 5.0; W = np.array([0.7, 0.3]); MU = np.array([+m, -m])
_logW = jnp.log(jnp.asarray(W)); _MU = jnp.asarray(MU)
_c = -0.5 * D * jnp.log(2 * jnp.pi)

def logdensity_fn(z):
    z0 = z[0]; quad_rest = jnp.sum(z[1:] ** 2)
    c0 = _logW[0] + _c - 0.5 * ((z0 - _MU[0]) ** 2 + quad_rest)
    c1 = _logW[1] + _c - 0.5 * ((z0 - _MU[1]) ** 2 + quad_rest)
    return jax.scipy.special.logsumexp(jnp.stack([c0, c1]))

def analytic_axis0_pdf(x):
    return W[0]*stats.norm.pdf(x, MU[0], 1) + W[1]*stats.norm.pdf(x, MU[1], 1)
def analytic_axis0_samples(n, rng):
    comp = (rng.random(n) >= W[0]).astype(int); return rng.standard_normal(n) + MU[comp]
def exact_mixture_draws(n, rng):
    n_plus = int(round(W[0]*n)); comp = np.ones(n, int); comp[:n_plus] = 0
    z = rng.standard_normal((n, D)); z[:, 0] += MU[comp]; return z, comp

NCH = 32; L, STEP, K = 2.0, 0.5, 20; B0, P_JUMP = 0.05, 0.5; SEED = 20260627
RS = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
R_V1, BURN_V1 = int(1500*RS), int(500*RS); R_V2 = int(2500*RS)

def run(comp, init_pos, n_rounds, key):
    st = comp["init_states"](jnp.asarray(init_pos), jax.random.fold_in(key, 0))
    keys = jax.random.split(key, n_rounds)
    pos = np.empty((n_rounds, NCH, D)); acc = np.empty((n_rounds, NCH))
    for r in range(n_rounds):
        st, (p, ec, a) = comp["round"](st, keys[r]); pos[r] = np.asarray(p); acc[r] = np.asarray(a)
    return pos, acc

def make(move, drop_jacobian=False):
    return make_teleport_composite(logdensity_fn, D, NCH, L=L, step_size=STEP, K=K,
                                   move=move, b0=B0, p_jump=P_JUMP, eps=B0,
                                   drop_jacobian=drop_jacobian)

def gates_from_truth(comp, boot_rng, seed):
    """V2 + V4 metrics from a truth init."""
    init, _ = exact_mixture_draws(NCH, np.random.default_rng(seed))
    pos, acc = run(comp, init, R_V2, jax.random.key(seed+5))
    z0 = pos[:, :, 0]; frac = (z0 > 0).mean(1)
    w2 = float(frac.mean()); se2, _ = block_bootstrap_se(frac, rng=boot_rng)
    half = R_V2 // 2; w1, w2h = float(frac[:half].mean()), float(frac[half:].mean())
    s1, _ = block_bootstrap_se(frac[:half], rng=boot_rng); s2, _ = block_bootstrap_se(frac[half:], rng=boot_rng)
    se_diff = np.hypot(s1, s2)
    thin = pos[half::5].reshape(-1, D); plus = thin[thin[:,0]>0]; minus = thin[thin[:,0]<0]
    pmp, pmm = plus[:,0].mean(), minus[:,0].mean(); pvp, pvm = plus[:,0].var(), minus[:,0].var()
    wvar = thin[:,1].var()
    tau = integrated_autocorr_time(frac); sp = max(1, int(round(tau)))
    pool = pos[half::sp].reshape(-1, D)
    rng = np.random.default_rng(seed+6)
    ca = pool[:,0].copy(); rng.shuffle(ca); ca = ca[:4000]; aa = analytic_axis0_samples(len(ca), rng)
    ks0 = stats.ks_2samp(ca, aa)
    c1 = pool[:,1].copy(); rng.shuffle(c1); c1 = c1[:4000]; a1 = rng.standard_normal(len(c1))
    ks1 = stats.ks_2samp(c1, a1)
    de_acc = float(acc.mean()); min_eig = float(np.linalg.eigvalsh(np.cov(thin.T)).min())
    return dict(pos=pos, frac=frac, w=w2, se=se2, w1=w1, w2h=w2h, se_diff=se_diff,
                pmp=pmp, pmm=pmm, pvp=pvp, pvm=pvm, wvar=wvar,
                ks_p0=ks0.pvalue, ks_p1=ks1.pvalue, de_acc=de_acc, min_eig=min_eig)

def main():
    t0 = time.time(); boot = np.random.default_rng(SEED); summary = {}
    moves = ["gamma1", "near", "snooker"]
    fig, axes = plt.subplots(len(moves)+1, 2, figsize=(12, 4.0*(len(moves)+1)))
    for mi, move in enumerate(moves):
        comp = make(move)
        print(f"\n===== move={move} =====", flush=True)
        # V1 weight recovery from single +mode init
        init_single = np.zeros((NCH, D)); init_single[:, 0] = m
        pos1, acc1 = run(comp, init_single, R_V1, jax.random.key(SEED+1))
        frac1 = (pos1[:, :, 0] > 0).mean(1); w_rec = float(frac1[BURN_V1:].mean())
        se_rec, _ = block_bootstrap_se(frac1[BURN_V1:], rng=boot)
        # vanilla contrast (once)
        if mi == 0:
            st_v = comp["init_states"](jnp.asarray(init_single), jax.random.key(SEED+2))
            total = R_V1*K; done = 0; tail = []
            while done < total:
                nt = min(2000, total-done)
                ck = jax.random.split(jax.random.fold_in(jax.random.key(SEED+3), done), nt*NCH).reshape(nt, NCH)
                st_v, pv = comp["mclmc_only"](st_v, ck)
                if done >= total - 4000: tail.append(np.asarray(pv[:,:,0]))
                done += nt
            w_van = float((np.concatenate(tail) > 0).mean())
        print(f"  V1 weight recovery w(+)={w_rec:.4f}+/-{se_rec:.4f} (0.70); vanilla={w_van:.4f}", flush=True)
        # V2/V4 from truth
        g = gates_from_truth(comp, boot, SEED+10+mi)
        print(f"  V2 w@truth={g['w']:.4f}+/-{g['se']:.4f}  drift|{g['w2h']-g['w1']:.4f}|<{3*g['se_diff']:.4f}", flush=True)
        print(f"  V2 moments +{g['pmp']:.3f}/{g['pmm']:.3f} var {g['pvp']:.3f}/{g['pvm']:.3f} within {g['wvar']:.3f}", flush=True)
        print(f"  V2 KS axis0 p={g['ks_p0']:.3f} axis1 p={g['ks_p1']:.3f}; V4 acc={g['de_acc']:.3f} min_eig={g['min_eig']:.3f}", flush=True)
        c = dict(
            v1=abs(w_rec-0.70) < max(3*se_rec, 0.03),
            v2a=abs(g['w']-0.70) < 3*g['se'],
            v2b=abs(g['w2h']-g['w1']) < 3*g['se_diff'],
            v2c=(abs(g['pmp']-m)<0.10 and abs(g['pmm']+m)<0.10 and abs(g['pvp']-1)<0.10 and abs(g['pvm']-1)<0.10),
            v2d=g['ks_p0']>0.05, v3=(g['ks_p0']>0.05 and g['ks_p1']>0.05),
            v4=(0.05<g['de_acc']<0.95 and g['min_eig']>0))
        summary[move] = dict(w_rec=w_rec, se_rec=se_rec, w_van=w_van, **{k: g[k] for k in
            ['w','se','w1','w2h','se_diff','pmp','pmm','pvp','pvm','wvar','ks_p0','ks_p1','de_acc','min_eig']},
            checks={k: bool(v) for k, v in c.items()}, all_pass=bool(all(c.values())))
        # plots: weight trace (recovery) + axis0 marginal
        axes[mi,0].plot(frac1, lw=0.6); axes[mi,0].axhline(0.70, color='k', ls='--')
        axes[mi,0].axhline(w_rec, color='C1', alpha=.6); axes[mi,0].set_title(f"{move}: V1 recovery w={w_rec:.3f}")
        thin = g['pos'][R_V2//2::5].reshape(-1, D)
        axes[mi,1].hist(thin[:,0], bins=60, density=True, alpha=.5)
        xs = np.linspace(-9,9,400); axes[mi,1].plot(xs, analytic_axis0_pdf(xs), 'k-')
        axes[mi,1].set_title(f"{move}: axis0 (KS p={g['ks_p0']:.3f})")

    # ---- snooker bias demo: drop Jacobian ----
    print("\n===== snooker DROP-JACOBIAN bias demo =====", flush=True)
    comp_nj = make("snooker", drop_jacobian=True)
    g_nj = gates_from_truth(comp_nj, boot, SEED+99)
    g_wj = summary["snooker"]
    print(f"  with-Jac  w@truth={g_wj['w']:.4f}+/-{g_wj['se']:.4f}  moments +{g_wj['pmp']:.3f} var {g_wj['pvp']:.3f}", flush=True)
    print(f"  drop-Jac  w@truth={g_nj['w']:.4f}+/-{g_nj['se']:.4f}  moments +{g_nj['pmp']:.3f} var {g_nj['pvp']:.3f}", flush=True)
    bias_w = abs(g_nj['w']-0.70); bias_var = abs(g_nj['pvp']-1.0)
    # "visibly biased" = off truth by > 3 SE in weight OR per-mode var off by >0.10
    bias_shown = (bias_w > 3*g_nj['se']) or (bias_var > 0.10) or (abs(g_nj['pmp']-m)>0.10)
    print(f"  -> drop-Jac bias_w={bias_w:.4f} (3SE={3*g_nj['se']:.4f}) bias_var={bias_var:.4f} : "
          f"{'VISIBLE BIAS (Jacobian needed -- correct)' if bias_shown else 'no visible bias (inconclusive)'}", flush=True)
    summary["snooker_dropjac"] = dict(w=g_nj['w'], se=g_nj['se'], pmp=g_nj['pmp'], pvp=g_nj['pvp'],
                                      pvm=g_nj['pvm'], wvar=g_nj['wvar'], bias_shown=bool(bias_shown))
    # bias plot: per-mode +mode VARIANCE bar (with vs without Jac) and axis0 marginal
    axes[-1,0].bar(["with Jac\nw", "drop Jac\nw"], [g_wj['w'], g_nj['w']], color=["C0","C3"])
    axes[-1,0].axhline(0.70, color='k', ls='--', label='truth 0.70'); axes[-1,0].legend(fontsize=8)
    axes[-1,0].set_title(f"snooker weight: drop-Jac bias_w={bias_w:.3f}")
    thin_nj = g_nj['pos'][R_V2//2::5].reshape(-1, D)
    xs = np.linspace(-9,9,400)
    axes[-1,1].plot(xs, analytic_axis0_pdf(xs), 'k-', label='analytic')
    axes[-1,1].hist(thin_nj[:,0], bins=60, density=True, alpha=.5, label='drop Jac', color='C3')
    axes[-1,1].legend(fontsize=8); axes[-1,1].set_title("snooker drop-Jac axis0 marginal")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "validate_easy.png"), dpi=110); plt.close(fig)

    with open(os.path.join(HERE, "validate_easy.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"\n--- PROPOSED VERDICTS (UNCERTIFIED) ---")
    for move in moves:
        s = summary[move]; print(f"  {move:8s}: {'UNBIASED' if s['all_pass'] else 'FAIL'}  checks={s['checks']}")
    print(f"  snooker drop-Jac visibly biased: {summary['snooker_dropjac']['bias_shown']}")
    print(f"total {time.time()-t0:.1f}s")

def make_truth_pool(comp, seed):
    init, _ = exact_mixture_draws(NCH, np.random.default_rng(seed))
    pos, _ = run(comp, init, R_V2, jax.random.key(seed+5))
    return pos[R_V2//2::5].reshape(-1, D)

if __name__ == "__main__":
    main()
