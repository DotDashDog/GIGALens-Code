"""GATE C: SA-MCMC mixture mover on the CURVED representative testbed.

Pre-registered (method discipline):
 cause: the curved thin ridge defeats AFFINE moves (chord lands off-ridge); the
   SA mixture proposes NEAR an existing on-ridge point (curvature robust) and the
   self-inclusive deletion preserves pi^{otimes N}. Prediction: (C1) from a
   truth-populated init the SA mixture HOLDS weight 0.60 with no drift and correct
   per-mode base moments; (C2) from a WRONG-but-populated weight init (0.30) the SA
   mixture RE-EQUILIBRATES to 0.60 while linear DE stays STUCK (curvature blocks
   cross-mode affine moves); (C3) SA cross-mode flip throughput >> linear DE.
 falsifier: weight off 0.60 by >3 SE, drift >3 SE, OR DE re-equilibrates as well as
   SA (=> curvature not actually blocking affine moves => testbed too easy).
GATE-1 (anti-gaming) static within-mode linear-DE acceptance is reported.

Run:  python validate_curved.py
"""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from curved_testbed import build_target, static_linear_de_acceptance
from sa_move import make_sa_composite
from de_mclmc import make_composite
from validate_analytic import integrated_autocorr_time, block_bootstrap_se

# ----- locked curved config (GATE-1 calibrated) and sampler knobs ------------
B_CURV, N_CURVE = 0.70, 4
N_CHAINS, L, STEP, K = 64, 2.0, 0.2, 20      # step 0.2: MCLMC resolves the curvature
N_SA, BW = 64, 0.20                          # bw 0.20: on-ridge hops + fast mixing
R = int(os.environ.get("R","2000"))
SEED = 20260628

def classify_frac(tgt, pos):                  # fraction in mode A (=0)
    return (tgt.classify(np.asarray(pos)) == 0).mean(axis=-1)

def run_sa(comp, init, nr, key, tgt):
    st = comp["init_states"](jnp.asarray(init), jax.random.fold_in(key, 0))
    keys = jax.random.split(key, nr)
    pos = np.empty((nr, N_CHAINS, init.shape[1])); subs = np.empty(nr)
    for r in range(nr):
        st, (p, ec, sub) = comp["round"](st, keys[r])
        pos[r] = np.asarray(p); subs[r] = float(np.asarray(sub).mean())
    return pos, subs

def run_de(comp, init, nr, key):
    st = comp["init_states"](jnp.asarray(init), jax.random.fold_in(key, 0))
    keys = jax.random.split(key, nr)
    pos = np.empty((nr, N_CHAINS, init.shape[1])); acc = np.empty(nr)
    for r in range(nr):
        st, (p, ec, a) = comp["round"](st, keys[r])
        pos[r] = np.asarray(p); acc[r] = float(np.asarray(a).mean())
    return pos, acc

def flips_per_round(tgt, pos):
    lab = tgt.classify(pos)                    # (nr, N)
    return float((lab[1:] != lab[:-1]).sum(axis=1).mean())

def main():
    t0 = time.time()
    tgt = build_target(b=B_CURV, n_curve=N_CURVE)
    D = tgt.D
    imm = jnp.asarray(tgt.within_mode_cov(mode=1))
    g1 = static_linear_de_acceptance(tgt, n=60000)
    print(f"GATE-1 static within-mode linear-DE acceptance = {g1*100:.2f}% "
          f"(target 0.5-1.5%; Gaussian testbeds ~8.7%)")

    sa = make_sa_composite(tgt.logp, D, N_CHAINS, L=L, step_size=STEP, K=K, n_sa=N_SA,
                           proposal="mixture", bandwidth=BW, kernel_cov=jnp.eye(D),
                           inverse_mass_matrix=imm)
    de = make_composite(tgt.logp, D, N_CHAINS, L=L, step_size=STEP, K=K,
                        b0=0.05, p_jump=0.5, inverse_mass_matrix=imm)
    boot = np.random.default_rng(SEED)

    # ---------- C1: invariance from truth-populated init ----------------------
    print("\n[C1] SA-mixture invariance from truth init (w_A=0.60) ...")
    init_truth, lbl = tgt.exact_draws_balanced(N_CHAINS, np.random.default_rng(SEED))
    pos_sa, subs_sa = run_sa(sa, init_truth, R, jax.random.key(SEED + 1), tgt)
    fa = classify_frac(tgt, pos_sa)
    wA = float(fa.mean()); seA, blk = block_bootstrap_se(fa, rng=boot)
    half = R // 2
    w1, w2 = float(fa[:half].mean()), float(fa[half:].mean())
    s1, _ = block_bootstrap_se(fa[:half], rng=boot); s2, _ = block_bootstrap_se(fa[half:], rng=boot)
    sdiff = np.hypot(s1, s2)
    print(f"   w_A={wA:.4f} +/- {seA:.4f}  |est-0.60|={abs(wA-0.60):.4f} (3SE={3*seA:.4f})")
    print(f"   drift 1st={w1:.4f} 2nd={w2:.4f} |diff|={abs(w2-w1):.4f} (3SE_diff={3*sdiff:.4f})")
    # per-mode BASE-space moments (unwarp), 2nd half thinned
    thin = pos_sa[half::5].reshape(-1, D)
    Xun = thin - tgt._cmask_np[None, :] * (tgt.b * (thin[:, 0:1] ** 2 - tgt._s2))
    labt = tgt.classify(thin)
    mom = {}
    for k, nm in [(0, "A"), (1, "B")]:
        Xk = Xun[labt == k]
        mom[nm] = (Xk[:, 1].mean(), Xk[:, 1].std(), Xk[:, 0].std())
        tgt_mu1 = tgt.mu[k, 1]
        print(f"   mode {nm}: base dim1 mean={mom[nm][0]:.3f}(tgt {tgt_mu1:+.1f}) "
              f"std={mom[nm][1]:.3f}(tgt {tgt.s_thin}) ridge dim0 std={mom[nm][2]:.3f}(tgt {tgt.s_ridge})")
    min_eig = float(np.linalg.eigvalsh(np.cov(pos_sa[half::5].reshape(-1, D).T)).min())
    flips_sa = flips_per_round(tgt, pos_sa)
    print(f"   SA sub-rate={subs_sa.mean():.3f} flips/round={flips_sa:.1f} min_eig={min_eig:.3e}")

    # ---------- C2: re-equilibration from WRONG populated weight (0.30) -------
    print("\n[C2] re-equilibration from WRONG populated init w_A=0.30 (truth 0.60) ...")
    rngw = np.random.default_rng(SEED + 7)
    nA = int(round(0.30 * N_CHAINS)); compl = np.ones(N_CHAINS, int); compl[:nA] = 0
    Xw = rngw.standard_normal((N_CHAINS, D)) * np.sqrt(tgt.var)[None, :] + tgt.mu[compl]
    init_wrong = tgt.f_np(Xw)
    pos_saw, _ = run_sa(sa, init_wrong, R, jax.random.key(SEED + 8), tgt)
    pos_dew, accw = run_de(de, init_wrong, R, jax.random.key(SEED + 9))
    fa_saw = classify_frac(tgt, pos_saw); fa_dew = classify_frac(tgt, pos_dew)
    flips_saw = flips_per_round(tgt, pos_saw); flips_dew = flips_per_round(tgt, pos_dew)
    wsa_end = float(fa_saw[half:].mean()); wde_end = float(fa_dew[half:].mean())
    print(f"   SA-mixture w_A: start~0.30 -> end {wsa_end:.3f} (truth 0.60) flips/rd={flips_saw:.1f}")
    print(f"   linear-DE  w_A: start~0.30 -> end {wde_end:.3f} (truth 0.60) flips/rd={flips_dew:.2f} (DE acc={accw.mean():.3f})")

    # =============================== PLOTS ===================================
    pre = os.path.join(HERE, "C_curved")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(fa, lw=0.6, label="SA-mixture"); axes[0].axhline(0.60, color="k", ls="--", label="truth 0.60")
    axes[0].axhline(w1, color="C0", alpha=.5, label=f"1st {w1:.3f}"); axes[0].axhline(w2, color="C1", alpha=.5, label=f"2nd {w2:.3f}")
    axes[0].set_title(f"C1 invariance from truth (GATE-1 linDE acc={g1*100:.1f}%)"); axes[0].set_xlabel("round"); axes[0].set_ylabel("frac mode A"); axes[0].legend(fontsize=8)
    axes[1].plot(fa_saw, lw=0.7, label=f"SA-mixture end {wsa_end:.3f}")
    axes[1].plot(fa_dew, lw=0.7, label=f"linear-DE end {wde_end:.3f}")
    axes[1].axhline(0.60, color="k", ls="--", label="truth 0.60"); axes[1].axhline(0.30, color="grey", ls=":", label="init 0.30")
    axes[1].set_title("C2 re-equilibration from wrong populated init"); axes[1].set_xlabel("round"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(pre + "_weight.png", dpi=110); plt.close(fig)

    # ridge geometry: dim0 vs dim2 (curved) for SA samples colored by mode, + chords
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    s = pos_sa[half::5].reshape(-1, D); lab = tgt.classify(s)
    ax.scatter(s[lab == 0, 0], s[lab == 0, 2], s=2, alpha=.3, label="mode A")
    ax.scatter(s[lab == 1, 0], s[lab == 1, 2], s=2, alpha=.3, label="mode B")
    ax.set_xlabel("dim0 (ridge)"); ax.set_ylabel("dim2 (curved)"); ax.legend(); ax.set_title("SA samples lie ON the curved ridges (dim0 vs dim2)")
    fig.tight_layout(); fig.savefig(pre + "_ridge.png", dpi=110); plt.close(fig)

    np.savez(pre + ".npz", g1=g1, fa=fa, wA=wA, seA=seA, w1=w1, w2=w2, sdiff=sdiff,
             fa_saw=fa_saw, fa_dew=fa_dew, wsa_end=wsa_end, wde_end=wde_end,
             flips_sa=flips_sa, flips_saw=flips_saw, flips_dew=flips_dew,
             min_eig=min_eig, sub_rate=float(subs_sa.mean()), de_acc=float(accw.mean()),
             config=str(sa["config"]))

    # ----------------------- PROPOSED verdict --------------------------------
    c1 = abs(wA - 0.60) < 3 * seA
    c1b = abs(w2 - w1) < 3 * sdiff
    c1c = (abs(mom["A"][0] - tgt.mu[0, 1]) < 0.1 and abs(mom["B"][0] - tgt.mu[1, 1]) < 0.1)
    c2 = abs(wsa_end - 0.60) < 0.05 and abs(wde_end - 0.30) < 0.10     # SA fixes, DE stuck
    c3 = flips_saw > 5 * max(flips_dew, 1e-6)
    c4 = (0.05 < subs_sa.mean() < 0.95) and (min_eig > 0)
    Mk = lambda b: "PASS" if b else "FAIL"
    print("\n========== GATE C CHECKS (PROPOSED / UNCERTIFIED) ==========")
    print(f" GATE-1 static linDE acc = {g1*100:.2f}%  (calibration; want 0.5-1.5%)")
    print(f" C1  invariance w_A      {wA:.4f}+/-{seA:.4f} vs .60 -> {Mk(c1)}")
    print(f" C1b no drift            |{w2-w1:.4f}|<{3*sdiff:.4f}        -> {Mk(c1b)}")
    print(f" C1c per-mode base means within 0.10                  -> {Mk(c1c)}")
    print(f" C2  SA->0.60 ({wsa_end:.3f}) & DE stuck ({wde_end:.3f})    -> {Mk(c2)}")
    print(f" C3  flips SA {flips_saw:.1f} >> DE {flips_dew:.2f}            -> {Mk(c3)}")
    print(f" C4  sub-rate {subs_sa.mean():.3f} & min_eig {min_eig:.2e}    -> {Mk(c4)}")
    print(f" total wall {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
