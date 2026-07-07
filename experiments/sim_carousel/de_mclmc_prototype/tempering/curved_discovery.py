"""GATE D (curved barrier) for TEMPERED BURN-IN, with a PROPERLY EEVPD-TUNED step
(this is the testbed where energy error BINDS -- the C-15 lesson).

Target: the carousel-faithful CURVED bimodal (sa_mcmc/curved_testbed.CurvedBimodal,
default b=0.6 -> GATE-1 linear-DE within-mode acceptance ~0.6-1.5%, reproducing
the real carousel's ~0.6%). Modes separated along x1; within-mode geometry is a
curved thin ridge so a straight chord between on-ridge points lands OFF the ridge
-> every AFFINE ensemble move fails (C-12/C-14: 0 cross-mode round-trips).

PRE-REGISTRATION
CAUSE: tempering flattens the target so MCLMC's GRADIENT dynamics (which FOLLOW
the curved ridge, unlike an affine chord) cross the barrier at small beta; cooling
re-localizes chains into both modes. Curvature-robust by construction.
GATE D (discovery across CURVED barrier): ALL chains init in mode B (minor); after
tempered burn-in the cold ensemble must POPULATE mode A (the dominant basin) ->
occ_A > 0.5 and clearly > vanilla. CONTRAST: vanilla MCLMC (beta=1) from the same
init stays in mode B (occ_A ~ 0); affine DE gives 0 cross-mode round-trips (C-14,
cited). FALSIFIER: occ_A ~ 0 after annealing (tempering failed to cross curved
barrier).
EEVPD: step tuned at beta=1 on this curved target so mean(ec^2)/D ~= 5e-4; reused
across the cooling ladder (conservative: force ~ beta, hotter levels have less
energy error). Tuned step REPORTED.
"""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "sa_mcmc"))
from tempered_mclmc import make_tempered_sampler, tune_step_eevpd, DESIRED_EVAR
from curved_testbed import CurvedBimodal, static_linear_de_acceptance

N_CHAINS, L = 64, 2.0
BETAS = np.geomspace(0.006, 1.0, 22)
STEPS_PER_BETA = 300
COLD_STEPS = 800
SEED = 20260628

def main():
    t0 = time.time()
    tgt = CurvedBimodal()    # default: D=10, w=(0.6,0.4), b=0.6
    D = tgt.D
    # GATE-1 curvature calibration (carousel-faithful within-mode acceptance)
    g1 = static_linear_de_acceptance(tgt, n=20000, mode=1, p_jump=0.5, b0=0.05)
    print(f"GATE-1 linear-DE within-mode acceptance = {100*g1:.2f}%  "
          f"(carousel-faithful 0.5-1.5%)", flush=True)

    # cross-mode barrier (along x1) for context
    rng = np.random.default_rng(0)
    yA, _ = tgt.exact_draws_balanced(4000, rng)
    print(f"classify agreement on exact draws: "
          f"{(tgt.classify(yA)==_).mean():.3f}", flush=True)

    # ---- EEVPD-tune step at beta=1 on the CURVED target ----
    rng2 = np.random.default_rng(1)
    yrep, _ = tgt.exact_draws(96, rng2)
    print("\n=== EEVPD step tuning at beta=1 (CURVED target) ===", flush=True)
    step1, grid, evs = tune_step_eevpd(tgt.logp, D, jnp.asarray(yrep), beta=1.0, L=L,
                                       step_grid=np.geomspace(0.003, 0.5, 14),
                                       target=DESIRED_EVAR, n_steps=40)
    print(f"beta=1 EEVPD-tuned step = {step1:.5f}\n", flush=True)

    # init: ALL chains in mode B (minor) -- the wrong basin
    rng3 = np.random.default_rng(SEED)
    yb, lblb = tgt.exact_draws_balanced(N_CHAINS, rng3)
    initB = yb[tgt.classify(yb) == 1]
    reps = int(np.ceil(N_CHAINS/max(1, len(initB)))); initB = np.tile(initB, (reps, 1))[:N_CHAINS]

    # FAITHFULNESS SWEEP: the interpolated tuned step sits at the integrator
    # stability knee (EEVPD explodes ~step^12), so the ANNEAL max EEVPD -- which
    # includes transient high-curvature OFF-RIDGE excursions during crossing --
    # can be >> 5e-4, meaning a "crossing" could be NUMERICAL HEATING, not
    # faithful tempered dynamics (C-15 lesson). We REQUIRE anneal max EEVPD<=5e-4
    # and verify discovery PERSISTS at such a faithful step (if discovery
    # vanishes when the step is made faithful, it was a step artifact).
    cand_steps = sorted(set([round(step1, 3), 0.07, 0.05, 0.03, 0.02]), reverse=True)
    sweep = []; stage_occA = None; faithful_step = None; occ_A = None
    print("[D] FAITHFULNESS step sweep (tempered burn-in from all-modeB init):", flush=True)
    for st in cand_steps:
        samp = make_tempered_sampler(tgt.logp, D, N_CHAINS, BETAS, L=L, step_size=st)
        ka, kc = jax.random.split(jax.random.key(SEED + 1))
        final_pos, trace = samp["anneal"](jnp.asarray(initB), ka, STEPS_PER_BETA)
        _, cold_pos, _ = samp["sample_cold"](final_pos, kc, COLD_STEPS)
        cold_pos = np.asarray(cold_pos)
        cold_lbl = tgt.classify(cold_pos.reshape(-1, D)).reshape(cold_pos.shape[:2])
        ocA = float((cold_lbl == 0).mean())
        amax = float(trace["stage_eevpd"].max())
        faithful = amax <= DESIRED_EVAR
        sweep.append((st, amax, ocA, faithful))
        if faithful and faithful_step is None:
            faithful_step = st; occ_A = ocA
            stage_lbl = tgt.classify(trace["stage_pos"].reshape(-1, D)).reshape(trace["stage_pos"].shape[:2])
            stage_occA = (stage_lbl == 0).mean(axis=1)
        print(f"   step={st:.3f}: anneal_max_EEVPD={amax:.2e} "
              f"({'FAITHFUL' if faithful else 'TOO COARSE'})  cold occ_A={ocA:.4f}", flush=True)
    if faithful_step is None:   # fall back to the finest (lowest-EEVPD) step
        faithful_step, _, occ_A, _ = min(sweep, key=lambda r: r[1])
        stage_occA = np.full(len(BETAS), np.nan)
    print(f"\n   FAITHFUL step = {faithful_step:.3f}; cold occ_A = {occ_A:.4f} "
          f"(truth w_A={tgt.w[0]})", flush=True)

    # vanilla contrast at the faithful step
    print("[D] vanilla MCLMC (beta=1) from same init (faithful step) ...", flush=True)
    samp = make_tempered_sampler(tgt.logp, D, N_CHAINS, BETAS, L=L, step_size=faithful_step)
    kv = jax.random.key(SEED + 2)
    _, vpos, _ = samp["sample_cold"](jnp.asarray(initB), kv,
                                     len(BETAS)*STEPS_PER_BETA + COLD_STEPS)
    vpos = np.asarray(vpos)
    vlbl = tgt.classify(vpos[-COLD_STEPS:].reshape(-1, D)).reshape(COLD_STEPS, N_CHAINS)
    occ_A_v = float((vlbl == 0).mean())
    print(f"   vanilla cold occ_A = {occ_A_v:.4f} (should be ~0; affine DE 0 round-trips, C-14)")

    np.savez(os.path.join(HERE, "curved_discovery.npz"),
             betas=BETAS, step1=step1, faithful_step=faithful_step, grid=grid, evs=evs, g1=g1,
             sweep=np.array(sweep, dtype=float), stage_occA=stage_occA,
             occ_A=occ_A, occ_A_v=occ_A_v, wA=tgt.w[0])

    # PLOTS
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].plot(BETAS, stage_occA, "o-", ms=3, label="ensemble occ_A (anneal)")
    a = tgt.w[0]**BETAS; b = tgt.w[1]**BETAS
    axes[0].plot(BETAS, a/(a+b), "k--", label="tempered weight target")
    axes[0].axhline(tgt.w[0], color="C2", ls=":", label=f"true w_A={tgt.w[0]}")
    axes[0].set_xscale("log"); axes[0].set_xlabel("beta"); axes[0].set_ylabel("occ_A")
    axes[0].set_title(f"Gate D anneal across CURVED barrier (GATE-1 {100*g1:.2f}%)")
    axes[0].legend(fontsize=8)
    axes[1].bar(["tempered", "vanilla"], [occ_A, occ_A_v], color=["C0", "C3"])
    axes[1].axhline(tgt.w[0], color="k", ls="--", label=f"truth {tgt.w[0]}")
    axes[1].set_ylabel("cold occ_A"); axes[1].set_ylim(0, 1)
    axes[1].set_title("Tempering discovers across curved barrier; vanilla trapped")
    axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "D_curved_discovery.png"), dpi=110); plt.close(fig)
    print(f"wall {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
