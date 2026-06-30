"""GPU carousel test of TEMPERED BURN-IN (C-18) on the REAL minimal-carousel
posterior -- the production scenario SA/DE could not solve: ALL chains start in the
~1e-5 secondary basin (bad-MAP situation); does annealing beta->1 DISCOVER the global
basin and DRAIN the secondary?

PRE-REGISTERED (cause/prediction/falsifier):
  cause: flattening the target (beta<1) lowers the secondary->global barrier so MCLMC's
    GRADIENT dynamics cross it (curvature-robust, follows the ridge); cooling tracks the
    tempered weight which sharpens to the true (~all-global) occupancy at beta=1.
  prediction: cold occ(global) -> ~1.0 (secondary drains); vanilla MCLMC (beta=1, no
    anneal) stays ~0 (trapped, C-10). Per-stage EEVPD stays <= target (faithful anneal).
  falsifier: cold occ(global) ~ 0 (no discovery) => tempering does NOT transfer to the
    real carousel (like SA, C-17) -- a real negative, not assumed.

EEVPD-tuned step on the REAL posterior (NOT hand-set); compared to the production
adapted step ss0=0.0484. Honest within-mode MM (upper_cov). Saves arrays + trace.
"""
import os, sys, numpy as np, jax, jax.numpy as jnp, time
jax.config.update("jax_enable_x64", True)
SCR_DIAG = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/minimal_case_recheck_plots/_scripts"
HERE = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/de_mclmc_prototype"
sys.path.insert(0, SCR_DIAG); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "tempering"))
from build_model import build
from tempered_mclmc import make_tempered_sampler, tune_step_eevpd, DESIRED_EVAR
def pr(*a): print(*a, flush=True)

pm = build()
def logp(z): return pm.log_prob(z)[0]
D = 14; THR = 4.40; COL = 9
prep = np.load(os.path.join(SCR_DIAG, "basin_prep.npz"))
upper_cov = jnp.asarray(prep["upper_cov"]); L0 = float(prep["L_final"]); ss0 = float(prep["ss_final"])
z_best = np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/minimal_case/map/arrays.npz")["z_best"].reshape(-1)
sec_center = np.asarray(z_best)
NCH = 16
NPZ = os.path.join(HERE, "tempering_carousel.npz")

def occ_global(P):  # P (...,D) -> frac with col9>THR
    return float((np.asarray(P)[..., COL] > THR).mean())

def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    init_sec = sec_center[None, :] + 1e-3 * rng.standard_normal((NCH, D))   # ALL in secondary
    pr(f"all-secondary init: occ(global)={occ_global(init_sec):.3f} (should be 0); "
       f"logp={float(logp(jnp.asarray(init_sec[0]))):.1f}  L0={L0:.2f} ss0(prod)={ss0:.4f}")

    # --- EEVPD-tune the step on the REAL posterior at beta=1 (secondary init) ---
    pr("\n[tune] EEVPD step tuning at beta=1 on the real carousel ...")
    step_t, sg, sev = tune_step_eevpd(logp, D, jnp.asarray(init_sec), beta=1.0, L=L0,
                                      imm=upper_cov, n_steps=40, seed=1)
    pr(f"[tune] EEVPD-tuned step={step_t:.5f}  (production adapted ss0={ss0:.4f})")
    step = float(step_t)

    betas = list(np.geomspace(0.02, 1.0, 12))
    pr(f"\nbetas (geom 0.02->1.0, {len(betas)}): {[round(b,3) for b in betas]}")
    samp = make_tempered_sampler(logp, D, NCH, betas, L=L0, step_size=step, imm=upper_cov)

    # --- anneal (discovery) ---
    SPB = 400
    pr(f"\n[anneal] cooling ladder, {SPB} steps/beta ...")
    final_pos, trace = samp["anneal"](init_sec, jax.random.key(7), SPB)
    occ_by_beta = [occ_global(trace["stage_pos"][i]) for i in range(len(betas))]
    pr("  beta   occ(global)  stage_EEVPD(<=%.0e faithful)" % DESIRED_EVAR)
    for i, b in enumerate(betas):
        flag = "ok" if trace["stage_eevpd"][i] <= DESIRED_EVAR else "COARSE"
        pr(f"  {b:5.3f}   {occ_by_beta[i]:.3f}        {trace['stage_eevpd'][i]:.2e} {flag}")
    pr(f"[anneal] post-anneal occ(global)={occ_global(final_pos):.3f}")

    # --- cold sampling at beta=1 from the annealed ensemble ---
    pr("\n[cold] sampling at beta=1 from annealed ensemble (2000 steps) ...")
    _, cold_pos, _ = samp["sample_cold"](jnp.asarray(final_pos), jax.random.key(11), 2000)
    cold_pos = np.asarray(cold_pos)                              # (T,n,D)
    occ_cold = occ_global(cold_pos[-1000:])
    pr(f"[cold] occ(global) last1000={occ_cold:.3f}  (truth ~1.0; secondary ~1e-5)")

    # --- vanilla control: beta=1 only from the SAME secondary init (no anneal) ---
    pr("\n[vanilla] beta=1 MCLMC from all-secondary init (no anneal; should stay trapped) ...")
    _, van_pos, _ = samp["sample_cold"](jnp.asarray(init_sec), jax.random.key(13), 2000)
    van_pos = np.asarray(van_pos)
    occ_van = occ_global(van_pos[-1000:])
    pr(f"[vanilla] occ(global) last1000={occ_van:.3f}  (should stay ~0 = trapped)")

    np.savez(NPZ, betas=np.asarray(betas), occ_by_beta=np.asarray(occ_by_beta),
             stage_eevpd=trace["stage_eevpd"], occ_cold=occ_cold, occ_van=occ_van,
             step_tuned=step, ss0=ss0, cold_col9=cold_pos[-1000:][:, :, COL].astype(np.float32),
             van_col9=van_pos[-1000:][:, :, COL].astype(np.float32))
    pr("\n========== RESULT (PROPOSED/UNCERTIFIED) ==========")
    pr(f"  tempered burn-in cold occ(global) = {occ_cold:.3f}")
    pr(f"  vanilla (trapped) cold occ(global) = {occ_van:.3f}")
    disc = occ_cold > occ_van + 0.2
    pr(f"  DISCOVERY (tempering >> vanilla): {'YES' if disc else 'NO'}  | wall {time.time()-t0:.0f}s")
    pr("  ->", NPZ)

if __name__ == "__main__":
    main()
