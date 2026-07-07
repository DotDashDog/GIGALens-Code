"""Clean re-run of TEMPERED BURN-IN on the real carousel, using the USER's MCLMC
STEP-SIZE ADAPTATION (the `step_size_adapt` algorithm from mclmc.py) per step instead
of a fixed/position-blind tuned step. This fixes the C-19 confound (a step EEVPD-tuned
at the secondary init was too coarse for the annealed/global geometry).

Faithful replication of mclmc.full_mclmc_with_adapt_sharded.step_size_adapt (the
DEFAULT, step_size_adapt_use_psmile=False) with its defaults:
  desired_energy_var=5e-4, trust_in_estimate=1.5, num_effective_samples=150
  => decay_rate=(150-1)/(150+1). handle_nans imported read-only from blackjax_updated_utils.
The step adapts PER-CHAIN, PER-STEP to hit EEVPD=mean(dE^2)/D=desired_energy_var, warm-
started across the cooling ladder. MCLMC kernel + handle_nans imported read-only; nothing
modified.

PRE-REG: with a per-step-adapted (faithful everywhere) step, separate FREEZE-OUT from the
step confound. prediction: per-stage EEVPD ~5e-4 (faithful); occ(global) climbs 0->X.
If occ caps near the decoupling occupancy (<<1.0) at FAITHFUL EEVPD => freeze-out is the
real ceiling (=> PT/adaptive needed), not my step bug. falsifier of "tempering transfers":
occ stays ~0 (=> no discovery even with good step).
"""
import os, sys, numpy as np, jax, jax.numpy as jnp, time
jax.config.update("jax_enable_x64", True)
SCR_DIAG = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/minimal_case_recheck_plots/_scripts"
HERE = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/de_mclmc_prototype"
sys.path.insert(0, SCR_DIAG); sys.path.insert(0, HERE)
from build_model import build
from gigalens_research.inference.blackjax_updated_utils import (
    _build_kernel_shardmap, isokinetic_mclachlan_smart, _single_init, handle_nans)
def pr(*a): print(*a, flush=True)

pm = build()
def logp(z): return pm.log_prob(z)[0]
D = 14; THR = 4.40; COL = 9
prep = np.load(os.path.join(SCR_DIAG, "basin_prep.npz"))
upper_cov = jnp.asarray(prep["upper_cov"]); L0 = float(prep["L_final"]); ss0 = float(prep["ss_final"])
z_best = np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/minimal_case/map/arrays.npz")["z_best"].reshape(-1)
sec_center = np.asarray(z_best)
NCH = 16
NPZ = os.path.join(HERE, "tempering_carousel_adapt.npz")

# --- user's step_size_adapt defaults (mclmc.full_mclmc_with_adapt_sharded) ---
DEVAR = 5e-4; TRUST = 1.5; NESS = 150; DECAY = (NESS - 1.0) / (NESS + 1.0)

def adapt_one(prev_state, next_state, info, step_size, adapt_state, nan_key):
    """Faithful copy of mclmc.step_size_adapt (the default, non-psmile)."""
    time_, x_avg, ss_max = adapt_state
    success, state, ss_max, ec = handle_nans(
        prev_state, next_state, step_size, ss_max, info.energy_change, nan_key)
    xi = jnp.square(ec) / (D * DEVAR) + 1e-8
    weight = jnp.exp(-0.5 * jnp.square(jnp.log(xi) / (6.0 * TRUST)))
    weighted_x = weight * (xi / jnp.power(step_size, 6.0))
    x_avg = DECAY * x_avg + weighted_x
    time_ = DECAY * time_ + weight
    new_step = jnp.power(x_avg / time_, -1.0 / 6.0)
    new_step = jnp.minimum(new_step, ss_max)
    return state, new_step, (time_, x_avg, ss_max), xi

def occ_global(P): return float((np.asarray(P)[..., COL] > THR).mean())

def run_beta(kern, states, ss, ast, keys):
    """K adaptive MCLMC steps at one beta. ss:(n,) per-chain step; ast: tuple of (n,)."""
    def scan_step(carry, kt):
        states, ss, ast = carry
        def per(s, step, a, k):
            kk, kn = jax.random.split(k)
            ns, info = kern(rng_key=kk, state=s, L=L0, step_size=step)
            s2, new_step, a2, xi = adapt_one(s, ns, info, step, a, kn)
            return s2, new_step, a2, s2.position, jnp.square(info.energy_change)
        s2, new_step, a2, pos, ec2 = jax.vmap(per)(states, ss, ast, kt)
        return (s2, new_step, a2), (pos, new_step, ec2)
    (states, ss, ast), (pos, sshist, ec2) = jax.lax.scan(scan_step, (states, ss, ast), keys)
    return states, ss, ast, pos, sshist, ec2

def build_kern(beta):
    tf = lambda z: beta * logp(z)
    return _build_kernel_shardmap(logdensity_fn=tf, inverse_mass_matrix=upper_cov,
                                  integrator=isokinetic_mclachlan_smart), tf

def init_at(positions, tf, key):
    keys = jax.random.split(key, positions.shape[0])
    return jax.vmap(lambda p, k: _single_init(p, tf, k))(jnp.asarray(positions), keys)

def fresh_adapt(n, step0):
    return (jnp.zeros(n), jnp.zeros(n), jnp.full(n, 1.0)), jnp.full(n, step0)

def main():
    t0 = time.time(); rng = np.random.default_rng(0)
    init_sec = sec_center[None, :] + 1e-3 * rng.standard_normal((NCH, D))
    pr(f"all-secondary init occ(global)={occ_global(init_sec):.3f}; L0={L0:.2f} ss0={ss0:.4f} "
       f"adapt: DEVAR={DEVAR} TRUST={TRUST} decay={DECAY:.4f}")
    betas = list(np.geomspace(0.02, 1.0, 12)); SPB = 500
    jitted = {b: jax.jit(lambda s, ss, a, k, _kern=build_kern(b)[0]: run_beta(_kern, s, ss, a, k)) for b in betas}
    tfns = {b: build_kern(b)[1] for b in betas}

    positions = jnp.asarray(init_sec)
    ast, ss = fresh_adapt(NCH, ss0)             # carry step across betas, reset averages per beta
    key = jax.random.key(7)
    occ_by_beta, ev_by_beta, ssmed_by_beta = [], [], []
    pr("\n[anneal] per-step adaptive step; betas geom 0.02->1.0")
    pr("  beta   occ(global)  EEVPD(mean dE^2/D)  step(med)")
    for b in betas:
        key, ki, ks = jax.random.split(key, 3)
        states = init_at(positions, tfns[b], ki)            # momentum refresh @ beta
        ast = (jnp.zeros(NCH), jnp.zeros(NCH), ast[2])       # reset running avgs, keep ss_max
        keys = jax.random.split(ks, SPB * NCH).reshape(SPB, NCH)
        states, ss, ast, pos, sshist, ec2 = jitted[b](states, ss, ast, keys)
        positions = states.position
        ev = float(np.mean(np.asarray(ec2)[SPB//2:]) / D)   # EEVPD over 2nd half
        occ = occ_global(positions); ssm = float(np.median(np.asarray(ss)))
        occ_by_beta.append(occ); ev_by_beta.append(ev); ssmed_by_beta.append(ssm)
        pr(f"  {b:5.3f}   {occ:.3f}        {ev:.2e}            {ssm:.4f}")
    pr(f"[anneal] post-anneal occ(global)={occ_global(positions):.3f}")

    # cold sampling at beta=1 (continue adapting)
    pr("\n[cold] beta=1 adaptive sampling from annealed ensemble (2500 steps) ...")
    kern1 = build_kern(1.0)[0]; cold_jit = jax.jit(lambda s, ss, a, k: run_beta(kern1, s, ss, a, k))
    key, ki, ks = jax.random.split(key, 3)
    st = init_at(positions, tfns[1.0], ki); ast = (jnp.zeros(NCH), jnp.zeros(NCH), ast[2])
    keys = jax.random.split(ks, 2500 * NCH).reshape(2500, NCH)
    st, ss, ast, cpos, cssh, cec2 = cold_jit(st, ss, ast, keys)
    cpos = np.asarray(cpos); occ_cold = occ_global(cpos[-1500:])
    ev_cold = float(np.mean(np.asarray(cec2)[-1500:]) / D)
    pr(f"[cold] occ(global) last1500={occ_cold:.3f}  EEVPD={ev_cold:.2e}  step(med)={float(np.median(ss)):.4f}")

    # vanilla control: beta=1 from secondary init (adaptive), should stay trapped
    pr("\n[vanilla] beta=1 adaptive from all-secondary init (no anneal) ...")
    ast_v, ss_v = fresh_adapt(NCH, ss0)
    key, ki, ks = jax.random.split(key, 3)
    stv = init_at(init_sec, tfns[1.0], ki)
    keys = jax.random.split(ks, 2500 * NCH).reshape(2500, NCH)
    stv, ssv, astv, vpos, vssh, vec2 = cold_jit(stv, ss_v, ast_v, keys)
    vpos = np.asarray(vpos); occ_van = occ_global(vpos[-1500:])
    pr(f"[vanilla] occ(global) last1500={occ_van:.3f} (should stay ~0)")

    np.savez(NPZ, betas=np.asarray(betas), occ_by_beta=np.asarray(occ_by_beta),
             ev_by_beta=np.asarray(ev_by_beta), ssmed_by_beta=np.asarray(ssmed_by_beta),
             occ_cold=occ_cold, ev_cold=ev_cold, occ_van=occ_van,
             cold_col9=cpos[-1500:][:, :, COL].astype(np.float32))
    pr("\n========== RESULT (PROPOSED/UNCERTIFIED) ==========")
    pr(f"  tempered burn-in (adaptive step) cold occ(global) = {occ_cold:.3f}  (truth ~1.0)")
    pr(f"  vanilla (trapped)                cold occ(global) = {occ_van:.3f}")
    pr(f"  per-stage EEVPD all ~{DEVAR:.0e}? {'YES (faithful)' if max(ev_by_beta) < 5*DEVAR else 'some COARSE: '+str([f'{e:.1e}' for e in ev_by_beta])}")
    pr(f"  -> occ cap at faithful step ⇒ {'FREEZE-OUT is the ceiling' if occ_cold < 0.85 and occ_cold > occ_van+0.2 else 'see numbers'}")
    pr(f"  wall {time.time()-t0:.0f}s  -> {NPZ}")

if __name__ == "__main__":
    main()
