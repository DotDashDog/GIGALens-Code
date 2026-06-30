"""DRILL-DOWN (protocol rule 1): is the Gate-A weight deficit (occ_+ ~ 0.51 vs
0.70) fixable by a better anneal schedule, or a FUNDAMENTAL freeze-out?

THEORY: a one-shot cooling anneal sets the between-mode weight at the FREEZE-OUT
beta_f -- the largest beta at which the ensemble still re-equilibrates the weight
within the allotted steps. Vanilla MCLMC crosses the barrier only for
barrier=beta*12.5 <~ 8, i.e. beta <~ 0.64; at beta_f~0.64 the tempered weight is
w0^bf/(w0^bf+w1^bf) = 0.7^.64/(0.7^.64+0.3^.64) ~ 0.637. So even with INFINITE
steps per stage the one-shot weight should PLATEAU near ~0.64, NOT reach 0.70,
because the cold barrier (12.5) is uncrossable.

PREDICTION: increasing steps/stage and ladder density moves occ_+ UP from 0.51
toward ~0.64 but PLATEAUS below 0.70. FALSIFIER: occ_+ reaches 0.70 within SE
for some affordable schedule (then it was under-resourcing, not freeze-out).
"""
import os, sys, time
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tempered_mclmc import make_tempered_sampler

D, m = 10, 5.0
W = np.array([0.7, 0.3]); MU = np.array([+m, -m])
_logW = jnp.log(jnp.asarray(W)); _MU = jnp.asarray(MU); _c = -0.5 * D * jnp.log(2*jnp.pi)
def logdensity_fn(z):
    z0 = z[0]; qr = jnp.sum(z[1:]**2)
    c0 = _logW[0]+_c-0.5*((z0-_MU[0])**2+qr); c1 = _logW[1]+_c-0.5*((z0-_MU[1])**2+qr)
    return jax.scipy.special.logsumexp(jnp.stack([c0, c1]))
def tw_plus(beta):
    a = W[0]**beta; b = W[1]**beta; return a/(a+b)

N_CHAINS, L, STEP = 64, 2.0, 0.5
SEED = 20260628
N_SEEDS = 4

LADDERS = {
    "geom15": np.geomspace(0.04, 1.0, 15),
    "geom30": np.geomspace(0.04, 1.0, 30),
    "densehot1": np.concatenate([np.geomspace(0.04, 0.5, 10),
                                 np.linspace(0.55, 1.0, 20)]),
}
EFFORTS = [200, 800, 2400]

def occ_after_anneal(samp, init_wrong, steps_per_beta, seed):
    key = jax.random.key(seed)
    final_pos, trace = samp["anneal"](jnp.asarray(init_wrong), key, steps_per_beta)
    fp = np.asarray(final_pos)
    return float((fp[:, 0] > 0).mean()), trace

def main():
    t0 = time.time()
    init_wrong = np.zeros((N_CHAINS, D)); init_wrong[:, 0] = MU[1]
    results = {}     # (ladder,effort) -> (mean_occ, se_occ)
    traj_store = {}
    for lname, betas in LADDERS.items():
        samp = make_tempered_sampler(logdensity_fn, D, N_CHAINS, betas, L=L, step_size=STEP)
        for eff in EFFORTS:
            occs = []
            tr0 = None
            for s in range(N_SEEDS):
                occ, tr = occ_after_anneal(samp, init_wrong, eff, SEED + 1000*s + eff)
                occs.append(occ)
                if tr0 is None: tr0 = tr
            occs = np.asarray(occs)
            results[(lname, eff)] = (float(occs.mean()),
                                     float(occs.std(ddof=1)/np.sqrt(N_SEEDS)))
            traj_store[(lname, eff)] = ((tr0["stage_pos"][:, :, 0] > 0).mean(axis=1),
                                        betas)
            print(f"   {lname:10s} eff={eff:5d}: occ_+ = {occs.mean():.4f} "
                  f"+/- {occs.std(ddof=1)/np.sqrt(N_SEEDS):.4f}", flush=True)

    # freeze-out floor estimate: tempered weight at beta_f (barrier 8 -> bf=8/12.5)
    bf = 8.0/12.5
    floor = tw_plus(bf)
    print(f"\n   predicted freeze-out floor (barrier~8, beta_f={bf:.3f}): {floor:.4f}")
    print(f"   true weight: 0.70", flush=True)

    np.savez(os.path.join(HERE, "drill_schedule.npz"),
             results_keys=[f"{k[0]}|{k[1]}" for k in results],
             results_vals=np.array([results[k] for k in results]),
             floor=floor, bf=bf,
             **{f"traj_{k[0]}_{k[1]}": traj_store[k][0] for k in traj_store},
             **{f"betas_{k[0]}_{k[1]}": traj_store[k][1] for k in traj_store})

    # plot 1: occ_+ vs effort, per ladder
    fig, ax = plt.subplots(figsize=(8, 4.7))
    for lname in LADDERS:
        ys = [results[(lname, e)][0] for e in EFFORTS]
        es = [results[(lname, e)][1] for e in EFFORTS]
        ax.errorbar(EFFORTS, ys, yerr=[3*e for e in es], marker="o", label=lname, capsize=3)
    ax.axhline(0.70, color="k", ls="--", label="true 0.70")
    ax.axhline(floor, color="r", ls=":", label=f"freeze-out floor ~{floor:.3f}")
    ax.set_xlabel("steps per beta stage"); ax.set_ylabel("post-anneal occ_+")
    ax.set_xscale("log"); ax.set_ylim(0.45, 0.75)
    ax.set_title("Drill: one-shot anneal weight plateaus at freeze-out floor (< 0.70)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "drill_occ_vs_effort.png"), dpi=110); plt.close(fig)

    # plot 2: anneal trajectory (occ_+ vs beta) for the best effort, each ladder
    fig, ax = plt.subplots(figsize=(8, 4.7))
    for lname in LADDERS:
        fr, betas = traj_store[(lname, EFFORTS[-1])]
        ax.plot(betas, fr, "o-", ms=3, label=f"{lname} occ_+")
    bb = np.geomspace(0.04, 1.0, 100)
    ax.plot(bb, [tw_plus(b) for b in bb], "k--", label="tempered weight target")
    ax.axvline(bf, color="r", ls=":", label=f"freeze-out beta_f~{bf:.2f}")
    ax.set_xlabel("beta"); ax.set_ylabel("ensemble occ_+"); ax.set_xscale("log")
    ax.set_title("Anneal trajectory: occ_+ detaches from tempered weight at beta_f")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(HERE, "drill_trajectory.png"), dpi=110); plt.close(fig)
    print(f"\n   wall {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
