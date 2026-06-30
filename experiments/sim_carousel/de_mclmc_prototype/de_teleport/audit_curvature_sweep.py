"""ORCHESTRATOR/GRADER AUDIT (not the proposer's run): validate that GATE C is not
ill-posed and that the off-ridge proxy actually PREDICTS mode round-trips.

The proposer's GATE-1 swept only b in {6,9,12} (all HARDER than the carousel point),
where linear-DE round-trips were 0 for everyone -> the pre-registered ">=10x baseline"
metric is undefined (0/0) and the off-ridge proxy was never checked against round-trips.

This sweep goes to LOWER curvature b in {1,2,3,4,6}, for the linear-DE baseline and the
near-teleport (the best affine candidate by off-ridge geometry), counting cross-mode
round-trips on the SAME balanced init / honest within-mode MM as curved_gates.py.

PRE-REGISTRATION (cause/prediction/falsifier):
  Cause hypothesis: round-trips are governed by the cross-mode off-ridge landing
    (near's within-mode off-ridge ~5.9 << linDE gamma_big ~30), so as curvature b
    falls, BOTH methods regain round-trips, and near should regain them at a HIGHER
    b (break later) than linDE -- i.e. near is genuinely more curvature-robust, just
    not enough at the carousel point b*=6.
  Prediction (direction+magnitude): at small b (~1) both round-trip freely (tens-
    hundreds). near's round-trips stay > linDE's across the sweep; both -> 0 by b~6.
    If near were NO better than linDE at every b, its geometric advantage would be
    illusory.
  Falsifier of "test passable": if EVERY method gives 0 round-trips at EVERY b down
    to 1, the testbed is ill-posed (unpassable) and the b*=6 zero is uninformative.
  Falsifier of "off-ridge predictive": if near (lower off-ridge) does NOT round-trip
    more than linDE at the curvatures where round-trips are nonzero, off-ridge does
    not predict mixing.

Plots before metrics. CPU only. Saves incrementally.
"""
import os, sys, time, json
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
from curved_testbed import CurvedTarget
from de_teleport import make_teleport_composite
from de_mclmc import make_composite

D = 10; NCH = 16; K = 20; SEED = 20260627
R = int(sys.argv[1]) if len(sys.argv) > 1 else 800
BS = [1.0, 2.0, 3.0, 4.0, 6.0]
NPZ = os.path.join(HERE, "audit_curvature_sweep.npz")
JSN = os.path.join(HERE, "audit_curvature_sweep.json")
def pr(*a): print(*a, flush=True)


def balanced_init(t, seed):
    rng = np.random.default_rng(seed)
    a = t.exact_draws_mode(NCH//2, 0, rng); b = t.exact_draws_mode(NCH//2, 1, rng)
    out = np.empty((NCH, D)); out[0::2] = a; out[1::2] = b
    return out


def run(comp, init, n_rounds, seed):
    st = comp["init_states"](jnp.asarray(init), jax.random.key(seed))
    keys = jax.random.split(jax.random.key(seed+7), n_rounds)
    pos = np.empty((n_rounds, NCH, D)); acc = np.empty((n_rounds, NCH))
    for r in range(n_rounds):
        st, (p, ec, a) = comp["round"](st, keys[r]); pos[r] = np.asarray(p); acc[r] = np.asarray(a)
    return pos, acc


def round_trips(modes):
    R_, C = modes.shape; tot = 0; cross = 0
    for c in range(C):
        d = np.diff(modes[:, c]); ups = int((d == 1).sum()); dns = int((d == -1).sum())
        tot += min(ups, dns); cross += int(np.abs(d).sum())
    return tot, cross


def measure(comp, t, init, seed):
    pos, acc = run(comp, init, R, seed)
    modes = t.classify(pos)
    w = (modes == 1).mean(1)
    rt, cross = round_trips(modes)
    return dict(de_acc=float(acc.mean()), rt=int(rt), cross=int(cross),
                w_mean=float(w[R//2:].mean()))


def main():
    t0 = time.time(); pr(f"R={R} sweep b={BS}")
    store = {"b": np.array(BS)}; summ = {"R": R, "rows": []}

    for b in BS:
        t = CurvedTarget(D=D, b=b)
        Cg = jnp.asarray(t.within_mode_cov(0)); Lg = jnp.linalg.cholesky(Cg)
        init = balanced_init(t, SEED)
        # linear DE baseline (unmodified de_mclmc)
        base = make_composite(t.logp, D, NCH, L=10.0, step_size=0.05, K=K,
                              b0=0.1, p_jump=0.3, inverse_mass_matrix=Cg, eps_scale=Lg)
        mr_lin = measure(base, t, init, SEED+1)
        # near-teleport (best affine candidate; isotropic jitter as in curved_gates)
        cn = make_teleport_composite(t.logp, D, NCH, L=10.0, step_size=0.05, K=K,
                                     move="near", b0=0.1, p_jump=0.5,
                                     inverse_mass_matrix=Cg, eps_scale=None)
        mr_near = measure(cn, t, init, SEED+1)
        # gamma1 teleport (spot check)
        cg = make_teleport_composite(t.logp, D, NCH, L=10.0, step_size=0.05, K=K,
                                     move="gamma1", b0=0.1, p_jump=0.5,
                                     inverse_mass_matrix=Cg, eps_scale=Lg)
        mr_g1 = measure(cg, t, init, SEED+1)
        row = dict(b=b, linDE=mr_lin, near=mr_near, gamma1=mr_g1)
        summ["rows"].append(row)
        pr(f"  b={b:4.1f} | linDE rt={mr_lin['rt']:4d} acc={mr_lin['de_acc']*100:6.3f}% "
           f"| near rt={mr_near['rt']:4d} acc={mr_near['de_acc']*100:6.3f}% "
           f"| gamma1 rt={mr_g1['rt']:4d} acc={mr_g1['de_acc']*100:6.3f}%")
        # incremental save
        for nm in ("linDE", "near", "gamma1"):
            store[f"rt_{nm}"] = np.array([next((rr[nm]['rt'] for rr in summ['rows'] if rr['b']==bb), np.nan) for bb in BS])
            store[f"acc_{nm}"] = np.array([next((rr[nm]['de_acc'] for rr in summ['rows'] if rr['b']==bb), np.nan) for bb in BS])
        np.savez(NPZ, **store)
        with open(JSN, "w") as f: json.dump(summ, f, indent=2, default=float)

    # ---- plot: round-trips vs b ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for nm, c in [("linDE","tab:blue"),("near","tab:orange"),("gamma1","tab:green")]:
        ax[0].plot(BS, store[f"rt_{nm}"], "o-", color=c, label=nm)
    ax[0].axvline(6.0, color="k", ls=":", label="b*=6 (carousel)")
    ax[0].set_xlabel("curvature b"); ax[0].set_ylabel("mode round-trips (16 chains)")
    ax[0].set_title(f"Round-trips vs curvature (R={R}) — test passable at low b?"); ax[0].legend()
    for nm, c in [("linDE","tab:blue"),("near","tab:orange"),("gamma1","tab:green")]:
        ax[1].semilogy(BS, np.maximum(store[f"rt_{nm}"], 0.5), "o-", color=c, label=nm)
    ax[1].axvline(6.0, color="k", ls=":")
    ax[1].set_xlabel("curvature b"); ax[1].set_ylabel("round-trips (log, floored 0.5)")
    ax[1].set_title("breaking curvature: does near break LATER than linDE?"); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "audit_curvature_sweep.png"), dpi=110); plt.close(fig)
    pr(f"\nsaved audit_curvature_sweep.png/.npz/.json  total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
