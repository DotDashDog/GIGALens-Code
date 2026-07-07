"""GATE-1 (faithful, anti-gaming) + GATE C (curvature robustness) on the curved
testbed, driving the REAL MCLMC kernel.

GATE-1: run the LINEAR DE baseline (de_mclmc.make_composite, the unmodified module)
on the curved target with the honest within-mode mass matrix and a balanced 16-chain
init. Calibrate curvature b so the linear-DE acceptance reproduces the carousel's low
~0.5-1.5% (an order of magnitude below the Gaussian carousel_testbed's ~8.7%). Report
the measured number so the orchestrator can verify calibration. If >5% the testbed is
too easy (ill-posed) -> increase b.

GATE C: at the calibrated b*, run each candidate move on the SAME balanced init and
compare effective mode-mixing (cross-mode round-trips, DE/move acceptance, weight
recovery) against the linear-DE baseline. A method "wins" only if it materially beats
linear DE on round-trips while staying unbiased.

PRE-REGISTRATION (cause/prediction/falsifier):
  Cause: all affine proposals land off the curved ridge (measured in offridge_diag:
  off-ridge 17-700 vs on-manifold 3), so cross-mode acceptance stays ~0.1-1% and
  round-trips stay low for EVERY affine move -- including gamma=1 teleport, near-
  teleport, and snooker.
  Prediction: at b*, round-trips for gamma1/near/snooker are within ~2-3x of linear
  DE (same order), NOT the >10x needed to call it a working mode-hop. near (best in
  geometry) is the most likely to edge ahead but stays << on-manifold mixing.
  Falsifier: any move gives >=10x the linear-DE round-trips at equal cost AND stays
  unbiased -> it beats the curvature wall.

Plots before metrics. Verdicts PROPOSED / UNCERTIFIED. Usage: curved_gates.py [Rscale]
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
from de_mclmc import make_composite                      # the UNMODIFIED linear-DE baseline
from validate_analytic import integrated_autocorr_time, block_bootstrap_se

D = 10; NCH = 16; K = 20; SEED = 20260627
RS = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
R = int(1200 * RS)


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
    modes = t.classify(pos)                     # (R,NCH) 0=plus,1=minus
    w = (modes == 1).mean(1)                    # frac in minus per round
    rt, cross = round_trips(modes)
    return dict(de_acc=float(acc.mean()), rt=rt, cross=cross,
                w_mean=float(w[R//2:].mean()), w_series=w, pos=pos, modes=modes)


def main():
    t0 = time.time(); store = {}; print(f"R={R}", flush=True)

    # ---------------- GATE-1: linear-DE acceptance vs b ----------------------
    print("\n===== GATE-1: linear DE (de_mclmc) acceptance vs curvature b =====", flush=True)
    g1 = []
    for b in [6.0, 9.0, 12.0]:
        t = CurvedTarget(D=D, b=b)
        Cg = jnp.asarray(t.within_mode_cov(0)); Lg = jnp.linalg.cholesky(Cg)
        comp = make_composite(t.logp, D, NCH, L=10.0, step_size=0.05, K=K,
                              b0=0.1, p_jump=0.3, inverse_mass_matrix=Cg, eps_scale=Lg)
        mr = measure(comp, t, balanced_init(t, SEED), SEED)
        g1.append((b, mr)); print(f"  b={b:4.1f}  linDE acc={mr['de_acc']*100:6.3f}%  "
                                  f"round-trips={mr['rt']:3d}  w(minor/minus)={mr['w_mean']:.3f}", flush=True)
    # pick b* with acc closest to 1% (and <5%)
    cand = [(b, mr) for b, mr in g1 if mr['de_acc'] < 0.05]
    b_star, base_mr = min((cand or g1), key=lambda x: abs(x[1]['de_acc'] - 0.01))
    print(f"  -> b* = {b_star} (linDE acc {base_mr['de_acc']*100:.3f}%, target 0.5-1.5%)", flush=True)
    store["gate1_b"] = np.array([x[0] for x in g1])
    store["gate1_acc"] = np.array([x[1]['de_acc'] for x in g1])
    store["gate1_rt"] = np.array([x[1]['rt'] for x in g1])
    store["b_star"] = b_star

    # ---------------- GATE C: compare moves at b* ----------------------------
    print(f"\n===== GATE C: move comparison at b*={b_star} =====", flush=True)
    t = CurvedTarget(D=D, b=b_star)
    Cg = jnp.asarray(t.within_mode_cov(0)); Lg = jnp.linalg.cholesky(Cg)
    init = balanced_init(t, SEED)
    rows = {}
    # baseline linear DE (re-measure at b* with same init/seed as the moves)
    base = make_composite(t.logp, D, NCH, L=10.0, step_size=0.05, K=K,
                          b0=0.1, p_jump=0.3, inverse_mass_matrix=Cg, eps_scale=Lg)
    rows["linDE"] = measure(base, t, init, SEED+1)
    for move in ["gamma1", "snooker"]:
        comp = make_teleport_composite(t.logp, D, NCH, L=10.0, step_size=0.05, K=K,
                                       move=move, b0=0.1, p_jump=0.5,
                                       inverse_mass_matrix=Cg, eps_scale=Lg)
        rows[move] = measure(comp, t, init, SEED+1)
    # near requires isotropic jitter -> use a small isotropic b0 in whitened... pass b0=0.1
    comp = make_teleport_composite(t.logp, D, NCH, L=10.0, step_size=0.05, K=K,
                                   move="near", b0=0.1, p_jump=0.5,
                                   inverse_mass_matrix=Cg, eps_scale=None)
    rows["near"] = measure(comp, t, init, SEED+1)

    for name, mr in rows.items():
        print(f"  {name:8s} move-acc={mr['de_acc']*100:6.3f}%  round-trips={mr['rt']:3d}  "
              f"crossings={mr['cross']:3d}  w(minor)={mr['w_mean']:.3f}", flush=True)
    rt_lin = max(rows["linDE"]["rt"], 1)
    print("\n  round-trip ratio vs linDE (>=10x needed to beat the wall):", flush=True)
    for name, mr in rows.items():
        print(f"    {name:8s}: {mr['rt']/rt_lin:.2f}x", flush=True)

    # ---------------- plots ----------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for name, mr in rows.items():
        ax[0].plot(mr["w_series"], lw=0.6, label=f"{name} (rt={mr['rt']})")
    ax[0].axhline(t.w[1], color='k', ls='--', label=f"truth minor {t.w[1]:.2f}")
    ax[0].set_xlabel("round"); ax[0].set_ylabel("frac chains in minor (minus) mode")
    ax[0].set_title(f"GATE C mixing on curved target (b*={b_star})"); ax[0].legend(fontsize=7)
    names = list(rows.keys())
    ax[1].bar(names, [rows[n]["rt"] for n in names])
    ax[1].set_ylabel("round-trips (16 chains)"); ax[1].set_title("mode round-trips by move")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "curved_gates.png"), dpi=110); plt.close(fig)

    store.update({f"rt_{n}": rows[n]["rt"] for n in names})
    store.update({f"acc_{n}": rows[n]["de_acc"] for n in names})
    store.update({f"w_{n}": rows[n]["w_mean"] for n in names})
    store.update({f"wser_{n}": rows[n]["w_series"] for n in names})
    np.savez(os.path.join(HERE, "curved_gates.npz"),
             **{k: np.asarray(v) for k, v in store.items()})
    summ = dict(b_star=b_star,
                gate1=[dict(b=x[0], linDE_acc_pct=x[1]['de_acc']*100, rt=x[1]['rt']) for x in g1],
                gateC={n: dict(move_acc_pct=rows[n]['de_acc']*100, round_trips=rows[n]['rt'],
                               crossings=rows[n]['cross'], w_minor=rows[n]['w_mean'],
                               rt_ratio_vs_linDE=rows[n]['rt']/rt_lin) for n in names})
    with open(os.path.join(HERE, "curved_gates.json"), "w") as f:
        json.dump(summ, f, indent=2, default=float)
    print(f"\nsaved curved_gates.png/.npz/.json  total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
