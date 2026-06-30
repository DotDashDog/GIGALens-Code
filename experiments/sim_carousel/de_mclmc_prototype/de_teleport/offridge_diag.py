"""DECISIVE GEOMETRIC DIAGNOSTIC (plots before metrics): does a straight-chord
affine move land OFF the curved ridge, and does the gamma=1 NEAR-teleport escape?

This needs NO MCLMC -- it is pure proposal geometry + target density. It is the
cheapest test of the standing hypothesis ("every affine variant inherits the
chord-off-ridge failure").

PRE-REGISTRATION (method discipline)
------------------------------------
Cause hypothesis: the carousel's low within-mode DE acceptance is because a
proposal built from a STRAIGHT chord (z_a - z_b) between two on-ridge points lands
off the CURVED ridge. On the curved testbed, on-manifold points have off-ridge
Mahalanobis ~ sqrt(D-1) ~ 3.0; a chord proposal should land much further off.

Predictions (direction + magnitude), at b calibrated so gamma_big within-mode
single-step acceptance ~ 0.5-1.5% (carousel ~0.6%):
  P1  gamma_big within-mode   : off-ridge >> 3  (predict ~7-15), accept ~1%.
  P2  gamma=1 random-partner  : off-ridge EVEN larger (bigger chord), accept lower.
  P3  gamma=1 NEAR-teleport   : if it lands near an on-manifold chain a, off-ridge
        ~ O(3) and acceptance HIGH; this is the curvature-robust candidate.
        BUT the residual (z_i - z_b*) is a within-mode chord; if the nearest
        neighbour b* is not close, off-ridge stays large. We MEASURE which.
  P4  snooker cross-mode      : straight line through anchor -> off-ridge large for
        a curved target; accept low.

Falsifier of the whole curvature story: if gamma_big within-mode off-ridge ~ 3
(on-manifold) AND acceptance is high, curvature is NOT the killer and the testbed
is mis-calibrated (Gate-1 would also fail).

Primary output is the PLOT (off-ridge distributions + accept vs off-ridge); the
numbers are supportive. All verdicts PROPOSED / UNCERTIFIED.
"""
import os, sys, json
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from curved_testbed import CurvedTarget

SEED = 20260627
D = 10
NPAIR = 4000          # proposals per move type for statistics


def logp_batch(t, Y):
    """vectorized logp over (N,D) -> (N,)."""
    return np.asarray(jax.vmap(t.logp)(jnp.asarray(Y)))


def accept_rate(t, Z_cur, Z_prop, log_jac=None):
    lc = logp_batch(t, Z_cur); lp = logp_batch(t, Z_prop)
    dl = lp - lc
    if log_jac is not None:
        dl = dl + log_jac
    a = np.minimum(1.0, np.exp(np.clip(dl, -700, 50)))
    return float(a.mean()), dl


# ---------------- move proposal generators (numpy) ---------------------------
def gen_chord(Z_cur, pool, gamma, rng, near=False):
    """z' = z_i + gamma*(z_a - z_b). pool: complement positions (P,D).
    near=True picks b = nearest pool point to z_i (the near-teleport)."""
    N = Z_cur.shape[0]; P = pool.shape[0]
    a = rng.integers(0, P, N)
    if near:
        d2 = ((Z_cur[:, None, :] - pool[None, :, :]) ** 2).sum(-1)   # (N,P)
        b = d2.argmin(1)
    else:
        b = rng.integers(0, P - 1, N); b = np.where(b >= a, b + 1, b)
    return Z_cur + gamma * (pool[a] - pool[b]), a, b


def gen_snooker(Z_cur, pool, rng):
    """DREAM/ter-Braak&Vrugt snooker: anchor c, project (z_a-z_b) onto line (z_i-z_c),
    z' = z_i + gamma_s*zp ; returns z', log-jacobian (d-1)*log(||z'-z_c||/||z_i-z_c||)."""
    N = Z_cur.shape[0]; P = pool.shape[0]; D = Z_cur.shape[1]
    out = np.empty_like(Z_cur); logj = np.empty(N)
    for i in range(N):
        idx = rng.choice(P, 3, replace=False)
        za, zb, zc = pool[idx[0]], pool[idx[1]], pool[idx[2]]
        F = Z_cur[i] - zc
        FF = max(float(F @ F), 1e-300)
        zp = F * (((za - zb) @ F) / FF)
        gs = 1.2 + rng.random()
        zprop = Z_cur[i] + gs * zp
        XpZ = np.linalg.norm(zprop - zc) + 1e-300
        XZ = np.linalg.norm(Z_cur[i] - zc) + 1e-300
        out[i] = zprop; logj[i] = (D - 1) * np.log(XpZ / XZ)
    return out, logj


# =============================================================================
def main():
    rng = np.random.default_rng(SEED)
    store = {}

    # ---- GATE-1 calibration: sweep curvature b -> within-mode gamma_big accept
    g_big = 2.38 / np.sqrt(2 * D)
    print(f"gamma_big = {g_big:.4f}  (D={D})")
    print("\n[GATE-1 calib] within-mode (mode=plus) single-step gamma_big acceptance vs b:")
    b_grid = [0.0, 0.8, 1.5, 2.1, 3.0, 4.0, 6.0]
    calib = []
    for b in b_grid:
        t = CurvedTarget(D=D, b=b)
        Zc = t.exact_draws_mode(NPAIR, 0, rng)          # current (in-mode)
        pool = t.exact_draws_mode(64, 0, rng)           # in-mode complement
        Zp, _, _ = gen_chord(Zc, pool, g_big, rng)
        acc, _ = accept_rate(t, Zc, Zp)
        _, offc = t.offridge_decomp(Zc); _, offp = t.offridge_decomp(Zp)
        calib.append((b, acc, float(np.median(offp))))
        print(f"   b={b:4.1f}  accept={acc*100:6.3f}%   median off-ridge(prop)={np.median(offp):5.1f}"
              f"  (on-manifold ~{np.sqrt(D-1):.1f})")
    store["calib_b"] = np.array([c[0] for c in calib])
    store["calib_acc"] = np.array([c[1] for c in calib])
    store["calib_off"] = np.array([c[2] for c in calib])

    # pick b* : smallest b with accept in [0.5%,1.5%]; else closest to 1%
    in_band = [c for c in calib if 0.005 <= c[1] <= 0.015]
    if in_band:
        b_star = in_band[0][0]
    else:
        b_star = min(calib, key=lambda c: abs(c[1] - 0.01))[0]
    print(f"\n[GATE-1] chosen b* = {b_star}  (target within-mode gamma_big accept 0.5-1.5%)")
    store["b_star"] = b_star

    # ---- full off-ridge decomposition at b* ---------------------------------
    t = CurvedTarget(D=D, b=b_star)
    # within-mode regime (mixing): everything in mode plus
    Zc_w = t.exact_draws_mode(NPAIR, 0, rng)
    pool_w = t.exact_draws_mode(64, 0, rng)
    # cross-mode regime (teleport): current in plus, want to reach minus.
    Zc_x = t.exact_draws_mode(NPAIR, 0, rng)
    pool_minus = t.exact_draws_mode(64, 1, rng)        # complement = minus chains
    pool_plus = t.exact_draws_mode(64, 0, rng)         # same-mode chains (for near b*)

    results = {}
    _, off_true = t.offridge_decomp(t.exact_draws_mode(NPAIR, 0, rng))
    results["true_draws"] = dict(off=off_true, acc=1.0, dl=np.zeros(NPAIR))

    # within-mode moves
    for name, gamma, near in [("Wbig", g_big, False), ("W1_rand", 1.0, False),
                              ("W1_near", 1.0, True)]:
        Zp, a, b = gen_chord(Zc_w, pool_w, gamma, rng, near=near)
        acc, dl = accept_rate(t, Zc_w, Zp)
        _, offp = t.offridge_decomp(Zp)
        results[name] = dict(off=offp, acc=acc, dl=dl)
        print(f"  [within] {name:8s} accept={acc*100:6.3f}%  median off-ridge={np.median(offp):5.1f}")
    # within-mode snooker (same-mode pool)
    Zp, logj = gen_snooker(Zc_w, pool_w, rng)
    acc, dl = accept_rate(t, Zc_w, Zp, log_jac=logj)
    _, offp = t.offridge_decomp(Zp)
    results["Wsnooker"] = dict(off=offp, acc=acc, dl=dl)
    print(f"  [within] {'Wsnooker':8s} accept={acc*100:6.3f}%  median off-ridge={np.median(offp):5.1f}")

    # cross-mode teleport moves (target mode = minus); off-ridge measured vs nearest mode
    # gamma=1 random: z_i + (z_a[minus] - z_b[?]); for a teleport we need z_b ~ z_i (plus).
    # random partner b from minus pool -> NOT a teleport; near uses b*=nearest plus chain.
    Zp_r, a, b = gen_chord(Zc_x, np.concatenate([pool_plus, pool_minus]), 1.0, rng)  # random over all
    acc, dl = accept_rate(t, Zc_x, Zp_r); _, offp = t.offridge_decomp(Zp_r)
    frac_minus = (t.classify(Zp_r) == 1).mean()
    results["X1_rand"] = dict(off=offp, acc=acc, dl=dl, frac_minus=float(frac_minus))
    print(f"  [cross ] {'X1_rand':8s} accept={acc*100:6.3f}%  median off-ridge={np.median(offp):5.1f}"
          f"  frac landing in minus={frac_minus:.2f}")

    # near-teleport cross-mode: a from MINUS pool (target mode), b = nearest PLUS chain to z_i
    a_idx = rng.integers(0, pool_minus.shape[0], NPAIR)
    d2 = ((Zc_x[:, None, :] - pool_plus[None, :, :]) ** 2).sum(-1)
    b_idx = d2.argmin(1)
    Zp_n = Zc_x + (pool_minus[a_idx] - pool_plus[b_idx])     # ~ z_a(minus) + (z_i - z_b*)(plus)
    acc, dl = accept_rate(t, Zc_x, Zp_n); _, offp = t.offridge_decomp(Zp_n)
    frac_minus = (t.classify(Zp_n) == 1).mean()
    # how close is the nearest same-mode neighbour? (residual chord magnitude)
    resid = np.sqrt(((Zc_x - pool_plus[b_idx]) ** 2).sum(1))
    results["Xnear"] = dict(off=offp, acc=acc, dl=dl, frac_minus=float(frac_minus))
    print(f"  [cross ] {'Xnear':8s} accept={acc*100:6.3f}%  median off-ridge={np.median(offp):5.1f}"
          f"  frac->minus={frac_minus:.2f}  median residual||z_i-z_b*||={np.median(resid):.2f}")
    store["xnear_resid_med"] = float(np.median(resid))

    # ---------- PLOT (primary) ----------
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    order = ["true_draws", "Wbig", "W1_rand", "W1_near", "Wsnooker", "X1_rand", "Xnear"]
    data = [np.clip(results[k]["off"], 0, 60) for k in order]
    bp = ax[0].boxplot(data, labels=order, showfliers=False)
    ax[0].axhline(np.sqrt(D - 1), color="g", ls="--", label=f"on-manifold ~{np.sqrt(D-1):.1f}")
    ax[0].set_ylabel("OFF-ridge Mahalanobis (sigs units)")
    ax[0].set_title(f"Off-ridge of proposals (b*={b_star}); on-manifold=green")
    ax[0].legend(fontsize=8); ax[0].tick_params(axis='x', rotation=30)
    # accept vs median off-ridge
    accs = [results[k]["acc"] * 100 for k in order]
    offs = [np.median(results[k]["off"]) for k in order]
    ax[1].scatter(offs, accs)
    for k, x, y in zip(order, offs, accs):
        ax[1].annotate(k, (x, y), fontsize=8)
    ax[1].set_xlabel("median off-ridge Mahalanobis"); ax[1].set_ylabel("acceptance %")
    ax[1].set_title("acceptance falls as proposals leave the ridge")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "offridge_diag.png"), dpi=110); plt.close(fig)

    # calibration plot
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    ax[0].plot(store["calib_b"], store["calib_acc"] * 100, "o-")
    ax[0].axhspan(0.5, 1.5, color="g", alpha=0.2, label="target 0.5-1.5%")
    ax[0].axvline(b_star, color="r", ls=":"); ax[0].set_xlabel("curvature b"); ax[0].set_ylabel("within-mode gamma_big accept %")
    ax[0].set_title("GATE-1 calibration"); ax[0].legend(fontsize=8); ax[0].set_yscale("log")
    ax[1].plot(store["calib_b"], store["calib_off"], "s-", color="C2")
    ax[1].axhline(np.sqrt(D - 1), color="g", ls="--"); ax[1].set_xlabel("curvature b")
    ax[1].set_ylabel("median off-ridge of proposal"); ax[1].set_title("chord lands further off as b grows")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "offridge_calib.png"), dpi=110); plt.close(fig)

    # ---------- save ----------
    np.savez(os.path.join(HERE, "offridge_diag.npz"),
             **{k: np.asarray(v) for k, v in store.items()},
             **{f"off_{k}": results[k]["off"] for k in order},
             acc=np.array([results[k]["acc"] for k in order]),
             order=np.array(order))
    summ = {k: dict(acc_pct=results[k]["acc"] * 100,
                    off_median=float(np.median(results[k]["off"])),
                    off_p90=float(np.percentile(results[k]["off"], 90)))
            for k in order}
    summ["b_star"] = b_star; summ["on_manifold_offridge"] = float(np.sqrt(D - 1))
    with open(os.path.join(HERE, "offridge_diag.json"), "w") as f:
        json.dump(summ, f, indent=2, default=float)
    print("\nsaved offridge_diag.png / offridge_calib.png / .npz / .json")
    print(json.dumps(summ, indent=2, default=lambda o: float(o)))


if __name__ == "__main__":
    main()
