"""DIAGNOSTIC: WHY did SA-MCMC freeze (0 hops, no draining) on the real carousel?
(C-17). Measure the mechanism instead of asserting it.

PRE-REGISTERED HYPOTHESIS (orchestrator):
  The secondary basin is ~1e-5 MASS but only ~7 logp below global per-point => it is
  tiny in VOLUME, so the 8 secondary chains pack tightly. SA deletes chain n with
  weight lambda_n = q(theta_n|others)/p(theta_n). Tight packing => high mutual KDE
  density q_sec; low target p_sec. If q_sec/q_glob ~ p_glob/p_sec (~e^7), the ratio
  CANCELS => lambda_sec ~ lambda_glob => secondary NOT preferentially deleted => no
  draining, no hops.
PREDICTION: log q_sec - log q_glob ~ +7 ; log p_glob - log p_sec ~ +7 ; lambda_sec ~
  lambda_glob (within ~1-2 nats); secondary pairwise distances << global.
FALSIFIER: lambda_sec >> lambda_glob (secondary strongly favored for deletion) yet no
  draining => mechanism is elsewhere (proposal landing / cross-mode), hypothesis wrong.

Also measured: within-mode packing (Mahalanobis pairwise), proposal landing (logp drop
of a KDE proposal, by source mode), and a BANDWIDTH sweep of the lambda gap. Transparent
re-implementation of the sa_move KDE/deletion (so it is instrumentable); verified to match.
"""
import os, sys, numpy as np, jax, jax.numpy as jnp, time
jax.config.update("jax_enable_x64", True)
SCR_DIAG = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/minimal_case_recheck_plots/_scripts"
HERE = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/de_mclmc_prototype"
sys.path.insert(0, SCR_DIAG); sys.path.insert(0, HERE)
from build_model import build
def pr(*a): print(*a, flush=True)

pm = build()
def logp1(z): return pm.log_prob(z)[0]
logp_v = jax.jit(jax.vmap(logp1))
D = 14; THR = 4.40; COL = 9
prep = np.load(os.path.join(SCR_DIAG, "basin_prep.npz"))
upper_cov = jnp.asarray(prep["upper_cov"])
z_best = np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/minimal_case/map/arrays.npz")["z_best"].reshape(-1)
sec_center = np.asarray(z_best)
sz = np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/minimal_case/mclmc/arrays.npz")["samples_z"]
flat = sz.reshape(-1, D); up_samples = flat[flat[:, COL] > THR]; lo_samples = flat[flat[:, COL] <= THR]
pr(f"n up(global) samples={up_samples.shape[0]}  n lo(secondary) samples={lo_samples.shape[0]}")

# Cholesky of the within-mode metric for KDE kernel + Mahalanobis
Lcov = jnp.linalg.cholesky(upper_cov)                       # upper_cov = L L^T
def maha_pairwise(X):                                       # Mahalanobis dists in upper_cov metric
    X = np.asarray(X); n = X.shape[0]; ds = []
    Linv_np = np.linalg.inv(np.asarray(Lcov))
    W = (Linv_np @ (X - X.mean(0)).T).T                     # whiten
    for i in range(n):
        for j in range(i+1, n):
            ds.append(np.linalg.norm(W[i]-W[j]))
    return np.array(ds)

def kde_logq_loo(S, bw):
    """log q(theta_n | others) over the other rows, KDE with H = bw^2 upper_cov. (Nx,)"""
    S = jnp.asarray(S); Nx = S.shape[0]
    Hc = bw * Lcov
    logdetH = 2.0 * jnp.sum(jnp.log(jnp.diagonal(Hc)))
    lognorm = -0.5 * (D * jnp.log(2*jnp.pi) + logdetH)
    def pair(n, m):
        d = S[n] - S[m]; w = jax.scipy.linalg.solve_triangular(Hc, d, lower=True)
        return lognorm - 0.5 * jnp.dot(w, w)
    G = jax.vmap(lambda n: jax.vmap(lambda m: pair(n, m))(jnp.arange(Nx)))(jnp.arange(Nx))
    G = G + jnp.diag(jnp.full((Nx,), -jnp.inf))            # exclude self
    return np.asarray(jax.scipy.special.logsumexp(G, axis=1) - jnp.log(Nx-1))

def main():
    t0 = time.time()
    rng = np.random.default_rng(0)
    # balanced ensemble exactly like the frozen run: 8 secondary (ball around z_best), 8 global
    sec = sec_center[None, :] + 1e-3 * rng.standard_normal((8, D))
    glob = np.asarray(up_samples[rng.choice(up_samples.shape[0], 8, replace=False)])
    S = np.concatenate([sec, glob], 0); is_sec = np.array([1]*8 + [0]*8)
    lp = np.asarray(logp_v(jnp.asarray(S)))
    pr("\n--- per-point target density (the p in lambda=q/p) ---")
    pr(f"  logp secondary: mean={lp[is_sec==1].mean():.2f}  | global: mean={lp[is_sec==0].mean():.2f}"
       f"  | gap(glob-sec)={lp[is_sec==0].mean()-lp[is_sec==1].mean():+.2f}  (predicted ~+7)")
    pr(f"  col9 sec={S[is_sec==1][:,COL].mean():.3f}  glob={S[is_sec==0][:,COL].mean():.3f} (THR {THR})")

    pr("\n--- within-mode PACKING (Mahalanobis pairwise in upper_cov metric) ---")
    dsec = maha_pairwise(sec); dglob = maha_pairwise(glob)
    pr(f"  secondary pairwise: median={np.median(dsec):.3f}  | global: median={np.median(dglob):.3f}"
       f"  | ratio glob/sec={np.median(dglob)/max(np.median(dsec),1e-9):.2f}x")

    pr("\n--- lambda = q/p decomposition, BANDWIDTH sweep (the core test) ---")
    pr("  bw     logq_sec  logq_glob  dq(sec-glob)   dp(glob-sec)   loglam_sec  loglam_glob  dlam(sec-glob)")
    rows = []
    for bw in [0.05, 0.1, 0.2, 0.5, 1.0]:
        logq = kde_logq_loo(S, bw)
        loglam = logq - lp                                 # lambda_n = q_n/p_n (deletion weight)
        qs, qg = logq[is_sec==1].mean(), logq[is_sec==0].mean()
        ls, lg = loglam[is_sec==1].mean(), loglam[is_sec==0].mean()
        dp = lp[is_sec==0].mean() - lp[is_sec==1].mean()
        pr(f"  {bw:4.2f}  {qs:8.2f}  {qg:8.2f}  {qs-qg:+11.2f}   {dp:+11.2f}   {ls:9.2f}  {lg:9.2f}  {ls-lg:+11.2f}")
        rows.append((bw, qs, qg, ls, lg, dp))
    pr("  [draining needs dlam(sec-glob) >> 0 (secondary preferentially deleted).")
    pr("   hypothesis: dq(sec-glob) ~ dp(glob-sec) ~ +7 => they cancel => dlam ~ 0 => NO draining.]")

    pr("\n--- proposal LANDING (KDE proposal y = theta + bw*chol(upper_cov)*xi; logp drop) ---")
    for bw in [0.05, 0.2, 0.5]:
        drops_s, drops_g, cross = [], [], 0
        for i in range(S.shape[0]):
            xi = rng.standard_normal((16, D))
            Y = S[i][None,:] + bw * (xi @ np.asarray(Lcov).T)
            lpy = np.asarray(logp_v(jnp.asarray(Y)))
            drop = lpy - lp[i]
            (drops_s if is_sec[i] else drops_g).append(drop.mean())
            cross += int(np.sum((Y[:,COL] > THR) != (S[i,COL] > THR)))
        pr(f"  bw={bw:4.2f}: logp drop sec={np.mean(drops_s):+.2f}  glob={np.mean(drops_g):+.2f}"
           f"  | within-mode proposals cross THR: {cross}/{S.shape[0]*16}")

    pr(f"\nwall {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
