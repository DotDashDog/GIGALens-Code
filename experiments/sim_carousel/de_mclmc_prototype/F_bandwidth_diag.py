"""DIAGNOSTIC (not a fix): isolate WHY kernel-hop acceptance ~ 0 on the D=14
carousel-faithful testbed. Hypothesis: too-narrow KDE -> q_C(z_i)~0 at the
current (non-center) point while q_C(z')~large at the on-center proposal ->
Hastings ratio q(z_i)/q(z') ~ 0 -> systematic rejection. Prediction: acceptance
rises sharply once eps >~ 1 (kernels overlap); cross-mode hops into the tight
secondary stay low (global-cov kernel overshoots). Falsifier: accept stays ~0 at
eps=2..4 -> not overlap, real bug.

Measures, for a BALANCED complement (8 sec + 8 glob exact draws):
  - median log q_C(z_i) for z_i typical SAME-mode points  vs  log q_C(z') for
    on-center proposals  (the Hastings gap)
  - WITHIN-mode hop acceptance (propose from same-mode centers)
  - CROSS-mode hop acceptance (propose from other-mode centers)
over eps in {0.25,0.5,1,2,4}. Pure proposal/accept algebra, no MCLMC, fast.
"""
import os, sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import carousel_testbed as T
def pr(*a): print(*a, flush=True)

mu_np, cov_np, sec, glob = T.fit_clusters()
mix = T.Mixture(mu_np, cov_np, np.array([0.3, 0.7]))
Cg = np.asarray(cov_np[1]); Lg = np.linalg.cholesky(Cg)
D = T.D

def kde_logq(z, centers, eps):
    # (1/m) sum N(z; c_j, eps^2 Cg) using Cg-Cholesky
    diff = (z[None, :] - centers)                    # (m,D)
    w = np.linalg.solve(Lg, diff.T).T                # whiten by Cg
    quad = (w*w).sum(1) / eps**2
    lognorm = -0.5*(D*np.log(2*np.pi) + 2*D*np.log(eps) + 2*np.sum(np.log(np.diag(Lg))))
    from scipy.special import logsumexp
    return logsumexp(lognorm - 0.5*quad) - np.log(centers.shape[0])

rng = np.random.default_rng(0)
# balanced complement: 8 sec + 8 glob exact draws
csec = mu_np[0] + rng.standard_normal((8, D)) @ np.asarray(mix._chol_np[0]).T
cglob = mu_np[1] + rng.standard_normal((8, D)) @ np.asarray(mix._chol_np[1]).T
comp = np.vstack([csec, cglob])
# typical current points in each mode
zglob = mu_np[1] + rng.standard_normal((200, D)) @ np.asarray(mix._chol_np[1]).T
zsec = mu_np[0] + rng.standard_normal((200, D)) @ np.asarray(mix._chol_np[0]).T

def logp(z): return float(mix.logp(jnp.asarray(z)))

pr(f"{'eps':>5} | {'med logq(z_i,glob)':>18} {'med logq(zprop_glob)':>20} | "
   f"{'within-acc%':>11} {'cross-acc%(glob->sec)':>21}")
for eps in [0.25, 0.5, 1.0, 2.0, 4.0]:
    # Hastings gap: logq at typical glob current points vs at on-center proposals
    lq_cur = np.array([kde_logq(z, comp, eps) for z in zglob[:60]])
    # on-center proposals from glob centers
    props = []
    for _ in range(60):
        j = rng.integers(8, 16); props.append(comp[j] + eps*(Lg @ rng.standard_normal(D)))
    lq_prop = np.array([kde_logq(p, comp, eps) for p in props])
    # WITHIN-mode acceptance: current z_i in glob, propose from glob centers
    nacc_w = 0; ntot = 120
    for _ in range(ntot):
        zi = zglob[rng.integers(0, 200)]
        j = rng.integers(8, 16); zp = comp[j] + eps*(Lg @ rng.standard_normal(D))
        la = (logp(zp)-logp(zi)) + (kde_logq(zi, comp, eps)-kde_logq(zp, comp, eps))
        if np.log(rng.random()) < la: nacc_w += 1
    # CROSS-mode acceptance: current z_i in glob, propose from SEC centers (hop to tight mode)
    nacc_c = 0
    for _ in range(ntot):
        zi = zglob[rng.integers(0, 200)]
        j = rng.integers(0, 8); zp = comp[j] + eps*(Lg @ rng.standard_normal(D))
        la = (logp(zp)-logp(zi)) + (kde_logq(zi, comp, eps)-kde_logq(zp, comp, eps))
        if np.log(rng.random()) < la: nacc_c += 1
    pr(f"{eps:>5} | {np.median(lq_cur):>18.1f} {np.median(lq_prop):>20.1f} | "
       f"{100*nacc_w/ntot:>11.1f} {100*nacc_c/ntot:>21.1f}")
pr("\nInterpretation: if within-acc jumps from ~0 to sizeable as eps crosses ~1,"
   "\nthe cause is KDE non-overlap (q(z_i)~0), confirming the bandwidth hypothesis.")
