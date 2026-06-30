"""DIAGNOSTIC (model-free): is the carousel's within-mode geometry CURVED / non-Gaussian,
so that a LINEAR DE difference-vector proposal lands OFF the ridge (-> rejected)?

The Gaussian testbed gives healthy DE acceptance (8-15%) at every weight (E_weight_diag),
so the acceptance-killer is NOT the linear features it captures (scale mismatch, axis
alignment, anisotropy/cond, displacement) NOR the weight. The one thing a Gaussian fit
CANNOT represent is curvature of the within-mode ridge. A Gaussian is (nearly) closed
under the linear DE proposal prop = z_i + gamma*(z_a - z_b): linear combinations of
Gaussian draws stay on-distribution. A CURVED thin ridge is not: linear combinations
leave the ridge into a density void.

We test this with the REAL MCLMC samples only (no model / GPU). Whiten the GLOBAL cluster
by its OWN covariance (so the Gaussian model is isotropic-unit). For genuine samples and
for DE proposals (gamma=1 jump, gamma=gamma_big local) we measure the kNN distance to the
cloud. The expected spread inflation of a linear combo is identical for a Gaussian, so we
compare the REAL cluster to a SYNTHETIC Gaussian control with the SAME N and covariance.

Pre-registration:
  hypothesis: the global cluster is a curved/non-Gaussian thin ridge; linear DE proposals
    land off it.
  prediction: off-manifold ratio (median kNN of proposals / median kNN of genuine samples)
    is SUBSTANTIALLY larger for the REAL cluster than for the Gaussian control, for BOTH
    gamma=1 and gamma_big; the excess GROWS with gamma (bigger linear step -> further off
    a curved ridge).
  falsifier: real ratio ~ Gaussian ratio (within ~10%) -> the cluster is effectively
    Gaussian and curvature is NOT the killer; the 0.6%-vs-testbed gap is unexplained.
CPU, samples only. PROPOSED / UNCERTIFIED.
"""
import os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import carousel_testbed as T

SEED = 20260627
GAMMA_BIG = 2.38 / np.sqrt(2.0 * T.D)   # 0.4500


def pr(*a): print(*a, flush=True)


def whiten(X, mu, L):
    # L lower-Cholesky of cov; whitened = L^{-1}(x-mu)
    return np.linalg.solve(L, (X - mu).T).T


def knn_med(query, reference, k=5):
    tree = cKDTree(reference)
    d, _ = tree.query(query, k=k + 1)   # +1: drop self if query subset of ref (here disjoint)
    return np.median(d[:, 1:].mean(axis=1))


def de_proposals(W, gamma, n, rng):
    """W: whitened cloud (M,D). Draw n proposals z_i + gamma*(z_a - z_b)."""
    M = W.shape[0]
    i = rng.integers(0, M, n); a = rng.integers(0, M, n); b = rng.integers(0, M, n)
    return W[i] + gamma * (W[a] - W[b])


def off_manifold_ratio(W, gamma, rng, n=4000, k=5):
    M = W.shape[0]
    # reference = half the cloud; genuine query = other half; proposals from reference
    perm = rng.permutation(M)
    ref = W[perm[:M // 2]]
    genuine = W[perm[M // 2:M // 2 + n]]
    props = de_proposals(W[perm[:M // 2]], gamma, n, rng)
    d_gen = knn_med(genuine, ref, k)
    d_prop = knn_med(props, ref, k)
    return d_prop / d_gen, d_gen, d_prop


def main():
    t = time.time()
    mu, cov, sec, glob = T.fit_clusters()
    rng = np.random.default_rng(SEED)

    results = {}
    for cname, X, c in [("global", glob, cov[1]), ("secondary", sec, cov[0])]:
        L = np.linalg.cholesky(c)
        m = X.mean(0)
        Wr = whiten(X, m, L)                              # real whitened cloud (~unit cov)
        # synthetic Gaussian control: same N, identity cov in whitened space
        n_ctrl = min(Wr.shape[0], 20000)
        rng.shuffle(Wr); Wr = Wr[:n_ctrl]
        Wg = rng.standard_normal((n_ctrl, T.D))           # Gaussian control (unit)
        pr(f"\n=== {cname} cluster (N={X.shape[0]}, using {n_ctrl}) ===")
        pr(f"  whitened real cloud: mean|.|={np.abs(Wr.mean(0)).max():.3f} "
           f"var range [{Wr.var(0).min():.2f},{Wr.var(0).max():.2f}] (Gaussian->~1)")
        # non-Gaussianity scalars: mean abs skew/excess-kurtosis across whitened axes
        from scipy.stats import skew, kurtosis
        sk = np.abs(skew(Wr, axis=0)).mean(); ku = np.abs(kurtosis(Wr, axis=0)).mean()
        pr(f"  whitened |skew| mean={sk:.3f}  |excess-kurt| mean={ku:.3f} (Gaussian->0)")
        for gamma, gname in [(1.0, "jump g=1"), (GAMMA_BIG, f"local g={GAMMA_BIG:.3f}")]:
            r_real, dg_r, dp_r = off_manifold_ratio(Wr, gamma, rng)
            r_gau, dg_g, dp_g = off_manifold_ratio(Wg, gamma, rng)
            excess = r_real / r_gau
            pr(f"  {gname:18s} off-manifold ratio: real={r_real:.3f}  gauss={r_gau:.3f}  "
               f"REAL/GAUSS excess={excess:.3f}")
            results[f"{cname}_{gname}"] = (r_real, r_gau, excess)

    # plot: excess off-manifold ratio (real/gauss) for both clusters & both gammas
    fig, ax = plt.subplots(figsize=(9, 4.5))
    labels = list(results.keys()); ex = [results[k][2] for k in labels]
    rr = [results[k][0] for k in labels]; rg = [results[k][1] for k in labels]
    x = np.arange(len(labels)); wd = 0.35
    ax.bar(x - wd/2, rr, wd, label="real cluster")
    ax.bar(x + wd/2, rg, wd, label="Gaussian control")
    ax.axhline(1.0, color="k", ls=":", alpha=0.6)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("off-manifold ratio (median kNN prop / genuine)")
    ax.set_title("Linear DE proposals land OFF the real ridge but ON the Gaussian\n"
                 "(real >> Gaussian => curved/non-Gaussian within-mode geometry)")
    ax.legend(fontsize=9)
    for xi, k in zip(x, labels):
        ax.annotate(f"x{results[k][2]:.1f}", (xi, max(rr[x.tolist().index(xi)],
                    rg[x.tolist().index(xi)])), ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "E_curvature_diag.png"), dpi=110); plt.close(fig)
    np.savez(os.path.join(HERE, "E_curvature_diag.npz"),
             **{k: np.array(v) for k, v in results.items()})
    pr(f"\nDONE {time.time()-t:.1f}s -> E_curvature_diag.png / .npz")


if __name__ == "__main__":
    main()
