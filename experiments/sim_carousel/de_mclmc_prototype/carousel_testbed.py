"""CAROUSEL-FAITHFUL analytic testbed for the DE-MCLMC composite sampler.

Two Gaussians fit to the carousel's OWN clusters (split of the real MCLMC z-samples
by col9 at 4.40) are used as an analytic mixture target with EXACT logp and EXACT
sampler. This reproduces the real carousel geometry -- ~5x within-mode scale mismatch
between modes, top principal axes aligned to ~0.5 deg, within-mode cond ~2.7e6, and the
real inter-mode displacement -- but with a MEASURABLE weight ratio w=(0.3 sec, 0.7 glob)
(NOT the real ~1e-5) so that weight-recovery is a usable unbiasedness test.

The composite is driven by the REAL MCLMC kernel (_build_kernel_shardmap +
isokinetic_mclachlan_smart) with the HONEST global-mode covariance as the inverse mass
matrix -- structurally identical to the carousel composite, just analytic.

This module ONLY builds the target + helpers. It does NOT modify de_mclmc.py.
"""
import os, sys
import numpy as np
import jax, jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# ---------------------------------------------------------------- carousel data
ARRAYS = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/minimal_case/mclmc/arrays.npz"
COL = 9
THR = 4.40
D = 14
# honest MCLMC knobs used by the real carousel composite (basin_prep.npz)
L0 = 10.580742116860897
SS0 = 0.04838185714806612
# weights: index 0 = secondary (col9 < THR), index 1 = global (col9 > THR)
W = np.array([0.30, 0.70])


def fit_clusters():
    """Return (mu, cov) for the two clusters: [0]=secondary, [1]=global."""
    sz = np.load(ARRAYS)["samples_z"].reshape(-1, D)
    c9 = sz[:, COL]
    sec = sz[c9 < THR]
    glob = sz[c9 > THR]
    mu = np.stack([sec.mean(0), glob.mean(0)])
    cov = np.stack([np.cov(sec.T), np.cov(glob.T)])
    return mu, cov, sec, glob


# Build target object (JAX) ----------------------------------------------------
class Mixture:
    """Analytic 2-component Gaussian mixture with exact logp, sampler, classifier."""

    def __init__(self, mu, cov, w):
        self.mu = jnp.asarray(mu)              # (2, D)
        self.cov = jnp.asarray(cov)            # (2, D, D)
        self.w = jnp.asarray(w)               # (2,)
        self.logw = jnp.log(self.w)
        # Cholesky factors for logpdf (cond ~2.7e6 -> float64 Cholesky is fine)
        self.chol = jnp.linalg.cholesky(self.cov)               # lower L, (2,D,D)
        self.D = mu.shape[1]
        # log normalizers: -0.5*(D*log(2pi) + logdet) ; logdet = 2*sum(log diag L)
        logdet = 2.0 * jnp.sum(jnp.log(jnp.diagonal(self.chol, axis1=1, axis2=2)), axis=1)
        self.lognorm = -0.5 * (self.D * jnp.log(2 * jnp.pi) + logdet)   # (2,)
        # numpy copies for exact sampling on CPU
        self._mu_np = np.asarray(mu)
        self._chol_np = np.asarray(self.chol)
        self._w_np = np.asarray(w)

    def _comp_logpdf(self, z):
        # z: (D,) -> (2,) per-component log N
        d = z[None, :] - self.mu                      # (2,D)
        # solve L y = d  for each component
        y = jax.vmap(lambda Lc, dc: jax.scipy.linalg.solve_triangular(Lc, dc, lower=True))(
            self.chol, d)                             # (2,D)
        quad = jnp.sum(y * y, axis=1)                 # (2,)
        return self.lognorm - 0.5 * quad

    def logp(self, z):
        comp = self.logw + self._comp_logpdf(z)
        return jax.scipy.special.logsumexp(comp)

    def responsibility_logp(self, z):
        """Return the per-component (logw + logpdf) vector; argmax = hard assignment."""
        return self.logw + self._comp_logpdf(z)

    def classify(self, Z):
        """Hard mode assignment (0=sec,1=glob) by max posterior responsibility.
        Z: (..., D) numpy -> (...,) int."""
        Z = np.asarray(Z)
        flat = Z.reshape(-1, self.D)
        # vectorized responsibility on numpy for speed
        out = np.empty(flat.shape[0], int)
        for k in range(2):
            d = flat - self._mu_np[k]
            y = np.linalg.solve(self._chol_np[k], d.T).T
            quad = np.sum(y * y, axis=1)
            lp = float(self.lognorm[k]) + np.log(self._w_np[k]) - 0.5 * quad
            if k == 0:
                acc = lp[:, None]
            else:
                acc = np.concatenate([acc, lp[:, None]], axis=1)
        out = np.argmax(acc, axis=1)
        return out.reshape(Z.shape[:-1])

    def exact_draws(self, n, rng):
        """n exact mixture draws + their component labels."""
        comp = (rng.random(n) >= self._w_np[0]).astype(int)
        z = np.empty((n, self.D))
        std = rng.standard_normal((n, self.D))
        for i in range(n):
            k = comp[i]
            z[i] = self._mu_np[k] + self._chol_np[k] @ std[i]
        return z, comp

    def exact_draws_balanced(self, n, rng):
        """n draws with DETERMINISTIC counts round(w*n) per component (exact weight,
        no binomial init transient). Returns z, comp."""
        n_sec = int(round(self._w_np[0] * n))
        comp = np.ones(n, int)
        comp[:n_sec] = 0
        z = np.empty((n, self.D))
        std = rng.standard_normal((n, self.D))
        for i in range(n):
            k = comp[i]
            z[i] = self._mu_np[k] + self._chol_np[k] @ std[i]
        return z, comp


def build_target():
    mu, cov, sec, glob = fit_clusters()
    mix = Mixture(mu, cov, W)
    return mix, dict(mu=mu, cov=cov, sec=sec, glob=glob)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    mix, info = build_target()
    rng = np.random.default_rng(0)
    z, comp = mix.exact_draws(2000, rng)
    print("logp at a draw:", float(mix.logp(jnp.asarray(z[0]))))
    cls = mix.classify(z)
    print("classify vs true label agreement:", (cls == comp).mean())
    print("recovered weight(global):", cls.mean(), " target 0.70")
    # Mahalanobis separation: each mode's mean under the OTHER mode's metric
    for k, name in [(0, "sec"), (1, "glob")]:
        d = info["mu"][1 - k] - info["mu"][k]
        y = np.linalg.solve(np.asarray(mix._chol_np[k]), d)
        print(f"  sep: other-mean under {name} metric = {np.sqrt((y*y).sum()):.2f} sigma")
