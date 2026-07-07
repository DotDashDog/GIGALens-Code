"""CURVED bimodal testbed: a 2-component Gaussian mixture pushed through a smooth
INVERTIBLE triangular ("banana"/Rosenbrock) warp so that the WITHIN-MODE geometry
is a curved thin ridge. A straight chord between two on-ridge points lands well
OFF the ridge -- reproducing the real carousel's chord-off-ridge difficulty
(which the Gaussian carousel_testbed.py does NOT, because that difficulty is
SEPARATION, not curvature).

GEOMETRY
--------
Base ("latent") space x: 2-Gaussian mixture, DIAGONAL covariance.
  * mode 0 ("plus")  : mean [+m, 0, ..., 0], weight w0
  * mode 1 ("minus") : mean [-m, 0, ..., 0], weight 1-w0
  * within-mode std:  axis 0 = sig0 (the LONG ridge direction),
                      soft axes 1..D-1 = sigs (THIN), per-mode scale via `mode_scale`.
Warp f : x -> y (triangular, lower):
      y0 = x0
      y_k = x_k + b * c_k * (x0^2 - s^2),   k = 1..D-1
  Inverse f^{-1}: x0 = y0 ; x_k = y_k - b*c_k*(y0^2 - s^2).
  Jacobian J_{f^{-1}} is lower-triangular with UNIT diagonal => |det| = 1, so
      log p_Y(y) = log p_X(f^{-1}(y))   (exact, by change of variables).
  s := m  => the warp vanishes at BOTH mode centers (x0=+-m, x0^2=m^2=s^2), so the
  two modes stay separated cleanly along axis 0 and the curvature is purely the
  LOCAL within-mode ridge bending (controlled by b).

WHY THIS IS A CURVED RIDGE (and tunable)
----------------------------------------
Within mode `plus`, x0 ranges ~ m +- sig0 and the (thin) soft coords ~ N(0,sigs^2).
The image traces a parabola y_k ~ b c_k (x0^2 - s^2) of soft-direction "rise"
~ b c_k * 2 m sig0 across the ridge, with thickness sigs. A straight chord between
two ridge points at x0 = m +- sig0 deviates from the parabola by ~ b c_k sig0^2 at
the midpoint. The OFF-RIDGE ratio (chord deviation / soft thickness) ~ b sig0^2/sigs,
so curvature strength `b` is the single knob calibrated in GATE-1.

OFF-RIDGE DECOMPOSITION (clean, no tangent algebra)
---------------------------------------------------
Pull any proposal y' back to x' = f^{-1}(y'). In x-space the mode is an axis-aligned
Gaussian, so the Mahalanobis distance to the nearest mode splits exactly into:
  * ALONG-ridge   = |x'0 - (+-m)| / sig0          (axis 0)
  * OFF-ridge     = ||x'_soft||  in sigs units     (soft axes)
A proposal is "on-manifold" iff its OFF-ridge Mahalanobis is O(1); a chord-off-ridge
failure shows up as OFF-ridge >> 1. `offridge_decomp` returns both.

This module ONLY builds the target + helpers. It imports nothing from the MCLMC
stack and modifies no shared module.
"""
import numpy as np
import jax, jax.numpy as jnp


class CurvedTarget:
    def __init__(self, D=10, m=5.0, sig0=1.0, sigs=0.3, b=2.1, w0=0.6,
                 c=None, mode_scale=(1.0, 1.0), n_warp=None):
        self.D = D
        self.m = float(m)
        self.sig0 = float(sig0)
        self.sigs = float(sigs)
        self.b = float(b)
        self.s2 = self.m ** 2          # s = m  -> warp vanishes at mode centers
        self.w = np.array([w0, 1.0 - w0], float)
        self.mode_scale = np.asarray(mode_scale, float)   # per-mode soft-scale multiplier
        # warp coefficient per soft axis. Default: warp the first `n_warp` soft axes.
        if c is None:
            c = np.zeros(D)
            nw = (D - 1) if n_warp is None else int(n_warp)
            # mixed-magnitude coefficients so the curved direction is not a single
            # symmetric (1,1,..) ray (closer to a low-rank but non-trivial bend).
            base = np.array([1.0, 0.7, -0.5, 0.9, -0.6, 0.4, 0.8, -0.3, 0.55])
            for k in range(1, 1 + nw):
                c[k] = base[(k - 1) % len(base)]
        self.c = np.asarray(c, float)              # (D,), c[0]=0

        # latent means / per-mode diagonal std
        self.mu_x = np.zeros((2, D)); self.mu_x[0, 0] = +m; self.mu_x[1, 0] = -m
        self.std_x = np.zeros((2, D))
        for k in range(2):
            self.std_x[k, 0] = sig0
            self.std_x[k, 1:] = sigs * self.mode_scale[k]

        # jax copies
        self._c = jnp.asarray(self.c)
        self._mu_x = jnp.asarray(self.mu_x)
        self._std_x = jnp.asarray(self.std_x)
        self._logw = jnp.log(jnp.asarray(self.w))
        self._s2 = float(self.s2)
        self._bc = self.b * self._c                # (D,)

    # ---- warp & inverse (triangular, |detJ|=1) ------------------------------
    def f_np(self, x):
        x = np.asarray(x)
        off = self.b * self.c * (x[..., :1] ** 2 - self.s2)
        off[..., 0] = 0.0
        return x + off

    def finv_np(self, y):
        y = np.asarray(y)
        off = self.b * self.c * (y[..., :1] ** 2 - self.s2)
        off[..., 0] = 0.0
        return y - off

    def finv_jax(self, y):
        off = self._bc * (y[..., :1] ** 2 - self._s2)
        off = off.at[..., 0].set(0.0)
        return y - off

    # ---- target log density (per single y vector, for the MCLMC kernel) -----
    def logp(self, y):
        # y:(D,). x = f^{-1}(y); |detJ|=1.
        off = self._bc * (y[0] ** 2 - self._s2)
        off = off.at[0].set(0.0)
        x = y - off
        d = x[None, :] - self._mu_x                       # (2,D)
        z = d / self._std_x                               # whiten (diag)
        quad = jnp.sum(z * z, axis=1)                     # (2,)
        logdet = 2.0 * jnp.sum(jnp.log(self._std_x), axis=1)   # (2,)
        lognorm = -0.5 * (self.D * jnp.log(2 * jnp.pi) + logdet)
        comp = self._logw + lognorm - 0.5 * quad
        return jax.scipy.special.logsumexp(comp)

    # ---- exact draws (push base mixture through f) --------------------------
    def exact_draws(self, n, rng):
        comp = (rng.random(n) >= self.w[0]).astype(int)     # 0=plus,1=minus
        x = self.mu_x[comp] + self.std_x[comp] * rng.standard_normal((n, self.D))
        return self.f_np(x), comp

    def exact_draws_balanced(self, n, rng):
        n0 = int(round(self.w[0] * n))
        comp = np.ones(n, int); comp[:n0] = 0
        x = self.mu_x[comp] + self.std_x[comp] * rng.standard_normal((n, self.D))
        return self.f_np(x), comp

    def exact_draws_mode(self, n, mode, rng):
        x = self.mu_x[mode] + self.std_x[mode] * rng.standard_normal((n, self.D))
        return self.f_np(x)

    # ---- classify by latent axis-0 sign (modes at +-m) ----------------------
    def classify(self, Y):
        x = self.finv_np(np.asarray(Y))
        return (x[..., 0] < 0).astype(int)                  # 0=plus,1=minus

    # ---- off-ridge / along-ridge Mahalanobis decomposition ------------------
    def offridge_decomp(self, Y):
        """For points Y (...,D): pull back to x, assign to nearest mode, return
        (along_ridge, off_ridge) Mahalanobis in (sig0, sigs) units."""
        Y = np.asarray(Y); shp = Y.shape[:-1]
        x = self.finv_np(Y).reshape(-1, self.D)
        out_along = np.empty(x.shape[0]); out_off = np.empty(x.shape[0])
        for k in range(2):
            zk = (x - self.mu_x[k]) / self.std_x[k]
            mh = np.sqrt((zk ** 2).sum(1))
            if k == 0:
                best = mh.copy(); along = np.abs(zk[:, 0]); off = np.sqrt((zk[:, 1:] ** 2).sum(1))
            else:
                better = mh < best
                best = np.minimum(best, mh)
                along = np.where(better, np.abs(zk[:, 0]), along)
                off = np.where(better, np.sqrt((zk[:, 1:] ** 2).sum(1)), off)
        return along.reshape(shp), off.reshape(shp)

    def within_mode_cov(self, mode=0, n=20000, seed=0):
        """Empirical y-space within-mode covariance (honest MCLMC preconditioner)."""
        rng = np.random.default_rng(seed)
        Y = self.exact_draws_mode(n, mode, rng)
        return np.cov(Y.T)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    t = CurvedTarget()
    rng = np.random.default_rng(0)
    Y, comp = t.exact_draws(5000, rng)
    # sanity: classify agreement, density finite, off-ridge of TRUE draws ~ O(1)
    cls = t.classify(Y)
    print("classify vs true label:", (cls == comp).mean())
    print("logp at a draw:", float(t.logp(jnp.asarray(Y[0]))))
    along, off = t.offridge_decomp(Y)
    print(f"TRUE-draw off-ridge Mahal: mean={off.mean():.2f} (expect ~sqrt(D-1)={np.sqrt(t.D-1):.2f})")
    print(f"recovered weight(plus): {(cls==0).mean():.3f} target {t.w[0]}")
