"""CURVED representative testbed: a bimodal target whose WITHIN-MODE geometry is
a curved thin ridge, so that a straight difference/chord vector between two
on-ridge points lands OFF the ridge.  This is the feature the Gaussian testbeds
do NOT have and that defeats every AFFINE ensemble move (DE / snooker / kernel
mode-matching).  The Gaussian carousel_testbed gives ~8.7% linear-DE acceptance;
the REAL carousel gives ~0.6% -- the gap is within-mode curvature, reproduced here.

CONSTRUCTION (smooth invertible TRIANGULAR warp -> analytic logp, unit Jacobian)
-------------------------------------------------------------------------------
Base space X (D dims), 2-Gaussian mixture, weights w (comparable masses by default):
  mode A: x1 ~ N(+Delta, s_thin^2);  mode B: x1 ~ N(-Delta, s_thin^2)
  x0 ~ N(0, s_ridge^2)   (the RIDGE direction: broad);  x2..x_{D-1} ~ N(0,s_thin^2)
  (each mode is a thin sheet extended along x0 -> a ridge; modes separated along x1)
Triangular "banana" warp f:  y0 = x0;   y_k = x_k + b*(x0^2 - s_ridge^2), k>=1.
  As you travel ALONG a ridge (x0 varies), every soft coord bows by b*x0^2 -> the
  ridge is CURVED; a straight chord between two on-ridge points at x0=u,v lands
  off-ridge by ~ b*(u^2 - v^2) in every soft direction.
Inverse:  x0 = y0;  x_k = y_k - b*(y0^2 - s_ridge^2).  J_{f^-1} is triangular with
  UNIT diagonal -> |det J| = 1, so  log p_Y(y) = log p_X(f^-1(y)).
exact_mixture_draws: draw x from the base mixture, push y = f(x).

GATE-1 (anti-gaming): tune b so a LINEAR DE move has WITHIN-mode acceptance
~0.5-1.5% (reproducing the carousel's ~0.6%).  measure_linear_de_acceptance()
drives de_mclmc.make_composite from a single-mode init and returns the DE accept.

This module ONLY builds the target + helpers; modifies nothing shared.
"""
import os, sys
import numpy as np
import jax, jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))


class CurvedBimodal:
    """Bimodal curved-ridge target with exact logp (jax), exact draws, classifier."""

    def __init__(self, D=10, w=(0.6, 0.4), Delta=4.0, s_ridge=3.0, s_thin=0.30,
                 b=0.6, n_curve=2):
        # dim0 = ridge (broad);  dim1 = mode-separation (thin);  dims 2..2+n_curve-1
        # = CURVED thin dims (get the banana bow);  remaining dims = thin nuisance.
        # Bowing only a FEW dims (not all D-1) keeps the curvature in a low-dim
        # subspace as in the real carousel, and keeps the calibration well-behaved.
        self.D = D
        self.w = np.asarray(w, float)
        self.Delta = Delta; self.s_ridge = s_ridge; self.s_thin = s_thin; self.b = b
        self.n_curve = n_curve
        self.curve_dims = np.arange(2, 2 + n_curve)
        var = np.full(D, s_thin ** 2); var[0] = s_ridge ** 2
        self.var = var
        mu = np.zeros((2, D)); mu[0, 1] = +Delta; mu[1, 1] = -Delta   # separated along x1
        self.mu = mu
        cmask = np.zeros(D); cmask[self.curve_dims] = 1.0
        self._cmask = jnp.asarray(cmask); self._cmask_np = cmask
        # jax constants
        self._logw = jnp.log(jnp.asarray(self.w))
        self._mu = jnp.asarray(mu); self._var = jnp.asarray(var)
        self._s2 = float(s_ridge ** 2); self._b = float(b)
        self._lognorm = -0.5 * (D * jnp.log(2 * jnp.pi) + jnp.sum(jnp.log(self._var)))

    # ---- warp & inverse (bow only the curve_dims) ----------------------------
    def finv(self, y):
        return y - self._cmask * (self._b * (y[0] ** 2 - self._s2))
    def f_np(self, x):
        bow = self.b * (x[..., 0:1] ** 2 - self._s2)            # (...,1)
        return x + self._cmask_np[None, :] * bow if x.ndim > 1 else x + self._cmask_np * bow

    # ---- target logp in Y (unit Jacobian) ------------------------------------
    def logp(self, y):
        x = self.finv(y)
        d = x[None, :] - self._mu                      # (2,D)
        quad = jnp.sum(d * d / self._var[None, :], axis=1)
        comp = self._logw + self._lognorm - 0.5 * quad
        return jax.scipy.special.logsumexp(comp)

    # ---- exact sampling ------------------------------------------------------
    def _draw_x(self, comp, rng):
        n = len(comp)
        x = rng.standard_normal((n, self.D)) * np.sqrt(self.var)[None, :]
        x += self.mu[comp]
        return x
    def exact_draws(self, n, rng):
        comp = (rng.random(n) >= self.w[0]).astype(int)   # 0=modeA,1=modeB
        return self.f_np(self._draw_x(comp, rng)), comp
    def exact_draws_balanced(self, n, rng):
        nA = int(round(self.w[0] * n)); comp = np.ones(n, int); comp[:nA] = 0
        return self.f_np(self._draw_x(comp, rng)), comp

    # ---- hard classifier (which mode) ----------------------------------------
    def classify(self, Y):
        Y = np.asarray(Y); flat = Y.reshape(-1, self.D)
        X = flat - self._cmask_np[None, :] * (self.b * (flat[:, 0:1] ** 2 - self._s2))
        lp = np.empty((flat.shape[0], 2))
        for k in range(2):
            d = X - self.mu[k]
            lp[:, k] = np.log(self.w[k]) - 0.5 * np.sum(d * d / self.var[None, :], axis=1)
        return np.argmax(lp, axis=1).reshape(Y.shape[:-1])

    def within_mode_cov(self, mode=1, n=20000, seed=0):
        """Empirical within-mode covariance in Y (for a curvature-aware kernel)."""
        rng = np.random.default_rng(seed)
        comp = np.full(n, mode, int)
        Y = self.f_np(self._draw_x(comp, rng))
        return np.cov(Y.T)


def build_target(**kw):
    return CurvedBimodal(**kw)


def static_linear_de_acceptance(tgt, n=20000, mode=1, p_jump=0.5, b0=0.05, seed=5):
    """STATIC GATE-1 metric (unconfounded by MCLMC): mean Metropolis acceptance of
    a LINEAR DE difference-vector jump between ON-RIDGE points of one curved mode.
    Draw on-ridge triples (z_i, z_a, z_b); propose z_i + gamma*(z_a-z_b) + jitter
    with gamma = 1.0 w.p. p_jump else 2.38/sqrt(2D) (exactly de_mclmc's scheme);
    return mean min(1, p(prop)/p(z_i)).  This is what an affine ensemble move 'sees'."""
    rng = np.random.default_rng(seed)
    D = tgt.D
    comp = np.full(3 * n, mode, int)
    Y = tgt.f_np(tgt._draw_x(comp, rng)).reshape(3, n, D)
    zi, za, zb = Y[0], Y[1], Y[2]
    gamma_big = 2.38 / np.sqrt(2.0 * D)
    gamma = np.where(rng.random(n) < p_jump, 1.0, gamma_big)[:, None]
    jit = b0 * rng.standard_normal((n, D))
    prop = zi + gamma * (za - zb) + jit
    lp = jax.vmap(tgt.logp)
    dlp = np.asarray(lp(jnp.asarray(prop))) - np.asarray(lp(jnp.asarray(zi)))
    return float(np.mean(np.minimum(1.0, np.exp(np.minimum(0.0, dlp)))))


# ----------------- GATE-1: linear-DE within-mode acceptance ------------------
def measure_linear_de_acceptance(tgt, n_chains=64, K=20, rounds=300,
                                 b0=0.05, p_jump=0.5, L=2.0, step=0.5, seed=1,
                                 inverse_mass_matrix=None):
    """Drive de_mclmc.make_composite from a SINGLE-mode (curved mode B) init and
    return the mean DE acceptance = WITHIN-mode linear-DE acceptance."""
    sys.path.insert(0, os.path.dirname(HERE))
    from de_mclmc import make_composite
    D = tgt.D
    comp = make_composite(tgt.logp, D, n_chains, L=L, step_size=step, K=K,
                          b0=b0, p_jump=p_jump, inverse_mass_matrix=inverse_mass_matrix)
    rng = np.random.default_rng(seed)
    init, _ = tgt.exact_draws_balanced(n_chains, rng)
    # keep ONLY mode-B points -> all within one curved mode
    lbl = tgt.classify(init)
    initB = init[lbl == 1]
    # replicate to fill n_chains
    reps = int(np.ceil(n_chains / max(1, len(initB))))
    initB = np.tile(initB, (reps, 1))[:n_chains]
    st = comp["init_states"](jnp.asarray(initB), jax.random.key(seed))
    keys = jax.random.split(jax.random.key(seed + 1), rounds)
    accs = []
    import numpy as _np
    for r in range(rounds):
        st, (p, ec, acc) = comp["round"](st, keys[r])
        accs.append(float(_np.asarray(acc).mean()))
    return float(np.mean(accs[rounds // 3:]))   # drop transient


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    tgt = build_target()
    rng = np.random.default_rng(0)
    y, comp = tgt.exact_draws(5000, rng)
    print("classify agreement:", (tgt.classify(y) == comp).mean(),
          " recovered w(A):", (tgt.classify(y) == 0).mean(), " target", tgt.w[0])
    # curvature check: straight chord between two on-ridge mode-B points -> off-ridge?
    rngc = np.random.default_rng(2)
    cB = np.ones(400, int); X = tgt._draw_x(cB, rngc); Y = tgt.f_np(X)
    # pick pairs, midpoint of chord, measure log-density drop vs endpoints
    import numpy as _np
    a, b = Y[:200], Y[200:400]
    mid = 0.5 * (a + b)
    lp = jax.vmap(tgt.logp)
    drop = _np.asarray(lp(jnp.asarray(mid))) - 0.5 * (_np.asarray(lp(jnp.asarray(a))) + _np.asarray(lp(jnp.asarray(b))))
    print(f"chord-midpoint mean logp drop vs endpoints (curvature): {drop.mean():.2f} nats (more negative = more curved)")
    print("logp finite:", _np.isfinite(_np.asarray(lp(jnp.asarray(y[:50])))).all())
