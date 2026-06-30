r"""Known-answer targets for the LAPS CPU validation harness.

Each target exposes, in the UNCONSTRAINED ``z`` space the LAPS sampler operates
on (``log_prob(z)`` with ``z`` shape ``(dim,)``):

* ``log_prob(z)`` -- scalar log density (unnormalized is fine; LAPS only uses
  gradients and the ensemble, never the normalizer).
* ground-truth per-dim moments used by ``metrics.py``:
    - ``mean_true``   (d,)  = E[x_i]
    - ``ex2_true``    (d,)  = E[x_i^2]      (the b^2 target moment)
    - ``var_x2_true`` (d,)  = Var[x_i^2]    (the b^2 denominator + noise floor)
    - ``var_true``    (d,)  = Var[x_i] = E[x_i^2] - E[x_i]^2  (for Var recovery)
* ``dim``.
* ``make_qz(mode)`` -> a Gaussian surrogate exposing ``.sample((n,), seed=key)``,
  ``.mean()``, ``.covariance()`` -- the chain initializer the sampler expects
  (matches ``qz.sample((num_chains,), seed=k_init)`` in ``LAPS_late_adjusted``).
    - mode="warm": moments ~ the true posterior (mean_true, full Cov).
    - mode="off" : inflated 2x (std) + shifted by +1 sigma  (for E-C).
    - mode="cold" is handled by the SAMPLER's cold init, not qz.

GROUND-TRUTH PROVENANCE (honest reporting):
  T_iso / T_ill / T_corr are zero-mean Gaussians, so every marginal moment is
  ANALYTIC: for x_i ~ N(0, s2_i),  E[x_i]=0, E[x_i^2]=s2_i, Var[x_i^2]=2 s2_i^2.
  T_curve (banana) has NO closed-form marginal moments here; its mean/ex2/var_x2
  are estimated from a LARGE exact reference sample (1e6 forward draws of the
  underlying Gaussian pushed through the unit-Jacobian twist) with a FIXED seed.
  The reference-sample standard error is reported in ``T_curve.moment_se`` so the
  grader can see the irreducible ground-truth uncertainty (it is << the b^2<0.01
  success scale at 1e6 draws).
"""

import numpy as np
import jax
import jax.numpy as jnp


# --------------------------------------------------------------------------- #
# Joint-shape ground truth from a reference sample (cross + curvature moments) #
# --------------------------------------------------------------------------- #
def _joint_truth_from_sample(x, curv_idx=(0, 1)):
    r"""Joint-shape truth from a large reference sample ``x`` (n_ref, d).

    Returns the cross second-moment matrix ``E[x_i x_j]`` and its per-entry
    sampling variance ``Var[x_i x_j]`` (the b^2-style denominator), PLUS the
    banana curvature monomial ``E[x_i^2 x_j]`` (``curv_idx=(i,j)``) and its
    variance ``Var[x_i^2 x_j]``. For the banana ``E[x_0^2 x_1] = 2 b`` analytically
    (Isserlis on the forward map); for a zero-mean Gaussian it is 0 -- the
    curvature signature is unique to the banana, which is the whole point.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    cross_true = (x.T @ x) / n                          # E[x_i x_j]  (d,d)
    # Var[x_i x_j] = E[(x_i x_j)^2] - E[x_i x_j]^2
    xx = np.einsum("ni,nj->nij", x, x)                  # (n,d,d) product per draw
    var_cross_true = xx.var(axis=0, ddof=0)             # (d,d)
    i, j = curv_idx
    m = x[:, i] ** 2 * x[:, j]                           # curvature monomial draws
    curv_true = float(m.mean())
    var_curv_true = float(m.var(ddof=0))
    return {
        "cross_true": cross_true,
        "var_cross_true": np.maximum(var_cross_true, 1e-30),
        "curv_idx": (int(i), int(j)),
        "curv_true": curv_true,
        "var_curv_true": max(var_curv_true, 1e-30),
        "curv_se": float(np.sqrt(var_curv_true / n)),
        "n_ref": int(n),
    }


# --------------------------------------------------------------------------- #
# Gaussian surrogate qz (the chain initializer LAPS consumes)                  #
# --------------------------------------------------------------------------- #
class GaussianQZ:
    """Minimal qz: ``.sample((n,), seed=key)`` / ``.mean()`` / ``.covariance()``."""

    def __init__(self, mean, cov):
        self._mean = np.asarray(mean, dtype=np.float64)
        self._cov = np.asarray(cov, dtype=np.float64)
        self.dim = self._mean.shape[-1]
        # symmetrize then Cholesky (jitter for numerical PSD safety)
        cov_sym = 0.5 * (self._cov + self._cov.T)
        jitter = 1e-12 * np.eye(self.dim)
        self._chol = np.linalg.cholesky(cov_sym + jitter)

    def mean(self):
        return self._mean

    def covariance(self):
        return self._cov

    def sample(self, sample_shape=(), seed=None):
        if seed is None:
            raise ValueError("GaussianQZ.sample requires an explicit jax PRNG `seed`.")
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        sample_shape = tuple(int(s) for s in sample_shape)
        n = int(np.prod(sample_shape)) if sample_shape else 1
        z = jax.random.normal(seed, (n, self.dim), dtype=jnp.float64)
        chol = jnp.asarray(self._chol, dtype=jnp.float64)
        mean = jnp.asarray(self._mean, dtype=jnp.float64)
        x = mean + z @ chol.T
        return x.reshape(sample_shape + (self.dim,)) if sample_shape else x[0]


# --------------------------------------------------------------------------- #
# Gaussian target base (analytic moments)                                      #
# --------------------------------------------------------------------------- #
class _GaussianTarget:
    """Mean-zero (by default) Gaussian with covariance ``cov``; analytic moments."""

    def __init__(self, name, cov, mean=None):
        self.name = name
        self.cov = np.asarray(cov, dtype=np.float64)
        self.dim = self.cov.shape[0]
        self.mean_vec = (np.zeros(self.dim) if mean is None
                         else np.asarray(mean, dtype=np.float64))
        self.prec = np.linalg.inv(self.cov)
        # analytic marginal moments
        diag = np.diag(self.cov)
        self.mean_true = self.mean_vec.copy()
        self.var_true = diag.copy()                       # Var[x_i] = Cov_ii
        self.ex2_true = self.mean_vec ** 2 + diag          # E[x_i^2]
        # Var[x_i^2] for x_i ~ N(m_i, s2_i): 2 s2_i^2 + 4 m_i^2 s2_i
        self.var_x2_true = 2.0 * diag ** 2 + 4.0 * self.mean_vec ** 2 * diag
        self.moment_se = np.zeros(self.dim)                # analytic -> no SE

        prec_j = jnp.asarray(self.prec, dtype=jnp.float64)
        mean_j = jnp.asarray(self.mean_vec, dtype=jnp.float64)

        def log_prob(z):
            d = z - mean_j
            return -0.5 * d @ prec_j @ d

        self.log_prob = log_prob
        self._joint_truth = None

    def joint_truth(self, n_ref=1_000_000, seed=20260629):
        """Joint-shape truth (cross + curvature moments) from a reference draw.

        Gaussian cross-moments are analytic but the cross/curvature VARIANCES
        (b^2-style denominators) need 4th moments, so we use a large fixed-seed
        reference sample for uniformity with the banana target. Cached."""
        if self._joint_truth is None:
            rng = np.random.default_rng(seed)
            x = rng.multivariate_normal(self.mean_vec, self.cov, size=n_ref)
            self._joint_truth = _joint_truth_from_sample(x)
        return self._joint_truth

    def make_qz(self, mode="warm"):
        if mode in ("warm", "cold"):
            return GaussianQZ(self.mean_true, self.cov)
        if mode == "off":
            # inflate covariance by 4x (2x in std) and shift mean by +1 true sigma
            shifted = self.mean_true + np.sqrt(np.diag(self.cov))
            return GaussianQZ(shifted, 4.0 * self.cov)
        raise ValueError(f"mode must be warm|off|cold, got {mode!r}")


# --------------------------------------------------------------------------- #
# Banana / Rosenbrock target (reference-sample moments)                        #
# --------------------------------------------------------------------------- #
class _BananaTarget:
    r"""Mild unimodal banana: a unit-Jacobian twist of an isotropic Gaussian.

    Base ``u ~ N(0, I_d)``.  Forward map (unit lower-triangular Jacobian):
        x_0 = u_0
        x_1 = u_1 + b (u_0^2 - 1)
        x_i = u_i             (i >= 2)
    so log p(x) = -1/2 [ x_0^2 + (x_1 - b(x_0^2 - 1))^2 + sum_{i>=2} x_i^2 ].
    Unimodal (smooth bijection of a Gaussian); curvature set by ``b`` (mild).
    Marginal moments have no clean closed form here -> estimated from 1e6 exact
    forward draws (fixed seed); the standard error is recorded in ``moment_se``.
    """

    def __init__(self, name="T_curve", dim=8, b=0.15, n_ref=1_000_000, seed=20260629):
        self.name = name
        self.dim = dim
        self.b = float(b)
        bj = jnp.asarray(self.b, dtype=jnp.float64)

        def log_prob(z):
            twist = z[1] - bj * (z[0] ** 2 - 1.0)
            quad = z[0] ** 2 + twist ** 2 + jnp.sum(z[2:] ** 2)
            return -0.5 * quad

        self.log_prob = log_prob

        # ground truth from a large exact reference sample of the forward map
        rng = np.random.default_rng(seed)
        u = rng.standard_normal((n_ref, dim))
        x = u.copy()
        x[:, 1] = u[:, 1] + self.b * (u[:, 0] ** 2 - 1.0)
        self.mean_true = x.mean(axis=0)
        self.var_true = x.var(axis=0, ddof=0)
        x2 = x ** 2
        self.ex2_true = x2.mean(axis=0)
        self.var_x2_true = x2.var(axis=0, ddof=0)
        # standard errors of the reference-sample moment estimates
        self.moment_se = np.sqrt(self.var_x2_true / n_ref)   # SE of E[x_i^2]
        # full reference covariance for the warm qz surrogate
        self._ref_cov = np.cov(x, rowvar=False)
        self.n_ref = n_ref
        # joint-shape truth from the SAME exact reference sample (curvature lives
        # in E[x_0^2 x_1] = 2b; the linear cross-cov E[x_0 x_1] is ~0 for the
        # banana, so marginal AND linear-correlation metrics are blind to it).
        self._joint_truth = _joint_truth_from_sample(x, curv_idx=(0, 1))

    def joint_truth(self, **_):
        return self._joint_truth

    def make_qz(self, mode="warm"):
        if mode in ("warm", "cold"):
            return GaussianQZ(self.mean_true, self._ref_cov)
        if mode == "off":
            shifted = self.mean_true + np.sqrt(np.diag(self._ref_cov))
            return GaussianQZ(shifted, 4.0 * self._ref_cov)
        raise ValueError(f"mode must be warm|off|cold, got {mode!r}")


# --------------------------------------------------------------------------- #
# Rotated, dense, ill-conditioned Gaussian (P6a stiff bed T_rot)               #
# --------------------------------------------------------------------------- #
class _RotatedGaussianTarget(_GaussianTarget):
    r"""Strongly-correlated, ill-conditioned, DENSE Gaussian, zero mean.

    Diagonal variances ``D`` log-spaced over ``1e-2 .. 1e2`` (cond ~1e4), then
    conjugated by a FIXED random rotation ``Q`` (orthogonal, QR of a fixed-seed
    matrix) so the covariance ``Sigma = Q D Q^T`` is DENSE (large off-diagonals).

    Ground truth is analytic (zero-mean Gaussian): per-dim
    ``E[x_i^2] = Sigma_ii`` (the diagonal of the rotated covariance) and
    ``Var[x_i^2] = 2 Sigma_ii^2`` -- both handled by ``_GaussianTarget``.

    THE POINT (diagonal-vs-dense open item): a diagonal ``1/Var`` preconditioner
    uses only ``diag(Sigma)`` and CANNOT whiten ``Sigma`` (its eigenbasis is the
    rotated ``Q``, not the axes). ``make_qz("warm")`` gives the FULL ``Sigma`` (so
    the chain *init* is perfect); the stress is entirely on Phase-2's diagonal
    metric.  ``make_qz("warm_diag")`` gives a diagonal-only surrogate
    (``diag(Sigma)``) for an axis-aligned-init contrast.
    """

    def __init__(self, name="T_rot", dim=12, cond=1e4, seed=11):
        # log-spaced eigen-variances spanning `cond` (1e-2 .. 1e2 at cond=1e4)
        hi = np.sqrt(cond)
        eig = np.logspace(np.log10(1.0 / hi), np.log10(hi), dim)
        # fixed random rotation Q (orthogonal) from QR of a fixed-seed matrix
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
        cov = (Q * eig) @ Q.T                       # Q diag(eig) Q^T, dense
        cov = 0.5 * (cov + cov.T)                    # symmetrize
        self.eig = eig
        self.Q = Q
        super().__init__(name, cov)

    def make_qz(self, mode="warm"):
        if mode == "warm_diag":
            # axis-aligned surrogate: same per-dim variances, NO off-diagonals.
            return GaussianQZ(self.mean_true, np.diag(np.diag(self.cov)))
        return super().make_qz(mode)


# --------------------------------------------------------------------------- #
# Factory: the four pre-registered beds + the two P6a stiff beds               #
# --------------------------------------------------------------------------- #
def make_target(name):
    name = name.lower()
    if name in ("t_iso", "iso"):
        return _GaussianTarget("T_iso", np.eye(8))
    if name in ("t_ill", "ill"):
        # diagonal, variances log-spaced 1e-1 .. 1e2 (cond ~1e3), d=8
        variances = np.logspace(-1, 2, 8)
        return _GaussianTarget("T_ill", np.diag(variances))
    if name in ("t_corr", "corr"):
        # fixed random PSD covariance, moderate off-diagonal correlation, d=8
        rng = np.random.default_rng(7)
        d = 8
        A = rng.standard_normal((d, d))
        corr = A @ A.T
        dinv = np.diag(1.0 / np.sqrt(np.diag(corr)))
        corr = dinv @ corr @ dinv                  # unit-diagonal correlation
        # moderate: shrink off-diagonals toward identity so |rho| stays sane
        corr = 0.5 * corr + 0.5 * np.eye(d)
        stds = np.sqrt(np.logspace(-0.5, 0.5, d))  # mild scale spread
        cov = np.outer(stds, stds) * corr
        return _GaussianTarget("T_corr", cov)
    if name in ("t_curve", "curve", "banana"):
        return _BananaTarget("T_curve", dim=8, b=0.15)
    # --- P6a stiff beds ---
    if name in ("t_rot", "rot"):
        # rotated, dense, ill-conditioned (cond ~1e4) Gaussian, d=12
        return _RotatedGaussianTarget("T_rot", dim=12, cond=1e4)
    if name in ("t_banana_hi", "banana_hi"):
        # STRONG banana (b=0.5, ~3.3x the T_curve curvature), unimodal, d=12
        return _BananaTarget("T_banana_hi", dim=12, b=0.5)
    raise ValueError(f"unknown target {name!r}")


ALL_TARGETS = ["T_iso", "T_ill", "T_corr", "T_curve"]
STIFF_TARGETS = ["T_rot", "T_banana_hi"]
