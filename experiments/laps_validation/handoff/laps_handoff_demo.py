r"""Runnable demo of the LAPS notebook handoff flow, using a MOCK gigalens model
(no real lens available here). Shows EXACTLY what the notebook flow produces:

    run_laps(prob_model, qz, init_mode="cold")  ->  diagnose(res)  ->  compare_warm_cold(...)

Run (in-container):
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 python3 laps_handoff_demo.py

The mock mimics the gigalens interface the wrappers consume:
  * ``prob_model.log_prob(z)`` -> (log_density, aux)  [LAPS takes [0]]
  * ``qz.sample((n,), seed) / .mean() / .covariance()``  (SVI surrogate)
A real notebook passes the actual gigalens ``prob_model`` + SVI ``qz`` instead.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp

from laps_handoff import run_laps, diagnose, compare_warm_cold


# ---- MOCK gigalens prob_model + qz (matches the gigalens interface) ---------
class _MockProbModel:
    def __init__(self, mean, cov):
        self.prec = jnp.asarray(np.linalg.inv(cov), jnp.float64)
        self.mean = jnp.asarray(mean, jnp.float64)

    def log_prob(self, z):                       # z: (dim,) single position
        d = z - self.mean
        return -0.5 * d @ self.prec @ d, jnp.asarray(0.0)   # (log_density, aux)


class _MockQZ:
    def __init__(self, mean, cov):
        self._mean = np.asarray(mean, np.float64)
        self._cov = np.asarray(cov, np.float64)
        self.dim = self._mean.shape[-1]
        self._chol = np.linalg.cholesky(self._cov + 1e-12 * np.eye(self.dim))

    def mean(self):
        return self._mean

    def covariance(self):
        return self._cov

    def sample(self, sample_shape=(), seed=None):
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        n = int(np.prod(sample_shape)) if sample_shape else 1
        z = jax.random.normal(seed, (n, self.dim), dtype=jnp.float64)
        x = jnp.asarray(self._mean) + z @ jnp.asarray(self._chol).T
        return x.reshape(sample_shape + (self.dim,)) if sample_shape else x[0]


def make_mock_gigalens_model(dim=8, seed=0):
    """A correlated SPD Gaussian standing in for a lens posterior in z-space."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, dim))
    cov = A @ A.T + dim * np.eye(dim)
    mean = rng.standard_normal(dim)
    return _MockProbModel(mean, cov), _MockQZ(mean, cov), mean, cov


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "demo_out")
    os.makedirs(out, exist_ok=True)
    prob_model, qz, mean, cov = make_mock_gigalens_model(dim=8, seed=0)
    names = [f"p{j}" for j in range(8)]

    print("=== 1) run_laps (cold, validated defaults) ===")
    res = run_laps(prob_model, qz, init_mode="cold", num_chains=512, seed=0)
    s = np.asarray(res.samples)
    s = s.reshape(-1, s.shape[-1])      # (M,K,d)->(M*K,d) for mean/cov; legacy (M,d) noop
    print(f"  samples shape={s.shape} finite={np.all(np.isfinite(s))} "
          f"switch@{res.switch_index}/{res.phase1_len} order={res.integrator_order}")

    print("\n=== 2) diagnose (ground-truth-free health) ===")
    health = diagnose(res, param_names=names,
                      out_png=os.path.join(out, "demo_diagnose.png"))

    print("\n=== 3) compare_warm_cold (cross-validation) ===")
    cmp = compare_warm_cold(prob_model, qz, param_names=names, num_chains=512, seed=0,
                            out_png=os.path.join(out, "demo_warm_cold.png"))

    # OPTIONAL sanity vs the known mock truth (a real lens has none; demo only)
    print("\n=== (demo-only) recovery vs known mock truth ===")
    print(f"  cold |mean_hat-mean|_max = {np.abs(s.mean(0)-mean).max():.3e}")
    print(f"  cold cov rel-err max     = "
          f"{np.abs(np.cov(s, rowvar=False)-cov).max()/np.abs(cov).max():.3e}")

    print("\nDEMO OK. PNGs:")
    print(" ", health["png"])
    print(" ", cmp["png"])


if __name__ == "__main__":
    main()
