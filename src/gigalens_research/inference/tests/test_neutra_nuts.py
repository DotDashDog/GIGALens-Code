"""Tests for gigalens_research.inference.neutra_nuts.

Runnable BOTH via pytest and as a plain script::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \\
    PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:<repo>/src \\
    <env>/bin/python test_neutra_nuts.py

The env used to develop these (gigalens_oldapi) has no pytest, so the script
runner below is the primary path. Fixed seeds throughout.

Diagnostics (R-hat, ESS, acceptance rate, divergence counts) are always
PRINTED before the pass/fail assertion is checked, not just asserted --
per the "plots/diagnostics before metrics" rule, a silently-passing number
should never be the only thing on screen.

No arviz: `split_rhat` and `effective_sample_size` below are small local
implementations of the standard (Vehtari et al. 2021 / Stan) split-Rhat and
Geyer initial-positive-sequence ESS estimators.
"""

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from gigalens_research.inference.neutra_nuts import neutra_nuts

_DT = jnp.float64


# --------------------------------------------------------------------------
# local diagnostics (no arviz)
# --------------------------------------------------------------------------
def split_rhat(chains):
    """Standard split-Rhat (Vehtari et al. 2021 / Stan), per parameter.

    chains: (n_chains, n_draws, dim) array-like.
    Returns: (dim,) array of R-hat values.
    """
    chains = np.asarray(chains, dtype=np.float64)
    n_chains, n_draws, dim = chains.shape
    n_half = n_draws // 2
    split = np.concatenate(
        [chains[:, :n_half, :], chains[:, n_draws - n_half:, :]], axis=0
    )  # (2*n_chains, n_half, dim)
    m, n = split.shape[0], split.shape[1]
    chain_means = split.mean(axis=1)  # (m, dim)
    chain_vars = split.var(axis=1, ddof=1)  # (m, dim)
    W = chain_vars.mean(axis=0)  # (dim,) within-chain variance
    B = n * chain_means.var(axis=0, ddof=1)  # (dim,) between-chain variance
    var_hat = (n - 1) / n * W + B / n
    return np.sqrt(var_hat / W)


def _autocovariance(x):
    """Biased autocovariance of a 1-D series via FFT (Stan's normalization)."""
    n = len(x)
    x = x - x.mean()
    size = 1
    while size < 2 * n:
        size *= 2
    fx = np.fft.rfft(x, n=size)
    acov = np.fft.irfft(fx * np.conjugate(fx), n=size)[:n].real
    return acov / n


def effective_sample_size(chains):
    """Multi-chain ESS per parameter via Geyer's initial positive sequence
    (Stan's combine-chains approach), following Gelman et al. BDA3 / the
    Stan reference manual's ESS section. chains: (n_chains, n_draws, dim).
    Returns: (dim,) array of ESS values (summed across chains).
    """
    chains = np.asarray(chains, dtype=np.float64)
    n_chains, n_draws, dim = chains.shape
    ess = np.zeros(dim)
    for d in range(dim):
        acov_chains = np.stack(
            [_autocovariance(chains[c, :, d]) for c in range(n_chains)]
        )  # (n_chains, n_draws)
        var_within = acov_chains[:, 0].mean()
        chain_means = chains[:, :, d].mean(axis=1)
        var_between = (
            n_draws * chain_means.var(ddof=1) if n_chains > 1 else 0.0
        )
        var_plus = ((n_draws - 1) * var_within + var_between) / n_draws
        acov_mean = acov_chains.mean(axis=0)
        rho_hat = 1.0 - (var_within - acov_mean) / var_plus
        rho_hat[0] = 1.0

        tau = 1.0
        t = 1
        while t + 1 < n_draws:
            pair_sum = rho_hat[t] + rho_hat[t + 1]
            if pair_sum < 0:
                break
            tau += 2 * pair_sum
            t += 2
        ess[d] = n_chains * n_draws / tau
    return ess


# --------------------------------------------------------------------------
# target log-densities
# --------------------------------------------------------------------------
DIM_GAUSS = 4
MU_TRUE = jnp.array([1.5, -2.0, 0.5, 3.0], dtype=_DT)
SIGMA_TRUE = jnp.array([1.0, 2.0, 0.5, 1.5], dtype=_DT)
RHO_TRUE = 0.8


def _make_correlated_gaussian_log_prob():
    dim = DIM_GAUSS
    R = RHO_TRUE * jnp.ones((dim, dim), dtype=_DT) + (1 - RHO_TRUE) * jnp.eye(
        dim, dtype=_DT
    )
    D = jnp.diag(SIGMA_TRUE)
    Sigma = D @ R @ D
    Sigma_inv = jnp.linalg.inv(Sigma)
    sign, logdet = jnp.linalg.slogdet(Sigma)

    def log_prob(x):
        d = x - MU_TRUE
        quad = d @ Sigma_inv @ d
        return -0.5 * quad - 0.5 * logdet - 0.5 * dim * jnp.log(2 * jnp.pi)

    return log_prob


def _standard_normal_log_prob(x):
    return -0.5 * jnp.sum(x**2)


def _funnel_log_prob(theta):
    """Neal's funnel, dim = len(theta): v ~ N(0, 3^2), x_i | v ~ N(0, exp(v))."""
    v = theta[0]
    x = theta[1:]
    log_p_v = -0.5 * (v / 3.0) ** 2 - jnp.log(3.0) - 0.5 * jnp.log(2 * jnp.pi)
    log_p_x = jnp.sum(
        -0.5 * x**2 * jnp.exp(-v) - 0.5 * v - 0.5 * jnp.log(2 * jnp.pi)
    )
    return log_p_v + log_p_x


# --------------------------------------------------------------------------
# 1. correlated Gaussian recovery
# --------------------------------------------------------------------------
def test_correlated_gaussian_recovery():
    log_prob = _make_correlated_gaussian_log_prob()
    result = neutra_nuts(
        log_prob,
        dim=DIM_GAUSS,
        n_chains=8,
        num_warmup=1000,
        num_results=1000,
        seed=0,
        target_accept=0.8,
    )
    samples = np.asarray(result["samples"], dtype=np.float64)  # (8, 1000, 4)
    assert samples.shape == (8, 1000, DIM_GAUSS)
    assert np.all(np.isfinite(samples))

    rhat = split_rhat(samples)
    ess = effective_sample_size(samples)
    flat = samples.reshape(-1, DIM_GAUSS)
    emp_mean = flat.mean(axis=0)
    emp_std = flat.std(axis=0, ddof=1)
    emp_corr = np.corrcoef(flat.T)
    mu_true = np.asarray(MU_TRUE)
    sigma_true = np.asarray(SIGMA_TRUE)
    mc_se = sigma_true / np.sqrt(ess)

    print("    -- correlated Gaussian diagnostics --")
    print(f"    split R-hat per dim: {rhat}")
    print(f"    ESS per dim:         {ess}")
    print(f"    empirical mean:      {emp_mean}")
    print(f"    true mean:           {mu_true}")
    print(f"    |mean err| (units of MC SE): {np.abs(emp_mean - mu_true) / mc_se}")
    print(f"    empirical std:       {emp_std}")
    print(f"    true std:            {sigma_true}")
    print(f"    relative std error:  {np.abs(emp_std - sigma_true) / sigma_true}")
    off_diag = emp_corr[~np.eye(DIM_GAUSS, dtype=bool)]
    print(f"    empirical off-diag correlations: {off_diag}")
    print(f"    acceptance rate per chain: {np.asarray(result['acceptance_rate'])}")
    print(f"    num_divergences per chain: {np.asarray(result['num_divergences'])}")

    assert np.max(rhat) < 1.02, f"split R-hat too high: {rhat}"
    assert np.all(
        np.abs(emp_mean - mu_true) <= 4 * mc_se
    ), f"mean recovery outside 4 MC SE: err={emp_mean - mu_true}, mc_se={mc_se}"
    assert np.all(
        np.abs(emp_std - sigma_true) / sigma_true <= 0.10
    ), f"marginal std off by more than 10%: {emp_std} vs {sigma_true}"
    assert np.all(
        np.abs(off_diag - RHO_TRUE) <= 0.10
    ), f"cross-correlation off by more than 0.1: {off_diag}"


# --------------------------------------------------------------------------
# 2. divergence accounting on a pathological target (Neal's funnel)
# --------------------------------------------------------------------------
def test_divergence_accounting_funnel():
    dim = 4
    result = neutra_nuts(
        _funnel_log_prob,
        dim=dim,
        n_chains=4,
        num_warmup=200,
        num_results=200,
        seed=1,
        target_accept=0.8,
    )
    samples = np.asarray(result["samples"], dtype=np.float64)
    num_div = np.asarray(result["num_divergences"])

    print("    -- funnel divergence diagnostics --")
    print(f"    samples shape: {samples.shape}")
    print(f"    num_divergences per chain: {num_div}")
    print(f"    total divergences: {num_div.sum()} / {4 * 200} draws")

    assert "num_divergences" in result
    assert "is_divergent" in result
    assert samples.shape == (4, 200, dim)
    assert np.all(np.isfinite(samples)), "sampler produced non-finite draws"
    assert np.issubdtype(num_div.dtype, np.integer), f"dtype was {num_div.dtype}"
    assert np.all(np.isfinite(num_div))
    assert np.all(num_div >= 0)
    # A finite, well-defined, non-negative count is all we require; on this
    # pathological target > 0 is plausible but not asserted (see docstring).


# --------------------------------------------------------------------------
# 3. determinism
# --------------------------------------------------------------------------
def test_determinism():
    kwargs = dict(dim=2, n_chains=2, num_warmup=50, num_results=50, target_accept=0.8)
    result_a = neutra_nuts(_standard_normal_log_prob, seed=7, **kwargs)
    result_b = neutra_nuts(_standard_normal_log_prob, seed=7, **kwargs)
    result_c = neutra_nuts(_standard_normal_log_prob, seed=8, **kwargs)

    samples_a = np.asarray(result_a["samples"])
    samples_b = np.asarray(result_b["samples"])
    samples_c = np.asarray(result_c["samples"])

    print(f"    same-seed samples identical: {np.array_equal(samples_a, samples_b)}")
    print(
        f"    diff-seed samples identical: {np.array_equal(samples_a, samples_c)}"
    )

    assert np.array_equal(samples_a, samples_b), "same seed did not reproduce samples"
    assert np.array_equal(
        np.asarray(result_a["step_size"]), np.asarray(result_b["step_size"])
    )
    assert not np.array_equal(
        samples_a, samples_c
    ), "different seeds produced identical samples"


# --------------------------------------------------------------------------
# 4. acceptance rate near target on an easy Gaussian
# --------------------------------------------------------------------------
def test_acceptance_rate_gaussian():
    result = neutra_nuts(
        _standard_normal_log_prob,
        dim=4,
        n_chains=8,
        num_warmup=500,
        num_results=500,
        seed=3,
        target_accept=0.8,
    )
    acc = np.asarray(result["acceptance_rate"])
    mean_acc = float(np.mean(acc))
    print("    -- acceptance-rate diagnostics --")
    print(f"    per-chain acceptance rate: {acc}")
    print(f"    mean acceptance rate:      {mean_acc}")
    assert 0.6 <= mean_acc <= 0.95, f"mean acceptance rate {mean_acc} outside [0.6, 0.95]"


# --------------------------------------------------------------------------
# script runner
# --------------------------------------------------------------------------
def _all_tests():
    return [
        (n, g)
        for n, g in sorted(globals().items())
        if n.startswith("test_") and callable(g)
    ]


if __name__ == "__main__":
    print(f"jax float64 enabled: {jnp.zeros(1).dtype == _DT}")
    n_fail = 0
    for name, fn in _all_tests():
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception as e:  # noqa
            n_fail += 1
            import traceback

            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
    print(
        f"\n{'ALL GREEN' if n_fail == 0 else str(n_fail) + ' FAILED'} "
        f"({len(_all_tests())} tests)"
    )
    raise SystemExit(1 if n_fail else 0)
