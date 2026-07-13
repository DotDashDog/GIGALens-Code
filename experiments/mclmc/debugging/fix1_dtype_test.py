"""Fast x64 dtype-consistency test for full_mclmc_with_adapt_sharded.

Reproduces the notebook setup that breaks under jax_enable_x64 + mixed precision, WITHOUT the
expensive lensing model: a toy float64 logdensity, a float32 qz (so positions promote to float64
while mean/cov stay float32 -- exactly the notebook condition). Tiny (dim=5, 2 chains, ~12 steps),
CPU. Pass = traces & runs with no dtype TypeError.

Usage: python fix1_dtype_test.py
"""
import os, sys
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

home = os.path.expanduser("~/")
sys.path.insert(0, os.path.join(home, "GIGALens-Code/src"))

from gigalens_research.inference.blackjax_updated_utils import (
    init_multi, _build_kernel_shardmap, isokinetic_mclachlan_smart,
)
from gigalens.jax.experimental.mclmc import full_mclmc_with_adapt_sharded
from blackjax.adaptation.mclmc_adaptation import MCLMCAdaptationState

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions


def logdensity(z):
    # mimic high_precision: reduction returns float64 regardless of z dtype
    return -0.5 * jnp.sum(z.astype(jnp.float64) ** 2)


def run_case(name, broken_momentum=False):
    dim, n_chains = 5, 2
    key = jax.random.key(0)
    ik, tk = jax.random.split(key)

    # float32 qz -> sample promotes to float64 under x64, mean()/covariance() stay float32
    # (exactly the notebook's bootstrap-qz condition).
    qz = tfd.MultivariateNormalTriL(
        loc=jnp.zeros(dim, dtype=jnp.float32),
        scale_tril=jnp.eye(dim, dtype=jnp.float32),
    )
    positions = qz.sample((n_chains,), seed=ik)
    print(f"[{name}] positions dtype={positions.dtype}  qz.mean dtype={qz.mean().dtype} "
          f"qz.cov dtype={qz.covariance().dtype}")

    state = init_multi(positions, ik, logdensity)
    if broken_momentum:
        # force the pre-fix bug: float32 initial momentum vs float64 positions
        state = state._replace(momentum=state.momentum.astype(jnp.float32))
    print(f"[{name}] state momentum dtype={state.momentum.dtype} position dtype={state.position.dtype}")

    kernel = lambda imm: _build_kernel_shardmap(
        logdensity_fn=logdensity, integrator=isokinetic_mclachlan_smart, inverse_mass_matrix=imm)

    params_init = MCLMCAdaptationState(
        L=jnp.sqrt(jnp.array(dim, dtype=jnp.float32)),
        step_size=jnp.sqrt(jnp.array(dim, dtype=jnp.float32)) * 0.25,
        inverse_mass_matrix=qz.covariance(),   # float32
    )

    hist, params = full_mclmc_with_adapt_sharded(
        kernel=kernel, num_burnin_steps=10, num_results=2,
        state_init=state, params_init=params_init,
        svi_mean=qz.mean(),                    # float32
        rng_key=tk, frac_tune1=0.2, frac_tune2=0.6, frac_tune3=0.2,
        desired_energy_var=5e-4, num_chains=n_chains, num_effective_samples=100,
        svi_mass_matrix_weight=10.0 * n_chains,
        step_size_adapt_use_psmile=False, windowed_mass_matrix=True, progress_bar=False,
    )
    print(f"[{name}] OK -> step_size[:, -1]={np.asarray(hist.step_size)[:, -1]} "
          f"final eps dtype={hist.step_size.dtype}")


if __name__ == "__main__":
    # 1) sanity that the test CAN catch the bug: force float32 momentum -> expect failure
    if "--check-detect" in sys.argv:
        try:
            run_case("BROKEN", broken_momentum=True)
            print("[BROKEN] UNEXPECTEDLY PASSED -- test does not detect the bug!")
        except TypeError as e:
            print(f"[BROKEN] correctly raised TypeError: {str(e)[:120]}")
    # 2) the real path with the fixes in place -> must pass
    run_case("FIXED", broken_momentum=False)
    print("ALL GOOD")
