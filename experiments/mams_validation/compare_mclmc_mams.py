"""Rough apples-to-apples MCLMC vs MAMS on a 10D, kappa=1e6 Gaussian.

Matched on gradient evaluations (both use isokinetic McLachlan => 2 grads/step, so
matching integrator steps == matching gradient calls). Reports BOTH sampling
efficiency (ESS per gradient on the stiffest direction) and bias (per-dimension
variance recovery vs known truth), because ESS/grad is bias-blind and unbiasedness
is the whole point of the adjusted sampler.
"""
import os
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gigalens_research.inference.blackjax_updated_utils import (
    _build_kernel_shardmap, _build_adjusted_kernel_shardmap,
    isokinetic_mclachlan_smart, init_multi,
)
from gigalens_research.inference.mclmc import full_mclmc_with_adapt_sharded
from gigalens_research.inference.mams import full_mams_with_adapt_sharded
import blackjax
from blackjax.diagnostics import effective_sample_size

DIM = 10
KAPPA = 1e6
N_CHAINS = 16
SEED = 0

# ---- target: diagonal Gaussian, log-spaced variances spanning kappa ----
variances = jnp.asarray(np.logspace(0.0, np.log10(KAPPA), DIM))   # [1, ..., 1e6]
inv_var = 1.0 / variances

def log_prob(z):
    return -0.5 * jnp.sum(z * z * inv_var)

print(f"target: {DIM}D Gaussian, variances {np.array(variances)}")
print(f"condition number = {float(variances.max()/variances.min()):.3e}")

key = jax.random.key(SEED)
k_init, k_mclmc, k_mams = jax.random.split(key, 3)

# init chains in the typical set (draw from the true Gaussian) to remove burn-in confound
init_pos = jax.random.normal(k_init, (N_CHAINS, DIM)) * jnp.sqrt(variances)[None, :]
state_init = init_multi(init_pos, k_init, log_prob)

svi_mean = jnp.zeros(DIM)
# Deliberately UNINFORMATIVE preconditioner start (identity): force the shared dense
# windowed-Welford adaptation to actually discover the 1e6 spread. Both samplers use
# the identical adaptation code, so this is a controlled, fair stressor.
imm_init = jnp.eye(DIM)
MCLMCAdaptationState = blackjax.adaptation.mclmc_adaptation.MCLMCAdaptationState
params_init = MCLMCAdaptationState(
    L=jnp.sqrt(DIM), step_size=jnp.sqrt(DIM) * 0.25, inverse_mass_matrix=imm_init
)

NUM_BURNIN = 4000
NUM_RESULTS = 4000

common = dict(
    num_burnin_steps=NUM_BURNIN, num_results=NUM_RESULTS,
    state_init=state_init, params_init=params_init, svi_mean=svi_mean,
    frac_tune1=0.2, frac_tune2=0.6, frac_tune3=0.2,
    num_chains=N_CHAINS, svi_mass_matrix_weight=float(N_CHAINS),
    windowed_mass_matrix=True, progress_bar=False,
)

def analyze(name, hist, num_results, params_final=None):
    pos = np.asarray(hist.position)             # (chains, total_steps, dim)
    res = pos[:, -num_results:, :]              # results phase
    if hasattr(hist, "num_integration_steps"):
        nall = np.asarray(hist.num_integration_steps)[0]   # per step (shared across chains)
        ntune, nres = nall[:-num_results], nall[-num_results:]
        print(f"  [{name}] trajectory n: tuning mean={ntune.mean():.2f} max={ntune.max()} "
              f"| results mean={nres.mean():.2f}")
    if params_final is not None:
        print(f"  [{name}] tuned L={float(np.asarray(params_final.L)):.4f} "
              f"step_size={float(np.asarray(hist.step_size)[0,-1]):.4f} "
              f"L/eps={float(np.asarray(params_final.L))/float(np.asarray(hist.step_size)[0,-1]):.3f}")
    # per-integration-steps -> gradients (McLachlan = 2 grad/step). Per chain, results phase.
    n_int = np.asarray(hist.num_integration_steps) if hasattr(hist, "num_integration_steps") \
        else np.ones(pos.shape[:2])
    n_int_res = n_int[:, -num_results:]
    grads_per_chain = 2.0 * float(n_int_res[0].sum())   # shared across chains
    total_grads = grads_per_chain * res.shape[0]
    # ESS per dimension, pooled across chains
    ess = np.asarray(effective_sample_size(jnp.asarray(res)))   # (dim,)
    # recovered second moment vs truth (zero-mean target)
    est_var = res.reshape(-1, DIM).var(axis=0)
    rel_var_err = est_var / np.asarray(variances) - 1.0
    acc = float(np.asarray(hist.acceptance_rate)[:, -num_results:].mean()) \
        if hasattr(hist, "acceptance_rate") else float("nan")
    print(f"\n=== {name} ===")
    print(f"  results integrator steps/chain : {n_int_res[0].sum():.0f}  "
          f"(avg n/transition = {n_int_res.mean():.2f})")
    print(f"  gradient calls (per chain)     : {grads_per_chain:.0f}")
    print(f"  mean acceptance (results)      : {acc:.3f}")
    print(f"  ESS per dim                    : {np.array2string(ess, precision=0)}")
    print(f"  worst-dim ESS                  : {ess.min():.0f}")
    print(f"  ESS/grad (worst dim, total)    : {ess.min()/total_grads:.3e}")
    print(f"  rel. variance error per dim    : {np.array2string(rel_var_err, precision=3)}")
    print(f"  |rel var err| worst dim (1e6)  : {abs(rel_var_err[-1]):.4f}")
    print(f"  max |rel var err| over dims    : {np.max(np.abs(rel_var_err)):.4f}")
    return dict(name=name, res=res, ess=ess, est_var=est_var, rel_var_err=rel_var_err,
                grads_per_chain=grads_per_chain, total_grads=total_grads,
                n_int_res=n_int_res, acc=acc)

print("\nRunning MCLMC ...")
kern_mclmc = lambda imm: _build_kernel_shardmap(
    logdensity_fn=log_prob, integrator=isokinetic_mclachlan_smart, inverse_mass_matrix=imm)
hist_mclmc, pf_mclmc = full_mclmc_with_adapt_sharded(
    kernel=kern_mclmc, rng_key=k_mclmc, desired_energy_var=5e-4, **common)
r_mclmc = analyze("MCLMC", hist_mclmc, NUM_RESULTS, pf_mclmc)

print("\nRunning MAMS ...")
kern_mams = lambda imm: _build_adjusted_kernel_shardmap(
    logdensity_fn=log_prob, integrator=isokinetic_mclachlan_smart, inverse_mass_matrix=imm)
hist_mams, pf_mams = full_mams_with_adapt_sharded(
    kernel=kern_mams, rng_key=k_mams, target_acceptance=0.9, **common)
r_mams = analyze("MAMS", hist_mams, NUM_RESULTS, pf_mams)

# ---- plots (plots before the summary verdict) ----
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
dims = np.arange(DIM)

ax[0].axhline(0, color="k", lw=0.8)
ax[0].plot(dims, r_mclmc["rel_var_err"], "o-", label="MCLMC")
ax[0].plot(dims, r_mams["rel_var_err"], "s-", label="MAMS")
ax[0].set_xlabel("dimension (variance 1 -> 1e6)"); ax[0].set_ylabel("est/true variance - 1")
ax[0].set_title("Bias: variance recovery"); ax[0].legend()

ax[1].plot(dims, r_mclmc["ess"], "o-", label="MCLMC")
ax[1].plot(dims, r_mams["ess"], "s-", label="MAMS")
ax[1].set_xlabel("dimension"); ax[1].set_ylabel("ESS (pooled)")
ax[1].set_title("ESS per dimension"); ax[1].set_yscale("log"); ax[1].legend()

# running estimate of stiffest-direction variance vs gradient count
for r, style in [(r_mclmc, "-"), (r_mams, "--")]:
    res = r["res"]; stiff = res[:, :, -1]              # (chains, results)
    nstep = res.shape[1]
    grads_axis = np.linspace(0, r["grads_per_chain"], nstep)
    running = np.cumsum(stiff**2, axis=1) / (np.arange(nstep) + 1)
    ax[2].plot(grads_axis, running.mean(0), style, label=r["name"])
ax[2].axhline(float(variances[-1]), color="k", lw=0.8, label="truth")
ax[2].set_xlabel("gradient calls (per chain, results)")
ax[2].set_ylabel("running E[z^2], stiffest dim")
ax[2].set_title("Stiffest-direction 2nd moment"); ax[2].legend()

out = os.path.join(os.path.dirname(__file__), "compare_mclmc_mams.png")
fig.tight_layout(); fig.savefig(out, dpi=110)
print(f"\nsaved plot -> {out}")
