"""MCLMC vs MAMS on a 2-mode Gaussian mixture (weights 0.3 / 0.7), 10D.

The two modes are separated along axis 0; the other 9 axes are shared unimodal.
Chains are initialized 50/50 across the two modes, so recovering the true 0.7
weight in mode B REQUIRES barrier crossing -- this tests mode mixing, weight
recovery, and per-mode bias, matched on gradient evaluations.

Note: the dense windowed preconditioner learns the OVERALL mixture covariance
(stretched along axis 0), which shrinks the whitened inter-mode distance and so
makes crossing feasible. Both samplers share that adaptation code.
"""
import os, sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.special as jsp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gigalens_research.inference.blackjax_updated_utils import (
    _build_kernel_shardmap, _build_adjusted_kernel_shardmap,
    isokinetic_mclachlan_smart, init_multi,
)
from gigalens.jax.experimental.mclmc import full_mclmc_with_adapt_sharded
from gigalens.jax.experimental.mams import full_mams_with_adapt_sharded
import blackjax
from blackjax.diagnostics import effective_sample_size

DIM = 10
N_CHAINS = 16
SEED = 0
SEP = 8.0                      # mode separation along axis 0 (in raw sigma) -- fairly separate
W = np.array([0.3, 0.7])       # mode weights (A, B)

mu_A = jnp.zeros(DIM).at[0].set(-SEP / 2)
mu_B = jnp.zeros(DIM).at[0].set(+SEP / 2)
logw = jnp.log(jnp.asarray(W))

def log_prob(z):
    lpA = logw[0] - 0.5 * jnp.sum((z - mu_A) ** 2)
    lpB = logw[1] - 0.5 * jnp.sum((z - mu_B) ** 2)
    return jsp.logsumexp(jnp.stack([lpA, lpB]))

# true overall moments along axis 0
true_mean0 = float(W[0] * (-SEP / 2) + W[1] * (SEP / 2))     # = 0.2*SEP
true_var0 = float(W[0] * (1 + (SEP / 2) ** 2) + W[1] * (1 + (SEP / 2) ** 2) - true_mean0 ** 2)
print(f"target: 10D, 2 modes at axis0 = +/-{SEP/2}, weights {W}, "
      f"true E[z0]={true_mean0:.3f}, true Var[z0]={true_var0:.3f}")

key = jax.random.key(SEED)
k_init, k_mclmc, k_mams = jax.random.split(key, 3)

# init 50/50 across modes (8 in A, 8 in B), small within-mode jitter
half = N_CHAINS // 2
centers = jnp.where((jnp.arange(N_CHAINS) < half)[:, None], mu_A[None, :], mu_B[None, :])
init_pos = centers + jax.random.normal(k_init, (N_CHAINS, DIM))
state_init = init_multi(init_pos, k_init, log_prob)

svi_mean = jnp.zeros(DIM).at[0].set(true_mean0)        # true overall mean
imm_init = jnp.eye(DIM)
MCLMCAdaptationState = blackjax.adaptation.mclmc_adaptation.MCLMCAdaptationState
params_init = MCLMCAdaptationState(L=jnp.sqrt(DIM), step_size=jnp.sqrt(DIM) * 0.25,
                                   inverse_mass_matrix=imm_init)

NUM_BURNIN = 4000
# Match RESULTS-phase gradient evaluations, NOT transitions: MAMS tunes to ~n
# steps/transition (measured ~3.94 here) while MCLMC is always 1 step/transition,
# so MCLMC gets proportionally more transitions to reach the same gradient budget.
TARGET_RESULT_GRADS = 32000          # per chain, results phase
MAMS_AVG_N = 3.94                    # measured in the transition-matched pilot run
NUM_RESULTS_MCLMC = TARGET_RESULT_GRADS // 2                        # n=1, 2 grad/step
NUM_RESULTS_MAMS = int(round(TARGET_RESULT_GRADS / 2 / MAMS_AVG_N))
print(f"results transitions: MCLMC={NUM_RESULTS_MCLMC} MAMS={NUM_RESULTS_MAMS} "
      f"(target ~{TARGET_RESULT_GRADS} grads/chain each)")
common = dict(
    num_burnin_steps=NUM_BURNIN,
    state_init=state_init, params_init=params_init, svi_mean=svi_mean,
    frac_tune1=0.2, frac_tune2=0.6, frac_tune3=0.2,
    num_chains=N_CHAINS, svi_mass_matrix_weight=float(N_CHAINS),
    windowed_mass_matrix=True, progress_bar=False,
)

def analyze(name, hist, num_results):
    pos = np.asarray(hist.position)               # (chains, total, dim)
    res = pos[:, -num_results:, :]
    z0 = res[:, :, 0]
    in_B = z0 > 0.0                               # midpoint between modes is 0
    weight_B = in_B.mean()
    # mode crossings per chain (sign flips of z0 across consecutive steps)
    crossings = int(np.sum(np.abs(np.diff((z0 > 0).astype(int), axis=1))))
    # per-mode fidelity (within-mode mean of axis 0 vs +/-SEP/2)
    mA = z0[~in_B].mean() if (~in_B).any() else np.nan
    mB = z0[in_B].mean() if in_B.any() else np.nan
    mean0 = res.reshape(-1, DIM)[:, 0].mean()
    var0 = res.reshape(-1, DIM)[:, 0].var()
    if hasattr(hist, "num_integration_steps"):
        n_int_res = np.asarray(hist.num_integration_steps)[:, -num_results:]
    else:
        n_int_res = np.ones(res.shape[:2])
    grads_per_chain = 2.0 * float(n_int_res[0].sum())
    ess0 = float(np.asarray(effective_sample_size(jnp.asarray(res[:, :, :1]))).reshape(-1)[0])
    ess1 = float(np.asarray(effective_sample_size(jnp.asarray(res[:, :, 1:2]))).reshape(-1)[0])
    acc = float(np.asarray(hist.acceptance_rate)[:, -num_results:].mean()) \
        if hasattr(hist, "acceptance_rate") else float("nan")
    print(f"\n=== {name} ===")
    print(f"  grads/chain={grads_per_chain:.0f}  acc={acc:.3f}  "
          f"avg n/transition={n_int_res.mean():.2f}")
    print(f"  recovered weight(mode B)  : {weight_B:.3f}   (truth 0.700)")
    print(f"  mode crossings (all chains): {crossings}")
    print(f"  within-mode mean z0  A/B  : {mA:.3f} / {mB:.3f}   (truth -{SEP/2:.1f}/+{SEP/2:.1f})")
    print(f"  overall E[z0]/Var[z0]     : {mean0:.3f} / {var0:.3f}   "
          f"(truth {true_mean0:.3f} / {true_var0:.3f})")
    print(f"  ESS axis0 (multimodal)    : {ess0:.0f}")
    print(f"  ESS axis1 (within-mode)   : {ess1:.0f}")
    return dict(name=name, res=res, z0=z0, weight_B=weight_B, grads_per_chain=grads_per_chain)

print("\nRunning MCLMC ...")
kern_mclmc = lambda imm: _build_kernel_shardmap(
    logdensity_fn=log_prob, integrator=isokinetic_mclachlan_smart, inverse_mass_matrix=imm)
hist_mclmc, _ = full_mclmc_with_adapt_sharded(kernel=kern_mclmc, rng_key=k_mclmc,
                                              num_results=NUM_RESULTS_MCLMC,
                                              desired_energy_var=5e-4, **common)
r_mclmc = analyze("MCLMC", hist_mclmc, NUM_RESULTS_MCLMC)

print("\nRunning MAMS ...")
kern_mams = lambda imm: _build_adjusted_kernel_shardmap(
    logdensity_fn=log_prob, integrator=isokinetic_mclachlan_smart, inverse_mass_matrix=imm)
hist_mams, _ = full_mams_with_adapt_sharded(kernel=kern_mams, rng_key=k_mams,
                                            num_results=NUM_RESULTS_MAMS,
                                            target_acceptance=0.9, **common)
r_mams = analyze("MAMS", hist_mams, NUM_RESULTS_MAMS)

# ---- plots ----
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
# true axis-0 marginal density
xs = np.linspace(-SEP / 2 - 4, SEP / 2 + 4, 400)
true_pdf = (W[0] * np.exp(-0.5 * (xs + SEP / 2) ** 2) + W[1] * np.exp(-0.5 * (xs - SEP / 2) ** 2)) \
    / np.sqrt(2 * np.pi)
for r, c in [(r_mclmc, "C0"), (r_mams, "C1")]:
    ax[0].hist(r["z0"].reshape(-1), bins=80, density=True, histtype="step", color=c, label=r["name"])
ax[0].plot(xs, true_pdf, "k-", lw=1.2, label="truth")
ax[0].set_xlabel("z[0] (multimodal axis)"); ax[0].set_ylabel("density")
ax[0].set_title(f"Axis-0 marginal (modes +/-{SEP/2:.0f}, w=0.3/0.7)"); ax[0].legend()

for r, style in [(r_mclmc, "-"), (r_mams, "--")]:
    z0 = r["z0"]; nstep = z0.shape[1]
    grads_axis = np.linspace(0, r["grads_per_chain"], nstep)
    running_wB = np.cumsum((z0 > 0).astype(float), axis=1).mean(0) / (np.arange(nstep) + 1)
    ax[1].plot(grads_axis, running_wB, style, label=r["name"])
ax[1].axhline(0.7, color="k", lw=0.8, label="truth 0.7"); ax[1].axhline(0.5, color="0.6", lw=0.6, ls=":")
ax[1].set_xlabel("gradient calls (per chain, results)"); ax[1].set_ylabel("running weight(mode B)")
ax[1].set_title("Weight recovery (init 50/50)"); ax[1].set_ylim(0.3, 0.9); ax[1].legend()

# per-chain z0 traces (mixing visual), first few chains
for r, c in [(r_mclmc, "C0"), (r_mams, "C1")]:
    z0 = r["z0"]
    for ci in range(min(4, z0.shape[0])):
        ax[2].plot(z0[ci, ::20], c, lw=0.4, alpha=0.5)
ax[2].axhline(SEP / 2, color="0.5", lw=0.5); ax[2].axhline(-SEP / 2, color="0.5", lw=0.5)
ax[2].set_xlabel("results step / 20"); ax[2].set_ylabel("z[0]")
ax[2].set_title("z0 traces (blue=MCLMC, orange=MAMS)")

out = os.path.join(os.path.dirname(__file__), "compare_bimodal.png")
fig.tight_layout(); fig.savefig(out, dpi=110)
print(f"\nsaved plot -> {out}")
