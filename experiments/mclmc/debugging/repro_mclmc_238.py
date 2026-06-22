"""Minimal repro of MCLMC_JIT crash for DESI-238.
Mirrors cells 0-6 + tiny variant of cell 12 from MCLMC_238.ipynb.

Run with environment-controlled compile probes:
  REPRO_TAG=<tag> python repro_mclmc_238.py
"""
import os, sys, time, traceback, faulthandler, signal
faulthandler.enable()
faulthandler.register(signal.SIGUSR1, all_threads=True)

TAG = os.environ.get("REPRO_TAG", "default")
print(f"[repro:{TAG}] start, pid={os.getpid()}", flush=True)

# Mirror cell 0 path setup, but be careful not to disturb a sidecar overlay
# (e.g., a tfp-nightly directory) that the user may have placed at sys.path[1].
home = os.path.expanduser("~/")
_conda_env_marker = "/.conda/envs/"
_conda_idx = next((i for i, p in enumerate(sys.path) if _conda_env_marker in p), None)
if _conda_idx is not None:
    conda_env = sys.path.pop(_conda_idx)
    sys.path.append(home + "/gigalens/src")
    sys.path.append(conda_env)
else:
    sys.path.append(home + "/gigalens/src")
sys.path.append(home + "/GIGALens-Code/")

# Switch to the data dir so relative paths in cell 3 resolve
os.chdir(home + "/GIGALens-Code/alternate_inference")
print(f"[repro:{TAG}] cwd={os.getcwd()}", flush=True)

t0 = time.perf_counter()
print(f"[repro:{TAG}] importing jax...", flush=True)
import jax
print(f"[repro:{TAG}] jax devices: {jax.devices()}  ({time.perf_counter()-t0:.1f}s)", flush=True)

import numpy as np
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

# Optional source-level workaround for the XLA dot-autotune FATAL on the
# `jnp.linalg.norm` inside blackjax's partially_refresh_momentum.
if os.environ.get("REPRO_PATCH_NORM", "0") == "1":
    import blackjax.mcmc.integrators as _bj_int
    from blackjax.mcmc.integrators import ravel_pytree as _bj_ravel
    from jax.random import normal as _jax_normal
    def _safe_norm(x):
        # Avoid lowering through dot(x, x); use explicit sum-of-squares + sqrt
        return jnp.sqrt(jnp.sum(x * x))
    def _patched_normalized_flatten_array(x, tol=1e-13):
        n = _safe_norm(x)
        return jnp.where(n > tol, x / n, x), n
    def _patched_partially_refresh_momentum(momentum, rng_key, step_size, L):
        m, unravel_fn = _bj_ravel(momentum)
        dim = m.shape[0]
        nu = jnp.sqrt((jnp.exp(2 * step_size / L) - 1.0) / dim)
        z = nu * _jax_normal(rng_key, shape=m.shape, dtype=m.dtype)
        v = m + z
        new_momentum = unravel_fn(v / _safe_norm(v))
        return jax.lax.cond(jnp.isinf(L), lambda _: momentum,
                            lambda _: new_momentum, operand=None)
    _bj_int._normalized_flatten_array = _patched_normalized_flatten_array
    _bj_int.partially_refresh_momentum = _patched_partially_refresh_momentum
    print(f"[repro:{TAG}] PATCHED _normalized_flatten_array & partially_refresh_momentum", flush=True)

from gigalens.jax.inference import ModellingSequence
from gigalens.jax.prob_model import BackwardProbModel
from gigalens.jax.physical_model import PhysicalModel
from gigalens.simulator import SimulatorConfig
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear

print(f"[repro:{TAG}] imports done ({time.perf_counter()-t0:.1f}s)", flush=True)

# Cell 2: priors (verbatim)
lens_prior = tfd.JointDistributionSequential(
    [
        tfd.JointDistributionNamed(dict(
            theta_E=tfd.LogNormal(jnp.log(2.0), 0.5),
            gamma=tfd.TruncatedNormal(2, 0.5, 1, 3),
            e1=tfd.TruncatedNormal(0, 0.25, -0.3, 0.3),
            e2=tfd.TruncatedNormal(0, 0.25, -0.3, 0.3),
            center_x=tfd.Normal(0, 0.1),
            center_y=tfd.Normal(0, 0.1),
        )),
        tfd.JointDistributionNamed(dict(gamma1=tfd.Normal(0, 0.1), gamma2=tfd.Normal(0, 0.1))),
    ]
)
def _ll(rs_loc, rs_w, ns_lo, ns_hi, e1w, e2w, cx_loc, cx_w, cy_loc, cy_w):
    return tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(rs_loc), rs_w),
        n_sersic=tfd.Uniform(ns_lo, ns_hi),
        e1=tfd.TruncatedNormal(0, e1w, -0.5, 0.5) if e1w == 0.1 else tfd.TruncatedNormal(0, e1w, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0, e2w, -0.5, 0.5) if e2w == 0.1 else tfd.TruncatedNormal(0, e2w, -0.3, 0.3),
        center_x=tfd.Normal(cx_loc, cx_w),
        center_y=tfd.Normal(cy_loc, cy_w),
    ))

lens_light_prior = tfd.JointDistributionSequential([
    tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(0.05), 0.1),
        n_sersic=tfd.Uniform(1, 15),
        e1=tfd.TruncatedNormal(0, 0.1, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0, 0.1, -0.5, 0.5),
        center_x=tfd.Normal(-3.25, 0.05),
        center_y=tfd.Normal(-3.25, 0.05),
    )),
    tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(0.25), 0.1),
        n_sersic=tfd.Uniform(1, 10),
        e1=tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
        center_x=tfd.Normal(0, 0.05),
        center_y=tfd.Normal(0, 0.05),
    )),
    tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(0.25), 0.1),
        n_sersic=tfd.Uniform(1, 10),
        e1=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
        center_x=tfd.Normal(0, 0.05),
        center_y=tfd.Normal(0, 0.05),
    )),
])
source_light_prior = tfd.JointDistributionSequential([
    tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(0.2), 0.5),
        n_sersic=tfd.Uniform(0.01, 15),
        e1=tfd.TruncatedNormal(0, 0.25, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0, 0.25, -0.5, 0.5),
        center_x=tfd.Normal(0, 0.5),
        center_y=tfd.Normal(0, 0.5),
    )),
    tfd.JointDistributionNamed(dict(
        R_sersic=tfd.LogNormal(jnp.log(0.2), 0.5),
        n_sersic=tfd.Uniform(0.01, 15),
        e1=tfd.TruncatedNormal(0, 0.25, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0, 0.25, -0.5, 0.5),
        center_x=tfd.Normal(0, 0.5),
        center_y=tfd.Normal(0, 0.5),
    )),
])
prior = tfd.JointDistributionSequential([lens_prior, lens_light_prior, source_light_prior])

# Cell 3
kernel = np.load("desi238data/psf94.npy").astype(np.float32)
sim_config = SimulatorConfig(delta_pix=0.065, num_pix=120, supersample=1, kernel=kernel)
phys_model = PhysicalModel(
    [epl.EPL(50), shear.Shear()],
    [sersic.SersicEllipse(use_lstsq=True)] * 3,
    [sersic.SersicEllipse(use_lstsq=True)] * 2,
)
observed_img = np.load("desi238data/cutout238b.npy").astype(np.float32, copy=False)
prob_model = BackwardProbModel(prior, jnp.array(observed_img), background_rms=0.007616264, exp_time=1197.699462)
model_seq = ModellingSequence(phys_model, prob_model, sim_config)

# Cell 6
map_best = jnp.load("best.npy")
default_start = jnp.diag(jnp.ones((map_best.shape[-1],))) * 1e-3
qz = tfd.MultivariateNormalTriL(loc=jnp.squeeze(map_best), scale_tril=default_start)

print(f"[repro:{TAG}] model setup complete ({time.perf_counter()-t0:.1f}s); dim={map_best.shape[-1]}", flush=True)

# Tiny MCLMC_JIT call
from alternate_inference.mclmc_alt import MCLMC_JIT
import alternate_inference.mclmc_alt as _mcl

if os.environ.get("REPRO_PATCH_DOT", "0") == "1":
    from blackjax.mcmc.integrators import (
        ravel_pytree as _bj_ravel,
        generalized_two_stage_integrator as _bj_g2s,
        format_isokinetic_state_output as _bj_format,
        euclidean_position_update_fn as _bj_pos_update,
    )
    def _safe_norm(x):
        return jnp.sqrt(jnp.sum(x * x))
    def _safe_normalized_flatten_array(x, tol=1e-13):
        n = _safe_norm(x)
        return jnp.where(n > tol, x / n, x), n
    def _patched_esh_smart(inverse_mass_matrix):
        if len(inverse_mass_matrix.shape) != 2:
            raise ValueError("inverse_mass_matrix must have 2 dimensions.")
        chol_inverse_mass_matrix = jnp.linalg.cholesky(inverse_mass_matrix)
        def _matvec_no_dot(M, v):
            # M @ v expressed as elementwise multiply + reduce (no dot_general HLO)
            return jnp.sum(M * v[None, :], axis=-1)
        def update(momentum, logdensity_grad, step_size, coef,
                   previous_kinetic_energy_change=None, is_last_call=False):
            del is_last_call
            flatten_grads, unravel_fn = _bj_ravel(logdensity_grad)
            flatten_grads = _matvec_no_dot(chol_inverse_mass_matrix.T, flatten_grads)
            flatten_momentum, _ = _bj_ravel(momentum)
            dims = flatten_momentum.shape[0]
            normalized_gradient, gradient_norm = _safe_normalized_flatten_array(flatten_grads)
            momentum_proj = jnp.sum(flatten_momentum * normalized_gradient)
            delta = step_size * coef * gradient_norm / (dims - 1)
            zeta = jnp.exp(-delta)
            new_momentum_raw = (
                normalized_gradient * (1 - zeta) * (1 + zeta + momentum_proj * (1 - zeta))
                + 2 * zeta * flatten_momentum
            )
            new_momentum_normalized, _ = _safe_normalized_flatten_array(new_momentum_raw)
            gr = unravel_fn(_matvec_no_dot(chol_inverse_mass_matrix, new_momentum_normalized))
            next_momentum = unravel_fn(new_momentum_normalized)
            kinetic_energy_change = (
                delta - jnp.log(2)
                + jnp.log(1 + momentum_proj + (1 - momentum_proj) * zeta**2)
            ) * (dims - 1)
            if previous_kinetic_energy_change is not None:
                kinetic_energy_change += previous_kinetic_energy_change
            return next_momentum, gr, kinetic_energy_change
        return update
    _mcl.esh_dynamics_momentum_update_one_step_smart = _patched_esh_smart
    print(f"[repro:{TAG}] PATCHED esh_dynamics_momentum_update_one_step_smart "
          "(jnp.dot -> jnp.sum(a*b); jnp.linalg.norm -> sqrt(sum(x*x)))", flush=True)

n_hmc = int(os.environ.get("REPRO_NHMC", 4))
nb = int(os.environ.get("REPRO_NB", 10))
nr = int(os.environ.get("REPRO_NR", 5))
use_sm = os.environ.get("REPRO_SHARD_MAP", "0") == "1"
print(f"[repro:{TAG}] calling MCLMC_JIT n_hmc={n_hmc} nb={nb} nr={nr} use_shard_map={use_sm}", flush=True)
sys.stdout.flush(); sys.stderr.flush()

t1 = time.perf_counter()
try:
    debug_hist = MCLMC_JIT(
        model_seq, qz,
        n_hmc=n_hmc, num_burnin_steps=nb, num_results=nr,
        desired_energy_variance=5e-4, frac_tune1=0.2, frac_tune2=0.6, frac_tune3=0.2,
        seed=0, debug_output=True, step_size_adapt_use_psmile=False,
        use_shard_map=use_sm, progress_bar=False,
    )
    debug_hist.position.block_until_ready()
    print(f"[repro:{TAG}] DONE in {time.perf_counter()-t1:.1f}s; samples shape={debug_hist.position.shape}", flush=True)
    # Exercise the indexing patterns the user runs in the notebook so we
    # catch JAX 0.10 ShardingTypeError on gather of sharded outputs.
    print(f"[repro:{TAG}] step_size[0,-1]={float(debug_hist.step_size[0, -1]):.6g}", flush=True)
    print(f"[repro:{TAG}] position[0,-1,0]={float(debug_hist.position[0, -1, 0]):.6g}", flush=True)
    print(f"[repro:{TAG}] L[-1,-1]={float(debug_hist.L[-1, -1]):.6g}", flush=True)
except Exception as e:
    print(f"[repro:{TAG}] EXCEPTION after {time.perf_counter()-t1:.1f}s: {type(e).__name__}: {e}", flush=True)
    traceback.print_exc()
    sys.exit(2)
