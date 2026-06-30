"""Fixed-knob (adaptation DISABLED) MCLMC runner for the src4 col9 double-well
escape test. Controls init positions (z-space), inverse_mass_matrix, L,
step_size, n_steps, n_chains explicitly. Reuses the EXACT kernel primitives the
real run used (_build_kernel_shardmap + isokinetic_mclachlan_smart) so the only
difference vs the cached run is that NO knob adaptation happens.

Single-device vmap-over-chains + scan-over-steps (no shard_map): keeps it simple
and OOM-safe for modest chains x steps. Stores ONLY col9 + energy_change per step
to bound device memory.
"""
import os, sys, numpy as np, jax, jax.numpy as jnp, time
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/minimal_case_recheck_plots/_scripts")
from build_model import build
from gigalens_research.inference.blackjax_updated_utils import (
    _build_kernel_shardmap, isokinetic_mclachlan_smart, init_multi, _single_init)

DIM = 14
DESIRED_EVAR = 5e-4   # same default the real run used; only for xi reporting
COL = 9
THRESH = 4.40

_pm = None
def pm():
    global _pm
    if _pm is None:
        _pm = build()
    return _pm

def log_prob_single(z):
    # z: (14,) -> scalar.  pm.log_prob returns (log_post, red_chi2).
    return pm().log_prob(z)[0]

def make_runner(imm):
    """4-GPU runner: chains sharded across local devices via pmap (each device runs
    an independent vmap-over-its-chains scan; chains are independent so NO cross-device
    collectives). n_chains must be divisible by jax.local_device_count(). Stores only
    col9 + energy_change + logp per step."""
    kernel = _build_kernel_shardmap(
        logdensity_fn=log_prob_single,
        inverse_mass_matrix=imm,
        integrator=isokinetic_mclachlan_smart,
    )
    def per_device(init_pos, init_keys, step_keys, L, step_size):
        # init_pos (cpd,14); init_keys (cpd,); step_keys (n_steps, cpd)
        states = jax.vmap(lambda p, k: _single_init(p, log_prob_single, k))(init_pos, init_keys)
        def step(carry, sk):
            def per_chain(s, k):
                ns, info = kernel(rng_key=k, state=s, L=L, step_size=step_size)
                return ns, (ns.position[COL], info.energy_change, info.logdensity, info.nonans)
            return jax.vmap(per_chain)(carry, sk)
        _, out = jax.lax.scan(step, states, step_keys)
        return out  # each (n_steps, cpd)
    pdev = jax.pmap(per_device, in_axes=(0, 0, 0, None, None))
    def run(init_positions, L, step_size, n_steps, seed):
        nc = init_positions.shape[0]
        nd = jax.local_device_count()
        assert nc % nd == 0, f"n_chains {nc} not divisible by n_devices {nd}"
        cpd = nc // nd
        ip = jnp.asarray(init_positions).reshape(nd, cpd, DIM)
        ikey, rkey = jax.random.split(jax.random.key(seed))
        init_keys = jax.random.split(ikey, nd * cpd).reshape(nd, cpd)
        # step_keys per device: (nd, n_steps, cpd)
        step_keys = jax.random.split(rkey, nd * n_steps * cpd).reshape(nd, n_steps, cpd)
        c9, ec, logp, nonan = pdev(ip, init_keys, step_keys,
                                   jnp.asarray(L), jnp.asarray(step_size))
        # each (nd, n_steps, cpd) -> (n_steps, nd*cpd)
        def merge(a):
            a = np.asarray(a)
            return np.moveaxis(a, 0, 1).reshape(a.shape[1], nd * cpd)
        return merge(c9), merge(ec), merge(logp), merge(nonan)
    return run

def init_ball(center, n, scale, seed):
    rng = np.random.default_rng(seed)
    return jnp.asarray(np.asarray(center)[None, :] + scale * rng.standard_normal((n, DIM)))

def analyze(c9):
    # c9: (T, C) numpy.  first-passage = first step col9 crosses above THRESH.
    T, C = c9.shape
    above = c9 > THRESH
    mfpt = np.full(C, np.nan)
    for ci in range(C):
        idx = np.nonzero(above[:, ci])[0]
        if idx.size:
            mfpt[ci] = idx[0]
    # committed escape: stays above for >=200 consecutive steps after crossing
    committed = np.full(C, np.nan)
    for ci in range(C):
        a = above[:, ci].astype(int)
        # find first index where next 200 are all above
        run = 0
        for t in range(T):
            run = run + 1 if a[t] else 0
            if run >= 200:
                committed[ci] = t - 199
                break
    return mfpt, committed

def escape_fraction_curve(c9, steps_grid):
    mfpt, _ = analyze(c9)
    return np.array([np.mean(mfpt <= s) for s in steps_grid])

def col9_ess(c9):
    # within-run ESS of col9 per chain via the package ESS helper (single chain).
    from gigalens_research.inference.blackjax_updated_utils import _ess_shardmap
    ess = []
    for ci in range(c9.shape[1]):
        try:
            e = float(np.asarray(_ess_shardmap(
                jnp.asarray(c9[:, ci][None, :, None]),
                chain_axis=0, sample_axis=1)).ravel()[0])
        except Exception:
            e = float("nan")
        ess.append(e)
    return np.array(ess)
