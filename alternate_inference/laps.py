"""LAPS: Late Adjusted Parallel Sampler

Based on Robnik & Seljak (2026), "Faster parallel MCMC: Metropolis adjustment
is best served warm", arXiv:2601.16696.

Two-phase ensemble sampler:
  Phase 1 (unadjusted): MCLMC with ensemble-based step size/L adaptation
      via the equipartition bias proxy, plus mass matrix adaptation.
  Phase 2 (adjusted): MAMS (Metropolis-adjusted MCLMC) with step size
      bisection targeting an acceptance rate, frozen mass matrix.
"""

import functools
import time
from collections import namedtuple

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

# JAX >= 0.8: jax.shard_map; older releases: jax.experimental.shard_map.shard_map
try:
    _shard_map = jax.shard_map  # type: ignore[attr-defined]
except AttributeError:
    from jax.experimental.shard_map import shard_map as _shard_map

from blackjax.mcmc.integrators import IntegratorState
from blackjax.adaptation.mass_matrix import welford_algorithm, WelfordAlgorithmState

import gigalens.jax.simulator as sim
from alternate_inference.mclmc_alt import (
    init_multi,
    _build_kernel_shardmap,
    welford_combine,
    isokinetic_mclachlan_smart,
)


# ── shard_map-compatible MCLMC base kernel (reused from mclmc_alt) ───────────
# _build_kernel_shardmap is imported above.


# ── Unadjusted scan body ─────────────────────────────────────────────────────

UnadjHist = namedtuple("UnadjHist", [
    "position", "step_size", "L", "D_tilde", "max_delta",
    "inverse_mass_matrix",
])

AdjHist = namedtuple("AdjHist", [
    "position", "step_size", "L", "acceptance",
    "inverse_mass_matrix",
])


def full_laps_sharded(
    kernel_builder,
    num_unadjusted_steps,
    num_adjusted_steps,
    num_results,
    state_init,
    init_step_size,
    init_L,
    init_inverse_mass_matrix,
    svi_mean,
    rng_key,
    num_chains,
    C=0.025,
    alpha=1.0,
    kappa=4,
    switch_threshold=0.01,
    adj_target_accept=0.65,
    adj_n_steps=10,
    adapt_mass_matrix=False,
    mass_matrix_num_effective_samples=1000,
    svi_mass_matrix_weight=20.0,
):
    """Two-scan sharded LAPS implementation.

    Parameters
    ----------
    kernel_builder : callable
        ``lambda imm: _build_kernel_shardmap(logdensity_fn, imm, integrator)``
    C : float
        Bias-to-step-size proportionality constant (Eq. in Sec 3.1).
    alpha : float
        L proportionality constant (Sec 3.2).
    kappa : int
        2 * integrator order (4 for MN2 / leapfrog).
    switch_threshold : float
        max_i (1 - V_ii)^2 threshold for switching to adjusted phase.
    adj_target_accept : float
        Target mean acceptance rate during adjusted phase.
    adj_n_steps : int
        Integrator steps per MAMS proposal.
    """
    num_devices = len(jax.devices())
    num_chains = (num_chains // num_devices) * num_devices
    if num_chains == 0:
        raise ValueError(f"num_chains must be >= num_devices ({num_devices})")
    chains_per_device = num_chains // num_devices
    dim = state_init.position.shape[-1]

    decay_rate_mm = (mass_matrix_num_effective_samples - 1.0) / (
        mass_matrix_num_effective_samples + 1.0
    )
    _, _, welford_cov = welford_algorithm(is_diagonal_matrix=False)

    mesh = jax.make_mesh((num_devices,), ("device",))
    key_sharding = NamedSharding(mesh, P(None, "device"))
    state_sharding = NamedSharding(mesh, P("device"))

    # ================================================================
    # Phase 1 — Unadjusted
    # ================================================================

    def unadj_step(carry, xs):
        i, rng_keys = xs
        states, step_size, L, inv_mm, welford_state, switched = carry

        kernel = kernel_builder(inv_mm)

        # Per-chain kernel step (vmap, no axis_name)
        chain_keys = rng_keys
        new_states, _infos = jax.vmap(
            lambda s, k: kernel(rng_key=k, state=s, L=L, step_size=step_size)
        )(states, chain_keys)

        # ── Equipartition bias (cross-chain, cross-device) ──
        positions = new_states.position
        grads = new_states.logdensity_grad
        n_dev = jax.lax.axis_size("device")
        n_total = chains_per_device * n_dev

        local_sum_x = jnp.sum(positions, axis=0)
        local_sum_xg = jnp.sum(positions * grads, axis=0)
        local_sum_g = jnp.sum(grads, axis=0)
        local_sum_x2 = jnp.sum(jnp.square(positions), axis=0)

        global_sum_x = jax.lax.psum(local_sum_x, "device")
        global_sum_xg = jax.lax.psum(local_sum_xg, "device")
        global_sum_g = jax.lax.psum(local_sum_g, "device")
        global_sum_x2 = jax.lax.psum(local_sum_x2, "device")

        x_mean = global_sum_x / n_total
        # V_ii = -E[(x_i - mean(x_i)) * ∂_i log p]
        #      = -(mean(x*g) - mean(x)*mean(g))
        V_diag = -(global_sum_xg / n_total - x_mean * (global_sum_g / n_total))

        D_tilde = jnp.mean(jnp.square(1.0 - V_diag))
        max_delta = jnp.max(jnp.square(1.0 - V_diag))

        diag_var = global_sum_x2 / n_total - jnp.square(x_mean)
        # Normalize by mass matrix diagonal → preconditioned-space scale
        diag_mm = jnp.diag(inv_mm)
        sigma_typical = jnp.sqrt(jnp.clip(
            jnp.mean(diag_var / jnp.maximum(diag_mm, 1e-10)), 1e-10
        ))

        # ── Step size & L adaptation (only while not switched) ──
        new_step_size = jnp.power(jnp.clip(C * D_tilde, 1e-12), 1.0 / kappa)
        new_L = alpha * sigma_typical

        _sel = lambda c, a, b: jnp.where(c, a, b)
        step_size = _sel(switched, step_size, new_step_size)
        L = _sel(switched, L, new_L)

        # ── Mass matrix adaptation (Welford + EMA decay) ──
        # Disabled by default: not part of the paper's LAPS algorithm.
        # Continuous adaptation destabilizes V_ii by changing the kernel every step.
        if adapt_mass_matrix:
            deltas = positions - x_mean[jnp.newaxis, :]
            local_m2 = jnp.einsum("ci,cj->ij", deltas, deltas)
            m2_step = jax.lax.psum(local_m2, "device")

            update = WelfordAlgorithmState(x_mean, m2_step, n_total)
            new_welford = welford_combine(welford_state, update)
            new_welford = new_welford._replace(
                m2=new_welford.m2 * decay_rate_mm,
                sample_size=new_welford.sample_size * decay_rate_mm,
            )
            do_mm = jnp.logical_and(~switched, i >= 10)
            sample_cov = welford_cov(new_welford)[0]
            inv_mm = jnp.where(do_mm, sample_cov, inv_mm)
            welford_state = jax.tree.map(
                lambda a, b: jnp.where(do_mm, a, b), new_welford, welford_state
            )

        # ── Switch condition ──
        switched = jnp.logical_or(switched, max_delta < switch_threshold)

        h = UnadjHist(
            position=new_states.position,
            step_size=jnp.broadcast_to(step_size, (chains_per_device,)),
            L=jnp.broadcast_to(L, (chains_per_device,)),
            D_tilde=jnp.broadcast_to(D_tilde, (chains_per_device,)),
            max_delta=jnp.broadcast_to(max_delta, (chains_per_device,)),
            inverse_mass_matrix=jnp.broadcast_to(
                inv_mm[jnp.newaxis], (chains_per_device, dim, dim)
            ),
        )
        return (new_states, step_size, L, inv_mm, welford_state, switched), h

    # ── Unadjusted scan setup ──

    unadj_key, adj_key = jax.random.split(rng_key)
    unadj_keys = jax.random.split(
        unadj_key, num_unadjusted_steps * num_chains
    ).reshape(num_unadjusted_steps, num_chains)
    unadj_keys = jax.device_put(unadj_keys, key_sharding)
    state_init = jax.device_put(state_init, state_sharding)

    welford_start = WelfordAlgorithmState(
        svi_mean,
        init_inverse_mass_matrix * svi_mass_matrix_weight,
        svi_mass_matrix_weight,
    )

    unadj_carry_specs = (P("device"), P(), P(), P(), P(), P())

    @jax.jit
    @functools.partial(
        _shard_map,
        mesh=mesh,
        in_specs=(
            (None, P(None, "device")),
            P("device"), P(), P(), P(), P(), P(),
        ),
        out_specs=(unadj_carry_specs, P("device")),
    )
    def run_unadjusted(xs, states, ss, L, imm, welf, switched):
        carry, hist = jax.lax.scan(
            unadj_step,
            init=(states, ss, L, imm, welf, switched),
            xs=xs,
        )
        hist = jax.tree.map(lambda x: jnp.moveaxis(x, 0, 1), hist)
        return carry, hist

    unadj_carry, unadj_hist = run_unadjusted(
        (jnp.arange(num_unadjusted_steps, dtype=jnp.int32), unadj_keys),
        state_init,
        jnp.float32(init_step_size),
        jnp.float32(init_L),
        init_inverse_mass_matrix,
        welford_start,
        jnp.bool_(False),
    )
    final_states, final_ss, final_L, final_imm, _, _ = unadj_carry

    # ================================================================
    # Phase 2 — Adjusted (MAMS)
    # ================================================================

    total_adj_iters = num_adjusted_steps + num_results

    def adj_step(carry, xs):
        i, rng_keys = xs
        states, step_size, L, inv_mm, eps_low, eps_high = carry

        kernel = kernel_builder(inv_mm)

        def mams_proposal(state, key):
            refresh_key, step_key, mh_key = jax.random.split(key, 3)

            # Full velocity refreshment (uniform on unit sphere)
            z = jax.random.normal(refresh_key, shape=(dim,))
            u_new = z / jnp.linalg.norm(z)
            state = IntegratorState(
                state.position, u_new, state.logdensity, state.logdensity_grad
            )

            # N deterministic integration steps, accumulating energy error.
            # L_no_refresh disables partial refreshment (matching BlackJAX's
            # adjusted_mclmc default of L_proposal_factor=inf).
            step_keys = jax.random.split(step_key, adj_n_steps)
            L_no_refresh = jnp.float32(1e30)

            def inner(carry, k):
                s, cum_de = carry
                s_new, info = kernel(rng_key=k, state=s, L=L_no_refresh, step_size=step_size)
                return (s_new, cum_de + info.energy_change), None

            # Derive initial cum_de from varying data to preserve VMA
            init_ce = state.logdensity * 0.0
            (proposal, total_de), _ = jax.lax.scan(
                inner, (state, init_ce), step_keys
            )

            # MH accept/reject
            accept_prob = jnp.minimum(1.0, jnp.exp(-total_de))
            accept = jax.random.uniform(mh_key) < accept_prob
            result = jax.tree.map(
                lambda a, b: jnp.where(accept, a, b), proposal, state
            )
            return result, accept

        chain_keys = rng_keys
        new_states, accepts = jax.vmap(mams_proposal)(states, chain_keys)

        # Cross-chain / cross-device mean acceptance
        n_dev = jax.lax.axis_size("device")
        n_total = chains_per_device * n_dev
        local_accept_sum = jnp.sum(accepts.astype(jnp.float32))
        global_accept_sum = jax.lax.psum(local_accept_sum, "device")
        mean_accept = global_accept_sum / n_total

        # Bisection step size adaptation (log-scale)
        do_adapt = i < num_adjusted_steps
        new_eps_low = jnp.where(mean_accept > adj_target_accept, step_size, eps_low)
        new_eps_high = jnp.where(
            mean_accept <= adj_target_accept, step_size, eps_high
        )
        candidate_ss = jnp.exp(
            0.5 * (jnp.log(new_eps_low) + jnp.log(new_eps_high))
        )
        step_size = jnp.where(do_adapt, candidate_ss, step_size)
        eps_low = jnp.where(do_adapt, new_eps_low, eps_low)
        eps_high = jnp.where(do_adapt, new_eps_high, eps_high)

        L_mams = step_size * adj_n_steps
        h = AdjHist(
            position=new_states.position,
            step_size=jnp.broadcast_to(step_size, (chains_per_device,)),
            L=jnp.broadcast_to(L_mams, (chains_per_device,)),
            acceptance=accepts.astype(jnp.float32),
            inverse_mass_matrix=jnp.broadcast_to(
                inv_mm[jnp.newaxis], (chains_per_device, dim, dim)
            ),
        )
        return (new_states, step_size, L, inv_mm, eps_low, eps_high), h

    # ── Adjusted scan setup ──

    adj_keys = jax.random.split(
        adj_key, total_adj_iters * num_chains
    ).reshape(total_adj_iters, num_chains)
    adj_keys = jax.device_put(adj_keys, key_sharding)

    adj_carry_specs = (P("device"), P(), P(), P(), P(), P())

    @jax.jit
    @functools.partial(
        _shard_map,
        mesh=mesh,
        in_specs=(
            (None, P(None, "device")),
            P("device"), P(), P(), P(), P(), P(),
        ),
        out_specs=(adj_carry_specs, P("device")),
    )
    def run_adjusted(xs, states, ss, L, imm, eps_lo, eps_hi):
        carry, hist = jax.lax.scan(
            adj_step,
            init=(states, ss, L, imm, eps_lo, eps_hi),
            xs=xs,
        )
        hist = jax.tree.map(lambda x: jnp.moveaxis(x, 0, 1), hist)
        return carry, hist

    # Bisection initial bounds: [eps/100, eps*100]
    eps_low_init = final_ss / 100.0
    eps_high_init = final_ss * 100.0

    adj_carry, adj_hist = run_adjusted(
        (jnp.arange(total_adj_iters, dtype=jnp.int32), adj_keys),
        final_states,
        final_ss,
        final_L,
        final_imm,
        eps_low_init,
        eps_high_init,
    )

    # Extract sampling results (after adjusted burn-in)
    samples = jax.tree.map(lambda x: x[:, num_adjusted_steps:], adj_hist)
    return (unadj_hist, adj_hist, samples), adj_carry


# ── Diagnostics ───────────────────────────────────────────────────────────────


def plot_laps_diagnostics(unadj_hist, adj_hist, num_adjusted_steps,
                          switch_threshold=0.01, smooth_kernel_size=30):
    import matplotlib.pyplot as plt
    import numpy as np

    n_unadj = unadj_hist.step_size.shape[1]
    n_adj_total = adj_hist.step_size.shape[1]
    phase_boundary = n_unadj
    sampling_start = n_unadj + num_adjusted_steps

    fig, axs = plt.subplots(5, 1, sharex=True)
    fig.set_size_inches(10, 10)
    ax_ss, ax_L, ax_eig, ax_bias, ax_accept = axs

    # ── Step size (both phases) ──
    ss = jnp.concatenate([unadj_hist.step_size, adj_hist.step_size], axis=1)
    ax_ss.plot(ss.T)
    ax_ss.set_title("Step Size")
    ax_ss.set_ylabel("Step Size")
    ax_ss.set_yscale("log")

    # ── Trajectory length (both phases) ──
    L = jnp.concatenate([unadj_hist.L, adj_hist.L], axis=1)
    ax_L.plot(L.T)
    ax_L.set_title("Trajectory Length (L)")
    ax_L.set_ylabel("L")

    # ── Covariance eigenvalues (chain 0, both phases) ──
    imm_all = jnp.concatenate(
        [unadj_hist.inverse_mass_matrix[0], adj_hist.inverse_mass_matrix[0]], axis=0
    )
    eigvals = jax.vmap(lambda x: jnp.linalg.eig(x)[0])(imm_all)
    min_eig = jnp.min(eigvals, axis=1)
    max_eig = jnp.max(eigvals, axis=1)
    mean_eig = jnp.mean(eigvals, axis=1)

    final_eigvals = jnp.real(jnp.linalg.eig(adj_hist.inverse_mass_matrix[0, 0])[0])
    ax_eig.plot(min_eig, label="Min", color="blue")
    ax_eig.axhline(jnp.min(final_eigvals), color="blue", linestyle="--")
    ax_eig.plot(mean_eig, label="Mean", color="black")
    ax_eig.axhline(jnp.mean(final_eigvals), color="black", linestyle="--")
    ax_eig.plot(max_eig, label="Max", color="red")
    ax_eig.axhline(jnp.max(final_eigvals), color="red", linestyle="--")
    ax_eig.legend()
    ax_eig.set_title("Covariance Eigenvalues")
    ax_eig.set_yscale("log")
    ax_eig.set_ylabel("Eigenvalue")

    # ── Equipartition bias (unadjusted phase only) ──
    ax_bias.plot(unadj_hist.D_tilde[0], label=r"$\tilde{D}$", color="blue")
    ax_bias.plot(unadj_hist.max_delta[0], label=r"$\max_i (1-V_{ii})^2$", color="red")
    ax_bias.axhline(switch_threshold, color="black", linestyle="--", label="Switch threshold")
    ax_bias.set_title("Equipartition Bias (Unadjusted Phase)")
    ax_bias.set_yscale("log")
    ax_bias.set_ylabel("Bias")
    ax_bias.legend()

    # ── Acceptance rate (adjusted phase only) ──
    accept = np.array(adj_hist.acceptance)
    mean_accept = np.mean(accept, axis=0)
    kernel = np.ones(smooth_kernel_size) / smooth_kernel_size
    smoothed = np.convolve(mean_accept, kernel, mode="same")
    x_adj = np.arange(n_unadj, n_unadj + n_adj_total)
    ax_accept.plot(x_adj, mean_accept, alpha=0.3, color="blue")
    ax_accept.plot(x_adj, smoothed, alpha=1.0, color="blue")
    ax_accept.set_title("Mean Acceptance Rate (Adjusted Phase)")
    ax_accept.set_ylabel("Acceptance")
    ax_accept.set_xlabel("Step")

    for ax in axs:
        ax.axvline(phase_boundary, color="red", linestyle="--")
        ax.axvline(sampling_start, color="green", linestyle="--")

    plt.tight_layout()
    plt.show()


# ── Top-level API ─────────────────────────────────────────────────────────────


def LAPS_JIT(
    model_seq,
    qz,
    n_hmc=16,
    num_unadjusted_steps=300,
    num_adjusted_steps=200,
    num_results=500,
    seed=0,
    C=0.025,
    alpha=1.0,
    adj_target_accept=0.65,
    adj_n_steps=10,
    adapt_mass_matrix=False,
    mass_matrix_num_effective_samples=1000,
):
    """Late Adjusted Parallel Sampler.

    Parameters
    ----------
    model_seq : ModellingSequence
    qz : MultivariateNormalTriL
        SVI surrogate — used for initial positions, mass matrix, and Welford prior.
    n_hmc : int
        Number of parallel chains (rounded down to multiple of device count).
    num_unadjusted_steps : int
        Max steps for the unadjusted phase.
    num_adjusted_steps : int
        Burn-in iterations in the adjusted (MAMS) phase.
    num_results : int
        Sampling iterations in the adjusted phase (one sample per iteration).
    C, alpha : float
        Paper hyperparameters for step size and L (Sec 3).
    adj_target_accept : float
        Target MH acceptance rate during adjusted burn-in.
    adj_n_steps : int
        Integrator steps per MAMS proposal.
    """
    lens_sim = sim.LensSimulator(model_seq.phys_model, model_seq.sim_config, bs=1)

    def log_prob(z):
        return model_seq.prob_model.log_prob(lens_sim, z)[0]

    integrator = isokinetic_mclachlan_smart
    kernel_builder = lambda imm: _build_kernel_shardmap(
        logdensity_fn=log_prob, inverse_mass_matrix=imm, integrator=integrator
    )

    rng_key = jax.random.key(seed)
    init_key, run_key = jax.random.split(rng_key)

    n_chains = n_hmc
    state_multi = init_multi(qz.sample((n_chains,), seed=init_key), init_key, log_prob)
    dim = state_multi.position.shape[-1]

    init_L = jnp.sqrt(jnp.float32(dim))
    init_step_size = jnp.sqrt(jnp.float32(dim)) * 0.25

    starttime = time.perf_counter()
    (unadj_hist, adj_hist, samples), adj_carry = full_laps_sharded(
        kernel_builder=kernel_builder,
        num_unadjusted_steps=num_unadjusted_steps,
        num_adjusted_steps=num_adjusted_steps,
        num_results=num_results,
        state_init=state_multi,
        init_step_size=init_step_size,
        init_L=init_L,
        init_inverse_mass_matrix=qz.covariance(),
        svi_mean=qz.mean(),
        rng_key=run_key,
        num_chains=n_chains,
        C=C,
        alpha=alpha,
        adj_target_accept=adj_target_accept,
        adj_n_steps=adj_n_steps,
        adapt_mass_matrix=adapt_mass_matrix,
        mass_matrix_num_effective_samples=mass_matrix_num_effective_samples,
        svi_mass_matrix_weight=10.0 * n_chains,
    )
    total_time = time.perf_counter() - starttime
    print(f"LAPS sampling took {total_time:.1f} s")

    return (unadj_hist, adj_hist, samples), adj_carry
