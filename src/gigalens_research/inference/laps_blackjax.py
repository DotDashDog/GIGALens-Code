"""Bare-bones GIGALens wrapper around blackjax's stock LAPS.

This is intentionally a thin translation layer: it takes the usual
GIGALens ``model_seq`` + surrogate ``qz`` inputs, builds the quantities
that ``blackjax.adaptation.laps.laps`` expects (log-density, per-chain
initial sampler, mesh) and forwards everything else unchanged. The
default algorithm is the blackjax default -- no GIGALens-side tweaks --
so this wrapper is suitable for A/B comparisons against the in-tree
re-implementation in ``laps.py``.

See https://github.com/blackjax-devs/blackjax/blob/1.5/blackjax/adaptation/laps.py
for the upstream reference.
"""

from __future__ import annotations

import contextlib
import time

import jax
import jax.numpy as jnp
import blackjax.util as _blackjax_util
from blackjax.adaptation.laps import laps as _blackjax_laps

# NOTE: experimental; NOT migrated to the scene API (it built a legacy LensSimulator from
# model_seq.phys_model, removed with the old gigalens API). Module stays IMPORT-SAFE;
# LAPS_blackjax raises if CALLED. Restore from git b82397c to re-enable.
# The first parameter is named ``prob_model`` to match what live callers now pass (a scene
# ProbModel), but the UNREACHABLE body below is still written against the old
# ``model_seq`` shape — it must be rewritten, not just renamed, when this is restored.


@contextlib.contextmanager
def _shard_map_check_vma_false():
    """Temporarily disable the VMA type-check on blackjax's ``shard_map`` calls.

    Recent JAX dev builds tighten the "varying manual axes" (VMA) type-check
    inside ``shard_map`` (``check_vma=True`` by default). blackjax's
    ``mclmc.handle_high_energy`` / ``handle_nans`` build one branch of a
    ``jax.lax.cond`` via ``jnp.zeros_like(...)``, whose aval drops the
    ``V:chains`` tag carried by the other branch. The algorithm is
    semantically unchanged, but the new check rejects it. JAX's own cond
    error message explicitly points at ``check_vma=False`` as the documented
    workaround. We swap in a thin wrapper around ``jax.shard_map`` that
    forwards everything unchanged except for forcing ``check_vma=False`` for
    the duration of the LAPS call, and restore the original symbol on exit,
    so no blackjax source file is modified.

    The older experimental ``shard_map`` spelt this kwarg ``check_rep``;
    blackjax imports ``jax.shard_map`` (new API), so ``check_vma`` is the
    right name here.
    """
    original = _blackjax_util.shard_map

    def _shard_map_no_check(*args, **kwargs):
        kwargs["check_vma"] = False
        return jax.shard_map(*args, **kwargs)

    _blackjax_util.shard_map = _shard_map_no_check
    try:
        yield
    finally:
        _blackjax_util.shard_map = original


def LAPS_blackjax(
    prob_model,
    qz,
    n_hmc: int = 128,
    num_unadjusted_steps: int = 1000,
    num_adjusted_steps: int = 2000,
    seed: int = 0,
    microcanonical: bool = True,
    alpha: float = 1.9,
    save_frac: float = 0.2,
    C: float = 0.1,
    early_stop: bool = True,
    r_end: float = 0.01,
    bias_type: int = 3,
    diagonal_preconditioning: bool = True,
    integrator_coefficients=None,
    steps_per_sample: int = 15,
    acc_prob=None,
    superchain_size: int = 1,
    diagnostics: bool = True,
    all_chains_info=None,
    observables_for_bias=lambda x: x,
    contract=lambda x: 0.0,
):
    """Run blackjax's stock LAPS on a GIGALens model.

    Parameters
    ----------
    model_seq
        A GIGALens ``ModellingSequence`` providing ``phys_model``,
        ``sim_config`` and ``prob_model``.
    qz
        Surrogate posterior (typically the VI fit) in the unconstrained
        space. Used both for per-chain initialization via ``qz.sample``
        and to set the dimensionality.
    n_hmc
        Total number of parallel chains (rounded down to a multiple of
        the available device count).
    num_unadjusted_steps, num_adjusted_steps
        Mapped directly to blackjax's ``num_steps1`` (unadjusted burn-in)
        and ``num_steps2`` (adjusted refinement). Blackjax decides
        internally how many adjusted samples to keep based on
        ``steps_per_sample`` and the chosen integrator.
    seed
        Integer seed used for ``jax.random.key``.

    All remaining keyword arguments are forwarded verbatim to
    ``blackjax.adaptation.laps.laps`` and default to blackjax's own
    defaults so the algorithm is unmodified.

    Returns
    -------
    dict with keys ``final_state`` (blackjax ``HMCState``), ``info``
    (per-phase diagnostics if ``diagnostics=True``, else ``None``),
    ``gradient_calls_per_step`` and ``acc_prob`` (the target acceptance
    actually used by blackjax after its dimension-dependent defaults).
    """

    raise NotImplementedError(
        "LAPS_blackjax was not migrated to the scene API; it used the legacy "
        "LensSimulator, which was removed with the old gigalens API. Restore from git "
        "b82397c if needed.")
    lens_sim = sim.LensSimulator(
        model_seq.phys_model,
        model_seq.sim_config,
        bs=1,
    )

    def log_prob(z):
        return model_seq.prob_model.log_prob(lens_sim, z)[0]

    dim = int(jnp.size(qz.mean()))

    def sample_init(key):
        return qz.sample(seed=key)
    # def sample_init(key):
    #     start = model_seq.prob_model.prior.sample(seed=key)
    #     params = jnp.stack(model_seq.prob_model.bij.inverse(start)).T
    #     return params

    num_devices = len(jax.devices())
    n_hmc = (n_hmc // num_devices) * num_devices
    if n_hmc == 0:
        raise ValueError(
            f"n_hmc must be >= num_devices ({num_devices}); got 0 after rounding."
        )
    mesh = jax.make_mesh((num_devices,), ("chains",))

    rng_key = jax.random.key(seed)

    start = time.perf_counter()
    with _shard_map_check_vma_false():
        info, gradient_calls_per_step, used_acc_prob, final_state = _blackjax_laps(
            log_prob,
            sample_init,
            dim,
            num_unadjusted_steps,
            num_adjusted_steps,
            n_hmc,
            mesh,
            rng_key,
            microcanonical=microcanonical,
            alpha=alpha,
            save_frac=save_frac,
            C=C,
            early_stop=early_stop,
            r_end=r_end,
            bias_type=bias_type,
            diagonal_preconditioning=diagonal_preconditioning,
            integrator_coefficients=integrator_coefficients,
            steps_per_sample=steps_per_sample,
            acc_prob=acc_prob,
            observables_for_bias=observables_for_bias,
            all_chains_info=all_chains_info,
            diagnostics=diagnostics,
            contract=contract,
            superchain_size=superchain_size,
        )
    elapsed = time.perf_counter() - start
    print(f"Sampling took {elapsed:.2f} s")

    return {
        "final_state": final_state,
        "info": info,
        "gradient_calls_per_step": gradient_calls_per_step,
        "acc_prob": used_acc_prob,
    }
