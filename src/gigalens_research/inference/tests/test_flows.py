"""Tests for gigalens_research.inference.flows.

Runnable BOTH via pytest and as a plain script::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \\
    PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:<repo>/src \\
    <env>/bin/python test_flows.py

The env used to develop these (gigalens_oldapi) has no pytest, so the script
runner below is the primary path. Fixed seeds throughout; the nontrivial flows
are tested with RANDOMLY PERTURBED params (identity-init flows have trivial
Jacobians -- perturbing is what makes the round-trip / Jacobian tests real).
"""

import math
import os

# Enforce float64 even if a caller forgot; harmless if already set.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
from jax import random as jr
from jax.nn.initializers import glorot_normal, normal

jax.config.update("jax_enable_x64", True)

import optax

from gigalens_research.inference.flows import (
    Affine,
    Sequential,
    make_identity_flow,
    make_numpyro_iaf_flow,
    make_whitened_spline_flow,
    neg_elbo_loss,
    forward_kl_loss,
    _std_normal_logprob,
    _create_mask,
    _DT,
)

DIM = 4
RTOL_JAC = 1e-6
ATOL_RT = 1e-8


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _perturb(params, key, scale):
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jr.split(key, len(leaves))
    new = [leaf + scale * jr.normal(k, leaf.shape, dtype=leaf.dtype)
           for leaf, k in zip(leaves, keys)]
    return jax.tree_util.tree_unflatten(treedef, new)


def _build_iaf(perturb_scale=0.5, seed=0):
    init_params, make_bij = make_numpyro_iaf_flow(jr.PRNGKey(seed), DIM)
    params = _perturb(init_params, jr.PRNGKey(seed + 100), perturb_scale)
    return init_params, params, make_bij


def _build_spline(perturb_scale=0.1, seed=1, qz_loc=None, qz_scale_tril=None):
    if qz_loc is None:
        qz_loc = jnp.array([0.3, -0.7, 1.1, -0.2], dtype=_DT)
    if qz_scale_tril is None:
        # nontrivial lower-triangular whitening
        A = np.array([[1.3, 0.0, 0.0, 0.0],
                      [0.4, 0.9, 0.0, 0.0],
                      [-0.2, 0.5, 1.1, 0.0],
                      [0.1, -0.3, 0.2, 0.8]])
        qz_scale_tril = jnp.asarray(A, dtype=_DT)
    init_params, make_bij = make_whitened_spline_flow(
        jr.PRNGKey(seed), DIM, qz_loc, qz_scale_tril)
    params = _perturb(init_params, jr.PRNGKey(seed + 100), perturb_scale)
    return init_params, params, make_bij, qz_loc, qz_scale_tril


def _rand_batch(seed, n=6, dim=DIM):
    return jr.normal(jr.PRNGKey(seed), (n, dim), dtype=_DT)


# --------------------------------------------------------------------------
# 1. round-trip
# --------------------------------------------------------------------------
def _roundtrip(make_bij, params, seed):
    bij = make_bij(params)
    u = _rand_batch(seed)
    z = bij.forward(u)
    u2 = bij.inverse(z)
    assert u2.dtype == _DT
    err_fi = float(jnp.max(jnp.abs(u2 - u)))
    z2 = bij.inverse(_rand_batch(seed + 1))
    z3 = bij.forward(z2)
    # forward(inverse(z)) == z on independently drawn z
    zz = _rand_batch(seed + 1)
    err_if = float(jnp.max(jnp.abs(bij.forward(bij.inverse(zz)) - zz)))
    assert err_fi < ATOL_RT, f"inverse(forward(u)) err={err_fi:g}"
    assert err_if < ATOL_RT, f"forward(inverse(z)) err={err_if:g}"
    return err_fi, err_if


def test_roundtrip_iaf():
    _, params, make_bij = _build_iaf()
    e = _roundtrip(make_bij, params, seed=10)
    print(f"    IAF roundtrip max err: {max(e):.2e}")


def test_roundtrip_spline():
    _, params, make_bij, _, _ = _build_spline()
    e = _roundtrip(make_bij, params, seed=11)
    print(f"    spline roundtrip max err: {max(e):.2e}")


# --------------------------------------------------------------------------
# 2. Jacobian ground truth: fldj(u) == slogdet(jac(forward)(u))  (THE test)
# --------------------------------------------------------------------------
def _jac_check(make_bij, params, seed, label):
    bij = make_bij(params)
    us = _rand_batch(seed, n=5)
    worst = 0.0
    for u in us:
        fldj = float(bij.forward_log_det_jacobian(u, event_ndims=1))
        J = jax.jacobian(bij.forward)(u)
        sign, ladj = jnp.linalg.slogdet(J)
        assert float(sign) > 0, f"{label}: non-positive Jacobian det"
        d = abs(fldj - float(ladj))
        worst = max(worst, d)
        assert d < 1e-6, f"{label}: fldj={fldj:.8f} vs slogdet={float(ladj):.8f}"
    print(f"    {label} fldj-vs-slogdet worst |diff|: {worst:.2e}")


def test_jacobian_iaf():
    _, params, make_bij = _build_iaf()
    _jac_check(make_bij, params, seed=20, label="IAF")


def test_jacobian_spline():
    _, params, make_bij, _, _ = _build_spline()
    _jac_check(make_bij, params, seed=21, label="spline")


# --------------------------------------------------------------------------
# 3. consistency: fldj(u) == -ildj(forward(u))
# --------------------------------------------------------------------------
def _consistency(make_bij, params, seed, label):
    bij = make_bij(params)
    u = _rand_batch(seed)
    fldj = bij.forward_log_det_jacobian(u, event_ndims=1)
    ildj = bij.inverse_log_det_jacobian(bij.forward(u), event_ndims=1)
    err = float(jnp.max(jnp.abs(fldj + ildj)))
    assert err < 1e-8, f"{label}: fldj != -ildj (err={err:g})"
    print(f"    {label} fldj+ildj max: {err:.2e}")


def test_consistency_iaf():
    _, params, make_bij = _build_iaf()
    _consistency(make_bij, params, seed=30, label="IAF")


def test_consistency_spline():
    _, params, make_bij, _, _ = _build_spline()
    _consistency(make_bij, params, seed=31, label="spline")


# --------------------------------------------------------------------------
# 4. init behaviour
# --------------------------------------------------------------------------
def test_spline_init_is_affine_whitening():
    """At init the spline flow equals the affine whitening map EXACTLY."""
    init_params, _params, make_bij, qz_loc, qz_L = _build_spline()
    bij = make_bij(init_params)  # UNPERTURBED init
    u = _rand_batch(40, n=8)
    z = bij.forward(u)
    z_ref = qz_loc + u @ qz_L.T
    err = float(jnp.max(jnp.abs(z - z_ref)))
    assert err < 1e-10, f"spline init != affine whitening (err={err:g})"
    # fldj at init == constant log|det qz_scale_tril| (couplings contribute 0)
    fldj = bij.forward_log_det_jacobian(u, event_ndims=1)
    logdetL = float(jnp.sum(jnp.log(jnp.abs(jnp.diag(qz_L)))))
    assert float(jnp.max(jnp.abs(fldj - logdetL))) < 1e-10
    print(f"    spline init == affine whitening (err {err:.2e}); "
          f"fldj==log|detL|={logdetL:.4f}")


def test_iaf_init_reproduces_numpyro_scheme():
    """Pin NumPyro's AutoIAFNormal init scheme (numpyro is not installed here,
    so we pin against jax's glorot_normal / normal directly).

    NumPyro init (masked_dense.py): per Dense layer W ~ glorot_normal() masked,
    b ~ normal(stddev=1e-2). This is NOT an identity init -- at init the flow is
    a small random *near-identity* perturbation. We verify (a) the drawn params
    match glorot_normal/normal bit-for-bit for the first block/layer given the
    documented key-splitting, (b) log_scale stays within the [-5, 3] clamp, and
    (c) the flow is genuinely non-identity yet close-ish to identity.
    """
    key = jr.PRNGKey(7)
    init_params, make_bij = make_numpyro_iaf_flow(key, DIM)
    masks = [jnp.asarray(m, dtype=_DT)
             for m in _create_mask(DIM, [DIM, DIM], np.arange(DIM), 2)]

    # (a) reproduce block 0, layer 0 W,b from the exact key path
    block_keys = jr.split(key, 3)              # num_flows=3
    k0 = block_keys[0]
    _, kw, kb = jr.split(k0, 3)                 # first layer of _init_arn_params
    W0_exp = glorot_normal()(kw, masks[0].shape, _DT)
    b0_exp = normal()(kb, masks[0].shape[-1:], _DT)
    W0, b0 = init_params["blocks"][0][0]
    assert jnp.allclose(W0, W0_exp) and jnp.allclose(b0, b0_exp), \
        "IAF init does not match glorot_normal/normal scheme"

    # (b) log_scale within clamp; (c) non-identity but near identity
    bij = make_bij(init_params)
    u = _rand_batch(41, n=8)
    z = bij.forward(u)
    dev = float(jnp.max(jnp.abs(z - u)))
    assert dev > 1e-6, "IAF init is unexpectedly the exact identity"
    assert dev < 5.0, "IAF init deviates implausibly far from identity"
    # forward(0) = mean(0) (scale*0 == 0): bias-dominated -> small
    z0 = bij.forward(jnp.zeros((DIM,), dtype=_DT))
    print(f"    IAF init matches glorot/normal scheme; max|T(u)-u|={dev:.3f}; "
          f"|T(0)|={float(jnp.max(jnp.abs(z0))):.3e}")


# --------------------------------------------------------------------------
# 5. losses
# --------------------------------------------------------------------------
def test_identity_losses():
    init_params, make_bij = make_identity_flow(DIM)
    # target = N(0, I): reverse-KL of identity flow is exactly 0
    nelbo = neg_elbo_loss(init_params, make_bij, _std_normal_logprob,
                          jr.PRNGKey(50), n_draws=4096, dim=DIM)
    assert abs(float(nelbo)) < 1e-10, f"identity neg_elbo={float(nelbo):g} != 0"
    # forward-KL sanity: identity flow -> mean(-base.log_prob(z))
    z = _rand_batch(51, n=64)
    fkl = forward_kl_loss(init_params, make_bij, z)
    ref = float(jnp.mean(-_std_normal_logprob(z)))
    assert abs(float(fkl) - ref) < 1e-10
    print(f"    identity neg_elbo={float(nelbo):.2e}; "
          f"fwd_kl matches mean(-logN)={ref:.4f}")


def test_reverse_kl_decreases():
    """Reverse-KL decreases over a few adam steps on a correlated 2-D Gaussian."""
    dim = 2
    rho = 0.9
    cov = jnp.array([[1.0, rho], [rho, 1.0]], dtype=_DT)
    prec = jnp.linalg.inv(cov)
    logdet = float(jnp.linalg.slogdet(cov)[1])

    def target_log_prob(z):
        q = jnp.einsum("...i,ij,...j->...", z, prec, z)
        return -0.5 * (q + dim * math.log(2 * math.pi) + logdet)

    qz_loc = jnp.zeros(dim, dtype=_DT)
    qz_L = jnp.eye(dim, dtype=_DT)  # start from white base -> couplings do the work
    init_params, make_bij = make_whitened_spline_flow(
        jr.PRNGKey(3), dim, qz_loc, qz_L, num_layers=6, num_bins=8)

    opt = optax.adam(5e-3)
    state = opt.init(init_params)

    def loss_fn(p, key):
        return neg_elbo_loss(p, make_bij, target_log_prob, key,
                             n_draws=512, dim=dim)

    @jax.jit
    def step(p, state, key):
        l, g = jax.value_and_grad(loss_fn)(p, key)
        updates, state = opt.update(g, state, p)
        p = optax.apply_updates(p, updates)
        return p, state, l

    key = jr.PRNGKey(999)
    p = init_params
    losses = []
    for i in range(250):
        key, sk = jr.split(key)
        p, state, l = step(p, state, sk)
        losses.append(float(l))
    l0 = float(np.mean(losses[:10]))
    l1 = float(np.mean(losses[-10:]))
    assert np.isfinite(l1), "loss became non-finite"
    assert l1 < l0 - 0.05, f"reverse-KL did not decrease: {l0:.4f} -> {l1:.4f}"
    print(f"    reverse-KL: {l0:.4f} -> {l1:.4f} (250 adam steps)")


# --------------------------------------------------------------------------
# 5b. out-of-range finiteness (regression: NaN grads past the spline range)
# --------------------------------------------------------------------------
# Real-world failure this pins: whitened demo-posterior samples reached
# max|T^-1(z)| = 31.4 with spline_range=6.0; forward_kl_loss was FINITE but
# its gradient was all-NaN (untaken spline branch differentiated at
# out-of-range inputs -> citardauq sqrt(0)/vanishing-denominator NaNs leaked
# through jnp.where). Fixed by the double-where trick in _rqs_forward/_inverse.
def _grad_finite(fn, params):
    g = jax.grad(fn)(params)
    return all(bool(jnp.all(jnp.isfinite(l)))
               for l in jax.tree_util.tree_leaves(g))


def _out_of_range_checks(make_bij, params, pts, label):
    """Value+grad finiteness of both directions and both ldjs at given pts."""
    def fwd_loss(p):
        b = make_bij(p)
        return jnp.sum(b.forward(pts)) + jnp.sum(
            b.forward_log_det_jacobian(pts, event_ndims=1))

    def inv_loss(p):
        b = make_bij(p)
        return jnp.sum(b.inverse(pts)) + jnp.sum(
            b.inverse_log_det_jacobian(pts, event_ndims=1))

    for name, fn in [("forward+fldj", fwd_loss), ("inverse+ildj", inv_loss)]:
        v = fn(params)
        assert bool(jnp.isfinite(v)), f"{label} {name}: non-finite value"
        assert _grad_finite(fn, params), f"{label} {name}: non-finite grad"


def test_out_of_range_grads_spline():
    """Spline: points at 5x the range -> exact identity for C, finite grads."""
    R = 6.0
    _init, params, make_bij, qz_loc, qz_L = _build_spline()
    pts = 5.0 * R * jr.normal(jr.PRNGKey(70), (16, DIM), dtype=_DT)
    _out_of_range_checks(make_bij, params, pts, "spline")
    # forward_kl_loss grad finite on far-out z (inverse maps them far outside
    # the spline box through the affine whitening)
    z_far = qz_loc + (5.0 * R * jr.normal(jr.PRNGKey(71), (16, DIM), dtype=_DT)) @ qz_L.T
    assert bool(jnp.isfinite(forward_kl_loss(params, make_bij, z_far)))
    assert _grad_finite(lambda p: forward_kl_loss(p, make_bij, z_far), params), \
        "spline forward_kl grad NaN at out-of-range z"
    # EXACT identity of the coupling stack C outside the box: strip the affine
    bij = make_bij(params)
    couplings = Sequential(bij.bijectors[:-1])
    far = jnp.full((DIM,), 5.0 * R, dtype=_DT)
    assert bool(jnp.all(couplings.forward(far) == far)), "C(far) != far exactly"
    assert bool(jnp.all(couplings.inverse(far) == far))
    assert float(couplings.forward_log_det_jacobian(far, event_ndims=1)) == 0.0
    print("    spline out-of-range: exact identity, all grads finite")


def test_out_of_range_grads_iaf():
    """IAF has no bounded range but must also stay finite far out (clamp)."""
    _init, params, make_bij = _build_iaf()
    pts = 30.0 * jr.normal(jr.PRNGKey(72), (16, DIM), dtype=_DT)
    _out_of_range_checks(make_bij, params, pts, "IAF")
    assert _grad_finite(lambda p: forward_kl_loss(p, make_bij, pts), params), \
        "IAF forward_kl grad NaN at far-out z"
    print("    IAF far-out points: values and grads finite")


def test_spline_boundary_fldj():
    """Value AND grad of fldj/ildj finite at exactly +/- spline_range."""
    R = 6.0
    _init, params, make_bij, _, _ = _build_spline()
    for sgn in (+1.0, -1.0):
        pts = jnp.full((2, DIM), sgn * R, dtype=_DT)

        def floss(p):
            b = make_bij(p)
            return jnp.sum(b.forward_log_det_jacobian(pts, event_ndims=1))

        def iloss(p):
            b = make_bij(p)
            return jnp.sum(b.inverse_log_det_jacobian(pts, event_ndims=1))

        for name, fn in [("fldj", floss), ("ildj", iloss)]:
            v = fn(params)
            assert bool(jnp.isfinite(v)), f"{name} at {sgn:+.0f}R non-finite"
            assert _grad_finite(fn, params), f"grad {name} at {sgn:+.0f}R non-finite"
    print("    spline boundary (+/-range): fldj/ildj values and grads finite")


_REPRO_BASE = ("/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/"
               "flow-precond/experiments/flow_precond/")


def test_real_reproducer_fkl_step():
    """Real demo data: one adam fkl step must stay finite (the exact failure
    that motivated the double-where fix). Skips if the npz artifacts are absent.
    """
    cache_p = _REPRO_BASE + "demo_validation_out/map_svi_cache.npz"
    gate_p = _REPRO_BASE + "gate_i_out/gate_i_arrays.npz"
    if not (os.path.exists(cache_p) and os.path.exists(gate_p)):
        print("    [skipped] reproducer npz files not found")
        return
    import optax
    c = np.load(cache_p)
    qz_loc = jnp.asarray(c["qz_loc"], dtype=_DT)
    qz_L = jnp.asarray(c["qz_scale_tril"], dtype=_DT)
    z = jnp.asarray(np.load(gate_p)["samples_vanilla"].reshape(-1, 22), dtype=_DT)
    init_params, make_bij = make_whitened_spline_flow(
        jr.PRNGKey(0), 22, qz_loc, qz_L,
        num_layers=6, num_bins=8, spline_range=6.0, hidden_dims=(64, 64))
    u = make_bij(init_params).inverse(z)
    max_u = float(jnp.max(jnp.abs(u)))
    assert max_u > 6.0, "reproducer no longer exercises out-of-range points"
    loss0 = float(forward_kl_loss(init_params, make_bij, z))
    assert np.isfinite(loss0)
    g = jax.grad(forward_kl_loss)(init_params, make_bij, z)
    assert all(bool(jnp.all(jnp.isfinite(l)))
               for l in jax.tree_util.tree_leaves(g)), "fkl grad NaN (real data)"
    opt = optax.adam(1e-4)
    upd, _ = opt.update(g, opt.init(init_params), init_params)
    p1 = optax.apply_updates(init_params, upd)
    assert all(bool(jnp.all(jnp.isfinite(l)))
               for l in jax.tree_util.tree_leaves(p1)), "params NaN after step"
    loss1 = float(forward_kl_loss(p1, make_bij, z))
    assert np.isfinite(loss1)
    print(f"    real data: max|T^-1(z)|={max_u:.1f}, fkl {loss0:.3f}->{loss1:.3f}, "
          f"grads/params finite")


# --------------------------------------------------------------------------
# 6. dtype
# --------------------------------------------------------------------------
def test_dtype_float64():
    for label, (make_bij, params) in {
        "IAF": (_build_iaf()[2], _build_iaf()[1]),
        "spline": (_build_spline()[2], _build_spline()[1]),
    }.items():
        bij = make_bij(params)
        u = _rand_batch(60)
        assert bij.forward(u).dtype == _DT, label
        assert bij.inverse(bij.forward(u)).dtype == _DT, label
        assert bij.forward_log_det_jacobian(u, event_ndims=1).dtype == _DT, label
    print("    all outputs float64")


# --------------------------------------------------------------------------
# script runner
# --------------------------------------------------------------------------
def _all_tests():
    return [(n, g) for n, g in sorted(globals().items())
            if n.startswith("test_") and callable(g)]


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
    print(f"\n{'ALL GREEN' if n_fail == 0 else str(n_fail) + ' FAILED'} "
          f"({len(_all_tests())} tests)")
    raise SystemExit(1 if n_fail else 0)
