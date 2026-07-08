"""Normalizing-flow library for flow-preconditioned MCMC (NeuTra-style).

Every builder returns ``(init_params, make_bijector)`` where
``make_bijector(params)`` yields an object exposing the tfp-style bijector
interface used by :mod:`gigalens_research.inference.flow_precond`::

    b.forward(u)                              # latent u -> target z   (fast direction)
    b.inverse(z)                              # target z -> latent u
    b.forward_log_det_jacobian(u, event_ndims=1)   # log|det dz/du|
    b.inverse_log_det_jacobian(z, event_ndims=1)   # log|det du/dz|

All operate on ``(..., dim)`` with ``event_ndims=1`` (the last axis is the
event), are pure functions of an explicit ``params`` pytree (no captured
state), and are jit/grad friendly so optax can train ``params`` directly.

The base distribution N(0, I) is handled OUTSIDE these builders (by the
caller / the loss functions here); the bijectors only implement T.

tfp-jax caveat (verified in this env, tfp 0.26.0-dev on the JAX substrate):
``tfb.AutoregressiveNetwork`` and ``masked_autoregressive_default_template``
are broken (the keras shim is a stub), so the IAF MADE network below is
hand-written as pure-jax functions over an explicit params pytree. The RQ
spline is likewise hand-written (Durkan et al. 2019): ``tfb.Rational
QuadraticSpline`` works eagerly but traces a data-dependent shape under
``jax.jit`` (breaks the jitted training/sampling loop), so this module avoids
tfp bijectors entirely and depends only on JAX.

All flow parameters are float64 (the pipeline runs with JAX_ENABLE_X64=1).
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import random as jr
from jax.nn.initializers import glorot_normal, normal

_DT = jnp.float64
_LOG2PI = math.log(2.0 * math.pi)


# ---------------------------------------------------------------------------
# Bijector interface + composition
# ---------------------------------------------------------------------------
class _Bijector:
    """Minimal tfp-style bijector base (event_ndims == 1 always)."""

    def forward(self, u):
        raise NotImplementedError

    def inverse(self, z):
        raise NotImplementedError

    def forward_log_det_jacobian(self, u, event_ndims=1):
        raise NotImplementedError

    def inverse_log_det_jacobian(self, z, event_ndims=1):
        # Generic, always-correct fallback: ildj(z) = -fldj(inverse(z)).
        assert event_ndims == 1
        return -self.forward_log_det_jacobian(self.inverse(z), event_ndims=1)


class Sequential(_Bijector):
    """Compose bijectors in *list order*: ``forward`` applies ``bijectors[0]``
    first, then ``bijectors[1]``, ... (unlike ``tfb.Chain`` which is reversed).
    This mirrors NumPyro's ``TransformedDistribution(base, flows)`` where the
    base sample is pushed through ``flows[0]``, then ``flows[1]``, ....
    """

    def __init__(self, bijectors):
        self.bijectors = list(bijectors)

    def forward(self, u):
        x = u
        for b in self.bijectors:
            x = b.forward(x)
        return x

    def inverse(self, z):
        x = z
        for b in reversed(self.bijectors):
            x = b.inverse(x)
        return x

    def forward_log_det_jacobian(self, u, event_ndims=1):
        assert event_ndims == 1
        x = u
        total = jnp.zeros(u.shape[:-1], dtype=u.dtype)
        for b in self.bijectors:
            total = total + b.forward_log_det_jacobian(x, event_ndims=1)
            x = b.forward(x)
        return total

    def inverse_log_det_jacobian(self, z, event_ndims=1):
        assert event_ndims == 1
        return -self.forward_log_det_jacobian(self.inverse(z), event_ndims=1)


class Permute(_Bijector):
    """Fixed permutation of the last axis (no parameters, fldj == 0).

    Reproduces NumPyro ``PermuteTransform``: forward is ``x[..., permutation]``.
    """

    def __init__(self, permutation):
        self.perm = np.asarray(permutation)
        self.inv = np.argsort(self.perm)

    def forward(self, u):
        return u[..., self.perm]

    def inverse(self, z):
        return z[..., self.inv]

    def forward_log_det_jacobian(self, u, event_ndims=1):
        assert event_ndims == 1
        return jnp.zeros(u.shape[:-1], dtype=u.dtype)

    def inverse_log_det_jacobian(self, z, event_ndims=1):
        assert event_ndims == 1
        return jnp.zeros(z.shape[:-1], dtype=z.dtype)


class Affine(_Bijector):
    """Fixed affine whitening map ``y = loc + scale_tril @ x`` (constants, not
    trained). ``scale_tril`` is lower-triangular; fldj = sum(log|diag|)."""

    def __init__(self, loc, scale_tril):
        self.loc = jnp.asarray(loc, dtype=_DT)
        self.L = jnp.asarray(scale_tril, dtype=_DT)
        # Precompute L^{-1} for a batched triangular solve: x = (y-loc) @ Linv.T
        self.Linv = jax.scipy.linalg.solve_triangular(
            self.L, jnp.eye(self.L.shape[-1], dtype=_DT), lower=True
        )
        self._logdet = jnp.sum(jnp.log(jnp.abs(jnp.diag(self.L))))

    def forward(self, u):
        return self.loc + u @ self.L.T

    def inverse(self, z):
        return (z - self.loc) @ self.Linv.T

    def forward_log_det_jacobian(self, u, event_ndims=1):
        assert event_ndims == 1
        return jnp.broadcast_to(self._logdet, u.shape[:-1])

    def inverse_log_det_jacobian(self, z, event_ndims=1):
        assert event_ndims == 1
        return jnp.broadcast_to(-self._logdet, z.shape[:-1])


# ---------------------------------------------------------------------------
# BUILDER 3 -- identity flow
# ---------------------------------------------------------------------------
class _Identity(_Bijector):
    def forward(self, u):
        return u

    def inverse(self, z):
        return z

    def forward_log_det_jacobian(self, u, event_ndims=1):
        assert event_ndims == 1
        return jnp.zeros(u.shape[:-1], dtype=u.dtype)

    def inverse_log_det_jacobian(self, z, event_ndims=1):
        assert event_ndims == 1
        return jnp.zeros(z.shape[:-1], dtype=z.dtype)


def make_identity_flow(dim):
    """Trivial flow: empty params, identity bijector via the shared interface."""
    del dim
    init_params = {}

    def make_bijector(params):
        del params
        return _Identity()

    return init_params, make_bijector


# ===========================================================================
# BUILDER 1 -- NumPyro AutoIAFNormal reproduction
# ===========================================================================
# Faithful reproduction of NumPyro's AutoIAFNormal guide transform.
# Sources (pyro-ppl/numpyro @ master), reproduced exactly below:
#   * numpyro/infer/autoguide.py -- class AutoIAFNormal:
#       __init__(..., num_flows=3, hidden_dims=None, skip_connections=False,
#                nonlinearity=stax.Elu)
#       _get_posterior(): hidden_dims defaults to [latent_dim, latent_dim];
#         flows = []; for i in range(num_flows):
#             if i > 0: flows.append(PermuteTransform(jnp.arange(latent_dim)[::-1]))
#             flows.append(InverseAutoregressiveTransform(arn_i))
#         -> stack for num_flows=3 is [IAF0, Permute, IAF1, Permute, IAF2].
#   * numpyro/distributions/flows.py -- InverseAutoregressiveTransform:
#       forward  __call__: mean, log_scale = arn(x); log_scale clamped to
#         [-5, 3] via _clamp_preserve_gradients; return exp(log_scale)*x + mean.
#       _inverse: x = fori_loop(0, dim, _update_x, zeros); each step
#         mean, log_scale = arn(x); x = (y - mean) * exp(-clamp(log_scale)).
#       log_abs_det_jacobian: clamp(log_scale).sum(-1)  (triangular Jacobian).
#       _clamp_preserve_gradients(x,mn,mx) = x + stop_gradient(clip(x,mn,mx)-x).
#   * numpyro/nn/auto_reg_nn.py -- AutoregressiveNN / create_mask /
#       sample_mask_indices (MADE masks, reproduced verbatim below); default
#       param_dims=[1,1] -> outputs (mean, log_scale); nonlinearity between
#       masked layers, none after the last; skip_connections wiring.
#   * numpyro/nn/masked_dense.py -- MaskedDense(mask, W_init=glorot_normal(),
#       b_init=normal()): apply is dot(inputs, W*mask) + b; init draws
#       W~glorot_normal, b~normal(stddev=1e-2). This is the init scheme we
#       reproduce so the flow behaves as NumPyro's does at init (a small random
#       *near-identity* perturbation -- NOT exact identity; see test 4).

_LS_MIN, _LS_MAX = -5.0, 3.0  # flows.py: log_scale_min_clip / log_scale_max_clip


def _clamp_preserve_gradients(x, mn, mx):
    # numpyro/distributions/flows.py:_clamp_preserve_gradients
    return x + jax.lax.stop_gradient(jnp.clip(x, mn, mx) - x)


def _sample_mask_indices(input_dim, hidden_dim):
    # numpyro/nn/auto_reg_nn.py:sample_mask_indices
    indices = np.linspace(1, input_dim, num=hidden_dim)
    return np.round(indices)


def _create_mask(input_dim, hidden_dims, permutation, output_dim_multiplier):
    """numpyro/nn/auto_reg_nn.py:create_mask (non-conditional MADE masks).

    Returns a list of masks (one per Dense layer), each of shape
    (fan_in, fan_out). Reproduced verbatim in numpy (masks are static).
    """
    permutation = np.asarray(permutation)
    var_index = np.zeros(permutation.shape[0])
    var_index[permutation] = np.arange(input_dim)

    input_indices = 1 + var_index
    hidden_indices = [_sample_mask_indices(input_dim - 1, h) for h in hidden_dims]
    output_indices = np.tile(var_index + 1, output_dim_multiplier)

    masks = [hidden_indices[0][None, :] >= input_indices[:, None]]
    for i in range(1, len(hidden_dims)):
        masks.append(hidden_indices[i][None, :] >= hidden_indices[i - 1][:, None])
    masks.append(output_indices[None, :] > hidden_indices[-1][:, None])
    return masks


def _arn_apply(params, x, masks):
    """MADE apply, reproducing numpyro AutoregressiveNN.apply_fun with
    param_dims=[1, 1] and nonlinearity=ELU (stax.Elu).

    ``params`` is a list of (W, b) per masked Dense layer; ``masks`` the static
    MADE masks. Returns (mean, log_scale), each shape x.shape (..., dim).
    """
    input_dim = x.shape[-1]
    h = x
    n = len(masks)
    for i, (mask, (W, b)) in enumerate(zip(masks, params)):
        # MaskedDense: dot(inputs, W * mask) + b
        h = h @ (W * mask) + b
        if i < n - 1:
            h = jax.nn.elu(h)  # stax.Elu; no nonlinearity after last layer
    # reshape (..., 2*dim) -> (..., 2, dim); moveaxis(-2, 0) -> (2, ..., dim)
    out = jnp.reshape(h, x.shape[:-1] + (2, input_dim))
    out = jnp.moveaxis(out, -2, 0)
    mean, log_scale = out[0], out[1]
    return mean, log_scale


def _init_arn_params(key, masks):
    """Init one MADE network exactly as NumPyro/stax does: per Dense layer,
    W ~ glorot_normal() masked by the layer mask, b ~ normal(stddev=1e-2).
    (numpyro/nn/masked_dense.py: W_init=glorot_normal(), b_init=normal().)
    """
    w_init = glorot_normal()
    b_init = normal()  # jax default stddev = 1e-2
    params = []
    for mask in masks:
        key, kw, kb = jr.split(key, 3)
        W = w_init(kw, mask.shape, _DT)
        b = b_init(kb, mask.shape[-1:], _DT)
        params.append((W, b))
    return params


class _IAFBlock(_Bijector):
    """One InverseAutoregressiveTransform block (no permute).

    forward is the fast single-pass IAF direction (latent x -> y); inverse is
    the exact iterative fori_loop from NumPyro (D passes for a D-dim input).
    """

    def __init__(self, arn_params, masks):
        self.params = arn_params
        self.masks = masks

    def _mean_logscale(self, x):
        mean, log_scale = _arn_apply(self.params, x, self.masks)
        log_scale = _clamp_preserve_gradients(log_scale, _LS_MIN, _LS_MAX)
        return mean, log_scale

    def forward(self, u):
        # numpyro flows.py call_with_intermediates: exp(log_scale)*x + mean
        mean, log_scale = self._mean_logscale(u)
        return jnp.exp(log_scale) * u + mean

    def inverse(self, z):
        # numpyro flows.py _inverse: fori_loop over dims, x starts at zeros.
        dim = z.shape[-1]

        def _update_x(_, x):
            mean, log_scale = self._mean_logscale(x)
            return (z - mean) * jnp.exp(-log_scale)

        return jax.lax.fori_loop(0, dim, _update_x, jnp.zeros_like(z))

    def forward_log_det_jacobian(self, u, event_ndims=1):
        assert event_ndims == 1
        # numpyro flows.py log_abs_det_jacobian: clamp(log_scale).sum(-1).
        # Jacobian is triangular (MADE autoregressive), diag = exp(log_scale).
        _, log_scale = self._mean_logscale(u)
        return jnp.sum(log_scale, axis=-1)

    def inverse_log_det_jacobian(self, z, event_ndims=1):
        assert event_ndims == 1
        return -self.forward_log_det_jacobian(self.inverse(z), event_ndims=1)


def make_numpyro_iaf_flow(key, dim, num_flows=3, hidden_dims=None):
    """Reproduce NumPyro's AutoIAFNormal transform T (base N(0,I) is external).

    Defaults match AutoIAFNormal: num_flows=3, hidden_dims=[dim, dim] (2 hidden
    layers), ELU, skip_connections=False. The returned bijector's ``forward``
    is the fast IAF latent->target map, evaluated (with its fldj) every sampler
    step; ``inverse`` is the exact iterative solve.
    """
    if hidden_dims is None:
        hidden_dims = [dim, dim]  # autoguide.py: [latent_dim, latent_dim]

    # Every IAF block shares the same MADE mask set: identity permutation
    # (arange(dim)); the between-block reversal is done by Permute, exactly as
    # AutoIAFNormal separates PermuteTransform from the ARN's own permutation.
    permutation = np.arange(dim)
    masks_np = _create_mask(dim, hidden_dims, permutation, output_dim_multiplier=2)
    masks = [jnp.asarray(m, dtype=_DT) for m in masks_np]

    keys = jr.split(key, num_flows)
    init_params = {"blocks": [_init_arn_params(k, masks) for k in keys]}

    rev = np.arange(dim)[::-1]  # PermuteTransform(jnp.arange(dim)[::-1])

    def make_bijector(params):
        blocks = params["blocks"]
        seq = []
        for i in range(num_flows):
            if i > 0:
                seq.append(Permute(rev))
            seq.append(_IAFBlock(blocks[i], masks))
        return Sequential(seq)

    return init_params, make_bijector


# ===========================================================================
# BUILDER 2 -- whitened RQ-spline coupling flow (identity C at init)
# ===========================================================================
# Activation for the conditioner MLPs: tanh (smooth, standard for coupling
# conditioners). The final conditioner layer is ZERO-initialized, so at init
# every coupling layer outputs the identity transform (see below) and
# C = identity exactly; hence T(u) = qz_loc + qz_scale_tril @ u at init.
_SOFTPLUS_UNIT = math.log(math.e - 1.0)  # softplus(x + this)=1 exactly at x=0


def _mlp_apply(mlp_params, x):
    """Standard MLP with tanh hidden activations; last layer is linear."""
    h = x
    n = len(mlp_params)
    for i, (W, b) in enumerate(mlp_params):
        h = h @ W + b
        if i < n - 1:
            h = jnp.tanh(h)
    return h


def _init_mlp(key, in_dim, hidden_dims, out_dim):
    """Hidden layers: glorot_normal W, zero bias. Final layer: ZERO W and b
    (guarantees identity coupling at init)."""
    w_init = glorot_normal()
    sizes = [in_dim] + list(hidden_dims) + [out_dim]
    params = []
    last = len(sizes) - 2
    for i in range(len(sizes) - 1):
        key, kw = jr.split(key)
        fi, fo = sizes[i], sizes[i + 1]
        if i == last:
            W = jnp.zeros((fi, fo), dtype=_DT)  # zero final layer -> identity C
        else:
            W = w_init(kw, (fi, fo), _DT)
        b = jnp.zeros((fo,), dtype=_DT)
        params.append((W, b))
    return params


# --- Hand-written monotonic rational-quadratic spline -----------------------
# Durkan, Bekasov, Murray, Papamakarios (2019), "Neural Spline Flows"
# [arXiv:1906.04032], Appendix A. Elementwise over the last axis; params carry a
# trailing bin axis. Linear (identity, slope 1) tails outside [-B, B], so the
# transform is identity there. Hand-written (rather than tfb.RationalQuadratic
# Spline) because the tfp-jax spline traces a data-dependent shape under jit
# (breaks jax.jit) -- the spec explicitly permits custom bijectors.
def _raw_to_spline_params(raw, num_bins, spline_range):
    """raw (..., 3*num_bins-1) -> (widths, heights, slopes).

    At raw == 0: softmax -> uniform widths/heights (each 2R/num_bins), and
    softplus(0 + log(e-1)) -> unit interior slopes; with the fixed unit boundary
    slopes this is the EXACT identity on [-R, R] (and identity outside). This is
    the identity-init recipe that makes C = identity at init.
    """
    total = 2.0 * spline_range
    rw = raw[..., :num_bins]
    rh = raw[..., num_bins : 2 * num_bins]
    rs = raw[..., 2 * num_bins :]
    widths = jax.nn.softmax(rw, axis=-1) * total
    heights = jax.nn.softmax(rh, axis=-1) * total
    slopes = jax.nn.softplus(rs + _SOFTPLUS_UNIT)  # interior knot slopes (K-1)
    return widths, heights, slopes


def _spline_knots(widths, heights, slopes, B):
    """Cumulative knot coords and full derivative array (unit boundary slopes)."""
    z = jnp.zeros_like(widths[..., :1])
    xk = jnp.concatenate([z - B, -B + jnp.cumsum(widths, axis=-1)], axis=-1)
    yk = jnp.concatenate([z - B, -B + jnp.cumsum(heights, axis=-1)], axis=-1)
    ones = jnp.ones_like(slopes[..., :1])
    d = jnp.concatenate([ones, slopes, ones], axis=-1)  # (..., K+1)
    return xk, yk, d


def _gather(a, idx):
    return jnp.take_along_axis(a, idx[..., None], axis=-1)[..., 0]


def _rqs_forward(x, widths, heights, slopes, B):
    """Forward spline + log|dy/dx|, elementwise; identity (logdet 0) outside."""
    K = widths.shape[-1]
    xk, yk, d = _spline_knots(widths, heights, slopes, B)
    inside = (x > -B) & (x < B)
    k = jnp.clip(jnp.sum(x[..., None] >= xk[..., 1:-1], axis=-1), 0, K - 1)
    xkk, wk = _gather(xk, k), _gather(widths, k)
    ykk, hk = _gather(yk, k), _gather(heights, k)
    dk, dk1 = _gather(d, k), _gather(d, k + 1)
    s = hk / wk
    xi = jnp.clip((x - xkk) / wk, 0.0, 1.0)
    xi1 = 1.0 - xi
    denom = s + (dk1 + dk - 2.0 * s) * xi * xi1
    y_in = ykk + hk * (s * xi * xi + dk * xi * xi1) / denom
    deriv = s * s * (dk1 * xi * xi + 2.0 * s * xi * xi1 + dk * xi1 * xi1) / (denom * denom)
    y = jnp.where(inside, y_in, x)
    logdet = jnp.where(inside, jnp.log(deriv), 0.0)
    return y, logdet


def _rqs_inverse(y, widths, heights, slopes, B):
    """Inverse spline + log|dx/dy|, elementwise; identity outside."""
    K = widths.shape[-1]
    xk, yk, d = _spline_knots(widths, heights, slopes, B)
    inside = (y > -B) & (y < B)
    k = jnp.clip(jnp.sum(y[..., None] >= yk[..., 1:-1], axis=-1), 0, K - 1)
    xkk, wk = _gather(xk, k), _gather(widths, k)
    ykk, hk = _gather(yk, k), _gather(heights, k)
    dk, dk1 = _gather(d, k), _gather(d, k + 1)
    s = hk / wk
    dy = y - ykk
    delta = dk1 + dk - 2.0 * s
    a = hk * (s - dk) + dy * delta
    b = hk * dk - dy * delta
    c = -s * dy
    disc = b * b - 4.0 * a * c
    xi = jnp.clip(2.0 * c / (-b - jnp.sqrt(jnp.maximum(disc, 0.0))), 0.0, 1.0)
    xi1 = 1.0 - xi
    denom = s + delta * xi * xi1
    deriv = s * s * (dk1 * xi * xi + 2.0 * s * xi * xi1 + dk * xi1 * xi1) / (denom * denom)
    x_in = xkk + xi * wk
    x = jnp.where(inside, x_in, y)
    logdet = jnp.where(inside, -jnp.log(deriv), 0.0)
    return x, logdet


class _RQSplineCoupling(_Bijector):
    """Masked RQ-spline coupling layer.

    ``active`` (bool, shape (dim,)) marks the transformed dims; the passive
    dims pass through unchanged and are the *only* input to the conditioner
    (active dims are zeroed before the MLP), so the Jacobian is block-triangular
    and fldj is the sum of per-active-dim spline log-slopes.
    """

    def __init__(self, mlp_params, active_mask, num_bins, spline_range):
        self.mlp = mlp_params
        self.active = jnp.asarray(active_mask, dtype=bool)
        self.activef = self.active.astype(_DT)
        self.passivef = (~self.active).astype(_DT)
        self.num_bins = num_bins
        self.B = float(spline_range)
        self.dim = int(active_mask.shape[0])
        self.pcount = 3 * num_bins - 1

    def _params(self, x):
        cond_in = x * self.passivef  # depend only on passive dims
        raw = _mlp_apply(self.mlp, cond_in)  # (..., dim * pcount)
        raw = jnp.reshape(raw, x.shape[:-1] + (self.dim, self.pcount))
        return _raw_to_spline_params(raw, self.num_bins, self.B)

    def forward(self, u):
        w, h, s = self._params(u)
        y, _ = _rqs_forward(u, w, h, s, self.B)
        return jnp.where(self.active, y, u)

    def inverse(self, z):
        # Passive dims are unchanged, so conditioner input z*passive == u*passive:
        # the spline is reconstructed exactly, then inverted (single closed-form
        # pass) on the active dims.
        w, h, s = self._params(z)
        x, _ = _rqs_inverse(z, w, h, s, self.B)
        return jnp.where(self.active, x, z)

    def forward_log_det_jacobian(self, u, event_ndims=1):
        assert event_ndims == 1
        w, h, s = self._params(u)
        _, ldj = _rqs_forward(u, w, h, s, self.B)  # per-dim
        return jnp.sum(ldj * self.activef, axis=-1)

    def inverse_log_det_jacobian(self, z, event_ndims=1):
        assert event_ndims == 1
        w, h, s = self._params(z)
        _, ldj = _rqs_inverse(z, w, h, s, self.B)
        return jnp.sum(ldj * self.activef, axis=-1)


def make_whitened_spline_flow(
    key,
    dim,
    qz_loc,
    qz_scale_tril,
    num_layers=6,
    num_bins=8,
    spline_range=6.0,
    hidden_dims=(64, 64),
):
    """T(u) = qz_loc + qz_scale_tril @ C(u), C = stack of RQ-spline couplings
    with alternating even/odd binary masks.

    At init C = identity EXACTLY (zero final conditioner layer + uniform
    bins/unit slopes), so T equals the affine whitening map (the SVI solution).
    fldj includes both the couplings and the constant log|det qz_scale_tril|.
    """
    dim = int(dim)
    idx = np.arange(dim)
    pcount = 3 * num_bins - 1
    out_dim = dim * pcount

    keys = jr.split(key, num_layers)
    masks = []
    mlp_params = []
    for i in range(num_layers):
        active = (idx % 2 == (i % 2))  # alternate even / odd indices
        masks.append(active)
        mlp_params.append(_init_mlp(keys[i], dim, hidden_dims, out_dim))

    init_params = {"couplings": mlp_params}
    affine = Affine(qz_loc, qz_scale_tril)

    def make_bijector(params):
        seq = []
        for i in range(num_layers):
            seq.append(
                _RQSplineCoupling(
                    params["couplings"][i], masks[i], num_bins, spline_range
                )
            )
        seq.append(affine)  # applied last: y = qz_loc + qz_scale_tril @ C(u)
        return Sequential(seq)

    return init_params, make_bijector


# ===========================================================================
# LOSSES
# ===========================================================================
def _std_normal_logprob(u):
    """log N(u; 0, I) summed over the event (last) axis."""
    return -0.5 * jnp.sum(u * u + _LOG2PI, axis=-1)


def neg_elbo_loss(
    params,
    make_bijector,
    target_log_prob_fn,
    key,
    n_draws,
    dim,
    base_log_prob_u_fn=None,
    base_sample_fn=None,
):
    """Reverse-KL objective (loss = -ELBO, LOWER is better).

        loss = E_{u~N(0,I)}[ base.log_prob(u) - fldj(u) - target_log_prob(T(u)) ]
             = E_q[ log q(z) - log p(z) ]                 (= KL(q || p) + const)

    where log q(z) = base.log_prob(u) - fldj(u) by change of variables and
    z = T(u). Minimizing this trains the pushforward q = T#N(0,I) toward the
    (unnormalized) target p. ``target_log_prob_fn`` maps (n, dim) -> (n,).
    """
    if base_sample_fn is None:
        base_sample_fn = lambda k, n: jr.normal(k, (n, dim), dtype=_DT)
    if base_log_prob_u_fn is None:
        base_log_prob_u_fn = _std_normal_logprob

    bij = make_bijector(params)
    u = base_sample_fn(key, n_draws)
    z = bij.forward(u)
    fldj = bij.forward_log_det_jacobian(u, event_ndims=1)
    return jnp.mean(base_log_prob_u_fn(u) - fldj - target_log_prob_fn(z))


def forward_kl_loss(params, make_bijector, samples_z, base_log_prob_u_fn=None):
    """Maximum-likelihood (forward-KL) objective on given target samples z:

        loss = -mean( base.log_prob(T^{-1}(z)) + ildj(z) )   (LOWER is better)

    i.e. the negative mean log-density the flow assigns to ``samples_z``.
    """
    if base_log_prob_u_fn is None:
        base_log_prob_u_fn = _std_normal_logprob
    bij = make_bijector(params)
    u = bij.inverse(samples_z)
    ildj = bij.inverse_log_det_jacobian(samples_z, event_ndims=1)
    return -jnp.mean(base_log_prob_u_fn(u) + ildj)
