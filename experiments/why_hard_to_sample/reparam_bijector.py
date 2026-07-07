"""Pure-jnp monotone cubic (PCHIP / Fritsch-Carlson) scalar bijector for the
T25/T26 Rs reparameterization (pre-registered 2026-07-03, docs/logs/
why-hard-to-sample.md, section "T25 ... + T26 ...").

WHY THIS FILE EXISTS
--------------------
T25 derives a PURE coordinate change on the z_Rs column only (Route A or B). The
new sampler coordinate is ``u``; the model still consumes the baseline
unconstrained coordinate ``z``. This module is the scalar leaf

    w : u -> z              (forward maps the SAMPLER coord u to the MODEL coord z)

that gets composed BEFORE the system's existing unconstraining+pack bijector
chain (see systems/carousel_min_newA/system.py). Composing w before the chain and
adding ``log|w'(u)|`` to the forward-log-det-jacobian leaves the theta-space
posterior byte-identical (family preservation, T25 HARD gate) while changing the
geometry the sampler sees.

The hot path the harness exercises is (traced from source, cited):
  * ProbModel.log_like  : ``x = self.bij.forward(list(z.T))``      (scene_prob_model.py:285)
  * ProbModel.log_prior : ``self.bij.forward(zc)`` +
                          ``self.bij.forward_log_det_jacobian(zc)`` (scene_prob_model.py:310-311)
  * MCLMC_JIT calls only ``prob_model.log_prob`` (mclmc.py:44-45); it never
    touches ``bij``. ``bij.inverse`` is used ONLY in the MAP stage
    (inference.py:132), which T26 does not run.
So the sampling hot path needs ``forward`` + ``forward_log_det_jacobian`` only,
both pure-jnp and differentiable here. ``inverse`` (60-iter bisection) is provided
for OFFLINE init mapping (z_init -> u_init) and the family round-trip gate; it is
not on the gradient path.

We deliberately do NOT depend on tfp spline bijectors (RationalQuadraticSpline /
etc.) -- their API/version risk in the shifter image is exactly the kind of
hidden coupling the pre-registration warns against. Everything here is numpy
(fitting) + jnp (evaluation), C1 by construction (cubic Hermite with monotone
Fritsch-Carlson slopes), analytic derivative for the fldj term.

AFFINE NORMALIZATION (pre-reg requirement)
------------------------------------------
The transform is normalized so that ``dz/du = 1`` at ``u = z_init`` and
``w(z_init) = z_init`` (i.e. the new coordinate matches the baseline z_Rs scale
near the typical-set init point). This keeps qz'=MVNDiag(., 1e-3) and the MCLMC
mass-matrix adaptation init comparable to the baseline arm. The normalization is
BAKED INTO the fitted knots by the derivation (t25_transforms.py); this class
verifies it (``dz/du`` at the z_init knot is asserted ~1 at fit time) and records
z_init in the artifact.

Deterministic: no randomness anywhere. Fitting raises loudly if the knots are not
strictly monotone (never silently "fixes" a bad profile).
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np


# ===========================================================================
# numpy-side fitting (PCHIP / Fritsch-Carlson monotone Hermite slopes)
# ===========================================================================

def pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Monotone C1 Hermite slopes at the knots via the Fritsch-Carlson rule
    (the same construction scipy.interpolate.PchipInterpolator uses). For a
    monotone-increasing (x, y) sample this returns non-negative slopes so the
    cubic Hermite interpolant is monotone with no overshoot. Pure numpy."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = x.shape[0]
    if n < 2:
        raise ValueError("pchip_slopes needs at least 2 knots")
    h = np.diff(x)
    if np.any(h <= 0):
        raise ValueError("pchip_slopes requires STRICTLY increasing x knots")
    delta = np.diff(y) / h                      # secant slopes, length n-1
    m = np.zeros(n, dtype=np.float64)

    # interior knots: weighted harmonic mean where secants share sign, else 0
    for k in range(1, n - 1):
        d0, d1 = delta[k - 1], delta[k]
        if d0 == 0.0 or d1 == 0.0 or (np.sign(d0) != np.sign(d1)):
            m[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            m[k] = (w1 + w2) / (w1 / d0 + w2 / d1)

    # endpoints: non-centered 3-point rule, clamped to preserve shape/sign
    def _end(d_end, d_next, h_end, h_next):
        d = ((2.0 * h_end + h_next) * d_end - h_end * d_next) / (h_end + h_next)
        if np.sign(d) != np.sign(d_end):
            d = 0.0
        elif (np.sign(d_end) != np.sign(d_next)) and (abs(d) > abs(3.0 * d_end)):
            d = 3.0 * d_end
        return d

    m[0] = _end(delta[0], delta[1], h[0], h[1]) if n > 2 else delta[0]
    m[-1] = _end(delta[-1], delta[-2], h[-1], h[-2]) if n > 2 else delta[-1]
    return m


def fit_monotone_cubic(x: np.ndarray, y: np.ndarray, *,
                       require_increasing: bool = True):
    """Validate + fit. Returns (x, y, slopes) as float64 numpy arrays.

    Raises unless x is strictly increasing and y is (strictly) monotone
    increasing -- a non-monotone coordinate map is not invertible and must not be
    silently accepted (the pre-registration's PCHIP-monotone hard requirement)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.shape != y.shape:
        raise ValueError(f"x,y shape mismatch {x.shape} vs {y.shape}")
    if x.shape[0] < 4:
        raise ValueError("need >=4 knots for a meaningful monotone cubic fit")
    if np.any(np.diff(x) <= 0):
        raise ValueError("knot x must be STRICTLY increasing")
    dy = np.diff(y)
    if require_increasing and np.any(dy <= 0):
        bad = int(np.sum(dy <= 0))
        raise ValueError(
            f"knot y is NOT strictly increasing ({bad} non-positive steps); the "
            "coordinate map would be non-invertible. Refuse to fit (T25 monotone "
            "gate). Check the profile derivation.")
    m = pchip_slopes(x, y)
    if np.any(m < 0) or not np.all(np.isfinite(m)):
        raise ValueError(
            "fitted Hermite slopes are negative or non-finite -> non-monotone "
            "interpolant. Refuse to fit.")
    return x, y, m


# ===========================================================================
# jnp-side bijector (forward + fldj differentiable; inverse via bisection)
# ===========================================================================

class MonotoneCubicBijector:
    """Scalar monotone C1 cubic Hermite map ``forward: u -> z``.

    Knots (x=u, y=z, slopes m) are precomputed and fixed. ``forward`` and
    ``forward_log_det_jacobian`` are pure jnp and differentiable (used on the MCLMC
    hot path). Outside the knot range the map is extended LINEARLY with the end
    slope (keeps C1 + monotone). ``inverse`` is a fixed 60-iteration bisection
    (offline: init mapping + family round-trip gate).
    """

    def __init__(self, x_knots, y_knots, slopes, *, z_init=None, meta=None):
        # numpy knots are the source of truth; jnp copies are built lazily so
        # this class is importable/usable under a jax-free login python (the
        # numpy fitting + bisection inverse + fldj-FD checks need no jax).
        self._x_np = np.asarray(x_knots, dtype=np.float64)
        self._y_np = np.asarray(y_knots, dtype=np.float64)
        self._m_np = np.asarray(slopes, dtype=np.float64)
        self.n = int(self._x_np.shape[0])
        self.z_init = None if z_init is None else float(z_init)
        self.meta = dict(meta or {})
        self._x = self._y = self._m = None  # jnp copies (lazy)

    def _ensure_jnp(self):
        if self._x is None:
            import jax
            import jax.numpy as jnp
            # ensure_compile_time_eval: if the first call lands INSIDE a jit
            # trace (e.g. blackjax's kernel), plain jnp.asarray would create
            # constants owned by THAT trace; caching them leaks tracers into
            # later traces (UnexpectedTracerError, hit 2026-07-03 in T26).
            # This context always materializes concrete device arrays.
            with jax.ensure_compile_time_eval():
                self._x = jnp.asarray(self._x_np)
                self._y = jnp.asarray(self._y_np)
                self._m = jnp.asarray(self._m_np)

    @property
    def x(self):
        self._ensure_jnp(); return self._x

    @property
    def y(self):
        self._ensure_jnp(); return self._y

    @property
    def m(self):
        self._ensure_jnp(); return self._m

    # -- construction helpers -------------------------------------------------
    @classmethod
    def fit(cls, u_knots, z_knots, *, z_init=None, meta=None):
        x, y, m = fit_monotone_cubic(u_knots, z_knots)
        return cls(x, y, m, z_init=z_init, meta=meta)

    @classmethod
    def from_npz(cls, path):
        d = np.load(path, allow_pickle=True)
        meta = {}
        if "meta_json" in d.files:
            meta = json.loads(str(d["meta_json"]))
        z_init = float(d["z_init"]) if "z_init" in d.files else None
        return cls(d["u_knots"], d["z_knots"], d["slopes"],
                   z_init=z_init, meta=meta)

    def to_npz(self, path, *, extra=None):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        payload = dict(
            u_knots=self._x_np, z_knots=self._y_np, slopes=self._m_np,
            meta_json=json.dumps(self.meta),
        )
        if self.z_init is not None:
            payload["z_init"] = np.float64(self.z_init)
        if extra:
            payload.update(extra)
        np.savez(path, **payload)
        return path

    # -- forward (u -> z), differentiable -------------------------------------
    def _cell(self, u):
        import jax.numpy as jnp
        idx = jnp.clip(jnp.searchsorted(self.x, u, side="right") - 1, 0, self.n - 2)
        return idx

    def forward(self, u):
        import jax.numpy as jnp
        u = jnp.asarray(u)
        i = self._cell(u)
        x0 = self.x[i]; x1 = self.x[i + 1]
        y0 = self.y[i]; y1 = self.y[i + 1]
        m0 = self.m[i]; m1 = self.m[i + 1]
        h = x1 - x0
        t = (u - x0) / h
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        y_in = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
        # linear extrapolation outside the knot span (end slope; C1 + monotone)
        y_lo = self.y[0] + self.m[0] * (u - self.x[0])
        y_hi = self.y[-1] + self.m[-1] * (u - self.x[-1])
        y = jnp.where(u < self.x[0], y_lo, jnp.where(u > self.x[-1], y_hi, y_in))
        return y

    def derivative(self, u):
        """dz/du (strictly positive for a monotone map). Analytic Hermite deriv."""
        import jax.numpy as jnp
        u = jnp.asarray(u)
        i = self._cell(u)
        x0 = self.x[i]; x1 = self.x[i + 1]
        y0 = self.y[i]; y1 = self.y[i + 1]
        m0 = self.m[i]; m1 = self.m[i + 1]
        h = x1 - x0
        t = (u - x0) / h
        t2 = t * t
        dh00 = 6 * t2 - 6 * t
        dh10 = 3 * t2 - 4 * t + 1
        dh01 = -6 * t2 + 6 * t
        dh11 = 3 * t2 - 2 * t
        din = (dh00 * y0 + dh10 * h * m0 + dh01 * y1 + dh11 * h * m1) / h
        d = jnp.where(u < self.x[0], self.m[0],
                      jnp.where(u > self.x[-1], self.m[-1], din))
        return d

    def forward_log_det_jacobian(self, u):
        """log|dz/du| -- the term added to the chain's forward-log-det-jacobian."""
        import jax.numpy as jnp
        return jnp.log(self.derivative(u))

    # -- inverse (z -> u): OFFLINE 60-iteration bisection ---------------------
    def inverse(self, z, iters=60):
        """Invert the monotone map by bisection. Vectorized over z. Uses numpy
        so it is safe to call at build time (init mapping, family round-trip
        gate); not needed on the gradient path."""
        z = np.asarray(z, dtype=np.float64)
        scalar = (z.ndim == 0)
        zf = np.atleast_1d(z)
        # bracket: knot span padded generously; expand if the target is outside.
        span = self._x_np[-1] - self._x_np[0]
        lo = np.full_like(zf, self._x_np[0] - max(span, 1.0) * 4.0)
        hi = np.full_like(zf, self._x_np[-1] + max(span, 1.0) * 4.0)
        flo = self._forward_np(lo) - zf
        fhi = self._forward_np(hi) - zf
        # widen if not bracketed (linear tails guarantee eventual bracket)
        for _ in range(40):
            need = (flo > 0) | (fhi < 0)
            if not np.any(need):
                break
            lo = np.where(flo > 0, lo - max(span, 1.0) * 4.0, lo)
            hi = np.where(fhi < 0, hi + max(span, 1.0) * 4.0, hi)
            flo = self._forward_np(lo) - zf
            fhi = self._forward_np(hi) - zf
        for _ in range(int(iters)):
            mid = 0.5 * (lo + hi)
            fm = self._forward_np(mid) - zf
            pos = fm > 0
            hi = np.where(pos, mid, hi)
            lo = np.where(pos, lo, mid)
        out = 0.5 * (lo + hi)
        return float(out[0]) if scalar else out

    def _forward_np(self, u):
        """numpy mirror of forward (for the bisection inverse only)."""
        u = np.asarray(u, dtype=np.float64)
        x, y, m = self._x_np, self._y_np, self._m_np
        n = x.shape[0]
        i = np.clip(np.searchsorted(x, u, side="right") - 1, 0, n - 2)
        x0 = x[i]; x1 = x[i + 1]; y0 = y[i]; y1 = y[i + 1]; m0 = m[i]; m1 = m[i + 1]
        h = x1 - x0
        t = (u - x0) / h
        t2 = t * t; t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        y_in = h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1
        y_lo = y[0] + m[0] * (u - x[0])
        y_hi = y[-1] + m[-1] * (u - x[-1])
        return np.where(u < x[0], y_lo, np.where(u > x[-1], y_hi, y_in))

    def derivative_np(self, u):
        u = np.asarray(u, dtype=np.float64)
        x, y, m = self._x_np, self._y_np, self._m_np
        n = x.shape[0]
        i = np.clip(np.searchsorted(x, u, side="right") - 1, 0, n - 2)
        x0 = x[i]; x1 = x[i + 1]; y0 = y[i]; y1 = y[i + 1]; m0 = m[i]; m1 = m[i + 1]
        h = x1 - x0
        t = (u - x0) / h
        t2 = t * t
        dh00 = 6 * t2 - 6 * t
        dh10 = 3 * t2 - 4 * t + 1
        dh01 = -6 * t2 + 6 * t
        dh11 = 3 * t2 - 2 * t
        din = (dh00 * y0 + dh10 * h * m0 + dh01 * y1 + dh11 * h * m1) / h
        return np.where(u < x[0], m[0], np.where(u > x[-1], m[-1], din))


class ColumnReparamBijector:
    """Compose a scalar monotone leaf ``w: u -> z`` onto ONE column of a
    list-of-columns -> theta-dict inner bijector (the system's existing
    ``model.bijector`` = Chain([event_space_bij, pack_sequence_as])).

    Mimics EXACTLY the three methods ProbModel calls on ``self.bij``:
      * ``forward(cols)``                     (scene_prob_model.py:285,310)
      * ``forward_log_det_jacobian(cols)``    (scene_prob_model.py:311)
      * ``inverse(theta_dict)``  (MAP-only; OFFLINE here — init map + family gate)

    Chain rule: theta = B(m(u)) where m replaces column ``col`` with w(u_col) and
    is identity elsewhere. Because m is block-diagonal (identity off ``col``),

        log|det J_theta/u| = log|det J_B(z)|  +  log|w'(u_col)|

    i.e. the inner chain's fldj evaluated at the substituted z-columns PLUS the
    scalar leaf's log-derivative. Pure composition; the theta-space posterior is
    byte-identical to the baseline (family preservation)."""

    def __init__(self, inner, leaf: "MonotoneCubicBijector", col: int):
        self.inner = inner
        self.leaf = leaf
        self.col = int(col)

    def _substitute(self, cols):
        """Return the column list with column ``col`` mapped u -> z = w(u)."""
        z_cols = list(cols)
        z_cols[self.col] = self.leaf.forward(cols[self.col])
        return z_cols

    def forward(self, cols):
        return self.inner.forward(self._substitute(cols))

    def forward_log_det_jacobian(self, cols, *args, **kwargs):
        z_cols = self._substitute(cols)
        inner_fldj = self.inner.forward_log_det_jacobian(z_cols, *args, **kwargs)
        leaf_fldj = self.leaf.forward_log_det_jacobian(cols[self.col])
        return inner_fldj + leaf_fldj

    def inverse(self, theta):
        """theta-dict -> u columns (OFFLINE). inner.inverse gives z columns; the
        leaf inverse (bisection) maps the reparam'd column z -> u."""
        z_cols = list(self.inner.inverse(theta))
        z_cols[self.col] = self.leaf.inverse(np.asarray(z_cols[self.col]))
        return z_cols


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_column(param_names, token="Rs") -> int:
    """Index of the sorted-key column whose scalar token is `token` (longest
    suffix wins so 'Rs' is not matched inside 'alpha_Rs'). Raises unless unique."""
    names = [str(n) for n in param_names]
    cand = [i for i, n in enumerate(names)
            if (n == token or n.endswith(token))
            and not (token == "Rs" and n.endswith("alpha_Rs"))]
    if len(cand) != 1:
        raise ValueError(f"column token {token!r} not unique in {names}: {cand}")
    return cand[0]


def attach_reparam_leaf(prob_model, param_names, artifact_path, *, token="Rs",
                        loud=True):
    """Wrap ``prob_model.bij`` with a :class:`ColumnReparamBijector` that composes
    the monotone leaf stored in ``artifact_path`` onto the ``token`` column
    (sorted-key order). Mutates ``prob_model.bij`` in place and returns
    ``(leaf, col, sha256)``. Prints WHICH artifact + its sha256 (pre-reg D:
    "loud print of which artifact + its sha256"). Generic: no gigalens import."""
    col = resolve_column(param_names, token)
    sha = sha256_file(artifact_path)
    leaf = MonotoneCubicBijector.from_npz(artifact_path)
    leaf._ensure_jnp()   # eager, OUTSIDE any trace (belt-and-braces vs tracer leak)
    inner = prob_model.bij
    prob_model.bij = ColumnReparamBijector(inner, leaf, col)
    if loud:
        print(f"[reparam] attached leaf on column {col} (token {token!r}, "
              f"param {list(param_names)[col]})")
        print(f"[reparam] artifact = {os.path.abspath(artifact_path)}")
        print(f"[reparam] artifact sha256 = {sha}")
        print(f"[reparam] leaf route = {leaf.meta.get('route')}, "
              f"n_knots = {leaf.n}, z_init = {leaf.z_init}")
    return leaf, col, sha


# ===========================================================================
# OFFLINE self-test (login python; numpy only, jax optional) -- exercises the
# PCHIP fitting, the bisection inverse, and the fldj-vs-finite-difference check
# on a synthetic profile. Run:  python reparam_bijector.py --selftest
# ===========================================================================

def _selftest() -> int:
    import numpy as _np
    print("[selftest] synthetic variance-stabilizing profile ...")
    # synthetic lambda(z): rises sharply toward low z (a funnel), calm at high z
    z = _np.linspace(-6.0, 12.0, 512)
    # smooth funnel: conditional precision rises sharply toward low z, calm high
    lam = 1.0 + 400.0 * _np.exp(-(z - (-4.0)) / 1.2)
    lam = _np.clip(lam, 1.0, 1e6)
    sqrt_lam = _np.sqrt(lam)
    # u(z) = int sqrt(lambda) dz ; normalize so du/dz=1 and u=z_init at z_init
    u_raw = _np.concatenate([[0.0], _np.cumsum(0.5 * (sqrt_lam[1:] + sqrt_lam[:-1]) * _np.diff(z))])
    z_init = 3.0
    g0 = _np.interp(z_init, z, sqrt_lam)
    u0 = _np.interp(z_init, z, u_raw)
    u = z_init + (u_raw - u0) / g0
    # w = u -> z : fit z as a function of u (u strictly increasing in z)
    w = MonotoneCubicBijector.fit(u, z, z_init=z_init, meta={"route": "selftest"})

    ok = True
    # (1) round-trip inverse
    z_test = _np.linspace(-5.0, 11.0, 97)
    u_of_z = w.inverse(z_test)
    z_rt = w._forward_np(u_of_z)
    rt_err = float(_np.max(_np.abs(z_rt - z_test)))
    print(f"[selftest] inverse round-trip max|z - w(w^-1(z))| = {rt_err:.3e}")
    ok = ok and rt_err < 1e-7

    # (2) du/dz = 1 at z_init  (=> dz/du = 1 at u=z_init)
    d_at_init = float(w.derivative_np(_np.array([z_init]))[0])
    print(f"[selftest] dz/du at u=z_init = {d_at_init:.6f} (want ~1.0)")
    ok = ok and abs(d_at_init - 1.0) < 5e-3

    # (3) fldj vs finite-difference of log|w'|
    uu = _np.linspace(u.min() + 0.5, u.max() - 0.5, 200)
    eps = 1e-6
    fd = (w._forward_np(uu + eps) - w._forward_np(uu - eps)) / (2 * eps)
    analytic = w.derivative_np(uu)
    dmax = float(_np.max(_np.abs(_np.log(analytic) - _np.log(fd))))
    print(f"[selftest] max|log w'_analytic - log w'_FD| = {dmax:.3e}")
    ok = ok and dmax < 1e-5

    # (4) monotonicity of forward
    zz = w._forward_np(_np.sort(uu))
    mono = bool(_np.all(_np.diff(zz) > 0))
    print(f"[selftest] forward strictly increasing on the grid: {mono}")
    ok = ok and mono

    # (5) refuse a non-monotone profile
    try:
        MonotoneCubicBijector.fit(_np.array([0., 1., 2., 3.]),
                                  _np.array([0., 1., 0.5, 2.]))
        print("[selftest] ERROR: non-monotone fit did NOT raise")
        ok = False
    except ValueError:
        print("[selftest] non-monotone profile correctly rejected")

    print(f"[selftest] RESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        sys.exit(_selftest())
    print("reparam_bijector: pass --selftest to run the offline numeric checks.")
