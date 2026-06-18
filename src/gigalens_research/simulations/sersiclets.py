import functools

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax import jit
from jax.scipy.special import gammaln
from lenstronomy.LightModel.Profiles.shapelets import Shapelets as LenstronomyShapelets

import gigalens.profile


_PI_NEG_QUARTER = float(jnp.pi) ** -0.25


# def _phi_basis_1d(x, n_max):
#     """Compute the orthonormal Hermite (shapelet) basis ``phi_n(x)`` for
#     ``n = 0, ..., n_max`` using the numerically stable three-term recurrence
#     on the *normalized* functions:

#         phi_0(x) = pi^(-1/4) * exp(-x^2/2)
#         phi_1(x) = sqrt(2) * x * phi_0(x)
#         phi_n(x) = sqrt(2/n) * x * phi_{n-1}(x) - sqrt((n-1)/n) * phi_{n-2}(x)

#     Unlike the textbook approach (compute H_n(x) via its recurrence, then
#     multiply by the tiny prefactor 1/sqrt(2^n * n! * sqrt(pi)) and the
#     Gaussian envelope exp(-x^2/2)), this recurrence keeps every intermediate
#     value bounded uniformly in n and x (|phi_n(x)| <= O(1)), so it does not
#     overflow in float32 for any n_max or x, and its VJP is finite everywhere.
#     """
#     phi = jnp.empty((n_max + 1, *x.shape), dtype=x.dtype)
#     phi = phi.at[0].set(_PI_NEG_QUARTER * jnp.exp(-x ** 2 / 2))
#     if n_max >= 1:
#         phi = phi.at[1].set(jnp.sqrt(2.0) * x * phi[0])
#     for n in range(2, n_max + 1):
#         phi = phi.at[n].set(
#             jnp.sqrt(2.0 / n) * x * phi[n - 1]
#             - jnp.sqrt((n - 1) / n) * phi[n - 2]
#         )
#     return phi

# def _radial_laguerre_basis(r, n_max, n_sersic):
#     alpha = 2 * n_sersic - 1
#     #* Using recurrence from wikipedia for now. Should change to normalized for stability
#     L = jnp.empty((n_max + 1, *r.shape), dtype=r.dtype)
#     L = L.at[0].set(jnp.ones_like(r))
#     L = L.at[1].set(-r + alpha + 1)
    
#     for n in range(1, n_max):
#         L = L.at[n+1].set(
#             ((2 * n + 1 + alpha - r) * L[n] - (n + alpha) * L[n - 1]) / (n+1)
#         )
#     return L


def _normalized_laguerre_functions(u, p_max, k):
    """Stable recurrence for the normalized associated-Laguerre functions

        h_p(u) = sqrt(p! / Gamma(p + k + 1)) * L_p^k(u) * exp(-u/2),  p = 0..p_max,

    for an arbitrary (real, > -1) Laguerre index ``k``. The envelope exp(-u/2)
    and the Gamma normalisation are folded into the seeds so every intermediate
    stays bounded; these satisfy ``int_0^inf h_p h_{p'} u^k du = delta_{p p'}``.
    """
    env = jnp.exp(-u / 2.0)
    h = jnp.empty((p_max + 1, *u.shape), dtype=u.dtype)
    h = h.at[0].set(jnp.exp(-0.5 * gammaln(k + 1.0)) * env)
    if p_max >= 1:
        h = h.at[1].set(jnp.exp(-0.5 * gammaln(k + 2.0)) * (k + 1.0 - u) * env)
    for p in range(1, p_max):
        inv = 1.0 / jnp.sqrt((p + 1.0) * (p + k + 1.0))
        sub = jnp.sqrt(p * (p + k) / ((p + 1.0) * (p + k + 1.0)))
        h = h.at[p + 1].set((2.0 * p + 1.0 + k - u) * inv * h[p] - sub * h[p - 1])
    return h


def _radial_laguerre_basis_normalized(u, n_max, n_sersic):
    """Orthonormal associated-Laguerre radial basis, computed with a
    numerically stable recurrence on the *normalized* functions.

    Returns ``g_l(u)`` for ``l = 0, ..., n_max``, where

        g_l(u) = sqrt(l! / Gamma(l + k + 1)) * L_l^k(u) * exp(-u/2),
        k = 2*n_sersic - 1,

    i.e. the bare generalized Laguerre polynomial L_l^k carrying both the
    sersiclet envelope exp(-u/2) (Eq. B7) and the orthonormalisation constant
    sqrt(l! / Gamma(l + k + 1)) (Eq. B6/B8 of Andrae, Melchior & Jahnke 2011).
    These satisfy the plain orthonormality

        \\int_0^inf g_l(u) g_{l'}(u) u^k du = delta_{l l'} .

    To recover the full sersiclet radial part R_l of Eq. (B7), multiply by the
    l-independent factor sqrt(b^{2 n_S} / beta^{2 n_S}); the argument expected
    here is u = b (r / beta)^(1 / n_sersic).

    Like the Hermite shapelet recurrence, the envelope exp(-u/2) and the Gamma
    normalisation are folded directly into the seeds, so every intermediate
    value stays bounded (|g_l| <= O(1)) and the recurrence does not overflow in
    float32 for any n_max or u. The bare polynomial is recovered as
    L_l^k(u) = g_l(u) * exp(u/2) * sqrt(Gamma(l + k + 1) / l!).
    """
    return _normalized_laguerre_functions(u, n_max, 2.0 * n_sersic - 1.0)


# class EllipticalSersiclets(gigalens.profile.LightProfile):
#     """Elliptical *polar* sersiclet light profile (Andrae, Melchior & Jahnke 2011).

#     Sersiclets are intrinsically polar (Sec. 2.2 of the paper rules out a
#     Cartesian construction): each basis function is a product of a radial part
#     R_l(r) -- the orthonormal associated-Laguerre function -- and an azimuthal
#     Fourier part. We use the real cos/sin form so the model is real-valued for
#     fitting real images:

#         m = 0 :  B_{l,0}(r, phi) = R_l(r)
#         m > 0 :  B_{l,m}^c(r, phi) = R_l(r) cos(m phi)
#                  B_{l,m}^s(r, phi) = R_l(r) sin(m phi)

#     with the radial order l and azimuthal order m running over 2 l + |m| <= n_max
#     (the same number of modes, (n_max+1)(n_max+2)/2, as the Cartesian shapelet
#     basis of equal n_max). At n_sersic = 0.5 this reduces to polar shapelets.

#     The radial argument is u = 2 b_n (R/beta)^(1/n_sersic), where R is the
#     elliptical radius and b_n is the standard Sersic coefficient (same
#     approximation as gigalens' Sersic profile). The factor of 2 absorbs the 1/2
#     in Eq. (B7)'s envelope -- the paper notes b may absorb such factors -- so
#     that the ground state B_{0,0} ~ exp(-b_n (R/beta)^(1/n_sersic)) is exactly a
#     Sersic profile with scale radius beta (its "ground state", per the paper).
#     """

#     _name = "SERSICLETS"
#     _params = ["beta", "n_sersic", "e1", "e2", "center_x", "center_y"]

#     def __init__(self, n_max, use_lstsq=False):
#         super(EllipticalSersiclets, self).__init__(use_lstsq=use_lstsq)
#         self.n_max = n_max
#         # Enumerate polar modes (l, m) with 2 l + |m| <= n_max. For each m > 0 we
#         # emit a cosine and a sine mode; m = 0 is a single (real) mode.
#         mode_l, mode_m, mode_is_sin = [], [], []
#         for l in range(n_max // 2 + 1):
#             for m in range(0, n_max - 2 * l + 1):
#                 mode_l.append(l); mode_m.append(m); mode_is_sin.append(0)  # cos / m=0
#                 if m > 0:
#                     mode_l.append(l); mode_m.append(m); mode_is_sin.append(1)  # sin
#         self.mode_l = jnp.array(mode_l)
#         self.mode_m = jnp.array(mode_m)
#         self.mode_is_sin = jnp.array(mode_is_sin, dtype=bool)
#         self.n_layers = len(mode_l)
#         self.depth = self.n_layers
#         decimal_places = len(str(self.n_layers))
#         self._amp_names = [f"amp{str(i).zfill(decimal_places)}" for i in range(self.n_layers)]
#         for name in self._amp_names:
#             self._params.append(name)

#     @functools.partial(jit, static_argnums=(0,))
#     def _polar(self, x, y, cx, cy, e1, e2):
#         """Elliptical radius R and position angle phi in the circularised frame.

#         Same shear/axis-ratio transform as the Sersic profile's ``distance``;
#         we additionally return the azimuthal angle for the Fourier part.
#         """
#         phi_e = jnp.arctan2(e2, e1) / 2
#         c = jnp.sqrt(e1 ** 2 + e2 ** 2)
#         q = (1 - c) / (1 + c)
#         dx, dy = x - cx, y - cy
#         cos_p, sin_p = jnp.cos(phi_e), jnp.sin(phi_e)
#         xt1 = (cos_p * dx + sin_p * dy) * jnp.sqrt(q)
#         xt2 = (-sin_p * dx + cos_p * dy) / jnp.sqrt(q)
#         R = jnp.sqrt(xt1 ** 2 + xt2 ** 2)
#         phi = jnp.arctan2(xt2, xt1)
#         return R, phi

#     @functools.partial(jit, static_argnums=(0,))
#     def light(self, x, y, n_sersic, e1, e2, center_x, center_y, beta, **amp):
#         R, phi = self._polar(x, y, center_x, center_y, e1, e2)
#         bn = jnp.exp(0.6950 + jnp.log(n_sersic) - 0.1789 / n_sersic)
#         ratio = R / beta

#         #* NaN-proofing (mirrors the Sersic profile): for r far outside the
#         #* profile, ratio^(1/n_sersic) overflows and the brightness is zero
#         #* anyway, but a raw 0*NaN would poison the gradient. Zero the ratio
#         #* before the power, then zero the basis where it is unsafe.
#         is_safe = (1.0 / n_sersic) * jnp.log10(ratio + 1e-10) < 30
#         safe_ratio = jnp.where(is_safe, ratio, 0.0)

#         # u = 2 b_n (R/beta)^(1/n_sersic): factor 2 makes B_{0,0} ~ exp(-b_n ...),
#         # i.e. exactly a Sersic profile with scale radius beta (see class docstring).
#         u = 2.0 * bn * safe_ratio ** (1.0 / n_sersic)
#         g = _radial_laguerre_basis_normalized(u, self.n_max, n_sersic)  # (n_max+1, *R.shape)

#         radial = g[self.mode_l]  # (P, *R.shape)
#         m_col = self.mode_m.reshape((-1,) + (1,) * R.ndim)
#         sin_col = self.mode_is_sin.reshape((-1,) + (1,) * R.ndim)
#         angle = m_col * phi  # (P, *R.shape)
#         azimuthal = jnp.where(sin_col, jnp.sin(angle), jnp.cos(angle))
#         basis = radial * azimuthal
#         basis = jnp.where(is_safe[None, ...], basis, 0.0)

#         if self.use_lstsq:
#             return basis
#         else:
#             amps = jnp.stack([amp[name] for name in self._amp_names], axis=0)
#             return jnp.einsum('ij,i...j->...j', amps, basis)


# Envelope-based safety cutoff: beyond u ~ U_MAX_SAFE the envelope exp(-u/2) is
# negligible, so we zero the basis there. This (rather than the looser Sersic-
# exponent cut) also keeps the r^|m| factor of the continuous basis from
# overflowing, since it bounds u and hence r within the evaluated region.
_U_MAX_SAFE = 70.0


class EllipticalSersiclets(gigalens.profile.LightProfile):
    """Continuity-corrected elliptical polar sersiclets (QHO-consistent variant).

    The paper's basis (``EllipticalSersiclets``) uses a single Laguerre index
    k = 2 n_sersic - 1 for every azimuthal order m, which makes the m > 0 modes
    discontinuous at r = 0 (their angular dependence does not vanish at the
    centre). Restoring the r^|m| prefactor and letting the Laguerre index depend
    on m fixes this while preserving orthogonality (only same-m modes interact):

        R_{p,m}(r) = N_{p,m} r^|m| L_p^{k_m}(u) exp(-u/2),
        k_m = 2 n_sersic (|m| + 1) - 1,   u = 2 b_n (r/beta)^(1/n_sersic),

    with radial order p, azimuthal order m, modes truncated at 2 p + |m| <= n_max
    (the 2D harmonic-oscillator total quantum number). At n_sersic = 0.5 this
    becomes k_m = |m|, u = (r/beta_s)^2, i.e. exactly the 2D QHO / polar-shapelet
    basis r^|m| L_p^|m|(r^2) exp(-r^2/2) -- the smooth limit the paper's version
    cannot reach. The m = 0, p = 0 ground state is still a Sersic profile.

    Real cos/sin convention: m = 0 is one real mode; m > 0 contributes a cosine
    and a sine mode. Mode count is (n_max+1)(n_max+2)/2, as for the paper's basis.
    """

    _name = "SERSICLETS"
    _params = ["beta", "n_sersic", "e1", "e2", "center_x", "center_y"]

    def __init__(self, n_max, use_lstsq=False):
        super(EllipticalSersiclets, self).__init__(use_lstsq=use_lstsq)
        self.n_max = n_max
        # Modes (p, m) with 2 p + |m| <= n_max; m = 0 real, m > 0 -> cos & sin.
        modes = []
        for p in range(n_max // 2 + 1):
            for m in range(0, n_max - 2 * p + 1):
                modes.append((p, m, False))
                if m > 0:
                    modes.append((p, m, True))
        self.modes = modes
        self.n_layers = len(modes)
        self.depth = self.n_layers
        decimal_places = len(str(self.n_layers))
        self._amp_names = [f"amp{str(i).zfill(decimal_places)}" for i in range(self.n_layers)]
        for name in self._amp_names:
            self._params.append(name)

    @functools.partial(jit, static_argnums=(0,))
    def _polar(self, x, y, cx, cy, e1, e2):
        phi_e = jnp.arctan2(e2, e1) / 2
        c = jnp.sqrt(e1 ** 2 + e2 ** 2)
        q = (1 - c) / (1 + c)
        dx, dy = x - cx, y - cy
        cos_p, sin_p = jnp.cos(phi_e), jnp.sin(phi_e)
        xt1 = (cos_p * dx + sin_p * dy) * jnp.sqrt(q)
        xt2 = (-sin_p * dx + cos_p * dy) / jnp.sqrt(q)
        R = jnp.sqrt(xt1 ** 2 + xt2 ** 2)
        phi = jnp.arctan2(xt2, xt1)
        return R, phi

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y, n_sersic, e1, e2, center_x, center_y, beta, **amp):
        R, phi = self._polar(x, y, center_x, center_y, e1, e2)
        bn = jnp.exp(0.6950 + jnp.log(n_sersic) - 0.1789 / n_sersic)
        ratio = R / beta

        # Envelope-based NaN-proofing: zero (before any power) where the envelope
        # is negligible, i.e. u > _U_MAX_SAFE  <=>  ratio > (U_MAX/(2 b_n))^n_sersic.
        ratio_max = (_U_MAX_SAFE / (2.0 * bn)) ** n_sersic
        is_safe = ratio < ratio_max
        safe_ratio = jnp.where(is_safe, ratio, 0.0)
        u = 2.0 * bn * safe_ratio ** (1.0 / n_sersic)

        # Per-m radial functions h_p with k_m = 2 n_sersic (|m| + 1) - 1.
        h_by_m = {}
        for m in range(self.n_max + 1):
            p_max_m = (self.n_max - m) // 2
            if p_max_m < 0:
                break
            k_m = 2.0 * n_sersic * (m + 1.0) - 1.0
            h_by_m[m] = _normalized_laguerre_functions(u, p_max_m, k_m)

        basis = []
        for (p, m, is_sin) in self.modes:
            # r^|m| ~ safe_ratio^m (the beta^|m| constant is absorbed into the amp);
            # safe_ratio is zeroed where unsafe, so the power never overflows.
            radial = safe_ratio ** m * h_by_m[m][p]
            azimuthal = jnp.sin(m * phi) if is_sin else jnp.cos(m * phi)
            basis.append(jnp.where(is_safe, radial * azimuthal, 0.0))
        basis = jnp.stack(basis, axis=0)  # (P, *R.shape)

        if self.use_lstsq:
            return basis
        else:
            amps = jnp.stack([amp[name] for name in self._amp_names], axis=0)
            return jnp.einsum('ij,i...j->...j', amps, basis)