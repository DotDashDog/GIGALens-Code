#!/usr/bin/env python
"""Replace complex source light with one plain Sersic per clump, preserving size and SNR.

Motivation
----------
The translated cluster model (``translate_old_params.py``) carries nine source planes
whose light is a mixture of ``SERSIC_ELLIPSE`` and 45-coefficient ``SHAPELETS``. For a
*simple* mock -- one where every source is a smooth blob you can reason about, and where
the free-parameter count is small enough to actually sample -- you want each source
replaced by a single ``SersicEllipse`` at fixed ``n_sersic`` that is still recognisably
the same object: same place, same size, same detectability.

This module does that swap. It is deliberately a **measurement**, not a reparametrisation:
nothing here reads ``R_sersic`` or ``beta`` off the input profiles and rescales it. The
input light is rendered and its size, shape and brightness are measured the way you would
measure them on an image. That is the only approach that works when the light is a sum of
profiles with different centres, when the shapelet coefficients dominate the Sersic they
were fit alongside, or when a component's amplitude is negative -- all three occur here.

What is preserved, and how
--------------------------
Per clump (see *Grouping*), four things transfer:

======================  ========================================================
``center_x/center_y``   ``|I|``-weighted first moment of the rendered light.
``e1``/``e2``           ``|I|``-weighted second moments, converted through
                        gigalens' own ``q = (1-c)/(1+c)`` convention, so the
                        output ellipticity reproduces the input's isophote
                        shape exactly for a pure ellipse (verified).
``R_sersic``            The circularised half-light radius of ``|I|`` inside
                        ``aperture``. For the output Sersic, ``R_sersic`` **is**
                        the half-light radius, so this is a like-for-like
                        transfer rather than a fitted proxy.
``Ie``                  Solved so the swapped source reproduces a chosen
                        brightness target -- by default its image-plane
                        signal-to-noise ratio (see *Amplitude*).
======================  ========================================================

``n_sersic`` is whatever you ask for (default 2.0) and is the one property deliberately
*not* preserved: fixing it is the point of the exercise.

Why ``|I|`` and not ``I``
-------------------------
Three Sersic amplitudes in this model are negative, and two composite clumps are ~80%
internally cancelling (``net/|F|`` of 0.23 and 0.20). A *signed* flux-weighted centroid
of such a source is meaningless -- the weights sum to nearly zero and the moments blow
up. Positional statistics therefore use ``|I|``. The *amplitude* is a separate question
and is handled on the signed image, where cancellation is physical and must be respected.

Grouping
--------
``group="clump"`` (default) emits one Sersic per set of components sharing a centre. That
is the structure the old API actually had: a ``SersicShapelets`` was a Sersic plus a
Shapelets locked to the same ``center_x``/``center_y``, and a plane holding two of them is
a two-clump source. Collapsing such a plane to a single Sersic would place it at the
flux-weighted midpoint of two objects several arcsec apart -- somewhere neither of them is.
``group="plane"`` does exactly that, on request.

Amplitude
---------
``match="snr"`` (default, needs ``datasets``) renders each clump on its own through the
real forward model -- that band's PSF, mask and error map -- and matches

    SNR = sqrt( sum_{unmasked} (model / sigma)^2 )

the matched-filter significance of the source in that band. This is what "same SNR"
should mean: it is computed on the *lensed, convolved, signed* image, so magnification,
seeing and internal cancellation are all accounted for without being modelled separately.
The forward model is linear in ``Ie``, so ``SNR`` is exactly proportional to ``|Ie|`` and
the match is a single division -- not a fit, and not iterative.

``match="flux"`` needs no data and matches the source-plane integrated flux instead
(``flux_kind="abs"`` or ``"net"``) through the analytic Sersic integral. Cheaper, and
wrong in the ways described above whenever magnification varies across the source.

The aperture is a real choice
-----------------------------
Sersic wings are unbounded and their weight grows steeply with ``n``. Five of this model's
components sit at ``n = 3.8`` to ``9.1``, and for the clumps they dominate the measured
half-light radius grows with the aperture you measure in -- a factor 1.7 to 2.6 between 1"
and 30" (measured, four of thirteen clumps). There is no aperture-free answer;
``aperture`` (default 10 arcsec, which holds >=96% of ``|F|`` for every clump here)
is therefore explicit, and :meth:`SwapReport.summary` prints the half-light radius at
``aperture`` and at ``aperture/3`` side by side so an aperture-sensitive source is visible
rather than implied.

Sign
----
``sign="abs"`` (default) gives every output Sersic a positive ``Ie``. A negative-surface-
brightness source is unphysical, and gigalens warns on it; the three that appear here are
artefacts of an unconstrained lstsq solve, not measurements. ``sign="signed"`` keeps the
sign of the matched quantity if you want the swap to be sign-faithful instead.

Usage
-----
::

    from translate_old_params import build
    from sersic_swap import sersic_swap

    model, spec = build("MAP_best_31JulNFW_fixedcosmo_fixedLowZ.json")
    params = model.to_params({})

    # datasets built as usual, one per source plane, in cutout_extensions order
    result = sersic_swap(model, params, n_sersic=2.0, datasets=datasets)
    result.summary()

    simple_model, simple_params = result.model, result.params

The returned model keeps the mass plane, the cosmology and every plane redshift and name
untouched -- only ``plane.light`` is rebuilt. Because ``sees`` matches by object identity,
**datasets must be rebuilt against the new components**::

    new_datasets = [ds(ext, pl.light) for ext, pl in
                    zip(cutout_extensions(spec), [p for p in simple_model.planes if p.has_light])]
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Polynomial gigalens uses for b_n. Duplicated rather than imported so the measurement
# and the profile cannot silently disagree about which approximation is in force: if
# gigalens changes its polynomial, _check_bn() below fails loudly instead of leaving a
# few-percent flux error to be discovered downstream.
_BN_COEFFS = (9.76405972e+01, -1.06311838e+01, 3.50444477e-01, -1.99301509e-03,
              4.88184193e+01, 2.82927599e+00, 4.03956637e-01)

#: Radial samples per clump (log-spaced) and angular samples per shell.
N_R, N_T = 768, 256
#: Innermost sampled radius, as a fraction of the SOURCE's own smallest characteristic
#: scale (``R_sersic`` or ``beta``) -- deliberately not a fraction of the aperture. A
#: high-n Sersic is cuspy enough that the two differ by orders of magnitude: measuring an
#: n=9.08 profile over an 889" span with the cutoff at 1e-5 of the *aperture* puts it at
#: 0.011 R_e, which truncates 2.6% of the flux and biases r_half 7% HIGH (the missing
#: flux is all central, so "half of what is left" sits further out). Tied to the source
#: instead, the same case is exact to 2e-4. Caught by selftest(), not in review.
R_LO_FRAC = 1e-6


def bn(n_sersic: float) -> float:
    """gigalens' ``b_n`` (sersic.py ``_BN_COEFFS``), valid for ``n > 0.1``."""
    c = _BN_COEFFS
    return ((c[0]*n_sersic**3 + c[1]*n_sersic**2 + c[2]*n_sersic + c[3])
            / (c[4]*n_sersic**2 + c[5]*n_sersic + c[6]))


def sersic_total_flux(Ie: float, R_sersic: float, n_sersic: float) -> float:
    """Analytic ``\\int Ie * exp(-bn((R/Rs)^(1/n) - 1)) dA`` over the whole plane.

    Valid for the elliptical profile as well as the circular one: gigalens' elliptical
    coordinate map scales one axis by ``sqrt(q)`` and the other by ``1/sqrt(q)``, so its
    Jacobian is 1 and the integral is unchanged. Evaluated in logs -- the individual
    factors ``exp(bn)`` and ``bn^(2n)`` both overflow float64 by ``n ~ 9``, while their
    ratio stays O(1).
    """
    b = bn(n_sersic)
    log_shape = b + math.lgamma(2*n_sersic) - 2*n_sersic*math.log(b)
    return Ie * R_sersic**2 * 2*math.pi*n_sersic * math.exp(log_shape)


# --------------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------------
@dataclass(frozen=True)
class Clump:
    """One measured clump of source light."""

    plane: int                      #: index into ``model.planes``
    plane_name: str
    members: Tuple[int, ...]        #: indices into ``model.planes[plane].light``
    member_names: Tuple[str, ...]
    center_x: float
    center_y: float
    e1: float
    e2: float
    q: float                        #: axis ratio, ``(1-|e|)/(1+|e|)``
    phi: float                      #: major-axis position angle, radians
    r_half: float                   #: circularised half-light radius of ``|I|``
    r_half_inner: float             #: same, measured inside ``aperture/3``
    flux_abs: float                 #: ``\\int |I| dA`` inside ``aperture``
    flux_net: float                 #: ``\\int I dA`` inside ``aperture``
    aperture: float
    inner: float                    #: fraction of ``|F|`` in the innermost shell
    outer: float                    #: fraction of ``|F|`` in the outermost shell

    @property
    def aperture_sensitive(self) -> bool:
        """True when the half-light radius depends materially on the aperture.

        A source whose ``r_half`` moves by >10% when the aperture is cut to a third has
        no aperture-free size; the reported one is a statement about ``aperture``, not
        about the object.
        """
        return abs(self.r_half_inner - self.r_half) > 0.10 * self.r_half


def _polar_grid(cx, cy, q, phi, r_lo, r_hi, n_r, n_t):
    """Log-polar sampling in the source's own elliptical frame.

    Points sit at gigalens' circularised elliptical radius ``R`` and at angle ``theta``
    in the *sheared* frame, so the area element is exactly ``R dR dtheta``. A uniform
    Cartesian grid cannot do this job: resolving an ``n >= 4`` core and reaching its
    wings differ by four orders of magnitude in radius, and a grid fine enough for the
    core over a box wide enough for the wings is ~10^8 points. (Measured: a 1201^2
    Cartesian grid returns ``r_half`` a factor 10 too small for an ``n=4`` Sersic.)
    """
    ln_r = np.linspace(math.log(r_lo), math.log(r_hi), n_r)
    theta = (np.arange(n_t) + 0.5) * (2*math.pi/n_t)
    R = np.exp(ln_r)[:, None]
    T = theta[None, :]
    u, v = R*np.cos(T)/math.sqrt(q), R*np.sin(T)*math.sqrt(q)
    cos_p, sin_p = math.cos(phi), math.sin(phi)
    X = cx + cos_p*u - sin_p*v
    Y = cy + sin_p*u + cos_p*v
    dA = R**2 * (ln_r[1] - ln_r[0]) * (2*math.pi/n_t)
    return X, Y, ln_r, np.broadcast_to(dA, X.shape)


def _evaluate(profiles, param_dicts, X, Y):
    import jax.numpy as jnp
    total = np.zeros(X.shape, dtype=float)
    jx, jy = jnp.asarray(X), jnp.asarray(Y)
    for profile, prm in zip(profiles, param_dicts):
        total = total + np.asarray(profile.light(jx, jy, **prm), dtype=float)
    return total


def _inner_radius(param_dicts, aperture: float) -> float:
    """Innermost sampled radius: ``R_LO_FRAC`` times the smallest scale actually present.

    ``R_sersic`` and ``beta`` are read here purely to size the quadrature grid -- they do
    not enter any measured quantity. A component type exposing neither falls back to the
    aperture, which is safe for a profile with no small-scale structure and wrong for one
    that has it, so add the scale rather than relying on the fallback.
    """
    scales = [float(d[k]) for d in param_dicts for k in ("R_sersic", "beta")
              if k in d and float(d[k]) > 0]
    return max(min(scales, default=aperture) * R_LO_FRAC, aperture * 1e-13)


def _moments(profiles, param_dicts, cx, cy, aperture, n_r, n_t, n_iter=8):
    """Iterate centroid/ellipticity to self-consistency, then read off the growth curve.

    The grid is rebuilt in the current best frame each pass, so once ``q``/``phi``
    converge the samples lie along the source's own isophotes.
    """
    r_lo = _inner_radius(param_dicts, aperture)
    q, phi = 1.0, 0.0
    for _ in range(n_iter):
        X, Y, _, dA = _polar_grid(cx, cy, q, phi, r_lo, aperture, n_r, n_t)
        w = np.abs(_evaluate(profiles, param_dicts, X, Y)) * dA
        W = w.sum()
        if not W > 0:
            raise ValueError(
                "clump renders identically zero over the aperture; nothing to measure")
        cx_new, cy_new = (w*X).sum()/W, (w*Y).sum()/W
        dx, dy = X - cx_new, Y - cy_new
        Q = np.array([[(w*dx*dx).sum(), (w*dx*dy).sum()],
                      [(w*dx*dy).sum(), (w*dy*dy).sum()]]) / W
        lam, vec = np.linalg.eigh(Q)                  # ascending: lam[1] is the major axis
        q_new = math.sqrt(max(lam[0], 0.0) / lam[1])
        phi_new = math.atan2(vec[1, 1], vec[0, 1])
        moved = math.hypot(cx_new-cx, cy_new-cy) + abs(q_new - q)
        cx, cy, q, phi = cx_new, cy_new, q_new, phi_new
        if moved < 1e-9:
            break
    return cx, cy, q, phi


def _growth(profiles, param_dicts, cx, cy, q, phi, aperture, n_r, n_t):
    """Return (r_half, flux_abs, flux_net, inner, outer) inside ``aperture``."""
    r_lo = _inner_radius(param_dicts, aperture)
    X, Y, ln_r, dA = _polar_grid(cx, cy, q, phi, r_lo, aperture, n_r, n_t)
    I = _evaluate(profiles, param_dicts, X, Y)
    w = np.abs(I) * dA
    shell = w.sum(axis=1)
    cum = np.cumsum(shell)
    # cum[k] holds ALL of shell k, so it is the flux inside that annulus' OUTER edge.
    # Interpolating against the shell centre instead biases r_half low by dln_r/2 --
    # 0.7% at n_r=768, systematic and in the same direction for every source.
    r_out = np.exp(ln_r + 0.5*(ln_r[1] - ln_r[0]))
    r_half = float(np.exp(np.interp(0.5*cum[-1], cum, np.log(r_out))))
    return (r_half, float(w.sum()), float((I*dA).sum()),
            float(shell[0]/cum[-1]), float(shell[-1]/cum[-1]))


def _clump_groups(plane, plane_params, group: str):
    """Partition a plane's light components into clumps, in first-appearance order."""
    if group == "plane":
        return [tuple(range(len(plane.light)))]
    if group != "clump":
        raise ValueError(f"group must be 'clump' or 'plane'; got {group!r}")
    order: List[Tuple[float, float]] = []
    buckets: Dict[Tuple[float, float], List[int]] = {}
    for j in range(len(plane.light)):
        prm = plane_params["light"][j]
        key = (round(float(prm["center_x"]), 6), round(float(prm["center_y"]), 6))
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(j)
    return [tuple(buckets[k]) for k in order]


def measure_clumps(model, params, *, aperture: float = 10.0, group: str = "clump",
                   n_r: int = N_R, n_t: int = N_T) -> List[Clump]:
    """Measure every source clump in ``model`` at ``params``. Renders only; no data used."""
    out: List[Clump] = []
    for i, plane in enumerate(model.planes):
        if not plane.has_light:
            continue
        pl_params = params["planes"][i]
        for members in _clump_groups(plane, pl_params, group):
            profiles = [plane.light[j].profile for j in members]
            prm = [{k: float(v) for k, v in pl_params["light"][j].items()}
                   for j in members]
            cx0 = float(np.mean([d["center_x"] for d in prm]))
            cy0 = float(np.mean([d["center_y"] for d in prm]))
            cx, cy, q, phi = _moments(profiles, prm, cx0, cy0, aperture, n_r, n_t)
            r_half, f_abs, f_net, inner, outer = _growth(
                profiles, prm, cx, cy, q, phi, aperture, n_r, n_t)
            r_in, _, _, _, _ = _growth(
                profiles, prm, cx, cy, q, phi, aperture/3.0, n_r, n_t)
            c = (1 - q) / (1 + q)
            out.append(Clump(
                plane=i, plane_name=plane.name or f"plane{i}", members=members,
                member_names=tuple(plane.light[j].name or f"light{j}" for j in members),
                center_x=cx, center_y=cy, e1=c*math.cos(2*phi), e2=c*math.sin(2*phi),
                q=q, phi=phi, r_half=r_half, r_half_inner=r_in,
                flux_abs=f_abs, flux_net=f_net, aperture=aperture,
                inner=inner, outer=outer))
    return out


# --------------------------------------------------------------------------------
# Image-plane SNR
# --------------------------------------------------------------------------------
def _dataset_for_plane(model, datasets) -> Dict[int, Any]:
    """Map plane index -> dataset, via ``sees`` object identity.

    Positional pairing is refused on purpose. Planes are ordered by redshift and the
    cutouts are named by source number; in this model the two orders agree for three of
    nine planes, and a mispairing renders perfectly while being wrong.
    """
    by_plane: Dict[int, Any] = {}
    for ds in datasets:
        sees = getattr(ds, "sees", None)
        if sees is None or isinstance(sees, str):
            raise ValueError(
                "every dataset must carry an explicit sees=<plane light components> so "
                "it can be matched to a plane by identity; got sees=%r. A model with "
                "one cutout per source redshift cannot be paired by position." % (sees,))
        ids = {id(c) for c in sees}
        hits = [i for i, p in enumerate(model.planes)
                if p.has_light and ids <= {id(c) for c in p.light}]
        if len(hits) != 1:
            raise ValueError(
                f"dataset sees={len(ids)} component(s) matching {len(hits)} planes; "
                "expected exactly 1. Rebuild the datasets against this model's own "
                "Components -- sees matches by object identity, not by value.")
        if hits[0] in by_plane:
            raise ValueError(f"two datasets both claim plane {hits[0]}")
        by_plane[hits[0]] = ds
    return by_plane


def _clump_snr(model, params, plane_index, members, dataset) -> float:
    """Matched-filter SNR of one clump's lensed, PSF-convolved model in its own band."""
    import jax.numpy as jnp
    from gigalens.jax.scene_simulator import SceneSimulator

    seen = [model.planes[plane_index].light[j] for j in members]
    sim = SceneSimulator(model, dataset.sim_config, sees=seen)
    image = jnp.asarray(sim.simulate(params), dtype=jnp.float64)
    resid = jnp.where(dataset.mask, image / dataset.error_map, 0.0)
    return float(jnp.sqrt(jnp.sum(resid**2)))


# --------------------------------------------------------------------------------
# The swap
# --------------------------------------------------------------------------------
@dataclass
class SwapReport:
    """Result of :func:`sersic_swap`, plus what it achieved."""

    model: Any
    params: Dict[str, Any]
    clumps: List[Clump]
    rows: List[Dict[str, Any]] = field(default_factory=list)
    n_sersic: float = 2.0
    match: str = "snr"
    aperture: float = 10.0

    def summary(self) -> None:
        print(f"sersic_swap: n_sersic={self.n_sersic}, match={self.match!r}, "
              f"aperture={self.aperture}\"")
        head = (f"{'plane':<18}{'clump':<14}{'from':<24}{'R_sersic':>9}{'r/3\"':>8}"
                f"{'e1':>7}{'e2':>7}{'Ie':>12}")
        if self.match == "snr":
            head += f"{'SNR':>10}{'SNR out/in':>11}"
        head += f"{'|F| out/in':>11}"
        print(head)
        flagged = []
        for row in self.rows:
            line = (f"{row['plane_name']:<18}{row['clump_name']:<14}{row['from']:<24}"
                    f"{row['R_sersic']:9.4f}{row['r_half_inner']:8.4f}"
                    f"{row['e1']:7.3f}{row['e2']:7.3f}{row['Ie']:12.5g}")
            if self.match == "snr":
                line += f"{row['snr_in']:10.4g}{row['snr_ratio']:11.5f}"
            line += f"{row['flux_ratio']:11.4f}"
            print(line)
            if row["aperture_sensitive"]:
                flagged.append(row)
        if flagged:
            print(f"\n  aperture-sensitive ({len(flagged)}): r_half moves >10% between "
                  f"{self.aperture/3:g}\" and {self.aperture:g}\". The quoted size is a "
                  "statement about the aperture, not the object:")
            for row in flagged:
                print(f"    {row['plane_name']:<18}{row['clump_name']:<12}"
                      f"{row['r_half_inner']:.4f} -> {row['R_sersic']:.4f}"
                      f"  ({row['R_sersic']/row['r_half_inner']:.2f}x)")
        cancel = [r for r in self.rows if abs(r["net_over_abs"]) < 0.5]
        if cancel:
            print(f"\n  internally cancelling ({len(cancel)}): |net flux| < 50% of "
                  "integrated |flux|, so 'total flux' is not a well-posed target for "
                  "these; the SNR match uses the signed lensed image and is unaffected:")
            for row in cancel:
                print(f"    {row['plane_name']:<18}{row['clump_name']:<12}"
                      f"net/|F| = {row['net_over_abs']:+.3f}")


def sersic_swap(model, params, *, n_sersic: float = 2.0, aperture: float = 10.0,
                match: str = "snr", datasets: Optional[Sequence[Any]] = None,
                flux_kind: str = "abs", ellipticity: str = "measured",
                sign: str = "abs", group: str = "clump",
                n_r: int = N_R, n_t: int = N_T) -> SwapReport:
    """Return a copy of ``model`` with every source clump replaced by one Sersic.

    Parameters
    ----------
    model, params
        The complex scene and a full structured parameter dict for it
        (``model.to_params(...)``). Free parameters in the *light* are not supported --
        the swap measures a rendered scene, so the light must be at a definite value.
        Mass and cosmology components are carried over by identity, priors intact.
    n_sersic
        Sersic index of every output source. Fixed, not fitted; this is the property the
        swap deliberately does not preserve.
    aperture
        Radius in arcsec within which size and flux are measured. Not a cosmetic default:
        see the module docstring. ``SwapReport.summary`` flags every clump whose size
        depends on it.
    match
        ``"snr"`` -- match each clump's matched-filter SNR in its own band; requires
        ``datasets``. ``"flux"`` -- match its source-plane integrated flux instead; no
        data needed.
    datasets
        The ``ImageData`` list, one per source plane, each carrying an explicit ``sees``.
        Paired to planes by object identity, never by position.
    flux_kind
        ``"abs"`` (integral of ``|I|``) or ``"net"`` (signed integral), for
        ``match="flux"`` only.
    ellipticity
        ``"measured"`` (from the second moments) or ``"circular"`` (``e1=e2=0``).
    sign
        ``"abs"`` -- every output ``Ie`` is positive. ``"signed"`` -- keep the sign of the
        matched quantity, reproducing negative-surface-brightness sources faithfully.
    group
        ``"clump"`` (one Sersic per shared centre) or ``"plane"`` (one per source plane).
    """
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.profiles.light.sersic import SersicEllipse

    _check_bn()
    if match not in ("snr", "flux"):
        raise ValueError(f"match must be 'snr' or 'flux'; got {match!r}")
    if match == "snr" and not datasets:
        raise ValueError(
            "match='snr' measures the lensed, PSF-convolved model against the real "
            "error map and mask, so it needs datasets=. Pass match='flux' for a "
            "data-free swap -- but note that discards magnification and seeing.")
    if flux_kind not in ("abs", "net"):
        raise ValueError(f"flux_kind must be 'abs' or 'net'; got {flux_kind!r}")
    if ellipticity not in ("measured", "circular"):
        raise ValueError(f"ellipticity must be 'measured' or 'circular'; got {ellipticity!r}")
    if sign not in ("abs", "signed"):
        raise ValueError(f"sign must be 'abs' or 'signed'; got {sign!r}")
    if n_sersic <= 0.1:
        raise ValueError(
            f"n_sersic must exceed 0.1; got {n_sersic}. gigalens' b_n polynomial "
            "diverges from the analytic solution below that (sersic.py domain).")

    clumps = measure_clumps(model, params, aperture=aperture, group=group,
                            n_r=n_r, n_t=n_t)
    by_plane = _dataset_for_plane(model, datasets) if match == "snr" else {}

    # ---- targets, measured on the INPUT model -----------------------------------
    targets: List[float] = []
    for clump in clumps:
        if match == "snr":
            if clump.plane not in by_plane:
                raise ValueError(
                    f"plane {clump.plane} ({clump.plane_name}) has light but no dataset; "
                    "match='snr' needs one band per source plane.")
            targets.append(_clump_snr(model, params, clump.plane, clump.members,
                                      by_plane[clump.plane]))
        else:
            targets.append(clump.flux_abs if flux_kind == "abs" else clump.flux_net)

    # ---- geometry of the replacements -------------------------------------------
    def light_for(clump: Clump, Ie: float, index: int) -> Component:
        e1, e2 = (clump.e1, clump.e2) if ellipticity == "measured" else (0.0, 0.0)
        if math.hypot(e1, e2) >= 1.0:               # gigalens: q = (1-c)/(1+c) -> NaN
            raise ValueError(
                f"{clump.plane_name}: measured |e| = {math.hypot(e1, e2):.4f} >= 1, "
                "which gigalens renders as NaN. The second moments are degenerate; "
                "use ellipticity='circular' or a different aperture.")
        return Component(SersicEllipse(use_lstsq=False), {
            "R_sersic": clump.r_half, "n_sersic": float(n_sersic),
            "e1": e1, "e2": e2,
            "center_x": clump.center_x, "center_y": clump.center_y, "Ie": Ie,
        }, name=f"sersic_{index}")

    def assemble(amplitudes: Sequence[float]):
        per_plane: Dict[int, List[Component]] = {}
        for clump, Ie in zip(clumps, amplitudes):
            bucket = per_plane.setdefault(clump.plane, [])
            bucket.append(light_for(clump, Ie, len(bucket)))
        planes = []
        for i, plane in enumerate(model.planes):
            planes.append(Plane(redshift=plane.redshift,
                                deflection_ratio=plane.deflection_ratio,
                                mass=plane.mass, light=per_plane.get(i, []),
                                name=plane.name))
        return LensModel(planes, cosmo=model.cosmo)

    # ---- amplitudes --------------------------------------------------------------
    if match == "flux":
        unit = [sersic_total_flux(1.0, c.r_half, n_sersic) for c in clumps]
        amps = [t/u for t, u in zip(targets, unit)]
    else:
        # The forward model is linear in Ie, so one probe render per clump at Ie=1 gives
        # the exact scaling. Warnings are suppressed only around the probe: Ie=1 is not
        # the amplitude anyone asked about, and the real one is checked below.
        probe = assemble([1.0]*len(clumps))
        probe_params = probe.to_params({})
        amps = []
        for k, (clump, target) in enumerate(zip(clumps, targets)):
            unit_snr = _clump_snr(probe, probe_params, clump.plane,
                                  (row_index_in_plane(clumps, k),),
                                  by_plane[clump.plane])
            if not unit_snr > 0:
                raise ValueError(
                    f"{clump.plane_name}: the replacement Sersic renders zero signal in "
                    "its band (entirely masked, or outside the cutout). Cannot match SNR.")
            amps.append(target/unit_snr)

    if sign == "abs":
        amps = [abs(a) for a in amps]
    else:
        # SNR is unsigned by construction, so "keep the sign" has to mean the sign of the
        # source being replaced -- its NET flux, since that is what a telescope adds up.
        # Without this branch sign='signed' would silently be a no-op under match='snr'.
        amps = [abs(a) * (-1.0 if c.flux_net < 0 else 1.0)
                for a, c in zip(amps, clumps)] if match == "snr" else amps

    # Deliberately NOT suppressed: with sign='signed' a negative Ie warns, and that
    # warning is the whole point of having asked for it.
    new_model = assemble(amps)
    new_params = new_model.to_params({})

    # ---- verify, on the model that is actually returned --------------------------
    # Nothing above is asserted; everything is re-measured through the same estimators
    # that produced the targets, so a bug in the estimator cancels and a bug in the
    # assembly does not.
    achieved = measure_clumps(new_model, new_params, aperture=aperture, group=group,
                              n_r=n_r, n_t=n_t)
    if len(achieved) != len(clumps):
        raise AssertionError(
            f"swap produced {len(achieved)} clumps from {len(clumps)}; regrouping failed")

    rows = []
    for k, (clump, new, Ie, target) in enumerate(zip(clumps, achieved, amps, targets)):
        row: Dict[str, Any] = {
            "plane": clump.plane, "plane_name": clump.plane_name,
            "clump_name": new.member_names[0],
            "from": "+".join(sorted({n.rstrip("_0123456789") or n
                                     for n in clump.member_names})),
            "R_sersic": clump.r_half, "r_half_inner": clump.r_half_inner,
            "r_half_achieved": new.r_half, "e1": new.e1, "e2": new.e2, "Ie": Ie,
            "flux_in_abs": clump.flux_abs, "flux_in_net": clump.flux_net,
            "flux_out": new.flux_abs,
            "flux_ratio": new.flux_abs/clump.flux_abs if clump.flux_abs else math.nan,
            "net_over_abs": (clump.flux_net/clump.flux_abs
                             if clump.flux_abs else math.nan),
            "aperture_sensitive": clump.aperture_sensitive,
        }
        if abs(new.center_x - clump.center_x) > 1e-6*max(1.0, aperture) or \
                abs(new.center_y - clump.center_y) > 1e-6*max(1.0, aperture):
            raise AssertionError(
                f"clump {k} ({clump.plane_name}) was measured at "
                f"({clump.center_x:.6f}, {clump.center_y:.6f}) but its replacement "
                f"measures at ({new.center_x:.6f}, {new.center_y:.6f}); the swapped "
                "clumps are not in the order they were measured.")
        if match == "snr":
            row["snr_in"] = target
            row["snr_out"] = _clump_snr(
                new_model, new_params, clump.plane,
                (_component_at(new_model, new_params, clump.plane,
                               clump.center_x, clump.center_y),),
                by_plane[clump.plane])
            row["snr_ratio"] = row["snr_out"]/target if target else math.nan
        rows.append(row)

    return SwapReport(model=new_model, params=new_params, clumps=clumps, rows=rows,
                      n_sersic=float(n_sersic), match=match, aperture=aperture)


def row_index_in_plane(clumps: Sequence[Clump], k: int) -> int:
    """Position of ``clumps[k]`` among the clumps sharing its plane."""
    return sum(1 for c in clumps[:k] if c.plane == clumps[k].plane)


def _component_at(model, params, plane_index: int, cx: float, cy: float,
                  tol: float = 1e-6) -> int:
    """Index of the light Component on ``plane_index`` centred at ``(cx, cy)``.

    Deliberately a *different* route to the same component than
    :func:`row_index_in_plane`, which walks positions. The verification pass uses this
    one so that a bug in the positional bookkeeping cannot cancel between the probe
    render and the check: the two would then disagree about which clump they measured,
    and the reported SNR ratio would leave 1.0.
    """
    hits = []
    for j in range(len(model.planes[plane_index].light)):
        prm = params["planes"][plane_index]["light"][j]
        if (abs(float(prm["center_x"]) - cx) <= tol
                and abs(float(prm["center_y"]) - cy) <= tol):
            hits.append(j)
    if len(hits) != 1:
        raise AssertionError(
            f"plane {plane_index}: {len(hits)} swapped components sit at "
            f"({cx:.6f}, {cy:.6f}); expected exactly 1. Clump grouping and component "
            "assembly disagree.")
    return hits[0]


# --------------------------------------------------------------------------------
def _check_bn() -> None:
    """Fail loudly if gigalens' b_n polynomial no longer matches the one duplicated here."""
    from gigalens.jax.profiles.light import sersic as _s
    theirs = np.asarray(_s._BN_COEFFS, dtype=float)
    if not np.allclose(theirs, np.asarray(_BN_COEFFS, dtype=float), rtol=1e-12):
        raise RuntimeError(
            "gigalens' Sersic b_n coefficients have changed; sersic_swap's copy is stale "
            "and its analytic flux normalisation would be silently wrong. Update "
            "_BN_COEFFS here to match sersic.py.")


def selftest(verbose: bool = True) -> bool:
    """Check the measurement against two profile families with analytic truth.

    Neither case is a Sersic-swap in disguise: the first asks the estimator to recover a
    Sersic's own ``R_sersic``, ``e1``/``e2``, centre and closed-form total flux; the
    second feeds it a single shapelet coefficient, which is a circular Gaussian of width
    ``beta``, with half-light radius ``beta*sqrt(2 ln 2)`` and total flux
    ``2 sqrt(pi) amp beta^2`` -- values the Sersic machinery knows nothing about.

    The ``n=9.08`` case is the model's worst real source and needs a ~890 arcsec radial
    span to hold 1-1e-4 of its flux; it is here because that is the regime where a
    Cartesian grid fails and where the aperture question below becomes unavoidable.
    """
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.profiles.light.shapelets import Shapelets

    _check_bn()
    ok = True

    def report(label, got, want, tol):
        nonlocal ok
        bad = abs(got - want) > tol * max(abs(want), 1e-12)
        ok = ok and not bad
        if verbose:
            print(f"  {'FAIL' if bad else 'ok  '}  {label:<34} {got:14.6g} "
                  f"want {want:14.6g}")

    profile = SersicEllipse(use_lstsq=False)
    for R_sersic, n, e1, e2 in ((0.5, 1.0, 0.0, 0.0), (0.5, 2.0, 0.3, -0.2),
                                (0.9, 4.0, -0.15, 0.25), (0.836, 9.08, -0.178, 0.219)):
        prm = dict(R_sersic=R_sersic, n_sersic=n, e1=e1, e2=e2,
                   center_x=1.5, center_y=-2.0, Ie=7.0)
        # Radius holding 1-1e-4 of the flux, from the exact incomplete-gamma inversion.
        # It grows as R_sersic*(x/bn)^n: ~7x R_sersic at n=1, but ~1000x at n=9. An
        # earlier closed-form guess here returned 64" for the n=9.08 case instead of
        # ~890" and the selftest failed on it -- which is exactly the aperture problem
        # this module warns about, showing up in its own test.
        from scipy.special import gammaincinv
        span = R_sersic * (float(gammaincinv(2*n, 1 - 1e-4)) / bn(n))**n
        cx, cy, q, phi = _moments([profile], [prm], 1.5, -2.0, span, N_R, N_T)
        r_half, f_abs, _, _, _ = _growth([profile], [prm], cx, cy, q, phi,
                                         span, N_R, N_T)
        c = (1 - q)/(1 + q)
        if verbose:
            print(f"pure Sersic R={R_sersic} n={n} (radial span {span:.3g}\")")
        report("r_half == R_sersic", r_half, R_sersic, 2e-3)
        report("e1", c*math.cos(2*phi), e1, 5e-3) if e1 else None
        report("e2", c*math.sin(2*phi), e2, 5e-3) if e2 else None
        report("centre offset", math.hypot(cx-1.5, cy+2.0), 0.0, 1.0)
        report("flux == analytic", f_abs, sersic_total_flux(7.0, R_sersic, n), 2e-3)

    for beta, amp in ((0.25, 3.0), (0.0263, 120.0)):
        shapelet = Shapelets(2, use_lstsq=False)
        prm = {k: 0.0 for k in shapelet._amp_names}
        prm[shapelet._amp_names[0]] = amp
        prm.update(beta=beta, center_x=0.7, center_y=1.3)
        cx, cy, q, phi = _moments([shapelet], [prm], 0.7, 1.3, beta*12, N_R, N_T)
        r_half, f_abs, _, _, _ = _growth([shapelet], [prm], cx, cy, q, phi,
                                         beta*12, N_R, N_T)
        if verbose:
            print(f"amp00-only shapelet beta={beta} (a circular Gaussian)")
        report("r_half == beta*sqrt(2 ln 2)", r_half, beta*math.sqrt(2*math.log(2)), 2e-3)
        report("flux == 2 sqrt(pi) amp beta^2", f_abs,
               2*math.sqrt(math.pi)*amp*beta**2, 2e-3)
        report("axis ratio == 1", q, 1.0, 1e-6)

    if verbose:
        print("selftest:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import argparse
    import os
    import jax
    jax.config.update("jax_enable_x64", True)

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--selftest", action="store_true",
                        help="check the measurement against analytic truth and exit")
    parser.add_argument("--aperture", type=float, default=10.0)
    parser.add_argument("--n-sersic", type=float, default=2.0)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    here = os.path.dirname(os.path.abspath(__file__))
    from translate_old_params import build

    model, spec = build(args.json or os.path.join(
        here, "MAP_best_31JulNFW_fixedcosmo_fixedLowZ.json"))
    # No datasets on the command line, so this is the data-free variant; see the module
    # docstring for why match="snr" is the one you want for an actual mock.
    sersic_swap(model, model.to_params({}), n_sersic=args.n_sersic,
                aperture=args.aperture, match="flux").summary()
