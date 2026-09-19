"""Tied core-Sersic lens light (2026-09-19, user decision after C-6 in
docs/logs/vela-f140w-modelling.md): the core-Sersic family with NO free core parameters.

    R_b = R_e * 10**f(n),   f(n) = log10(ceiling_frac) - slope_dex_per_n * w * log(1 + exp((n_knee - n) / w))
    gamma = 0 (flat core; a gamma > 0 core is an integrable but real central singularity the data
              cannot constrain), alpha = 5 (transition sharpness, unresolvable at 0.4 px).

f(n) is the DC-5 option-B median rule (2% of R_e for n >= 4.5, falling by 0.6 dex per unit n below:
1% at n = 4, 0.25% at 3, 0.06% at 2) with its hinge max(0, n_knee - n) replaced by a softplus of
width ``knee_width`` (in units of n) so the gradient of the likelihood with respect to n is
continuous (the hinge's slope jump would inject an energy error into MCLMC for every lens whose n
posterior straddles the knee). The softplus is at most 0.1 dex below the hinge (at the knee) and
agrees with it to < 0.01 dex beyond 1.2 n-units either side. No scatter: the SAME rule renders the
truth (generator: ``lens_light_profile: core_sersic_tied``) and the fit model
(:class:`TiedCoreSersic`), so the core is a modelling convention shared by both, like alpha.

Why: with R_b free the fit's posterior has a hockey-stick valley (R_b against n, with a "no core"
shelf holding ~6% of the posterior at Delta chi2 ~ 7) that costs MCLMC 10x in ESS on every parameter;
fixing R_b restores the pure-Sersic efficiency (C-6). The lens mass is unchanged by any of it.
"""
from __future__ import annotations

import functools
from typing import Any, Dict, Optional

import numpy as np

TIED_CORE_RULE: Dict[str, float] = {
    "ceiling_frac": 0.02,      # R_b / R_e for n >> n_knee
    "n_knee": 4.5,
    "slope_dex_per_n": 0.6,    # fall of log10(R_b / R_e) per unit n below the knee
    "knee_width": 0.25,        # softplus width in n
    "gamma": 0.0,
    "alpha": 5.0,
}
TIED_CORE_RULE_KEYS = ("ceiling_frac", "n_knee", "slope_dex_per_n", "knee_width")


def tied_core_log10_frac(n_sersic: Any, rule: Optional[Dict[str, Any]] = None, xp: Any = np) -> Any:
    """log10(R_b / R_e) for Sersic index ``n_sersic`` (array-friendly; ``xp`` = numpy or jax.numpy)."""
    r = {**TIED_CORE_RULE, **(rule or {})}
    w = float(r["knee_width"])
    hinge = w * xp.logaddexp(0.0, (float(r["n_knee"]) - n_sersic) / w)   # smooth max(0, n_knee - n)
    return xp.log10(float(r["ceiling_frac"])) - float(r["slope_dex_per_n"]) * hinge


def tied_core_radius(R_sersic: Any, n_sersic: Any, rule: Optional[Dict[str, Any]] = None, xp: Any = np) -> Any:
    """R_b [same units as ``R_sersic``] under the tied rule."""
    return R_sersic * 10.0 ** tied_core_log10_frac(n_sersic, rule, xp)


def _make_profile_class():
    import jax.numpy as jnp
    from jax import jit
    from gigalens.jax.profiles.light import sersic

    class TiedCoreSersic(sersic.CoreSersic):
        """``CoreSersic`` with R_b = rule(R_sersic, n_sersic), gamma and alpha constants from the
        rule: the free parameters are exactly those of ``SersicEllipse``."""
        _name = "TIED_CORE_SERSIC"
        _params = ["R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y"]
        _domains = dict((k, v) for k, v in getattr(sersic.CoreSersic, "_domains", {}).items()
                        if k in ("R_sersic", "n_sersic", "e1", "e2", "center_x", "center_y"))

        def __init__(self, use_lstsq: bool = False, rule: Optional[Dict[str, Any]] = None, **kwargs):
            self.rule = {**TIED_CORE_RULE, **(rule or {})}
            unknown = set(self.rule) - set(TIED_CORE_RULE)
            if unknown:
                raise KeyError(f"TiedCoreSersic rule: unknown keys {sorted(unknown)}")
            super().__init__(use_lstsq=use_lstsq, **kwargs)

        @functools.partial(jit, static_argnums=(0,))
        def light(self, x, y, R_sersic, n_sersic, e1, e2, center_x, center_y, Ie=None):
            Rb = tied_core_radius(R_sersic, n_sersic, self.rule, jnp)
            return sersic.CoreSersic.light(
                self, x, y, R_sersic, n_sersic, Rb, float(self.rule["alpha"]), float(self.rule["gamma"]),
                e1, e2, center_x, center_y, Ie)

    return TiedCoreSersic


_CLS = None


def TiedCoreSersic(use_lstsq: bool = False, rule: Optional[Dict[str, Any]] = None, **kwargs):
    """Factory (lazy import of gigalens/jax): a ``TiedCoreSersic`` profile instance."""
    global _CLS
    if _CLS is None:
        _CLS = _make_profile_class()
    return _CLS(use_lstsq=use_lstsq, rule=rule, **kwargs)
