"""SHIM -- NFW_ELLIPSE_SLOPE moved into the gigalens package at user request
(2026-07-04): gigalens/src/gigalens/jax/profiles/mass/nfw_ellipse_slope.py.

This re-export keeps the harness scripts (t29_slope_class_gpu.py and any
committed history) working unchanged. Import from gigalens directly in new
code:

    from gigalens.jax.profiles.mass.nfw_ellipse_slope import NFW_ELLIPSE_SLOPE
"""
from gigalens.jax.profiles.mass.nfw_ellipse_slope import (  # noqa: F401
    H_LNR,
    NFW_ELLIPSE_SLOPE,
    Rs_of_s_thetaE,
    _NBISECT,
    _XHI,
    _XLO,
    _dsigma_dx,
    _g,
    s_of_Rs_thetaE,
    sigma_of_x,
    x_of_s,
)
