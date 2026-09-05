"""Drop-in cosmology Component for the ersatz carousel with the ratio-pair + wa chart.

Why: with the plain gaussianized-box chart the (Om0, w0, wa) posterior is a curved
filament that rotates ~54 deg in sampling coordinates; MCLMC's frozen metric cannot
follow it, chains stall at the wa=2 wall, and wa reaches bulk-ESS 16 in 160k draws
(lab log ``docs/logs/ersatz-carousel-cosmology.md``). Here the two data-stiff
directions are the deflection ratios of the lowest and highest source planes and the
third coordinate is wa itself, so the filament is a straight axis.

Usage in ``gen_improved_sersic_ersatz_carousel.ipynb`` (cell that builds the model)::

    from ersatz_cosmo_chart import cosmo_ratio_pair_wa
    model = LensModel([FIT.lens_plane] + filtered_planes,
                      cosmo=cosmo_ratio_pair_wa(), unconstrain="gaussian")

The prior DENSITY is unchanged (uniform box) except for the documented support
amendment Om0 in (0.05, 0.99) (the chart folds at both Om0 edges; 0 of 160k baseline
draws lie below Om0=0.10). The z layout (column order and names) is identical to the
baseline prior file's, so downstream tooling keyed by ``z_param_names`` is unaffected.
"""
from __future__ import annotations

from gigalens.jax.cosmo import w0waCDM_Cosmo
from gigalens.jax.scene import Component

from gigalens_research.priors import RatioPairWaUniform, deflection_ratio_pair_wa_fn

Z_LENS = 0.49
Z_SOURCE_REF = 1.432
#: Lowest and highest source-plane redshifts of the ersatz carousel (largest
#: ratio-contour crossing angle among the available pairs).
Z_PAIR = (0.962, 4.090)
OM0_BOUNDS = (0.05, 0.99)
W0_BOUNDS = (-2.0, -1.0 / 3.0)
WA_BOUNDS = (-3.0, 2.0)
#: Measured 2026-09-05 on a (41, 31, 21) grid: the chart folds in a thin slab at
#: w0 > -0.67, wa in (0.25, 0.5) (all Om0) with whitened det down to -2.07e-3 -- an
#: intrinsic degeneracy (w_eff -> 0 makes dark energy matter-like) shared by every
#: pair tested. The baseline posterior has 0 of 160,000 draws with w0 > -0.75 at
#: -0.1 < wa < 0.6 (max w0 there is -0.784), so the slab carries no mass.
DET_ATOL = 3e-3
ROUNDTRIP_ATOL = 1e-9


def cosmo_ratio_pair_wa(*, skip_validation: bool = False) -> Component:
    """The ersatz-carousel cosmology Component with (Om0, w0, wa) grouped under the
    ratio-pair + wa chart. ``skip_validation=True`` skips the ~30 s grid validator
    (tests only); the validation report is on ``prior.ratio_pair_wa_report``."""
    cosmo = w0waCDM_Cosmo(z_lens=Z_LENS, z_source_ref=Z_SOURCE_REF)
    r_pair_fn = deflection_ratio_pair_wa_fn(cosmo, Z_PAIR, fixed=dict(H0=70.0, k=0.0))
    prior = RatioPairWaUniform(
        r_pair_fn, OM0_BOUNDS, W0_BOUNDS, WA_BOUNDS,
        det_atol=DET_ATOL, roundtrip_atol=ROUNDTRIP_ATOL, skip_validation=skip_validation)
    return Component(cosmo, {"H0": 70.0, "k": 0.0, ("Om0", "w0", "wa"): prior})
