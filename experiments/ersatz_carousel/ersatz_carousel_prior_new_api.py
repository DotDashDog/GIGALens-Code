#!/usr/bin/env python
"""New-API translation of ``ersatz_carousel_prior.py``.

Every distribution/number below is copied unchanged from ``ersatz_carousel_prior.py``;
nothing here is fit, derived, or guessed. What changes is only the container each one
sits in, mechanically:

Mass priors
-----------
Old ``Prior(profile, dict_of_tfd)`` -> new ``Component(profile, priors_dict, name=...)``.
The new API has no ``Prior`` wrapper at all (verified: no ``class Prior`` anywhere under
``gigalens/jax``); a ``Component``'s second argument -- called ``priors``, not ``params``
-- takes the *same* dict, and gigalens introspects each value's type itself (a bare
number is fixed, a ``tfd.Distribution`` is a free parameter) exactly as the old ``Prior``
did. Six mass ``Component``s come out (``upper_right_halo`` included, though -- per
``translate_old_params.STRUCTURE`` -- it is commented out in ``boiler(2).py`` and absent
from every fit; translated here anyway since it is defined in the source file, unused
there too). Names match ``translate_old_params.STRUCTURE["lens_names"]`` where the same
five components appear there, for the same profile in the same position, so the two
translation scripts refer to one lens by one name.

Source-light priors: two structural changes
--------------------------------------------
1. **``CombinedProfile``/``DoubleSersic`` are gone** (confirmed: neither class exists
   under current ``gigalens/src`` -- both are old-API-only). A plane simply holds a list
   of light ``Component``s, so every combined ``Prior`` here becomes several ``Component``s
   in a Python list, split by the exact same suffix convention ``translate_old_params.py``
   already reverse-engineers from ``combined_profile.py`` (see its ``_split_one_level``):
   a non-shared parameter of sub-profile ``i`` is suffixed ``_{i}``; a ``shared_params``
   entry (default ``center_x``/``center_y``/``e1``/``e2``) is written bare once, and that
   nests for ``DoubleSersic`` (itself a ``CombinedProfile`` of two ``SersicEllipse``,
   default ``shared_params``) wrapped inside an outer ``CombinedProfile`` with
   ``shared_params=[]``. Splitting each source's flat dict by that same convention gives
   the exact leaf components below -- e.g. ``source1_prior``'s ``CombinedProfile([DoubleSersic(...),
   SersicEllipse(...)], shared_params=[])`` becomes 3 leaf Sersics (``sersic_0_0``,
   ``sersic_0_1``, ``sersic_1``), matching ``translate_old_params._expand_components``'s
   own path/tag convention (``tag = "_" + "_".join(path)``) so the two scripts name leaves
   the same way.

2. **A parameter two sub-profiles used to *share* (one sampled value, not two) needs
   ``shared()``, not a copied distribution.** ``DoubleSersic``'s default
   ``shared_params`` and ``source3_prior``'s explicit ``shared_params=['center_x',
   'center_y', 'e1', 'e2']`` both mean the old fit drew ONE value for that parameter and
   reused it in both sub-profiles' light. Writing the same ``tfd.Distribution`` twice --
   once per leaf ``Component`` -- would instead draw two INDEPENDENT values, silently
   changing the model (two Sersics that no longer share a centre/ellipticity). The new
   API's ``shared()`` handle (``gigalens.jax.scene.shared``: "reuse the *same instance* at
   multiple sites to share") is exactly the mechanism for this, and is used below wherever
   the old ``shared_params`` applied: ``source1_prior``'s ``center_x_0``/``center_y_0``/
   ``e1_0``/``e2_0`` (shared between its two ``DoubleSersic`` sub-Sersics), and
   ``source3_prior``'s bare ``center_x``/``center_y``/``e1``/``e2`` (shared between its
   two top-level Sersics). Every other combined source (``source45_prior``,
   ``source1213_prior``) has ``shared_params=[]`` and needs no ``shared()`` at all -- each
   leaf already has its own independent distribution in the original dict.

``deflection_ratio`` -> ``Plane(deflection_ratio=...)``, not a cosmology
-------------------------------------------------------------------------
Every source profile here is built with ``cosmo_sample=False``, and no cosmology
``Prior``/``Component`` is defined anywhere in this file: the old model samples each
source's deflection ratio directly, rather than deriving it from a cosmology + redshift
(contrast ``translate_old_params.py``, which *does* have a fixed cosmology and therefore
derives the ratio -- a different old fit). The new API's ``Plane`` geometry is "exactly
one of ``deflection_ratio`` (no cosmology) or ``redshift`` (with cosmology)"
(``scene.py``); with no cosmology ``Component`` supplied here, ``deflection_ratio`` is the
correct field, and it accepts a ``tfd.Distribution`` exactly like any ``Component``
prior -- so each source's ``deflection_ratio`` prior moves onto its ``Plane`` unchanged,
still the same free parameter it always was. This is the only place a ``Plane`` needs
building at all: mass ``Component``s are left bare below, exactly as unassembled as they
were as bare ``Prior``s in the original file (this file was never the place they got
grouped into a lens plane; that stays whoever's job it already was).

Each source's ``z_source`` comment is kept for reference only, unused as a value here,
exactly as it was already unused (commented out) in ``ersatz_carousel_prior.py`` --
translating ``deflection_ratio`` needs no redshift at all.

Not translated: none of this changes what is or isn't a free parameter. ``Ie`` stays
absent everywhere it was absent (``use_lstsq=True`` throughout, both files), and every
bound, scale, and location number is verbatim from ``ersatz_carousel_prior.py``.
"""
from __future__ import annotations

import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

from gigalens.jax.scene import Component, Plane, shared
from gigalens.jax.profiles.mass.nfw_ellipse_slope import NFW_ELLIPSE_SLOPE
from gigalens.jax.profiles.mass.piemd import DPIE
from gigalens.jax.profiles.mass.shear import Shear
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.light.sersic import SersicEllipse

# --------------------------------------------------------------------------------
# Mass priors -- bare Components, unassembled, exactly as they were bare Priors.
# Names match translate_old_params.STRUCTURE["lens_names"] (same profile, same
# position) for the five that appear in that STRUCTURE.
# --------------------------------------------------------------------------------
halo_model = Component(
    NFW_ELLIPSE_SLOPE(),
    dict(
        center_x=tfd.Normal(6.69965551, 1),
        center_y=tfd.Normal(4.80431651, 1),
        e1=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
        s_E=tfd.Uniform(0., 0.8),
        theta_E=tfd.Uniform(12, 14),
    ),
    name="cluster_halo",
)

# fixed position, PIEMD
ld_free_ellip_model = Component(
    DPIE(),
    dict(
        center_x=tfd.Normal(11.80977389, 0.1),
        center_y=tfd.Normal(23.0283886, 0.1),
        theta_E=tfd.TruncatedNormal(1.6730331, 0.5, 1, 2.5),
        r_cut=tfd.LogNormal(jnp.log(10), 1),
        r_core=tfd.LogNormal(jnp.log(0.05), 0.01),
        e1=tfd.TruncatedNormal(0, 0.1, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0, 0.1, -0.5, 0.5),
    ),
    name="bcg_dpie",
)

shear_model = Component(
    Shear(),
    dict(
        gamma1=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        gamma2=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
    ),
    name="ext_shear",
)

le_free_model = Component(
    DPIE(),
    dict(
        center_x=tfd.Normal(-21.17580938, 0.1),
        center_y=tfd.Normal(-24.25810504, 0.1),
        e1=tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
        theta_E=tfd.TruncatedNormal(1.6711541, 0.1, 1, 2.5),
        r_cut=tfd.LogNormal(jnp.log(10), 1),
        r_core=tfd.LogNormal(jnp.log(0.05), 0.01),
    ),
    name="le_dpie",
)

group_halo_free_model = Component(
    DPIE(),
    dict(
        center_x=tfd.Normal(-15.10088063, .1),
        center_y=tfd.Normal(-4.66657821, .1),
        e1=tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
        theta_E=tfd.TruncatedNormal(0.8151327, 0.1, 0.2, 1.5),
        r_cut=tfd.LogNormal(jnp.log(10), 1),
        r_core=tfd.LogNormal(jnp.log(0.05), 0.01),
    ),
    name="group_dpie",
)

# Commented out in boiler(2).py and absent from every JSON dump (translate_old_params.py
# STRUCTURE); translated anyway since it is defined here, same as in the old file.
upper_right_halo = Component(
    EPL(),
    dict(
        center_x=tfd.Normal(32, .5),
        center_y=tfd.Normal(22, .5),
        e1=tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0, 0.05, -0.3, 0.3),
        theta_E=tfd.LogNormal(jnp.log(1), 0.05),
        gamma=tfd.TruncatedNormal(2, 0.5, 1, 3),
    ),
    name="upper_right_halo",
)

# --------------------------------------------------------------------------------
# Source-light priors -- one Plane per source, bundling deflection_ratio (the
# CombinedProfile dict's own top-level key, now Plane's, since it has nowhere else to
# live) with its light Components (the CombinedProfile's former sub-profiles, now a
# flat list -- see module docstring for the suffix/shared() derivation of each).
# --------------------------------------------------------------------------------

# source 1_2 (z_source = 0.962): CombinedProfile([DoubleSersic(...), SersicEllipse(...)],
# shared_params=[]). DoubleSersic's default shared_params=[center_x,center_y,e1,e2]
# ties its two Sersics' centre/ellipticity together -- one sampled value, not two.
_source1_center_x_0 = shared(tfd.Normal(7.67187389, 5), name="source1_2_center_x_0")
_source1_center_y_0 = shared(tfd.Normal(3.31911655, 5), name="source1_2_center_y_0")
_source1_e1_0 = shared(tfd.TruncatedNormal(0., 0.1, -0.3, 0.3), name="source1_2_e1_0")
_source1_e2_0 = shared(tfd.TruncatedNormal(0., 0.1, -0.3, 0.3), name="source1_2_e2_0")

sersic_0_0 = Component(
    SersicEllipse(use_lstsq=True),
    dict(
        center_x=_source1_center_x_0,
        center_y=_source1_center_y_0,
        e1=_source1_e1_0,
        e2=_source1_e2_0,
        n_sersic=tfd.Uniform(.25, 10),
        R_sersic=tfd.LogNormal(jnp.log(0.4), 0.15),
    ),
    name="sersic_0_0",
)
sersic_0_1 = Component(
    SersicEllipse(use_lstsq=True),
    dict(
        center_x=_source1_center_x_0,
        center_y=_source1_center_y_0,
        e1=_source1_e1_0,
        e2=_source1_e2_0,
        n_sersic=tfd.Uniform(.25, 10),
        R_sersic=tfd.LogNormal(jnp.log(0.4), 0.15),
    ),
    name="sersic_0_1",
)
sersic_1 = Component(
    SersicEllipse(use_lstsq=True),
    dict(
        center_x=tfd.Normal(10, 5),
        center_y=tfd.Normal(3, 5),
        e1=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic=tfd.Uniform(1, 10),
        R_sersic=tfd.LogNormal(jnp.log(0.4), 0.15),
    ),
    name="sersic_1",
)
source1_plane = Plane(
    # z_source = 0.962 -- kept for reference only, same as the commented-out value in
    # ersatz_carousel_prior.py; no cosmology is defined here, so no redshift is needed.
    deflection_ratio=tfd.Uniform(0.5, 1),
    light=[sersic_0_0, sersic_0_1, sersic_1],
    name="source1_2",
)

# source 3 (z_source = 1.166): CombinedProfile([SersicEllipse, SersicEllipse],
# shared_params=['center_x', 'center_y', 'e1', 'e2']) -- explicit, but the same as
# DoubleSersic's default. Both Sersics share one centre/ellipticity.
_source3_center_x = shared(tfd.Normal(6.79821086, 5), name="source3_center_x")
_source3_center_y = shared(tfd.Normal(7.91570776, 5), name="source3_center_y")
_source3_e1 = shared(tfd.TruncatedNormal(0., 0.1, -0.3, 0.3), name="source3_e1")
_source3_e2 = shared(tfd.TruncatedNormal(0., 0.1, -0.3, 0.3), name="source3_e2")

source3_sersic_0 = Component(
    SersicEllipse(use_lstsq=True),
    dict(
        center_x=_source3_center_x,
        center_y=_source3_center_y,
        e1=_source3_e1,
        e2=_source3_e2,
        n_sersic=tfd.Uniform(.25, 10),
        R_sersic=tfd.LogNormal(jnp.log(0.4), 0.15),
    ),
    name="sersic_0",
)
source3_sersic_1 = Component(
    SersicEllipse(use_lstsq=True),
    dict(
        center_x=_source3_center_x,
        center_y=_source3_center_y,
        e1=_source3_e1,
        e2=_source3_e2,
        n_sersic=tfd.Uniform(.25, 10),
        R_sersic=tfd.Uniform(1e-3, 1),
    ),
    name="sersic_1",
)
source3_plane = Plane(
    # z_source = 1.166 -- reference only.
    deflection_ratio=tfd.Uniform(0.5, 1),
    light=[source3_sersic_0, source3_sersic_1],
    name="source3",
)

# source 4_5 (z_source = 1.432): CombinedProfile([SersicEllipse, SersicEllipse],
# shared_params=[]) -- no sharing; two fully independent Sersics.
source45_sersic_0 = Component(
    SersicEllipse(use_lstsq=True),
    dict(
        center_x=tfd.Normal(4.63, 5),
        center_y=tfd.Normal(3.79, 5),
        e1=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic=tfd.Uniform(.25, 10),
        R_sersic=tfd.Uniform(1e-3, 1),
    ),
    name="sersic_0",
)
source45_sersic_1 = Component(
    SersicEllipse(use_lstsq=True),
    dict(
        center_x=tfd.Normal(4.78792923, 5),
        center_y=tfd.Normal(0.84347007, 5),
        e1=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic=tfd.Uniform(.25, 10),
        R_sersic=tfd.Uniform(1e-3, 1),
    ),
    name="sersic_1",
)
# deflection_ratio = 1 in the original (a fixed constant, not a distribution) -- a
# Component/Plane prior value may be either, and gigalens classifies it as fixed the
# same way it would classify a fixed float anywhere else.
source45_plane = Plane(
    # z_source = 1.432 -- reference only.
    deflection_ratio=1,
    light=[source45_sersic_0, source45_sersic_1],
    name="source4_5",
)

# source 9 (z_source = 1.506): plain SersicEllipse, never combined.
source9_sersic = Component(
    SersicEllipse(use_lstsq=True),
    dict(
        center_x=tfd.Normal(-7, 5),
        center_y=tfd.Normal(-13, 5),
        e1=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic=tfd.Uniform(.25, 10),
        R_sersic=tfd.Uniform(1e-3, 1),
    ),
    name="sersic",
)
source9_plane = Plane(
    # z_source = 1.506 -- reference only.
    deflection_ratio=tfd.Uniform(0.75, 1.25),
    light=[source9_sersic],
    name="source9",
)

# source 7 (z_source = 1.627): plain SersicEllipse, never combined.
source7_sersic = Component(
    SersicEllipse(use_lstsq=True),
    dict(
        center_x=tfd.Normal(0, 5),
        center_y=tfd.Normal(0, 5),
        e1=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic=tfd.Uniform(.25, 10),
        R_sersic=tfd.Uniform(1e-3, 1),
    ),
    name="sersic",
)
source7_plane = Plane(
    # z_source = 1.627 -- reference only.
    deflection_ratio=tfd.Uniform(1., 1.5),
    light=[source7_sersic],
    name="source7",
)

# source 6 (z_source = 1.656): plain SersicEllipse, never combined.
source6_sersic = Component(
    SersicEllipse(use_lstsq=True),
    dict(
        center_x=tfd.Normal(0, 5),
        center_y=tfd.Normal(0, 5),
        e1=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic=tfd.Uniform(.25, 10),
        R_sersic=tfd.Uniform(1e-3, 1),
    ),
    name="sersic",
)
source6_plane = Plane(
    # z_source = 1.656 -- reference only.
    deflection_ratio=tfd.Uniform(1., 1.5),
    light=[source6_sersic],
    name="source6",
)

# source 12_13 (z_source = 3.086): CombinedProfile([SersicEllipse, SersicEllipse],
# shared_params=[]) -- no sharing; two fully independent Sersics.
source1213_sersic_0 = Component(
    SersicEllipse(use_lstsq=True),
    dict(
        center_x=tfd.Normal(2.906817674636841, 5),
        center_y=tfd.Normal(4.523301601409912, 5),
        e1=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic=tfd.Uniform(.25, 10),
        R_sersic=tfd.Uniform(1e-3, 1),
    ),
    name="sersic_0",
)
source1213_sersic_1 = Component(
    SersicEllipse(use_lstsq=True),
    dict(
        center_x=tfd.Normal(6.670745849609375, 5),
        center_y=tfd.Normal(5.6769537925720215, 5),
        e1=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic=tfd.Uniform(.25, 10),
        R_sersic=tfd.Uniform(1e-3, 1),
    ),
    name="sersic_1",
)
source1213_plane = Plane(
    # z_source = 3.086 -- reference only.
    deflection_ratio=tfd.Uniform(1, 1.5),
    light=[source1213_sersic_0, source1213_sersic_1],
    name="source12_13",
)

# source 8 (z_source = 3.549): plain SersicEllipse, never combined.
source8_sersic = Component(
    SersicEllipse(use_lstsq=True),
    dict(
        center_x=tfd.Normal(6.1898108, 5),
        center_y=tfd.Normal(6.8792906, 5),
        e1=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic=tfd.Uniform(.25, 10),
        R_sersic=tfd.Uniform(1e-3, 1),
    ),
    name="sersic",
)
source8_plane = Plane(
    # z_source = 3.549 -- reference only.
    deflection_ratio=tfd.Uniform(1, 1.5),
    light=[source8_sersic],
    name="source8",
)

# source 11 (z_source = 4.090): plain SersicEllipse, never combined.
source11_sersic = Component(
    SersicEllipse(use_lstsq=True),
    dict(
        center_x=tfd.Normal(4.4566746, 5),
        center_y=tfd.Normal(2.0770147, 5),
        e1=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        e2=tfd.TruncatedNormal(0., 0.1, -0.3, 0.3),
        n_sersic=tfd.Uniform(.25, 10),
        R_sersic=tfd.Uniform(1e-3, 1),
    ),
    name="sersic",
)
source11_plane = Plane(
    # z_source = 4.090 -- reference only.
    deflection_ratio=tfd.Uniform(1., 1.5),
    light=[source11_sersic],
    name="source11",
)
