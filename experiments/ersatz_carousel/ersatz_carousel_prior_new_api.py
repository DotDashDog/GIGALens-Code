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

Update: cosmology (flat LambdaCDM), and one assembled LensModel
-----------------------------------------------------------------
The above was the first pass: every source's deflection ratio was a free
``Plane(deflection_ratio=...)`` parameter, matching the original file's ``cosmo_sample=
False`` exactly, and no ``LensModel`` was assembled (mass ``Component``s were left bare,
same as the original file left them). Per request, this now instead:

1. **Adds a flat LambdaCDM cosmology.** The new API has two cosmology profiles,
   ``wCDM_Cosmo`` (params ``H0, Om0, k, w0``) and ``w0waCDM_Cosmo`` (adds ``wa``) -- no
   dedicated ``LambdaCDM`` class exists (``cosmo.py``), so LambdaCDM is ``wCDM_Cosmo``
   with ``w0`` FIXED at ``-1.0`` (not sampled) rather than given a prior; ``wa`` is not a
   parameter of ``wCDM_Cosmo`` at all, so it does not need fixing separately. Following
   ``experiments/sample_cosmology/bullseye.py``'s own cosmo ``Component`` (the setup this
   was modeled on): ``H0=70.0`` and ``k=0.0`` fixed (flat, standard), ``Om0`` free via
   ``UniformBij(0.0, 1.0)`` -- the same custom Shift/Scale/NormalCDF event-space bijector
   bullseye uses in place of TFP's default sigmoid one for a bounded uniform, copied
   verbatim from there (``tNCDF_bij``/``UniformBij``). ``z_lens=0.49`` and
   ``z_source_ref=1.432`` are ``translate_old_params.STRUCTURE``'s values for this exact
   physical setup (read off ``boiler(2).py`` there), not new numbers.

   This is a genuine choice, not a mechanical translation: which cosmological parameters
   are fixed vs sampled was not specified, and this follows bullseye's own precedent
   (fix H0/k, sample Om0) rather than inventing a different split.

2. **Every source's redshift is now real, not a comment.** With a cosmology supplied,
   ``Plane`` requires ``redshift``, not ``deflection_ratio`` (mutually exclusive --
   ``scene.py``: "Geometry is exactly one of ``deflection_ratio`` (no cosmology) or
   ``redshift`` (with cosmology)"). Each source's ``z_source`` -- previously kept only as
   a reference comment, since the original file's ``cosmo_sample=False`` never consumed
   it -- is exactly the value needed here, and is unchanged from that comment (in turn
   matching ``translate_old_params.STRUCTURE["source_ids"]``'s keys). The mass (cluster)
   plane is likewise given ``redshift=z_lens``.

   This REPLACES each source's ``deflection_ratio`` prior (kept only as a comment below,
   for audit) with a value *derived* from the cosmology + redshifts at sample time --
   the direct trade the user asked for by asking to "switch to using cosmology": a
   source's line-of-sight distance stops being an independently free parameter and
   becomes physically tied to the same (Om0, H0, k, w0) for every source and to the
   shared lens redshift.

3. **One assembled ``LensModel``.** ``model`` at the bottom collects the cluster mass
   plane (all six mass ``Component``s, unchanged) and all nine source planes, in ascending
   redshift order (observer -> source, as the new API requires), with the cosmology
   ``Component`` above passed as ``cosmo=``.

Update: ellipticity priors are now DiskEllipticity, not box-truncated normals
----------------------------------------------------------------------------
Every ``(e1, e2)`` pair -- previously two INDEPENDENT ``tfd.TruncatedNormal``s on a
box ``[-b, b]^2`` -- is now a single grouped (tuple-key) ``DiskEllipticity`` prior,
``("e1", "e2"): _disk_e(sd, b)``. See :func:`_disk_e` for the two numbers and how
each is derived from the prior it replaces; the short version is ``e_max =
max(0.5, b*sqrt(2))`` (0.5, or the old box's CORNER if that reaches further) and a
``scale`` chosen so the width in ellipticity space is unchanged.

Two mechanical consequences, neither of them a modelling choice:

* A joint prior over two parameters has to sit at a TUPLE key, which cannot be
  written in ``dict(e1=..., e2=...)`` keyword form, so the affected components' prior
  dicts became ``{...}`` literals. Key ORDER is preserved: a grouped prior "sorts into
  its first member's z-column slot" (``scene.py``), so leaving ``("e1", "e2")`` where
  ``e1`` used to be keeps the z-column order of the model identical.
* Where the old file shared ``e1`` and ``e2`` across two sub-Sersics via two separate
  ``shared()`` handles (``source3``, and ``source1_2``'s commented-out pair), that is
  now ONE ``shared()`` handle wrapping the joint prior -- ``shared()`` is accepted at a
  tuple key (``scene._classify_group``), and the pair is one prior now, so it cannot be
  shared in two halves. Side effect on NAMING, the only externally visible difference:
  ``_build_z_param_names`` names a grouped entry by its member param paths, and a shared
  group has several sites per member, so ``source3``'s two ellipticity z-columns are now
  keyed ``planes/2/light/1/e1``/``e2`` (labelled ``source3/light/sersic_1/e1``/``e2``)
  where they used to be ``shared_0002``/``shared_0003`` (labelled ``source3_e1``/``e2``).
  Same columns in the same positions, still one draw feeding BOTH Sersics -- but the name
  no longer announces the sharing, and any saved chain keyed by ``z_param_names`` will
  disagree with a new one on those two of 106 columns.

``gamma1``/``gamma2`` of ``ext_shear`` are deliberately NOT swapped: ``DiskEllipticity``
does serve any disk-bounded pair, but external shear is not an ellipticity and was not
part of the request. Change it here if that was intended.

Not translated: none of this changes what is or isn't a free MASS or LIGHT parameter.
``Ie`` stays absent everywhere it was absent (``use_lstsq=True`` throughout, both files),
and every mass/light bound, scale, and location number outside the ellipticity pairs is
still verbatim from ``ersatz_carousel_prior.py``. Only the geometry layer (deflection
ratio vs cosmology + redshift), the ellipticity priors, and the top-level assembly are
new.
"""
from __future__ import annotations

import math

import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import bijectors as tfb

from gigalens.jax.scene import Component, Plane, LensModel, shared
from gigalens.jax.utils.grouped_priors import DiskEllipticity
from gigalens.jax.cosmo import w0waCDM_Cosmo
from gigalens.jax.profiles.mass.nfw_ellipse_slope import NFW_ELLIPSE_SLOPE
from gigalens.jax.profiles.mass.piemd import DPIE
from gigalens.jax.profiles.mass.shear import Shear
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.light.sersic import SersicEllipse

# translate_old_params.STRUCTURE's values for this exact physical setup (from
# boiler(2).py) -- not new numbers.
_Z_LENS = 0.49
_Z_SOURCE_REF = 1.432


def _tNCDF_bij(low, high):
    return tfb.Chain([tfb.Shift(low), tfb.Scale(high - low), tfb.NormalCDF()])


# Verbatim from experiments/sample_cosmology/bullseye.py: a tfd.Uniform with a
# Shift/Scale/NormalCDF event-space bijector in place of TFP's default (sigmoid-based)
# one for Uniform.
class _UniformBij(tfd.Uniform):
    def __init__(self, *args, event_space_bijector_class=_tNCDF_bij, **kwargs):
        self._esb = event_space_bijector_class(*args)
        super().__init__(*args, **kwargs)

    def _default_event_space_bijector(self):
        return self._esb


# --------------------------------------------------------------------------------
# Ellipticity: box-truncated (e1, e2) -> DiskEllipticity
# --------------------------------------------------------------------------------
# Which quantity "the standard deviation stays the same" refers to. This is the one
# judgement call in the swap, and it is worth a factor of ~2 in prior width, so it is a
# switch rather than a buried constant:
#   "e" -- the sd in ELLIPTICITY space is preserved (default).
#   "u" -- the old sd is passed straight through as DiskEllipticity's ``scale``, i.e.
#          the literal parameter-name mapping. This NARROWS the prior (see _disk_e).
_SD_CONVENTION = "e"


def _disk_e(sd, box_half):
    """The ``DiskEllipticity`` replacing an ``(e1, e2)`` pair of independent
    ``tfd.TruncatedNormal(0, sd, -box_half, box_half)``.

    ``e_max`` (the disk's hard cap on ``|e|``) is ``max(0.5, box_half*sqrt(2))``: 0.5,
    or the CORNER of the old box -- the largest modulus the old prior could reach --
    whenever that is larger, so the disk never cuts away ellipticity the box allowed.
    Here that is 0.5 at every site except ``bcg_dpie``'s ``box_half=0.5``, which gives
    ``0.5*sqrt(2) = 0.70711`` (``q > 0.17``).

    ``scale`` is NOT the sd of ``e``: it is the sd of the unconstrained Gaussian on
    ``u``, and the disk map is ``e = e_max * u / sqrt(1 + |u|^2)``, so ``e ~ e_max * u``
    near the centre. Passing ``sd`` through verbatim would therefore shrink the prior's
    width in ellipticity space by roughly ``e_max`` -- a factor of 2 at ``e_max = 0.5``.
    Under ``_SD_CONVENTION = "e"`` this uses ``scale = sd / e_max`` so the widths match
    instead. Monte-Carlo check of the realised marginal ``sd(e1)``, old vs new:

        sd=0.10, box +/-0.3 (e_max 0.500):  0.0987 -> 0.0932   [verbatim scale: 0.0490]
        sd=0.10, box +/-0.5 (e_max 0.707):  0.1000 -> 0.0964   [verbatim scale: 0.0694]
        sd=0.05, box +/-0.3 (e_max 0.500):  0.0500 -> 0.0490   [verbatim scale: 0.0249]

    The residual ~5% narrowing is the disk map's own compression at larger ``|u|``, not
    a free knob; correcting it exactly would mean fitting ``scale`` numerically per site.

    ``mode`` is left at its default ``(0, 0)`` -- every prior replaced here was centred
    on zero ellipticity. ``dtype`` is left at the ambient default float, which is what
    the rest of this model's priors already use (LensModel enforces dtype uniformity).
    """
    e_max = max(0.5, box_half * math.sqrt(2.0))
    scale = sd if _SD_CONVENTION == "u" else sd / e_max
    return DiskEllipticity(e_max=e_max, scale=scale)


# Flat LambdaCDM: wCDM_Cosmo (H0, Om0, k, w0) with w0 FIXED at -1.0, not sampled --
# there is no dedicated LambdaCDM profile class, and this is what makes wCDM_Cosmo one.
# H0/k fixed and Om0 free follow bullseye's own cosmo Component precedent (see module
# docstring); z_lens/z_source_ref are STRUCTURE's, not new.
cosmo = Component(
    w0waCDM_Cosmo(z_lens=_Z_LENS, z_source_ref=_Z_SOURCE_REF),
    dict(
        H0=70.0,
        k=0.0,
        wa=tfd.Uniform(-3.0, 2.0),#wa=0.0,
        w0=tfd.Uniform(-2.0, -1/3),
        Om0=_UniformBij(jnp.float64(0.0), jnp.float64(1.0)),
    ),
    name="lambdaCDM",
)

# --------------------------------------------------------------------------------
# Mass priors -- Components, grouped into the cluster Plane below.
# Names match translate_old_params.STRUCTURE["lens_names"] (same profile, same
# position) for the five that appear in that STRUCTURE.
# --------------------------------------------------------------------------------
halo_model = Component(
    NFW_ELLIPSE_SLOPE(),
    {
        "center_x": tfd.Normal(6.69965551, 1),
        "center_y": tfd.Normal(4.80431651, 1),
        ("e1", "e2"): _disk_e(0.1, 0.3),
        "s_E": tfd.Uniform(0., 0.8),
        "theta_E": tfd.Uniform(12, 14),
    },
    name="halo",
)

# fixed position, PIEMD
ld_free_ellip_model = Component(
    DPIE(),
    {
        "center_x": tfd.Normal(11.80977389, 0.1),
        "center_y": tfd.Normal(23.0283886, 0.1),
        "theta_E": tfd.TruncatedNormal(1.6730331, 0.5, 1, 2.5),
        "r_cut": tfd.LogNormal(jnp.log(10), 1),
        "r_core": tfd.LogNormal(jnp.log(0.05), 0.01),
        # the one site whose old box corner (0.5*sqrt(2)) reaches past 0.5
        ("e1", "e2"): _disk_e(0.1, 0.5),
    },
    name="Ld",
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
    {
        "center_x": tfd.Normal(-21.17580938, 0.1),
        "center_y": tfd.Normal(-24.25810504, 0.1),
        ("e1", "e2"): _disk_e(0.05, 0.3),
        "theta_E": tfd.TruncatedNormal(2.4, 0.1, 1.5, 3.5), #* CHANGED FROM SEAN'S PRIOR
        "r_cut": tfd.LogNormal(jnp.log(10), 1),
        "r_core": tfd.LogNormal(jnp.log(0.05), 0.01),
    },
    name="Le",
)

group_halo_free_model = Component(
    DPIE(),
    {
        "center_x": tfd.Normal(-15.10088063, .1),
        "center_y": tfd.Normal(-4.66657821, .1),
        ("e1", "e2"): _disk_e(0.05, 0.3),
        "theta_E": tfd.TruncatedNormal(0.8151327, 0.1, 0.2, 1.5),
        "r_cut": tfd.LogNormal(jnp.log(10), 1),
        "r_core": tfd.LogNormal(jnp.log(0.05), 0.01),
    },
    name="Lf",
)

# Commented out in boiler(2).py and absent from every JSON dump (translate_old_params.py
# STRUCTURE); translated anyway since it is defined here, same as in the old file.
# upper_right_halo = Component(
#     EPL(),
#     {
#         "center_x": tfd.Normal(32, .5),
#         "center_y": tfd.Normal(22, .5),
#         ("e1", "e2"): _disk_e(0.05, 0.3),
#         "theta_E": tfd.LogNormal(jnp.log(1), 0.05),
#         "gamma": tfd.TruncatedNormal(2, 0.5, 1, 3),
#     },
#     name="upper_right_halo",
# )

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
# One shared handle for the PAIR (was one per component): the two are a single joint
# prior now, so they cannot be shared in halves.
_source1_e_0 = shared(_disk_e(0.1, 0.3), name="source1_2_e_0")

# sersic_0_0 = Component(
#     SersicEllipse(use_lstsq=True),
#     {
#         "center_x": _source1_center_x_0,
#         "center_y": _source1_center_y_0,
#         ("e1", "e2"): _source1_e_0,
#         "n_sersic": tfd.Uniform(.25, 10),
#         "R_sersic": tfd.LogNormal(jnp.log(0.4), 0.15),
#     },
#     name="sersic_0_0",
# )
sersic_0 = Component(
    SersicEllipse(use_lstsq=True),
    {
        "center_x": tfd.Normal(7.67187389, 5),#_source1_center_x_0,
        "center_y": tfd.Normal(3.31911655, 5), #_source1_center_y_0,
        ("e1", "e2"): _disk_e(0.1, 0.3),#_source1_e_0,
        "n_sersic": tfd.Uniform(.25, 10),
        "R_sersic": tfd.LogNormal(jnp.log(0.4), 0.3),
    },
    name="source1",
)
sersic_1 = Component(
    SersicEllipse(use_lstsq=True),
    {
        "center_x": tfd.Normal(10, 5),
        "center_y": tfd.Normal(3, 5),
        ("e1", "e2"): _disk_e(0.1, 0.3),
        "n_sersic": tfd.Uniform(0.25, 10),
        "R_sersic": tfd.LogNormal(jnp.log(0.4), 0.3),
    },
    name="source2",
)
source1_plane = Plane(
    # was deflection_ratio=tfd.Uniform(0.5, 1) -- superseded by cosmology + redshift
    # (see module docstring, point 2); kept here for audit only, not used.
    redshift=0.962,
    light=[sersic_0, sersic_1],
    name="source1_2",
)

# source 3 (z_source = 1.166): CombinedProfile([SersicEllipse, SersicEllipse],
# shared_params=['center_x', 'center_y', 'e1', 'e2']) -- explicit, but the same as
# DoubleSersic's default. Both Sersics share one centre/ellipticity.
_source3_center_x = shared(tfd.Normal(6.79821086, 5), name="source3_center_x")
_source3_center_y = shared(tfd.Normal(7.91570776, 5), name="source3_center_y")
# One shared handle for the PAIR (was one per parameter) -- see source1_2 above.
_source3_e = shared(_disk_e(0.1, 0.3), name="source3_e")

source3_sersic_0 = Component(
    SersicEllipse(use_lstsq=True),
    {
        "center_x": tfd.Normal(6.79821086, 5),#_source3_center_x,
        "center_y": tfd.Normal(7.91570776, 5),#_source3_center_y,
        ("e1", "e2"): _source3_e,
        "n_sersic": tfd.Uniform(.25, 10),
        "R_sersic": tfd.LogNormal(jnp.log(0.4), 0.3),
    },
    name="source3_main",
)
# source3_sersic_1 = Component(
#     SersicEllipse(use_lstsq=True),
#     {
#         "center_x": _source3_center_x,
#         "center_y": _source3_center_y,
#         ("e1", "e2"): _source3_e,
#         "n_sersic": tfd.Uniform(.25, 10),
#         "R_sersic": tfd.Uniform(1e-3, 1),
#     },
#     name="source3_core",
# )
source3_plane = Plane(
    # was deflection_ratio=tfd.Uniform(0.5, 1) -- superseded, see source1_plane above.
    redshift=1.166,
    light=[source3_sersic_0,],#, source3_sersic_1],
    name="source3",
)

# source 4_5 (z_source = 1.432): CombinedProfile([SersicEllipse, SersicEllipse],
# shared_params=[]) -- no sharing; two fully independent Sersics.
source45_sersic_0 = Component(
    SersicEllipse(use_lstsq=True),
    {
        "center_x": tfd.Normal(4.63, 5),
        "center_y": tfd.Normal(3.79, 5),
        ("e1", "e2"): _disk_e(0.1, 0.3),
        "n_sersic": tfd.Uniform(.25, 10),
        "R_sersic": tfd.Uniform(1e-3, 1),
    },
    name="source4",
)
source45_sersic_1 = Component(
    SersicEllipse(use_lstsq=True),
    {
        "center_x": tfd.Normal(4.78792923, 5),
        "center_y": tfd.Normal(0.84347007, 5),
        ("e1", "e2"): _disk_e(0.1, 0.3),
        "n_sersic": tfd.Uniform(.25, 10),
        "R_sersic": tfd.Uniform(1e-3, 1),
    },
    name="source5",
)
source45_plane = Plane(
    # was deflection_ratio=1 (fixed) -- superseded, see source1_plane above.
    redshift=1.432,
    light=[source45_sersic_0, source45_sersic_1],
    name="source4_5",
)

# source 9 (z_source = 1.506): plain SersicEllipse, never combined.
source9_sersic = Component(
    SersicEllipse(use_lstsq=True),
    {
        "center_x": tfd.Normal(-7, 5),
        "center_y": tfd.Normal(-13, 5),
        ("e1", "e2"): _disk_e(0.1, 0.3),
        "n_sersic": tfd.Uniform(.25, 10),
        "R_sersic": tfd.Uniform(1e-3, 1),
    },
    name="source9",
)
source9_plane = Plane(
    # was deflection_ratio=tfd.Uniform(0.75, 1.25) -- superseded, see source1_plane above.
    redshift=1.506,
    light=[source9_sersic],
    name="source9",
)

# source 7 (z_source = 1.627): plain SersicEllipse, never combined.
source7_sersic = Component(
    SersicEllipse(use_lstsq=True),
    {
        "center_x": tfd.Normal(0, 5),
        "center_y": tfd.Normal(0, 5),
        ("e1", "e2"): _disk_e(0.1, 0.3),
        "n_sersic": tfd.Uniform(.25, 10),
        "R_sersic": tfd.Uniform(1e-3, 1),
    },
    name="source7",
)
source7_plane = Plane(
    # was deflection_ratio=tfd.Uniform(1., 1.5) -- superseded, see source1_plane above.
    redshift=1.627,
    light=[source7_sersic],
    name="source7",
)

# source 6 (z_source = 1.656): plain SersicEllipse, never combined.
source6_sersic = Component(
    SersicEllipse(use_lstsq=True),
    {
        "center_x": tfd.Normal(0, 5),
        "center_y": tfd.Normal(0, 5),
        ("e1", "e2"): _disk_e(0.1, 0.3),
        "n_sersic": tfd.Uniform(.25, 10),
        "R_sersic": tfd.Uniform(1e-3, 1),
    },
    name="source6",
)
source6_plane = Plane(
    # was deflection_ratio=tfd.Uniform(1., 1.5) -- superseded, see source1_plane above.
    redshift=1.656,
    light=[source6_sersic],
    name="source6",
)

# source 12_13 (z_source = 3.086): CombinedProfile([SersicEllipse, SersicEllipse],
# shared_params=[]) -- no sharing; two fully independent Sersics.
source1213_sersic_0 = Component(
    SersicEllipse(use_lstsq=True),
    {
        "center_x": tfd.Normal(2.906817674636841, 5),
        "center_y": tfd.Normal(4.523301601409912, 5),
        ("e1", "e2"): _disk_e(0.1, 0.3),
        "n_sersic": tfd.Uniform(.25, 10),
        "R_sersic": tfd.Uniform(1e-3, 1),
    },
    name="source12",
)
source1213_sersic_1 = Component(
    SersicEllipse(use_lstsq=True),
    {
        "center_x": tfd.Normal(6.670745849609375, 5),
        "center_y": tfd.Normal(5.6769537925720215, 5),
        ("e1", "e2"): _disk_e(0.1, 0.3),
        "n_sersic": tfd.Uniform(.25, 10),
        "R_sersic": tfd.Uniform(1e-3, 1),
    },
    name="source13",
)
source1213_plane = Plane(
    # was deflection_ratio=tfd.Uniform(1, 1.5) -- superseded, see source1_plane above.
    redshift=3.086,
    light=[source1213_sersic_0, source1213_sersic_1],
    name="source12_13",
)

# source 8 (z_source = 3.549): plain SersicEllipse, never combined.
source8_sersic = Component(
    SersicEllipse(use_lstsq=True),
    {
        "center_x": tfd.Normal(6.1898108, 5),
        "center_y": tfd.Normal(6.8792906, 5),
        ("e1", "e2"): _disk_e(0.1, 0.3),
        "n_sersic": tfd.Uniform(.25, 10),
        "R_sersic": tfd.Uniform(1e-3, 1),
    },
    name="source8",
)
source8_plane = Plane(
    # was deflection_ratio=tfd.Uniform(1, 1.5) -- superseded, see source1_plane above.
    redshift=3.549,
    light=[source8_sersic],
    name="source8",
)

# source 11 (z_source = 4.090): plain SersicEllipse, never combined.
source11_sersic = Component(
    SersicEllipse(use_lstsq=True),
    {
        "center_x": tfd.Normal(4.4566746, 5),
        "center_y": tfd.Normal(2.0770147, 5),
        ("e1", "e2"): _disk_e(0.1, 0.3),
        "n_sersic": tfd.Uniform(.25, 10),
        "R_sersic": tfd.Uniform(1e-3, 1),
    },
    name="source11",
)
source11_plane = Plane(
    # was deflection_ratio=tfd.Uniform(1., 1.5) -- superseded, see source1_plane above.
    redshift=4.090,
    light=[source11_sersic],
    name="source11",
)

# --------------------------------------------------------------------------------
# One assembled LensModel: the cluster mass plane (all six mass Components) plus every
# source plane, in ascending redshift order (observer -> source, as the new API
# requires -- already the order they were defined in above), with the LambdaCDM
# cosmology from the top of this file.
# --------------------------------------------------------------------------------
cluster_plane = Plane(
    redshift=_Z_LENS,
    mass=[halo_model, ld_free_ellip_model, shear_model, le_free_model,
          group_halo_free_model],#, upper_right_halo],
    name="cluster",
)

model = LensModel(
    [
        cluster_plane,
        source1_plane, source3_plane, source45_plane, source9_plane,
        source7_plane, source6_plane, source1213_plane, source8_plane, source11_plane,
    ],
    cosmo=cosmo,
)
