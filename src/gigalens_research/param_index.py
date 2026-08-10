"""The parameter index: one record per free parameter, in the scene's own path space.

Every free parameter of a scene-backed model has a *site path* — the place it
lives in the physical model — and the scene already knows it. Paths are tuples of
**strings**, and they are the params-tree keys::

    ("planes", "lens", "mass", "host", "theta_E")  -> plane "lens", mass "host"
    ("planes", "1", "light", "0", "R_sersic")      -> plane 1, light component 0
    ("planes", "1", "geometry", "redshift")        -> plane 1's geometry
    ("cosmo", "H0")                                -> cosmology

so ``kind`` / ``plane`` / ``component`` are read off the path rather than
re-derived. :func:`param_sites` walks ``LensModel._site_to_unique`` (the
authoritative ``(path, unique_key, component_index)`` list) and returns one
:class:`ParamSite` per free parameter, in a canonical order.

Keys, names and indices
-----------------------
A plane/component segment is the scene's ``name`` when it has one and ``str(index)``
when it does not (gigalens ``scene.py`` §names, commit ``cc5a078``). Every segment is
a ``str`` either way — never a bare ``int`` — because a dict mixing ``int`` and ``str``
keys cannot be flattened by JAX at all.

So a path segment is an *identity*, not a position, and the two must not be confused:

- :attr:`ParamSite.plane_keys` / :attr:`ParamSite.component_keys` are the raw key
  segments — what indexes a params tree.
- :attr:`ParamSite.planes` / :attr:`ParamSite.components` are integer **positions** in
  ``scene.planes``. They are resolved once, at index time, because a
  :class:`ParamSite` outlives the scene that could resolve them.

Both are selectable: ``select_sites(plane=1)`` and ``select_sites(plane="lens")`` are
equally valid. Positions are what ordering uses — sorting on the key text would put
plane ``"10"`` before ``"2"`` and would shuffle panels whenever a plane is renamed.

Why this module exists
----------------------
This replaces the older "3-group prefixed label space" (``theta_E__1``,
``lens_R_sersic__0``, ``src_Ie``, ``cosmo_H0``) that ``Posterior.grouped_free_x``
used to produce. That space was lossy in two ways that mattered:

- **Plane identity was destroyed.** Its ``__<i>`` suffix was a *global running
  index* across every plane, so ``theta_E__1`` could be plane 0's second mass
  component or plane 1's first — indistinguishable. Selecting by plane was
  therefore impossible downstream.
- **The lens/source light split was invented.** A :class:`Plane` carries only
  ``mass`` and ``light``; there is no lens-light/source-light distinction in the
  model. The old space synthesized one via "light is source light iff an earlier
  plane has mass" and baked it into labels (``$R_l$`` vs ``$R_s$``). Plane index
  is the real distinction, and it is what these records carry.

Identity vs display
-------------------
Two different strings, deliberately:

- :attr:`ParamSite.key` — the full path (``"planes/1/mass/0/theta_E"``). Stable,
  unambiguous, greppable. This is what ``plot_params=`` matches on.
- :func:`site_labels` — what a plot axis shows. Bare (``$\\theta_E$``) when the
  parameter is unique in the figure, disambiguated (``$\\theta_E^{(1,0)}$``) only
  when two columns would otherwise collide.

Shared parameters
-----------------
A ``shared()`` parameter is ONE free parameter feeding several sites, and is
counted once here (matching ``LensModel.num_free_params``) rather than fanned
out into duplicate, perfectly-correlated columns. Its :attr:`ParamSite.paths`
holds every site it feeds, and it matches a ``plane=`` selection if *any* of
those sites is on that plane.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = [
    "KINDS",
    "LATEX_LABELS",
    "ParamSite",
    "kind_of_key",
    "latex_label",
    "param_of_key",
    "param_sites",
    "path_of_key",
    "select_sites",
    "site_labels",
    "sites_to_matrix",
    "truth_row",
]


# ---------------------------------------------------------------------------
# Kinds and ordering
# ---------------------------------------------------------------------------

KIND_COSMOLOGY = "cosmology"
KIND_GEOMETRY = "geometry"
KIND_MASS = "mass"
KIND_LIGHT = "light"

#: The parameter classes, in canonical panel order: cosmology, then the plane
#: geometry it acts through (redshift / deflection_ratio), then mass, then light.
KINDS: Tuple[str, ...] = (KIND_COSMOLOGY, KIND_GEOMETRY, KIND_MASS, KIND_LIGHT)

_KIND_ORDER = {kind: i for i, kind in enumerate(KINDS)}

#: Accepted spellings for ``kind=`` selections.
_KIND_ALIASES = {
    "cosmology": KIND_COSMOLOGY,
    "cosmo": KIND_COSMOLOGY,
    "geometry": KIND_GEOMETRY,
    "geom": KIND_GEOMETRY,
    "mass": KIND_MASS,
    "light": KIND_LIGHT,
}


# ---------------------------------------------------------------------------
# LaTeX registry
# ---------------------------------------------------------------------------

#: LaTeX rendering keyed by the *bare* parameter name, as it appears in a
#: profile's ``_params`` (the last segment of a site path).
#:
#: Deliberately profile-agnostic: which component a parameter belongs to is
#: carried by the ``(plane, component)`` disambiguation suffix, not smuggled
#: into the symbol. The one exception is external shear, whose ``gamma1`` /
#: ``gamma2`` keep an explicit ``ext`` marker — without it they read as
#: components of the EPL slope ``gamma``, which is a different quantity.
LATEX_LABELS: Dict[str, str] = {
    # Mass — EPL / SIE / SIS
    "theta_E": r"$\theta_E$",
    "gamma": r"$\gamma$",
    "e1": r"$\epsilon_1$",
    "e2": r"$\epsilon_2$",
    "center_x": r"$x$",
    "center_y": r"$y$",
    # Mass — external shear
    "gamma1": r"$\gamma_{\rm ext,1}$",
    "gamma2": r"$\gamma_{\rm ext,2}$",
    # Mass — NFW / tNFW
    "Rs": r"$R_{\rm s}$",
    "alpha_Rs": r"$\alpha_{R_{\rm s}}$",
    "r_trunc": r"$r_{\rm trunc}$",
    # Mass — PIEMD / dPIE
    "r_core": r"$r_{\rm core}$",
    "r_cut": r"$r_{\rm cut}$",
    "Ra": r"$R_a$",
    "s_E": r"$s_E$",
    # Light — Sersic
    "R_sersic": r"$R_{\rm sersic}$",
    "n_sersic": r"$n_{\rm sersic}$",
    "Ie": r"$I_e$",
    # Light — shapelets
    "beta": r"$\beta$",
    # Geometry
    "redshift": r"$z$",
    "deflection_ratio": r"$\beta_{\rm defl}$",
    # Cosmology (see gigalens.jax.cosmo)
    "H0": r"$H_0$",
    "Om0": r"$\Omega_m$",
    "k": r"$\Omega_k$",
    "w0": r"$w_0$",
    "wa": r"$w_a$",
}


def latex_label(param: str, *, fallback: Optional[str] = None) -> str:
    """LaTeX for a bare parameter name, else ``fallback`` (default: the name).

    Unknown names (e.g. shapelet coefficients) pass through readably rather than
    raising — a corner plot of an unregistered profile should still draw.
    """
    return LATEX_LABELS.get(param, param if fallback is None else fallback)


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSite:
    """One free parameter of a scene, located in the model.

    Attributes
    ----------
    ukey : str
        The scene's unique key for this parameter — the key into a bijector
        output dict (``planes/lens/mass/host/theta_E``, ``shared_7``, ``coupled_3``,
        or a pipe-joined grouped-prior key).
    cidx : int or None
        Which component of a grouped / coupled prior's vector value this is
        (e.g. ``e1`` = 0, ``e2`` = 1 of a ``DiskEllipticity``). ``None`` for an
        ordinary scalar parameter.
    kind : str
        One of :data:`KINDS`.
    param : str
        Bare parameter name, e.g. ``theta_E``.
    paths : tuple of tuple of str
        Every scene site this free parameter feeds, as all-string path tuples.
        Length > 1 exactly when it is a ``shared()`` parameter.
    key : str
        Canonical stable identity: the primary site path, slash-joined.
    plane_tags : tuple of (int, str)
        ``(plane index, display tag)`` for every plane this parameter acts on. The
        tag is the scene's ``Plane(name=...)`` when it has one, else the index as a
        string. Resolved at index time because a ``ParamSite`` outlives the scene.
    comp_tags : tuple of (str, int, str)
        ``(role, index, display tag)`` for every component this parameter acts on.
        The tag is ``m1``/``l0`` when unnamed — unchanged — and ``m:host``/``l:host``
        when the scene names it. The role letter survives naming because Component
        names are unique only per *(plane, kind)*: a plane may hold a mass ``host``
        AND a light ``host``, and only the letter tells those apart.
    plane_keys : tuple of str
        The raw params-tree key of every plane this parameter acts on — the name when
        named, ``str(index)`` when not. What indexes a params dict, as against
        :attr:`planes`, which is the position in ``scene.planes``.
    component_keys : tuple of (str, str)
        ``(role, raw key)`` for every component this parameter acts on. Carries the
        role for the same reason :attr:`comp_tags` does: names are unique only per
        *(plane, kind)*, so ``host`` alone does not identify a component.
    group_name : str or None
        Display name of the ``shared()`` / ``coupled()`` handle behind this column,
        when it has one. A shared parameter spans several sites, so it has no single
        site to read a label from — this is the only name it can have.
    """

    ukey: str
    cidx: Optional[int]
    kind: str
    param: str
    paths: Tuple[tuple, ...]
    key: str
    # Positions and display tags are resolved against the scene at index time; a
    # ParamSite outlives the scene, so they cannot be recovered from the path later.
    plane_tags: Tuple[Tuple[int, str], ...] = ()
    comp_tags: Tuple[Tuple[str, int, str], ...] = ()
    plane_keys: Tuple[str, ...] = ()
    component_keys: Tuple[Tuple[str, str], ...] = ()
    group_name: Optional[str] = None

    @property
    def shared(self) -> bool:
        """True when one free parameter feeds several sites."""
        return len(self.paths) > 1

    @property
    def planes(self) -> FrozenSet[int]:
        """Every plane INDEX this parameter acts on (empty for cosmology).

        Read from :attr:`plane_tags` rather than from the path: a path segment is a
        key (a name, or ``str(index)``), and only the scene could say which position
        a name sits at — by the time anyone asks, it is gone.
        """
        return frozenset(i for i, _tag in self.plane_tags)

    @property
    def plane(self) -> Optional[int]:
        """The plane index, when unambiguous; ``None`` for cosmology or for a shared
        parameter spanning several planes (see :attr:`planes`)."""
        pl = self.planes
        return next(iter(pl)) if len(pl) == 1 else None

    @property
    def components(self) -> FrozenSet[Tuple[str, int]]:
        """Every ``(role, index)`` component this parameter acts on, where role
        is ``"mass"`` or ``"light"``. Empty for cosmology and geometry."""
        return frozenset((role, j) for role, j, _tag in self.comp_tags)

    @property
    def component(self) -> Optional[int]:
        """The component index within its ``(plane, role)``, when unambiguous."""
        cs = {idx for _role, idx in self.components}
        return next(iter(cs)) if len(cs) == 1 else None


# ---------------------------------------------------------------------------
# Building the index
# ---------------------------------------------------------------------------


def _plane_name(scene: Any, i: int) -> Optional[str]:
    """Plane ``i``'s scene name, or ``None``."""
    try:
        return getattr(scene.planes[i], "name", None) or None
    except (AttributeError, IndexError, TypeError):
        return None


def _comp_name(scene: Any, i: int, role: str, j: int) -> Optional[str]:
    """Component ``j``'s scene name within plane ``i``'s ``role`` list, or ``None``."""
    try:
        comps = scene.planes[i].mass if role == KIND_MASS else scene.planes[i].light
        return getattr(comps[j], "name", None) or None
    except (AttributeError, IndexError, TypeError):
        return None


def _plane_tag(scene: Any, i: int) -> str:
    """A plane's display tag: its name, else its index (§names)."""
    name = _plane_name(scene, i)
    return name if name else str(i)


def _comp_tag(scene: Any, i: int, role: str, j: int) -> str:
    """A component's display tag: ``m1``/``l0`` unnamed, ``m:host`` named.

    The role letter is kept in both forms. Scene Component names are unique only per
    *(plane, kind)*, so a mass ``host`` and a light ``host`` — one galaxy's two
    aspects — are distinguished by the letter alone.
    """
    name = _comp_name(scene, i, role, j)
    return f"{role[0]}:{name}" if name else f"{role[0]}{j}"


# -- params-tree keys ---------------------------------------------------------------
# The scene owns this mapping (``LensModel.plane_key`` / ``component_key``), so ask it
# rather than re-deriving. The fallback is ``str(index)``, NOT the name: a scene old
# enough to lack these accessors also predates names in keys, and keyed its paths
# positionally. Guessing the name there would build a lookup table that matches no
# path the scene actually emits.


def _plane_key(scene: Any, i: int) -> str:
    """Plane ``i``'s params-tree key."""
    fn = getattr(scene, "plane_key", None)
    if fn is not None:
        return str(fn(i))
    return str(i)


def _component_key(scene: Any, i: int, role: str, j: int) -> str:
    """Component ``j``'s params-tree key within plane ``i``'s ``role`` list."""
    fn = getattr(scene, "component_key", None)
    if fn is not None:
        return str(fn(i, role, j))
    return str(j)


def _index_maps(scene: Any) -> Tuple[Dict[str, int], Dict[Tuple[str, str, str], int]]:
    """``(plane key -> index, (plane key, role, component key) -> index)``.

    The inverse of the scene's own key derivation, built once per index so that every
    site path can be resolved back to the positions ordering and selection need.
    """
    planes: Dict[str, int] = {}
    comps: Dict[Tuple[str, str, str], int] = {}
    for i, plane in enumerate(getattr(scene, "planes", ()) or ()):
        pk = _plane_key(scene, i)
        planes[pk] = i
        for role in (KIND_MASS, KIND_LIGHT):
            for j, _c in enumerate(getattr(plane, role, ()) or ()):
                comps[(pk, role, _component_key(scene, i, role, j))] = j
    return planes, comps


def _as_path(path: Any) -> tuple:
    """Normalize a scene site path to an all-``str`` tuple.

    Scenes predating ``cc5a078`` emitted positional ``int`` segments; normalizing here
    means the rest of this module has exactly one path space to reason about, and
    ``path_of_key`` stays an exact inverse of :attr:`ParamSite.key` in both eras.
    """
    return tuple(str(seg) for seg in path)


def _group_name(scene: Any, ukey: str) -> Optional[str]:
    """The ``shared()``/``coupled()`` handle name behind ``ukey``, if the scene records
    one. Older scenes have no ``_unique_handles``; absence is not an error."""
    for uk, handle in getattr(scene, "_unique_handles", ()) or ():
        if uk == ukey:
            n = getattr(handle, "name", None)
            if n:
                return n
    return None


def _classify(path: tuple) -> Optional[Tuple[str, str]]:
    """``(kind, param)`` for a site path, or ``None`` if it is not a parameter
    we index (an unrecognized or non-profile site)."""
    if not path:
        return None
    if path[0] == "cosmo" and len(path) == 2:
        return KIND_COSMOLOGY, path[1]
    if path[0] == "planes" and len(path) == 4 and path[2] == "geometry":
        return KIND_GEOMETRY, path[3]
    if path[0] == "planes" and len(path) == 5 and path[2] in (KIND_MASS, KIND_LIGHT):
        return path[2], path[4]
    return None


def path_of_key(key: str) -> tuple:
    """The site path behind a slash-joined key: the exact inverse of
    :attr:`ParamSite.key`.

    Every segment stays a ``str``. An all-digit segment is a positional key, not an
    ``int`` to be recovered: params-tree keys are always strings (a dict mixing
    ``int`` and ``str`` keys cannot be flattened by JAX), and ``int("0") == 0`` would
    produce a tuple that indexes no params tree and compares equal to no scene path.
    """
    return tuple(key.split("/"))


def kind_of_key(key: str) -> Optional[str]:
    """The :data:`KINDS` entry a path key belongs to, or ``None`` if it isn't a
    parameter path (a synthetic ``shared_7`` key, say).

    For classifying label strings when no scene is at hand. Prefer
    :attr:`ParamSite.kind` when you have the records.
    """
    spec = _classify(path_of_key(key))
    return None if spec is None else spec[0]


def param_of_key(key: str) -> str:
    """The bare parameter name at the end of a path key (``theta_E``)."""
    return key.rsplit("/", 1)[-1]


def _sort_key(site: ParamSite) -> tuple:
    planes = site.planes
    comps = {idx for _role, idx in site.components}
    return (
        _KIND_ORDER.get(site.kind, len(KINDS)),
        min(planes) if planes else -1,
        min(comps) if comps else -1,
    )


def _scene_of(source: Any):
    """Accept a ``LensModel`` directly, or any Posterior-like object exposing
    one, so callers can pass whichever they have."""
    for attr in ("scene", "_scene_model"):
        scene = getattr(source, attr, None)
        if scene is not None:
            return scene
    if hasattr(source, "_site_to_unique"):
        return source
    raise TypeError(
        f"expected a scene LensModel or a scene-backed Posterior; got "
        f"{type(source).__name__}, which exposes neither `.scene` nor "
        "`._site_to_unique`. Legacy (non-scene) models have no path space and "
        "cannot be indexed."
    )


def param_sites(source: Any) -> List[ParamSite]:
    """Index every free parameter of a scene, in canonical order.

    ``source`` may be a scene ``LensModel`` or a scene-backed ``Posterior``.

    Order is cosmology, geometry, mass, light; then by plane, then by component;
    then the model's own definition order within a component (a stable sort over
    ``_site_to_unique``, which is built in definition order). Constants are not
    parameters and never appear.
    """
    scene = _scene_of(source)
    plane_at, comp_at = _index_maps(scene)

    grouped: Dict[Tuple[str, Optional[int]], List[tuple]] = {}
    order: List[Tuple[str, Optional[int]]] = []
    for path, ukey, cidx in scene._site_to_unique:
        # A shared() parameter appears once per site it feeds but is a single free
        # parameter, so collapse on (ukey, cidx). Grouped and coupled priors give
        # each of their vector components a distinct cidx, so those stay separate
        # columns — which is what we want.
        ident = (ukey, cidx)
        if ident not in grouped:
            grouped[ident] = []
            order.append(ident)
        grouped[ident].append(_as_path(path))

    sites: List[ParamSite] = []
    for ukey, cidx in order:
        paths = tuple(grouped[(ukey, cidx)])
        spec = _classify(paths[0])
        if spec is None:
            continue
        kind, param = spec
        # A path whose plane/component key is absent from the scene's own key maps is
        # dropped from the tags rather than guessed at. That is unreachable for a
        # scene that built these paths, and a silent wrong index would be worse than a
        # missing one: it would put a parameter on the wrong panel.
        plane_keys = tuple(sorted({
            p[1] for p in paths if p and p[0] == "planes"
        }))
        plane_tags = tuple(sorted({
            (plane_at[p[1]], _plane_tag(scene, plane_at[p[1]]))
            for p in paths if p and p[0] == "planes" and p[1] in plane_at
        }))
        component_keys = tuple(sorted({
            (p[2], p[3]) for p in paths
            if len(p) == 5 and p[0] == "planes" and p[2] in (KIND_MASS, KIND_LIGHT)
        }))
        comp_tags = tuple(sorted({
            (p[2],
             comp_at[(p[1], p[2], p[3])],
             _comp_tag(scene, plane_at[p[1]], p[2], comp_at[(p[1], p[2], p[3])]))
            for p in paths
            if len(p) == 5 and p[0] == "planes" and p[2] in (KIND_MASS, KIND_LIGHT)
            and p[1] in plane_at and (p[1], p[2], p[3]) in comp_at
        }))
        sites.append(
            ParamSite(
                ukey=ukey,
                cidx=cidx,
                kind=kind,
                param=param,
                paths=paths,
                key="/".join(paths[0]),
                plane_tags=plane_tags,
                comp_tags=comp_tags,
                plane_keys=plane_keys,
                component_keys=component_keys,
                group_name=_group_name(scene, ukey),
            )
        )
    sites.sort(key=_sort_key)  # stable: definition order survives within a component
    return sites


# ---------------------------------------------------------------------------
# Display labels
# ---------------------------------------------------------------------------


def _suffix(site: ParamSite) -> str:
    """The tag used to break a label collision: plane, then component as
    ``m<i>`` / ``l<i>``, e.g. ``0,m1`` for plane 0's mass component 1. Where the
    scene names a plane or component, its name replaces the index: ``lens,m:host``.

    The role letter is not decoration, named or not. Mass and light components are
    indexed separately within a plane, so plane 0's mass[0] and plane 0's light[0] are
    both "component 0" — and since ``e1``/``e2``/``center_x``/``center_y`` are
    parameter names common to mass *and* light profiles, a bare ``(plane,
    component)`` tag collides on exactly the parameters most likely to need it. Names
    do not retire the problem: they are unique only per (plane, kind), precisely so
    that one galaxy's mass and light may both be ``host``.

    A shared parameter spanning several sites gets them joined (``0+1``), which
    is the honest reading: it is one parameter acting in both.
    """
    parts: List[str] = []
    if site.plane_tags:
        # Sorted by plane INDEX, not by tag text, so renaming a plane cannot reorder a
        # shared parameter's "0+1" tag.
        parts.append("+".join(tag for _i, tag in sorted(site.plane_tags)))
    if site.comp_tags:
        # Sort by (index, role) exactly as before, so an unnamed model's tags are
        # byte-for-byte unchanged; only the rendered tag substitutes a name.
        comps = sorted(site.comp_tags, key=lambda t: (t[1], t[0]))
        parts.append("+".join(tag for _r, _i, tag in comps))
    return ",".join(parts)


def _mathtext_escape(s: str) -> str:
    """Escape a name for math mode.

    An unescaped ``_`` is the subscript operator, so a user's ``host_light`` renders
    as ``host`` subscript ``light`` — silently, without raising. Measured: the
    unescaped tag renders 142.8px wide against the escaped 150.6px. Names like
    ``main_deflector`` are the norm, so this is the common case, not an edge one.
    """
    return s.replace("\\", r"\backslash ").replace("_", r"\_")


def _decorate(base: str, site: ParamSite, *, latex: bool) -> str:
    tag = _suffix(site)
    if not tag:
        return base
    if latex and base.startswith("$") and base.endswith("$"):
        return f"${base[1:-1]}^{{({_mathtext_escape(tag)})}}$"
    return f"{base} ({tag})"


def site_labels(sites: Sequence[ParamSite], *, latex: bool = True) -> List[str]:
    """Display labels for ``sites``, disambiguated only where needed.

    A parameter that is unique among ``sites`` renders bare (``$\\theta_E$``);
    one that would collide with another gains its ``(plane, component)``
    superscript (``$\\theta_E^{(1,0)}$``). So the common single-plane model keeps
    clean axes, and a multiplane one stays unambiguous.

    Note this depends on the *selection*: plotting only plane 1 may render a
    parameter bare that would have been suffixed in the full-model plot. That is
    the intent — the suffix exists to separate columns, and there is nothing to
    separate from when the collision isn't on the figure.
    """
    bases = [latex_label(s.param) if latex else s.param for s in sites]
    collisions = {b for b, n in Counter(bases).items() if n > 1}
    labels = [
        _decorate(b, s, latex=latex) if b in collisions else b
        for s, b in zip(sites, bases)
    ]
    # Belt and braces: if a suffix still didn't separate two columns (e.g. two
    # shared parameters over the same planes), fall back to the full path, which
    # is unique by construction.
    still = {l for l, n in Counter(labels).items() if n > 1}
    return [s.key if l in still else l for s, l in zip(sites, labels)]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _as_set(value: Any) -> frozenset:
    """One-or-many normalization. Strings are scalars, not sequences of
    characters; numpy arrays are sequences, though they are not
    ``collections.abc.Sequence``."""
    if isinstance(value, (str, bytes)):
        return frozenset([value])
    if isinstance(value, np.ndarray):
        return frozenset(value.tolist())
    if isinstance(value, (list, tuple, set, frozenset)):
        return frozenset(value)
    return frozenset([value])


def _normalize_planes(plane: Any) -> Tuple[frozenset, frozenset]:
    """Split a ``plane=`` selection into integer indices and string keys.

    Both are accepted because both are meaningful: ``plane=1`` is the second plane
    whatever it is called, ``plane="lens"`` is that particular object wherever it sits.
    An all-digit *string* is treated as a key, not an index — the scene rejects
    all-digit names precisely so that ``"0"`` can only ever be the positional key of
    plane 0, which makes the two readings agree there anyway.
    """
    idxs, keys = set(), set()
    for p in _as_set(plane):
        if isinstance(p, bool):
            raise TypeError(f"plane must be an int or a name; got {p!r}.")
        if isinstance(p, (int, np.integer)):
            idxs.add(int(p))
        elif isinstance(p, str):
            keys.add(p)
        else:
            raise TypeError(
                f"plane must be an int (index), a str (name), or a list of those; "
                f"got {p!r}."
            )
    return frozenset(idxs), frozenset(keys)


def _normalize_kinds(kind: Any) -> frozenset:
    out = set()
    for k in _as_set(kind):
        if not isinstance(k, str):
            raise TypeError(f"kind must be a string or list of strings; got {k!r}.")
        norm = _KIND_ALIASES.get(k.lower())
        if norm is None:
            raise ValueError(
                f"unknown kind {k!r}; expected one of "
                f"{', '.join(repr(x) for x in KINDS)}."
            )
        out.add(norm)
    return frozenset(out)


def _normalize_components(
    component: Any,
) -> Tuple[frozenset, frozenset, frozenset, frozenset]:
    """Split a ``component=`` selection into four buckets.

    ``(bare indices, (role, index) pairs, bare names, (role, name) pairs)``. Accepts
    ``0``, ``[0, 1]``, ``("mass", 0)``, ``"host"``, ``("light", "host")``, or a list
    mixing them — a name is as valid an identifier as an index, and the role-pair form
    matters more for names than for indices, since ``host`` may well be both a mass
    and a light.
    """
    if _is_role_pair(component):
        raw = [component]  # a lone ("mass", 0) is one pair, not two selections
    elif isinstance(component, (list, tuple)):
        raw = list(component)
    else:
        raw = [component]
    idxs, pairs, names, name_pairs = set(), set(), set(), set()
    for item in raw:
        if _is_role_pair(item):
            role, ident = item
            role = role.lower()
            if role not in (KIND_MASS, KIND_LIGHT):
                raise ValueError(
                    f"component role must be 'mass' or 'light'; got {role!r}. (A "
                    f"2-tuple is read as a (role, component) pair — to select several "
                    f"components by name, pass a list: component=['a', 'b'].)"
                )
            if isinstance(ident, str):
                name_pairs.add((role, ident))
            else:
                pairs.add((role, int(ident)))
        elif isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            idxs.add(int(item))
        elif isinstance(item, str):
            names.add(item)
        else:
            raise TypeError(
                f"component must be an int (index), a str (name), a (role, index) or "
                f"(role, name) pair, or a list of those; got {item!r}."
            )
    return frozenset(idxs), frozenset(pairs), frozenset(names), frozenset(name_pairs)


def _is_role_pair(item: Any) -> bool:
    return (
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], str)
        and isinstance(item[1], (int, np.integer, str))
        and not isinstance(item[1], bool)
    )


def select_sites(
    sites: Sequence[ParamSite],
    *,
    kind: Any = None,
    plane: Any = None,
    component: Any = None,
    select: Optional[Callable[[ParamSite], bool]] = None,
) -> List[ParamSite]:
    """Subset ``sites``. Every supplied filter must match (they AND together);
    omitting all of them returns everything, which is the default everywhere.

    Parameters
    ----------
    kind : str or list of str, optional
        One or more of :data:`KINDS` (``"cosmo"`` and ``"geom"`` also accepted).
    plane : int, str, or list, optional
        Plane index (``1``) or scene name / params-tree key (``"lens"``). A shared
        parameter matches if *any* of its sites is on a selected plane.
    component : int, str, (role, index), (role, name), or list, optional
        Component index or name within its ``(plane, role)``. Pass ``("mass", 0)`` or
        ``("light", "host")`` to pin the role, or combine a bare index/name with
        ``kind=``/``plane=``. Pinning the role matters for names: one galaxy's mass
        and light may both be called ``host``.
    select : callable, optional
        Escape hatch: a predicate on :class:`ParamSite` for anything the keyword
        filters don't express, e.g. ``select=lambda s: s.param.startswith("e")``.

    Raises
    ------
    ValueError
        If the filters match nothing — an empty corner plot is never what the
        caller wanted, and a silent blank figure hides the typo.
    """
    out = list(sites)
    if kind is not None:
        wanted = _normalize_kinds(kind)
        out = [s for s in out if s.kind in wanted]
    if plane is not None:
        wanted_idxs, wanted_keys = _normalize_planes(plane)
        out = [
            s for s in out
            if (s.planes & wanted_idxs) or (frozenset(s.plane_keys) & wanted_keys)
        ]
    if component is not None:
        idxs, pairs, names, name_pairs = _normalize_components(component)
        out = [
            s
            for s in out
            if ({i for _r, i in s.components} & idxs)
            or (s.components & pairs)
            or ({n for _r, n in s.component_keys} & names)
            or (frozenset(s.component_keys) & name_pairs)
        ]
    if select is not None:
        out = [s for s in out if select(s)]

    if not out:
        raise ValueError(
            "no parameters matched the selection "
            f"(kind={kind!r}, plane={plane!r}, component={component!r}"
            f"{', select=<callable>' if select is not None else ''}). "
            f"{_available(sites)}"
        )
    return out


def _available(sites: Sequence[ParamSite]) -> str:
    kinds = sorted({s.kind for s in sites}, key=lambda k: _KIND_ORDER.get(k, 99))
    # Report each plane as index and key together: a name-based selection that missed
    # is otherwise undiagnosable from a list of bare indices.
    planes = sorted({(i, tag) for s in sites for i, tag in s.plane_tags})
    shown = [i if str(i) == tag else f"{i} ({tag!r})" for i, tag in planes]
    comps = sorted({n for s in sites for _r, n in s.component_keys if not n.isdigit()})
    return (
        f"This model has kind={kinds} and plane=[{', '.join(map(str, shown))}]"
        + (f"; named components {comps}" if comps else "")
        + f"; {len(sites)} free parameters in total."
    )


# ---------------------------------------------------------------------------
# Pulling values
# ---------------------------------------------------------------------------


def sites_to_matrix(sites: Sequence[ParamSite], x_flat: Dict[str, Any]) -> np.ndarray:
    """Stack ``sites`` out of a flat bijector output into an ``(n, len(sites))``
    matrix, one column per parameter, in ``sites`` order.

    ``x_flat`` is the scene bijector's ``{unique_key: array}`` dict — i.e. what
    ``Posterior.z_to_x`` returns.
    """
    cols = []
    for s in sites:
        try:
            value = x_flat[s.ukey]
        except KeyError as e:
            raise KeyError(
                f"parameter {s.key!r} (unique key {s.ukey!r}) is absent from the "
                "supplied flat params. Is this x from the same scene?"
            ) from e
        if s.cidx is not None:
            value = value[..., s.cidx]
        cols.append(np.asarray(value).reshape(-1))
    return np.vstack(cols).T


# ---------------------------------------------------------------------------
# Truth alignment
# ---------------------------------------------------------------------------

_LEGACY_GROUP_KEYS = {"lens_mass", "lens_light", "source_light"}


def _nested_get(truth: Any, path: tuple) -> Any:
    """Walk a scene-nested truth by site path. Returns ``None`` if absent.

    Path segments are strings, but a truth may have been written against an older
    scene (``{"planes": {0: ...}}``), hand-built with ``int`` keys, or stored as
    lists. All three are accepted: a truth is data the user brings, often persisted
    long before the model was rebuilt, so being liberal here costs nothing and saves
    a re-derivation.

    Names are *not* resolved to positions. A truth keyed ``{"lens": ...}`` joins a
    model whose plane is named ``lens``; against a model where that plane is unnamed
    the lookup simply misses, and :func:`truth_row` fills ``NaN`` and warns. That is
    the intended behaviour — silently pairing a truth's ``lens`` with position 0
    would be a guess, and the whole point of naming is that position and identity
    are different claims.
    """
    node = truth
    for seg in path:
        if isinstance(node, dict):
            if seg in node:
                node = node[seg]
                continue
            # A str path segment against an int-keyed truth, or vice versa.
            alt = int(seg) if isinstance(seg, str) and seg.isdigit() else str(seg)
            if alt in node:
                node = node[alt]
                continue
            return None
        elif isinstance(node, (list, tuple)):
            if not (isinstance(seg, str) and seg.isdigit()):
                return None
            idx = int(seg)
            if idx >= len(node):
                return None
            node = node[idx]
        else:
            return None
    return node


def _truth_getter(truth: Any) -> Callable[[ParamSite], Any]:
    """Resolve a truth into a per-site lookup.

    Accepts the two forms the scene API actually produces: a scene-nested point
    (``{"planes": {"lens": {"mass": {"host": {...}}}}, "cosmo": {...}}``) or a flat
    path-keyed dict (``{"planes/lens/mass/host/theta_E": ..., "cosmo/H0": ...}``,
    the idiom gigalens' own tests use).

    A truth built against a *renamed* model joins by name and so simply misses, which
    :func:`truth_row` reports as a warning rather than silently mispairing. Take the
    names from the file that defines the model.
    """
    if not isinstance(truth, dict):
        raise TypeError(
            f"truth must be a scene-nested dict or a path-keyed flat dict; got "
            f"{type(truth).__name__}."
        )
    if _LEGACY_GROUP_KEYS & set(truth):
        raise ValueError(
            "this truth is in the retired 3-group label space "
            "({'lens_mass': ..., 'lens_light': ..., 'source_light': ...}). Corner "
            "plots now work in the scene's path space: pass the scene-nested "
            "truth ({'planes': {'0': {'mass': {'0': {...}}}}, 'cosmo': {...}}) or a "
            "path-keyed dict ({'planes/0/mass/0/theta_E': ...}) instead. The "
            "3-group form could not say which plane a parameter was on."
        )
    if "planes" in truth or "cosmo" in truth:
        return lambda site: next(
            (v for v in (_nested_get(truth, p) for p in site.paths) if v is not None),
            None,
        )
    return lambda site: next(
        (
            truth[k]
            for k in ("/".join(map(str, p)) for p in site.paths)
            if k in truth
        ),
        None,
    )


def truth_row(
    sites: Sequence[ParamSite],
    truth: Any,
    *,
    what: str = "truth",
    warn: bool = True,
) -> np.ndarray:
    """A 1-D array of truth values in ``sites`` order, ``NaN`` where the truth
    does not define a parameter.

    Missing entries are filled rather than raised: a truth model and a fitted
    model legitimately differ in parameterization (approximating an image-based
    truth with a shapelet source, say), and ``corner`` simply draws no marker for
    a non-finite truth.
    """
    getter = _truth_getter(truth)
    values, missing = [], []
    for site in sites:
        value = getter(site)
        if value is None:
            missing.append(site.key)
            values.append(np.nan)
        else:
            values.append(float(np.squeeze(np.asarray(value))))
    if missing and warn:
        import warnings

        warnings.warn(
            f"{what} does not define {len(missing)} of {len(sites)} plotted "
            f"parameters; these get no marker: {missing}.",
            stacklevel=3,
        )
    return np.array(values)
