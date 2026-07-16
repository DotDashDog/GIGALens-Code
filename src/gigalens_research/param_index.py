"""The parameter index: one record per free parameter, in the scene's own path space.

Every free parameter of a scene-backed model has a *site path* — the place it
lives in the physical model — and the scene already knows it. Paths are tuples::

    ("planes", 1, "mass", 0, "theta_E")      -> plane 1, mass component 0
    ("planes", 1, "light", 0, "R_sersic")    -> plane 1, light component 0
    ("planes", 1, "geometry", "redshift")    -> plane 1's geometry
    ("cosmo", "H0")                          -> cosmology

so ``kind`` / ``plane`` / ``component`` are read off the path rather than
re-derived. :func:`param_sites` walks ``LensModel._site_to_unique`` (the
authoritative ``(path, unique_key, component_index)`` list) and returns one
:class:`ParamSite` per free parameter, in a canonical order.

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
        output dict (``planes/0/mass/0/theta_E``, ``shared_7``, ``coupled_3``,
        or a pipe-joined grouped-prior key).
    cidx : int or None
        Which component of a grouped / coupled prior's vector value this is
        (e.g. ``e1`` = 0, ``e2`` = 1 of a ``DiskEllipticity``). ``None`` for an
        ordinary scalar parameter.
    kind : str
        One of :data:`KINDS`.
    param : str
        Bare parameter name, e.g. ``theta_E``.
    paths : tuple of tuple
        Every scene site this free parameter feeds. Length > 1 exactly when it
        is a ``shared()`` parameter.
    key : str
        Canonical stable identity: the primary site path, slash-joined.
    """

    ukey: str
    cidx: Optional[int]
    kind: str
    param: str
    paths: Tuple[tuple, ...]
    key: str

    @property
    def shared(self) -> bool:
        """True when one free parameter feeds several sites."""
        return len(self.paths) > 1

    @property
    def planes(self) -> FrozenSet[int]:
        """Every plane this parameter acts on (empty for cosmology)."""
        return frozenset(
            p[1] for p in self.paths if p and p[0] == "planes" and isinstance(p[1], int)
        )

    @property
    def plane(self) -> Optional[int]:
        """The plane, when unambiguous; ``None`` for cosmology or for a shared
        parameter spanning several planes (see :attr:`planes`)."""
        pl = self.planes
        return next(iter(pl)) if len(pl) == 1 else None

    @property
    def components(self) -> FrozenSet[Tuple[str, int]]:
        """Every ``(role, index)`` component this parameter acts on, where role
        is ``"mass"`` or ``"light"``. Empty for cosmology and geometry."""
        return frozenset(
            (p[2], p[3])
            for p in self.paths
            if len(p) == 5 and p[0] == "planes" and p[2] in (KIND_MASS, KIND_LIGHT)
        )

    @property
    def component(self) -> Optional[int]:
        """The component index within its ``(plane, role)``, when unambiguous."""
        cs = {idx for _role, idx in self.components}
        return next(iter(cs)) if len(cs) == 1 else None


# ---------------------------------------------------------------------------
# Building the index
# ---------------------------------------------------------------------------


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
    """The site path behind a slash-joined key: the inverse of
    :attr:`ParamSite.key`. Numeric segments become ints, so the result compares
    equal to the scene's own path tuples."""
    return tuple(int(s) if s.isdigit() else s for s in key.split("/"))


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
        grouped[ident].append(tuple(path))

    sites: List[ParamSite] = []
    for ukey, cidx in order:
        paths = tuple(grouped[(ukey, cidx)])
        spec = _classify(paths[0])
        if spec is None:
            continue
        kind, param = spec
        sites.append(
            ParamSite(
                ukey=ukey,
                cidx=cidx,
                kind=kind,
                param=param,
                paths=paths,
                key="/".join(map(str, paths[0])),
            )
        )
    sites.sort(key=_sort_key)  # stable: definition order survives within a component
    return sites


# ---------------------------------------------------------------------------
# Display labels
# ---------------------------------------------------------------------------


def _suffix(site: ParamSite) -> str:
    """The tag used to break a label collision: plane, then component as
    ``m<i>`` / ``l<i>``, e.g. ``0,m1`` for plane 0's mass component 1.

    The role letter is not decoration. Mass and light components are indexed
    separately within a plane, so plane 0's mass[0] and plane 0's light[0] are
    both "component 0" — and since ``e1``/``e2``/``center_x``/``center_y`` are
    parameter names common to mass *and* light profiles, a bare ``(plane,
    component)`` tag collides on exactly the parameters most likely to need it.

    A shared parameter spanning several sites gets them joined (``0+1``), which
    is the honest reading: it is one parameter acting in both.
    """
    parts: List[str] = []
    planes = sorted(site.planes)
    if planes:
        parts.append("+".join(str(p) for p in planes))
    comps = sorted(site.components, key=lambda rc: (rc[1], rc[0]))
    if comps:
        parts.append("+".join(f"{role[0]}{idx}" for role, idx in comps))
    return ",".join(parts)


def _decorate(base: str, site: ParamSite, *, latex: bool) -> str:
    tag = _suffix(site)
    if not tag:
        return base
    if latex and base.startswith("$") and base.endswith("$"):
        return f"${base[1:-1]}^{{({tag})}}$"
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


def _normalize_planes(plane: Any) -> frozenset:
    out = set()
    for p in _as_set(plane):
        if isinstance(p, bool) or not isinstance(p, (int, np.integer)):
            raise TypeError(f"plane must be an int or list of ints; got {p!r}.")
        out.add(int(p))
    return frozenset(out)


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


def _normalize_components(component: Any) -> Tuple[frozenset, frozenset]:
    """Split a ``component=`` selection into bare indices and ``(role, index)``
    pairs. Accepts ``0``, ``[0, 1]``, ``("mass", 0)``, or a list mixing them."""
    if _is_role_pair(component):
        raw = [component]  # a lone ("mass", 0) is one pair, not two selections
    elif isinstance(component, (list, tuple)):
        raw = list(component)
    else:
        raw = [component]
    idxs, pairs = set(), set()
    for item in raw:
        if _is_role_pair(item):
            role, idx = item
            role = role.lower()
            if role not in (KIND_MASS, KIND_LIGHT):
                raise ValueError(
                    f"component role must be 'mass' or 'light'; got {role!r}."
                )
            pairs.add((role, int(idx)))
        elif isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            idxs.add(int(item))
        else:
            raise TypeError(
                f"component must be an int, a (role, index) pair, or a list of "
                f"those; got {item!r}."
            )
    return frozenset(idxs), frozenset(pairs)


def _is_role_pair(item: Any) -> bool:
    return (
        isinstance(item, tuple)
        and len(item) == 2
        and isinstance(item[0], str)
        and isinstance(item[1], (int, np.integer))
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
    plane : int or list of int, optional
        Plane index. A shared parameter matches if *any* of its sites is on a
        named plane.
    component : int, (role, index), or list, optional
        Component index within its ``(plane, role)``. Pass ``("mass", 0)`` to
        pin the role, or combine a bare index with ``kind=``/``plane=``.
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
        wanted_planes = _normalize_planes(plane)
        out = [s for s in out if s.planes & wanted_planes]
    if component is not None:
        idxs, pairs = _normalize_components(component)
        out = [
            s
            for s in out
            if ({i for _r, i in s.components} & idxs) or (s.components & pairs)
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
    planes = sorted({p for s in sites for p in s.planes})
    return (
        f"This model has kind={kinds} and plane={planes}; "
        f"{len(sites)} free parameters in total."
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
    """Walk a scene-nested truth by site path, tolerating int-vs-str keys and
    list-indexed planes/components. Returns ``None`` if absent."""
    node = truth
    for seg in path:
        if isinstance(node, dict):
            if seg in node:
                node = node[seg]
            elif str(seg) in node:
                node = node[str(seg)]
            else:
                return None
        elif isinstance(node, (list, tuple)) and isinstance(seg, int):
            if seg >= len(node):
                return None
            node = node[seg]
        else:
            return None
    return node


def _truth_getter(truth: Any) -> Callable[[ParamSite], Any]:
    """Resolve a truth into a per-site lookup.

    Accepts the two forms the scene API actually produces: a scene-nested point
    (``{"planes": {0: {"mass": {0: {...}}}}, "cosmo": {...}}``) or a flat
    path-keyed dict (``{"planes/0/mass/0/theta_E": ..., "cosmo/H0": ...}``,
    the idiom gigalens' own tests use).
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
            "truth ({'planes': {0: {'mass': {0: {...}}}}, 'cosmo': {...}}) or a "
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
