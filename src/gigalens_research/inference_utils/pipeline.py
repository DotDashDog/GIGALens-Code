"""GIGALens inference pipeline.

A small, generic driver that runs a sequence of inference *stages* (MAP, SVI,
HMC, MCLMC, ...) and caches each stage's results on disk so that re-runs only
recompute what actually changed.

Design summary
--------------
- Each algorithm is an :class:`InferenceStage` that declares what artifacts it
  ``requires`` from upstream and what it ``produces`` for downstream. The
  driver wires them by name and validates the chain before anything runs.
- Each stage's output is a :class:`StageResult` (raw arrays + metadata only);
  high-level objects (e.g. ``tfd.MultivariateNormalTriL``) are reconstructed
  by ``derive_artifacts`` from those arrays on demand.
- The on-disk layout is ``<out_dir>/<stage_name>/{manifest.json,arrays.npz}``
  with an ``input_hash`` covering the model, the stage's config, its seed, and
  the hashes of the upstream artifacts it consumed. On rerun, a stage is
  loaded from disk iff its input hash matches; mismatches invalidate the stage
  and every downstream stage.
- :class:`BridgeStage` is a pure-function stage for stitching: e.g. building a
  diagonal ``qz`` from a MAP optimum, or any other ad-hoc transform between
  stages. It re-runs every time (the ``fn`` is presumed cheap and pure) but
  participates in input-hash propagation like any other stage.

Typical use::

    pipeline = Pipeline(InferenceContext.from_modelling_sequence(model_seq))
    pipeline.add(MAPStage(num_steps=1000, n_samples=2000))
    pipeline.add(SVIStage(num_steps=5000, n_vi=1000))
    pipeline.add(HMCStage(n_hmc=64, num_results=1500))
    artifacts = pipeline.run(out_dir="results/system_4")

Non-standard stitching:
- Skip an upstream stage with ``pipeline.run(seed_artifacts={"z_best": z})``.
- Insert a custom transform with ``BridgeStage``.
- Replace a sampler by adding a different stage that produces the same key.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import hashlib
import io
import json
import os
import shutil
import time
import warnings
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import jax
import numpy as np
from jax import numpy as jnp


# ---------------------------------------------------------------------------
# Hashing utilities
# ---------------------------------------------------------------------------
#
# We need stable hashes for: numpy/jax arrays, TFP Distributions, dataclasses,
# dicts, lists, scalars, and the GIGALens model objects. The strategy is:
# (1) try ``jax.tree.flatten`` (works for TFP distributions, nested
# dicts/lists/tuples, and our model dataclasses once we coerce them), then
# (2) hash each leaf canonically. Closures, callables, and arbitrary objects
# fall back to a typed repr; callers should supply an explicit version string
# for those (this is what BridgeStage and the optimizer_id kwarg are for).
#
# These hashes are *cache keys*, not cryptographic commitments; we only need
# them to detect when an input changed, not to defend against adversaries.


_HEX_LEN = 16  # 64 bits is plenty for our cache-key collision budget


def _short(h: str) -> str:
    return h[:_HEX_LEN]


def _feed(h: "hashlib._Hash", obj: Any) -> None:
    """Mix ``obj`` into ``h`` using a canonical, type-tagged encoding."""
    if obj is None:
        h.update(b"none")
    elif isinstance(obj, bool):
        h.update(b"bool:"); h.update(b"1" if obj else b"0")
    elif isinstance(obj, (int, np.integer)):
        h.update(b"int:"); h.update(repr(int(obj)).encode())
    elif isinstance(obj, (float, np.floating)):
        h.update(b"float:"); h.update(repr(float(obj)).encode())
    elif isinstance(obj, (str, bytes)):
        h.update(b"str:"); h.update(obj.encode() if isinstance(obj, str) else obj)
    elif isinstance(obj, np.ndarray):
        h.update(b"arr:")
        h.update(str(obj.dtype).encode()); h.update(b":")
        h.update(str(obj.shape).encode()); h.update(b":")
        h.update(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, jax.Array):
        _feed(h, np.asarray(obj))
    elif isinstance(obj, dict):
        h.update(b"dict:")
        for k in sorted(obj.keys(), key=str):
            _feed(h, k); _feed(h, obj[k])
    elif isinstance(obj, (list, tuple)):
        h.update(b"seq:"); h.update(type(obj).__name__.encode())
        for x in obj:
            _feed(h, x)
    else:
        # Fallback: pytree-flatten (works for TFP Distributions, registered
        # custom classes, and most numerical structures), falling through to
        # a typed-repr only if flatten returns nothing useful.
        try:
            leaves, treedef = jax.tree_util.tree_flatten(obj)
        except Exception:
            leaves, treedef = None, None
        # An unregistered object flattens to itself as a single leaf
        # (``leaves == [obj]``); recursing on it would loop forever. Treat that
        # case as "not a pytree" and fall through to parameters/repr handling.
        if leaves and not (len(leaves) == 1 and leaves[0] is obj):
            h.update(b"pytree:"); h.update(type(obj).__name__.encode())
            h.update(b":"); h.update(str(treedef).encode())
            for leaf in leaves:
                _feed(h, leaf)
        elif hasattr(obj, "parameters") and isinstance(getattr(obj, "parameters"), dict):
            # TFP Distributions on older JAX/TFP releases that don't expose
            # parameters as pytree leaves still surface them via ``.parameters``.
            h.update(b"params:"); h.update(type(obj).__name__.encode()); h.update(b":")
            _feed(h, obj.parameters)
        else:
            warnings.warn(
                f"[pipeline.stable_hash] no structural hash for {type(obj).__name__}; "
                f"falling back to repr(). If this object affects results, set an "
                f"explicit version on the consuming stage instead of relying on "
                f"auto-hashing.",
                stacklevel=4,
            )
            h.update(b"repr:"); h.update(type(obj).__name__.encode())
            h.update(b":"); h.update(repr(obj).encode())


def stable_hash(obj: Any) -> str:
    """Canonical short hex hash of ``obj`` (see :func:`_feed` for the encoding)."""
    h = hashlib.sha256()
    _feed(h, obj)
    return _short(h.hexdigest())


# ---------------------------------------------------------------------------
# Inference context: model + simulator state shared by all stages
# ---------------------------------------------------------------------------


class _ScenePhysModelView:
    """A read-only ``phys_model``-shaped view over a scene ``LensModel`` (G1 C).

    ``model_card`` / ``InferenceContext.hash`` / ``posterior.py`` iterate
    ``phys_model.{lenses, lens_light, source_light}`` as lists of gigalens profile
    objects (reading ``.use_lstsq`` / ``.depth`` / ``vars(p)``). A scene LensModel groups
    things into planes/Components instead, so this shim re-derives those three profile
    lists by role:
      - ``lenses``       : every mass Component's profile (plane order)
      - ``source_light`` : light on lensed planes (``source_plane_light``)
      - ``lens_light``   : the remaining (non-lensed-plane) light
    This keeps the legacy read-only consumers working unchanged on a scene-backed model
    WITHOUT touching the old 3-group vocabulary anywhere else."""

    def __init__(self, scene_model):
        self._model = scene_model
        self.lenses = [c.profile for p in scene_model.planes for c in p.mass]
        src_ids = {id(c) for c in scene_model.source_plane_light()}
        self.source_light = [c.profile for c in scene_model.light_components
                             if id(c) in src_ids]
        self.lens_light = [c.profile for c in scene_model.light_components
                           if id(c) not in src_ids]


@dataclasses.dataclass(frozen=True)
class InferenceContext:
    """Everything stages need to read about the system being modeled.

    Stages should treat all fields as read-only. ``model_seq`` carries the
    JAX-side ``MAP``/``SVI``/``HMC`` implementations; the other fields are
    surfaced separately so they're easy to hash without touching closures
    inside ``ModellingSequence``.
    """

    phys_model: Any
    prob_model: Any
    sim_config: Any
    model_seq: Any

    @classmethod
    def from_modelling_sequence(cls, model_seq) -> "InferenceContext":
        # G1 dual-path: for a scene-backed ModellingSequence, expose a
        # phys_model-shaped VIEW derived from the scene LensModel so the legacy
        # read-only consumers (model_card / hash / posterior) are unchanged. The
        # legacy path passes the real PhysicalModel through untouched.
        if getattr(model_seq, "scene_model", None) is not None:
            phys_view = _ScenePhysModelView(model_seq.scene_model)
        else:
            phys_view = model_seq.phys_model
        return cls(
            phys_model=phys_view,
            prob_model=model_seq.prob_model,
            sim_config=model_seq.sim_config,
            model_seq=model_seq,
        )

    def hash(self) -> str:
        """Stable hash of the *modeling inputs* (not the ``model_seq`` impl).

        Picks up whichever noise-model attributes the prob_model actually
        carries: gigalens' ``ForwardProbModel`` exposes ``background_rms`` /
        ``exp_time``; ``BackwardProbModel`` exposes ``err_map``. Both forms
        (and any reasonable third-party variant that mixes them) are covered.
        The prob_model class name is folded in so two models with overlapping
        but differently-interpreted attributes don't alias.
        """
        pm = self.prob_model
        noise: Dict[str, Any] = {
            "class": type(pm).__name__,
            "observed_image": np.asarray(pm.observed_image),
        }
        for attr in ("background_rms", "exp_time", "err_map"):
            if hasattr(pm, attr):
                noise[attr] = np.asarray(getattr(pm, attr))
        return stable_hash({
            "phys_model": _hash_phys_model(self.phys_model),
            "prior": pm.prior,
            "noise": noise,
            "sim_config": _hash_sim_config(self.sim_config),
        })


def _is_profile(obj) -> bool:
    """Duck-type a gigalens profile (mass or light): carries a ``_name`` tag and
    a ``params`` list. Used to expand *composite* profiles (e.g.
    ``SersicShapelets``, which holds sub-profiles as public attributes) instead
    of feeding the opaque object to the hasher."""
    return (
        not isinstance(obj, (str, bytes))
        and hasattr(obj, "_name")
        and hasattr(obj, "params")
    )


def _hash_phys_model(pm) -> Dict[str, Any]:
    """Cheap, robust hash of a ``PhysicalModel``: profile classes + their
    public attributes (which is where things like ``EPL(niter=50)`` live).

    Composite profiles that nest other profile objects as public attributes
    (e.g. ``SersicShapelets.sersic`` / ``.shapelets``) are expanded recursively
    so the hash stays content-based; a raw profile object would otherwise reach
    the hasher's pytree fallback, which treats it as a self-referential leaf.
    """
    def _describe(p):
        attrs = {}
        for k, v in vars(p).items():
            if k.startswith("_"):
                continue
            attrs[k] = _describe(v) if _is_profile(v) else v
        return {"type": type(p).__name__, "attrs": attrs}

    def _profiles(plist):
        return [_describe(p) for p in plist]
    return {
        "lenses": _profiles(pm.lenses),
        "lens_light": _profiles(pm.lens_light),
        "source_light": _profiles(pm.source_light),
    }


def _hash_sim_config(sc) -> Dict[str, Any]:
    if dataclasses.is_dataclass(sc):
        d = dataclasses.asdict(sc)
    else:
        d = {k: v for k, v in vars(sc).items() if not k.startswith("_")}
    return d


# ---------------------------------------------------------------------------
# Model card: an explicit, loud summary of the *effective* forward model.
#
# Exists to make model misspecification visible. The fields that have
# historically been set wrong *silently* — PSF, noise model, pixel grid,
# likelihood precision, and whether the linear (lstsq) amplitudes carry any
# physical regularization — are reported explicitly, and anything that looks
# like a silent degradation (e.g. no PSF on data that are normally convolved)
# is surfaced in ``warnings``. A flexible model can hide misspecification
# behind a good chi^2; this card does not.
# ---------------------------------------------------------------------------


def _profile_depth(p) -> int:
    return int(getattr(p, "depth", getattr(p, "n_layers", 1)))


def model_card(ctx: "InferenceContext") -> Dict[str, Any]:
    """Return a JSON-safe summary of the effective forward model for ``ctx``.

    Written to ``<out_dir>/model_card.json`` by :meth:`Pipeline.run` and printed
    via :func:`format_model_card`. Use directly on any context/posterior:
    ``print(format_model_card(model_card(ctx)))``.
    """
    pm, sc, phys = ctx.prob_model, ctx.sim_config, ctx.phys_model
    warns: List[str] = []

    # --- PSF ---
    kernel = getattr(sc, "kernel", None)
    if kernel is None:
        psf: Dict[str, Any] = {"present": False}
        warns.append(
            "NO PSF: sim_config.kernel is None — the forward model applies no PSF "
            "convolution. If the data are PSF-convolved (they usually are), this "
            "model is MISSPECIFIED. Check the asset path / loader."
        )
    else:
        k = np.asarray(kernel)
        psf = {
            "present": True,
            "shape": list(k.shape),
            "sum": float(k.sum()),
            "sha1": hashlib.sha1(
                np.ascontiguousarray(k, dtype=np.float64).tobytes()
            ).hexdigest()[:12],
        }
        if not np.isclose(float(k.sum()), 1.0, atol=1e-3):
            warns.append(
                f"PSF kernel does not sum to 1 (sum={float(k.sum()):.4f}); check normalization."
            )

    # --- noise model ---
    # Detection order: legacy backward (``err_map``), legacy forward
    # (``background_rms``/``exp_time``), then the SCENE ProbModel's per-pixel ``error_map``
    # (G1: it builds the error map inside the Dataset, so it exposes neither err_map nor
    # background_rms/exp_time — but the noise IS present and used in the likelihood).
    # Report the error_map DIRECTLY; do NOT fabricate background_rms/exp_time from it.
    noise: Dict[str, Any] = {"prob_model": type(pm).__name__}
    scene_error_map = None
    if not hasattr(pm, "err_map"):
        try:
            scene_error_map = pm.error_map  # scene ProbModel property (single dataset)
        except Exception:
            # Missing (legacy) or raises for a multi-dataset scene model — treat as
            # "no single error_map to report" and fall through to the unknown branch.
            scene_error_map = None
    if hasattr(pm, "err_map"):
        em = np.asarray(pm.err_map)
        noise.update(kind="fixed_err_map (backward)", err_map_shape=list(em.shape),
                     err_map_min=float(em.min()), err_map_median=float(np.median(em)))
    elif hasattr(pm, "background_rms") and hasattr(pm, "exp_time"):
        noise.update(kind="forward (bkg_rms + poisson)",
                     background_rms=float(np.asarray(pm.background_rms)),
                     exp_time=float(np.asarray(pm.exp_time)))
    elif scene_error_map is not None:
        em = np.asarray(scene_error_map)
        noise.update(kind="fixed error_map (scene)", err_map_shape=list(em.shape),
                     err_map_min=float(em.min()), err_map_median=float(np.median(em)))
    else:
        noise["kind"] = "unknown"
        warns.append(
            "Noise model unrecognized: prob_model exposes neither err_map nor "
            "(background_rms, exp_time) nor a scene error_map."
        )

    # --- grid / precision (read directly; these are required fields) ---
    grid = {"num_pix": sc.num_pix, "delta_pix": sc.delta_pix, "supersample": sc.supersample}
    prec = getattr(sc, "likelihood_precision", None)  # physics-default-ok: reporting only

    def _prof(p) -> Dict[str, Any]:
        scalars = {k: v for k, v in vars(p).items()
                   if not k.startswith("_") and isinstance(v, (int, float, str, bool))}
        return {"type": type(p).__name__, **scalars}

    profiles = {
        "lens_mass": [_prof(p) for p in phys.lenses],
        "lens_light": [_prof(p) for p in phys.lens_light],
        "source_light": [_prof(p) for p in phys.source_light],
    }

    # --- linear amplitudes / regularization ---
    use_lstsq = any(getattr(p, "use_lstsq", False)  # physics-default-ok: reporting only
                    for p in list(phys.source_light) + list(phys.lens_light))
    n_basis = (sum(_profile_depth(p) for p in phys.lens_light)
               + sum(_profile_depth(p) for p in phys.source_light))
    amps = {
        "solved_by_lstsq": bool(use_lstsq),
        "n_basis": int(n_basis),
        # gigalens' lstsq adds only a tiny numerical jitter to the Gram matrix,
        # not a physical (e.g. ridge/curvature) prior on the amplitudes.
        "physical_regularization": None,
    }
    if use_lstsq and n_basis >= 50:
        warns.append(
            f"{n_basis} linear amplitudes solved by lstsq with NO physical regularization "
            "(numerical jitter only). High-order bases can fit noise or place flux in the "
            "lensing null space (unphysical source structure)."
        )

    card = {
        "psf": psf,
        "noise": noise,
        "grid": grid,
        "likelihood_precision": prec,
        "profiles": profiles,
        "linear_amplitudes": amps,
        "warnings": warns,
    }

    # --- scene-specific section (Phase 7b): trace mode, per-plane distances, sees ---
    # Only for scene-backed contexts; legacy card output is unchanged (no "scene" key).
    scene = _scene_card_section(ctx)
    if scene is not None:
        card["scene"] = scene
    return card


def _scene_card_section(ctx) -> Optional[Dict[str, Any]]:
    """Phase-7b scene card: amplitude mode, active trace mode, per-plane redshifts +
    (multiplane) distances from the model's own cosmology, and the per-dataset ``sees``.

    Returns None for a legacy (non-scene) context so the legacy card is unchanged."""
    model = getattr(getattr(ctx, "model_seq", None), "scene_model", None)
    if model is None:
        return None
    pm = ctx.prob_model
    out: Dict[str, Any] = {"amplitude_mode": getattr(pm, "mode", None)}

    # Active trace mode (deflection_ratio vs multiplane) from a built simulator.
    sims = getattr(pm, "simulators", None)
    out["trace_mode"] = (getattr(sims[0], "trace_mode", None)
                         if sims else None)

    # Per-plane geometry: redshift (cosmology) or deflection_ratio (single-plane).
    planes_info = []
    has_cosmo = model.cosmo is not None
    for i, p in enumerate(model.planes):
        geom = {"plane": i,
                "n_mass": len(p.mass), "n_light": len(p.light)}
        if has_cosmo:
            geom["redshift"] = p.redshift
        else:
            geom["deflection_ratio"] = p.deflection_ratio
        planes_info.append(geom)
    out["planes"] = planes_info

    # Multiplane: per-plane transverse comoving distances from the model's OWN cosmology
    # (cosmo.distance_matrix). Reportable when every plane redshift + the cosmology are
    # CONSTANTS (read from model.constants); if any is sampled there is no single value to
    # report on the card, which is noted rather than guessed.
    if has_cosmo and out["trace_mode"] == "multiplane":
        try:
            consts = model.constants
            zs = [consts["planes"][i]["geometry"]["redshift"]
                  for i in range(len(model.planes))]
            cosmo_params = consts.get("cosmo", {})
            D = np.asarray(model.cosmo.profile.distance_matrix(zs, **cosmo_params))
            # observer(0) -> plane_k transverse comoving distance (D has shape (N+1, N+1)).
            out["redshifts"] = [float(np.asarray(z).squeeze()) for z in zs]
            out["distances_obs_to_plane"] = [float(D[k + 1, 0])
                                             for k in range(len(model.planes))]
        except (KeyError, TypeError) as exc:
            out["distances_note"] = (
                "multiplane distances not reported (a plane redshift or a cosmology "
                f"param is sampled, not constant): {type(exc).__name__}")

    # Per-dataset sees selection (resolved Component profile types).
    resolved = getattr(pm, "_resolved_sees", None)
    if resolved is not None:
        out["sees"] = [[type(c.profile).__name__ for c in seen] for seen in resolved]
    return out


def format_model_card(card: Dict[str, Any]) -> str:
    """Human-readable one-screen summary of :func:`model_card`."""
    bar = "=" * 72
    lines = [bar, "EFFECTIVE FORWARD MODEL (model card)", bar]
    psf = card["psf"]
    if psf.get("present"):
        lines.append(f"  PSF        : present  shape={psf['shape']} "
                     f"sum={psf['sum']:.4f} sha1={psf['sha1']}")
    else:
        lines.append("  PSF        : ** ABSENT — NO PSF CONVOLUTION **")
    n = card["noise"]
    lines.append(f"  Noise      : {n.get('kind')}  ({n.get('prob_model')})")
    g = card["grid"]
    lines.append(f"  Grid       : num_pix={g['num_pix']} delta_pix={g['delta_pix']} "
                 f"supersample={g['supersample']}")
    lines.append(f"  Precision  : {card['likelihood_precision']}")
    a = card["linear_amplitudes"]
    lines.append(f"  Amplitudes : lstsq={a['solved_by_lstsq']} n_basis={a['n_basis']} "
                 f"physical_reg={a['physical_regularization']}")
    def _fmt(p):
        extra = f"(n_max={p['n_max']})" if "n_max" in p else ""
        return p["type"] + extra
    lines.append("  Source     : " + ", ".join(_fmt(p) for p in card["profiles"]["source_light"]))
    lines.append("  Lens mass  : " + ", ".join(p["type"] for p in card["profiles"]["lens_mass"]))
    lines.append("  Lens light : " + ", ".join(p["type"] for p in card["profiles"]["lens_light"]))
    # Phase 7b: scene section (only present for scene-backed models; legacy unchanged).
    s = card.get("scene")
    if s is not None:
        lines.append(f"  Trace      : {s.get('trace_mode')}  (amplitude mode "
                     f"{s.get('amplitude_mode')})")
        if "distances_obs_to_plane" in s:
            zr = ", ".join(f"z={z:.3g}->{d:.1f}Mpc"
                           for z, d in zip(s.get("redshifts", []),
                                           s["distances_obs_to_plane"]))
            lines.append(f"  Distances  : {zr}")
        elif "distances_note" in s:
            lines.append(f"  Distances  : {s['distances_note']}")
        if s.get("sees") is not None:
            lines.append("  Sees       : " + " | ".join(
                "[" + ", ".join(view) + "]" for view in s["sees"]))
    if card["warnings"]:
        lines.append("-" * 72)
        for w in card["warnings"]:
            lines.append("  [!] " + w)
    lines.append(bar)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# StageResult: the only thing a stage's ``run`` should return
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class StageResult:
    """Raw, pickleable output of a single stage.

    ``arrays`` holds the saved-to-disk arrays (np.ndarray). ``metadata`` holds
    JSON-serializable scalars: wall time, seed, num_steps, etc. Anything that
    needs to flow into downstream stages as a higher-level object (e.g. a TFP
    distribution) is reconstructed by the stage's ``derive_artifacts``.

    ``diagnostics`` holds optional debug arrays (e.g. an MCLMC tuning history):
    extra, often large, run-internal quantities that are *not* published as
    artifacts and never flow downstream. They are persisted separately from
    ``arrays`` so loading a posterior view doesn't drag them in. Populate them
    only when a stage is run with ``debug=True``.
    """

    arrays: Dict[str, np.ndarray]
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)
    diagnostics: Dict[str, np.ndarray] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class StageDiagnostics:
    """A stage's debug arrays plus enough context to plot them.

    Returned by :meth:`Pipeline.diagnostics` and :func:`diagnostics_from_disk`,
    and consumed by ``gigalens_research.plotting.plot_stage_diagnostics``,
    which dispatches on ``stage_class`` to the matching registered plotter.

    Attributes
    ----------
    stage_name : str
        The stage's instance name (its on-disk directory).
    stage_class : str
        The stage class name (e.g. ``"MCLMCStage"``); the plot-dispatch key.
    arrays : dict[str, np.ndarray]
        The captured debug arrays (e.g. ``step_size``, ``L``, ``xi``, ...).
    config : dict
        Plot-relevant config from ``InferenceStage.diagnostics_config`` (e.g.
        tuning-stage boundaries).
    ctx : InferenceContext
        The modeling context, for plotters that need the model/simulator.
    """

    stage_name: str
    stage_class: str
    arrays: Dict[str, np.ndarray]
    config: Dict[str, Any]
    ctx: "InferenceContext"

    def __bool__(self) -> bool:
        return bool(self.arrays)


# ---------------------------------------------------------------------------
# InferenceStage base class
# ---------------------------------------------------------------------------


class InferenceStage:
    """Base class for inference algorithms in the pipeline.

    Subclasses define:
    - ``name`` (class-level): default directory name on disk.
    - ``schema_version`` (class-level int): bump when the on-disk array
      layout for this stage changes.
    - ``requires`` / ``produces`` (class-level tuples of artifact names):
      what this stage reads from / writes to the shared artifact bag.
    - ``run(ctx, artifacts, seed) -> StageResult``: actually run the algorithm.
    - ``derive_artifacts(arrays) -> dict``: reconstruct the artifacts named
      in ``produces`` from the saved arrays. Default: identity over
      ``produces``.

    Optional per-instance overrides (passed to ``__init__``):
    - ``name``: distinct directory name (e.g. two MAPs in one pipeline).
    - ``seed``: stage-specific seed; falls back to the pipeline-wide seed.
    """

    name: ClassVar[str] = "stage"
    schema_version: ClassVar[int] = 1
    requires: ClassVar[Tuple[str, ...]] = ()
    produces: ClassVar[Tuple[str, ...]] = ()
    # Whether this stage's outputs can be reconstructed from disk. Sampler /
    # optimizer stages set True (the default); pure-function stages like
    # ``BridgeStage`` set False because they don't persist arrays and must
    # rebuild their artifacts by calling their ``fn`` again.
    cacheable_outputs: ClassVar[bool] = True

    def __init__(self, *, name: Optional[str] = None, seed: Optional[int] = None):
        self._name = name or type(self).name
        self._seed = seed

    @property
    def instance_name(self) -> str:
        return self._name

    def effective_seed(self, pipeline_seed: int) -> int:
        return pipeline_seed if self._seed is None else self._seed

    def config_hash_data(self) -> Any:
        """Return everything that should contribute to this stage's input
        hash *besides* the context, seed, and upstream artifacts.

        Default: all public instance attributes whose names don't start with
        ``_``. Override if your config contains unhashable closures (and
        replace them with version strings — see ``MAPStage.optimizer_id``).
        """
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    def run(self, ctx: InferenceContext, artifacts: Dict[str, Any], seed: int) -> StageResult:
        raise NotImplementedError

    def derive_artifacts(self, arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
        return {k: arrays[k] for k in self.produces if k in arrays}

    @classmethod
    def to_posterior(cls, arrays: Dict[str, np.ndarray], ctx: "InferenceContext"):
        """Build a :class:`~gigalens_research.inference_utils.posterior.Posterior`
        view from the stage's saved arrays. Override in subclasses that
        produce something worth viewing (samples, surrogate, point estimate).
        """
        raise TypeError(
            f"{cls.__name__} does not produce a posterior view "
            f"(no to_posterior implementation)."
        )

    def diagnostics_config(self) -> Dict[str, Any]:
        """Subset of this stage's config that its diagnostic plotter needs
        (e.g. tuning-stage boundaries). Stashed into the persisted manifest so
        :class:`StageDiagnostics` is self-contained on reload. Default: empty.
        """
        return {}


# ---------------------------------------------------------------------------
# BridgeStage: ad-hoc transforms between stages
# ---------------------------------------------------------------------------


class BridgeStage(InferenceStage):
    """A pure-function stage that maps some artifacts to others.

    Use for non-standard stitching, e.g. building a diagonal ``qz`` from a
    MAP optimum to skip SVI::

        BridgeStage(
            name="diag_qz_from_map",
            version="v1",
            requires=("z_best",),
            produces=("qz",),
            fn=lambda z_best: tfd.MultivariateNormalDiag(
                loc=jnp.asarray(z_best),
                scale_diag=jnp.full(z_best.shape[-1], 1e-2),
            ),
        )

    Bridges always re-run on pipeline invocation (the ``fn`` is presumed
    cheap and deterministic), but they still participate in input-hash
    propagation so downstream stages get cached/invalidated correctly.
    """

    schema_version: ClassVar[int] = 1
    cacheable_outputs: ClassVar[bool] = False

    def __init__(
        self,
        *,
        name: str,
        version: str,
        requires: Iterable[str],
        produces: Iterable[str],
        fn: Callable[..., Any],
    ):
        super().__init__(name=name)
        self.version = version
        self._requires = tuple(requires)
        self._produces = tuple(produces)
        self.fn = fn

    @property
    def requires(self) -> Tuple[str, ...]:  # type: ignore[override]
        return self._requires

    @property
    def produces(self) -> Tuple[str, ...]:  # type: ignore[override]
        return self._produces

    def config_hash_data(self) -> Any:
        return {"version": self.version}

    def run(self, ctx, artifacts, seed):
        t0 = time.perf_counter()
        out = self.fn(**{k: artifacts[k] for k in self._requires})
        if not isinstance(out, dict):
            if len(self._produces) != 1:
                raise ValueError(
                    f"BridgeStage {self._name!r}: fn returned a non-dict but "
                    f"produces declares {len(self._produces)} keys"
                )
            out = {self._produces[0]: out}
        missing = set(self._produces) - set(out.keys())
        if missing:
            raise ValueError(
                f"BridgeStage {self._name!r}: fn returned {set(out)} but "
                f"produces declares {set(self._produces)}; missing {missing}"
            )
        result = StageResult(
            arrays={},
            metadata={"wall_time_s": time.perf_counter() - t0,
                      "version": self.version},
        )
        # Stash the derived artifacts so the driver can publish them without
        # going through (nonexistent) arrays.
        result.metadata["_bridge_artifacts"] = out
        return result

    def derive_artifacts(self, arrays):
        # Bridges have no persisted arrays; the driver loads by re-running.
        return {}


# ---------------------------------------------------------------------------
# On-disk I/O
# ---------------------------------------------------------------------------


_MANIFEST_FILENAME = "manifest.json"
_ARRAYS_FILENAME = "arrays.npz"
_DIAGNOSTICS_FILENAME = "diagnostics.npz"
_PIPELINE_MANIFEST = "pipeline.json"


class PipelineMismatchError(RuntimeError):
    """Raised when ``resume='strict'`` finds a stage whose input hash
    disagrees with what's on disk."""


def _make_json_safe(obj: Any) -> Any:
    """Recursively coerce ``obj`` into something ``json.dump`` can handle."""
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()
                if not (isinstance(k, str) and k.startswith("_"))}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, (np.ndarray, jax.Array)):
        a = np.asarray(obj)
        return {"__array_summary__": True, "shape": list(a.shape), "dtype": str(a.dtype)}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def _write_manifest(path: str, manifest: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _read_manifest(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _save_stage(
    stage_dir: str,
    manifest: Dict[str, Any],
    arrays: Dict[str, np.ndarray],
    diagnostics: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    os.makedirs(stage_dir, exist_ok=True)
    if arrays:
        np.savez(os.path.join(stage_dir, _ARRAYS_FILENAME),
                 **{k: np.asarray(v) for k, v in arrays.items()})
    if diagnostics:
        np.savez(os.path.join(stage_dir, _DIAGNOSTICS_FILENAME),
                 **{k: np.asarray(v) for k, v in diagnostics.items()})
    _write_manifest(os.path.join(stage_dir, _MANIFEST_FILENAME), manifest)


def _load_npz(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        return {}
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def _load_stage_arrays(stage_dir: str) -> Dict[str, np.ndarray]:
    return _load_npz(os.path.join(stage_dir, _ARRAYS_FILENAME))


def _load_stage_diagnostics(stage_dir: str) -> Dict[str, np.ndarray]:
    return _load_npz(os.path.join(stage_dir, _DIAGNOSTICS_FILENAME))


def _move_aside(path: str) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%dT%H%M%S")
    new = f"{path}.stale-{stamp}"
    shutil.move(path, new)
    return new


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------


ResumeMode = Union[bool, Literal["strict"]]


class Pipeline:
    """A sequence of inference stages with on-disk per-stage caching.

    Parameters
    ----------
    ctx : InferenceContext
        Shared modeling context. Build one with
        ``InferenceContext.from_modelling_sequence(model_seq)``.
    seed : int
        Default seed for stages that don't set their own.

    Examples
    --------
    >>> p = Pipeline(InferenceContext.from_modelling_sequence(model_seq))
    >>> p.add(MAPStage(num_steps=1000, n_samples=2000))
    >>> p.add(SVIStage(num_steps=5000, n_vi=1000))
    >>> p.add(HMCStage(n_hmc=64, num_results=1500, num_burnin_steps=500))
    >>> artifacts = p.run(out_dir="results/system_4", resume=True)
    """

    def __init__(self, ctx: InferenceContext, seed: int = 0):
        self.ctx = ctx
        self.seed = int(seed)
        self.stages: List[InferenceStage] = []
        self.results: Dict[str, StageResult] = {}

    def add(self, stage: InferenceStage) -> "Pipeline":
        if any(s.instance_name == stage.instance_name for s in self.stages):
            raise ValueError(
                f"Stage name {stage.instance_name!r} is already in the pipeline; "
                f"pass a distinct name= when constructing the stage."
            )
        self.stages.append(stage)
        return self

    # -- public entry point ---------------------------------------------------

    def run(
        self,
        out_dir: Optional[str] = None,
        *,
        resume: ResumeMode = True,
        force: bool = False,
        seed_artifacts: Optional[Dict[str, Any]] = None,
        seed_artifact_ids: Optional[Dict[str, str]] = None,
        keep_stale: bool = True,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Execute the pipeline.

        Parameters
        ----------
        out_dir : str or None
            Directory to persist stage results into. ``None`` disables disk I/O.
        resume : bool or 'strict'
            - ``True``: load each stage from disk iff its input hash matches;
              otherwise (re)run it. Default.
            - ``'strict'``: as ``True``, but raise :class:`PipelineMismatchError`
              instead of re-running on any hash mismatch.
            - ``False``: ignore on-disk results and re-run every stage.
        force : bool
            If True, re-run every stage regardless of cached state. Equivalent
            to ``resume=False`` plus skipping the strict check.
        seed_artifacts : dict, optional
            Pre-populate the artifact bag (e.g. ``{"z_best": truth_z}`` to
            skip MAP and start SVI from a known point).
        seed_artifact_ids : dict, optional
            Explicit version strings for seeded artifacts that aren't easily
            auto-hashed (e.g. ``{"qz": "diag_around_truth_v1"}``).
        keep_stale : bool
            On invalidation, rename ``<stage>/`` to ``<stage>.stale-<ts>/``
            instead of deleting. Recoverable safety net.
        verbose : bool
            Print one line per stage with cache status and wall time.
        """
        if force:
            resume = False
        seed_artifacts = dict(seed_artifacts or {})
        seed_artifact_ids = dict(seed_artifact_ids or {})

        artifacts: Dict[str, Any] = dict(seed_artifacts)
        artifact_hashes: Dict[str, str] = {
            k: seed_artifact_ids.get(k) or stable_hash(v)
            for k, v in artifacts.items()
        }
        # Track which stage last wrote each artifact so we can name the loser
        # when an overwrite happens.
        artifact_owner: Dict[str, str] = {k: "<seed>" for k in artifacts}

        self._validate_dag(set(artifacts))

        if out_dir is not None:
            os.makedirs(out_dir, exist_ok=True)
        ctx_hash = self.ctx.hash()
        run_log: List[Dict[str, Any]] = []
        self.results = {}

        # Echo the effective forward model up front (before any compute), and
        # persist it, so a silent misspecification (e.g. a missing PSF) is
        # visible immediately rather than hidden behind a good chi^2.
        if jax.process_index() == 0:
            _card = model_card(self.ctx)
            print(format_model_card(_card))
            if out_dir is not None:
                _write_manifest(os.path.join(out_dir, "model_card.json"),
                                _make_json_safe(_card))

        for stage in self.stages:
            stage_dir = os.path.join(out_dir, stage.instance_name) if out_dir else None
            seed = stage.effective_seed(self.seed)

            upstream_hashes = {k: artifact_hashes[k] for k in stage.requires}
            input_hash = stable_hash({
                "ctx": ctx_hash,
                "class": type(stage).__name__,
                "schema": stage.schema_version,
                "config": stage.config_hash_data(),
                "seed": seed,
                "upstream": upstream_hashes,
            })

            result, status = self._run_or_load(
                stage, stage_dir, input_hash, upstream_hashes,
                artifacts, seed, resume, keep_stale,
            )

            new = self._publish(stage, result)
            for k, v in new.items():
                if k in artifacts and jax.process_index() == 0:
                    print(
                        f"[pipeline] WARNING: {stage.instance_name!r} overwrites "
                        f"artifact {k!r} previously produced by "
                        f"{artifact_owner.get(k, '?')!r}. "
                        f"Downstream stages will use {stage.instance_name!r}'s {k!r}."
                    )
                artifacts[k] = v
                artifact_hashes[k] = stable_hash(v)
                artifact_owner[k] = stage.instance_name
            self.results[stage.instance_name] = result

            if verbose and jax.process_index() == 0:
                wt = result.metadata.get("wall_time_s")
                print(f"[pipeline] {stage.instance_name}: {status}"
                      + (f" ({wt:.1f}s)" if isinstance(wt, (int, float)) else ""))
            run_log.append({
                "stage": stage.instance_name,
                "class": type(stage).__name__,
                "status": status,
                "input_hash": input_hash,
                "wall_time_s": result.metadata.get("wall_time_s"),
            })

        if out_dir is not None and jax.process_index() == 0:
            _write_manifest(
                os.path.join(out_dir, _PIPELINE_MANIFEST),
                {
                    "ctx_hash": ctx_hash,
                    "seed": self.seed,
                    "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                    "stages": run_log,
                },
            )

        return artifacts

    # -- internals ------------------------------------------------------------

    def _validate_dag(self, seeded_keys: set) -> None:
        """Check that all ``requires`` are satisfied and warn about shadowed artifacts.

        A *shadowed* artifact is one produced by stage A and then re-produced
        by a later stage B. Stage B's value will silently replace A's in the
        artifact bag — every stage that follows B gets B's version. This is
        intentional and useful (e.g. HessianSurrogate → SVI both produce
        ``qz``) but easy to set up by accident, so we warn here and again at
        runtime when the overwrite actually happens.
        """
        available: set = set(seeded_keys)
        # owner[key] = instance_name of the stage that most recently declared
        # it produces that key.
        owner: Dict[str, str] = {k: "<seed>" for k in seeded_keys}

        for stage in self.stages:
            missing = set(stage.requires) - available
            if missing:
                raise ValueError(
                    f"Stage {stage.instance_name!r} requires {sorted(missing)} "
                    f"but nothing earlier produces them. "
                    f"Available so far: {sorted(available)}."
                )
            for key in stage.produces:
                if key in available:
                    import warnings
                    warnings.warn(
                        f"[pipeline] artifact {key!r} is produced by both "
                        f"{owner[key]!r} and {stage.instance_name!r}. "
                        f"Stages after {stage.instance_name!r} will use "
                        f"{stage.instance_name!r}'s {key!r} and ignore "
                        f"{owner[key]!r}'s.",
                        stacklevel=4,
                    )
                owner[key] = stage.instance_name
            available.update(stage.produces)

    def _run_or_load(
        self,
        stage: InferenceStage,
        stage_dir: Optional[str],
        input_hash: str,
        upstream_hashes: Dict[str, str],
        artifacts: Dict[str, Any],
        seed: int,
        resume: ResumeMode,
        keep_stale: bool,
    ) -> Tuple[StageResult, str]:
        """Return ``(result, status)`` where status is 'loaded', 'ran',
        or 'rerun-mismatch'. Handles on-disk cache lookup, invalidation,
        and stale-aside moves."""
        manifest_path = (
            os.path.join(stage_dir, _MANIFEST_FILENAME) if stage_dir else None
        )

        # Read any existing manifest: even non-cacheable stages need it for the
        # strict-mode mismatch check and the "skip rewriting unchanged manifest"
        # optimization. The full cache-load shortcut is gated on cacheable_outputs.
        manifest: Optional[Dict[str, Any]] = None
        if (resume is not False) and manifest_path and os.path.exists(manifest_path):
            try:
                manifest = _read_manifest(manifest_path)
            except (OSError, json.JSONDecodeError) as e:
                warnings.warn(
                    f"[pipeline] could not read {manifest_path} ({e!r}); rerunning {stage.instance_name}",
                    stacklevel=3,
                )

        if manifest is not None:
            cached_hash = manifest.get("input_hash")
            if cached_hash == input_hash:
                if stage.cacheable_outputs:
                    arrays = _load_stage_arrays(stage_dir)
                    diagnostics = _load_stage_diagnostics(stage_dir)
                    return (
                        StageResult(arrays=arrays,
                                    metadata=dict(manifest.get("metadata") or {}),
                                    diagnostics=diagnostics),
                        "loaded",
                    )
                # Non-cacheable (bridges): fall through to re-run, but we'll
                # skip rewriting the manifest below since nothing changed.
            else:
                if resume == "strict":
                    raise PipelineMismatchError(
                        f"Stage {stage.instance_name!r}: cached input_hash "
                        f"{cached_hash!r} != current {input_hash!r}. "
                        f"Pass resume=True to re-run, force=True to overwrite, "
                        f"or use a different out_dir."
                    )
                if jax.process_index() == 0:
                    if keep_stale:
                        moved = _move_aside(stage_dir)
                        warnings.warn(
                            f"[pipeline] {stage.instance_name}: input hash changed; "
                            f"moved old run to {moved}",
                            stacklevel=3,
                        )
                    else:
                        shutil.rmtree(stage_dir)
                manifest = None  # forget the stale one; we'll rewrite

        if (resume is False) and stage_dir and os.path.exists(stage_dir):
            if jax.process_index() == 0:
                if keep_stale:
                    _move_aside(stage_dir)
                else:
                    shutil.rmtree(stage_dir)

        # Actually run the stage.
        result = stage.run(self.ctx, artifacts, seed)
        manifest_unchanged = (
            manifest is not None and manifest.get("input_hash") == input_hash
        )
        if stage_dir is not None and jax.process_index() == 0 and not manifest_unchanged:
            new_manifest = {
                "stage": stage.instance_name,
                "class": type(stage).__name__,
                "schema_version": stage.schema_version,
                "input_hash": input_hash,
                "upstream_hashes": upstream_hashes,
                "config": _make_json_safe(stage.config_hash_data()),
                "diagnostics_config": _make_json_safe(stage.diagnostics_config()),
                "seed": seed,
                "metadata": _make_json_safe(result.metadata),
                "arrays": sorted(result.arrays.keys()),
                "diagnostics": sorted(result.diagnostics.keys()),
                "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            }
            _save_stage(stage_dir, new_manifest, result.arrays, result.diagnostics)
        return result, "computed" if manifest_unchanged else "ran"

    # -- posterior view ------------------------------------------------------

    def posterior(self, stage: Optional[str] = None):
        """Return a :class:`Posterior` for one of the stages that just ran.

        With ``stage=None``, returns the *richest* terminal posterior: the
        last sampler stage if any, else the last SVI, else the last MAP.
        """
        if not self.results:
            raise RuntimeError("Pipeline has no results yet; call .run() first.")
        if stage is None:
            stage = self._pick_terminal_stage()
        if stage not in self.results:
            raise KeyError(
                f"No result for stage {stage!r}; available: {sorted(self.results)}."
            )
        stage_obj = next(s for s in self.stages if s.instance_name == stage)
        return type(stage_obj).to_posterior(self.results[stage].arrays, self.ctx)

    def diagnostics(self, stage: str) -> "StageDiagnostics":
        """Return the captured debug diagnostics for one stage.

        Only populated when the stage was run with ``debug=True``; otherwise
        ``StageDiagnostics.arrays`` is empty (and the object is falsy). Pass
        the result to ``gigalens_research.plotting.plot_stage_diagnostics``.
        """
        if not self.results:
            raise RuntimeError("Pipeline has no results yet; call .run() first.")
        if stage not in self.results:
            raise KeyError(
                f"No result for stage {stage!r}; available: {sorted(self.results)}."
            )
        stage_obj = next(s for s in self.stages if s.instance_name == stage)
        return StageDiagnostics(
            stage_name=stage,
            stage_class=type(stage_obj).__name__,
            arrays=dict(self.results[stage].diagnostics),
            config=dict(stage_obj.diagnostics_config()),
            ctx=self.ctx,
        )

    def _pick_terminal_stage(self) -> str:
        # Order from richest to leanest posterior; pick the last entry whose
        # stage class has its own ``to_posterior`` override. Bridges and
        # other stages without a view are skipped.
        scores = {"HMCStage": 2, "MCLMCStage": 2, "SVIStage": 1, "MAPStage": 0}
        best, best_score = None, -1
        for s in self.stages:
            if s.instance_name not in self.results:
                continue
            if type(s).to_posterior is InferenceStage.to_posterior:
                continue
            score = scores.get(type(s).__name__, 1)
            if score >= best_score:
                best, best_score = s.instance_name, score
        if best is None:
            raise RuntimeError("No stage in this pipeline produces a posterior view.")
        return best

    def _publish(self, stage: InferenceStage, result: StageResult) -> Dict[str, Any]:
        """Map a stage's result to the artifact-bag entries it announced via
        ``produces``. Bridges stash their outputs in metadata since they have
        no persisted arrays to derive from."""
        stash = result.metadata.get("_bridge_artifacts")
        if stash is not None:
            return {k: stash[k] for k in stage.produces}
        return dict(stage.derive_artifacts(result.arrays))


# ---------------------------------------------------------------------------
# Concrete stages
# ---------------------------------------------------------------------------
#
# These are thin adapters over ``ModellingSequence.{MAP,SVI,HMC}`` and the
# alternate inference functions. Each stage exposes its tunable knobs as
# plain kwargs (no nested dicts) and a single ``optimizer_id`` string for
# optimizers (which optax doesn't let us hash robustly).


import optax  # noqa: E402  (kept below pipeline core for narrative clarity)
import tensorflow_probability.substrates.jax as tfp  # noqa: E402

_tfd = tfp.distributions


import inspect as _inspect  # noqa: E402

_ADABELIEF_SUPPORTS_NESTEROV = (
    "nesterov" in _inspect.signature(optax.adabelief).parameters
)


def _default_map_optimizer() -> optax.GradientTransformation:
    kwargs = dict(b1=0.95, b2=0.99)
    if _ADABELIEF_SUPPORTS_NESTEROV:
        kwargs["nesterov"] = True
    return optax.adabelief(1e-2, **kwargs)


def _default_svi_optimizer() -> optax.GradientTransformation:
    return optax.adabelief(1e-4, b1=0.95, b2=0.99)


# Suffix the MAP optimizer id with whether nesterov is actually applied; the
# hash differs across optax versions and we'd rather invalidate than lie.
_DEFAULT_MAP_OPTIMIZER_ID = (
    "adabelief_1e-2_b1_0.95_b2_0.99"
    + ("_nesterov" if _ADABELIEF_SUPPORTS_NESTEROV else "")
)


class MAPStage(InferenceStage):
    """Multi-start MAP optimization. Wraps ``ModellingSequence.MAP``.

    Produces ``z_best`` (best parameter vector in unconstrained space),
    plus per-step ``lp_hist`` and ``chisq_hist`` for diagnostics/plotting.
    """

    name: ClassVar[str] = "map"
    schema_version: ClassVar[int] = 1
    requires: ClassVar[Tuple[str, ...]] = ()
    produces: ClassVar[Tuple[str, ...]] = ("z_best", "lp_hist", "chisq_hist")

    def __init__(
        self,
        *,
        num_steps: int = 350,
        n_samples: int = 500,
        optimizer_factory: Optional[Callable[[], optax.GradientTransformation]] = None,
        optimizer_id: Optional[str] = None,
        pbar_interval: int = 5,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(name=name, seed=seed)
        self.num_steps = int(num_steps)
        self.n_samples = int(n_samples)
        self._optimizer_factory = optimizer_factory or _default_map_optimizer
        self.optimizer_id = str(optimizer_id or _DEFAULT_MAP_OPTIMIZER_ID)
        self.pbar_interval = int(pbar_interval)

    def config_hash_data(self):
        return {
            "num_steps": self.num_steps,
            "n_samples": self.n_samples,
            "optimizer_id": self.optimizer_id,
            "pbar_interval": self.pbar_interval,
        }

    def run(self, ctx, artifacts, seed):
        t0 = time.perf_counter()
        samples, lps, chisqs = ctx.model_seq.MAP(
            optimizer=self._optimizer_factory(),
            start=None,
            n_samples=self.n_samples,
            num_steps=self.num_steps,
            seed=seed,
            output_type="best_step",
            pbar_interval=self.pbar_interval,
        )
        # ``samples`` is shape (num_steps, n_params), ``lps``/``chisqs`` are
        # (num_steps,). Pick the globally best step.
        lps_np = np.asarray(lps)
        chisqs_np = np.asarray(chisqs)
        samples_np = np.asarray(samples)
        best = int(np.nanargmax(lps_np))
        return StageResult(
            arrays={
                "z_best": samples_np[best],
                "lp_hist": lps_np,
                "chisq_hist": chisqs_np,
            },
            metadata={
                "wall_time_s": time.perf_counter() - t0,
                "num_steps": self.num_steps,
                "n_samples": self.n_samples,
                "best_step": best,
                "best_lp": float(lps_np[best]),
                "best_chisq": float(chisqs_np[best]),
            },
        )

    @classmethod
    def to_posterior(cls, arrays, ctx):
        from .posterior import PointEstimate
        return PointEstimate(
            ctx,
            z_best=arrays["z_best"],
            lp_hist=arrays.get("lp_hist"),
            chisq_hist=arrays.get("chisq_hist"),
        )


class SVIStage(InferenceStage):
    """Gaussian variational inference. Wraps ``ModellingSequence.SVI``.

    Requires ``z_best`` (a starting point in unconstrained space).
    Produces ``qz`` (``tfd.MultivariateNormalTriL``) and ``svi_loss_hist``.
    """

    name: ClassVar[str] = "svi"
    schema_version: ClassVar[int] = 1
    requires: ClassVar[Tuple[str, ...]] = ("z_best",)
    produces: ClassVar[Tuple[str, ...]] = ("qz", "svi_loss_hist")

    def __init__(
        self,
        *,
        num_steps: int = 500,
        n_vi: int = 250,
        init_scales: float = 1e-3,
        optimizer_factory: Optional[Callable[[], optax.GradientTransformation]] = None,
        optimizer_id: str = "adabelief_1e-4_b1_0.95_b2_0.99",
        pbar_interval: int = 5,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(name=name, seed=seed)
        self.num_steps = int(num_steps)
        self.n_vi = int(n_vi)
        self.init_scales = float(init_scales)
        self._optimizer_factory = optimizer_factory or _default_svi_optimizer
        self.optimizer_id = str(optimizer_id)
        self.pbar_interval = int(pbar_interval)

    def config_hash_data(self):
        return {
            "num_steps": self.num_steps,
            "n_vi": self.n_vi,
            "init_scales": self.init_scales,
            "optimizer_id": self.optimizer_id,
            "pbar_interval": self.pbar_interval,
        }

    def run(self, ctx, artifacts, seed):
        t0 = time.perf_counter()
        qz, loss_hist = ctx.model_seq.SVI(
            start=artifacts["z_best"],
            optimizer=self._optimizer_factory(),
            n_vi=self.n_vi,
            init_scales=self.init_scales,
            num_steps=self.num_steps,
            seed=seed,
            pbar_interval=self.pbar_interval,
        )
        return StageResult(
            arrays={
                "qz_loc": np.asarray(qz.loc),
                "qz_scale_tril": np.asarray(qz.scale_tril),
                "svi_loss_hist": np.asarray(loss_hist),
            },
            metadata={
                "wall_time_s": time.perf_counter() - t0,
                "num_steps": self.num_steps,
                "n_vi": self.n_vi,
                "final_loss": float(np.asarray(loss_hist)[-1]),
            },
        )

    def derive_artifacts(self, arrays):
        qz = _tfd.MultivariateNormalTriL(
            loc=jnp.asarray(arrays["qz_loc"]),
            scale_tril=jnp.asarray(arrays["qz_scale_tril"]),
        )
        return {"qz": qz, "svi_loss_hist": arrays["svi_loss_hist"]}

    @classmethod
    def to_posterior(cls, arrays, ctx):
        from .posterior import SurrogatePosterior
        qz = _tfd.MultivariateNormalTriL(
            loc=jnp.asarray(arrays["qz_loc"]),
            scale_tril=jnp.asarray(arrays["qz_scale_tril"]),
        )
        return SurrogatePosterior(ctx, qz=qz, loss_hist=arrays.get("svi_loss_hist"))


class HMCStage(InferenceStage):
    """Preconditioned HMC. Wraps ``ModellingSequence.HMC``.

    Requires ``qz`` (typically from SVI). Produces ``samples_z`` of canonical
    shape ``(num_chains, num_steps, n_params)``.
    """

    name: ClassVar[str] = "hmc"
    schema_version: ClassVar[int] = 1
    requires: ClassVar[Tuple[str, ...]] = ("qz",)
    produces: ClassVar[Tuple[str, ...]] = ("samples_z",)

    def __init__(
        self,
        *,
        n_hmc: int = 50,
        num_burnin_steps: int = 250,
        num_results: int = 750,
        init_eps: float = 0.3,
        init_l: int = 3,
        max_leapfrog_steps: int = 30,
        pbar_interval: int = 0,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(name=name, seed=seed)
        self.n_hmc = int(n_hmc)
        self.num_burnin_steps = int(num_burnin_steps)
        self.num_results = int(num_results)
        self.init_eps = float(init_eps)
        self.init_l = int(init_l)
        self.max_leapfrog_steps = int(max_leapfrog_steps)
        self.pbar_interval = int(pbar_interval)

    def run(self, ctx, artifacts, seed):
        t0 = time.perf_counter()
        samples = ctx.model_seq.HMC(
            q_z=artifacts["qz"],
            init_eps=self.init_eps,
            init_l=self.init_l,
            n_hmc=self.n_hmc,
            num_burnin_steps=self.num_burnin_steps,
            num_results=self.num_results,
            max_leapfrog_steps=self.max_leapfrog_steps,
            seed=seed,
            pbar_interval=self.pbar_interval,
        )
        canonical = _to_canonical_samples(samples)
        return StageResult(
            arrays={"samples_z": canonical},
            metadata={
                "wall_time_s": time.perf_counter() - t0,
                "num_chains": int(canonical.shape[0]),
                "num_steps": int(canonical.shape[1]),
                "n_params": int(canonical.shape[2]),
            },
        )

    @classmethod
    def to_posterior(cls, arrays, ctx):
        from .posterior import SamplerPosterior
        return SamplerPosterior(ctx, samples_z=arrays["samples_z"])


class HessianSurrogateStage(InferenceStage):
    """Laplace-approximation surrogate posterior from the log-posterior Hessian.

    Requires ``z_best`` (MAP estimate in unconstrained space).
    Produces ``qz`` (``tfd.MultivariateNormalTriL`` with the Laplace
    covariance) — the same artifact that :class:`SVIStage` produces, so this
    stage is a drop-in replacement for SVI when a fast, deterministic
    preconditioner is preferred.

    The Hessian is built column-by-column using Hessian-vector products, which
    is memory-safe at high shapelet orders (no vmap over the full basis).

    Parameters
    ----------
    fix_indefinite : bool, default ``True``
        Replace non-positive eigenvalues of ``-H`` with their absolute values
        before computing the covariance.  See :func:`HessianSurrogate` for
        the full discussion.
    eigenvalue_floor : float or None
        Lower bound on eigenvalues of ``-H`` as a fraction of the largest
        eigenvalue (after flipping). ``None`` → ``1e-8 * max_eigenvalue``.
    """

    name: ClassVar[str] = "hessian_surrogate"
    schema_version: ClassVar[int] = 1
    requires: ClassVar[Tuple[str, ...]] = ("z_best",)
    produces: ClassVar[Tuple[str, ...]] = ("qz",)

    def __init__(
        self,
        *,
        fix_indefinite: bool = True,
        eigenvalue_floor: Optional[float] = None,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(name=name, seed=seed)
        self.fix_indefinite = bool(fix_indefinite)
        self.eigenvalue_floor = eigenvalue_floor

    def run(self, ctx, artifacts, seed):
        from gigalens_research.inference.hessian_surrogate import HessianSurrogate
        import time as _time
        t0 = _time.perf_counter()
        qz = HessianSurrogate(
            ctx.model_seq,
            artifacts["z_best"],
            fix_indefinite=self.fix_indefinite,
            eigenvalue_floor=self.eigenvalue_floor,
        )
        evals = np.linalg.eigvalsh(np.asarray(qz.covariance()))
        return StageResult(
            arrays={
                "qz_loc": np.asarray(qz.loc),
                "qz_scale_tril": np.asarray(qz.scale_tril),
            },
            metadata={
                "wall_time_s": _time.perf_counter() - t0,
                "fix_indefinite": self.fix_indefinite,
                "cov_eigenvalue_min": float(evals.min()),
                "cov_eigenvalue_max": float(evals.max()),
                "cov_condition_number": float(evals.max() / max(evals.min(), 1e-300)),
            },
        )

    def derive_artifacts(self, arrays):
        qz = _tfd.MultivariateNormalTriL(
            loc=jnp.asarray(arrays["qz_loc"]),
            scale_tril=jnp.asarray(arrays["qz_scale_tril"]),
        )
        return {"qz": qz}

    @classmethod
    def to_posterior(cls, arrays, ctx):
        from .posterior import SurrogatePosterior
        qz = _tfd.MultivariateNormalTriL(
            loc=jnp.asarray(arrays["qz_loc"]),
            scale_tril=jnp.asarray(arrays["qz_scale_tril"]),
        )
        return SurrogatePosterior(ctx, qz=qz)


class MCLMCStage(InferenceStage):
    """MCLMC sampler. Wraps ``gigalens_research.inference.MCLMC_JIT``.

    Requires ``qz`` (used for chain initialization, initial mass matrix, and
    SVI-mean reference). Produces ``samples_z`` of canonical shape
    ``(num_chains, num_steps, n_params)``.
    """

    name: ClassVar[str] = "mclmc"
    schema_version: ClassVar[int] = 1
    requires: ClassVar[Tuple[str, ...]] = ("qz",)
    produces: ClassVar[Tuple[str, ...]] = ("samples_z",)

    def __init__(
        self,
        *,
        n_chains: int = 16,
        num_burnin_steps: int = 1000,
        num_results: int = 2000,
        desired_energy_variance: float = 5e-4,
        init_L: Optional[float] = None,
        init_step_size: Optional[float] = None,
        frac_tune1: float = 0.2,
        frac_tune2: float = 0.6,
        frac_tune3: float = 0.2,
        regularize_mass_matrix: bool = True,
        progress_bar: bool = False,
        debug: bool = False,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(name=name, seed=seed)
        self.n_chains = int(n_chains)
        self.num_burnin_steps = int(num_burnin_steps)
        self.num_results = int(num_results)
        self.desired_energy_variance = float(desired_energy_variance)
        self.init_L = None if init_L is None else float(init_L)
        self.init_step_size = None if init_step_size is None else float(init_step_size)
        self.frac_tune1 = float(frac_tune1)
        self.frac_tune2 = float(frac_tune2)
        self.frac_tune3 = float(frac_tune3)
        self.progress_bar = bool(progress_bar)
        self.debug = bool(debug)
        self.regularize_mass_matrix = bool(regularize_mass_matrix)
    def diagnostics_config(self):
        # What the MCLMC diagnostic plotter needs to draw the tuning-stage
        # boundaries (see plotting.diagnostics.plot_mclmc_diagnostics).
        return {
            "num_burnin_steps": self.num_burnin_steps,
            "num_results": self.num_results,
            "frac_tune1": self.frac_tune1,
            "frac_tune2": self.frac_tune2,
            "frac_tune3": self.frac_tune3,
        }

    def run(self, ctx, artifacts, seed):
        # Local import: keeps MCLMC's heavy blackjax dependency optional for
        # users who only need MAP/SVI/HMC.
        from gigalens_research.inference import MCLMC_JIT
        t0 = time.perf_counter()
        out = MCLMC_JIT(
            model_seq=ctx.model_seq,
            qz=artifacts["qz"],
            n_hmc=self.n_chains,
            num_burnin_steps=self.num_burnin_steps,
            num_results=self.num_results,
            desired_energy_variance=self.desired_energy_variance,
            init_L=self.init_L,
            init_step_size=self.init_step_size,
            frac_tune1=self.frac_tune1,
            frac_tune2=self.frac_tune2,
            frac_tune3=self.frac_tune3,
            regularize_mass_matrix=self.regularize_mass_matrix,
            progress_bar=self.progress_bar,
            seed=seed,
            debug_output=self.debug,
        )
        diagnostics: Dict[str, np.ndarray] = {}
        if self.debug:
            # debug_output=True returns the full tuning `Hist`; the kept draws
            # are the last `num_results` positions. We also capture the tuning
            # traces (step_size, L, xi, success mask, mass matrix) for the
            # diagnostic plotter. The inverse mass matrix is replicated across
            # chains, so we keep only chain 0 to bound the on-disk size.
            hist = out
            samples = np.asarray(hist.position[:, -self.num_results:, :])
            diagnostics = {
                "step_size": np.asarray(hist.step_size),
                "L": np.asarray(hist.L),
                "xi": np.asarray(hist.xi),
                "nonan": np.asarray(hist.nonan),
                "inverse_mass_matrix": np.asarray(hist.inverse_mass_matrix[:1]),
            }
        else:
            samples = np.asarray(out)
        samples_np = np.asarray(samples)
        return StageResult(
            arrays={"samples_z": samples_np},
            metadata={
                "wall_time_s": time.perf_counter() - t0,
                "num_chains": int(samples_np.shape[0]),
                "num_steps": int(samples_np.shape[1]),
                "n_params": int(samples_np.shape[2]),
                "debug": self.debug,
            },
            diagnostics=diagnostics,
        )

    @classmethod
    def to_posterior(cls, arrays, ctx):
        from .posterior import SamplerPosterior
        return SamplerPosterior(ctx, samples_z=arrays["samples_z"])


def _to_canonical_samples(samples) -> np.ndarray:
    """Normalize HMC's ``(num_steps, num_devices, n_per_device, n_params)``
    layout to the canonical ``(num_chains, num_steps, n_params)``.
    """
    arr = np.asarray(samples)
    if arr.ndim == 4:
        ns, nd, npc, npar = arr.shape
        # collapse devices × per_device → chains, then move chains to front
        arr = arr.reshape(ns, nd * npc, npar)
        arr = np.swapaxes(arr, 0, 1)
    elif arr.ndim == 3:
        # Already (num_chains, num_steps, n_params)
        pass
    else:
        raise ValueError(f"Unexpected samples ndim={arr.ndim}, shape={arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# Stage registry: needed by the on-disk Posterior loader
# ---------------------------------------------------------------------------


_STAGE_REGISTRY: Dict[str, type] = {}


def register_stage(cls: type) -> type:
    """Register a stage class so that ``posterior_from_disk`` and other
    name-based lookups can find it. Built-in stages are registered at import
    time; user-defined stages should call this decorator on their class."""
    _STAGE_REGISTRY[cls.__name__] = cls
    return cls


for _cls in (MAPStage, SVIStage, HessianSurrogateStage, HMCStage, MCLMCStage, BridgeStage):
    register_stage(_cls)


def posterior_from_disk(out_dir: str, stage: str, ctx: InferenceContext):
    """Reconstruct a :class:`Posterior` from a saved stage directory.

    ``out_dir`` is the pipeline directory (parent of the stage dir); ``stage``
    is the stage's instance name. ``ctx`` is needed to reconstruct any view
    that uses the bijector or simulator.
    """
    stage_dir = os.path.join(out_dir, stage)
    manifest_path = os.path.join(stage_dir, _MANIFEST_FILENAME)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"No manifest at {manifest_path}; either the stage didn't run "
            f"or {out_dir!r} is the wrong directory."
        )
    manifest = _read_manifest(manifest_path)
    class_name = manifest.get("class")
    stage_cls = _STAGE_REGISTRY.get(class_name)
    if stage_cls is None:
        raise KeyError(
            f"Unknown stage class {class_name!r} in {manifest_path}; "
            f"register it with register_stage() before loading."
        )
    arrays = _load_stage_arrays(stage_dir)
    return stage_cls.to_posterior(arrays, ctx)


def diagnostics_from_disk(out_dir: str, stage: str, ctx: InferenceContext) -> "StageDiagnostics":
    """Reconstruct a :class:`StageDiagnostics` from a saved stage directory.

    Mirrors :func:`posterior_from_disk`. ``arrays`` is empty unless the stage
    was run with ``debug=True``. Pass the result to
    ``gigalens_research.plotting.plot_stage_diagnostics``.
    """
    stage_dir = os.path.join(out_dir, stage)
    manifest_path = os.path.join(stage_dir, _MANIFEST_FILENAME)
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"No manifest at {manifest_path}; either the stage didn't run "
            f"or {out_dir!r} is the wrong directory."
        )
    manifest = _read_manifest(manifest_path)
    return StageDiagnostics(
        stage_name=stage,
        stage_class=manifest.get("class"),
        arrays=_load_stage_diagnostics(stage_dir),
        config=dict(manifest.get("diagnostics_config") or {}),
        ctx=ctx,
    )
