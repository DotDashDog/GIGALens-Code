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

    pipeline = Pipeline(InferenceContext.from_prob_model(prob_model))
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

from gigalens_research.paths import resolve_out_dir


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
    elif isinstance(obj, np.dtype):
        # Precision specs (SimulatorConfig.likelihood/basis/conv_precision) are
        # stored as numpy dtypes; hash by canonical name so they don't reach the
        # repr() fallback. ``str(dtype)`` is stable ("float32"/"float64").
        h.update(b"dtype:"); h.update(str(obj).encode())
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
        elif callable(obj) and getattr(obj, "__qualname__", None) is not None:
            # Functions / methods / classes: hash by stable module-qualified name, NOT
            # repr() (which embeds a memory address and so changes every process,
            # silently breaking resume). TFP's TransformedDistribution stashes helper
            # functions in ``.parameters`` (e.g. ``_default_kwargs_split_fn``), which is
            # how a DiskEllipticity / any transformed-distribution prior reaches here.
            # These are fixed library callables, so a name-based key is both stable and
            # correct (their identity, not their address, is what matters).
            h.update(b"func:")
            h.update(f"{getattr(obj, '__module__', '?')}.{obj.__qualname__}".encode())
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

    Stages should treat all fields as read-only. The scene ``prob_model`` is the
    one authoritative input — the JAX-side ``MAP``/``SVI``/``HMC`` free functions
    take it directly. ``phys_model`` and ``sim_config`` are derived views,
    surfaced separately so they're easy to hash.
    """

    phys_model: Any
    prob_model: Any
    sim_config: Any

    @classmethod
    def from_prob_model(cls, prob_model) -> "InferenceContext":
        # Scene-only (old gigalens API dropped): expose a phys_model-shaped VIEW derived
        # from the scene LensModel so the read-only consumers (model_card / hash /
        # posterior) read a stable profile listing without the old PhysicalModel.
        scene_model = getattr(prob_model, "model", None)
        if scene_model is None:
            raise TypeError(
                "InferenceContext.from_prob_model requires a scene ProbModel "
                "(gigalens.jax.scene_prob_model.ProbModel), which exposes the scene "
                "LensModel as `.model`. The legacy PhysicalModel path was removed with "
                "the old gigalens API.")
        # sim_config is used only for plotting extent / source-plane FOV defaults. Take
        # the first dataset that actually has one; a point-source-only ProbModel has no
        # imaging grid, so this is None there (nothing on the fit path reads it).
        sim_config = next(
            (getattr(d, "sim_config", None) for d in prob_model.datasets
             if getattr(d, "sim_config", None) is not None), None)
        return cls(
            phys_model=_ScenePhysModelView(scene_model),
            prob_model=prob_model,
            sim_config=sim_config,
        )

    def hash(self) -> str:
        """Stable hash of the *modeling inputs*.

        Multi-dataset-aware: a scene ``ProbModel`` carries a ``datasets`` list, so the
        fingerprint folds in every band's image, noise map, mask AND that band's own
        ``sim_config`` (the per-band PSF/grid lives there, not in the single
        ``ctx.sim_config``, which is only the first band's). For a legacy single-image
        prob_model (no ``datasets``), it falls back to the singular ``observed_image``
        plus whichever noise-model attributes the model carries: ``ForwardProbModel``
        exposes ``background_rms`` / ``exp_time``; ``BackwardProbModel`` exposes
        ``err_map``. The prob_model class name is folded in so two models with
        overlapping but differently-interpreted attributes don't alias.
        """
        pm = self.prob_model
        noise: Dict[str, Any] = {"class": type(pm).__name__}
        datasets = getattr(pm, "datasets", None)
        if datasets is not None:
            # Per-band: image + noise + mask, and each band's own sim_config (PSF/grid).
            # Non-imaging datasets (point-source positions/fluxes/delays, and any
            # future kinematics/visibility Dataset) carry no image/error_map/mask;
            # they are fingerprinted by their public numeric content instead.
            noise["datasets"] = [
                {"image": np.asarray(d.image),
                 "error_map": np.asarray(d.error_map),
                 "mask": np.asarray(d.mask)}
                if hasattr(d, "image") else _dataset_content(d)
                for d in datasets
            ]
            sim_config_hash: Any = [
                _hash_sim_config(sc) if (sc := getattr(d, "sim_config", None))
                is not None else None
                for d in datasets
            ]
        else:
            noise["observed_image"] = np.asarray(pm.observed_image)
            for attr in ("background_rms", "exp_time", "err_map"):
                if hasattr(pm, attr):
                    noise[attr] = np.asarray(getattr(pm, attr))
            sim_config_hash = _hash_sim_config(self.sim_config)
        return stable_hash({
            "phys_model": _hash_phys_model(self.phys_model),
            "prior": pm.prior,
            "noise": noise,
            "sim_config": sim_config_hash,
        })


def _dataset_content(d) -> Dict[str, Any]:
    """Content fingerprint of a non-imaging scene ``Dataset`` (e.g. point-source
    observables). Collects every public scalar/array attribute — for a
    ``PointSourceObsData`` that is the observed positions/fluxes/delays, all
    sigmas, and the solver constants (newton_steps, trust_region_frac, ...), so
    any change to the observation or its likelihood configuration invalidates
    the cache. Object-valued attributes (``source_component``, ``sim_config``)
    are skipped: the profile structure is already hashed via the phys-model
    view, and sim_config is hashed separately."""
    content: Dict[str, Any] = {"class": type(d).__name__}
    for k, v in vars(d).items():
        if k.startswith("_"):
            continue
        if v is None or isinstance(v, (bool, int, float, str, np.integer,
                                       np.floating, np.ndarray, jax.Array)):
            content[k] = np.asarray(v) if isinstance(v, jax.Array) else v
    return content


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
# Model card: delegated to gigalens.jax.utils.model_card (the scene-native
# rewrite of the card that used to live here). The pipeline adds its own
# context hash as an extra and keeps the historical call surface
# (model_card(ctx) / format_model_card(card)) so stages, experiment scripts
# and Pipeline.run are unchanged. The card is a REPORT: hard validation lives
# in the gigalens constructors/simulators, not here.
# ---------------------------------------------------------------------------


def model_card(ctx: "InferenceContext") -> Dict[str, Any]:
    """Return the JSON-safe scene model card for ``ctx``.

    Delegates to :func:`gigalens.jax.utils.model_card` on ``ctx.prob_model``
    (planes/datasets/adaptive-supersampling/point-source aware, structured
    ``advisories`` instead of free-text ``warnings``), adding the pipeline's
    stable input hash under ``card["extras"]["context_hash"]``. Written to
    ``<out_dir>/model_card.json`` by :meth:`Pipeline.run` and printed via
    :func:`format_model_card`.
    """
    from gigalens.jax.utils import model_card as _scene_model_card

    extras: Dict[str, Any] = {}
    try:
        extras["context_hash"] = ctx.hash()
    except Exception as exc:  # reporting must never break the run
        extras["context_hash"] = f"unavailable: {type(exc).__name__}: {exc}"
    return _scene_model_card(ctx.prob_model, extras=extras)


def format_model_card(card: Dict[str, Any]) -> str:
    """Human-readable one-screen summary of :func:`model_card`."""
    from gigalens.jax.utils import format_model_card as _fmt

    return _fmt(card)


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
        ``InferenceContext.from_prob_model(prob_model)``.
    seed : int
        Default seed for stages that don't set their own.

    Examples
    --------
    >>> p = Pipeline(InferenceContext.from_prob_model(prob_model))
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
            A *relative* path is resolved under the results root
            (``$GIGALENS_RESULTS_ROOT`` or ``$PSCRATCH/gigalens``; see
            :func:`gigalens_research.paths.results_root`); an absolute path is
            used verbatim.
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

        out_dir = resolve_out_dir(out_dir)
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
        scores = {"HMCStage": 2, "NUTSStage": 2, "MCLMCStage": 2, "MAMSStage": 2,
                  "SVIStage": 1, "MAPStage": 0}
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
# These are thin adapters over ``gigalens.jax.inference.{MAP,SVI,HMC}`` and the
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
    """Multi-start MAP optimization. Wraps ``gigalens.jax.inference.MAP``.

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
        from gigalens.jax.inference import MAP as _MAP
        samples, lps, chisqs = _MAP(
            ctx.prob_model,
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
    """Gaussian variational inference. Wraps ``gigalens.jax.inference.SVI``.

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
        from gigalens.jax.inference import SVI as _SVI
        qz, loss_hist = _SVI(
            ctx.prob_model,
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
    """Preconditioned HMC. Wraps ``gigalens.jax.inference.HMC``.

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
        from gigalens.jax.inference import HMC as _HMC
        samples = _HMC(
            ctx.prob_model,
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
            ctx.prob_model,
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
            prob_model=ctx.prob_model,
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
            # Empirical covariance of the kept draws (chains flattened together),
            # so the plotter can overlay the posterior-covariance eigenvalue
            # spread on the inverse-mass-matrix panel. Shape (n_params, n_params),
            # so it's cheap to store alongside the tuning traces.
            flat = samples.reshape(-1, samples.shape[-1])
            samples_cov = np.cov(flat, rowvar=False)
            diagnostics = {
                "step_size": np.asarray(hist.step_size),
                "L": np.asarray(hist.L),
                "xi": np.asarray(hist.xi),
                "nonan": np.asarray(hist.nonan),
                "inverse_mass_matrix": np.asarray(hist.inverse_mass_matrix[:1]),
                "samples_cov": np.asarray(samples_cov),
                # The kept draws themselves (unconstrained z-space), so the
                # surrogate corner plot can compare them against an MVN built
                # from the final inverse mass matrix. Duplicates the published
                # samples but only when debug=True.
                "samples_z": np.asarray(samples),
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


class PTMCLMCStage(InferenceStage):
    """Parallel-tempered MCLMC for MULTIMODAL posteriors. Wraps
    ``gigalens.jax.experimental.pt_mclmc.sample_pt_mclmc``.

    Use this instead of :class:`MCLMCStage` when you suspect (or know) the
    posterior has well-separated basins/modes that a single beta=1 MCLMC
    chain essentially never crosses -- see that module's own docstring for
    the full per-knob manual; this docstring intentionally does not
    duplicate it (every constructor argument below mirrors
    ``sample_pt_mclmc``'s, one-line summary only). Validated on exactly ONE
    system so far: a 33-dimensional strong-lensing (dPIE, "carousel")
    posterior with two basins (see
    ``GIGALens-Code/docs/logs/carousel-mclmc-sampling.md``, gates
    PT-0b/PT-1/PT-2/PT-6).

    Point-and-go entry point: requires only ``z_best`` (a MAP estimate), NOT
    ``qz`` -- every chain starts at ``z_best`` plus independent
    ``init_scale``-jitter, with a matching diagonal seed metric
    (``init_scale**2 * I``). This is the validated harness's cold-start "D2"
    entry point; there is no support for seeding different walkers in
    different basins.

    Cross-repo dependency: this stage only runs against a gigalens build
    that includes the experimental ``pt_mclmc`` module (added on branch
    ``pt-mclmc-experimental``, merged into gigalens ``linusu-dev-merge`` via
    PR seanxuseanxu/gigalens#66 on 2026-07-19; not on gigalens ``main`` or
    in any release); ``run`` raises a descriptive ``ImportError`` if it is
    missing.

    Requires ``z_best``. Produces ``samples_z`` of canonical shape
    ``(n_walkers, n_rounds - num_burnin_rounds, dim)`` -- the cold
    (``beta=1``) rung's post-burn-in draws.
    """

    name: ClassVar[str] = "pt_mclmc"
    schema_version: ClassVar[int] = 1
    requires: ClassVar[Tuple[str, ...]] = ("z_best",)
    produces: ClassVar[Tuple[str, ...]] = ("samples_z",)

    def __init__(
        self,
        *,
        beta_min: float,
        n_rungs: Optional[int] = None,
        betas: Optional[Iterable[float]] = None,
        n_walkers: int = 8,
        steps_per_round: int = 10,
        n_rounds: int = 2000,
        num_burnin_rounds: int = 1000,
        init_scale: float = 1e-3,
        adapt_metric: bool = True,
        metric_windows: Iterable[int] = (100, 250, 500),
        metric_estimator: str = "pooled",
        eevpd_target: float = 5e-4,
        step_size_init: float = 0.05,
        step_size_max: float = 5.0,
        decoherence_length: Optional[float] = None,
        indicator: Optional[Callable[[Any], Any]] = None,
        indicator_id: Optional[str] = None,
        progress_every: int = 0,
        debug: bool = False,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(name=name, seed=seed)
        # beta_min: hottest-rung inverse temperature. REQUIRED, no default --
        # the single most consequential knob (see sample_pt_mclmc's docstring
        # for the full manual). Tested 0.36 on the validated 33-dim system.
        self.beta_min = float(beta_min)
        self.n_rungs = None if n_rungs is None else int(n_rungs)
        self.betas = None if betas is None else tuple(float(b) for b in betas)
        self.n_walkers = int(n_walkers)
        self.steps_per_round = int(steps_per_round)
        self.n_rounds = int(n_rounds)
        # num_burnin_rounds: rounds DISCARDED from the published samples_z
        # while the cold chain equilibrates from its init basin (550-1500 on
        # the validated system, ladder-dependent). Must be < n_rounds.
        self.num_burnin_rounds = int(num_burnin_rounds)
        if self.num_burnin_rounds >= self.n_rounds:
            raise ValueError(
                f"num_burnin_rounds ({self.num_burnin_rounds}) must be < "
                f"n_rounds ({self.n_rounds}) -- otherwise zero post-burn-in "
                f"rounds would be published as samples_z."
            )
        self.init_scale = float(init_scale)
        self.adapt_metric = bool(adapt_metric)
        self.metric_windows = tuple(int(w) for w in metric_windows)
        self.metric_estimator = str(metric_estimator)
        self.eevpd_target = float(eevpd_target)
        self.step_size_init = float(step_size_init)
        self.step_size_max = float(step_size_max)
        self.decoherence_length = (
            None if decoherence_length is None else float(decoherence_length)
        )
        # indicator: optional diagnostic-only basin labeler, an unhashable
        # callable -- follows the MAPStage optimizer_id pattern: stored
        # privately (excluded from config_hash_data's public-attrs default),
        # with indicator_id (a short version string) required alongside it so
        # something hashable still feeds the cache input-hash.
        if indicator is not None and indicator_id is None:
            raise ValueError(
                "indicator was given without indicator_id: indicator is an "
                "unhashable callable and cannot contribute to the stage's "
                "config-hash cache key directly, so a short version string "
                "identifying it (e.g. 'z6>-22.35_v1') is required whenever "
                "indicator is supplied -- it feeds the cache input-hash in "
                "its place."
            )
        self._indicator = indicator
        self.indicator_id = indicator_id
        self.progress_every = int(progress_every)
        self.debug = bool(debug)

    def diagnostics_config(self):
        # What the PT-MCLMC diagnostic plotter needs (see
        # plotting.diagnostics.plot_pt_mclmc_diagnostics).
        return {
            "n_rounds": self.n_rounds,
            "num_burnin_rounds": self.num_burnin_rounds,
            "metric_windows": self.metric_windows,
            "steps_per_round": self.steps_per_round,
            "beta_min": self.beta_min,
            "n_walkers": self.n_walkers,
            "eevpd_target": self.eevpd_target,
        }

    def run(self, ctx, artifacts, seed):
        # Local import: the experimental PT-MCLMC sampler lives outside this
        # repo (gigalens branch pt-mclmc-experimental, PR #66) and is not part
        # of any merged gigalens release yet.
        try:
            from gigalens.jax.experimental.pt_mclmc import sample_pt_mclmc
        except ImportError as e:
            raise ImportError(
                "PTMCLMCStage requires a gigalens build that includes "
                "gigalens.jax.experimental.pt_mclmc (added on gigalens branch "
                "pt-mclmc-experimental; on linusu-dev-merge since PR #66, "
                "2026-07-19). Your installed gigalens does not have it."
            ) from e

        # Same seam as MCLMC_JIT (inference/mclmc.py): log_prob(z) ->
        # log_prob only (drop the reduced-chi2 companion value).
        def log_prob(z):
            return ctx.prob_model.log_prob(z)[0]

        z_init = np.asarray(artifacts["z_best"], dtype=np.float64).reshape(-1)

        t0 = time.perf_counter()
        result = sample_pt_mclmc(
            log_prob,
            z_init,
            beta_min=self.beta_min,
            n_rungs=self.n_rungs,
            betas=self.betas,
            n_walkers=self.n_walkers,
            steps_per_round=self.steps_per_round,
            n_rounds=self.n_rounds,
            seed=seed,
            init_scale=self.init_scale,
            adapt_metric=self.adapt_metric,
            metric_windows=self.metric_windows,
            metric_estimator=self.metric_estimator,
            eevpd_target=self.eevpd_target,
            step_size_init=self.step_size_init,
            step_size_max=self.step_size_max,
            decoherence_length=self.decoherence_length,
            indicator=self._indicator,
            progress_every=self.progress_every,
            store_all_rungs=False,
        )
        wall_time_s = time.perf_counter() - t0

        # Canonical (num_chains, num_steps, n_params) layout: drop burn-in
        # rounds, then move the walker axis first (cold_positions is
        # (n_rounds, n_walkers, dim)).
        samples_z = np.asarray(
            result.cold_positions[self.num_burnin_rounds:]
        ).transpose(1, 0, 2)

        round_trips_total = int(np.asarray(result.round_trips).sum())
        transport_warning = round_trips_total == 0
        if transport_warning:
            warnings.warn(
                "PTMCLMCStage: ladder never completed a round trip -- for a "
                "multimodal target these samples are unreliable; check "
                "beta_min / .summary() diagnostics",
                stacklevel=2,
            )

        swap_attempts = np.asarray(result.swap_attempts, dtype=np.float64)
        swap_accepts = np.asarray(result.swap_accepts, dtype=np.float64)
        swap_acceptance = np.divide(
            swap_accepts, swap_attempts,
            out=np.full_like(swap_accepts, np.nan),
            where=swap_attempts > 0,
        ).tolist()

        metadata = {
            "wall_time_s": wall_time_s,
            "num_chains": int(samples_z.shape[0]),
            "num_steps": int(samples_z.shape[1]),
            "n_params": int(samples_z.shape[2]),
            "betas": np.asarray(result.betas, dtype=np.float64).tolist(),
            "n_rungs": int(result.n_rungs),
            "num_burnin_rounds": int(self.num_burnin_rounds),
            "swap_acceptance": swap_acceptance,
            "round_trips_total": round_trips_total,
            "n_nan_reverts": int(result.n_nan_reverts),
            "u0_identity_rel": float(result.u0_identity_rel),
            "metric_frozen": bool(result.metric_frozen),
            "transport_warning": bool(transport_warning),
            "debug": self.debug,
        }

        diagnostics: Dict[str, np.ndarray] = {}
        if self.debug:
            # Full-run (all rounds, incl. burn-in) diagnostic arrays from the
            # PTMCLMCResult -- never published as samples_z, only kept for the
            # diagnostic plotter.
            diagnostics = {
                "cold_positions": np.asarray(result.cold_positions),
                "cold_logdensity": np.asarray(result.cold_logdensity),
                "eevpd": np.asarray(result.eevpd),
                "step_size_mean": np.asarray(result.step_size_mean),
                "swap_attempts": np.asarray(result.swap_attempts),
                "swap_accepts": np.asarray(result.swap_accepts),
                "betas": np.asarray(result.betas),
                "inv_mass_final": np.asarray(result.inv_mass_final),
                "round_trips": np.asarray(result.round_trips),
            }
            if self._indicator is not None:
                diagnostics.update({
                    "cold_indicator": np.asarray(result.cold_indicator),
                    "round_trips_pocket": np.asarray(result.round_trips_pocket),
                    "round_trips_main": np.asarray(result.round_trips_main),
                    "swap_attempts_by_class": np.asarray(result.swap_attempts_by_class),
                    "swap_accepts_by_class": np.asarray(result.swap_accepts_by_class),
                })

        return StageResult(
            arrays={"samples_z": samples_z},
            metadata=metadata,
            diagnostics=diagnostics,
        )

    @classmethod
    def to_posterior(cls, arrays, ctx):
        from .posterior import SamplerPosterior
        return SamplerPosterior(ctx, samples_z=arrays["samples_z"])


class MAMSStage(InferenceStage):
    """MAMS sampler (Metropolis-adjusted microcanonical). Wraps
    ``gigalens_research.inference.MAMS_JIT``.

    The Metropolis-adjusted counterpart to :class:`MCLMCStage`: same interface
    and outputs, but asymptotically unbiased (each trajectory ends in an
    accept/reject step), with the step size tuned by dual averaging to a target
    acceptance rate rather than to an energy-variance setpoint.

    Requires ``qz`` (used for chain initialization, initial mass matrix, and
    SVI-mean reference). Produces ``samples_z`` of canonical shape
    ``(num_chains, num_steps, n_params)``.
    """

    name: ClassVar[str] = "mams"
    schema_version: ClassVar[int] = 1
    requires: ClassVar[Tuple[str, ...]] = ("qz",)
    produces: ClassVar[Tuple[str, ...]] = ("samples_z",)

    def __init__(
        self,
        *,
        n_chains: int = 16,
        num_burnin_steps: int = 1000,
        num_results: int = 2000,
        target_acceptance: float = 0.9,
        init_L: Optional[float] = None,
        init_step_size: Optional[float] = None,
        frac_tune1: float = 0.2,
        frac_tune2: float = 0.6,
        frac_tune3: float = 0.2,
        regularize_mass_matrix: bool = True,
        L_max_ratio: float = 4.0,
        max_integration_steps: int = 60,
        progress_bar: bool = False,
        debug: bool = False,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(name=name, seed=seed)
        self.n_chains = int(n_chains)
        self.num_burnin_steps = int(num_burnin_steps)
        self.num_results = int(num_results)
        self.target_acceptance = float(target_acceptance)
        self.init_L = None if init_L is None else float(init_L)
        self.init_step_size = None if init_step_size is None else float(init_step_size)
        self.frac_tune1 = float(frac_tune1)
        self.frac_tune2 = float(frac_tune2)
        self.frac_tune3 = float(frac_tune3)
        self.progress_bar = bool(progress_bar)
        self.debug = bool(debug)
        self.regularize_mass_matrix = bool(regularize_mass_matrix)
        self.L_max_ratio = float(L_max_ratio)
        self.max_integration_steps = int(max_integration_steps)

    def diagnostics_config(self):
        # What the MAMS diagnostic plotter needs to draw the tuning-stage
        # boundaries and the acceptance-rate target line (see
        # plotting.diagnostics.plot_mams_diagnostics).
        return {
            "num_burnin_steps": self.num_burnin_steps,
            "num_results": self.num_results,
            "frac_tune1": self.frac_tune1,
            "frac_tune2": self.frac_tune2,
            "frac_tune3": self.frac_tune3,
            "target_acceptance": self.target_acceptance,
        }

    def run(self, ctx, artifacts, seed):
        # Local import: keeps MAMS's heavy blackjax dependency optional for
        # users who only need MAP/SVI/HMC.
        from gigalens_research.inference import MAMS_JIT
        t0 = time.perf_counter()
        out = MAMS_JIT(
            prob_model=ctx.prob_model,
            qz=artifacts["qz"],
            n_hmc=self.n_chains,
            num_burnin_steps=self.num_burnin_steps,
            num_results=self.num_results,
            target_acceptance=self.target_acceptance,
            init_L=self.init_L,
            init_step_size=self.init_step_size,
            frac_tune1=self.frac_tune1,
            frac_tune2=self.frac_tune2,
            frac_tune3=self.frac_tune3,
            regularize_mass_matrix=self.regularize_mass_matrix,
            L_max_ratio=self.L_max_ratio,
            max_integration_steps=self.max_integration_steps,
            progress_bar=self.progress_bar,
            seed=seed,
            debug_output=self.debug,
        )
        diagnostics: Dict[str, np.ndarray] = {}
        if self.debug:
            # debug_output=True returns the full tuning `Hist`; the kept draws
            # are the last `num_results` positions. We also capture the tuning
            # traces (step_size, L, acceptance_rate, trajectory length in
            # integrator steps, success mask, mass matrix) for the diagnostic
            # plotter. The inverse mass matrix is replicated across chains, so we
            # keep only chain 0 to bound the on-disk size.
            hist = out
            samples = np.asarray(hist.position[:, -self.num_results:, :])
            # Empirical covariance of the kept draws (chains flattened together),
            # so the plotter can overlay the posterior-covariance eigenvalue
            # spread on the inverse-mass-matrix panel.
            flat = samples.reshape(-1, samples.shape[-1])
            samples_cov = np.cov(flat, rowvar=False)
            diagnostics = {
                "step_size": np.asarray(hist.step_size),
                "L": np.asarray(hist.L),
                "acceptance_rate": np.asarray(hist.acceptance_rate),
                "num_integration_steps": np.asarray(hist.num_integration_steps),
                "nonan": np.asarray(hist.nonan),
                "inverse_mass_matrix": np.asarray(hist.inverse_mass_matrix[:1]),
                "samples_cov": np.asarray(samples_cov),
                # The kept draws themselves (unconstrained z-space), so the
                # surrogate corner plot can compare them against an MVN built
                # from the final inverse mass matrix. Duplicates the published
                # samples but only when debug=True.
                "samples_z": np.asarray(samples),
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


class NUTSStage(InferenceStage):
    """NUTS with per-chain window adaptation. Wraps
    ``gigalens.jax.inference.NUTS``.

    The wrapped implementation is EXPERIMENTAL (restored WIP code; see its
    module docstring) and is being validated separately -- this adapter only
    standardizes its pipeline contract and persists its gradient-evaluation
    accounting; it does not certify the sampler.

    Requires ``qz`` (chain initialization and the initial inverse mass matrix
    handed to window adaptation). Produces ``samples_z`` of canonical shape
    ``(num_chains, num_steps, n_params)``.

    ``num_burnin_steps`` has NO default by design (benchmark-campaign policy:
    burn-in budgets must be an explicit, recorded choice).

    With ``count_grad_evals=True`` (the default) the stage persists the
    per-chain gradient-evaluation curves into its diagnostics:

    - ``grad_evals_burnin``: shape ``(n_chains,)``, warmup gradient count per
      chain (one gradient per leapfrog step; velocity Verlet).
    - ``grad_evals_cumulative``: shape ``(n_chains, num_results)``, entry
      ``[c, t]`` = total gradients chain ``c`` had spent once retained step
      ``t`` was complete, burn-in included -- the curve to pair with a
      running R-hat ("gradients to R-hat < 1.01").
    """

    name: ClassVar[str] = "nuts"
    schema_version: ClassVar[int] = 1
    requires: ClassVar[Tuple[str, ...]] = ("qz",)
    produces: ClassVar[Tuple[str, ...]] = ("samples_z",)

    def __init__(
        self,
        *,
        num_burnin_steps: int,
        n_chains: int = 16,
        num_results: int = 500,
        init_step_size: float = 1.0,
        target_acceptance_rate: float = 0.8,
        max_tree_depth: int = 8,
        count_grad_evals: bool = True,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(name=name, seed=seed)
        self.n_chains = int(n_chains)
        self.num_burnin_steps = int(num_burnin_steps)
        self.num_results = int(num_results)
        self.init_step_size = float(init_step_size)
        self.target_acceptance_rate = float(target_acceptance_rate)
        self.max_tree_depth = int(max_tree_depth)
        self.count_grad_evals = bool(count_grad_evals)

    def diagnostics_config(self):
        return {
            "num_burnin_steps": self.num_burnin_steps,
            "num_results": self.num_results,
            "target_acceptance_rate": self.target_acceptance_rate,
            "max_tree_depth": self.max_tree_depth,
        }

    def run(self, ctx, artifacts, seed):
        # Local import: keeps NUTS's blackjax dependency optional for users
        # who only need MAP/SVI/HMC (mirrors MCLMCStage/MAMSStage).
        from gigalens.jax.inference import NUTS as _NUTS
        t0 = time.perf_counter()
        out = _NUTS(
            ctx.prob_model,
            q_z=artifacts["qz"],
            n_chains=self.n_chains,
            num_burnin_steps=self.num_burnin_steps,
            num_results=self.num_results,
            init_step_size=self.init_step_size,
            target_acceptance_rate=self.target_acceptance_rate,
            max_tree_depth=self.max_tree_depth,
            seed=seed,
            count_grad_evals=self.count_grad_evals,
        )
        diagnostics: Dict[str, np.ndarray] = {}
        metadata: Dict[str, Any] = {}
        if self.count_grad_evals:
            samples, counts = out
            burnin = np.asarray(counts.burnin)
            cumulative = np.asarray(counts.cumulative)
            diagnostics = {
                "grad_evals_burnin": burnin,
                "grad_evals_cumulative": cumulative,
            }
            metadata = {
                # Total = summed over chains; sequential = the critical path
                # (slowest chain), the wall-clock-relevant count on
                # chain-parallel hardware.
                "grad_evals_total": int(cumulative[:, -1].sum()),
                "grad_evals_sequential": int(cumulative[:, -1].max()),
                "grads_per_step": 1,
                "integrator": "velocity_verlet",
            }
        else:
            samples = out
        # NUTS already returns the canonical (n_chains, num_results, n_params).
        samples_np = np.asarray(samples)
        return StageResult(
            arrays={"samples_z": samples_np},
            metadata={
                "wall_time_s": time.perf_counter() - t0,
                "num_chains": int(samples_np.shape[0]),
                "num_steps": int(samples_np.shape[1]),
                "n_params": int(samples_np.shape[2]),
                **metadata,
            },
            diagnostics=diagnostics,
        )

    @classmethod
    def to_posterior(cls, arrays, ctx):
        from .posterior import SamplerPosterior
        return SamplerPosterior(ctx, samples_z=arrays["samples_z"])


# Sentinel distinguishing "user didn't pass this kwarg" (use the preset value)
# from "user explicitly passed None" (e.g. ``p2_resample_at_chunk=None`` to
# force the resampler off from an otherwise-cold preset). Plain ``None``
# defaults can't do this because ``None`` is itself a meaningful certified
# value (it's the warm preset's "resampler off").
_UNSET = object()

# Certified LAPS hyperparameters (Robnik & Seljak 2026 port), extracted
# verbatim from the validation drivers -- do NOT hand-tune these here; if the
# certified numbers change, re-derive from the driver + re-certify, then
# update this dict.
#
# Cold ("prior" init, no MAP/SVI needed): certified at both M=128
# (``experiments/laps_validation/handoff/diag_resample128.py``, arm "R128a")
# and M=512 (``.../diag_resample.py``, arm "R1"); the M=128 point is the
# default operating point (cheaper), with ``p2_resample_min_survivors=24``
# (the M=128 guard -- diag_resample128.py's ``MIN_SURV_128``). The M=512 arm
# used the sampler's own default (32) and is not reproduced verbatim here
# (num_chains=128 is the stage default; pass ``num_chains=512,
# p2_resample_min_survivors=32`` to reproduce the 512-chain arm exactly).
# ``num_unadjusted_steps`` is the sampler's own default (300) in both drivers
# (never overridden by the certified arms). ``num_adjusted_steps=248``
# (T2a=13 + T2b=18 chunks @ the sampler's default ``p2_chunk_size=8``),
# ``p2_resample_at_chunk=13`` (T2a), ``early_stop=False``,
# ``track_chains=True`` throughout.
_LAPS_PRESETS: Dict[str, Dict[str, Any]] = {
    "cold": dict(
        init_mode="prior",
        num_chains=128,
        num_unadjusted_steps=300,
        num_adjusted_steps=248,
        early_stop=False,
        track_chains=True,
        p2_resample_at_chunk=13,
        p2_resample_min_survivors=24,
        p2_resample_mode="replace",
    ),
    # Warm (qz/SVI-surrogate init): certified at M=128 ("W128") and M=512
    # ("W"), same budget as the cold arms, NO resample kwargs -- the resample
    # lever is a no-op at its default (``p2_resample_at_chunk=None``) and the
    # drivers never turn it on for the warm arm.
    "warm": dict(
        init_mode="warm",
        num_chains=128,
        num_unadjusted_steps=300,
        num_adjusted_steps=248,
        early_stop=False,
        track_chains=True,
        p2_resample_at_chunk=None,
        p2_resample_min_survivors=32,  # sampler default; inert (resampler off)
        p2_resample_mode="replace",    # sampler default; inert (resampler off)
    ),
}


class LAPSStage(InferenceStage):
    """LAPS (Late-Adjusted Parallel Sampler; Robnik & Seljak 2026). Wraps
    ``gigalens_research.inference.laps_late_adjusted.LAPS_late_adjusted_JIT``.

    Two certified init presets (``init="cold"|"warm"``), matching the
    validated configurations in ``experiments/laps_validation/handoff/
    diag_resample.py`` (512 chains) and ``diag_resample128.py`` (128 chains,
    the default operating point here); see ``docs/logs/
    laps_prior_init_investigation.md`` (DC-7.3/DC-7.4) for the certification.

    * ``init="cold"``: no upstream artifact needed. Draws initial positions
      from the model's PRIOR (``init_mode="prior"``: constrained prior draw
      mapped through the bijector inverse to unconstrained space -- the
      "RECOMMENDED robust cold-start", see ``LAPS_late_adjusted_JIT``'s
      docstring), then rescues the resulting straggler mixture with the
      certified mid-Phase-2 resample lever (``p2_resample_at_chunk=13`` +
      companions). Requires nothing upstream (``requires == ()``).
    * ``init="warm"``: draws initial positions from the ``qz`` surrogate
      (SVI or Hessian-surrogate) the way :class:`MCLMCStage`/:class:`HMCStage`
      do. Requires ``qz`` (``requires == ("qz",)``).

    Any certified default can be overridden by passing it explicitly to the
    constructor (the override wins); anything not listed as a constructor
    kwarg can still be passed through ``extra_kwargs`` (forwarded verbatim to
    ``LAPS_late_adjusted_JIT``, e.g. ``schedule``, ``velocity_init``,
    ``p2_keep_per_chain``). Requires ``qz`` only in warm mode -- see
    :attr:`requires`, which is computed from the resolved ``init_mode``.
    """

    name: ClassVar[str] = "laps"
    schema_version: ClassVar[int] = 1
    produces: ClassVar[Tuple[str, ...]] = ("samples_z",)

    def __init__(
        self,
        init: str = "cold",
        *,
        num_chains: Any = _UNSET,
        num_unadjusted_steps: Any = _UNSET,
        num_adjusted_steps: Any = _UNSET,
        early_stop: Any = _UNSET,
        track_chains: Any = _UNSET,
        p2_resample_at_chunk: Any = _UNSET,
        p2_resample_min_survivors: Any = _UNSET,
        p2_resample_mode: Any = _UNSET,
        extra_kwargs: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(name=name, seed=seed)
        if init not in ("cold", "warm"):
            raise ValueError(f"init must be 'cold' or 'warm', got {init!r}")
        self.init = init

        preset = dict(_LAPS_PRESETS[init])
        overrides = dict(
            num_chains=num_chains,
            num_unadjusted_steps=num_unadjusted_steps,
            num_adjusted_steps=num_adjusted_steps,
            early_stop=early_stop,
            track_chains=track_chains,
            p2_resample_at_chunk=p2_resample_at_chunk,
            p2_resample_min_survivors=p2_resample_min_survivors,
            p2_resample_mode=p2_resample_mode,
        )
        for key, value in overrides.items():
            if value is not _UNSET:
                preset[key] = value
        self.config = preset
        self.extra_kwargs = dict(extra_kwargs or {})

    @property
    def requires(self) -> Tuple[str, ...]:  # type: ignore[override]
        # Only the warm preset's qz-surrogate init needs an upstream qz; cold
        # (prior) init needs nothing (see class docstring).
        return ("qz",) if self.config["init_mode"] == "warm" else ()

    def diagnostics_config(self):
        # Plot-relevant config for a future LAPS diagnostic plotter (phase
        # budgets + resample chunk), mirroring MCLMCStage.diagnostics_config.
        return dict(self.config)

    def run(self, ctx, artifacts, seed):
        # Local import: keeps LAPS's heavy blackjax dependency optional for
        # users who only need MAP/SVI/HMC (mirrors MCLMCStage/
        # HessianSurrogateStage's local imports).
        from gigalens_research.inference.laps_late_adjusted import (
            LAPS_late_adjusted_JIT,
        )

        t0 = time.perf_counter()
        kwargs = dict(self.config)
        init_mode = kwargs.pop("init_mode")
        num_chains = kwargs.pop("num_chains")
        kwargs.update(self.extra_kwargs)

        res = LAPS_late_adjusted_JIT(
            ctx.prob_model,
            artifacts.get("qz"),
            init_mode=init_mode,
            num_chains=num_chains,
            seed=seed,
            **kwargs,
        )

        # res.samples is already (num_chains, p2_keep_per_chain, dim) --
        # exactly the canonical (num_chains, num_steps, n_params) layout other
        # stages normalize to (p2_keep_per_chain plays the role of "steps";
        # p2_keep_per_chain=1, the default, reproduces the classic
        # one-sample-per-chain final ensemble).
        samples_np = np.asarray(res.samples)

        resample_info = res.resample_info
        resample_summary = None
        if resample_info is not None:
            resample_summary = {
                "chunk": int(resample_info["chunk"]),
                "skipped": bool(resample_info["skipped"]),
                "mode": resample_info.get("mode"),
                "n_survivors": int(resample_info["n_survivors"]),
                "n_stragglers": int(resample_info["n_stragglers"]),
                "cut": float(resample_info["cut"]),
                "eps0_rs": (
                    float(resample_info["eps0_rs"])
                    if resample_info.get("eps0_rs") is not None
                    else None
                ),
            }

        diagnostics: Dict[str, np.ndarray] = {
            "p1_step_size": np.asarray(res.p1_step_size),
            "p1_L": np.asarray(res.p1_L),
            "p1_D_tilde": np.asarray(res.p1_D_tilde),
            "p1_eevpd_obs": np.asarray(res.p1_eevpd_obs),
            "p1_eevpd_wanted": np.asarray(res.p1_eevpd_wanted),
            "p1_delta_max": np.asarray(res.p1_delta_max),
            "p1_nan_frac": np.asarray(res.p1_nan_frac),
            "p2_step_size": np.asarray(res.p2_step_size),
            "p2_accept": np.asarray(res.p2_accept),
            "p2_frozen": np.asarray(res.p2_frozen),
            "p2_settled_accept": np.asarray(res.p2_settled_accept),
            "precond_var": np.asarray(res.precond_var),
        }
        if resample_info is not None:
            if resample_info.get("stragglers") is not None:
                diagnostics["resample_stragglers"] = np.asarray(
                    resample_info["stragglers"])
            if resample_info.get("donors") is not None:
                diagnostics["resample_donors"] = np.asarray(
                    resample_info["donors"])

        return StageResult(
            arrays={"samples_z": samples_np},
            metadata={
                "wall_time_s": time.perf_counter() - t0,
                "num_chains": int(samples_np.shape[0]),
                "num_steps": int(samples_np.shape[1]),
                "n_params": int(samples_np.shape[2]),
                "n_samples_total": int(res.n_samples_total),
                "init": self.init,
                "init_mode": init_mode,
                "phase1_len": int(res.phase1_len),
                "switched": bool(res.switched),
                "switch_index": int(res.switch_index),
                "integrator_order": int(res.integrator_order),
                "target_accept": float(res.target_accept),
                "p2_final_step_size": float(res.p2_final_step_size),
                "p2_accept_last": float(np.asarray(res.p2_accept)[-1]),
                "resample_info": resample_summary,
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


for _cls in (MAPStage, SVIStage, HessianSurrogateStage, HMCStage, MCLMCStage,
             MAMSStage, LAPSStage, BridgeStage):
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
