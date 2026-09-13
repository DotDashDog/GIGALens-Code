"""Frozen source-plane domains with a differentiable interpolation operator.

A :class:`SourceDomain` is *static geometry*: vertex positions in the source plane and
the connectivity between them. It is built once from a pilot model and never changes
inside an inference algorithm (a Delaunay flip is a discontinuity of the likelihood;
a moving vertex can fold). The only mass dependence of a pixelized source is then the
ray-traced position at which the domain is *evaluated*, which makes the likelihood a
continuous, almost-everywhere differentiable function of the mass parameters.

Every domain exposes one operator, :meth:`SourceDomain.locate`: for source-plane
points it returns the ``K`` vertex ids and interpolation weights per point (``K = 4``
bilinear, ``K = 3`` barycentric), with all-zero weights outside the domain. Two
derived views are built on it:

* :meth:`SourceDomain.basis` -- the ``(I, ...)`` stack of one basis function per
  vertex (what an lstsq light profile returns);
* :meth:`SourceDomain.interpolate` -- ``sum_i values_i * phi_i(x, y)`` (what a
  forward-mode profile with explicit vertex values returns).

Both are pure JAX, jittable, and differentiable in the query points; the discrete
choice (which cell / triangle) carries no gradient, which is correct because a
piecewise-linear interpolant is continuous across cell boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

__all__ = [
    "SourceDomain",
    "RegularGridDomain",
    "TriangleMeshDomain",
    "adaptive_delaunay_domain",
    "kmeans_weights",
]


class SourceDomain:
    """Interface: static source-plane geometry + interpolation operator.

    Attributes set by every implementation:

    * ``vertex_xy`` -- ``(I, 2)`` float64 source-plane vertex positions;
    * ``edges`` -- ``(E, 2)`` int32 undirected unique edges (the graph the
      regularizers act on);
    * ``K`` -- number of vertices that carry weight for one query point.
    """

    vertex_xy: np.ndarray
    edges: np.ndarray
    K: int

    @property
    def n_basis(self) -> int:
        return int(self.vertex_xy.shape[0])

    # -- the one abstract operator -------------------------------------------------
    def locate(self, bx, by) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """``(vids, w)``, each ``(K,) + bx.shape``: vertex ids and interpolation
        weights of every query point. Outside the domain ``w == 0`` (and ``vids == 0``,
        a harmless dummy). Weights sum to one inside."""
        raise NotImplementedError

    # -- derived views ---------------------------------------------------------------
    def basis(self, bx, by) -> jnp.ndarray:
        """One basis image per vertex: ``(I,) + bx.shape``."""
        bx = jnp.asarray(bx)
        by = jnp.asarray(by)
        shape = bx.shape
        vids, w = self.locate(bx.reshape(-1), by.reshape(-1))  # (K, N)
        n = vids.shape[1]
        col = jnp.broadcast_to(jnp.arange(n)[None, :], vids.shape)
        out = jnp.zeros((self.n_basis, n), dtype=w.dtype).at[vids, col].add(w)
        return out.reshape((self.n_basis,) + tuple(shape))

    def interpolate(self, values, bx, by) -> jnp.ndarray:
        """``sum_i values_i phi_i`` at the query points.

        ``values`` is ``(I,)`` (one field) or ``(I, B)`` with ``B`` equal to the
        trailing (batch) axis of ``bx`` -- the scene simulator's convention, where the
        sample batch rides on the last axis of every array.
        """
        values = jnp.asarray(values)
        vids, w = self.locate(jnp.asarray(bx), jnp.asarray(by))  # (K, *shape)
        if values.ndim == 1:
            vals = values[vids]
        elif values.ndim == 2:
            b = values.shape[-1]
            if b != 1 and vids.shape[-1] != b:
                raise ValueError(
                    f"interpolate: values batch {b} does not match the query points' "
                    f"trailing batch axis {vids.shape[-1]}")
            bidx = jnp.arange(b).reshape((1,) * (vids.ndim - 1) + (b,))
            vals = values[vids, jnp.broadcast_to(bidx, vids.shape) if b != 1 else 0]
        else:
            raise ValueError(f"interpolate: values must be (I,) or (I, B); got {values.shape}")
        return jnp.sum(w * vals, axis=0)

    def render(self, values, *, half_size: float, npix: int,
               center=(0.0, 0.0)) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Sample the interpolated field on a square grid (for plotting only)."""
        cx, cy = center
        xs = np.linspace(cx - half_size, cx + half_size, npix)
        ys = np.linspace(cy - half_size, cy + half_size, npix)
        X, Y = np.meshgrid(xs, ys, indexing="xy")
        img = np.asarray(self.interpolate(jnp.asarray(values), jnp.asarray(X), jnp.asarray(Y)))
        return img, (cx - half_size, cx + half_size, cy - half_size, cy + half_size)

    def __repr__(self):
        return f"{type(self).__name__}(I={self.n_basis}, E={self.edges.shape[0]})"


# ------------------------------------------------------------------------------------
# Regular grid, bilinear interpolation
# ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RegularGridDomain(SourceDomain):
    """Regular ``ny x nx`` source-plane grid with bilinear interpolation.

    Vertex ``v = iy * nx + ix`` sits at ``(x0 + ix * dx, y0 + iy * dy)``. Row-major
    ``(ny, nx)`` field arrays flatten to the vertex order, so
    ``field.reshape(-1)`` is a valid ``values`` vector for :meth:`interpolate`.
    """

    x0: float
    y0: float
    dx: float
    dy: float
    nx: int
    ny: int
    K = 4

    def __post_init__(self):
        if self.nx < 2 or self.ny < 2:
            raise ValueError("RegularGridDomain needs at least 2 x 2 vertices")
        if not (self.dx > 0 and self.dy > 0):
            raise ValueError("RegularGridDomain spacing must be positive")
        ix, iy = np.meshgrid(np.arange(self.nx), np.arange(self.ny), indexing="xy")
        vx = self.x0 + ix.ravel() * self.dx
        vy = self.y0 + iy.ravel() * self.dy
        object.__setattr__(self, "vertex_xy", np.stack([vx, vy], axis=-1).astype(np.float64))
        v = np.arange(self.nx * self.ny).reshape(self.ny, self.nx)
        horiz = np.stack([v[:, :-1].ravel(), v[:, 1:].ravel()], axis=-1)
        vert = np.stack([v[:-1, :].ravel(), v[1:, :].ravel()], axis=-1)
        object.__setattr__(self, "edges", np.concatenate([horiz, vert]).astype(np.int32))

    @classmethod
    def centered(cls, *, center, half_size: float, n: int) -> "RegularGridDomain":
        """``n x n`` grid covering ``[c - half_size, c + half_size]^2``."""
        d = 2.0 * float(half_size) / (int(n) - 1)
        cx, cy = center
        return cls(x0=float(cx) - float(half_size), y0=float(cy) - float(half_size),
                   dx=d, dy=d, nx=int(n), ny=int(n))

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.ny, self.nx)

    def locate(self, bx, by):
        bx = jnp.asarray(bx)
        by = jnp.asarray(by)
        fx = (bx - self.x0) / self.dx
        fy = (by - self.y0) / self.dy
        inside = (fx >= 0) & (fx <= self.nx - 1) & (fy >= 0) & (fy <= self.ny - 1)
        ix = jnp.clip(jnp.floor(fx), 0, self.nx - 2).astype(jnp.int32)
        iy = jnp.clip(jnp.floor(fy), 0, self.ny - 2).astype(jnp.int32)
        tx = fx - ix
        ty = fy - iy
        v00 = iy * self.nx + ix
        vids = jnp.stack([v00, v00 + 1, v00 + self.nx, v00 + self.nx + 1])
        w = jnp.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty])
        w = jnp.where(inside[None], w, 0.0)
        vids = jnp.where(inside[None], vids, 0)
        return vids, w


# ------------------------------------------------------------------------------------
# Triangle mesh, barycentric interpolation with bucketed point location
# ------------------------------------------------------------------------------------
class TriangleMeshDomain(SourceDomain):
    """Static triangulation of source-plane vertices (Delaunay or otherwise).

    Point location is a bounded search: a uniform bucket grid over the mesh's bounding
    box lists, per bucket, every triangle whose bounding box overlaps it (padded to a
    fixed width so the gather is jittable). A query evaluates the barycentric
    coordinates in every candidate and keeps the one containing it. Cost per point is
    ``O(max candidates per bucket)``, independent of the mesh size.

    Triangles are re-oriented counter-clockwise at construction and a zero-area
    triangle raises: with static vertices there is no such thing as a fold at
    evaluation time, so nothing downstream needs a degenerate-triangle guard.
    """

    K = 3

    def __init__(self, vertex_xy, simplices, *, n_buckets: Optional[int] = None,
                 tol: float = 1e-6):
        V = np.asarray(vertex_xy, dtype=np.float64)
        S = np.asarray(simplices, dtype=np.int32)
        if V.ndim != 2 or V.shape[1] != 2:
            raise ValueError(f"vertex_xy must be (I, 2); got {V.shape}")
        if S.ndim != 2 or S.shape[1] != 3:
            raise ValueError(f"simplices must be (T, 3); got {S.shape}")
        a, b, c = V[S[:, 0]], V[S[:, 1]], V[S[:, 2]]
        area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        flip = area2 < 0
        S = S.copy()
        S[flip, 1], S[flip, 2] = S[flip, 2], S[flip, 1]
        area2 = np.abs(area2)
        bbox_area = float(np.prod(V.max(0) - V.min(0)))
        if np.any(area2 <= 1e-12 * max(bbox_area, 1e-300)):
            n_bad = int(np.sum(area2 <= 1e-12 * max(bbox_area, 1e-300)))
            raise ValueError(
                f"TriangleMeshDomain: {n_bad} degenerate (zero-area) triangle(s); a static "
                "mesh must not contain them (drop duplicate vertices / re-triangulate).")
        self.vertex_xy = V
        self.simplices = S
        self.edges = _unique_edges(S)
        self.tol = float(tol)

        # Barycentric coordinates via the per-triangle affine inverse:
        #   (u, v) = M_t (p - a_t),   w = (1 - u - v, u, v)
        a, b, c = V[S[:, 0]], V[S[:, 1]], V[S[:, 2]]
        e1 = b - a
        e2 = c - a
        det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        self._a = jnp.asarray(a)
        self._cu = jnp.asarray(np.stack([e2[:, 1] / det, -e2[:, 0] / det], axis=-1))
        self._cv = jnp.asarray(np.stack([-e1[:, 1] / det, e1[:, 0] / det], axis=-1))
        self._simplices = jnp.asarray(S)

        # Bucket grid over the (slightly padded) bounding box.
        lo = V.min(0)
        hi = V.max(0)
        pad = 1e-9 * np.maximum(hi - lo, 1.0)
        lo = lo - pad
        hi = hi + pad
        nb = int(n_buckets) if n_buckets is not None else max(4, int(np.ceil(np.sqrt(S.shape[0]))))
        self._nb = nb
        self._lo = lo
        self._bw = (hi - lo) / nb
        tri_lo = np.minimum(np.minimum(a, b), c)
        tri_hi = np.maximum(np.maximum(a, b), c)
        blo = np.clip(np.floor((tri_lo - lo) / self._bw).astype(int), 0, nb - 1)
        bhi = np.clip(np.floor((tri_hi - lo) / self._bw).astype(int), 0, nb - 1)
        buckets = [[] for _ in range(nb * nb)]
        for t in range(S.shape[0]):
            for j in range(blo[t, 1], bhi[t, 1] + 1):
                for i in range(blo[t, 0], bhi[t, 0] + 1):
                    buckets[j * nb + i].append(t)
        kmax = max(1, max(len(bk) for bk in buckets))
        cand = -np.ones((nb * nb, kmax), dtype=np.int32)
        for q, bk in enumerate(buckets):
            cand[q, : len(bk)] = bk
        self._cand = jnp.asarray(cand)
        self.max_candidates = kmax

    # -- constructors ----------------------------------------------------------------
    @classmethod
    def delaunay(cls, points_xy, **kwargs) -> "TriangleMeshDomain":
        """Delaunay triangulation of source-plane points (scipy)."""
        from scipy.spatial import Delaunay

        pts = np.asarray(points_xy, dtype=np.float64)
        if pts.shape[0] < 3:
            raise ValueError("Delaunay needs at least 3 points")
        tri = Delaunay(pts)
        return cls(pts, tri.simplices, **kwargs)

    # -- the operator ----------------------------------------------------------------
    def locate(self, bx, by):
        bx = jnp.asarray(bx)
        by = jnp.asarray(by)
        shape = bx.shape
        px = bx.reshape(-1)
        py = by.reshape(-1)
        nb = self._nb
        fx = (px - self._lo[0]) / self._bw[0]
        fy = (py - self._lo[1]) / self._bw[1]
        in_box = (fx >= 0) & (fx < nb) & (fy >= 0) & (fy < nb)
        bi = jnp.clip(jnp.floor(fx), 0, nb - 1).astype(jnp.int32)
        bj = jnp.clip(jnp.floor(fy), 0, nb - 1).astype(jnp.int32)
        cand = self._cand[bj * nb + bi]  # (N, Kmax)
        valid = cand >= 0
        cs = jnp.where(valid, cand, 0)
        dx = px[:, None] - self._a[cs, 0]
        dy = py[:, None] - self._a[cs, 1]
        u = self._cu[cs, 0] * dx + self._cu[cs, 1] * dy
        v = self._cv[cs, 0] * dx + self._cv[cs, 1] * dy
        wa = 1.0 - u - v
        score = jnp.minimum(jnp.minimum(wa, u), v)
        score = jnp.where(valid, score, -jnp.inf)
        best = jnp.argmax(score, axis=1)  # integer choice: no gradient, by design
        rows = jnp.arange(px.shape[0])
        tri = cs[rows, best]
        w = jnp.stack([wa[rows, best], u[rows, best], v[rows, best]])  # (3, N)
        inside = in_box & (score[rows, best] >= -self.tol)
        w = jnp.where(inside[None], w, 0.0)
        vids = jnp.where(inside[None], self._simplices[tri].T, 0)
        return vids.reshape((3,) + tuple(shape)), w.reshape((3,) + tuple(shape))

    def triangle_of(self, bx, by) -> np.ndarray:
        """Containing-triangle index per point, ``-1`` outside (diagnostics)."""
        vids, w = self.locate(jnp.asarray(bx).reshape(-1), jnp.asarray(by).reshape(-1))
        inside = np.asarray(jnp.sum(w, axis=0) > 0)
        # Recover the triangle from its vertex triple (unique in a valid mesh).
        key = {tuple(sorted(s)): t for t, s in enumerate(np.asarray(self.simplices))}
        vids = np.asarray(vids).T
        out = np.array([key.get(tuple(sorted(v)), -1) if ok else -1 for v, ok in zip(vids, inside)])
        return out


def _unique_edges(simplices: np.ndarray) -> np.ndarray:
    e = np.concatenate([simplices[:, [0, 1]], simplices[:, [1, 2]], simplices[:, [2, 0]]])
    e = np.sort(e, axis=1)
    return np.unique(e, axis=0).astype(np.int32)


# ------------------------------------------------------------------------------------
# Builders from a pilot model
# ------------------------------------------------------------------------------------
KMEANS_WEIGHT_SCHEMES = ("paper_eq12", "normalized_floor", "pyautoarray_current")


def kmeans_weights(image, *, scheme: str, floor: float) -> np.ndarray:
    """Per-pixel KMeans weights from a pilot lensed-source image.

    ``paper_eq12``: ``normalized + floor + max(signal)`` (Nightingale & Dye 2015 eq. 12
    as written); ``normalized_floor``: ``normalized + floor``;
    ``pyautoarray_current``: ``|img| / max(img)`` floored at ``floor``. All strictly
    positive and finite.
    """
    img = np.asarray(image, dtype=np.float64).reshape(-1)
    signal = np.clip(img, 0.0, None)
    rng = float(signal.max() - signal.min())
    normalized = (signal - signal.min()) / rng if rng > 0 else np.zeros_like(signal)
    if scheme == "paper_eq12":
        w = normalized + float(floor) + float(signal.max())
    elif scheme == "normalized_floor":
        w = normalized + float(floor)
    elif scheme == "pyautoarray_current":
        w = np.abs(img) / max(float(img.max()), np.finfo(float).tiny)
        w = np.where(w < float(floor), float(floor), w)
    else:
        raise ValueError(f"scheme must be one of {KMEANS_WEIGHT_SCHEMES}; got {scheme!r}")
    return np.where(np.isfinite(w) & (w > 0), w, float(floor))


def adaptive_delaunay_domain(
    *,
    model,
    params,
    plane: int,
    sim_config,
    pilot_image,
    n_vertices: int,
    region_half_size: float,
    weight_scheme: str,
    weight_floor: float,
    seed: int,
    pad_fraction: float = 0.0,
    kmeans_max_iter: int = 200,
) -> TriangleMeshDomain:
    """Brightness-adaptive Delaunay domain from a pilot model (the herculens /
    PyAutoLens staged idea, frozen in the SOURCE plane).

    Image-plane centres are placed by weighted KMeans on ``pilot_image`` (the pilot
    model's lensed-source image on the detector grid, ``(H, W)``) inside the box
    ``|x|, |y| <= region_half_size``; the centres are ray-traced to ``plane`` with the
    pilot ``params`` through ``model.trace_to_plane`` and Delaunay-triangulated there.
    Vertex density therefore follows both surface brightness and magnification.

    ``pad_fraction > 0`` appends a ring of vertices at ``(1 + pad_fraction)`` times the
    traced points' radius about their centroid, so ray-traced sub-pixels that leave the
    data-constrained region under a mass perturbation land on padding vertices (which the
    regularizer pulls towards their neighbours) instead of a hard zero cutoff.

    Nothing here is a silent default: the region, vertex count, weight scheme, floor and
    seed are all required so the construction is reproducible from the call alone.
    """
    from gigalens.simulator import LensWCS

    if weight_scheme not in KMEANS_WEIGHT_SCHEMES:
        raise ValueError(f"weight_scheme must be one of {KMEANS_WEIGHT_SCHEMES}; got {weight_scheme!r}")
    img = np.asarray(pilot_image, dtype=np.float64)
    wcs = LensWCS(n=sim_config.num_pix, supersample=1,
                  transform_pix2angle=sim_config.transform_pix2angle,
                  pix_scale=sim_config.delta_pix, shift=sim_config.angular_shift)
    X, Y = wcs.pixel_grid()
    if img.shape != X.shape:
        raise ValueError(f"pilot_image shape {img.shape} does not match the detector grid {X.shape}")
    x = X.ravel()
    y = Y.ravel()
    inside = (np.abs(x) <= region_half_size) & (np.abs(y) <= region_half_size)
    if int(inside.sum()) < n_vertices:
        raise ValueError(
            f"n_vertices={n_vertices} exceeds the {int(inside.sum())} pixels inside "
            f"region_half_size={region_half_size}")
    pts = np.stack([x[inside], y[inside]], axis=-1)
    w = kmeans_weights(img.ravel()[inside], scheme=weight_scheme, floor=weight_floor)

    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=int(n_vertices), init="k-means++", n_init=4,
                max_iter=int(kmeans_max_iter), random_state=int(seed), algorithm="lloyd")
    km.fit(pts, sample_weight=w)
    seeds_img = km.cluster_centers_.astype(np.float64)

    bx, by = model.trace_to_plane(params, jnp.asarray(seeds_img[:, 0])[:, None],
                                  jnp.asarray(seeds_img[:, 1])[:, None], plane)
    src = np.stack([np.asarray(bx).reshape(-1), np.asarray(by).reshape(-1)], axis=-1)
    if src.shape[0] != seeds_img.shape[0]:
        raise ValueError("trace_to_plane returned a batched result; pass unbatched pilot params")

    if pad_fraction > 0:
        c = src.mean(0)
        r = float(np.max(np.linalg.norm(src - c, axis=1))) * (1.0 + float(pad_fraction))
        m = max(8, int(np.ceil(np.sqrt(n_vertices))))
        ang = np.linspace(0, 2 * np.pi, m, endpoint=False)
        ring = c + r * np.stack([np.cos(ang), np.sin(ang)], axis=-1)
        src = np.concatenate([src, ring])

    domain = TriangleMeshDomain.delaunay(src)
    domain.image_seed_xy = seeds_img  # for diagnostics / plots
    domain.n_padding = int(src.shape[0] - seeds_img.shape[0])
    return domain
