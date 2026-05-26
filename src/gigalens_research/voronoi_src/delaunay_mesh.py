from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DelaunayMesh:
    """
    Frozen image-plane triangulation + lookup tables.

    Notes
    -----
    - Connectivity is defined in the image plane and is *static*.
    - Per evaluation, the lens model maps image-plane vertices to source-plane
      vertices; barycentric weights are then computed in the source plane using
      the static connectivity.
    """

    # Image-plane seed vertices (static)
    seed_xy: np.ndarray  # (I, 2) float32, arcsec

    # Triangulation (static)
    simplices: np.ndarray  # (T, 3) int32, vertex indices
    edges: np.ndarray  # (E, 2) int32, undirected unique edges

    # Sub-pixel lookup (static)
    subpix_xy: np.ndarray  # (J_sub, 2) float32, arcsec
    subpix_tri: np.ndarray  # (J_sub,) int32, triangle id or -1 if outside

    # Grid metadata
    n_seed_y: int
    n_seed_x: int
    extent: float  # half-size in arcsec (seeds cover [-extent, extent]^2)

    num_pix: int
    supersample: int


def _unique_edges_from_simplices(simplices: np.ndarray) -> np.ndarray:
    """Return unique undirected edges as sorted int32 pairs."""
    a = simplices[:, [0, 1]]
    b = simplices[:, [1, 2]]
    c = simplices[:, [2, 0]]
    edges = np.concatenate([a, b, c], axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges.astype(np.int32)


def _squeeze_lens_params(lens_params):
    """Return lens_params with each leaf cast to float and squeezed to scalar."""
    import jax.numpy as jnp

    out = []
    for p in lens_params:
        out.append({k: jnp.asarray(jnp.squeeze(jnp.asarray(v)), dtype=jnp.float32) for k, v in p.items()})
    return out


def _magnification_numpy(
    lenses,
    lens_params,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """Signed magnification mu(theta) = 1 / det(d beta / d theta) at image-plane points."""
    import jax
    import jax.numpy as jnp

    lp = _squeeze_lens_params(lens_params)
    x_j = jnp.asarray(x, dtype=jnp.float32)
    y_j = jnp.asarray(y, dtype=jnp.float32)
    pts = jnp.stack([x_j, y_j], axis=-1)

    def beta_vec(th):
        bx, by = th[0], th[1]
        for lens, p in zip(lenses, lp):
            ax, ay = lens.deriv(bx, by, **p)
            bx = bx - ax
            by = by - ay
        return jnp.stack([jnp.squeeze(bx), jnp.squeeze(by)])

    def mag_one(pt):
        jac = jax.jacfwd(beta_vec)(pt)
        det = jnp.linalg.det(jac)
        return 1.0 / det

    mu = jax.vmap(mag_one)(pts)
    return np.asarray(jax.device_get(mu), dtype=np.float32)


def _beta_numpy(lenses, lens_params, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Ray-trace (x,y) to the source plane using JAX lens models, returning numpy arrays.
    Intended for one-off mesh setup on CPU.
    """
    import jax
    import jax.numpy as jnp

    bx = jnp.array(x, dtype=jnp.float32)
    by = jnp.array(y, dtype=jnp.float32)
    for lens, p in zip(lenses, lens_params):
        ax, ay = lens.deriv(bx, by, **p)
        bx = bx - ax
        by = by - ay
    bx, by = jax.device_get(bx), jax.device_get(by)
    return np.asarray(bx, dtype=np.float32), np.asarray(by, dtype=np.float32)


def build_frozen_sourceplane_delaunay_from_truth(
    *,
    lenses,
    lens_params_truth,
    num_pix: int,
    delta_pix: float,
    supersample: int,
    n_seed_y: int,
    n_seed_x: int,
    extent: float,
) -> DelaunayMesh:
    """
    Build a Delaunay triangulation in the *source plane* using truth ray-traced
    seed points, then freeze:
      - simplices (connectivity)
      - per-subpixel triangle assignment computed at truth

    This is a stability-oriented Stage 0 variant: within an optimisation run
    we keep the combinatorial structure fixed.
    """
    # Seed vertices in image plane (regular grid)
    ys = np.linspace(-extent, extent, n_seed_y, dtype=np.float32)
    xs = np.linspace(-extent, extent, n_seed_x, dtype=np.float32)
    seed_X, seed_Y = np.meshgrid(xs, ys, indexing="xy")
    seed_xy = np.stack([seed_X.ravel(), seed_Y.ravel()], axis=-1).astype(np.float32)

    # Subpixel coordinates consistent with GIGALens
    import gigalens.simulator as _sim

    transform = (np.eye(2, dtype=np.float32) * np.float32(delta_pix)) / float(supersample)
    _, _, img_x, img_y = _sim.LensSimulatorInterface.get_coords(
        int(supersample), int(num_pix), transform
    )
    subpix_xy = np.stack([img_x.ravel(), img_y.ravel()], axis=-1).astype(np.float32)

    # Ray-trace seeds and subpixels with truth mass model
    seed_bx, seed_by = _beta_numpy(lenses, lens_params_truth, seed_xy[:, 0], seed_xy[:, 1])
    sub_bx, sub_by = _beta_numpy(lenses, lens_params_truth, subpix_xy[:, 0], subpix_xy[:, 1])
    seed_src = np.stack([seed_bx, seed_by], axis=-1)
    sub_src = np.stack([sub_bx, sub_by], axis=-1)

    # Source-plane Delaunay connectivity
    from scipy.spatial import Delaunay

    tri = Delaunay(seed_src.astype(np.float64))
    simplices = np.asarray(tri.simplices, dtype=np.int32)
    edges = _unique_edges_from_simplices(simplices)

    # Truth-time triangle assignment for each subpixel
    subpix_tri = tri.find_simplex(sub_src.astype(np.float64)).astype(np.int32)
    # Outside convex hull => -1
    subpix_tri = np.where(subpix_tri >= 0, subpix_tri, -np.ones_like(subpix_tri, dtype=np.int32))

    return DelaunayMesh(
        seed_xy=seed_xy,
        simplices=simplices,
        edges=edges,
        subpix_xy=subpix_xy,
        subpix_tri=subpix_tri,
        n_seed_y=int(n_seed_y),
        n_seed_x=int(n_seed_x),
        extent=float(extent),
        num_pix=int(num_pix),
        supersample=int(supersample),
    )


def _weighted_kmeans_sklearn(
    points: np.ndarray,
    weights: np.ndarray,
    n_clusters: int,
    seed: int,
    max_iter: int,
) -> np.ndarray:
    """Run weighted KMeans, using scikit-learn when available."""
    from sklearn.cluster import KMeans

    km = KMeans(
        n_clusters=int(n_clusters),
        init="k-means++",
        n_init=4,
        max_iter=int(max_iter),
        random_state=int(seed),
        algorithm="lloyd",
    )
    km.fit(points.astype(np.float64), sample_weight=weights.astype(np.float64))
    return km.cluster_centers_.astype(np.float32)


def _kmeans_weights_from_image(
    *,
    pix_xy_masked: np.ndarray,
    img_masked: np.ndarray,
    weight_floor: float,
    weight_scheme: str,
    mag_masked: np.ndarray | None = None,
) -> np.ndarray:
    """Compute per-pixel KMeans weights for adaptive mesh seeding."""
    signal = np.clip(img_masked, 0.0, None)
    signal_range = float(np.max(signal) - np.min(signal))
    if signal_range > 0:
        normalized = (signal - np.min(signal)) / signal_range
    else:
        normalized = np.zeros_like(signal)

    if weight_scheme == "paper_eq12":
        weights = normalized + float(weight_floor) + float(np.max(signal))
    elif weight_scheme == "pyautoarray_current":
        weights = np.abs(img_masked) / max(float(np.max(img_masked)), np.finfo(float).tiny)
        weights = np.where(weights < float(weight_floor), float(weight_floor), weights)
    elif weight_scheme == "normalized_floor":
        weights = normalized + float(weight_floor)
    elif weight_scheme == "brightness_times_invmag":
        if mag_masked is None:
            raise ValueError("mag_masked required for brightness_times_invmag weight scheme")
        invmag = np.sqrt(np.clip(np.abs(mag_masked), 1e-3, None))
        weights = (normalized + float(weight_floor)) * invmag
    else:
        raise ValueError(
            "weight_scheme must be one of 'paper_eq12', 'pyautoarray_current', "
            "'normalized_floor', or 'brightness_times_invmag'."
        )

    weights = np.asarray(weights, dtype=np.float64)
    return np.where(np.isfinite(weights) & (weights > 0), weights, float(weight_floor))


def build_brightness_adaptive_sourceplane_delaunay_from_truth(
    *,
    lenses,
    lens_params_truth,
    lensed_source_image: np.ndarray,
    num_pix: int,
    delta_pix: float,
    supersample: int,
    n_source_pixels: int,
    extent: float,
    weight_floor: float = 0.05,
    weight_scheme: str = "paper_eq12",
    seed: int = 0,
    max_iter: int = 200,
) -> DelaunayMesh:
    """
    Build a source-plane Delaunay mesh from brightness-adaptive image-plane
    centres, following the PyAutoLens / Nightingale et al. staged idea.

    We compute image-plane KMeans centres using weights from a previous lensed
    source model (here supplied by the truth-pinned lens-subtracted image), then
    ray-trace those centres to the source plane and freeze the Delaunay
    connectivity / subpixel lookup at truth.

    Stage-0 limitation: PyAutoLens uses Voronoi natural-neighbour
    interpolation, whereas the current experimental simulator uses Delaunay
    barycentric interpolation on the resulting centres.
    """
    import gigalens.simulator as _sim
    from scipy.spatial import Delaunay

    # Pixel-centre coordinates for placing adaptive image-plane centres.
    pix_transform = np.eye(2, dtype=np.float32) * np.float32(delta_pix)
    _, _, pix_x, pix_y = _sim.LensSimulatorInterface.get_coords(
        1, int(num_pix), pix_transform
    )
    pix_xy = np.stack([pix_x.ravel(), pix_y.ravel()], axis=-1).astype(np.float32)

    img = np.asarray(lensed_source_image, dtype=np.float32).reshape(-1)
    mask = (
        (pix_xy[:, 0] >= -extent)
        & (pix_xy[:, 0] <= extent)
        & (pix_xy[:, 1] >= -extent)
        & (pix_xy[:, 1] <= extent)
    )
    pix_xy_masked = pix_xy[mask]
    img_masked = img[mask]

    mag_masked = None
    if weight_scheme == "brightness_times_invmag":
        mag_masked = _magnification_numpy(
            lenses,
            lens_params_truth,
            pix_xy_masked[:, 0],
            pix_xy_masked[:, 1],
        )
    weights = _kmeans_weights_from_image(
        pix_xy_masked=pix_xy_masked,
        img_masked=img_masked,
        weight_floor=weight_floor,
        weight_scheme=weight_scheme,
        mag_masked=mag_masked,
    )

    if n_source_pixels > pix_xy_masked.shape[0]:
        raise ValueError(
            f"n_source_pixels={n_source_pixels} exceeds masked image pixels "
            f"({pix_xy_masked.shape[0]})."
        )

    seed_xy = _weighted_kmeans_sklearn(
        pix_xy_masked,
        weights,
        n_clusters=int(n_source_pixels),
        seed=int(seed),
        max_iter=int(max_iter),
    )

    # Subpixel coordinates used by the mapping matrix.
    transform = (np.eye(2, dtype=np.float32) * np.float32(delta_pix)) / float(supersample)
    _, _, img_x, img_y = _sim.LensSimulatorInterface.get_coords(
        int(supersample), int(num_pix), transform
    )
    subpix_xy = np.stack([img_x.ravel(), img_y.ravel()], axis=-1).astype(np.float32)

    seed_bx, seed_by = _beta_numpy(lenses, lens_params_truth, seed_xy[:, 0], seed_xy[:, 1])
    sub_bx, sub_by = _beta_numpy(lenses, lens_params_truth, subpix_xy[:, 0], subpix_xy[:, 1])
    seed_src = np.stack([seed_bx, seed_by], axis=-1)
    sub_src = np.stack([sub_bx, sub_by], axis=-1)

    tri = Delaunay(seed_src.astype(np.float64))
    simplices = np.asarray(tri.simplices, dtype=np.int32)
    edges = _unique_edges_from_simplices(simplices)
    subpix_tri = tri.find_simplex(sub_src.astype(np.float64)).astype(np.int32)
    subpix_tri = np.where(subpix_tri >= 0, subpix_tri, -np.ones_like(subpix_tri, dtype=np.int32))

    n_side = int(round(np.sqrt(n_source_pixels)))
    return DelaunayMesh(
        seed_xy=seed_xy.astype(np.float32),
        simplices=simplices,
        edges=edges,
        subpix_xy=subpix_xy,
        subpix_tri=subpix_tri,
        n_seed_y=n_side,
        n_seed_x=n_side,
        extent=float(extent),
        num_pix=int(num_pix),
        supersample=int(supersample),
    )


def build_brightness_adaptive_imageplane_delaunay_from_truth(
    *,
    lenses,
    lens_params_truth,
    lensed_source_image: np.ndarray,
    num_pix: int,
    delta_pix: float,
    supersample: int,
    n_source_pixels: int,
    extent: float,
    weight_floor: float = 0.05,
    weight_scheme: str = "paper_eq12",
    seed: int = 0,
    max_iter: int = 200,
) -> DelaunayMesh:
    """
    Brightness-adaptive KMeans seeds with Delaunay connectivity in the image plane.

    Sub-pixel triangle lookup is frozen at truth in image-plane coordinates; per
    evaluation the lens maps vertices and sub-pixels to the source plane for
    barycentric weights while keeping combinatorics static (mass-differentiable path).
    """
    import gigalens.simulator as _sim
    from scipy.spatial import Delaunay

    pix_transform = np.eye(2, dtype=np.float32) * np.float32(delta_pix)
    _, _, pix_x, pix_y = _sim.LensSimulatorInterface.get_coords(
        1, int(num_pix), pix_transform
    )
    pix_xy = np.stack([pix_x.ravel(), pix_y.ravel()], axis=-1).astype(np.float32)

    img = np.asarray(lensed_source_image, dtype=np.float32).reshape(-1)
    mask = (
        (pix_xy[:, 0] >= -extent)
        & (pix_xy[:, 0] <= extent)
        & (pix_xy[:, 1] >= -extent)
        & (pix_xy[:, 1] <= extent)
    )
    pix_xy_masked = pix_xy[mask]
    img_masked = img[mask]

    mag_masked = None
    if weight_scheme == "brightness_times_invmag":
        mag_masked = _magnification_numpy(
            lenses,
            lens_params_truth,
            pix_xy_masked[:, 0],
            pix_xy_masked[:, 1],
        )
    weights = _kmeans_weights_from_image(
        pix_xy_masked=pix_xy_masked,
        img_masked=img_masked,
        weight_floor=weight_floor,
        weight_scheme=weight_scheme,
        mag_masked=mag_masked,
    )

    if n_source_pixels > pix_xy_masked.shape[0]:
        raise ValueError(
            f"n_source_pixels={n_source_pixels} exceeds masked image pixels "
            f"({pix_xy_masked.shape[0]})."
        )

    seed_xy = _weighted_kmeans_sklearn(
        pix_xy_masked,
        weights,
        n_clusters=int(n_source_pixels),
        seed=int(seed),
        max_iter=int(max_iter),
    )

    transform = (np.eye(2, dtype=np.float32) * np.float32(delta_pix)) / float(supersample)
    _, _, img_x, img_y = _sim.LensSimulatorInterface.get_coords(
        int(supersample), int(num_pix), transform
    )
    subpix_xy = np.stack([img_x.ravel(), img_y.ravel()], axis=-1).astype(np.float32)

    tri = Delaunay(seed_xy.astype(np.float64))
    simplices = np.asarray(tri.simplices, dtype=np.int32)
    edges = _unique_edges_from_simplices(simplices)
    subpix_tri = tri.find_simplex(subpix_xy.astype(np.float64)).astype(np.int32)
    subpix_tri = np.where(subpix_tri >= 0, subpix_tri, -np.ones_like(subpix_tri, dtype=np.int32))

    n_side = int(round(np.sqrt(n_source_pixels)))
    return DelaunayMesh(
        seed_xy=seed_xy.astype(np.float32),
        simplices=simplices,
        edges=edges,
        subpix_xy=subpix_xy,
        subpix_tri=subpix_tri,
        n_seed_y=n_side,
        n_seed_x=n_side,
        extent=float(extent),
        num_pix=int(num_pix),
        supersample=int(supersample),
    )


def build_regular_imageplane_mesh(
    *,
    num_pix: int,
    delta_pix: float,
    supersample: int,
    n_seed_y: int,
    n_seed_x: int,
    extent: float,
) -> DelaunayMesh:
    """
    Build a deterministic 2-triangles-per-cell triangulation over a regular
    image-plane seed grid, and a static mapping from supersampled image pixels
    to triangle ids.

    Parameters
    ----------
    num_pix, delta_pix, supersample:
        Image-plane grid properties matching SimulatorConfig.
    n_seed_y, n_seed_x:
        Seed grid resolution.
    extent:
        Seeds cover [-extent, extent]^2 (arcsec) in the image plane.
    """

    # Seed vertices in image plane.
    ys = np.linspace(-extent, extent, n_seed_y, dtype=np.float32)
    xs = np.linspace(-extent, extent, n_seed_x, dtype=np.float32)
    seed_X, seed_Y = np.meshgrid(xs, ys, indexing="xy")
    seed_xy = np.stack([seed_X.ravel(), seed_Y.ravel()], axis=-1).astype(np.float32)

    # Triangulation: each grid cell -> two triangles with a fixed diagonal.
    # Vertex indexing: vid = iy * n_seed_x + ix.
    simplices = []
    for iy in range(n_seed_y - 1):
        for ix in range(n_seed_x - 1):
            v00 = iy * n_seed_x + ix
            v10 = iy * n_seed_x + (ix + 1)
            v01 = (iy + 1) * n_seed_x + ix
            v11 = (iy + 1) * n_seed_x + (ix + 1)
            # Diagonal v00 -> v11
            simplices.append((v00, v10, v11))
            simplices.append((v00, v11, v01))
    simplices = np.asarray(simplices, dtype=np.int32)
    edges = _unique_edges_from_simplices(simplices)

    # Supersampled image-plane coordinates (pixel centers), using the same
    # coordinate convention as gigalens' LensSimulatorInterface.get_coords.
    import gigalens.simulator as _sim

    transform = (np.eye(2, dtype=np.float32) * np.float32(delta_pix)) / float(supersample)
    _, _, img_x, img_y = _sim.LensSimulatorInterface.get_coords(
        int(supersample), int(num_pix), transform
    )
    subpix_xy = np.stack([img_x.ravel(), img_y.ravel()], axis=-1).astype(np.float32)

    # Static mapping from (x,y) to triangle id.
    # Determine seed cell index (ix_cell, iy_cell). Outside extent -> -1.
    dx_seed = (2 * extent) / (n_seed_x - 1)
    dy_seed = (2 * extent) / (n_seed_y - 1)
    x = subpix_xy[:, 0]
    y = subpix_xy[:, 1]

    inside = (x >= -extent) & (x <= extent) & (y >= -extent) & (y <= extent)
    ix = np.floor((x + extent) / dx_seed).astype(np.int32)
    iy = np.floor((y + extent) / dy_seed).astype(np.int32)
    # Clamp boundary points that land exactly on +extent to the last cell.
    ix = np.clip(ix, 0, n_seed_x - 2)
    iy = np.clip(iy, 0, n_seed_y - 2)

    # Choose triangle within cell via diagonal test.
    # Convert to local coords u,v in [0,1] within the cell.
    x0 = -extent + ix.astype(np.float32) * dx_seed
    y0 = -extent + iy.astype(np.float32) * dy_seed
    u = (x - x0) / dx_seed
    v = (y - y0) / dy_seed
    # Above diagonal (u+v >= 1) -> triangle (v00,v10,v11), else (v00,v11,v01)
    upper = (u + v) >= 1.0

    cell_id = iy * (n_seed_x - 1) + ix  # (num_cells,) flattened
    tri_base = 2 * cell_id
    tri_id = np.where(upper, tri_base + 0, tri_base + 1).astype(np.int32)
    tri_id = np.where(inside, tri_id, -np.ones_like(tri_id, dtype=np.int32))

    return DelaunayMesh(
        seed_xy=seed_xy,
        simplices=simplices,
        edges=edges,
        subpix_xy=subpix_xy,
        subpix_tri=tri_id,
        n_seed_y=int(n_seed_y),
        n_seed_x=int(n_seed_x),
        extent=float(extent),
        num_pix=int(num_pix),
        supersample=int(supersample),
    )

