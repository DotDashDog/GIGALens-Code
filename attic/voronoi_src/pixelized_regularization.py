from __future__ import annotations

from typing import Callable

import jax.numpy as jnp


def _add_ridge(H: jnp.ndarray, *, num_vertices: int, ridge_scale: float) -> jnp.ndarray:
    tr = jnp.trace(H)
    ridge = ridge_scale * jnp.maximum(tr / num_vertices, 1.0)
    return H + ridge * jnp.eye(num_vertices, dtype=H.dtype)


def regularization_matrix_constant_gradient(
    *,
    num_vertices: int,
    edges: jnp.ndarray,
    lam: jnp.ndarray,
    ridge_scale: float = 1e-8,
    vertex_positions: jnp.ndarray | None = None,
    vertex_lam: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Graph Laplacian penalty: lam * sum_edges (s_i - s_j)^2."""
    del vertex_positions, vertex_lam
    i = edges[:, 0]
    k = edges[:, 1]

    L = jnp.zeros((num_vertices, num_vertices), dtype=jnp.float32)
    L = L.at[i, i].add(1.0)
    L = L.at[k, k].add(1.0)
    L = L.at[i, k].add(-1.0)
    L = L.at[k, i].add(-1.0)

    H = lam.astype(jnp.float32) * L
    return _add_ridge(H, num_vertices=num_vertices, ridge_scale=ridge_scale)


def regularization_matrix_distance_weighted_gradient(
    *,
    num_vertices: int,
    edges: jnp.ndarray,
    lam: jnp.ndarray,
    ridge_scale: float = 1e-8,
    vertex_positions: jnp.ndarray,
    vertex_lam: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """lam * sum_edges w_ij (s_i - s_j)^2 with w_ij = 1 / |x_i - x_j|^2."""
    del vertex_lam
    if vertex_positions is None:
        raise ValueError("vertex_positions required for distance-weighted gradient regularization")

    i = edges[:, 0]
    k = edges[:, 1]
    xi = vertex_positions[i]
    xk = vertex_positions[k]
    dist2 = jnp.sum((xi - xk) ** 2, axis=-1)
    dist2 = jnp.maximum(dist2, jnp.float32(1e-12))
    w = 1.0 / dist2

    H = jnp.zeros((num_vertices, num_vertices), dtype=jnp.float32)
    lam_f = lam.astype(jnp.float32)
    H = H.at[i, i].add(lam_f * w)
    H = H.at[k, k].add(lam_f * w)
    H = H.at[i, k].add(-lam_f * w)
    H = H.at[k, i].add(-lam_f * w)
    return _add_ridge(H, num_vertices=num_vertices, ridge_scale=ridge_scale)


def regularization_matrix_curvature(
    *,
    num_vertices: int,
    edges: jnp.ndarray,
    lam: jnp.ndarray,
    ridge_scale: float = 1e-8,
    vertex_positions: jnp.ndarray | None = None,
    vertex_lam: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """lam * sum_i (s_i - mean(s_{N(i)}))^2 (constant-curvature / second-difference)."""
    del vertex_positions, vertex_lam
    i = edges[:, 0]
    k = edges[:, 1]

    degree = jnp.zeros((num_vertices,), dtype=jnp.float32)
    degree = degree.at[i].add(1.0)
    degree = degree.at[k].add(1.0)

    # Build neighbor-mean operator: (B s)_v = mean_{u in N(v)} s_u
    neighbor_sum = jnp.zeros((num_vertices, num_vertices), dtype=jnp.float32)
    neighbor_sum = neighbor_sum.at[i, k].add(1.0)
    neighbor_sum = neighbor_sum.at[k, i].add(1.0)
    inv_degree = jnp.where(degree > 0, 1.0 / degree, 0.0)
    B = neighbor_sum * inv_degree[:, None]

    I = jnp.eye(num_vertices, dtype=jnp.float32)
    M = I - B
    H = lam.astype(jnp.float32) * (M.T @ M)
    return _add_ridge(H, num_vertices=num_vertices, ridge_scale=ridge_scale)


def regularization_matrix_adaptive_split(
    *,
    num_vertices: int,
    edges: jnp.ndarray,
    lam: jnp.ndarray,
    ridge_scale: float = 1e-8,
    vertex_positions: jnp.ndarray | None = None,
    vertex_lam: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Per-vertex lambda field (Nightingale AdaptSplit-style).

    vertex_lam must be supplied with shape (num_vertices,); typically
    lambda_max on faint vertices and lambda_min on bright vertices.
    Edge penalty uses lam_ij = 0.5 * (lam_i + lam_j).
    """
    del vertex_positions
    if vertex_lam is None:
        raise ValueError("vertex_lam required for adaptive_split regularization")

    i = edges[:, 0]
    k = edges[:, 1]
    lam_i = vertex_lam[i].astype(jnp.float32)
    lam_k = vertex_lam[k].astype(jnp.float32)
    scale = lam.astype(jnp.float32)
    lam_edge = scale * 0.5 * (lam_i + lam_k)

    H = jnp.zeros((num_vertices, num_vertices), dtype=jnp.float32)
    H = H.at[i, i].add(lam_edge)
    H = H.at[k, k].add(lam_edge)
    H = H.at[i, k].add(-lam_edge)
    H = H.at[k, i].add(-lam_edge)
    return _add_ridge(H, num_vertices=num_vertices, ridge_scale=ridge_scale)


RegMatrixBuilder = Callable[..., jnp.ndarray]

REGULARIZATION_BUILDERS: dict[str, RegMatrixBuilder] = {
    "constant_gradient": regularization_matrix_constant_gradient,
    "distance_weighted_gradient": regularization_matrix_distance_weighted_gradient,
    "curvature": regularization_matrix_curvature,
    "adaptive_split": regularization_matrix_adaptive_split,
}


def build_regularization_matrix(
    kind: str,
    *,
    num_vertices: int,
    edges: jnp.ndarray,
    lam: jnp.ndarray,
    vertex_positions: jnp.ndarray | None = None,
    vertex_lam: jnp.ndarray | None = None,
    ridge_scale: float = 1e-8,
) -> jnp.ndarray:
    """Dispatch to a named regularization matrix builder."""
    if kind not in REGULARIZATION_BUILDERS:
        raise KeyError(f"Unknown regularization kind {kind!r}; choose from {sorted(REGULARIZATION_BUILDERS)}")
    return REGULARIZATION_BUILDERS[kind](
        num_vertices=num_vertices,
        edges=edges,
        lam=lam,
        ridge_scale=ridge_scale,
        vertex_positions=vertex_positions,
        vertex_lam=vertex_lam,
    )
