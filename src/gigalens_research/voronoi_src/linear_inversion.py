from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .pixelized_regularization import build_regularization_matrix


def solve_source_unconstrained(
    *,
    basis: jnp.ndarray,
    resid_no_source: jnp.ndarray,
    inv_var: jnp.ndarray,
    edges: jnp.ndarray,
    lam: float | jnp.ndarray,
    reg_kind: str = "constant_gradient",
    vertex_positions: jnp.ndarray | None = None,
    vertex_lam: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Analytic ridge-regularized linear inversion (same as prob model at fixed mass)."""
    D = jnp.einsum("ihw,hw->i", basis, resid_no_source * inv_var, precision="highest")
    F = jnp.einsum("ihw,khw,hw->ik", basis, basis, inv_var, precision="highest")
    lam_j = jnp.asarray(lam, dtype=jnp.float32)
    H = build_regularization_matrix(
        reg_kind,
        num_vertices=basis.shape[0],
        edges=jnp.asarray(edges, dtype=jnp.int32),
        lam=lam_j,
        vertex_positions=vertex_positions,
        vertex_lam=vertex_lam,
    )
    A = F + H
    I = basis.shape[0]
    trA = jnp.trace(A)
    jitter = jnp.float32(1e-6) * jnp.maximum(trA / I, 1.0)
    A = A + jitter * jnp.eye(I, dtype=A.dtype)
    chol = jnp.linalg.cholesky(A)
    y = jax.lax.linalg.triangular_solve(chol, D, left_side=True, lower=True)
    return jax.lax.linalg.triangular_solve(chol.T, y, left_side=True, lower=False)


def _build_reg_matrix_numpy(
    *,
    reg_kind: str,
    num_vertices: int,
    edges: np.ndarray,
    lam: float,
    vertex_positions: np.ndarray | None,
    vertex_lam: np.ndarray | None,
    ridge_scale: float,
) -> np.ndarray:
    """Build the regularization matrix in float64 numpy for the positive solver.

    This mirrors the JAX builders bit-for-bit but in higher precision so that
    a downstream Cholesky stays well-defined for large-lambda / wide-dynamic-
    range edge weights.
    """
    i = edges[:, 0].astype(np.int64)
    k = edges[:, 1].astype(np.int64)
    n = int(num_vertices)
    lam_f = float(lam)

    if reg_kind == "constant_gradient":
        H = np.zeros((n, n), dtype=np.float64)
        np.add.at(H, (i, i), lam_f)
        np.add.at(H, (k, k), lam_f)
        np.add.at(H, (i, k), -lam_f)
        np.add.at(H, (k, i), -lam_f)
    elif reg_kind == "distance_weighted_gradient":
        if vertex_positions is None:
            raise ValueError("vertex_positions required for distance_weighted_gradient")
        pos = np.asarray(vertex_positions, dtype=np.float64)
        dist2 = np.maximum(np.sum((pos[i] - pos[k]) ** 2, axis=-1), 1e-12)
        w = lam_f / dist2
        H = np.zeros((n, n), dtype=np.float64)
        np.add.at(H, (i, i), w)
        np.add.at(H, (k, k), w)
        np.add.at(H, (i, k), -w)
        np.add.at(H, (k, i), -w)
    elif reg_kind == "curvature":
        degree = np.zeros((n,), dtype=np.float64)
        np.add.at(degree, i, 1.0)
        np.add.at(degree, k, 1.0)
        inv_degree = np.where(degree > 0, 1.0 / np.maximum(degree, 1e-12), 0.0)
        B = np.zeros((n, n), dtype=np.float64)
        np.add.at(B, (i, k), 1.0)
        np.add.at(B, (k, i), 1.0)
        B = B * inv_degree[:, None]
        M = np.eye(n, dtype=np.float64) - B
        H = lam_f * (M.T @ M)
    elif reg_kind == "adaptive_split":
        if vertex_lam is None:
            raise ValueError("vertex_lam required for adaptive_split")
        vl = np.asarray(vertex_lam, dtype=np.float64)
        lam_edge = lam_f * 0.5 * (vl[i] + vl[k])
        H = np.zeros((n, n), dtype=np.float64)
        np.add.at(H, (i, i), lam_edge)
        np.add.at(H, (k, k), lam_edge)
        np.add.at(H, (i, k), -lam_edge)
        np.add.at(H, (k, i), -lam_edge)
    else:
        raise ValueError(f"Unknown reg_kind {reg_kind!r}")

    tr = float(np.trace(H))
    ridge = float(ridge_scale) * max(tr / n, 1.0)
    H = H + ridge * np.eye(n, dtype=np.float64)
    # Symmetrize defensively to suppress any asymmetry from rounding.
    return 0.5 * (H + H.T)


def solve_source_positive(
    *,
    basis: jnp.ndarray,
    resid_no_source: jnp.ndarray,
    inv_var: jnp.ndarray,
    edges: np.ndarray,
    lam: float,
    reg_kind: str = "constant_gradient",
    vertex_positions: np.ndarray | None = None,
    vertex_lam: np.ndarray | None = None,
    ridge_scale: float = 1e-8,
):
    """Non-negative source coefficients via scipy.optimize.lsq_linear.

    The regularization matrix is built in float64 numpy here to avoid the
    float32-Cholesky failures we hit on JAX-side builds at large lambda /
    large dynamic-range edge weights.
    """
    from scipy.optimize import lsq_linear

    basis_np = np.asarray(jax.device_get(basis), dtype=np.float64)
    resid_np = np.asarray(jax.device_get(resid_no_source), dtype=np.float64).reshape(-1)
    inv_var_np = np.asarray(jax.device_get(inv_var), dtype=np.float64).reshape(-1)
    edges_np = np.asarray(edges, dtype=np.int32)
    n_vertices = basis_np.shape[0]

    sqrt_w = np.sqrt(inv_var_np)
    design = basis_np.reshape(n_vertices, -1).T
    lhs_img = design * sqrt_w[:, None]
    rhs_img = resid_np * sqrt_w

    vpos_np = None if vertex_positions is None else np.asarray(vertex_positions, dtype=np.float64)
    vlam_np = None if vertex_lam is None else np.asarray(vertex_lam, dtype=np.float64)

    last_err: Exception | None = None
    for boost in (1.0, 1e2, 1e4, 1e6):
        H = _build_reg_matrix_numpy(
            reg_kind=reg_kind,
            num_vertices=n_vertices,
            edges=edges_np,
            lam=lam,
            vertex_positions=vpos_np,
            vertex_lam=vlam_np,
            ridge_scale=float(ridge_scale) * boost,
        )
        try:
            chol = np.linalg.cholesky(H)
            break
        except np.linalg.LinAlgError as exc:
            last_err = exc
            continue
    else:  # pragma: no cover - fallback for hopelessly ill-conditioned H
        raise last_err  # type: ignore[misc]

    lhs = np.vstack([lhs_img, chol])
    rhs = np.concatenate([rhs_img, np.zeros(n_vertices, dtype=np.float64)])
    result = lsq_linear(
        lhs,
        rhs,
        bounds=(0.0, np.inf),
        method="trf",
        tol=1e-8,
        lsmr_tol="auto",
        max_iter=500,
        verbose=0,
    )
    return jnp.asarray(result.x, dtype=jnp.float32), result


def vertex_lam_adaptive_split(
    reference_brightness: np.ndarray,
    *,
    lam_ratio: float = 1e-3,
) -> np.ndarray:
    """Relative per-vertex weights in [lam_ratio, 1]; bright vertices are closer to lam_ratio."""
    ref = np.clip(np.asarray(reference_brightness, dtype=np.float64), 0.0, None)
    if ref.size == 0:
        return np.full(0, 1.0, dtype=np.float64)
    r = ref - np.min(ref)
    denom = max(float(np.max(r)), np.finfo(float).tiny)
    normalized = r / denom
    return lam_ratio + (1.0 - lam_ratio) * (1.0 - normalized)
