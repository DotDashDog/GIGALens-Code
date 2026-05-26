from __future__ import annotations

import jax.numpy as jnp
import numpy as np
try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None

from voronoi_src.pixelized_regularization import (
    REGULARIZATION_BUILDERS,
    build_regularization_matrix,
)


def _random_spd_edges(num_vertices: int, rng: np.random.Generator) -> np.ndarray:
    """Simple path graph plus random chords for testing."""
    edges = [(i, i + 1) for i in range(num_vertices - 1)]
    for _ in range(num_vertices):
        i, k = rng.integers(0, num_vertices, size=2)
        if i != k:
            edges.append((min(i, k), max(i, k)))
    edges = np.unique(np.sort(np.asarray(edges, dtype=np.int32), axis=1), axis=0)
    return edges


def _quadratic_form(H: np.ndarray, s: np.ndarray) -> float:
    return float(s @ H @ s)


def test_constant_gradient_algebraic():
    rng = np.random.default_rng(0)
    n = 8
    edges = _random_spd_edges(n, rng)
    lam = 0.37
    s = rng.normal(size=(n,))
    H = np.asarray(
        build_regularization_matrix(
            "constant_gradient",
            num_vertices=n,
            edges=jnp.asarray(edges),
            lam=jnp.asarray(lam, dtype=jnp.float32),
            ridge_scale=0.0,
        )
    )
    manual = sum(lam * (s[i] - s[k]) ** 2 for i, k in edges)
    assert np.isclose(_quadratic_form(H, s), manual, rtol=1e-5, atol=1e-5)


def test_distance_weighted_algebraic():
    rng = np.random.default_rng(1)
    n = 10
    edges = _random_spd_edges(n, rng)
    pos = rng.normal(size=(n, 2)).astype(np.float32)
    lam = 0.21
    s = rng.normal(size=(n,))
    H = np.asarray(
        build_regularization_matrix(
            "distance_weighted_gradient",
            num_vertices=n,
            edges=jnp.asarray(edges),
            lam=jnp.asarray(lam, dtype=jnp.float32),
            vertex_positions=jnp.asarray(pos),
            ridge_scale=0.0,
        )
    )
    manual = sum(
        lam * (1.0 / max(float(np.sum((pos[i] - pos[k]) ** 2)), 1e-12)) * (s[i] - s[k]) ** 2
        for i, k in edges
    )
    assert np.isclose(_quadratic_form(H, s), manual, rtol=1e-5, atol=1e-4)


def test_curvature_algebraic():
    rng = np.random.default_rng(2)
    n = 9
    edges = _random_spd_edges(n, rng)
    lam = 0.15
    s = rng.normal(size=(n,))
    H = np.asarray(
        build_regularization_matrix(
            "curvature",
            num_vertices=n,
            edges=jnp.asarray(edges),
            lam=jnp.asarray(lam, dtype=jnp.float32),
            ridge_scale=0.0,
        )
    )
    neighbors: dict[int, list[int]] = {i: [] for i in range(n)}
    for i, k in edges:
        neighbors[i].append(k)
        neighbors[k].append(i)
    manual = sum(
        lam * (s[i] - float(np.mean(s[neighbors[i]]))) ** 2 for i in range(n) if neighbors[i]
    )
    assert np.isclose(_quadratic_form(H, s), manual, rtol=1e-4, atol=1e-3)


def test_regularization_spd(kind: str):
    rng = np.random.default_rng(3)
    n = 12
    edges = _random_spd_edges(n, rng)
    kwargs = dict(
        num_vertices=n,
        edges=jnp.asarray(edges),
        lam=jnp.asarray(0.1, dtype=jnp.float32),
    )
    if kind == "distance_weighted_gradient":
        kwargs["vertex_positions"] = jnp.asarray(rng.normal(size=(n, 2)), dtype=jnp.float32)
    if kind == "adaptive_split":
        kwargs["vertex_lam"] = jnp.linspace(1e-3, 1.0, n, dtype=jnp.float32)
    H = np.asarray(build_regularization_matrix(kind, **kwargs))
    eigvals = np.linalg.eigvalsh(H)
    assert np.all(eigvals > 0)


if __name__ == "__main__":
    test_constant_gradient_algebraic()
    test_distance_weighted_algebraic()
    test_curvature_algebraic()
    for kind in sorted(REGULARIZATION_BUILDERS):
        test_regularization_spd(kind)
    print("All regularization algebra tests passed.")
