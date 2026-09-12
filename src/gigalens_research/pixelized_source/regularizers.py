"""Quadratic priors on mesh-source coefficients.

A :class:`QuadraticRegularizer` is the Gaussian prior ``p(s) ∝ exp(-1/2 s^T H s)`` on
the vertex values of a :class:`~.profiles.MeshSource`. It owns its hyperparameters
(by name; they become ordinary sampled parameters of the profile) and exposes the
matrix and its log-determinant. Anything that can produce an SPD ``H`` -- graph
Laplacians, curvature operators, a GP precision on the vertices -- fits the interface.

The evidence term (:mod:`.likelihood`) is the only consumer; it needs ``matrix`` and
``logdet`` for one sample at a time (hyperparameters are scalars per call).
"""
from __future__ import annotations

from typing import Optional, Sequence, Union

import jax.numpy as jnp
import numpy as np

from .domain import SourceDomain

__all__ = ["QuadraticRegularizer", "GraphLaplacian"]


class QuadraticRegularizer:
    """Interface. ``hyperparams`` are appended to the owning profile's ``params``."""

    hyperparams: Sequence[str] = ()

    def matrix(self, **hyper) -> jnp.ndarray:
        """``(I, I)`` SPD precision matrix for one set of scalar hyperparameters."""
        raise NotImplementedError

    def logdet(self, **hyper) -> jnp.ndarray:
        """``log det H`` for the same hyperparameters (analytic where possible)."""
        raise NotImplementedError


class GraphLaplacian(QuadraticRegularizer):
    """``H(lam) = lam * (L + ridge * mean_eig(L) * I)`` on the domain's edge graph.

    ``kind``:
      * ``"gradient"``  -- ``s^T L s = sum_edges w_ij (s_i - s_j)^2`` (constant-gradient
        / zeroth-order penalty), with ``edge_weights`` ``None`` (unit),
        ``"inverse_distance"`` (``1 / |x_i - x_j|^2``) or an explicit ``(E,)`` array
        (e.g. an AdaptSplit-style per-edge lambda field).
      * ``"curvature"`` -- ``sum_i (s_i - mean_{j ~ i} s_j)^2`` (constant-curvature).

    ``L`` has a null space (the constants for ``gradient``), so the prior is made proper
    with a ridge that scales with ``lam`` -- ``ridge_scale`` times the mean eigenvalue --
    which is a weak Gaussian prior on the mean level. Because the ridge rides with
    ``lam``, ``log det H = I log lam + const`` exactly, and the constant is computed once
    from the eigenvalues (no ``slogdet`` inside the likelihood).

    Hyperparameter: ``lam > 0`` (give it e.g. a LogNormal prior; its scale is set by the
    basis normalization -- with barycentric bases and pixel-area flux conversion,
    ``lam ~ 1e-4 .. 1e-1`` was the useful range on the old prototype).
    """

    hyperparams = ("lam",)

    def __init__(self, domain: SourceDomain, *, kind: str, ridge_scale: float,
                 edge_weights: Union[None, str, np.ndarray] = None):
        n = domain.n_basis
        edges = np.asarray(domain.edges, dtype=np.int64)
        i, k = edges[:, 0], edges[:, 1]
        if kind == "gradient":
            if edge_weights is None:
                w = np.ones(edges.shape[0])
            elif isinstance(edge_weights, str):
                if edge_weights != "inverse_distance":
                    raise ValueError(f"edge_weights string must be 'inverse_distance'; got {edge_weights!r}")
                pos = np.asarray(domain.vertex_xy, dtype=np.float64)
                d2 = np.sum((pos[i] - pos[k]) ** 2, axis=-1)
                w = 1.0 / np.maximum(d2, 1e-12)
            else:
                w = np.asarray(edge_weights, dtype=np.float64).reshape(-1)
                if w.shape[0] != edges.shape[0]:
                    raise ValueError(f"edge_weights must have one entry per edge ({edges.shape[0]}); got {w.shape[0]}")
                if not np.all(np.isfinite(w)) or np.any(w < 0):
                    raise ValueError("edge_weights must be finite and non-negative")
            L = np.zeros((n, n))
            np.add.at(L, (i, i), w)
            np.add.at(L, (k, k), w)
            np.add.at(L, (i, k), -w)
            np.add.at(L, (k, i), -w)
        elif kind == "curvature":
            if edge_weights is not None:
                raise ValueError("edge_weights are only defined for kind='gradient'")
            deg = np.zeros(n)
            np.add.at(deg, i, 1.0)
            np.add.at(deg, k, 1.0)
            B = np.zeros((n, n))
            np.add.at(B, (i, k), 1.0)
            np.add.at(B, (k, i), 1.0)
            B = B * np.where(deg > 0, 1.0 / np.maximum(deg, 1.0), 0.0)[:, None]
            M = np.eye(n) - B
            L = M.T @ M
        else:
            raise ValueError(f"kind must be 'gradient' or 'curvature'; got {kind!r}")
        L = 0.5 * (L + L.T)
        eig = np.linalg.eigvalsh(L)
        mean_eig = float(np.mean(eig))
        if not (ridge_scale > 0):
            raise ValueError("ridge_scale must be > 0: the graph Laplacian is singular, so the "
                             "prior is improper (log det H = -inf) without a ridge")
        ridge = float(ridge_scale) * mean_eig
        self.kind = kind
        self.n = n
        self.ridge = ridge
        self.L_reg = jnp.asarray(L + ridge * np.eye(n))
        self._logdet_unit = float(np.sum(np.log(np.maximum(eig, 0.0) + ridge)))
        self.eigenvalues = eig

    def matrix(self, *, lam):
        return jnp.asarray(lam) * self.L_reg

    def logdet(self, *, lam):
        return self.n * jnp.log(jnp.asarray(lam)) + self._logdet_unit

    def __repr__(self):
        return f"GraphLaplacian(kind={self.kind!r}, I={self.n}, ridge={self.ridge:.3g})"
