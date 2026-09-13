"""Diagnostics for pixelized sources (ported from the retired voronoi_src)."""
from __future__ import annotations

import copy
from typing import Dict, Iterable, List

import jax.numpy as jnp
import numpy as np

__all__ = ["alternating_pattern_score", "hyperparameter_scan", "with_component_param"]


def alternating_pattern_score(values, edges, *, bright_fraction: float = 0.1) -> float:
    """Mean per-edge sign product ``sign(s_i - <s>) sign(s_k - <s>)`` over edges touching
    the brightest ``bright_fraction`` of vertices, in ``[-1, 1]``. Strongly negative
    means checkerboard / alternating tiles among the bright vertices (an
    under-regularized inversion); positive means locally smooth bright structure."""
    s = np.asarray(values, dtype=np.float64).reshape(-1)
    e = np.asarray(edges, dtype=np.int64)
    if e.size == 0 or s.size == 0:
        return 0.0
    thr = np.quantile(s, 1.0 - float(bright_fraction))
    bright = s >= thr
    i, k = e[:, 0], e[:, 1]
    m = bright[i] | bright[k]
    if not np.any(m):
        return 0.0
    d = s - s.mean()
    return float(np.mean(np.sign(d[i[m]]) * np.sign(d[k[m]])))


def with_component_param(model, params: Dict, plane: int, light: int, **values) -> Dict:
    """Copy of a structured params tree with one light component's entries replaced."""
    out = copy.deepcopy(params)
    node = model.component_params(out, plane, "light", light)
    for k, v in values.items():
        if k not in node:
            raise KeyError(f"component has no parameter {k!r}; has {sorted(node)}")
        node[k] = v
    return out


def hyperparameter_scan(term, model, params: Dict, plane: int, light: int, name: str,
                        values: Iterable[float]) -> List[Dict[str, float]]:
    """Evaluate a gigalens ``ImageLikelihoodTerm``'s ``evidence_terms`` on a grid of one
    regularizer hyperparameter, everything else fixed at ``params``. Rows carry
    ``chi2, sHs, logdetA, logdetH, log_like, minus2logZ``."""
    rows = []
    for v in values:
        p = with_component_param(model, params, plane, light, **{name: jnp.asarray(float(v))})
        t = term.evidence_terms(p)
        row = {name: float(v)}
        for key in ("chi2", "sHs", "logdetA", "logdetH", "log_like"):
            row[key] = float(np.asarray(t[key]).reshape(-1)[0])
        row["minus2logZ"] = -2.0 * row["log_like"]
        rows.append(row)
    return rows
