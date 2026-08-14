"""Dataset-kind dispatch and observation-uncertainty normalization.

A :class:`~gigalens.jax.scene_prob_model.ProbModel` holds a heterogeneous list of
``Dataset`` objects. The research-side view layer used to assume every one of them
was imaging: :meth:`Posterior.observed_for` read ``ds.image``, ``_band_extent`` read
``ds.sim_config``, and :attr:`ProbModel.simulators` was indexed by dataset position.
None of that holds once a point-source observation is in the list, so this module
provides the one thing those callers actually need — *what kind of observation is
this?* — plus the uncertainty normalization the point-source plotters run on.

Kinds are deliberately fine-grained. In particular the two point-source modules are
NOT the same kind:

- :data:`KIND_POINT_SOURCE` — :mod:`gigalens.jax.point_source_position`. A calibrated
  Gaussian in the image plane; its ``chi2`` is a real goodness-of-fit over
  ``2 n_images`` observables, so pulls, reduced-chi2 and the chi2 decomposition all
  mean what they look like.
- :data:`KIND_POINT_SOURCE_LOSS` — :mod:`gigalens.jax.point_source`, the three-term
  hand-weighted positions+flux+time-delay loss. Its ``chi2`` is a weighted loss with a
  stand-in event count, so plotting it as pulls against a unit Gaussian would claim a
  calibration that module never offers. The point-source plotters reject it by name
  rather than drawing something misleading.

Uncertainties
-------------
:func:`image_position_covariances` returns one 2x2 covariance per observed image, from
whichever form the dataset carries — scalar, per-image, per-coordinate, or a full
matrix. Everything downstream (pulls, error ellipses, the position chi2) is written
against that covariance, so the per-coordinate and full-covariance cases share one code
path instead of two that can drift.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

#: Imaging observation (``gigalens.jax.scene_prob_model.ImageData``).
KIND_IMAGE = "image"
#: Position-only / positions+channels point source, calibrated Gaussian
#: (``gigalens.jax.point_source_position``).
KIND_POINT_SOURCE = "point_source"
#: Three-term hand-weighted point-source loss (``gigalens.jax.point_source``).
KIND_POINT_SOURCE_LOSS = "point_source_loss"
#: Anything this module does not recognize.
KIND_UNKNOWN = "unknown"


def dataset_kind(dataset: Any) -> str:
    """Classify one ``Dataset`` into one of the ``KIND_*`` constants.

    Resolution is by ``isinstance`` against the gigalens classes first (the
    authority), with a duck-typed fallback so a locally-subclassed or
    not-yet-importable observation still lands on the right plotters instead of
    silently reaching the imaging path.
    """
    try:
        from gigalens.jax.scene_prob_model import ImageData
        if isinstance(dataset, ImageData):
            return KIND_IMAGE
    except ImportError:  # pragma: no cover - gigalens always present in practice
        pass

    try:
        from gigalens.jax.point_source_position import PointSourcePositionData
        if isinstance(dataset, PointSourcePositionData):
            return KIND_POINT_SOURCE
    except ImportError:  # pragma: no cover
        pass

    try:
        from gigalens.jax.point_source import PointSourceData
        if isinstance(dataset, PointSourceData):
            return KIND_POINT_SOURCE_LOSS
    except ImportError:  # pragma: no cover
        pass

    # Duck-typed fallback. ``is_pointsource`` is the marker the three-term module
    # sets (and the position module deliberately does not), so it separates the two
    # point-source kinds without importing either.
    if hasattr(dataset, "x_img") and hasattr(dataset, "y_img"):
        return (KIND_POINT_SOURCE_LOSS
                if getattr(dataset, "is_pointsource", False) else KIND_POINT_SOURCE)
    if hasattr(dataset, "image") and hasattr(dataset, "sim_config"):
        return KIND_IMAGE
    return KIND_UNKNOWN


def is_imaging(dataset: Any) -> bool:
    """``True`` for an observation that carries a pixel grid."""
    return dataset_kind(dataset) == KIND_IMAGE


#: Cross-image block magnitude, as a fraction of the largest per-image variance,
#: above which a full joint (2n, 2n) cov_img is treated as carrying real image-to-
#: image correlation. Below this, dropping the off-diagonal blocks to get a per-image
#: (n, 2, 2) covariance is numerical noise, not an approximation; above it, the term's
#: own likelihood (which whitens all 2n coordinates jointly, see
#: ``gigalens.jax.point_source_position._validate_cov_img``) is scoring a shared mode
#: this plotting layer's per-image chi2 decomposition (``mahalanobis_sq``) cannot
#: represent, so silently keeping only the diagonal blocks would make the panel
#: disagree with what was actually fit.
_CROSS_IMAGE_COV_RTOL = 1e-3


def _diagonal_blocks(cov2n: np.ndarray, n: int, source: str) -> np.ndarray:
    """Extract the ``n`` per-image 2x2 diagonal blocks from a full joint
    ``(2n, 2n)`` astrometric covariance.

    ``cov_img`` on a :class:`~gigalens.jax.point_source_position.PointSourcePositionData`
    is always stored in this INTERLEAVED form (``[x0, y0, x1, y1, ...]``, image
    ``i``'s x at index ``2i``, its y at ``2i+1``) once constructed — the constructor
    normalizes the ``(n, 2, 2)`` per-image-block input form into it too. Raises rather
    than silently truncating when the off-diagonal (cross-image) blocks are not
    negligible; see :data:`_CROSS_IMAGE_COV_RTOL`.
    """
    c = np.asarray(cov2n, dtype=float)
    blocks = np.stack([c[2 * i:2 * i + 2, 2 * i:2 * i + 2] for i in range(n)])
    off = c.copy()
    for i in range(n):
        off[2 * i:2 * i + 2, 2 * i:2 * i + 2] = 0.0
    off_mag = float(np.abs(off).max()) if off.size else 0.0
    scale = float(np.abs(np.diag(c)).max())
    if scale > 0 and off_mag > _CROSS_IMAGE_COV_RTOL * scale:
        raise ValueError(
            f"{source} has shape {c.shape} (the full joint (2n, 2n) astrometric "
            f"covariance) with a non-negligible cross-image block (max off-diagonal "
            f"block entry {off_mag:.3e}, {off_mag / scale:.3e} of the largest "
            f"variance, vs. a {_CROSS_IMAGE_COV_RTOL:.0e} tolerance). This plotting "
            "layer's chi2 decomposition (mahalanobis_sq) sums PER-IMAGE quadratic "
            "forms and cannot represent image-to-image correlation, so extracting the "
            "diagonal blocks here would silently disagree with the likelihood's "
            "actual jointly-whitened scored chi2 for this dataset. The panel needs "
            "the joint form (the term's own whiten) before this dataset can be "
            "plotted correctly.")
    return blocks


def _as_covariance(sig: np.ndarray, n: int, source: str) -> np.ndarray:
    """Normalize one astrometric-uncertainty array to ``(n, 2, 2)`` covariances.

    Accepted forms:

    - scalar -> one isotropic sigma shared by every image and coordinate,
    - ``(n,)`` -> per-image isotropic sigma,
    - ``(n, 2)`` -> per-image, per-coordinate sigma (x, y),
    - ``(n, 2, 2)`` -> per-image full covariance,
    - ``(2n, 2n)`` -> the full joint astrometric covariance
      :class:`~gigalens.jax.point_source_position.PointSourcePositionData` actually
      stores as ``cov_img`` (INTERLEAVED ``[x0, y0, x1, y1, ...]``); reduced to its
      ``n`` diagonal blocks via :func:`_diagonal_blocks`, which raises rather than
      dropping a non-negligible cross-image correlation silently.

    The 1-D and 2-D forms hold **standard deviations** (the dataset's own
    convention), so they are squared onto the diagonal. The 3-D form is the only
    sensible 2x2 generalization of "the uncertainty of this image" and is therefore
    read as a **covariance**, not as a matrix of standard deviations and not as a
    Cholesky factor or a precision matrix. That reading is checked, not assumed: a
    non-symmetric or non-positive-definite matrix raises here rather than being
    whitened into a plausible-looking wrong pull.
    """
    sig = np.asarray(sig, dtype=float)
    if sig.ndim == 0:
        var = np.full((n, 2), float(sig) ** 2)
        return np.einsum("ij,jk->ijk", var, np.eye(2))
    if sig.shape == (n,):
        var = np.stack([sig, sig], axis=1) ** 2
        return np.einsum("ij,jk->ijk", var, np.eye(2))
    if sig.shape == (n, 2):
        return np.einsum("ij,jk->ijk", sig ** 2, np.eye(2))
    if sig.shape == (n, 2, 2):
        asym = np.abs(sig - np.swapaxes(sig, -1, -2)).max()
        scale = np.abs(sig).max()
        if asym > 1e-9 * max(scale, 1.0):
            raise ValueError(
                f"{source} has shape (n, 2, 2) but is not symmetric (max |C - C^T| = "
                f"{asym:.3e}). This layer reads a 2x2 per-image uncertainty as a "
                "COVARIANCE matrix; a Cholesky factor, a precision matrix or a "
                "matrix of standard deviations would be whitened into pulls that "
                "look reasonable and are wrong. Pass the covariance, or teach "
                "gigalens_research.inference_utils.datasets the new convention.")
        eigs = np.linalg.eigvalsh(sig)
        if not np.all(eigs > 0):
            raise ValueError(
                f"{source} has a non-positive-definite covariance (min eigenvalue "
                f"{eigs.min():.3e}); a Gaussian likelihood is undefined there, so "
                "there is no honest pull or error ellipse to draw.")
        return sig
    if sig.shape == (2 * n, 2 * n):
        return _diagonal_blocks(sig, n, source)
    raise ValueError(
        f"{source} has shape {sig.shape}, which is none of the recognized forms for "
        f"{n} images: scalar, ({n},), ({n}, 2) standard deviations, ({n}, 2, 2), or "
        f"({2 * n}, {2 * n}) covariances. Refusing to broadcast — a "
        "broadcastable-but-wrong shape would silently misweight every residual.")


def image_position_covariances(dataset: Any) -> np.ndarray:
    """Per-image astrometric covariance of a point-source dataset, ``(n, 2, 2)``.

    Reads an explicit ``cov_img`` if the dataset carries one, else normalizes
    ``sigma_img`` through :func:`_as_covariance`. Both paths end in covariances, so
    callers never branch on which uncertainty form the dataset was built with.
    """
    n = int(dataset.n_images)
    cov = getattr(dataset, "cov_img", None)
    if cov is not None:
        return _as_covariance(cov, n, "point-source dataset cov_img")
    sig = getattr(dataset, "sigma_img", None)
    if sig is None:
        raise AttributeError(
            f"{type(dataset).__name__} carries neither cov_img nor sigma_img, so it "
            "has no astrometric uncertainty to whiten residuals with.")
    return _as_covariance(sig, n, "point-source dataset sigma_img")


def whiten(residuals: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Whiten per-image 2-vector residuals by their covariances.

    ``residuals`` is ``(n, 2)`` (or ``(n, 2, k)`` for ``k`` posterior draws) and
    ``cov`` is ``(n, 2, 2)``. Returns the same shape as ``residuals``.

    The whitening is ``L^-1 r`` with ``C = L L^T`` (Cholesky), so the returned
    components are uncorrelated unit-variance pulls whose squared sum is exactly the
    Mahalanobis distance ``r^T C^-1 r`` that the likelihood scores. For a diagonal
    covariance this reduces elementwise to ``r / sigma`` — the per-coordinate pull —
    so the correlated and uncorrelated cases are the same plot, not two conventions.
    """
    res = np.asarray(residuals, dtype=float)
    cov = np.asarray(cov, dtype=float)
    squeeze = res.ndim == 2
    if squeeze:
        res = res[..., None]
    chol = np.linalg.cholesky(cov)                      # (n, 2, 2), lower
    # Stacked solve: (n, 2, 2) against (n, 2, k) -> (n, 2, k), one triangular
    # system per image.
    out = np.linalg.solve(chol, res)
    return out[..., 0] if squeeze else out


def mahalanobis_sq(residuals: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """``r^T C^-1 r`` per image: ``(n,)`` (or ``(n, k)`` for ``k`` draws)."""
    w = whiten(residuals, cov)
    return np.sum(w ** 2, axis=1)


def covariance_ellipse(cov2: np.ndarray, n_sigma: float = 1.0
                       ) -> Tuple[float, float, float]:
    """``(width, height, angle_deg)`` of the ``n_sigma`` error ellipse of one 2x2
    covariance, in the format :class:`matplotlib.patches.Ellipse` expects.

    Width and height are full axis lengths (twice the semi-axes), so the patch drawn
    from them encloses the set ``{r : r^T C^-1 r <= n_sigma^2}``.
    """
    cov2 = np.asarray(cov2, dtype=float)
    vals, vecs = np.linalg.eigh(cov2)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    width, height = 2.0 * n_sigma * np.sqrt(np.maximum(vals, 0.0))
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    return float(width), float(height), angle


def isotropic_sigma(cov: np.ndarray) -> np.ndarray:
    """One representative sigma per image (arcsec): ``sqrt(tr(C) / 2)``, ``(n,)``.

    The rms of the two principal axes. Used only where a single scalar scale per
    image is needed for framing or for a log axis reference line — never in a
    residual, which always goes through the full :func:`whiten`.
    """
    cov = np.asarray(cov, dtype=float)
    return np.sqrt(0.5 * (cov[:, 0, 0] + cov[:, 1, 1]))
