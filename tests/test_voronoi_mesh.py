"""Regression tests for the pixelized-source (voronoi_src) numerics.

Both bugs were found in the 2026-09-12 audit of ``gigalens_research.voronoi_src``:

1. ``build_regular_imageplane_mesh`` split each grid cell along the v00 -> v11
   diagonal but tested membership against the *other* diagonal (``u + v >= 1``),
   so half of all sub-pixels were assigned a triangle that does not contain them
   (barycentric weights down to -1, then silently clipped by the simulator).
2. ``PixelizedSourceSimulator._barycentric_weights`` guarded a degenerate
   (folded) triangle with ``sign(denom) * eps + eps``, which is exactly zero for a
   slightly negative denom: division by zero, and NaN gradients through the
   ``jnp.where`` mask for every parameter path touching a fold.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from gigalens.jax.profiles.mass.epl import EPL
from gigalens.simulator import LensWCS, SimulatorConfig
from gigalens_research.voronoi_src.delaunay_mesh import build_regular_imageplane_mesh
from gigalens_research.voronoi_src.pixelized_simulator import PixelizedSourceSimulator


def _bary_numpy(p, a, b, c):
    """Barycentric weights (N, 3) of points p in triangles (a, b, c), float64."""
    p, a, b, c = (np.asarray(x, dtype=np.float64) for x in (p, a, b, c))
    v0, v1, v2 = b - a, c - a, p - a
    den = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
    u = (v2[:, 0] * v1[:, 1] - v2[:, 1] * v1[:, 0]) / den
    v = (v0[:, 0] * v2[:, 1] - v0[:, 1] * v2[:, 0]) / den
    return np.stack([1.0 - u - v, u, v], axis=-1)


# --------------------------------------------------------------------- bug 1: diagonal
@pytest.mark.parametrize("supersample", [1, 2])
@pytest.mark.parametrize("n_seed", [(5, 7), (6, 6)])
def test_regular_mesh_every_subpixel_lies_in_its_assigned_triangle(supersample, n_seed):
    n_seed_y, n_seed_x = n_seed
    mesh = build_regular_imageplane_mesh(
        num_pix=16, delta_pix=0.1, supersample=supersample,
        n_seed_y=n_seed_y, n_seed_x=n_seed_x, extent=0.6,
    )
    inside = mesh.subpix_tri >= 0
    # extent (0.6) < image half-size (0.8): both inside and outside sub-pixels exist
    assert inside.any() and (~inside).any()

    vid = mesh.simplices[mesh.subpix_tri[inside]]
    w = _bary_numpy(
        mesh.subpix_xy[inside], mesh.seed_xy[vid[:, 0]], mesh.seed_xy[vid[:, 1]], mesh.seed_xy[vid[:, 2]]
    )
    # Containment: no negative weight beyond float32 rounding (was -1 before the fix).
    assert w.min() >= -1e-5, f"min barycentric weight {w.min():.3f}: sub-pixels outside assigned triangle"
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-5)

    # Sub-pixels flagged outside really are outside the seed box.
    out = mesh.subpix_xy[~inside]
    assert (np.abs(out) > mesh.extent + 1e-6).any(axis=1).all()

    # Every triangle is used by at least one sub-pixel (cells are much larger than sub-pixels).
    assert np.unique(mesh.subpix_tri[inside]).size == mesh.simplices.shape[0]


def test_regular_mesh_triangles_are_consistently_oriented():
    mesh = build_regular_imageplane_mesh(
        num_pix=8, delta_pix=0.1, supersample=1, n_seed_y=4, n_seed_x=5, extent=0.3
    )
    a, b, c = (mesh.seed_xy[mesh.simplices[:, k]].astype(np.float64) for k in range(3))
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    assert (area2 > 0).all()


def test_regular_mesh_subpixel_grid_matches_simulator_wcs():
    """The mesh lookup table is indexed in the simulator's LensWCS ravel order."""
    mesh = build_regular_imageplane_mesh(
        num_pix=10, delta_pix=0.05, supersample=2, n_seed_y=4, n_seed_x=4, extent=0.2
    )
    X, Y = LensWCS(n=10, supersample=2, pix_scale=0.05).pixel_grid()
    assert mesh.subpix_xy.shape == (400, 2)
    np.testing.assert_allclose(mesh.subpix_xy, np.stack([X.ravel(), Y.ravel()], axis=-1), atol=1e-6)


def test_simulator_refuses_mesh_built_for_another_grid():
    mesh = build_regular_imageplane_mesh(
        num_pix=10, delta_pix=0.05, supersample=1, n_seed_y=4, n_seed_x=4, extent=0.2
    )
    cfg = SimulatorConfig(delta_pix=0.05, num_pix=10, supersample=2)
    with pytest.raises(ValueError, match="does not match the simulator"):
        PixelizedSourceSimulator(lenses=[], lens_light_profiles=[], sim_config=cfg, mesh=mesh)


# ------------------------------------------------------- bug 2: degenerate triangles
@pytest.mark.parametrize("tiny", [0.0, -1e-7, 1e-7])
def test_barycentric_weights_finite_with_finite_gradients_on_degenerate_triangle(tiny):
    bw = PixelizedSourceSimulator._barycentric_weights
    a = jnp.array([0.0, 0.0])
    b = jnp.array([1.0, 0.0])
    c = jnp.array([1.0, tiny])  # (nearly) collinear; tiny < 0 is the inverted case
    p = jnp.array([0.3, 0.2])

    wa, wb, wc, deg = bw(p, a, b, c)
    assert bool(deg)
    assert np.all(np.isfinite(np.asarray([wa, wb, wc])))

    def masked_total(p_, c_):
        # exactly the simulator's masking pattern: where(valid, w, 0)
        wa, wb, wc, deg = bw(p_, a, b, c_)
        valid = ~deg
        return jnp.where(valid, wa, 0.0) + jnp.where(valid, wb, 0.0) + jnp.where(valid, wc, 0.0)

    g_p, g_c = jax.grad(masked_total, argnums=(0, 1))(p, c)
    assert np.all(np.isfinite(np.asarray(g_p))), g_p  # sub-pixel path (was NaN)
    assert np.all(np.isfinite(np.asarray(g_c))), g_c  # vertex / mass-parameter path (was NaN)
    # A masked-out triangle contributes nothing, so its gradient is exactly zero.
    assert np.all(np.asarray(g_p) == 0) and np.all(np.asarray(g_c) == 0)


def test_barycentric_weights_healthy_triangle_and_orientation():
    bw = PixelizedSourceSimulator._barycentric_weights
    a, b, c = jnp.array([0.0, 0.0]), jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])
    p = jnp.array([0.25, 0.25])
    wa, wb, wc, deg = bw(p, a, b, c)
    assert not bool(deg)
    np.testing.assert_allclose(np.asarray([wa, wb, wc]), [0.5, 0.25, 0.25], atol=1e-6)
    # Clockwise orientation (negative signed area) is not degenerate and gives the same weights.
    wa2, wb2, wc2, deg2 = bw(p, a, c, b)
    assert not bool(deg2)
    np.testing.assert_allclose(np.asarray([wa2, wc2, wb2]), [0.5, 0.25, 0.25], atol=1e-6)
    # Gradient wrt p is exact: d(wb)/dp = (1, 0)
    np.testing.assert_allclose(np.asarray(jax.grad(lambda q: bw(q, a, b, c)[1])(p)), [1.0, 0.0], atol=1e-6)


# -------------------------------------------------------------- end-to-end smoke
def test_pixelized_basis_partition_of_unity_under_a_lens():
    """Basis images sum to the pixel area wherever every sub-pixel is inside the mesh.

    Uses the new-API EPL profile (``deriv`` contract unchanged) and no PSF, so the
    only transformation is barycentric interpolation + pooling * pixel area.
    """
    num_pix, delta_pix, ss = 12, 0.1, 2
    mesh = build_regular_imageplane_mesh(
        num_pix=num_pix, delta_pix=delta_pix, supersample=ss, n_seed_y=7, n_seed_x=7, extent=0.5
    )
    cfg = SimulatorConfig(delta_pix=delta_pix, num_pix=num_pix, supersample=ss)
    sim = PixelizedSourceSimulator(lenses=[EPL(50)], lens_light_profiles=[], sim_config=cfg, mesh=mesh)
    lens_params = [dict(theta_E=0.35, gamma=2.0, e1=0.05, e2=-0.03, center_x=0.0, center_y=0.0)]
    out = sim.basis_and_lens_light((lens_params, []))
    basis = np.asarray(out.basis_images)
    assert basis.shape == (mesh.seed_xy.shape[0], num_pix, num_pix)
    assert np.all(np.isfinite(basis)) and basis.min() >= 0.0

    total = basis.sum(axis=0)
    inside = (mesh.subpix_tri >= 0).reshape(num_pix * ss, num_pix * ss)
    full_pixels = inside.reshape(num_pix, ss, num_pix, ss).all(axis=(1, 3))
    assert full_pixels.any()
    np.testing.assert_allclose(total[full_pixels], delta_pix**2, rtol=1e-4)
    assert int(out.degenerate_subpix) == 0
