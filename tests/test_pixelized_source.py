"""Tests for gigalens_research.pixelized_source (frozen domains, scene-API profiles,
regularized evidence term, correlated field).

Ground truths used: scipy's point location for the triangle mesh, exact reproduction
of linear functions by both interpolants, an independent float64 numpy evaluation of
the Suyu/WD03 evidence, and the exact white-noise limit of the correlated field.
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp
from scipy.spatial import Delaunay

from gigalens.jax.profiles.light.sersic import SersicEllipse
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.scene import Component, LensModel, Plane
from gigalens.jax.scene_prob_model import ImageData, ProbModel
from gigalens.jax.scene_simulator import SceneSimulator
from gigalens.simulator import SimulatorConfig
from gigalens_research.pixelized_source import (
    CorrelatedFieldSource,
    GraphLaplacian,
    MeshSource,
    RegularGridDomain,
    RegularizedImageData,
    TriangleMeshDomain,
    adaptive_delaunay_domain,
)
from gigalens_research.pixelized_source.diagnostics import (
    alternating_pattern_score,
    hyperparameter_scan,
)

tfd = tfp.distributions
F = np.float64

MASS_TRUTH = dict(theta_E=0.9, gamma=2.0, e1=0.05, e2=-0.03, center_x=0.0, center_y=0.0)
SRC_TRUTH = dict(R_sersic=0.25, n_sersic=1.5, e1=0.0, e2=0.0, center_x=0.05, center_y=-0.02, Ie=20.0)


# ====================================================================== domains
def _random_points(rng, n, lo, hi):
    return rng.uniform(lo, hi, size=(n, 2))


def test_regular_grid_partition_of_unity_and_zero_outside():
    dom = RegularGridDomain.centered(center=(0.0, 0.0), half_size=1.0, n=6)
    rng = np.random.default_rng(0)
    p = _random_points(rng, 3000, -1.5, 1.5)
    basis = np.asarray(dom.basis(jnp.asarray(p[:, 0]), jnp.asarray(p[:, 1])))  # (I, N)
    inside = np.all(np.abs(p) <= 1.0, axis=1)
    assert basis.min() >= 0.0
    np.testing.assert_allclose(basis.sum(0)[inside], 1.0, atol=1e-12)
    np.testing.assert_array_equal(basis.sum(0)[~inside], 0.0)
    assert dom.edges.shape == (2 * 6 * 5, 2)


def test_regular_grid_reproduces_linear_functions():
    dom = RegularGridDomain.centered(center=(0.3, -0.2), half_size=0.7, n=5)
    a, b, c = 0.4, -1.3, 2.1
    values = a + b * dom.vertex_xy[:, 0] + c * dom.vertex_xy[:, 1]
    rng = np.random.default_rng(1)
    p = _random_points(rng, 500, -0.35, 0.45)  # strictly inside
    got = np.asarray(dom.interpolate(jnp.asarray(values), jnp.asarray(p[:, 0]), jnp.asarray(p[:, 1])))
    np.testing.assert_allclose(got, a + b * p[:, 0] + c * p[:, 1], atol=1e-12)


def test_regular_grid_batched_interpolate_matches_loop():
    dom = RegularGridDomain.centered(center=(0.0, 0.0), half_size=1.0, n=4)
    rng = np.random.default_rng(2)
    B = 3
    vals = rng.normal(size=(dom.n_basis, B))
    bx = rng.uniform(-1, 1, size=(5, 6, B))
    by = rng.uniform(-1, 1, size=(5, 6, B))
    got = np.asarray(dom.interpolate(jnp.asarray(vals), jnp.asarray(bx), jnp.asarray(by)))
    for k in range(B):
        one = np.asarray(dom.interpolate(jnp.asarray(vals[:, k]), jnp.asarray(bx[..., k]), jnp.asarray(by[..., k])))
        np.testing.assert_allclose(got[..., k], one, atol=1e-12)


def _mesh(rng, n=60):
    pts = _random_points(rng, n, -1.0, 1.0)
    return TriangleMeshDomain.delaunay(pts), Delaunay(pts)


def test_triangle_mesh_point_location_matches_scipy():
    rng = np.random.default_rng(3)
    dom, tri = _mesh(rng)
    q = _random_points(rng, 4000, -1.3, 1.3)
    ref = tri.find_simplex(q)
    got = dom.triangle_of(jnp.asarray(q[:, 0]), jnp.asarray(q[:, 1]))
    # Agreement away from edges; on a shared edge either neighbour is valid (weights agree).
    vids, w = dom.locate(jnp.asarray(q[:, 0]), jnp.asarray(q[:, 1]))
    w = np.asarray(w)
    interior = (ref >= 0) & (w.min(0) > 1e-9)
    np.testing.assert_array_equal(got[interior], ref[interior])
    np.testing.assert_array_equal(got[ref < 0], -1)
    assert (ref >= 0).sum() > 1000 and (ref < 0).sum() > 100


def test_triangle_mesh_partition_of_unity_and_linear_reproduction():
    rng = np.random.default_rng(4)
    dom, tri = _mesh(rng)
    q = _random_points(rng, 3000, -1.3, 1.3)
    inside = tri.find_simplex(q) >= 0
    basis = np.asarray(dom.basis(jnp.asarray(q[:, 0]), jnp.asarray(q[:, 1])))
    assert basis.min() >= -1e-6
    np.testing.assert_allclose(basis.sum(0)[inside], 1.0, atol=1e-9)
    np.testing.assert_array_equal(basis.sum(0)[~inside], 0.0)
    a, b, c = -0.7, 0.9, 1.7
    values = a + b * dom.vertex_xy[:, 0] + c * dom.vertex_xy[:, 1]
    got = np.asarray(dom.interpolate(jnp.asarray(values), jnp.asarray(q[:, 0]), jnp.asarray(q[:, 1])))
    np.testing.assert_allclose(got[inside], (a + b * q[:, 0] + c * q[:, 1])[inside], atol=1e-9)


def test_triangle_mesh_rejects_degenerate_triangles():
    V = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="degenerate"):
        TriangleMeshDomain(V, np.array([[0, 1, 2], [0, 1, 3]]))


def test_domain_operators_jit_and_differentiate():
    rng = np.random.default_rng(5)
    dom, tri = _mesh(rng)
    values = jnp.asarray(rng.normal(size=dom.n_basis))
    f = jax.jit(lambda x, y: dom.interpolate(values, x, y))
    q = jnp.asarray([0.1, -0.2])
    val = f(q[0], q[1])
    g = jax.grad(lambda p: f(p[0], p[1]))(q)
    assert np.isfinite(float(val)) and np.all(np.isfinite(np.asarray(g)))
    # Inside a triangle the interpolant is the plane through its three vertex values.
    t = int(tri.find_simplex(np.asarray(q)[None])[0])
    a, b, c = dom.vertex_xy[tri.simplices[t]]
    va, vb, vc = (float(values[i]) for i in tri.simplices[t])
    M = np.array([[b[0] - a[0], c[0] - a[0]], [b[1] - a[1], c[1] - a[1]]])
    grad_plane = np.linalg.solve(M.T, np.array([vb - va, vc - va]))
    np.testing.assert_allclose(np.asarray(g), grad_plane, atol=1e-9)
    # basis is jittable too (regular grid)
    grid = RegularGridDomain.centered(center=(0, 0), half_size=1, n=4)
    B = jax.jit(grid.basis)(jnp.zeros((3, 3, 2)), jnp.zeros((3, 3, 2)))
    assert B.shape == (16, 3, 3, 2)


# ====================================================================== scene glue
def _cfg(num_pix=24):
    return SimulatorConfig(delta_pix=0.1, num_pix=num_pix, supersample=1, kernel=None)


def _mass_component(free_theta_E=False):
    priors = dict(MASS_TRUTH)
    if free_theta_E:
        priors["theta_E"] = tfd.Normal(F(MASS_TRUTH["theta_E"]), F(0.05))
    return Component(EPL(50), priors, name="lens")


def _mock(seed=0, noise_rms=0.05):
    """Noisy lensed-Sersic mock on the scene API + its noiseless lensed-source image."""
    cfg = _cfg()
    src = Component(SersicEllipse(use_lstsq=False), dict(SRC_TRUTH), name="src")
    truth = LensModel([Plane(mass=[_mass_component()]), Plane(light=[src])])
    params = truth.to_params({})
    clean = np.asarray(SceneSimulator(truth, cfg).simulate(params))
    rng = np.random.default_rng(seed)
    img = clean + noise_rms * rng.standard_normal(clean.shape)
    err = np.full(clean.shape, noise_rms)
    return cfg, img, err, clean, truth, params


def _z(model, **leaf_values):
    """Unconstrained ``z`` from CONSTRAINED leaf values, everything else at the prior's
    unconstrained origin. (A Normal prior's z column is the raw value -- z = 0 is
    theta_E = 0, where the EPL gradient is NaN -- so never hand-write z for mass.)"""
    params = model.constrained(jnp.zeros(model.num_free_params))
    for path, v in leaf_values.items():
        plane, kind, comp, name = path.split("/")
        params["planes"][plane][kind][comp][name] = jnp.asarray(F(v))
    return jnp.asarray(model.unconstrained(params))


def _mesh_model(domain, regularizer=None, free_theta_E=False, lam_prior=None):
    """``regularizer=None`` means a flat-prior basis: GraphLaplacian with lam FIXED tiny."""
    if regularizer is None:
        regularizer = GraphLaplacian(domain, kind="gradient", ridge_scale=1e-6)
        lam_prior = 1e-12
    priors = {"lam": lam_prior if lam_prior is not None else tfd.LogNormal(F(np.log(1e-2)), F(2.0))}
    src = Component(MeshSource(domain, regularizer), priors, name="mesh")
    model = LensModel([Plane(mass=[_mass_component(free_theta_E)]), Plane(light=[src])])
    return model, src


def test_mesh_source_requires_a_regularizer_hyperparameter():
    dom = RegularGridDomain.centered(center=(0.0, 0.0), half_size=1.0, n=4)
    with pytest.raises(ValueError, match="not addressable"):
        MeshSource(dom, None)


def test_mesh_source_basis_partition_of_unity_through_scene_simulator():
    cfg, img, err, clean, truth, _ = _mock()
    dom = RegularGridDomain.centered(center=(0.0, 0.0), half_size=2.0, n=9)
    model, _ = _mesh_model(dom)
    params = model.to_params({})
    sim = SceneSimulator(model, cfg)
    stacked = np.asarray(sim.lstsq_simulate(params, jnp.asarray(img), jnp.asarray(err),
                                            return_stacked=True))
    assert stacked.shape == (1, 24, 24, dom.n_basis)
    assert stacked.min() >= 0.0
    # No PSF, supersample 1, every ray-traced pixel centre inside the domain: the basis
    # functions partition unity on every detector pixel.
    np.testing.assert_allclose(stacked[0].sum(-1), 1.0, atol=1e-12)


def _independent_evidence(term, model, params, image, err, reg):
    """float64 numpy Suyu/WD03 evidence from the rendered basis stack."""
    X = np.asarray(term.simulator.lstsq_simulate(params, jnp.asarray(image), jnp.asarray(err),
                                                 return_stacked=True))[0]
    X = X.reshape(-1, X.shape[-1]) / err.reshape(-1)[:, None]
    y = image.reshape(-1) / err.reshape(-1)
    lam = float(model.component_params(params, 1, "light", 0)["lam"])
    H = lam * np.asarray(reg.L_reg)
    A = X.T @ X + H
    A = 0.5 * (A + A.T)
    A = A + 1e-6 * max(np.mean(np.diag(A)), 1.0) * np.eye(A.shape[0])
    s = np.linalg.solve(A, X.T @ y)
    r = y - X @ s
    chi2 = float(r @ r)
    sHs = float(s @ H @ s)
    logdetA = float(np.linalg.slogdet(A)[1])
    logdetH = float(np.linalg.slogdet(H)[1])
    norm = float(np.sum(np.log(2 * np.pi * err ** 2)))
    return dict(chi2=chi2, sHs=sHs, logdetA=logdetA, logdetH=logdetH,
                log_like=-0.5 * (chi2 + sHs + logdetA - logdetH + norm), coeffs=s)


def test_regularized_term_matches_independent_numpy_evidence():
    cfg, img, err, clean, truth, _ = _mock()
    dom = RegularGridDomain.centered(center=(0.0, 0.0), half_size=0.8, n=8)
    reg = GraphLaplacian(dom, kind="gradient", ridge_scale=1e-6)
    model, src = _mesh_model(dom, reg)
    data = RegularizedImageData(img, cfg, error_map=err, sees=[src])
    prob = ProbModel(model, data, mode="lstsq")
    term = prob.terms[0]
    z = jnp.asarray([np.log(3e-3)])  # lam's unconstrained coordinate (LogNormal -> log)
    params = model.constrained(z)
    got = {k: float(np.asarray(v).reshape(-1)[0]) for k, v in term.evidence_terms(params).items()
           if k != "coeffs"}
    ref = _independent_evidence(term, model, params, img, err, reg)
    for k in ("chi2", "sHs", "logdetA", "logdetH", "log_like"):
        np.testing.assert_allclose(got[k], ref[k], rtol=1e-9, err_msg=k)
    np.testing.assert_allclose(np.asarray(term.coefficients(params))[0], ref["coeffs"], rtol=1e-8)
    # and the ProbModel surface is wired to it
    ll, red = prob.log_like(z)
    np.testing.assert_allclose(float(ll), ref["log_like"], rtol=1e-9)
    np.testing.assert_allclose(float(red), ref["chi2"] / img.size, rtol=1e-9)
    # analytic log det H agrees with slogdet
    np.testing.assert_allclose(float(reg.logdet(lam=3e-3)), ref["logdetH"], rtol=1e-9)


def test_regularized_term_batched_matches_unbatched():
    cfg, img, err, *_ = _mock()
    dom = RegularGridDomain.centered(center=(0.0, 0.0), half_size=0.8, n=6)
    reg = GraphLaplacian(dom, kind="curvature", ridge_scale=1e-6)
    model, src = _mesh_model(dom, reg, free_theta_E=True)
    prob = ProbModel(model, RegularizedImageData(img, cfg, error_map=err, sees=[src]))
    zs = jnp.stack([_z(model, **{"0/mass/lens/theta_E": t, "1/light/mesh/lam": l})
                    for t, l in ((0.9, 1e-2), (1.0, 1e-1), (0.8, 1e-3))])
    ll_b, red_b = prob.log_like(zs)
    assert ll_b.shape == (3,)
    for k in range(3):
        ll, red = prob.log_like(zs[k])
        np.testing.assert_allclose(float(ll_b[k]), float(ll), rtol=1e-10)
        np.testing.assert_allclose(float(red_b[k]), float(red), rtol=1e-10)


def test_regularized_term_weak_prior_limit_is_plain_lstsq():
    cfg, img, err, *_ = _mock()
    dom = RegularGridDomain.centered(center=(0.0, 0.0), half_size=0.8, n=6)
    reg = GraphLaplacian(dom, kind="gradient", ridge_scale=1e-6)
    model, src = _mesh_model(dom, reg)
    prob = ProbModel(model, RegularizedImageData(img, cfg, error_map=err, sees=[src]))
    term = prob.terms[0]
    params = model.constrained(jnp.asarray([np.log(1e-14)]))
    chi2_reg = float(term.evidence_terms(params)["chi2"][0])
    im = np.asarray(SceneSimulator(model, cfg).lstsq_simulate(params, jnp.asarray(img), jnp.asarray(err)))
    chi2_plain = float(np.sum(((im - img) / err) ** 2))
    np.testing.assert_allclose(chi2_reg, chi2_plain, rtol=1e-6)
    # chi2 is non-decreasing in lam
    chi2s = [float(term.evidence_terms(model.constrained(jnp.asarray([np.log(l)])))["chi2"][0])
             for l in (1e-6, 1e-3, 1e0, 1e3)]
    assert all(b >= a * (1 - 1e-9) for a, b in zip(chi2s, chi2s[1:])), chi2s


def test_regularized_prob_model_gradients_and_evidence_optimal_lambda():
    """Mock at peak SNR ~100 (noise 0.2) on a 16x16 grid (spacing 0.107").

    Derived expectations (docs/logs/pixelized-source.md, 2026-09-12): the unregularized
    floor is chi2/nu = 0.88 (256 basis functions for 576 pixels over-fit), and the
    evidence-optimal lambda must bring it to 1 +- sqrt(2/576) = 0.06. At noise 0.05 the
    same grid floors at 3.08 -- bilinear interpolation bias on the n=1.5 Sersic cusp, not
    a regularizer property (floor 1.68 at n=24, 1.11 at n=32) -- so that configuration is
    NOT a valid test of the regularizer.
    """
    cfg, img, err, clean, truth, _ = _mock(noise_rms=0.2)
    dom = RegularGridDomain.centered(center=(0.0, 0.0), half_size=0.8, n=16)
    reg = GraphLaplacian(dom, kind="gradient", ridge_scale=1e-6)
    model, src = _mesh_model(dom, reg, free_theta_E=True)
    prob = ProbModel(model, RegularizedImageData(img, cfg, error_map=err, sees=[src]))
    z = _z(model, **{"0/mass/lens/theta_E": 0.9, "1/light/mesh/lam": 1e-2})
    lp, red = prob.log_prob(z)
    g = jax.grad(lambda zz: prob.log_prob(zz)[0])(z)
    assert np.isfinite(float(lp)) and np.all(np.isfinite(np.asarray(g)))
    assert float(g[1]) != 0.0  # lam is live in the likelihood

    params = model.constrained(z)
    lams = np.logspace(-6, 3, 10)
    rows = hyperparameter_scan(prob.terms[0], model, params, 1, 0, "lam", lams)
    ll = np.array([r["log_like"] for r in rows])
    chi2nu = np.array([r["chi2"] for r in rows]) / img.size
    assert np.all(np.isfinite(ll))
    best = int(np.argmax(ll))
    assert 0 < best < len(lams) - 1, f"evidence peaked at the grid edge: {ll}"
    assert chi2nu[0] < 0.95, f"unregularized floor should over-fit: {chi2nu[0]}"
    # At the evidence-optimal lambda the reconstruction fits the data (chi2/nu ~ 1 +- 0.06)
    # and the bright source vertices are not checkerboarded.
    assert 0.85 < chi2nu[best] < 1.2, rows[best]
    p_best = model.constrained(_z(model, **{"0/mass/lens/theta_E": 0.9, "1/light/mesh/lam": lams[best]}))
    s = np.asarray(prob.terms[0].component_coefficients(p_best, src))[0]
    assert alternating_pattern_score(s, dom.edges) > 0.0
    # And the rendered source peaks near the truth centre.
    src_img, ext = dom.render(s, half_size=0.8, npix=81)
    iy, ix = np.unravel_index(np.argmax(src_img), src_img.shape)
    xs = np.linspace(ext[0], ext[1], 81)
    assert abs(xs[ix] - SRC_TRUTH["center_x"]) < 0.12 and abs(xs[iy] - SRC_TRUTH["center_y"]) < 0.12


def test_regularized_data_refuses_when_nothing_is_regularized():
    cfg, img, err, *_ = _mock()
    src = Component(SersicEllipse(use_lstsq=True),
                    {k: v for k, v in SRC_TRUTH.items() if k != "Ie"}, name="sersic")
    model = LensModel([Plane(mass=[_mass_component(free_theta_E=True)]), Plane(light=[src])])
    with pytest.raises(ValueError, match="none of the seen light components"):
        ProbModel(model, RegularizedImageData(img, cfg, error_map=err, sees=[src]))
    with pytest.raises(ValueError, match="no meaning in mode"):
        ProbModel(model, RegularizedImageData(img, cfg, error_map=err, sees=[src]), mode="forward")


# ====================================================================== correlated field
def test_correlated_field_white_limit_is_exact_and_variance_is_normalized():
    dom = RegularGridDomain.centered(center=(0.0, 0.0), half_size=0.8, n=12)
    prof = CorrelatedFieldSource(dom)
    assert prof.params[:3] == ["mean_log", "log_amp", "slope"] and len(prof.params) == 3 + 144
    rng = np.random.default_rng(6)
    xi = jnp.asarray(rng.standard_normal(dom.n_basis))
    vals = np.asarray(prof.values(F(0.3), F(np.log(0.5)), F(0.0), xi))
    np.testing.assert_allclose(vals, np.exp(0.3 + 0.5 * np.asarray(xi)), rtol=1e-12)
    # slope > 0: unit-variance field on average over draws
    draws = jnp.asarray(rng.standard_normal((dom.n_basis, 200)))
    f = jax.vmap(lambda x: prof.field(x.reshape(dom.shape), F(1.5)), in_axes=1)(draws)
    assert abs(float(jnp.mean(f ** 2)) - 1.0) < 0.15
    # smoother than white: neighbour correlation is positive
    fa = np.asarray(f)
    corr = np.mean(fa[:, :, :-1] * fa[:, :, 1:]) / np.mean(fa ** 2)
    assert corr > 0.5


def test_correlated_field_forward_prob_model_is_finite_and_differentiable():
    cfg, img, err, *_ = _mock()
    dom = RegularGridDomain.centered(center=(0.0, 0.0), half_size=0.8, n=8)
    prof = CorrelatedFieldSource(dom)
    priors = {
        "mean_log": tfd.Normal(F(np.log(5.0)), F(1.0)),
        "log_amp": tfd.Normal(F(0.0), F(0.5)),
        "slope": tfd.LogNormal(F(np.log(2.0)), F(0.5)),
        **prof.xi_prior(),
    }
    src = Component(prof, priors, name="field")
    model = LensModel([Plane(mass=[_mass_component(free_theta_E=True)]), Plane(light=[src])])
    assert model.num_free_params == 1 + 3 + 64
    prob = ProbModel(model, ImageData(img, cfg, error_map=err, sees=[src]), mode="forward")
    z = _z(model, **{"0/mass/lens/theta_E": 0.9, "1/light/field/mean_log": np.log(5.0),
                     "1/light/field/log_amp": 0.0, "1/light/field/slope": 2.0})
    assert model.z_param_names[0].endswith("theta_E")
    lp, red = prob.log_prob(z)
    g = np.asarray(jax.grad(lambda zz: prob.log_prob(zz)[0])(z))
    assert np.isfinite(float(lp)) and np.all(np.isfinite(g))
    assert g[0] != 0.0 and np.count_nonzero(g[4:]) > 32  # mass and excitations are live
    im = np.asarray(SceneSimulator(model, cfg).simulate(model.constrained(z)))
    assert im.shape == (24, 24) and np.all(np.isfinite(im)) and im.min() >= 0.0
    # batched evaluation
    lp2, _ = prob.log_prob(jnp.stack([z, z + 0.1]))
    assert lp2.shape == (2,) and np.all(np.isfinite(np.asarray(lp2)))


# ====================================================================== pilot builder
def test_adaptive_delaunay_domain_from_pilot_model():
    cfg, img, err, clean, truth, params = _mock()
    dom = adaptive_delaunay_domain(
        model=truth, params=params, plane=1, sim_config=cfg, pilot_image=clean,
        n_vertices=40, region_half_size=1.15, weight_scheme="normalized_floor",
        weight_floor=0.05, seed=0, pad_fraction=0.2,
    )
    assert dom.n_basis == 40 + dom.n_padding and dom.n_padding >= 8
    assert np.all(np.abs(dom.image_seed_xy) <= 1.15)
    # Vertex density follows brightness: more seeds where the pilot image is bright.
    X = np.linspace(-1.15, 1.15, 24)
    bright = clean > 0.5 * clean.max()
    # Use it on the scene: basis partitions unity wherever the traced pixel lands inside.
    model, src = _mesh_model(dom)
    stacked = np.asarray(SceneSimulator(model, cfg).lstsq_simulate(
        model.to_params({}), jnp.asarray(img), jnp.asarray(err), return_stacked=True))[0]
    tot = stacked.sum(-1)
    assert stacked.min() >= -1e-6
    assert np.all((tot < 1e-9) | (np.abs(tot - 1.0) < 1e-9))
    assert (tot > 0.5).mean() > 0.3
    assert bright.any() and np.all(np.abs(tot[bright] - 1.0) < 1e-9)  # data region covered
