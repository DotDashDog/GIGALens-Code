from __future__ import annotations

import argparse
import json
import os
import sys
import time
from os.path import expanduser

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax as tfp
import yaml

HOME = expanduser("~/")
for p in (
    os.path.join(HOME, "gigalens", "src"),
    os.path.join(HOME, "GIGALens-Code"),
    os.path.join(HOME, "GIGALens-Code", "source_modeling"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

from gigalens.jax.physical_model import PhysicalModel
from gigalens.simulator import SimulatorConfig
from gigalens.jax.profiles.mass import epl, shear
from gigalens.jax.profiles.light import sersic

from voronoi_src.delaunay_mesh import (
    build_regular_imageplane_mesh,
    build_frozen_sourceplane_delaunay_from_truth,
)
from voronoi_src.pixelized_simulator import PixelizedSourceSimulator
from voronoi_src.pixelized_prob_model import PixelizedSourceProbModel
from voronoi_src.voronoi_diagnostics import Stage0Diagnostics, plot_stage0_panel, render_source_on_grid

tfd = tfp.distributions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--system-index", type=int, default=0)
    p.add_argument("--results-dir", type=str, default=os.path.join(HOME, "GIGALens-Code", "source_modeling", "voronoi_src", "results"))
    p.add_argument("--num-steps", type=int, default=500)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seed", type=int, default=25)
    p.add_argument("--extent", type=float, default=2.5)
    p.add_argument("--background-rms", type=float, default=0.2)
    p.add_argument("--exp-time", type=float, default=100.0)
    p.add_argument(
        "--lambda-init",
        type=float,
        default=1e-4,
        help="Initial regularization strength. The source basis normalization makes lambda~1 over-smooth.",
    )
    p.add_argument(
        "--lambda-prior-scale",
        type=float,
        default=3.0,
        help="LogNormal sigma for the regularization-strength prior.",
    )
    p.add_argument("--skip-checks", action="store_true")
    p.add_argument("--frozen-sourceplane", action="store_true",
                   help="Use truth-frozen source-plane Delaunay connectivity + "
                        "truth triangle assignment (stability-first).")
    p.add_argument(
        "--lambda-only",
        action="store_true",
        help="Freeze mass + lens light to truth; optimise only lambda.",
    )
    return p.parse_args()


def load_system(i: int):
    systems_dir = os.path.join(HOME, "GIGALens-Code", "SystemSaves")
    imgs = np.load(os.path.join(systems_dir, "100SystemsStandard80px.npz"))
    keys = imgs.files
    observed = jnp.array(imgs[keys[i]], dtype=jnp.float32)

    with open(os.path.join(systems_dir, "100SystemsStandardParams.yaml"), "r") as f:
        truth = yaml.safe_load(f)
    # truth structure is lists; convert to jnp arrays consistent with helpers.params_lists_to_jax
    from helpers import params_lists_to_jax, index_params

    true_params_all = params_lists_to_jax(truth)
    true_params = index_params(true_params_all, i)
    return observed, true_params


def build_prior(true_params, *, lambda_loc: float = 1e-4, lambda_scale: float = 3.0):
    # For Stage 0 debugging we prefer *soft* priors (no hard truncations),
    # because MAP can easily step outside truncated supports, producing -inf
    # log-probabilities and making the diagnostics useless.
    lens_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    theta_E=tfd.LogNormal(jnp.log(1.25), 0.4),
                    gamma=tfd.Normal(2.0, 0.5),
                    e1=tfd.Normal(0.0, 0.2),
                    e2=tfd.Normal(0.0, 0.2),
                    center_x=tfd.Normal(0.0, 0.06),
                    center_y=tfd.Normal(0.0, 0.06),
                )
            ),
            tfd.JointDistributionNamed(
                dict(
                    gamma1=tfd.Normal(0.0, 0.1),
                    gamma2=tfd.Normal(0.0, 0.1),
                )
            ),
        ]
    )

    def finite_centered_prior(profile_params):
        """A numerically finite 'fixed' prior centred on truth.

        `vela_utilities.fixed_prior` uses Uniform(val +/- 1e-6). For large
        parameters (notably lens-light Ie) that interval can collapse in
        float32, producing non-finite bijector inverses / Jacobians. A narrow
        Normal gives finite unconstrained coordinates, and the MAP code pins
        these coordinates when needed.
        """
        dists = {}
        for key, value in profile_params.items():
            val = jnp.asarray(jnp.squeeze(value), dtype=jnp.float32)
            scale = jnp.maximum(jnp.abs(val) * jnp.float32(1e-5), jnp.float32(1e-3))
            dists[key] = tfd.Normal(val, scale)
        return tfd.JointDistributionNamed(dists)

    lens_light_truth = true_params[1][0].copy()
    # We'll keep Ie fixed too by pinning all lens-light params.
    lens_light_prior = tfd.JointDistributionSequential([finite_centered_prior(lens_light_truth)])

    pixel_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    **{"lambda": tfd.LogNormal(jnp.log(jnp.float32(lambda_loc)), jnp.float32(lambda_scale))},
                )
            )
        ]
    )

    return tfd.JointDistributionSequential([lens_prior, lens_light_prior, pixel_prior])


def sanity_forward_consistency(
    *,
    observed: jnp.ndarray,
    true_params,
    sim_config: SimulatorConfig,
    background_rms: float,
    exp_time: float,
):
    """
    Compare chi^2 of the truth parametric simulation against the data.

    This is not expected to be exactly 1 because the 100Systems set can include
    noise realizations, but it should be sane and provides a baseline for the
    pixelised reconstruction.
    """
    from gigalens.jax.simulator import LensSimulator
    from gigalens.jax.prob_model import ForwardProbModel

    phys = PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [sersic.SersicEllipse(use_lstsq=False)],
        [sersic.SersicEllipse(use_lstsq=False)],
    )
    lens_sim = LensSimulator(phys, sim_config, bs=1)
    # Build a trivial prior just to instantiate the model; we will directly simulate.
    # This is used only for baseline chi^2.
    im_sim = lens_sim.simulate(true_params)
    err = jnp.sqrt(jnp.float32(background_rms) ** 2 + jnp.clip(im_sim, 0, np.inf) / jnp.float32(exp_time))
    chi2 = jnp.mean(((im_sim - observed) / err) ** 2)
    return float(jnp.squeeze(chi2))


def sanity_lambda_scan(model_seq, simulator, z0, n=12):
    # Scan lambda over decades around 1.0 in unconstrained space by editing physical space.
    prob = model_seq.prob_model
    x0 = prob.bij.forward(list(z0.T))
    lam_vals = jnp.logspace(-4, 4, n)
    lzs = []
    for lam in lam_vals:
        x = (x0[0], x0[1], [{"lambda": jnp.array(lam, dtype=jnp.float32)}])
        z = jnp.stack(prob.bij.inverse(x)).T
        lp, chi = prob.log_prob(simulator, z)
        lzs.append(lp[0] if lp.ndim else lp)
    return np.asarray(lam_vals), np.asarray(jax.device_get(jnp.array(lzs)))


def run_simple_map(
    *,
    prob_model: PixelizedSourceProbModel,
    simulator: PixelizedSourceSimulator,
    optimizer: optax.GradientTransformation,
    num_steps: int,
    n_samples: int,
    seed: int,
    start_z: jnp.ndarray,
):
    """
    Lightweight MAP optimizer for the experimental pixelised model.

    This avoids `gigalens.jax.inference.ModellingSequence.MAP`, which always
    constructs a `LensSimulator` internally. Here we must use
    `PixelizedSourceSimulator`.
    """
    key = jax.random.PRNGKey(seed)
    start_z = jnp.squeeze(start_z)
    z0 = jnp.tile(start_z[None, :], (n_samples, 1))
    noise = 1e-2 * jax.random.normal(key, shape=z0.shape, dtype=z0.dtype)
    z0 = z0 + noise

    def loss(z):
        lp, chisq = prob_model.log_prob(simulator, z)
        lp = jnp.where(jnp.isfinite(lp), lp, -jnp.inf)
        chisq = jnp.where(jnp.isfinite(chisq), chisq, jnp.inf)
        return -jnp.mean(lp) / jnp.size(prob_model.observed_image), (lp, chisq)

    loss_and_grad = jax.jit(jax.value_and_grad(loss, has_aux=True))
    opt_state = optimizer.init(z0)

    def one_step(carry, _):
        z, opt_state = carry
        (l, (lp, chisq)), g = loss_and_grad(z)
        z_eval = z
        updates, opt_state = optimizer.update(g, opt_state)
        z_candidate = optax.apply_updates(z, updates)
        # Pin lens-light parameters to their start_z values.
        # Parameter ordering for this prior is:
        #   [lens_mass (8), lens_light (7), lambda (1)] => total 16 dims.
        z_candidate = z_candidate.at[:, 8:15].set(start_z[8:15])
        # Reject non-finite parameter updates sample-wise.
        finite_update = jnp.all(jnp.isfinite(z_candidate), axis=-1, keepdims=True)
        z = jnp.where(finite_update, z_candidate, z)
        # best sample at this step
        best_idx = jnp.nanargmax(lp)
        best_z = z_eval[best_idx][None, :]
        best_lp = lp[best_idx]
        best_chi = chisq[best_idx]
        return (z, opt_state), (best_z, best_lp, best_chi)

    (_, _), hist = jax.lax.scan(one_step, (z0, opt_state), xs=None, length=num_steps)
    best_z_hist, best_lp_hist, best_chi_hist = hist
    # pick best step overall
    best_step = jnp.nanargmax(best_lp_hist)
    return (
        best_z_hist[best_step],
        best_lp_hist[best_step],
        best_chi_hist,
    )


def run_lambda_only_map(
    *,
    prob_model: PixelizedSourceProbModel,
    simulator: PixelizedSourceSimulator,
    optimizer: optax.GradientTransformation,
    num_steps: int,
    seed: int,
    start_z_full: jnp.ndarray,
):
    """
    Stage 0 stability check: optimise only z_lambda (last coordinate),
    holding all other z coordinates fixed.
    """
    key = jax.random.PRNGKey(seed)
    start_z_full = jnp.squeeze(start_z_full)
    zlam0 = start_z_full[-1]
    zlam = zlam0 + 0.1 * jax.random.normal(key, shape=())
    opt_state = optimizer.init(zlam)

    def loss(zlam_scalar):
        z_full = start_z_full.at[-1].set(zlam_scalar)
        lp, chisq = prob_model.log_prob(simulator, z_full[jnp.newaxis, :])
        lp = jnp.where(jnp.isfinite(lp), lp, -jnp.inf)
        chisq = jnp.where(jnp.isfinite(chisq), chisq, jnp.inf)
        # match inference.py convention: average over image pixels
        return -jnp.mean(lp) / jnp.size(prob_model.observed_image), (lp[0], chisq[0])

    loss_and_grad = jax.jit(jax.value_and_grad(loss, has_aux=True))

    def one_step(carry, _):
        zlam, opt_state = carry
        (l, (lp, chisq)), g = loss_and_grad(zlam)
        zlam_eval = zlam
        updates, opt_state = optimizer.update(g, opt_state)
        zlam_candidate = optax.apply_updates(zlam, updates)
        zlam = jnp.where(jnp.isfinite(zlam_candidate), zlam_candidate, zlam)
        return (zlam, opt_state), (zlam_eval, lp, chisq)

    (_, _), hist = jax.lax.scan(one_step, (zlam, opt_state), xs=None, length=num_steps)
    zlam_hist, lp_hist, chisq_hist = hist
    best_step = jnp.nanargmax(lp_hist)
    best_z_full = start_z_full.at[-1].set(zlam_hist[best_step])[jnp.newaxis, :]
    return best_z_full, lp_hist[best_step], chisq_hist

def main():
    args = parse_args()
    print(f"JAX devices: {jax.devices()}")
    print(f"local device count: {jax.local_device_count()}")

    observed, true_params = load_system(args.system_index)

    # Model: EPL + shear; lens light is SersicEllipse (fixed); source is pixelised.
    phys_model = PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [sersic.SersicEllipse(use_lstsq=False)],
        [],  # no parametric source light
    )
    # Simulator config matches hundredsystems.py
    kernel = np.load(os.path.join(HOME, "gigalens", "src", "gigalens", "assets", "psf.npy")).astype(np.float32)
    sim_config = SimulatorConfig(delta_pix=0.065, num_pix=80, supersample=2, kernel=kernel)

    if not args.skip_checks:
        baseline_chi2 = sanity_forward_consistency(
            observed=observed,
            true_params=true_params,
            sim_config=sim_config,
            background_rms=args.background_rms,
            exp_time=args.exp_time,
        )
        print(f"Baseline (truth parametric) mean chi^2: {baseline_chi2:.3f}")

    mesh = build_regular_imageplane_mesh(
        num_pix=sim_config.num_pix,
        delta_pix=sim_config.delta_pix,
        supersample=sim_config.supersample,
        n_seed_y=args.n_seed,
        n_seed_x=args.n_seed,
        extent=args.extent,
    )
    if args.frozen_sourceplane:
        mesh = build_frozen_sourceplane_delaunay_from_truth(
            lenses=phys_model.lenses,
            lens_params_truth=true_params[0],
            num_pix=sim_config.num_pix,
            delta_pix=sim_config.delta_pix,
            supersample=sim_config.supersample,
            n_seed_y=args.n_seed,
            n_seed_x=args.n_seed,
            extent=args.extent,
        )
    print(f"Mesh: I={mesh.seed_xy.shape[0]} vertices, T={mesh.simplices.shape[0]} triangles, E={mesh.edges.shape[0]} edges")

    prior = build_prior(
        true_params,
        lambda_loc=args.lambda_init,
        lambda_scale=args.lambda_prior_scale,
    )

    # Simulator and prob model
    simulator = PixelizedSourceSimulator(
        lenses=phys_model.lenses,
        lens_light_profiles=phys_model.lens_light,
        sim_config=sim_config,
        mesh=mesh,
    )
    prob_model = PixelizedSourceProbModel(
        prior,
        observed_image=observed,
        background_rms=args.background_rms,
        exp_time=args.exp_time,
        edges=mesh.edges,
    )
    # Build truth-ish z0: use truth lens_light params, lens mass from truth,
    # and initialize lambda on the empirically appropriate source-basis scale.
    truth_mass = true_params[0]
    truth_light = [true_params[1][0]]
    x0 = (truth_mass, truth_light, [{"lambda": jnp.array(args.lambda_init, dtype=jnp.float32)}])
    z0 = jnp.stack(prob_model.bij.inverse(x0)).T  # (1, n_params)
    truth_terms = prob_model.debug_terms(simulator, jnp.squeeze(z0))
    print(
        "Truth diagnostic: "
        f"z_finite={bool(~truth_terms['z_any_nonfinite'])}, "
        f"logZ={float(jnp.squeeze(truth_terms['logZ'])):.3e}, "
        f"prior_lp={float(jnp.squeeze(truth_terms['prior_lp'])):.3e}, "
        f"fldj={float(jnp.squeeze(truth_terms['fldj'])):.3e}, "
        f"chi2_mean={float(jnp.squeeze(truth_terms['chi2_mean'])):.3f}"
    )

    if not args.skip_checks:
        lam_vals, lzs = sanity_lambda_scan(type("W", (), {"prob_model": prob_model})(), simulator, z0)
        print("Lambda scan (log10 lambda, log posterior target):")
        for lv, lz in zip(lam_vals, lzs):
            print(f"  {np.log10(lv): .2f}: {float(lz): .3e}")

    # MAP
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adabelief(1e-3, b1=0.95, b2=0.99, nesterov=True),
    )
    t0 = time.perf_counter()
    if args.lambda_only:
        best_z, best_lp, best_chi_hist = run_lambda_only_map(
            prob_model=prob_model,
            simulator=simulator,
            optimizer=optimizer,
            num_steps=args.num_steps,
            seed=args.seed,
            start_z_full=jnp.squeeze(z0),
        )
    else:
        best_z, best_lp, best_chi_hist = run_simple_map(
            prob_model=prob_model,
            simulator=simulator,
            optimizer=optimizer,
            num_steps=args.num_steps,
            n_samples=args.n_samples,
            seed=args.seed,
            start_z=z0,
        )
    jax.block_until_ready(best_z)
    dt = time.perf_counter() - t0
    print(f"MAP wall time: {dt:.1f} s")

    best_x = prob_model.bij.forward(list(best_z.T))
    out = simulator.basis_and_lens_light((best_x[0], best_x[1]))
    basis = out.basis_images
    lens_light = out.lens_light
    resid = (observed - lens_light).astype(jnp.float32)
    D = jnp.einsum("ihw,hw->i", basis, resid * prob_model.inv_var)
    F = jnp.einsum("ihw,khw,hw->ik", basis, basis, prob_model.inv_var)
    lam = best_x[2][0]["lambda"].astype(jnp.float32)
    from voronoi_src.pixelized_regularization import regularization_matrix_constant_gradient
    H = regularization_matrix_constant_gradient(num_vertices=basis.shape[0], edges=jnp.array(mesh.edges), lam=lam)
    A = F + H
    chol = jnp.linalg.cholesky(A)
    y = jax.lax.linalg.triangular_solve(chol, D, left_side=True, lower=True)
    s = jax.lax.linalg.triangular_solve(chol.T, y, left_side=True, lower=False)

    src_img = jnp.einsum("i,ihw->hw", s, basis)
    model = lens_light + src_img
    lens_only_norm_resid = (observed - lens_light) / jnp.sqrt(1.0 / prob_model.inv_var)
    norm_resid = (observed - model) / jnp.sqrt(1.0 / prob_model.inv_var)
    basis_coverage = jnp.sum(jnp.abs(basis) > 0, axis=0)

    # For a quick source-plane visualization, ray-trace seed points and nearest-neighbour render.
    seed_xy = jnp.array(mesh.seed_xy)
    seed_bx, seed_by = simulator._beta(best_x[0], seed_xy[:, 0], seed_xy[:, 1])
    src_plane, src_extent = render_source_on_grid(
        source_values=s,
        seed_xy_src=jnp.stack([seed_bx, seed_by], axis=-1),
    )

    # Save outputs
    out_dir = os.path.join(args.results_dir, f"system{args.system_index:02d}")
    os.makedirs(out_dir, exist_ok=True)

    # MAP loss history is in best_chi (shape depends on ModellingSequence); keep a simple best-step series if available
    # If not available, plot a constant.
    map_loss = jax.device_get(jnp.atleast_1d(best_chi_hist).reshape(-1))
    image_half_extent = 0.5 * sim_config.num_pix * sim_config.delta_pix
    image_extent = (
        -image_half_extent,
        image_half_extent,
        -image_half_extent,
        image_half_extent,
    )

    diag = Stage0Diagnostics(
        data=np.array(jax.device_get(observed)),
        model=np.array(jax.device_get(model)),
        norm_resid=np.array(jax.device_get(norm_resid)),
        source_plane=src_plane,
        map_loss=np.array(map_loss),
        title=f"Stage0 system {args.system_index:02d} | lambda={float(jnp.squeeze(lam)):.3g} | deg_subpix={int(jax.device_get(out.degenerate_subpix))}",
        image_extent=image_extent,
        source_extent=src_extent,
        lens_light=np.array(jax.device_get(lens_light)),
        lensed_source=np.array(jax.device_get(src_img)),
        basis_coverage=np.array(jax.device_get(basis_coverage)),
    )
    fig_path = os.path.join(out_dir, "stage0_panel.png")
    plot_stage0_panel(diag, save_path=fig_path)

    summary = {
        "system_index": args.system_index,
        "map_seconds": dt,
        "degenerate_subpix": int(jax.device_get(out.degenerate_subpix)),
        "lambda": float(jnp.squeeze(lam)),
        "lambda_init": float(args.lambda_init),
        "lambda_prior_scale": float(args.lambda_prior_scale),
        "chi2_mean": float(jnp.mean((norm_resid) ** 2)),
        "lens_only_chi2_mean": float(jnp.mean(lens_only_norm_resid**2)),
        "lensed_source_min": float(jnp.min(src_img)),
        "lensed_source_max": float(jnp.max(src_img)),
        "lensed_source_sum": float(jnp.sum(src_img)),
        "lens_light_max": float(jnp.max(lens_light)),
        "data_max": float(jnp.max(observed)),
        "source_coeff_min": float(jnp.min(s)),
        "source_coeff_max": float(jnp.max(s)),
        "source_coeff_negative_fraction": float(jnp.mean(s < 0)),
        "basis_coverage_min": float(jnp.min(basis_coverage)),
        "basis_coverage_max": float(jnp.max(basis_coverage)),
        "best_log_target": float(jnp.squeeze(best_lp)),
    }
    summary.update({
        f"truth_{k}": float(jnp.squeeze(v))
        for k, v in truth_terms.items()
        if k not in {"degenerate_subpix"}
    })
    summary["truth_degenerate_subpix"] = int(jax.device_get(truth_terms["degenerate_subpix"]))
    # Debug terms (helps diagnose non-finite targets).
    terms = prob_model.debug_terms(simulator, jnp.squeeze(best_z))
    summary.update({k: float(jnp.squeeze(v)) for k, v in terms.items() if k not in {"degenerate_subpix"}})
    summary["degenerate_subpix"] = int(jax.device_get(terms["degenerate_subpix"]))
    with open(os.path.join(out_dir, "stage0_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {fig_path}")
    print(f"Saved: {os.path.join(out_dir, 'stage0_summary.json')}")


if __name__ == "__main__":
    main()

