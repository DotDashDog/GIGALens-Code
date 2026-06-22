from __future__ import annotations

import argparse
import json
import os
import sys
from os.path import expanduser

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

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

from gigalens.jax.simulator import LensSimulator

from voronoi_src.delaunay_mesh import (
    build_brightness_adaptive_sourceplane_delaunay_from_truth,
    build_frozen_sourceplane_delaunay_from_truth,
)
from voronoi_src.diagnostics.quality_metrics import (
    alternating_pattern_score,
    lambda_evidence_scan,
    pick_evidence_optimal_lambda,
)
from voronoi_src.linear_inversion import solve_source_positive, solve_source_unconstrained
from voronoi_src.pixelized_prob_model import PixelizedSourceProbModel
from voronoi_src.pixelized_simulator import PixelizedSourceSimulator
from voronoi_src.tests.test_stage0 import build_prior, load_system
from voronoi_src.voronoi_diagnostics import render_source_on_grid


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--system-index", type=int, default=0)
    p.add_argument("--n-seed", type=int, default=30)
    p.add_argument("--extent", type=float, default=2.5)
    p.add_argument("--background-rms", type=float, default=0.2)
    p.add_argument("--exp-time", type=float, default=100.0)
    p.add_argument(
        "--lambda-values",
        type=float,
        nargs="+",
        default=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join(
            HOME,
            "GIGALens-Code",
            "source_modeling",
            "voronoi_src",
            "truth_pinned_lambda_scan",
        ),
    )
    p.add_argument(
        "--positive",
        action="store_true",
        help="Solve the regularized linear inversion with non-negative source coefficients.",
    )
    p.add_argument(
        "--adaptive-mesh",
        action="store_true",
        help="Use brightness-adaptive weighted-KMeans image-plane centres traced to the source plane.",
    )
    p.add_argument(
        "--weight-floor",
        type=float,
        default=0.01,
        help="Brightness-adaptive KMeans floor weight, analogous to W_floor in the paper.",
    )
    p.add_argument(
        "--adaptive-weight-scheme",
        type=str,
        default="paper_eq12",
        choices=("paper_eq12", "pyautoarray_current", "normalized_floor"),
        help="Weight map used for the adaptive KMeans image mesh.",
    )
    p.add_argument(
        "--source-display-npix",
        type=int,
        default=120,
        help="Resolution of the rendered source-plane diagnostic only; does not affect the inversion.",
    )
    return p.parse_args()


def _log_norm(arr):
    finite = np.asarray(arr)[np.isfinite(arr)]
    positive = finite[finite > 0]
    if positive.size == 0:
        return None
    vmin = max(np.percentile(positive, 1), np.min(positive[positive > 0]))
    vmax = np.percentile(positive, 99.8)
    if vmax <= vmin:
        vmax = positive.max()
    return LogNorm(vmin=vmin, vmax=vmax) if vmax > vmin else None


def _weighted_centroid(img, extent):
    arr = np.asarray(img, dtype=np.float64)
    weights = np.clip(arr, 0.0, None)
    total = np.sum(weights)
    if not np.isfinite(total) or total <= 0:
        return (float("nan"), float("nan"))
    x0, x1, y0, y1 = extent
    yy = np.linspace(y0, y1, arr.shape[0])
    xx = np.linspace(x0, x1, arr.shape[1])
    X, Y = np.meshgrid(xx, yy, indexing="xy")
    return (float(np.sum(X * weights) / total), float(np.sum(Y * weights) / total))


def _peak_xy(img, extent):
    arr = np.asarray(img)
    iy, ix = np.unravel_index(np.nanargmax(arr), arr.shape)
    x0, x1, y0, y1 = extent
    x = np.linspace(x0, x1, arr.shape[1])[ix]
    y = np.linspace(y0, y1, arr.shape[0])[iy]
    return (float(x), float(y))


def _render_true_source_on_grid(*, source_profile, source_params, source_extent, out_npix):
    x0, x1, y0, y1 = source_extent
    yy = jnp.linspace(y0, y1, out_npix)
    xx = jnp.linspace(x0, x1, out_npix)
    X, Y = jnp.meshgrid(xx, yy, indexing="xy")
    img = source_profile.light(X, Y, **source_params)
    return np.array(jax.device_get(img))


def _solve_positive_source(*, basis, resid_no_source, inv_var, edges, lam):
    """Solve weighted regularized least squares with source coefficients >= 0."""
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

    sqrt_lam = np.sqrt(float(lam))
    lhs_reg = np.zeros((edges_np.shape[0], n_vertices), dtype=np.float64)
    lhs_reg[np.arange(edges_np.shape[0]), edges_np[:, 0]] = sqrt_lam
    lhs_reg[np.arange(edges_np.shape[0]), edges_np[:, 1]] = -sqrt_lam
    rhs_reg = np.zeros(edges_np.shape[0], dtype=np.float64)

    # Match the small null-space ridge used by the JAX regularization matrix.
    degree = np.bincount(edges_np.reshape(-1), minlength=n_vertices).astype(np.float64)
    ridge = 1e-8 * max(float(lam) * float(np.mean(degree)), 1.0)
    lhs_ridge = np.sqrt(ridge) * np.eye(n_vertices, dtype=np.float64)
    rhs_ridge = np.zeros(n_vertices, dtype=np.float64)

    lhs = np.vstack([lhs_img, lhs_reg, lhs_ridge])
    rhs = np.concatenate([rhs_img, rhs_reg, rhs_ridge])
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


def _save_panel(*, observed, rows, image_extent, out_path):
    nrows = len(rows)
    fig, axs = plt.subplots(nrows, 5, figsize=(20, 3.8 * nrows), squeeze=False)

    for ir, row in enumerate(rows):
        lam = row["lambda"]
        chi2_mean = row["chi2_mean"]

        im0 = axs[ir, 0].imshow(
            observed,
            origin="lower",
            cmap="magma",
            norm=_log_norm(observed),
            extent=image_extent,
        )
        axs[ir, 0].set_title("Data")
        plt.colorbar(im0, ax=axs[ir, 0], fraction=0.046, pad=0.04)

        im1 = axs[ir, 1].imshow(
            row["model"],
            origin="lower",
            cmap="magma",
            norm=_log_norm(row["model"]),
            extent=image_extent,
        )
        axs[ir, 1].set_title(f"Model | lambda={lam:.0e}")
        plt.colorbar(im1, ax=axs[ir, 1], fraction=0.046, pad=0.04)

        resid_abs = float(np.nanmax(np.abs(row["norm_resid"])))
        resid_lim = max(1.0, min(resid_abs, 8.0))
        im2 = axs[ir, 2].imshow(
            row["norm_resid"],
            origin="lower",
            cmap="coolwarm",
            vmin=-resid_lim,
            vmax=resid_lim,
            extent=image_extent,
        )
        axs[ir, 2].set_title(f"Norm residual | chi2={chi2_mean:.3f}")
        plt.colorbar(im2, ax=axs[ir, 2], fraction=0.046, pad=0.04)

        im3 = axs[ir, 3].imshow(
            row["source_plane"],
            origin="lower",
            cmap="magma",
            extent=row["source_extent"],
        )
        axs[ir, 3].set_title("Reconstructed source")
        plt.colorbar(im3, ax=axs[ir, 3], fraction=0.046, pad=0.04)

        im4 = axs[ir, 4].imshow(
            row["true_source_plane"],
            origin="lower",
            cmap="magma",
            extent=row["source_extent"],
        )
        axs[ir, 4].set_title("True Sersic source")
        plt.colorbar(im4, ax=axs[ir, 4], fraction=0.046, pad=0.04)

        for ax in axs[ir, :3]:
            ax.set_xlabel("x [arcsec]")
            ax.set_ylabel("y [arcsec]")
        for ax in axs[ir, 3:]:
            ax.set_xlabel("source x [arcsec]")
            ax.set_ylabel("source y [arcsec]")

    fig.suptitle("Truth-pinned source linear inversion lambda scan", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    args = parse_args()
    print(f"JAX devices: {jax.devices()}")

    observed, true_params = load_system(args.system_index)
    phys_model = PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [sersic.SersicEllipse(use_lstsq=False)],
        [],
    )
    kernel = np.load(
        os.path.join(HOME, "gigalens", "src", "gigalens", "assets", "psf.npy")
    ).astype(np.float32)
    sim_config = SimulatorConfig(delta_pix=0.065, num_pix=80, supersample=2, kernel=kernel)

    lens_only_sim = LensSimulator(phys_model, sim_config, bs=1)
    lens_only_image = lens_only_sim.simulate({
        "lens_mass": {str(i): p for i, p in enumerate(true_params[0])},
        "lens_light": {"0": true_params[1][0]},
        "source_light": {},
    })
    source_only_phys_model = PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [],
        [sersic.SersicEllipse(use_lstsq=False)],
    )
    source_only_sim = LensSimulator(source_only_phys_model, sim_config, bs=1)
    lensed_source_for_mesh = source_only_sim.simulate({
        "lens_mass": {str(i): p for i, p in enumerate(true_params[0])},
        "lens_light": {},
        "source_light": {"0": true_params[2][0]},
    })

    if args.adaptive_mesh:
        mesh = build_brightness_adaptive_sourceplane_delaunay_from_truth(
            lenses=phys_model.lenses,
            lens_params_truth=true_params[0],
            lensed_source_image=np.array(jax.device_get(lensed_source_for_mesh)),
            num_pix=sim_config.num_pix,
            delta_pix=sim_config.delta_pix,
            supersample=sim_config.supersample,
            n_source_pixels=args.n_seed * args.n_seed,
            extent=args.extent,
            weight_floor=args.weight_floor,
            weight_scheme=args.adaptive_weight_scheme,
            seed=0,
        )
    else:
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
    print(
        f"Mesh: I={mesh.seed_xy.shape[0]} vertices, "
        f"T={mesh.simplices.shape[0]} triangles, E={mesh.edges.shape[0]} edges"
    )

    simulator = PixelizedSourceSimulator(
        lenses=phys_model.lenses,
        lens_light_profiles=phys_model.lens_light,
        sim_config=sim_config,
        mesh=mesh,
    )
    prior = build_prior(true_params, lambda_loc=1e-4, lambda_scale=3.0)
    prob_model = PixelizedSourceProbModel(
        prior,
        observed_image=observed,
        background_rms=args.background_rms,
        exp_time=args.exp_time,
        edges=mesh.edges,
    )

    out = simulator.basis_and_lens_light((true_params[0], [true_params[1][0]]))
    basis = out.basis_images
    lens_light = out.lens_light
    resid_no_source = (observed - lens_light).astype(jnp.float32)
    x_truth_mass = (true_params[0], [true_params[1][0]])
    evidence_scan = lambda_evidence_scan(
        prob_model=prob_model,
        simulator=simulator,
        x_truth_mass=x_truth_mass,
        lambda_values=args.lambda_values,
    )

    seed_xy = jnp.array(mesh.seed_xy)
    seed_bx, seed_by = simulator._beta(true_params[0], seed_xy[:, 0], seed_xy[:, 1])
    seed_xy_src = jnp.stack([seed_bx, seed_by], axis=-1)
    source_profile = sersic.SersicEllipse(use_lstsq=False)

    rows = []
    summary_rows = []
    for lam_float in args.lambda_values:
        lam = jnp.array(lam_float, dtype=jnp.float32)
        positive_solver = None
        if args.positive:
            s, positive_solver = solve_source_positive(
                basis=basis,
                resid_no_source=resid_no_source,
                inv_var=prob_model.inv_var,
                edges=mesh.edges,
                lam=lam_float,
            )
        else:
            s = solve_source_unconstrained(
                basis=basis,
                resid_no_source=resid_no_source,
                inv_var=prob_model.inv_var,
                edges=mesh.edges,
                lam=lam_float,
            )

        src_img = jnp.einsum("i,ihw->hw", s, basis, precision="highest")
        model = lens_light + src_img
        norm_resid = (observed - model) / jnp.sqrt(1.0 / prob_model.inv_var)
        chi2_mean = jnp.mean(norm_resid**2)
        src_plane, src_extent = render_source_on_grid(
            source_values=s,
            seed_xy_src=seed_xy_src,
            out_npix=args.source_display_npix,
        )
        true_src_plane = _render_true_source_on_grid(
            source_profile=source_profile,
            source_params=true_params[2][0],
            source_extent=src_extent,
            out_npix=src_plane.shape[0],
        )
        recon_centroid = _weighted_centroid(src_plane, src_extent)
        true_centroid = _weighted_centroid(true_src_plane, src_extent)
        recon_peak = _peak_xy(src_plane, src_extent)
        true_peak = _peak_xy(true_src_plane, src_extent)

        row = {
            "lambda": float(lam_float),
            "chi2_mean": float(chi2_mean),
            "model": np.array(jax.device_get(model)),
            "norm_resid": np.array(jax.device_get(norm_resid)),
            "source_plane": src_plane,
            "true_source_plane": true_src_plane,
            "source_extent": src_extent,
        }
        rows.append(row)
        summary_rows.append(
            {
                "lambda": float(lam_float),
                "chi2_mean": float(chi2_mean),
                "alternating_pattern_score": alternating_pattern_score(
                    np.asarray(jax.device_get(s)), mesh.edges
                ),
                "source_coeff_min": float(jnp.min(s)),
                "source_coeff_max": float(jnp.max(s)),
                "source_coeff_std": float(jnp.std(s)),
                "source_coeff_negative_fraction": float(jnp.mean(s < 0)),
                "lensed_source_min": float(jnp.min(src_img)),
                "lensed_source_max": float(jnp.max(src_img)),
                "lensed_source_std": float(jnp.std(src_img)),
                "reconstructed_source_positive_centroid_x": recon_centroid[0],
                "reconstructed_source_positive_centroid_y": recon_centroid[1],
                "true_source_centroid_x": true_centroid[0],
                "true_source_centroid_y": true_centroid[1],
                "reconstructed_source_peak_x": recon_peak[0],
                "reconstructed_source_peak_y": recon_peak[1],
                "true_source_peak_x": true_peak[0],
                "true_source_peak_y": true_peak[1],
                "positive_solver_success": (
                    bool(positive_solver.success) if positive_solver is not None else None
                ),
                "positive_solver_cost": (
                    float(positive_solver.cost) if positive_solver is not None else None
                ),
                "positive_solver_optimality": (
                    float(positive_solver.optimality) if positive_solver is not None else None
                ),
                "positive_solver_nit": int(positive_solver.nit) if positive_solver is not None else None,
            }
        )
        print(
            f"lambda={lam_float:.1e} chi2_mean={float(chi2_mean):.4f} "
            f"s_std={float(jnp.std(s)):.3f} src_max={float(jnp.max(src_img)):.3f}"
        )

    suffix = ""
    if args.adaptive_mesh:
        suffix += "_adaptive"
        suffix += f"_{args.adaptive_weight_scheme}"
        suffix += f"_wf{args.weight_floor:g}".replace(".", "p")
    if args.positive:
        suffix += "_positive"
    out_dir = os.path.join(args.results_dir, f"system{args.system_index:02d}_nseed{args.n_seed}{suffix}")
    os.makedirs(out_dir, exist_ok=True)
    image_half_extent = 0.5 * sim_config.num_pix * sim_config.delta_pix
    image_extent = (-image_half_extent, image_half_extent, -image_half_extent, image_half_extent)

    panel_path = os.path.join(out_dir, "truth_pinned_lambda_scan.png")
    _save_panel(
        observed=np.array(jax.device_get(observed)),
        rows=rows,
        image_extent=image_extent,
        out_path=panel_path,
    )
    summary = {
        "system_index": args.system_index,
        "n_seed": args.n_seed,
        "mesh_vertices": int(mesh.seed_xy.shape[0]),
        "mesh_triangles": int(mesh.simplices.shape[0]),
        "mesh_edges": int(mesh.edges.shape[0]),
        "degenerate_subpix": int(jax.device_get(out.degenerate_subpix)),
        "lens_only_chi2_mean": float(jnp.mean((resid_no_source**2) * prob_model.inv_var)),
        "positive": bool(args.positive),
        "adaptive_mesh": bool(args.adaptive_mesh),
        "weight_floor": float(args.weight_floor),
        "adaptive_weight_scheme": args.adaptive_weight_scheme,
        "adaptive_image": "truth_lensed_source_model" if args.adaptive_mesh else None,
        "lambda_scan": summary_rows,
        "evidence_scan": evidence_scan,
        "evidence_optimal": pick_evidence_optimal_lambda(evidence_scan),
    }
    summary_path = os.path.join(out_dir, "truth_pinned_lambda_scan.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved: {panel_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
