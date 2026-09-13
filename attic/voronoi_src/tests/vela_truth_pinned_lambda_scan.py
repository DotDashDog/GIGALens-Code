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

from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear
from gigalens.jax.simulator import _shared_kernel_component_conv
from gigalens.jax.physical_model import PhysicalModel
from objax.functional import average_pool_2d

from vela_utilities import (
    DEFAULT_BACKGROUND_RMS,
    DEFAULT_CAM,
    DEFAULT_EXP_TIME,
    DEFAULT_FILTER_TAG,
    DEFAULT_NUM_PIX,
    DEFAULT_SUPERSAMPLE,
    load_vela_sim_system,
    run_save_dir,
    source_plane_dir,
    system_save_dir,
    vela_system_model,
)
from voronoi_src.delaunay_mesh import (
    build_brightness_adaptive_imageplane_delaunay_from_truth,
    build_brightness_adaptive_sourceplane_delaunay_from_truth,
)
from voronoi_src.diagnostics.caustic_metrics import compute_degeneracy_stats, plot_caustic_vertex_overlay
from voronoi_src.diagnostics.quality_metrics import (
    alternating_pattern_score,
    pick_evidence_optimal_lambda,
    plot_vertex_density_maps,
)
from voronoi_src.linear_inversion import (
    solve_source_positive,
    solve_source_unconstrained,
    vertex_lam_adaptive_split,
)
from voronoi_src.pixelized_prob_model import PixelizedSourceProbModel
from voronoi_src.pixelized_simulator import PixelizedSourceSimulator
from voronoi_src.tests.test_stage0 import build_prior
from voronoi_src.tests.truth_pinned_lambda_scan import _peak_xy, _weighted_centroid
from voronoi_src.voronoi_diagnostics import render_source_on_grid

REG_VARIANT_CHOICES = (
    "constant_gradient",
    "distance_weighted_gradient",
    "curvature",
    "adaptive_split",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sim-num", type=str, default="04")
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--cam", type=str, default=DEFAULT_CAM)
    p.add_argument("--filter-tag", type=str, default=DEFAULT_FILTER_TAG)
    p.add_argument("--n-seed", type=int, default=30)
    p.add_argument("--extent", type=float, default=3.0)
    p.add_argument("--num-pix", type=int, default=DEFAULT_NUM_PIX)
    p.add_argument("--supersample", type=int, default=DEFAULT_SUPERSAMPLE)
    p.add_argument("--background-rms", type=float, default=DEFAULT_BACKGROUND_RMS)
    p.add_argument("--exp-time", type=float, default=DEFAULT_EXP_TIME)
    p.add_argument("--weight-floor", type=float, default=0.01)
    p.add_argument(
        "--adaptive-target",
        type=str,
        default="mclmc_shapelets",
        choices=("lens_img", "mclmc_shapelets"),
    )
    p.add_argument("--adaptive-n-max", type=int, default=15)
    p.add_argument(
        "--adaptive-weight-scheme",
        type=str,
        default="normalized_floor",
        choices=("paper_eq12", "pyautoarray_current", "normalized_floor", "brightness_times_invmag"),
    )
    p.add_argument(
        "--mesh-connectivity",
        type=str,
        default="sourceplane",
        choices=("sourceplane", "imageplane"),
        help="sourceplane: Delaunay in source plane (truth-pinned). imageplane: mass-differentiable.",
    )
    p.add_argument(
        "--reg-variants",
        type=str,
        nargs="+",
        default=["constant_gradient"],
        choices=REG_VARIANT_CHOICES,
    )
    p.add_argument("--source-display-npix", type=int, default=360)
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
            "truth_pinned_lambda_scan_vela",
        ),
    )
    p.add_argument("--no-positive", action="store_true")
    p.add_argument(
        "--alternating-threshold",
        type=float,
        default=-0.01,
        help="Alternating-pattern score must exceed this value (less negative is better).",
    )
    p.add_argument("--skip-density-maps", action="store_true")
    p.add_argument("--skip-caustic-panel", action="store_true")
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


def _source_image_cps_per_arcsec2(source_image_njy, meta):
    source_pixel_scale = float(meta["source_pixel_scale_arcsec"])
    sb_njy_per_arcsec2 = source_image_njy / (source_pixel_scale**2)
    return sb_njy_per_arcsec2 * 1e-9 / float(meta["photfnu_Jy"])


def _source_image_on_extent(source_image, *, source_center, source_pixel_scale, extent, out_npix):
    from scipy.interpolate import RegularGridInterpolator

    x0, x1, y0, y1 = extent
    cx, cy = source_center
    native_half = 0.5 * (source_image.shape[0] - 1) * float(source_pixel_scale)
    native_x = np.linspace(cx - native_half, cx + native_half, source_image.shape[0])
    native_y = np.linspace(cy - native_half, cy + native_half, source_image.shape[1])
    out_x = np.linspace(x0, x1, out_npix)
    out_y = np.linspace(y0, y1, out_npix)
    xx, yy = np.meshgrid(out_x, out_y, indexing="xy")
    points = np.stack([xx.ravel(), yy.ravel()], axis=-1)
    interpolator = RegularGridInterpolator(
        (native_x, native_y),
        source_image,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    return interpolator(points).reshape(out_npix, out_npix)


def _lstsq_source_component_image(lens_sim, prob_model, params):
    lens_params = params[0]
    lens_light_params = params[1]
    source_light_params = params[2]
    beta_x, beta_y = lens_sim._beta(lens_params)

    components = jnp.zeros((0, *lens_sim.img_X.shape))
    for light_model, p in zip(lens_sim.phys_model.lens_light, lens_light_params):
        components = jnp.concatenate(
            (components, light_model.light(lens_sim.img_X, lens_sim.img_Y, **p)),
            axis=0,
        )
    n_lens_light_components = components.shape[0]
    for light_model, p in zip(lens_sim.phys_model.source_light, source_light_params):
        components = jnp.concatenate(
            (components, light_model.light(beta_x, beta_y, **p)),
            axis=0,
        )

    components = jnp.nan_to_num(components)
    components = jnp.transpose(components, (3, 0, 1, 2))
    if lens_sim.flat_kernel is not None:
        components = _shared_kernel_component_conv(components, lens_sim.flat_kernel)
    if lens_sim.supersample != 1:
        components = average_pool_2d(
            components,
            size=(lens_sim.supersample, lens_sim.supersample),
            padding="SAME",
        )
    components = jnp.transpose(components, (0, 2, 3, 1))

    _, coeffs = lens_sim.lstsq_simulate(
        params,
        prob_model.observed_image,
        prob_model.err_map,
    )
    source_components = components[..., n_lens_light_components:]
    source_coeffs = jnp.atleast_1d(coeffs)[n_lens_light_components:]
    return jnp.squeeze(jnp.sum(source_components * source_coeffs, axis=-1))


def _adaptive_image_from_mclmc_shapelets(*, sim_config, observed, system_dir, n_max, background_rms, exp_time):
    run_dir = run_save_dir(system_dir, n_max)
    samples_path = os.path.join(run_dir, "mclmc_samples.npy")
    if not os.path.exists(samples_path):
        raise FileNotFoundError(f"Missing MCLMC samples: {samples_path}")

    samples = jnp.asarray(np.load(samples_path))
    if not bool(jnp.all(jnp.isfinite(samples))):
        raise ValueError(f"MCLMC samples contain non-finite values: {samples_path}")

    prob_model, lens_sim = vela_system_model(
        sim_config,
        observed,
        background_rms=background_rms,
        exp_time=exp_time,
        use_shapelets=True,
        n_max=n_max,
    )
    median_z = jnp.median(samples, axis=(0, 1))
    median_params = prob_model.bij.forward(list(median_z.T))
    adaptive_image = _lstsq_source_component_image(lens_sim, prob_model, median_params)
    return np.array(jax.device_get(adaptive_image), dtype=np.float32), samples_path


def _build_mesh(args, phys_model, true_params, adaptive_image, sim_config):
    common = dict(
        lenses=phys_model.lenses,
        lens_params_truth=true_params[0],
        lensed_source_image=adaptive_image,
        num_pix=sim_config.num_pix,
        delta_pix=sim_config.delta_pix,
        supersample=sim_config.supersample,
        n_source_pixels=args.n_seed * args.n_seed,
        extent=args.extent,
        weight_floor=args.weight_floor,
        weight_scheme=args.adaptive_weight_scheme,
        seed=0,
    )
    if args.mesh_connectivity == "imageplane":
        return build_brightness_adaptive_imageplane_delaunay_from_truth(**common)
    return build_brightness_adaptive_sourceplane_delaunay_from_truth(**common)


def _prob_model_kwargs_for_reg(reg_kind, seed_xy_src, seed_brightness):
    kwargs = dict(reg_kind=reg_kind)
    if reg_kind == "distance_weighted_gradient":
        kwargs["reg_vertex_positions"] = seed_xy_src
    elif reg_kind == "adaptive_split":
        kwargs["reg_vertex_lam"] = vertex_lam_adaptive_split(seed_brightness)
    return kwargs


def _solve_at_lambda(
    *,
    reg_kind,
    lam_float,
    basis,
    resid_no_source,
    inv_var,
    mesh,
    seed_xy_src,
    positive,
    vertex_lam,
):
    lam = float(lam_float)
    vpos = jnp.asarray(seed_xy_src, dtype=jnp.float32) if reg_kind == "distance_weighted_gradient" else None
    vl = jnp.asarray(vertex_lam, dtype=jnp.float32) if reg_kind == "adaptive_split" and vertex_lam is not None else None
    if positive:
        s, solver = solve_source_positive(
            basis=basis,
            resid_no_source=resid_no_source,
            inv_var=inv_var,
            edges=mesh.edges,
            lam=lam,
            reg_kind=reg_kind,
            vertex_positions=np.asarray(seed_xy_src) if vpos is not None else None,
            vertex_lam=np.asarray(vertex_lam) if vl is not None else None,
        )
        return s, solver
    s = solve_source_unconstrained(
        basis=basis,
        resid_no_source=resid_no_source,
        inv_var=inv_var,
        edges=mesh.edges,
        lam=lam,
        reg_kind=reg_kind,
        vertex_positions=vpos,
        vertex_lam=vl,
    )
    return s, None


def _save_reg_comparison_panel(*, observed, adaptive_image, variant_rows, image_extent, out_path):
    """variant_rows: list of dict with keys reg_kind, lambda_rows (list of row dicts)."""
    n_var = len(variant_rows)
    n_lam = len(variant_rows[0]["lambda_rows"])
    fig, axs = plt.subplots(n_var, n_lam, figsize=(3.2 * n_lam, 3.4 * n_var), squeeze=False)
    for iv, vrow in enumerate(variant_rows):
        for il, lrow in enumerate(vrow["lambda_rows"]):
            ax = axs[iv, il]
            src = lrow["source_plane"]
            true = lrow["true_source_plane"]
            vmin = min(np.nanmin(src), np.nanmin(true))
            vmax = max(np.nanmax(src), np.nanmax(true))
            ax.imshow(src, origin="lower", cmap="magma", extent=lrow["source_extent"], vmin=vmin, vmax=vmax)
            ax.set_title(
                f"{vrow['reg_kind']}\nλ={lrow['lambda']:.0e}\n"
                f"χ²={lrow['chi2_mean']:.3f} alt={lrow['alternating_score']:.3f}\n"
                f"-2lnZ={lrow['minus2logZ']:.1f}",
                fontsize=8,
            )
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("VELA truth-pinned reg-variant × lambda comparison (reconstructed source)", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    print(f"JAX devices: {jax.devices()}")

    observed, true_params, sim_config, meta = load_vela_sim_system(
        args.sim_num,
        args.rep,
        cam=args.cam,
        num_pix=args.num_pix,
        supersample=args.supersample,
        filter_tag=args.filter_tag,
    )
    sys_dir = system_save_dir(args.sim_num, args.rep, cam=args.cam, filter_tag=args.filter_tag)
    if args.adaptive_target == "lens_img":
        adaptive_image = np.load(os.path.join(sys_dir, "lens_img.npy")).astype(np.float32)
        adaptive_image_label = "lens_img.npy"
    else:
        adaptive_image, samples_path = _adaptive_image_from_mclmc_shapelets(
            sim_config=sim_config,
            observed=observed,
            system_dir=sys_dir,
            n_max=args.adaptive_n_max,
            background_rms=args.background_rms,
            exp_time=args.exp_time,
        )
        adaptive_image_label = os.path.relpath(samples_path, sys_dir)

    src_dir = source_plane_dir(args.sim_num, cam=args.cam, filter_tag=args.filter_tag)
    true_source_image_njy = np.load(os.path.join(src_dir, "source_image.npy")).astype(np.float32)
    true_source_image = _source_image_cps_per_arcsec2(true_source_image_njy, meta)

    phys_model = PhysicalModel(
        [epl.EPL(50), shear.Shear()],
        [sersic.SersicEllipse(use_lstsq=False)],
        [],
    )
    mesh = _build_mesh(args, phys_model, true_params, adaptive_image, sim_config)
    print(
        f"Mesh ({args.mesh_connectivity}): I={mesh.seed_xy.shape[0]} "
        f"T={mesh.simplices.shape[0]} E={mesh.edges.shape[0]}"
    )

    simulator = PixelizedSourceSimulator(
        lenses=phys_model.lenses,
        lens_light_profiles=phys_model.lens_light,
        sim_config=sim_config,
        mesh=mesh,
    )
    out0 = simulator.basis_and_lens_light((true_params[0], [true_params[1][0]]))
    basis = out0.basis_images
    lens_light = out0.lens_light
    resid_no_source = (observed - lens_light).astype(jnp.float32)

    seed_bx, seed_by = simulator._beta(true_params[0], jnp.array(mesh.seed_xy[:, 0]), jnp.array(mesh.seed_xy[:, 1]))
    seed_xy_src = np.asarray(jax.device_get(jnp.stack([seed_bx, seed_by], axis=-1)), dtype=np.float32)

    # Vertex brightness at seeds for adaptive_split
    half_extent = 0.5 * sim_config.num_pix * sim_config.delta_pix
    gx = np.linspace(-half_extent, half_extent, sim_config.num_pix)
    gy = np.linspace(-half_extent, half_extent, sim_config.num_pix)
    from scipy.interpolate import RegularGridInterpolator

    grid_interp = RegularGridInterpolator((gy, gx), adaptive_image, bounds_error=False, fill_value=0.0)
    seed_brightness = grid_interp(mesh.seed_xy[:, ::-1])
    vertex_lam_rel = vertex_lam_adaptive_split(seed_brightness)

    source_center = (
        float(jnp.squeeze(true_params[2][0]["center_x"])),
        float(jnp.squeeze(true_params[2][0]["center_y"])),
    )
    image_half_extent = half_extent
    image_extent = (-image_half_extent, image_half_extent, -image_half_extent, image_half_extent)

    variant_summaries = []
    variant_panel_rows = []

    for reg_kind in args.reg_variants:
        pm_kwargs = _prob_model_kwargs_for_reg(reg_kind, seed_xy_src, seed_brightness)
        prior = build_prior(
            true_params,
            lambda_loc=max(float(args.lambda_values[0]), 1e-8),
            lambda_scale=3.0,
        )
        prob_model = PixelizedSourceProbModel(
            prior,
            observed_image=observed,
            background_rms=args.background_rms,
            exp_time=args.exp_time,
            edges=mesh.edges,
            **pm_kwargs,
        )

        evidence_rows = []
        lambda_rows = []
        scan_rows = []

        for lam_float in args.lambda_values:
            s, solver = _solve_at_lambda(
                reg_kind=reg_kind,
                lam_float=lam_float,
                basis=basis,
                resid_no_source=resid_no_source,
                inv_var=prob_model.inv_var,
                mesh=mesh,
                seed_xy_src=seed_xy_src,
                positive=not args.no_positive,
                vertex_lam=vertex_lam_rel if reg_kind == "adaptive_split" else None,
            )
            src_img = jnp.einsum("i,ihw->hw", s, basis, precision="highest")
            model = lens_light + src_img
            norm_resid = (observed - model) / jnp.sqrt(1.0 / prob_model.inv_var)
            chi2_mean = float(jnp.mean(norm_resid**2))
            alt_score = alternating_pattern_score(np.asarray(jax.device_get(s)), mesh.edges)

            x_lam = (
                true_params[0],
                [true_params[1][0]],
                [{"lambda": jnp.array([lam_float], dtype=jnp.float32)}],
            )
            z_lam = jnp.squeeze(jnp.stack(prob_model.bij.inverse(x_lam)), axis=-1)
            terms = prob_model.debug_terms(simulator, z_lam)
            minus2logZ = float(-2.0 * terms["logZ"])

            src_plane, src_extent = render_source_on_grid(
                source_values=s,
                seed_xy_src=jnp.asarray(seed_xy_src),
                out_npix=args.source_display_npix,
            )
            true_src_plane = _source_image_on_extent(
                true_source_image,
                source_center=source_center,
                source_pixel_scale=meta["source_pixel_scale_arcsec"],
                extent=src_extent,
                out_npix=src_plane.shape[0],
            )

            row = {
                "lambda": float(lam_float),
                "chi2_mean": chi2_mean,
                "alternating_score": alt_score,
                "minus2logZ": minus2logZ,
                "logZ": float(terms["logZ"]),
                "sHs": float(terms["sHs"]),
                "source_plane": src_plane,
                "true_source_plane": true_src_plane,
                "source_extent": src_extent,
                "reconstructed_peak": _peak_xy(src_plane, src_extent),
                "true_peak": _peak_xy(true_src_plane, src_extent),
                "positive_solver_success": bool(solver.success) if solver is not None else None,
            }
            scan_rows.append(row)
            lambda_rows.append(row)
            evidence_rows.append(
                {
                    "lambda": float(lam_float),
                    "chi2_mean": chi2_mean,
                    "minus2logZ": minus2logZ,
                    "logZ": float(terms["logZ"]),
                    "alternating_score": alt_score,
                }
            )
            print(
                f"[{reg_kind}] λ={lam_float:.1e} chi2={chi2_mean:.4f} "
                f"alt={alt_score:.4f} -2lnZ={minus2logZ:.2f}"
            )

        best = pick_evidence_optimal_lambda(evidence_rows)
        best_row = next(r for r in scan_rows if r["lambda"] == best["lambda"])
        peak_err = np.hypot(
            best_row["reconstructed_peak"][0] - best_row["true_peak"][0],
            best_row["reconstructed_peak"][1] - best_row["true_peak"][1],
        )
        variant_summaries.append(
            {
                "reg_kind": reg_kind,
                "evidence_optimal_lambda": best["lambda"],
                "evidence_optimal_chi2_mean": best["chi2_mean"],
                "evidence_optimal_minus2logZ": best["minus2logZ"],
                "evidence_optimal_alternating_score": best_row["alternating_score"],
                "peak_offset_arcsec": float(peak_err),
                "phase1_pass_alternating": bool(best_row["alternating_score"] > args.alternating_threshold),
                "phase1_pass_peak": bool(
                    peak_err <= float(meta["source_pixel_scale_arcsec"])
                ),
                "lambda_scan": evidence_rows,
            }
        )
        variant_panel_rows.append({"reg_kind": reg_kind, "lambda_rows": lambda_rows})

    suffix = f"vela{args.sim_num}_cam{args.cam}_rep{args.rep:02d}_nseed{args.n_seed}"
    suffix += "_adaptmclmc" if args.adaptive_target == "mclmc_shapelets" else "_adaptlensimg"
    if args.adaptive_target == "mclmc_shapelets":
        suffix += f"_nmax{args.adaptive_n_max}"
    suffix += f"_{args.adaptive_weight_scheme}_wf{args.weight_floor:g}".replace(".", "p")
    suffix += f"_{args.mesh_connectivity}"
    suffix += "_" + "_".join(args.reg_variants)
    suffix += "_unconstrained" if args.no_positive else "_positive"
    out_dir = os.path.join(args.results_dir, suffix)
    os.makedirs(out_dir, exist_ok=True)

    # Compute the degeneracy stats up front so they end up in the summary even
    # if a downstream plotting helper raises.
    degeneracy = compute_degeneracy_stats(simulator, true_params[0])

    density_meta: dict = {}
    panel_path = None

    summary = {
        "sim_num": args.sim_num,
        "rep": args.rep,
        "cam": args.cam,
        "mesh_connectivity": args.mesh_connectivity,
        "reg_variants": args.reg_variants,
        "positive": not args.no_positive,
        "adaptive_weight_scheme": args.adaptive_weight_scheme,
        "adaptive_target": args.adaptive_target,
        "adaptive_image": adaptive_image_label,
        "alternating_threshold": args.alternating_threshold,
        "vertex_density": density_meta,
        "degeneracy": degeneracy,
        "variant_summaries": variant_summaries,
    }
    summary_path = os.path.join(out_dir, "vela_truth_pinned_lambda_scan.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    if len(args.reg_variants) > 1 or len(args.lambda_values) > 1:
        panel_path = os.path.join(out_dir, "reg_variant_lambda_comparison.png")
        try:
            _save_reg_comparison_panel(
                observed=np.asarray(jax.device_get(observed)),
                adaptive_image=adaptive_image,
                variant_rows=variant_panel_rows,
                image_extent=image_extent,
                out_path=panel_path,
            )
            print(f"Saved: {panel_path}")
        except Exception as exc:  # pragma: no cover - plot helper failure
            print(f"Warning: failed to save reg comparison panel: {exc}")

    if not args.skip_density_maps:
        try:
            density_meta.update(
                plot_vertex_density_maps(
                    seed_xy_image=mesh.seed_xy,
                    seed_xy_source=seed_xy_src,
                    extent_image=args.extent,
                    extent_source=args.extent,
                    out_path=os.path.join(out_dir, "vertex_density_maps.png"),
                    title=f"Vertex density ({args.adaptive_weight_scheme})",
                )
            )
            summary["vertex_density"] = density_meta
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
        except Exception as exc:  # pragma: no cover
            print(f"Warning: failed to save vertex density maps: {exc}")

    if not args.skip_caustic_panel:
        try:
            plot_caustic_vertex_overlay(
                lenses=phys_model.lenses,
                lens_params=true_params[0],
                seed_xy_image=mesh.seed_xy,
                extent=args.extent,
                out_path=os.path.join(out_dir, "caustic_vertex_overlay.png"),
            )
        except Exception as exc:  # pragma: no cover
            print(f"Warning: failed to save caustic vertex overlay: {exc}")


if __name__ == "__main__":
    main()
