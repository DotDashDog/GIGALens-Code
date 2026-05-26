from __future__ import annotations

import argparse
import json
import os
import sys
from os.path import expanduser

import jax
import jax.numpy as jnp
import numpy as np

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
from gigalens.model import PhysicalModel

from vela_utilities import (
    DEFAULT_BACKGROUND_RMS,
    DEFAULT_CAM,
    DEFAULT_EXP_TIME,
    DEFAULT_FILTER_TAG,
    DEFAULT_NUM_PIX,
    DEFAULT_SUPERSAMPLE,
    load_vela_sim_system,
    system_save_dir,
)
from voronoi_src.delaunay_mesh import (
    build_brightness_adaptive_imageplane_delaunay_from_truth,
    build_brightness_adaptive_sourceplane_delaunay_from_truth,
)
from voronoi_src.diagnostics.caustic_metrics import compute_degeneracy_stats
from voronoi_src.pixelized_prob_model import PixelizedSourceProbModel
from voronoi_src.pixelized_simulator import PixelizedSourceSimulator
from voronoi_src.tests.test_stage0 import build_prior
from voronoi_src.tests.vela_truth_pinned_lambda_scan import _adaptive_image_from_mclmc_shapelets


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2: FD vs autodiff gradient check at truth mass.")
    p.add_argument("--sim-num", type=str, default="04")
    p.add_argument("--rep", type=int, default=0)
    p.add_argument("--cam", type=str, default=DEFAULT_CAM)
    p.add_argument("--n-seed", type=int, default=30)
    p.add_argument("--extent", type=float, default=3.0)
    p.add_argument("--lambda", type=float, default=1e-4, dest="lam")
    p.add_argument("--fd-eps", type=float, default=1e-3)
    p.add_argument("--perturb-sigma", type=float, default=3.0,
                   help="Perturbation magnitude in unconstrained space for finite-perturbation test.")
    p.add_argument("--num-perturb", type=int, default=5)
    p.add_argument(
        "--mesh-connectivity",
        type=str,
        default="imageplane",
        choices=("sourceplane", "imageplane"),
    )
    p.add_argument(
        "--results-dir",
        type=str,
        default=os.path.join(HOME, "GIGALens-Code", "source_modeling", "voronoi_src", "phase2_gradient_check"),
    )
    return p.parse_args()


def main():
    args = parse_args()
    print(f"JAX devices: {jax.devices()}")
    observed, true_params, sim_config, meta = load_vela_sim_system(
        args.sim_num, args.rep, cam=args.cam, num_pix=DEFAULT_NUM_PIX, supersample=DEFAULT_SUPERSAMPLE
    )
    sys_dir = system_save_dir(args.sim_num, args.rep, cam=args.cam, filter_tag=DEFAULT_FILTER_TAG)
    adaptive_image, _ = _adaptive_image_from_mclmc_shapelets(
        sim_config=sim_config,
        observed=observed,
        system_dir=sys_dir,
        n_max=15,
        background_rms=DEFAULT_BACKGROUND_RMS,
        exp_time=DEFAULT_EXP_TIME,
    )

    phys_model = PhysicalModel([epl.EPL(50), shear.Shear()], [sersic.SersicEllipse(use_lstsq=False)], [])
    common = dict(
        lenses=phys_model.lenses,
        lens_params_truth=true_params[0],
        lensed_source_image=adaptive_image,
        num_pix=sim_config.num_pix,
        delta_pix=sim_config.delta_pix,
        supersample=sim_config.supersample,
        n_source_pixels=args.n_seed * args.n_seed,
        extent=args.extent,
        weight_floor=0.01,
        weight_scheme="normalized_floor",
    )
    if args.mesh_connectivity == "imageplane":
        mesh = build_brightness_adaptive_imageplane_delaunay_from_truth(**common)
    else:
        mesh = build_brightness_adaptive_sourceplane_delaunay_from_truth(**common)

    simulator = PixelizedSourceSimulator(
        lenses=phys_model.lenses,
        lens_light_profiles=phys_model.lens_light,
        sim_config=sim_config,
        mesh=mesh,
    )
    prior = build_prior(true_params, lambda_loc=args.lam, lambda_scale=3.0)
    prob_model = PixelizedSourceProbModel(
        prior,
        observed_image=observed,
        background_rms=DEFAULT_BACKGROUND_RMS,
        exp_time=DEFAULT_EXP_TIME,
        edges=mesh.edges,
        reg_kind="constant_gradient",
    )

    x_truth = (
        true_params[0],
        [true_params[1][0]],
        [{"lambda": jnp.array([args.lam], dtype=jnp.float32)}],
    )
    z_truth = jnp.squeeze(jnp.stack(prob_model.bij.inverse(x_truth)), axis=-1)
    n_z = int(z_truth.size)
    print(f"z_truth shape: {z_truth.shape}, n_z={n_z}")

    # Identify which group (lens_mass, lens_light, reg) each z coordinate
    # belongs to by perturbing it and inspecting which physical leaf changes.
    def _group_sizes(x_tree):
        flat, _ = jax.tree_util.tree_flatten(x_tree)
        return [int(np.prod(np.asarray(a.shape))) for a in flat]

    x_base = prob_model.bij.forward(list(z_truth[jnp.newaxis, :].T))
    flat_base, _ = jax.tree_util.tree_flatten(x_base)
    arr_base = np.concatenate([np.asarray(jax.device_get(a)).ravel() for a in flat_base])
    n_mass = sum(_group_sizes([x_base[0]]))
    n_light = sum(_group_sizes([x_base[1]]))

    def label_for_index(i: int) -> str:
        z_pert = z_truth.at[i].add(jnp.float32(0.05))
        x_pert = prob_model.bij.forward(list(z_pert[jnp.newaxis, :].T))
        flat_pert, _ = jax.tree_util.tree_flatten(x_pert)
        arr_pert = np.concatenate(
            [np.asarray(jax.device_get(a)).ravel() for a in flat_pert]
        )
        diffs = np.abs(arr_pert - arr_base)
        max_idx = int(np.argmax(diffs))
        if max_idx < n_mass:
            return f"lens_mass[{max_idx}]"
        elif max_idx < n_mass + n_light:
            return f"lens_light[{max_idx - n_mass}]"
        else:
            return f"reg[{max_idx - n_mass - n_light}]"

    def log_target(z):
        return prob_model.debug_terms(simulator, z)["log_target"]

    # Pre-jit the scalar function so each FD call is fast after warmup.
    log_target_jit = jax.jit(log_target)
    # Warmup compile
    lt0 = float(log_target_jit(z_truth))
    print(f"log_target(z_truth) = {lt0:.6g}")

    grad_fn = jax.jit(jax.grad(log_target))
    grad = grad_fn(z_truth)
    grad_np = np.asarray(jax.device_get(grad))

    fd_rows = []
    for i in range(n_z):
        e = jnp.zeros_like(z_truth).at[i].set(jnp.float32(args.fd_eps))
        f_plus = float(log_target_jit(z_truth + e))
        f_minus = float(log_target_jit(z_truth - e))
        fd = (f_plus - f_minus) / (2 * args.fd_eps)
        ad = float(grad_np[i])
        denom = max(abs(fd), abs(ad), 1e-12)
        rel_err = abs(ad - fd) / denom
        label = label_for_index(i)
        fd_rows.append(
            {
                "index": i,
                "label": label,
                "autodiff": ad,
                "finite_diff": fd,
                "abs_error": float(abs(ad - fd)),
                "relative_error": float(rel_err),
            }
        )
        print(f"  z[{i:2d}] {label:18s} ad={ad:+.6g} fd={fd:+.6g} relerr={rel_err:.3g}")

    mass_rows = [r for r in fd_rows if r["label"].startswith("lens_mass")]
    mass_max_relerr = max((r["relative_error"] for r in mass_rows), default=0.0)
    all_max_relerr = max((r["relative_error"] for r in fd_rows), default=0.0)

    rng = np.random.default_rng(0)
    perturb_ok = []
    mass_indices = [r["index"] for r in mass_rows]
    for _ in range(args.num_perturb):
        noise = rng.normal(scale=args.perturb_sigma, size=len(mass_indices))
        z_pert = z_truth
        for j, idx in enumerate(mass_indices):
            z_pert = z_pert.at[idx].add(jnp.float32(noise[j]))
        lt = float(log_target_jit(z_pert))
        perturb_ok.append(np.isfinite(lt))

    degeneracy = compute_degeneracy_stats(simulator, true_params[0])

    summary = {
        "mesh_connectivity": args.mesh_connectivity,
        "lambda": args.lam,
        "z_truth_n": n_z,
        "log_target_truth": lt0,
        "mesh_vertices": int(mesh.seed_xy.shape[0]),
        "mesh_edges": int(mesh.edges.shape[0]),
        "gradient_checks": fd_rows,
        "max_relative_error_mass": mass_max_relerr,
        "max_relative_error_all": all_max_relerr,
        "phase2_pass_gradient": bool(mass_max_relerr < 1e-3),
        "phase2_pass_finite_perturbations": bool(all(perturb_ok)),
        "finite_perturbations_ok": perturb_ok,
        "degeneracy": degeneracy,
    }

    out_dir = os.path.join(
        args.results_dir,
        f"vela{args.sim_num}_cam{args.cam}_rep{args.rep:02d}_{args.mesh_connectivity}",
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phase2_gradient_check.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "gradient_checks"}, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
