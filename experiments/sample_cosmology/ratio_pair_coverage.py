#!/usr/bin/env python3
"""Coverage maps of the ratio-PAIR chart over the (Om0, w0) prior box.

For each candidate redshift pair (z_a, z_b) of the 7-source-plane z_lens=0.49
system (reference plane z1_2 = 0.962), builds the actual
``gigalens_research.priors.ratio_pair_coords.RatioPairBijector`` on the full
prior box Om0 in [0, 1], w0 in [-2, -1/3] and measures, on a grid:

* ROUND-TRIP COVERAGE: box-normalized error of ``forward(inverse(theta))``
  through the real Newton chart — the end-to-end guarantee that every theta in
  the box is reachable from z-space. Expected failures concentrate in the
  Om0 -> 1 degenerate sliver (dark energy vanishes, w0 unidentifiable — no
  ratio chart can cover it; module doc).
* WHITENED JACOBIAN DETERMINANT: |det d(r_a, r_b)/d(Om0, w0)| normalized by
  (box widths / image widths) — the chart's local conditioning; its smallness
  marks the degenerate sliver, and a sign flip would mark a folded (unusable)
  pair.
* the no-preimage fraction of the r sampling box (sampler-wall share of
  z-space; a burn-in efficiency cost, not a correctness cost).

Writes ``ratio_pair_coverage.png`` + per-pair numbers to
``<results_root>/results/sample_cosmology/ratio_pair_coverage``. Import-safe;
run with ``--run``. CPU is fine (JAX_PLATFORMS=cpu on login nodes).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from gigalens_research.paths import results_root
from gigalens_research.priors.ratio_pair_coords import (
    RatioPairBijector, deflection_ratio_pair_fn, validate_ratio_pair,
)

Z_LENS = 0.49
Z_REF = 0.962          # z1_2, the reference source plane of the ratio system
FIXED = dict(H0=70.0, k=0.0, wa=0.0)

OM0_BOUNDS = (0.0, 1.0)
W0_BOUNDS = (-2.0, -1.0 / 3.0)
OM_TRUE, W_TRUE = 0.3, -1.0

# Candidate coordinate pairs: three low-z x high-z choices (band-overlay
# crossing angles 10-16 deg) and one same-group control (z3 x z9, 3.6 deg)
# expected to condition much worse.
PAIRS = [
    ("z3 x z11", 1.166, 4.090),
    ("z4_5 x z8", 1.432, 3.549),
    ("z9 x z12_13", 1.506, 3.086),
    ("z3 x z9 (control)", 1.166, 1.506),
]

OUT_DIR = os.path.join(results_root(), "results", "sample_cosmology",
                       "ratio_pair_coverage")


def measure_pair(z_a, z_b, n_grid):
    from gigalens.jax.cosmo import w0waCDM_Cosmo

    cosmo = w0waCDM_Cosmo(z_lens=Z_LENS, z_source_ref=Z_REF)
    r_pair_fn = deflection_ratio_pair_fn(cosmo, (z_a, z_b), fixed=FIXED)
    bij = RatioPairBijector(r_pair_fn, OM0_BOUNDS, W0_BOUNDS)

    om = jnp.linspace(OM0_BOUNDS[0], OM0_BOUNDS[1], n_grid)
    w = jnp.linspace(W0_BOUNDS[0], W0_BOUNDS[1], n_grid)
    om_mesh, w_mesh = jnp.meshgrid(om, w, indexing="ij")
    theta = jnp.stack([om_mesh, w_mesh], axis=-1)

    theta_rt = bij.forward(bij.inverse(theta))
    err = jnp.maximum(
        jnp.abs(theta_rt[..., 0] - om_mesh) / (OM0_BOUNDS[1] - OM0_BOUNDS[0]),
        jnp.abs(theta_rt[..., 1] - w_mesh) / (W0_BOUNDS[1] - W0_BOUNDS[0]))

    def _det(om_, w0_):
        r_stacked = lambda th: jnp.stack(r_pair_fn(th[0], th[1]))
        J = jax.jacfwd(r_stacked)(jnp.stack([om_, w0_]))
        return J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]

    det = jnp.vectorize(_det)(om_mesh, w_mesh)
    det_w = det * ((OM0_BOUNDS[1] - OM0_BOUNDS[0]) * (W0_BOUNDS[1] - W0_BOUNDS[0])
                   / (bij._r_widths[0] * bij._r_widths[1]))

    ra_g, rb_g = jnp.vectorize(r_pair_fn)(om_mesh, w_mesh)
    hist, _, _ = np.histogram2d(
        np.asarray(ra_g).ravel(), np.asarray(rb_g).ravel(),
        bins=max(16, n_grid // 4),
        range=[list(bij._ra_box), list(bij._rb_box)])
    empty_frac = float((hist == 0).mean())

    return dict(
        om=np.asarray(om_mesh), w=np.asarray(w_mesh),
        err=np.asarray(err), det_w=np.asarray(det_w),
        empty_frac=empty_frac,
        ra_box=bij._ra_box, rb_box=bij._rb_box,
    )


def main(n_grid, cov_tol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(OUT_DIR, exist_ok=True)
    fig, axes = plt.subplots(len(PAIRS), 2,
                             figsize=(12.5, 4.4 * len(PAIRS)))
    summary = {}
    for row, (label, z_a, z_b) in enumerate(PAIRS):
        m = measure_pair(z_a, z_b, n_grid)
        covered = m["err"] <= cov_tol
        frac = float(covered.mean())
        min_det = float(np.min(np.abs(m["det_w"])))
        sign = 1.0 if float(np.mean(m["det_w"])) >= 0.0 else -1.0
        wrong = sign * m["det_w"] < 0.0        # opposite-sign (folded) region
        sign_flip = bool(wrong.any())
        # det_atol guidance: RatioPairUniform tolerates flips only where
        # |det_w| <= det_atol, so the binding number is the LARGEST |det_w|
        # attained inside the wrong-sign region (plus its extent, to confirm
        # it hugs the degenerate box edges and carries ~no posterior mass).
        if sign_flip:
            flip_stats = dict(
                flip_area_frac=float(wrong.mean()),
                flip_max_abs_det_w=float(np.max(np.abs(m["det_w"][wrong]))),
                flip_om0_range=[float(m["om"][wrong].min()),
                                float(m["om"][wrong].max())],
                flip_w0_range=[float(m["w"][wrong].min()),
                               float(m["w"][wrong].max())],
            )
        else:
            flip_stats = dict(flip_area_frac=0.0, flip_max_abs_det_w=0.0,
                              flip_om0_range=None, flip_w0_range=None)
        summary[label] = dict(
            z_a=z_a, z_b=z_b, coverage_frac=frac,
            min_abs_det_w=min_det, det_sign_flip=sign_flip,
            r_box_empty_frac=m["empty_frac"],
            ra_box=list(m["ra_box"]), rb_box=list(m["rb_box"]),
            **flip_stats,
        )
        print(f"[{label}] coverage(err<={cov_tol:g}) = {frac:.4f}, "
              f"min|det_w| = {min_det:.3e}, sign flip = {sign_flip}, "
              f"r-box empty frac = {m['empty_frac']:.3f}")
        if sign_flip:
            print(f"    flip region: area {flip_stats['flip_area_frac']:.4f}, "
                  f"max|det_w| {flip_stats['flip_max_abs_det_w']:.3e}, "
                  f"Om0 {flip_stats['flip_om0_range']}, "
                  f"w0 {flip_stats['flip_w0_range']}")

        # Maximal certified Om0 sub-interval: the fold curve (all ratio
        # gradients parallel as matter vanishes) lives at low Om0 for every
        # pair, and the Om0=1 dark-energy-vanishing crossing at the right
        # edge; trim one grid cell past each and CERTIFY the trimmed box with
        # the strict validator (det_atol=0: no tolerated flips at all).
        om_step = float(m["om"][1, 0] - m["om"][0, 0])
        wrong_om = m["om"][wrong]
        left = wrong_om[wrong_om < 0.5]
        right = wrong_om[wrong_om >= 0.5]
        om_lo_trim = (float(left.max()) + om_step) if left.size else OM0_BOUNDS[0]
        om_hi_trim = (float(right.min()) - om_step) if right.size else OM0_BOUNDS[1]
        trimmed = (om_lo_trim, om_hi_trim)
        from gigalens.jax.cosmo import w0waCDM_Cosmo
        r_pair_fn = deflection_ratio_pair_fn(
            w0waCDM_Cosmo(z_lens=Z_LENS, z_source_ref=Z_REF), (z_a, z_b),
            fixed=FIXED)
        try:
            trim_report = validate_ratio_pair(
                r_pair_fn, trimmed, W0_BOUNDS,
                det_atol=0.0, roundtrip_atol=1e-9,
                n_grid=101, n_roundtrip=41, n_image_grid=161)
            certified = True
            trim_info = dict(
                om0_bounds=list(trimmed), certified=True,
                min_abs_det_w=abs(trim_report["min_signed_det_w"]),
                max_roundtrip_err=trim_report["max_roundtrip_err"],
                r_box_empty_frac=trim_report["r_box_empty_frac"])
        except ValueError as e:
            certified = False
            trim_info = dict(om0_bounds=list(trimmed), certified=False,
                             error=str(e)[:400])
        summary[label]["trimmed_box"] = trim_info
        print(f"    trimmed box Om0 = ({om_lo_trim:.4f}, {om_hi_trim:.4f}): "
              f"certified = {certified}"
              + (f", min|det_w| = {trim_info['min_abs_det_w']:.3e}, "
                 f"max rt err = {trim_info['max_roundtrip_err']:.2e}"
                 if certified else f"\n    {trim_info['error'][:200]}"))

        ax = axes[row, 0]
        pc = ax.pcolormesh(m["om"], m["w"],
                           np.log10(np.maximum(m["err"], 1e-17)),
                           vmin=-17, vmax=0, cmap="viridis", shading="auto")
        fig.colorbar(pc, ax=ax, label=r"$\log_{10}$ round-trip err (box units)")
        ax.contour(m["om"], m["w"], m["err"], levels=[cov_tol],
                   colors="r", linewidths=1.2)
        ax.plot(OM_TRUE, W_TRUE, "r*", ms=12, mec="k")
        ax.set_title(f"{label}: coverage {100 * frac:.1f}% "
                     f"(red = err {cov_tol:g} contour)")
        ax.set_xlabel(r"$\Omega_{m,0}$"); ax.set_ylabel(r"$w_0$")

        ax = axes[row, 1]
        pc = ax.pcolormesh(m["om"], m["w"],
                           np.log10(np.maximum(np.abs(m["det_w"]), 1e-17)),
                           cmap="magma", shading="auto")
        fig.colorbar(pc, ax=ax, label=r"$\log_{10}|\det_w J|$")
        if sign_flip:
            ax.contour(m["om"], m["w"], m["det_w"], levels=[0.0],
                       colors="c", linewidths=1.5)
        for edge in trim_info["om0_bounds"]:
            ax.axvline(edge, color="w", ls="--", lw=1.0)
        ax.plot(OM_TRUE, W_TRUE, "r*", ms=12, mec="k")
        ax.set_title(f"{label}: whitened |det J| "
                     f"(min {min_det:.1e}"
                     f"{', SIGN FLIP' if sign_flip else ''})")
        ax.set_xlabel(r"$\Omega_{m,0}$"); ax.set_ylabel(r"$w_0$")

    fig.suptitle(
        f"Ratio-pair chart coverage, lens z={Z_LENS}, ref z={Z_REF}, "
        f"box $\\Omega_m$ {OM0_BOUNDS}, $w_0$ ({W0_BOUNDS[0]}, "
        f"{W0_BOUNDS[1]:.3f})", y=1.0)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, "ratio_pair_coverage.png")
    fig.savefig(png, dpi=130, bbox_inches="tight")
    with open(os.path.join(OUT_DIR, "coverage_summary.json"), "w") as f:
        json.dump(dict(n_grid=n_grid, cov_tol=cov_tol, pairs=summary),
                  f, indent=2)
    print("saved:", png)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true",
                        help="Compute the maps and write the figure.")
    parser.add_argument("--n-grid", type=int, default=161)
    parser.add_argument("--cov-tol", type=float, default=1e-6,
                        help="Round-trip error (box units) counted as covered.")
    args = parser.parse_args()
    if args.run:
        main(args.n_grid, args.cov_tol)
    else:
        print("--run not passed; nothing computed (import-safe check only).")
