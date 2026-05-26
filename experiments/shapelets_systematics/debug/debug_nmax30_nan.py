"""Diagnostic for the n_max=30 MAP NaN/zero-gradient failure.

The plan is to reproduce a single forward pass of the fixed-lens / free-source
model on the failing system (vela01_rep03) and inspect:

  - the shapelet basis tensor coming out of ShapeletsFast.light
  - the convolved/downsampled design matrix that feeds the lstsq
  - the Gram matrix X^T X and its condition number / spectrum
  - the lstsq coefficients (Cholesky path and the NaN -> jnp.linalg.lstsq
    fallback path)
  - whether the reconstructed image is finite and resembles the source

Each diagnostic is reported for n_max=20 (known-good) and n_max=30 (failing).

The script is intentionally written to run on the login-node CPU (no GPU
required); it pinwheels through a fairly small number of forward calls.
"""

from __future__ import annotations

import argparse
import os
import sys

SHAPELETS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SHAPELETS_DIR not in sys.path:
    sys.path.insert(0, SHAPELETS_DIR)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from gigalens.jax.profiles.light import shapelets as sh

from vela_utilities import (
    build_true_params_shp,
    DEFAULT_BACKGROUND_RMS,
    DEFAULT_EXP_TIME,
    free_source_fixed_lens_model,
    load_vela_sim_system,
)


def _phi_basis_1d_f64(x, n_max):
    """float64 reference for ShapeletsFast's 1D phi basis recurrence."""
    x = np.asarray(x, dtype=np.float64)
    phi = np.empty((n_max + 1,) + x.shape, dtype=np.float64)
    phi[0] = (np.pi ** -0.25) * np.exp(-x ** 2 / 2)
    if n_max >= 1:
        phi[1] = np.sqrt(2.0) * x * phi[0]
    for n in range(2, n_max + 1):
        phi[n] = np.sqrt(2.0 / n) * x * phi[n - 1] - np.sqrt((n - 1) / n) * phi[n - 2]
    return phi


def inspect_basis_recurrence(n_max, x_max):
    """Compare the float32 ShapeletsFast recurrence against a float64 NumPy
    reference at the most stressful sample points."""
    x32 = jnp.linspace(-x_max, x_max, 4001, dtype=jnp.float32)
    phi32 = np.asarray(sh._phi_basis_1d(x32, n_max))  # shape (n_max+1, 4001)
    phi64 = _phi_basis_1d_f64(np.asarray(x32), n_max)
    abs_err = np.abs(phi32 - phi64)
    rel_err = abs_err / np.maximum(np.abs(phi64), 1e-12)
    return {
        "n_max": n_max,
        "x_max": float(x_max),
        "phi32_n_nan": int(np.isnan(phi32).sum()),
        "phi32_n_inf": int(np.isinf(phi32).sum()),
        "phi32_max_abs": float(np.nanmax(np.abs(phi32))),
        "phi32_min_abs_nonzero": float(
            np.min(np.abs(phi32)[np.abs(phi32) > 0])
        ) if np.any(np.abs(phi32) > 0) else 0.0,
        "max_abs_err_vs_f64": float(np.nanmax(abs_err)),
        "max_rel_err_vs_f64 (|phi|>1e-3)": float(
            np.nanmax(rel_err[np.abs(phi64) > 1e-3])
            if np.any(np.abs(phi64) > 1e-3) else 0.0
        ),
    }


def inspect_design_matrix(sim_num, rep, n_max):
    """Build the fixed-lens free-source model, run one forward pass via
    `lstsq_simulate`, and inspect the Cholesky / lstsq pipeline."""
    print(f"\n=== inspect_design_matrix: vela{sim_num}_rep{rep:02d} n_max={n_max} ===")
    observed_img, true_params, sim_config, _ = load_vela_sim_system(
        sim_num, rep, cam="12",
    )
    model_seq, lens_sim = free_source_fixed_lens_model(
        sim_config, observed_img, true_params,
        background_rms=DEFAULT_BACKGROUND_RMS,
        exp_time=DEFAULT_EXP_TIME,
        use_shapelets=True, n_max=n_max,
    )
    prob_model = model_seq.prob_model

    # Use the truth as our test point - same as run_vela_modeling does.
    # The fixed-lens model has source priors LogNormal(log(0.7), 0.4) for beta
    # plus tight Normal centers; the prior mean gets used here.
    sample = prob_model.prior.sample(seed=jax.random.PRNGKey(0))
    # Plug truth back in where it exists; we only really care about the
    # source-side parameters so any prior sample for them works.
    truth_with_shp = build_true_params_shp(true_params, sample[2][0])
    # `lstsq_simulate` expects each leaf to have a leading batch dim of size 1
    # (we use bs=1 simulators throughout this script).
    truth_batched = jax.tree.map(lambda x: jnp.asarray(x)[None] if jnp.asarray(x).ndim == 0
                                 else jnp.asarray(x), truth_with_shp)

    err_map = prob_model.err_map

    recon, coeffs = lens_sim.lstsq_simulate(
        truth_batched, observed_img, err_map,
    )
    recon = np.asarray(recon)
    coeffs = np.asarray(coeffs)

    # Re-build the design matrix the same way lstsq_simulate does, but at
    # python level so we can probe the linear-algebra step by hand.
    sersic_basis = np.asarray(
        lens_sim.phys_model.lens_light[0].light(
            lens_sim.img_X, lens_sim.img_Y, **truth_batched[1][0],
        )
    )
    # shapelets.light returns (n_layers, h, w, bs)
    beta_x, beta_y = lens_sim._beta(truth_batched[0])
    shp_basis = np.asarray(
        lens_sim.phys_model.source_light[0].light(
            beta_x, beta_y, **truth_batched[2][0],
        )
    )

    full_basis = np.concatenate([sersic_basis, shp_basis], axis=0)
    # nan_to_num matches the in-simulator behaviour.
    full_basis = np.nan_to_num(full_basis)

    # Convolve each component with the PSF if there is one.
    if lens_sim.flat_kernel is not None:
        psf = np.asarray(np.squeeze(lens_sim.flat_kernel))
        # SAME-convolve each component channel-by-channel.
        from scipy.signal import fftconvolve
        conv = np.empty_like(full_basis)
        for j in range(full_basis.shape[0]):
            conv[j, ..., 0] = fftconvolve(full_basis[j, ..., 0], psf, mode="same")
        full_basis = conv

    # Reshape for normal-equation analysis.
    # full_basis shape: (n_comp, h, w, bs=1)
    n_comp, h, w, _ = full_basis.shape
    W = (1.0 / np.asarray(err_map))
    X = (full_basis[..., 0] * W[None]).reshape(n_comp, -1).T  # (h*w, n_comp)
    Y = (np.asarray(observed_img) * W).reshape(-1, 1)         # (h*w, 1)

    print(f"basis shape (post conv): {full_basis.shape}  (n_comp={n_comp})")
    print(f"basis NaN entries: {int(np.isnan(full_basis).sum())}")
    print(f"basis Inf entries: {int(np.isinf(full_basis).sum())}")
    print(f"basis max |value|: {float(np.nanmax(np.abs(full_basis))):.3e}")
    nonzero_min = np.min(np.abs(full_basis)[np.abs(full_basis) > 0])
    print(f"basis min nonzero |value|: {float(nonzero_min):.3e}")

    # Component-wise L2 norms after weighting, which is what determines the
    # diagonal of X^T X.
    diag_XtX = np.sum(X * X, axis=0)
    print("\nComponent-wise diagonal of X^T X "
          f"(weighted basis squared sum), len={len(diag_XtX)}:")
    print(f"  min: {float(np.min(diag_XtX)):.3e}")
    print(f"  max: {float(np.max(diag_XtX)):.3e}")
    print(f"  ratio max/min: {float(np.max(diag_XtX) / max(np.min(diag_XtX), 1e-40)):.3e}")

    # SVD-based condition number (in float64 for an honest estimate).
    X64 = X.astype(np.float64)
    s = np.linalg.svd(X64, compute_uv=False)
    cond = s[0] / s[-1] if s[-1] > 0 else np.inf
    print(f"X singular values: max={s[0]:.3e}, min={s[-1]:.3e}, cond(X)={cond:.3e}")
    print(f"X^T X cond (cond(X)^2) ≈ {cond**2:.3e}")

    # Compare cond against float32 / float64 precision thresholds.
    print(f"float32 unit roundoff ~ 1.2e-7; float32 cond-loss threshold ~ {1/1.2e-7:.1e}")

    # Reproduce the simulator's Cholesky path in float32.
    XtX = X.T @ X  # (n_comp, n_comp)
    XtY = X.T @ Y  # (n_comp, 1)
    diag_mean = np.mean(np.diag(XtX))
    jitter = 1e-6 * max(diag_mean, 1.0)
    XtX_reg = XtX + jitter * np.eye(n_comp, dtype=XtX.dtype)
    print(f"\ndiag_mean(X^T X)={diag_mean:.3e}, jitter added to diag={jitter:.3e}")

    try:
        chol = np.linalg.cholesky(XtX_reg.astype(np.float32))
        chol_nan = bool(np.isnan(chol).any())
        chol_inf = bool(np.isinf(chol).any())
        coeffs_chol = np.linalg.solve(chol.T, np.linalg.solve(chol, XtY.astype(np.float32)))
        print(f"Cholesky (float32): NaN={chol_nan}, Inf={chol_inf}, "
              f"coeffs NaN/Inf? {np.isnan(coeffs_chol).any()} / {np.isinf(coeffs_chol).any()}")
    except np.linalg.LinAlgError as exc:
        print(f"Cholesky (float32) raised: {exc}")
        coeffs_chol = None

    # And the lstsq fallback path.
    coeffs_ls, *_ = np.linalg.lstsq(XtX_reg, XtY, rcond=None)
    print(f"lstsq fallback coeffs NaN: {np.isnan(coeffs_ls).any()}, "
          f"Inf: {np.isinf(coeffs_ls).any()}, "
          f"max|coeff|: {np.nanmax(np.abs(coeffs_ls)):.3e}")

    # Reconstruction quality with each set of coefficients.
    def _chisq(c):
        if c is None:
            return float("nan")
        recon_unweighted = full_basis[..., 0].reshape(n_comp, -1).T @ c
        recon_image = recon_unweighted.reshape(h, w)
        resid = (np.asarray(observed_img) - recon_image) / np.asarray(err_map)
        return float(np.mean(resid ** 2))

    print(f"\nChi^2 per pixel with Cholesky coeffs: {_chisq(coeffs_chol)}")
    print(f"Chi^2 per pixel with lstsq fallback coeffs: {_chisq(coeffs_ls)}")
    print(f"Chi^2 per pixel reported by lstsq_simulate (jax): "
          f"{float(np.mean(((np.asarray(observed_img) - recon) / np.asarray(err_map)) ** 2)):.3e}")
    print(f"recon (from jax lstsq_simulate) finite fraction: "
          f"{float(np.mean(np.isfinite(recon))):.3f}")
    print(f"coeffs (from jax lstsq_simulate) NaN={int(np.isnan(coeffs).sum())}, "
          f"Inf={int(np.isinf(coeffs).sum())}, "
          f"max |coeff|: {float(np.nanmax(np.abs(coeffs))) if coeffs.size else 0.0:.3e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sim", default="01")
    p.add_argument("--rep", type=int, default=3)
    p.add_argument("--n-max-list", type=int, nargs="+", default=[20, 30])
    args = p.parse_args()

    print(f"JAX devices: {jax.devices()}")

    print("\n--- 1. Float32 vs Float64 basis recurrence (1D) ---")
    # Image goes from roughly -13" to +13" -> at the prior mean beta ~ 0.7"
    # the normalized argument reaches ~ +/- 18, well outside [-5, +5].
    for n_max in args.n_max_list:
        for x_max in (5.0, 10.0, 18.0):
            print(inspect_basis_recurrence(n_max, x_max))

    print("\n--- 2. Forward pass through lstsq_simulate at vela{}, rep{:02d} ---"
          .format(args.sim, args.rep))
    for n_max in args.n_max_list:
        inspect_design_matrix(args.sim, args.rep, n_max)


if __name__ == "__main__":
    main()
