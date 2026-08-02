"""T1 -- Gaussian-clone run (docs/logs/why-hard-to-sample.md, section T1).

Run the IDENTICAL MCLMC pipeline on the Gaussian clone (built by build_clone.py)
that we ran on the real carousel posterior. If conditioning/parameterization as
seen in z-space (H2) is sufficient to explain the slowness, the clone -- which
keeps the real run's z-space second moments but is perfectly smooth and Gaussian
-- should be just as slow. If the clone is much easier, H2 is insufficient and
the hardness lives in the true density's fine structure (H1) or non-Gaussian
z-shape (H3).

Usage:
  python run_t1_clone.py --clone clone.npz \
      --data-dir /global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag \
      --out-dir ./t1_out --seed 1
"""
import argparse
import os

import numpy as np

from common import (
    assert_x64, compute_diagnostics, load_target, load_param_names,
    print_diagnostics, run_standard_mclmc,
)
from exp_config import STANDARD


def build_clone_target(clone_npz):
    """Return (GaussianCloneTarget as a prob_model-like stub, dim).

    The stub exposes `.log_prob(z)` returning a (logpdf, ()) tuple,
    because MCLMC_JIT does `prob_model.log_prob(z)[0]`. logpdf is the
    multivariate-normal log-density evaluated via the precomputed Cholesky
    (solve_triangular; no per-call covariance inversion), float64, jit/vmap-safe
    for a single z-vector of shape (dim,).
    """
    import jax.numpy as jnp
    from jax.scipy.linalg import solve_triangular

    data = np.load(clone_npz)
    mean = jnp.asarray(np.asarray(data["mean"], dtype=np.float64))
    L = jnp.asarray(np.asarray(data["cholesky"], dtype=np.float64))  # lower, cov = L L^T
    dim = int(mean.shape[0])
    # -0.5*dim*log(2pi) - sum(log(diag L)); logdet(cov) = 2*sum(log diag L)
    log_norm = -0.5 * dim * jnp.log(2.0 * jnp.pi) - jnp.sum(jnp.log(jnp.diag(L)))

    class GaussianCloneTarget:
        def log_prob(self, z):
            z = jnp.asarray(z)
            diff = z - mean
            # solve L y = diff  =>  y = L^{-1} diff ; quad = diff^T cov^{-1} diff = y^T y
            y = solve_triangular(L, diff, lower=True)
            quad = jnp.sum(y * y)
            logpdf = log_norm - 0.5 * quad
            return logpdf, ()   # MCLMC_JIT indexes [0] for the scalar log-density

    # MCLMC_JIT's `prob_model.log_prob(z)[0]` accepts this unchanged.
    return GaussianCloneTarget(), dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clone", required=True, help="clone npz from build_clone.py")
    ap.add_argument("--data-dir", required=True,
                    help="dir with z_best.npy for qz reconstruction (MAIN checkout)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    assert_x64()
    os.makedirs(args.out_dir, exist_ok=True)

    clone_stub, clone_dim = build_clone_target(args.clone)

    # ------------------------------------------------------------------
    # CRITICAL (single most important constraint in this experiment):
    # The clone run's qz MUST be the SAME qz as the real run -- the
    # MAP-centered isotropic 1e-2 ball rebuilt via load_carousel_target --
    # NOT a Gaussian built from the fitted clone mean/covariance.
    #
    # MCLMC_JIT uses qz THREE ways: initial positions (qz.sample), the INITIAL
    # inverse mass matrix (qz.covariance()), and the mass-matrix-regularization
    # anchor (qz.mean()). Handing the clone its own fitted covariance as qz
    # would give it a perfect preconditioner from step 0 and destroy the
    # experiment: the clone must face the SAME initialization and the SAME
    # mass-matrix-adaptation starting point as the real run.
    # ------------------------------------------------------------------
    _prob_model, qz, z_best, qz_dim, param_names = load_target(args.data_dir)
    if clone_dim != qz_dim:
        raise ValueError(
            f"clone dim ({clone_dim}) != qz event dim ({qz_dim}); "
            "clone and qz describe different parameter spaces.")

    if param_names is None:
        param_names = load_param_names(args.data_dir, clone_dim)

    print(f"[T1] clone dim={clone_dim}; qz = MAP-centered isotropic 1e-2 ball "
          f"(SAME as real run). config={STANDARD.to_dict()}")

    out_npz = os.path.join(args.out_dir, f"t1_clone_seed{args.seed}.npz")
    pos = run_standard_mclmc(
        clone_stub, qz, STANDARD, args.seed, out_npz,
        target_desc="Gaussian clone (full-cov fit to real run's z-space post-burn-in samples)",
        provenance={"clone_npz": os.path.abspath(args.clone),
                    "qz_data_dir": os.path.abspath(args.data_dir),
                    "qz": ("SAME as real run: qz rebuilt by load_target(data_dir) "
                           f"({type(qz).__name__}); carousel=MAP-centered 1e-2 ball, "
                           "sys60=saved SVI Gaussian, vela=saved bootstrap Gaussian")},
    )
    diag = compute_diagnostics(pos, STANDARD.num_results, param_names)
    print_diagnostics(diag, header=f"T1 clone seed {args.seed}")
    print(f"\n[T1] done. min bulk-ESS={diag['min_ess']:.4g} ({diag['min_ess_param']}), "
          f"max rank-Rhat={diag['max_rhat']:.4f} ({diag['max_rhat_param']})")


if __name__ == "__main__":
    main()
