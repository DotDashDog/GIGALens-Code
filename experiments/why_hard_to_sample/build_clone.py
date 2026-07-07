"""Build the Gaussian clone for T1 (docs/logs/why-hard-to-sample.md, section T1).

From a converged run's POST-BURN-IN samples in z-space, fit mean + FULL
covariance (float64). The clone target is that Gaussian's logpdf -- no lensing
simulator, no bijectors; it lives natively in z. This isolates H2 (rotated
ill-conditioning as seen in z-space) from {H1 micro-roughness, H3 non-Gaussian
z-shape}: the clone keeps the run's second-moment geometry but is smooth and
exactly Gaussian by construction.

Pure numpy (no jax) -- this is post-processing.

Blind-spot mitigation (T1 spec): the clone is only as trustworthy as the run it
is built from. If that run was secretly unconverged, its covariance -- and hence
the clone's difficulty -- is wrong. So we also record the SOURCE run's max
rank-R-hat and min bulk-ESS in the manifest, and the source npz path.

Usage:
  python build_clone.py --source-run t0_out/t0_seed1.npz --num-results 2000 \
      --out clone.npz [--data-dir <dir with names.npy for diagnostics labels>]
"""
import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np

from common import compute_diagnostics, git_commit, load_param_names


def _spd_cholesky(cov):
    """Symmetrize, then Cholesky. If not positive-definite, add the smallest
    jitter (scaled to the mean diagonal) that makes cholesky succeed. Returns
    (L_lower, cov_used, min_eig, jitter_added)."""
    cov = 0.5 * (cov + cov.T)
    w = np.linalg.eigvalsh(cov)
    min_eig = float(w.min())
    scale = float(np.mean(np.diag(cov)))
    jitter = 0.0
    cur = cov
    try:
        L = np.linalg.cholesky(cur)
        return L, cur, min_eig, 0.0
    except np.linalg.LinAlgError:
        pass
    # grow jitter geometrically from a tiny fraction of the diagonal scale
    j = max(scale, 1.0) * 1e-12
    for _ in range(60):
        cur = cov + j * np.eye(cov.shape[0])
        try:
            L = np.linalg.cholesky(cur)
            jitter = j
            return L, cur, min_eig, jitter
        except np.linalg.LinAlgError:
            j *= 10.0
    raise np.linalg.LinAlgError(
        f"cholesky failed even after jitter up to {j:.3e}; min_eig={min_eig:.3e}")


def _load_source_position(source_run):
    """Load a (chains, steps, dim) array + which key it came from.

    Back-compat (default, unchanged): an npz with a 'position' key (the
    run_standard_mclmc output: burnin+results steps). NEW (opt-in by input
    shape only, never changes the 'position' path): if 'position' is absent, or
    source_run is a pipeline MCLMCStage DIRECTORY, read 'samples_z' from
    arrays.npz -- the reference-run layout (post-burn-in-only samples). This
    lets T1 clone-build directly off the user's reference mclmc arrays."""
    if os.path.isdir(source_run):
        arr = os.path.join(source_run, "arrays.npz")
        if not os.path.isfile(arr):
            raise FileNotFoundError(f"{source_run} is a dir but has no arrays.npz")
        d = np.load(arr)
        if "samples_z" not in d.files:
            raise KeyError(f"{arr} has no 'samples_z'; keys={d.files}")
        return np.asarray(d["samples_z"], dtype=np.float64), "samples_z"
    d = np.load(source_run)
    if "position" in d.files:
        return np.asarray(d["position"], dtype=np.float64), "position"
    if "samples_z" in d.files:
        return np.asarray(d["samples_z"], dtype=np.float64), "samples_z"
    raise KeyError(
        f"{source_run} has neither 'position' nor 'samples_z'; keys={list(d.keys())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-run", required=True,
                    help="npz from a converged run (key 'position'), OR an mclmc "
                         "stage dir / npz with 'samples_z' (post-burn-in reference)")
    ap.add_argument("--num-results", type=int, required=True,
                    help="trailing (post-burn-in) steps of the source run to pool")
    ap.add_argument("--out", required=True, help="output clone npz path")
    ap.add_argument("--data-dir", default=None,
                    help="optional dir with names.npy for diagnostic param labels")
    ap.add_argument("--exclude-chains", default="",
                    help="comma-separated chain indices to DROP before pooling "
                         "(default '' = keep all; used for the OLD arm's metastable "
                         "chain 0). Preserves old all-chains behavior when empty.")
    args = ap.parse_args()

    position, source_key = _load_source_position(args.source_run)  # (chains, steps, dim)
    if position.ndim != 3:
        raise ValueError(f"source array must be 3-D (chains,steps,dim); got {position.shape}")

    excluded = [int(c) for c in args.exclude_chains.split(",") if c.strip() != ""]
    if excluded:
        n0 = position.shape[0]
        bad = [c for c in excluded if c < 0 or c >= n0]
        if bad:
            raise ValueError(f"--exclude-chains {bad} out of range for {n0} chains")
        keep = [c for c in range(n0) if c not in excluded]
        if not keep:
            raise ValueError("--exclude-chains removes every chain")
        print(f"[build_clone] excluding chains {excluded}; keeping {keep} "
              f"(of {n0}) -- logged choice (metastable transient).")
        position = position[keep]

    n_chains, n_total, dim = position.shape
    nr = args.num_results
    if nr > n_total:
        raise ValueError(f"--num-results {nr} exceeds source steps {n_total}")
    print(f"[build_clone] source_key={source_key} position={position.shape} nr={nr}")

    param_names = load_param_names(args.data_dir, dim)

    # Pool post-burn-in samples across the kept chains.
    post = position[:, -nr:, :]                       # (chains, nr, dim)
    pooled = post.reshape(-1, dim).astype(np.float64)  # (chains*nr, dim)

    mean = pooled.mean(axis=0)
    # full covariance in z-space (rowvar=False: variables are columns)
    cov = np.cov(pooled, rowvar=False).astype(np.float64)
    cov = 0.5 * (cov + cov.T)   # symmetrize

    L, cov_used, min_eig, jitter = _spd_cholesky(cov)
    print(f"[build_clone] dim={dim} pooled={pooled.shape[0]} "
          f"min_eig={min_eig:.3e} jitter_added={jitter:.3e}")

    # Source-run convergence read (blind-spot mitigation).
    diag = compute_diagnostics(position, nr, param_names)
    print(f"[build_clone] source run: min bulk-ESS={diag['min_ess']:.4g} "
          f"({diag['min_ess_param']}), max rank-Rhat={diag['max_rhat']:.4f} "
          f"({diag['max_rhat_param']})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez(
        args.out,
        mean=mean, cov=cov_used, cholesky=L,
        min_eig=min_eig, jitter=jitter, dim=dim,
    )

    manifest = {
        "clone_npz": os.path.abspath(args.out),
        "source_run": os.path.abspath(args.source_run),
        "source_key": source_key,
        "excluded_chains": excluded,
        "num_results_pooled": int(nr),
        "n_chains": int(n_chains),
        "n_pooled_samples": int(pooled.shape[0]),
        "dim": int(dim),
        "cov_min_eigenvalue": min_eig,
        "cholesky_jitter_added": jitter,
        # blind-spot mitigation: convergence of the run the clone is built from
        "source_run_min_bulk_ess": diag["min_ess"],
        "source_run_min_ess_param": diag["min_ess_param"],
        "source_run_max_rank_rhat": diag["max_rhat"],
        "source_run_max_rhat_param": diag["max_rhat_param"],
        "note": ("Clone is built from this run's post-burn-in z-samples; its "
                 "difficulty is only meaningful if the source run converged. "
                 "See source_run_* fields."),
        "git_commit": git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    man_path = os.path.splitext(args.out)[0] + ".manifest.json"
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[build_clone] saved {args.out} and {man_path}")


if __name__ == "__main__":
    main()
