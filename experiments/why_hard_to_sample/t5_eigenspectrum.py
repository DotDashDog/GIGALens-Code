"""T5 -- eigenspectrum / rotation read (docs/logs/why-hard-to-sample.md, quantifies H2).

Free post-processing on an EXISTING MCLMC run npz (no new sampling). Cause
hypothesis under test (H2): the posterior is essentially Gaussian but with a
large condition number whose stiff/soft directions are ROTATED combinations
of z-parameters (not axis-aligned), so a diagonal preconditioner cannot
represent them. "Gaussian but hard", no curvature required.

For the run's post-burn-in unconstrained (z-space) samples:
  1. pooled z samples -> full covariance (float64, symmetrized). Eigenspectrum
     (plot, log scale) + raw condition number.
  2. stiffest (smallest-variance / largest-curvature) and softest
     (largest-variance) eigendirections: participation across parameters
     (eigenvector components, sorted by |component|, with names or z[i]
     labels) -- axis-aligned vs. rotated verdict via the top-component^2
     fraction (close to 1 => axis-aligned; spread out => rotated).
  3. compares against what the run's preconditioner (inverse_mass_matrix)
     could represent: is it diagonal or (nearly) full? Generalized-eigenvalue
     spectrum of (cov, inverse_mass_matrix) gives the WHITENED condition
     number -- the conditioning the sampler actually experienced after
     preconditioning. Raw vs whitened condition numbers reported side by side.
  4. JSON + eigenspectrum plot + printed table + the T5
     prediction/falsifier check restated next to the actual numbers. Verdict
     is always `proposed (UNCERTIFIED)` -- this script does not adjudicate.

Preconditioner convention: `inverse_mass_matrix` in the run npz is M^-1 (the
sampler's estimate of the posterior covariance, since momentum ~ N(0, M) and
M is tuned towards the posterior precision). Whitening solves the generalized
eigenproblem  cov @ v = lambda * inverse_mass_matrix @ v ; lambda=1 for every
direction would mean the preconditioner exactly matches the true covariance
(perfect whitening). `inverse_mass_matrix` is stored per-step in the npz
(shape (1, n_steps, dim, dim), see boundary_mm.py precedent in
experiments/sim_carousel/_h1h2_diag/); the FINAL adapted matrix
`inverse_mass_matrix[0, -1]` is used.

Parameter names (KNOWN CAVEAT -- finding C-8, docs/logs/carousel-mclmc-
sampling.md): past analyses have REVERSED the z-index<->name mapping. As
established by reading this repo's own fix (`plotting/diagnostics.py
::_z_space_labels`, `plotting/convergence.py::_param_labels`): the bijector's
output dict iterates in reversed-alphabetical order, while the true sampler
column order is the ALPHABETICALLY-SORTED key order (JAX/TFP pytree
flattening sorts dict keys). `build_model.param_names()` and the cached
`names.npy` in _h1h2_diag/ both read names straight off the unsorted output
(`flatten_param_names(bij.forward(...))`) and are therefore almost certainly
REVERSED. This script does not use either; see `recover_param_names` below,
which replicates the validated in-repo fix directly (zero-probe + sorted
keys), falling back to raw z[i] (never those buggy sources) if the bijector
cannot be run.

Usage:
    python t5_eigenspectrum.py --run /path/to/run_over.npz \
        --out-dir /path/to/outdir [--num-results N] [--data-dir DIR]

`--run` also accepts a pipeline stage DIRECTORY (MCLMCStage layout:
manifest.json + arrays.npz + diagnostics.npz) in place of a single npz; see
`_load_stage_dir` for the empirically-determined samples_z / inverse_mass_matrix
conventions used in that mode.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# Run loading -- raise on anything missing/ambiguous, never silently default.
# ---------------------------------------------------------------------------

def _load_stage_dir(run_dir, num_results_override, tag="t5"):
    """Load post-burn-in z-samples + the final-adapted inverse_mass_matrix
    from a pipeline stage DIRECTORY (MCLMCStage layout: manifest.json +
    arrays.npz + diagnostics.npz), as an alternative to the single-npz
    `position`/`inverse_mass_matrix`-key path below.

    EMPIRICAL FINDING (inspected 2026-07-02 on testsys60 and the Vela
    reference run): `arrays.npz['samples_z']` has shape (chains, num_results,
    dim) -- its step axis already equals manifest config `num_results`
    exactly, NOT `num_burnin_steps + num_results`. `diagnostics.npz`'s own
    `samples_z` copy is bit-identical -- also post-burn-in only -- even
    though `diagnostics.npz['inverse_mass_matrix']` (needed here) DOES span
    the full `num_burnin_steps + num_results` step count (shape
    (1, num_burnin_steps+num_results, dim, dim) for both inspected runs), so
    this function still slices/selects it via `[0, -1]` (final adapted
    step), identical in spirit to the ndim==4 branch of the single-npz path
    below. This function does not assume the samples_z burn-in convention in
    general: it checks the step-axis length against both interpretations and
    raises if it matches neither.
    """
    manifest_path = os.path.join(run_dir, "manifest.json")
    arrays_path = os.path.join(run_dir, "arrays.npz")
    if not os.path.isfile(manifest_path) or not os.path.isfile(arrays_path):
        raise FileNotFoundError(
            f"--run={run_dir} is a directory but is missing manifest.json "
            f"and/or arrays.npz (expected pipeline MCLMCStage layout); "
            f"found: {sorted(os.listdir(run_dir))}")

    with open(manifest_path) as f:
        manifest = json.load(f)
    cfg = manifest.get("config", {})
    num_results_manifest = cfg.get("num_results")
    num_burnin_manifest = cfg.get("num_burnin_steps")
    if num_results_manifest is None:
        raise ValueError(f"{manifest_path} config has no 'num_results'; cannot verify samples_z")

    d = np.load(arrays_path)
    if "samples_z" not in d.files:
        raise KeyError(f"{arrays_path} missing 'samples_z'; found {d.files}")
    samples_z = np.asarray(d["samples_z"])
    if samples_z.ndim != 3:
        raise ValueError(f"samples_z must be (chains, steps, dim); got shape {samples_z.shape}")

    n_steps = samples_z.shape[1]
    total_incl_burnin = (num_burnin_manifest or 0) + num_results_manifest
    if n_steps == num_results_manifest:
        variant = "post_burnin_only"
    elif num_burnin_manifest is not None and n_steps == total_incl_burnin:
        variant = "includes_burnin"
    else:
        raise ValueError(
            f"{arrays_path} samples_z has {n_steps} steps on axis 1, matching "
            f"neither num_results={num_results_manifest} nor "
            f"num_burnin_steps+num_results={total_incl_burnin} from manifest "
            f"config; refusing to guess which interpretation applies.")

    diagnostics_path = os.path.join(run_dir, "diagnostics.npz")
    if not os.path.isfile(diagnostics_path):
        raise FileNotFoundError(
            f"--run={run_dir} pipeline stage dir has no diagnostics.npz; "
            f"inverse_mass_matrix (required for T5) is not available in arrays.npz")
    dd = np.load(diagnostics_path)

    if "samples_z" in dd.files:
        dsz = np.asarray(dd["samples_z"])
        if dsz.shape[1] not in (num_results_manifest, total_incl_burnin):
            raise ValueError(
                f"{diagnostics_path} samples_z has {dsz.shape[1]} steps on "
                f"axis 1, matching neither num_results={num_results_manifest} "
                f"nor num_burnin_steps+num_results={total_incl_burnin}")
        same = (dsz.shape == samples_z.shape and np.array_equal(dsz, samples_z))
        print(f"[{tag}] diagnostics.npz samples_z shape={dsz.shape} "
              f"vs arrays.npz samples_z shape={samples_z.shape}: "
              f"{'identical' if same else 'DIFFERENT'}")

    if "inverse_mass_matrix" not in dd.files:
        raise KeyError(f"{diagnostics_path} missing 'inverse_mass_matrix'; found {dd.files}")
    imm_raw = np.asarray(dd["inverse_mass_matrix"])
    # Same final-adapted-step selection logic as the single-npz path below,
    # applied to whatever ndim this run's inverse_mass_matrix actually has.
    if imm_raw.ndim == 4:
        imm = imm_raw[0, -1]
    elif imm_raw.ndim == 3:
        imm = imm_raw[-1]
    elif imm_raw.ndim == 2:
        imm = imm_raw
    else:
        raise ValueError(
            f"inverse_mass_matrix has unexpected ndim={imm_raw.ndim} "
            f"(shape {imm_raw.shape}); expected 2, 3, or 4 dims.")
    dim = samples_z.shape[-1]
    if imm.shape != (dim, dim):
        raise ValueError(
            f"resolved inverse_mass_matrix has shape {imm.shape}, expected "
            f"({dim}, {dim}) to match samples_z's last axis.")

    nr = int(num_results_manifest)
    if num_results_override is not None and int(num_results_override) != nr:
        print(f"[{tag}] NOTE: --num-results={num_results_override} overrides manifest num_results={nr}")
        nr = int(num_results_override)

    if nr <= 0 or nr > samples_z.shape[1]:
        raise ValueError(f"num_results={nr} invalid for samples_z.shape={samples_z.shape}")

    post = samples_z[:, -nr:, :]  # trailing nr rows; no-op if already post-burn-in-only and nr unchanged
    print(f"[{tag}] loaded pipeline stage dir {run_dir}")
    print(f"[{tag}]   manifest num_burnin_steps={num_burnin_manifest} num_results={num_results_manifest}")
    print(f"[{tag}]   arrays.npz samples_z shape={samples_z.shape} -> variant={variant}")
    print(f"[{tag}]   diagnostics.npz inverse_mass_matrix raw shape={imm_raw.shape} "
          f"-> final-adapted-step resolved shape={imm.shape}")
    return d, samples_z, post, nr, imm


def load_run(run_path, num_results_override):
    if os.path.isdir(run_path):
        return _load_stage_dir(run_path, num_results_override, tag="t5")
    if not os.path.isfile(run_path):
        raise FileNotFoundError(f"--run npz not found: {run_path}")
    d = np.load(run_path)
    if "position" not in d.files:
        raise KeyError(f"{run_path} is missing required key 'position'; found {d.files}")
    position = np.asarray(d["position"])
    if position.ndim != 3:
        raise ValueError(
            f"position must be (chains, steps, dim); got shape {position.shape}")

    if "nr" in d.files:
        nr = int(np.asarray(d["nr"]))
        if num_results_override is not None and int(num_results_override) != nr:
            print(f"[t5] NOTE: --num-results={num_results_override} overrides npz nr={nr}")
            nr = int(num_results_override)
    elif num_results_override is not None:
        nr = int(num_results_override)
    else:
        raise ValueError(
            f"{run_path} has no 'nr' key and --num-results was not given; cannot "
            "determine the post-burn-in slice. Refusing to silently default.")

    if nr <= 0 or nr > position.shape[1]:
        raise ValueError(f"num_results={nr} invalid for position.shape={position.shape}")

    post = position[:, -nr:, :]  # (chains, nr, dim) -- results phase only

    if "inverse_mass_matrix" not in d.files:
        raise KeyError(
            f"{run_path} is missing required key 'inverse_mass_matrix'; found {d.files}")
    imm_raw = np.asarray(d["inverse_mass_matrix"])
    if imm_raw.ndim == 4:
        # (chains_or_1, n_steps, dim, dim) -- take chain 0, final adapted step.
        imm = imm_raw[0, -1]
    elif imm_raw.ndim == 3:
        # (n_steps, dim, dim) or (chains, dim, dim); ambiguous only if square
        # dims coincide with step count, which we do not attempt to guess.
        imm = imm_raw[-1]
    elif imm_raw.ndim == 2:
        imm = imm_raw
    else:
        raise ValueError(
            f"inverse_mass_matrix has unexpected ndim={imm_raw.ndim} "
            f"(shape {imm_raw.shape}); expected 2, 3, or 4 dims.")
    dim = position.shape[-1]
    if imm.shape != (dim, dim):
        raise ValueError(
            f"resolved inverse_mass_matrix has shape {imm.shape}, expected "
            f"({dim}, {dim}) to match position's last axis.")

    return d, position, post, nr, imm


# ---------------------------------------------------------------------------
# Parameter names -- identical fallback cascade to t2_zspace_diagnostics.py.
# --data-dir is optional and not part of the pre-registered T5 CLI spec, but
# is accepted here (harmless, defaults to None -> z[i]) so T5 can reuse the
# same --data-dir as T2 when available, for more legible participation tables.
# ---------------------------------------------------------------------------

def recover_param_names(dim, data_dir):
    """Returns (names, source, live_import_error_or_None).

    source in {"live_bijector_sorted", "z_index_fallback", "z_index_no_data_dir"}.
    Deliberately does NOT use `build_model.param_names()` or a cached
    `names.npy` -- both are built without sorting the bijector output dict's
    keys and are therefore almost certainly REVERSED (see module docstring /
    finding C-8). Only the sorted zero-probe recovery is treated as trustworthy.
    """
    if not data_dir:
        return [f"z[{i}]" for i in range(dim)], "z_index_no_data_dir", None

    data_dir = os.path.abspath(data_dir)
    try:
        if data_dir not in sys.path:
            sys.path.insert(0, data_dir)
        from build_model import prob_model  # type: ignore
        probe = np.zeros((1, dim))
        out = prob_model.bij.forward(list(probe.T))
        if not isinstance(out, dict) or not out or any(
                isinstance(v, (dict, list)) for v in out.values()):
            raise TypeError(
                "bijector output is not the scene-API flat unique-key dict; "
                "sampler-column-order name recovery (sorted-keys fix) is only "
                "implemented for that form")
        names = sorted(out.keys())
        if len(names) != dim:
            raise ValueError(
                f"sorted bijector-output keys gave {len(names)} names for dim={dim}")
        return names, "live_bijector_sorted", None
    except Exception as e:  # noqa: BLE001
        live_err = f"{type(e).__name__}: {e}"

    return [f"z[{i}]" for i in range(dim)], "z_index_fallback", live_err


def print_name_warning(source, live_err):
    if source == "live_bijector_sorted":
        return
    print("=" * 78)
    print("[t5] WARNING: parameter names are NOT independently verified this session.")
    print(f"[t5]   name_source = {source}")
    if live_err:
        print(f"[t5]   live bijector probe failed: {live_err}")
    if source == "z_index_fallback":
        print("[t5]   Could not import build_model / run the bijector probe (likely no")
        print("[t5]   jax outside the Shifter container). Falling back to raw z[i]")
        print("[t5]   indices. Deliberately NOT falling back to build_model.param_names()")
        print("[t5]   or the cached names.npy -- both are unsorted and documented")
        print("[t5]   elsewhere in this repo (diagnostics.py/convergence.py) as REVERSED.")
    elif source == "z_index_no_data_dir":
        print("[t5]   --data-dir not given; falling back to raw z[i] indices.")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Linear algebra
# ---------------------------------------------------------------------------

def pooled_covariance(post_flat):
    """Full covariance in float64, symmetrized, of the pooled post-burn-in
    z-samples across all chains."""
    X = np.asarray(post_flat, dtype=np.float64)
    cov = np.cov(X, rowvar=False, ddof=1)
    cov = 0.5 * (cov + cov.T)
    return cov


def _ensure_pd(A, max_tries=8):
    """Symmetrize and, if needed, add jitter until Cholesky succeeds. Returns
    (A_stabilized, jitter_used). Raises if it cannot be stabilized."""
    A = 0.5 * (A + A.T)
    jitter = 0.0
    for _ in range(max_tries):
        try:
            np.linalg.cholesky(A + jitter * np.eye(A.shape[0]))
            return A + jitter * np.eye(A.shape[0]), jitter
        except np.linalg.LinAlgError:
            jitter = max(jitter * 10, 1e-12)
    raise np.linalg.LinAlgError(
        f"could not stabilize matrix to positive-definite after {max_tries} tries "
        f"(final jitter attempted={jitter:.3e})")


def eig_spectrum(cov):
    """Ascending eigenvalues/eigenvectors of a symmetric matrix.
    w[0]/V[:,0] = stiffest (smallest variance / highest curvature) direction.
    w[-1]/V[:,-1] = softest (largest variance) direction."""
    w, V = np.linalg.eigh(cov)
    return w, V


def participation_table(vec, names, top_n=None):
    """Eigenvector components sorted by |component| descending, with names."""
    comp2 = vec ** 2
    order = np.argsort(-np.abs(vec))
    if top_n is not None:
        order = order[:top_n]
    rows = [{"index": int(i), "param": names[i], "component": float(vec[i]),
             "component_sq": float(comp2[i])} for i in order]
    top_component_sq_fraction = float(comp2.max())  # already normalized: sum(comp2)=1
    return rows, top_component_sq_fraction


def preconditioner_diag_ratio(imm):
    diag = np.diag(imm)
    off = imm - np.diag(diag)
    max_abs_diag = float(np.max(np.abs(diag))) if diag.size else float("nan")
    max_abs_off = float(np.max(np.abs(off))) if off.size else float("nan")
    ratio_maxabs = max_abs_off / max_abs_diag if max_abs_diag > 0 else float("nan")
    fro_diag = float(np.linalg.norm(diag))
    fro_off = float(np.linalg.norm(off, "fro"))
    ratio_fro = fro_off / fro_diag if fro_diag > 0 else float("nan")
    return {"max_abs_offdiag": max_abs_off, "max_abs_diag": max_abs_diag,
            "ratio_maxabs_offdiag_to_diag": ratio_maxabs,
            "frobenius_offdiag": fro_off, "frobenius_diag": fro_diag,
            "ratio_frobenius_offdiag_to_diag": ratio_fro}


def whitened_spectrum(cov, imm):
    """Generalized eigenvalues of cov v = lambda * imm v (ascending). lambda=1
    everywhere would mean the preconditioner exactly captures the covariance."""
    from scipy.linalg import eigh as sp_eigh
    cov_pd, cov_jitter = _ensure_pd(cov)
    imm_pd, imm_jitter = _ensure_pd(imm)
    w = sp_eigh(cov_pd, imm_pd, eigvals_only=True)
    return np.asarray(w), cov_jitter, imm_jitter


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_eigenspectrum_plot(w_raw, w_whitened, out_dir, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    idx_raw = np.arange(1, len(w_raw) + 1)
    ax.semilogy(idx_raw, np.sort(w_raw)[::-1], "o-", label="raw cov eigenvalues", color="C0")
    idx_w = np.arange(1, len(w_whitened) + 1)
    ax.semilogy(idx_w, np.sort(w_whitened)[::-1], "s-",
                label="whitened (cov, inv_mass_matrix) eigenvalues", color="C1")
    ax.axhline(1.0, color="gray", ls=":", lw=1, label="whitened=1 (perfect precond.)")
    ax.set_xlabel("eigenvalue rank (descending)")
    ax.set_ylabel("eigenvalue (log scale)")
    ax.set_title(f"T5 eigenspectrum [{tag}]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, f"t5_eigenspectrum_{tag}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True,
                    help="path to a run npz (e.g. run_over.npz) OR a pipeline stage "
                         "directory (manifest.json + arrays.npz + diagnostics.npz)")
    p.add_argument("--num-results", type=int, default=None,
                    help="override/ required if the npz has no 'nr' key")
    p.add_argument("--out-dir", required=True, help="output directory (created if missing)")
    p.add_argument("--data-dir", default=None,
                    help="(optional, not in the pre-registered T5 CLI, harmless if omitted) "
                         "the _h1h2_diag dir, for name recovery only")
    p.add_argument("--participation-top-n", type=int, default=8,
                    help="how many top-loading params to print per eigendirection")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(os.path.normpath(args.run)))[0]

    d, position, post, nr, imm = load_run(args.run, args.num_results)
    n_chains, n_used, dim = post.shape
    post_flat = post.reshape(-1, dim)
    print(f"[t5] run={args.run}")
    print(f"[t5] position shape={position.shape}  post-burn-in slice: nr={nr} "
          f"-> post shape={post.shape}  flat n={post_flat.shape[0]}")
    print(f"[t5] inverse_mass_matrix resolved shape={imm.shape}")

    names, name_source, live_err = recover_param_names(dim, args.data_dir)
    print_name_warning(name_source, live_err)

    # -- 1. covariance + raw eigenspectrum ------------------------------------
    cov = pooled_covariance(post_flat)
    w_raw, V_raw = eig_spectrum(cov)
    raw_condition = float(w_raw[-1] / w_raw[0]) if w_raw[0] > 0 else float("inf")
    print(f"\n[t5] raw covariance eigenspectrum: "
          f"lambda_min={w_raw[0]:.4e}  lambda_max={w_raw[-1]:.4e}  "
          f"condition_number={raw_condition:.4e}  (log10={np.log10(raw_condition):.2f})")

    # -- 2. stiffest / softest participation ----------------------------------
    stiff_rows, stiff_top_frac = participation_table(
        V_raw[:, 0], names, top_n=args.participation_top_n)
    soft_rows, soft_top_frac = participation_table(
        V_raw[:, -1], names, top_n=args.participation_top_n)

    def _axis_verdict(top_frac, ndim):
        # purely descriptive threshold-free framing: report the number, and
        # the trivial axis-aligned reference (1/ndim = fully spread, i.e. an
        # eigenvector with equal-magnitude loadings on every axis).
        return {"top_component_sq_fraction": top_frac,
                "uniform_spread_reference": 1.0 / ndim}

    print(f"\n[t5] STIFFEST direction (lambda={w_raw[0]:.4e}); "
          f"top-{args.participation_top_n} participating params (by |component|):")
    for r in stiff_rows:
        print(f"    idx={r['index']:3d} {r['param']:32s} component={r['component']:+.4f} "
              f"component^2={r['component_sq']:.4f}")
    print(f"    top component^2 fraction = {stiff_top_frac:.4f} "
          f"(uniform-spread reference = {1.0/dim:.4f}; closer to 1 => axis-aligned, "
          f"closer to uniform => rotated combination)")

    print(f"\n[t5] SOFTEST direction (lambda={w_raw[-1]:.4e}); "
          f"top-{args.participation_top_n} participating params (by |component|):")
    for r in soft_rows:
        print(f"    idx={r['index']:3d} {r['param']:32s} component={r['component']:+.4f} "
              f"component^2={r['component_sq']:.4f}")
    print(f"    top component^2 fraction = {soft_top_frac:.4f} "
          f"(uniform-spread reference = {1.0/dim:.4f})")

    # -- 3. preconditioner form + whitened spectrum ---------------------------
    precond_diag = preconditioner_diag_ratio(imm)
    print(f"\n[t5] inverse_mass_matrix diagonality: "
          f"max|offdiag|/max|diag|={precond_diag['ratio_maxabs_offdiag_to_diag']:.4e}  "
          f"||offdiag||_F/||diag||_F={precond_diag['ratio_frobenius_offdiag_to_diag']:.4e}")

    try:
        w_whitened, cov_jitter, imm_jitter = whitened_spectrum(cov, imm)
        whitened_condition = float(w_whitened[-1] / w_whitened[0]) if w_whitened[0] > 0 else float("inf")
        whitened_status = "ok"
        whitened_note = None
        if cov_jitter or imm_jitter:
            whitened_note = (f"stabilizing jitter added: cov_jitter={cov_jitter:.3e}, "
                              f"imm_jitter={imm_jitter:.3e}")
            print(f"[t5] NOTE: {whitened_note}")
        print(f"[t5] whitened (cov vs inverse_mass_matrix) generalized eigenspectrum: "
              f"lambda_min={w_whitened[0]:.4e}  lambda_max={w_whitened[-1]:.4e}  "
              f"whitened_condition_number={whitened_condition:.4e} "
              f"(log10={np.log10(whitened_condition):.2f})")
        print(f"[t5] RAW vs WHITENED condition number: "
              f"{raw_condition:.4e}  ->  {whitened_condition:.4e}  "
              f"(improvement factor = {raw_condition/whitened_condition:.3g}x)")
    except np.linalg.LinAlgError as e:
        w_whitened = None
        whitened_condition = None
        whitened_status = "failed"
        whitened_note = str(e)
        print(f"[t5] WARNING: whitened generalized-eigenproblem failed: {e}")
        print("[t5]   whitened condition number NOT reported (not fabricated).")

    # -- eigenspectrum plot -----------------------------------------------------
    plot_path = None
    if whitened_status == "ok":
        plot_path = make_eigenspectrum_plot(w_raw, w_whitened, args.out_dir, tag)
        print(f"\n[t5] saved eigenspectrum plot -> {plot_path}")
    else:
        # still plot the raw spectrum alone so the run isn't silent on failure
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.semilogy(np.arange(1, dim + 1), np.sort(w_raw)[::-1], "o-", color="C0",
                    label="raw cov eigenvalues")
        ax.set_xlabel("eigenvalue rank (descending)"); ax.set_ylabel("eigenvalue (log)")
        ax.set_title(f"T5 raw eigenspectrum only [{tag}] (whitened spectrum FAILED)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        plot_path = os.path.join(args.out_dir, f"t5_eigenspectrum_{tag}.png")
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"\n[t5] saved RAW-ONLY eigenspectrum plot -> {plot_path} "
              "(whitened spectrum unavailable)")

    # -- printed full eigenvalue table ------------------------------------------
    print("\n[t5] full raw eigenvalue table (ascending; idx 0 = stiffest):")
    for k in range(dim):
        wv = w_whitened[k] if whitened_status == "ok" else None
        print(f"    rank={k:3d}  lambda_raw={w_raw[k]:.4e}"
              + (f"  lambda_whitened={wv:.4e}" if wv is not None else ""))

    # -- 4. JSON + falsifier check ------------------------------------------------
    prediction_text = (
        "H2 predicts: condition number >~ 1e4 with stiff directions strongly rotated "
        "(spread across many parameters), and the preconditioner form unable to "
        "capture them (whitened condition number still large).")
    falsifier_text = (
        "Falsifier: modest condition number (<~ 1e2) or axis-aligned stiffness that "
        "the existing preconditioner already represents (whitened condition number "
        "close to 1 / much smaller than the raw condition number).")

    out = {
        "experiment": "T5",
        "run": os.path.abspath(args.run),
        "run_tag": tag,
        "num_chains": int(n_chains),
        "num_results_used": int(nr),
        "dim": int(dim),
        "name_source": name_source,
        "name_source_live_import_error": live_err,
        "raw_condition_number": raw_condition,
        "raw_log10_condition_number": float(np.log10(raw_condition)),
        "raw_eigenvalues_ascending": w_raw.tolist(),
        "stiffest": {
            "eigenvalue": float(w_raw[0]),
            "participation_top": stiff_rows,
            **_axis_verdict(stiff_top_frac, dim),
        },
        "softest": {
            "eigenvalue": float(w_raw[-1]),
            "participation_top": soft_rows,
            **_axis_verdict(soft_top_frac, dim),
        },
        "preconditioner": {
            "shape": list(imm.shape),
            **precond_diag,
            "whitened_status": whitened_status,
            "whitened_note": whitened_note,
            "whitened_eigenvalues_ascending": (w_whitened.tolist()
                                                if whitened_status == "ok" else None),
            "whitened_condition_number": whitened_condition,
            "whitened_log10_condition_number": (float(np.log10(whitened_condition))
                                                 if whitened_condition else None),
        },
        "raw_vs_whitened_improvement_factor": (
            float(raw_condition / whitened_condition)
            if whitened_status == "ok" and whitened_condition else None),
        "eigenspectrum_plot_path": plot_path,
        "prediction": prediction_text,
        "falsifier": falsifier_text,
        "verdict": "proposed (UNCERTIFIED) -- see condition numbers/participation tables "
                   "above; this script does not adjudicate the falsifier, a grader must "
                   "inspect the plot/numbers.",
    }
    json_path = os.path.join(args.out_dir, f"t5_eigenspectrum_{tag}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[t5] saved full report -> {json_path}")
    print(f"[t5] prediction: {prediction_text}")
    print(f"[t5] falsifier: {falsifier_text}")
    print("[t5] verdict: proposed (UNCERTIFIED) -- inspect the numbers/plot above; "
          "this script does not certify a result.")


if __name__ == "__main__":
    main()
