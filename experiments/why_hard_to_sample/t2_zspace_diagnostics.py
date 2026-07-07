"""T2 -- z-space diagnostics (docs/logs/why-hard-to-sample.md, tests H3).

Free post-processing on an EXISTING MCLMC run npz (no new sampling). Cause
hypothesis under test (H3): sampling happens in unconstrained z-space; if the
pathology is only visible there (bound-adjacent mass turned into heavy tails
by a sigmoid-type bijector), theta-space diagnostics are structurally blind
to it.

For the run's post-burn-in unconstrained (z-space) samples:
  1. per-parameter bulk-ESS / rank-Rhat in z-space (arviz, matching the
     conventions in src/gigalens_research/inference_utils/posterior.py:
     SamplerPosterior._rhat/._ess -- max(rank-Rhat, folded-Rhat), bulk-ESS).
     Identifies and saves the worst-ESS z-directions (JSON), which T3 uses
     to choose transect directions.
  2. z-space cornerplots, chunked to <=4x4 panels/figure per the repo's
     cornerplot-legibility rule (docs/agent-operating-card.md): first figure
     = the 4 worst-ESS parameters, subsequent figures cover the rest in
     groups of <=4, ordered worst-ESS-first throughout. Plus per-parameter
     1-D z-marginal histograms (heavy tails / multimodality), chunked to
     <=16 panels (4x4 grid) per figure, same ordering.
  3. bound-crowding: fraction of posterior mass in the outer 1% (1%-wide band
     at EACH edge; union of both bands) of each parameter's PRIOR BOX in
     theta-space. Requires the bijector + priors (via build_model in
     --data-dir); SKIPPED loudly (never fabricated) if that cannot be
     imported (e.g. no jax outside the Shifter container -- see
     docs/env_setup.md). A z-space proxy (fraction with |z-median| >
     4*MAD per dimension) is always computed and reported, but is clearly
     labeled a PROXY, never a substitute for the pre-registered theta-space
     metric.
  4. JSON + printed table + the T2 falsifier check restated next to the
     actual numbers. This script proposes nothing beyond `proposed
     (UNCERTIFIED)` -- it does not decide pass/fail; a grader must.

Parameter names (KNOWN CAVEAT -- finding C-8, docs/logs/carousel-mclmc-sampling.md):
past analyses have REVERSED the z-index<->name mapping. Root cause (confirmed
by reading this repo's OWN fix, `src/gigalens_research/plotting/diagnostics.py
::_z_space_labels` and `plotting/convergence.py::_param_labels`, both with a
"COLUMN ORDER NOTE" docstring): the sampler builds z via `bij.forward(list(z.T))`;
JAX/TFP's `pack_sequence_as` traverses dict pytrees with keys SORTED
alphabetically, so sampler column ``i`` is the alphabetically-``i``-th unique
key -- but `tfd.JointDistributionNamed`'s bijector *output* dict iterates in
REVERSED-alphabetical order. Reading names straight off
`flatten_param_names(bij.forward(...))` (what
`experiments/sim_carousel/_h1h2_diag/build_model.py::param_names` and hence
the cached `names.npy` in that dir both do -- WITHOUT sorting) therefore gives
the WRONG column<->name assignment (column ``a`` gets the name for column
``DIM-1-a``). This script does NOT use `build_model.param_names()` or the
cached `names.npy` for that reason. Instead it replicates the validated
in-repo fix directly: probe the bijector with an all-zero column list and
`sorted()` the output dict's keys. If that cannot be run (no jax outside the
Shifter container -- see docs/env_setup.md) it falls back to raw `z[i]`
labels, loudly flagged. Never guesses.

Usage:
    python t2_zspace_diagnostics.py --run /path/to/run_over.npz \
        --out-dir /path/to/outdir [--num-results N] \
        [--data-dir /path/to/experiments/sim_carousel/_h1h2_diag]

`--run` also accepts a pipeline stage DIRECTORY (MCLMCStage layout:
manifest.json + arrays.npz[, diagnostics.npz]) in place of a single npz; see
`_load_stage_dir` for the empirically-determined post-burn-in-only samples_z
convention used in that mode.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


# ---------------------------------------------------------------------------
# ESS / Rhat -- mirrors src/gigalens_research/inference_utils/posterior.py
# SamplerPosterior._rhat / ._ess (rank-normalized, arviz-based). We hand-roll
# rather than import posterior.py because that module `import jax` at the top
# level, and this diagnostic is meant to run with just numpy+arviz+matplotlib
# (no jax required) wherever possible -- only part 3 (bijector) needs jax.
# ---------------------------------------------------------------------------

def _require_arviz():
    try:
        import arviz as az
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Rank-normalized R-hat/ESS diagnostics require ArviZ. It is part "
            "of the canonical gigalens env (docs/env_setup.md)."
        ) from e
    return az


def _az_diag(sz, fn):
    az = _require_arviz()
    ds = az.convert_to_dataset(np.asarray(sz))
    var = list(ds.data_vars)[0]
    return np.atleast_1d(np.asarray(fn(ds)[var].values))


def rank_rhat(sz):
    """max(rank-Rhat, folded-Rhat) per parameter -- matches SamplerPosterior._rhat."""
    az = _require_arviz()
    rank = _az_diag(sz, lambda ds: az.rhat(ds, method="rank"))
    folded = _az_diag(sz, lambda ds: az.rhat(ds, method="folded"))
    return np.maximum(rank, folded)


def bulk_ess(sz):
    """Rank-normalized bulk-ESS per parameter -- matches SamplerPosterior._ess."""
    return _az_diag(sz, lambda ds: _require_arviz().ess(ds, method="bulk"))


# ---------------------------------------------------------------------------
# Run loading -- raise on anything missing/ambiguous, never silently default.
# ---------------------------------------------------------------------------

def _load_stage_dir(run_dir, num_results_override, tag="t2"):
    """Load post-burn-in z-samples from a pipeline stage DIRECTORY (MCLMCStage
    layout: manifest.json + arrays.npz [+ diagnostics.npz]), as an alternative
    to the single-npz `position`-key path below.

    EMPIRICAL FINDING (inspected 2026-07-02 on testsys60 and the Vela
    reference run): `arrays.npz['samples_z']` has shape (chains, num_results,
    dim) -- its step axis already equals manifest config `num_results`
    exactly, NOT `num_burnin_steps + num_results`. `diagnostics.npz`'s own
    `samples_z` copy is bit-identical (same shape, same values) -- also
    post-burn-in only -- even though the OTHER diagnostics arrays in that
    file (step_size, L, xi, nonan) DO span the full
    `num_burnin_steps + num_results` step count. So for these two runs,
    MCLMCStage logs `samples_z` as post-burn-in ONLY; burn-in samples are not
    retained anywhere. This function does not assume that in general: it
    checks the step-axis length against both interpretations
    (`num_results` vs `num_burnin_steps + num_results`) and raises if it
    matches neither. If a future run's samples_z turns out to include
    burn-in, the trailing-`num_results`-rows slice below preserves the
    post-burn-in contract either way.
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

    # Cross-check diagnostics.npz's own samples_z copy, if present. Never
    # trusted silently -- just documented, and required to match one of the
    # same two interpretations (else raise).
    diagnostics_path = os.path.join(run_dir, "diagnostics.npz")
    if os.path.isfile(diagnostics_path):
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
    return d, samples_z, post, nr


def load_run(run_path, num_results_override):
    if os.path.isdir(run_path):
        return _load_stage_dir(run_path, num_results_override, tag="t2")
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
            print(f"[t2] NOTE: --num-results={num_results_override} overrides npz nr={nr}")
            nr = int(num_results_override)
    elif num_results_override is not None:
        nr = int(num_results_override)
    else:
        raise ValueError(
            f"{run_path} has no 'nr' key and --num-results was not given; cannot "
            "determine the post-burn-in slice. Refusing to silently default.")

    if nr <= 0 or nr > position.shape[1]:
        raise ValueError(
            f"num_results={nr} invalid for position.shape={position.shape}")

    post = position[:, -nr:, :]  # (chains, nr, dim) -- results phase only
    return d, position, post, nr


# ---------------------------------------------------------------------------
# Parameter names (see module docstring for the C-8 caveat / fallback order).
# ---------------------------------------------------------------------------

def recover_param_names(dim, data_dir):
    """Returns (names, source, live_import_error_or_None).

    source in {"live_bijector_sorted", "z_index_fallback", "z_index_no_data_dir"}.

    Deliberately does NOT use `build_model.param_names()` or a cached
    `names.npy`: both are built from `flatten_param_names(bij.forward(...))`
    WITHOUT sorting, which this repo's own diagnostics.py/convergence.py
    document as giving the REVERSED column<->name assignment (see module
    docstring / finding C-8). The only trustworthy path implemented here is
    the validated in-repo fix: zero-probe the bijector, sort the output
    dict's keys.
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
                "implemented for that form -- refusing to guess for a legacy "
                "nested structure")
        names = sorted(out.keys())
        if len(names) != dim:
            raise ValueError(
                f"sorted bijector-output keys gave {len(names)} names for dim={dim}")
        return names, "live_bijector_sorted", None
    except Exception as e:  # noqa: BLE001 -- any import/runtime failure -> fall back
        live_err = f"{type(e).__name__}: {e}"

    return [f"z[{i}]" for i in range(dim)], "z_index_fallback", live_err


def print_name_warning(source, live_err):
    if source == "live_bijector_sorted":
        return
    print("=" * 78)
    print("[t2] WARNING: parameter names are NOT independently verified this session.")
    print(f"[t2]   name_source = {source}")
    if live_err:
        print(f"[t2]   live bijector probe failed: {live_err}")
    if source == "z_index_fallback":
        print("[t2]   Could not import build_model / run the bijector probe (likely no")
        print("[t2]   jax outside the Shifter container). Falling back to raw z[i]")
        print("[t2]   indices. Deliberately NOT falling back to build_model.param_names()")
        print("[t2]   or the cached names.npy in --data-dir -- both are built from")
        print("[t2]   flatten_param_names(bij.forward(...)) WITHOUT sorting the output")
        print("[t2]   dict, which this repo's own diagnostics.py/convergence.py document")
        print("[t2]   as REVERSED relative to true sampler-column order (finding C-8).")
    elif source == "z_index_no_data_dir":
        print("[t2]   --data-dir not given; falling back to raw z[i] indices.")
    print("=" * 78)


# ---------------------------------------------------------------------------
# Part 3a: bound-crowding in theta-space (needs the bijector + priors).
# ---------------------------------------------------------------------------

def compute_bound_crowding(data_dir, post_flat, names, dim):
    """Returns (status, per_param_list_or_None, note_or_None).

    status in {"ok", "skipped_no_data_dir", "skipped_no_bijector"}.
    """
    if not data_dir:
        return "skipped_no_data_dir", None, "no --data-dir given"

    data_dir = os.path.abspath(data_dir)
    try:
        if data_dir not in sys.path:
            sys.path.insert(0, data_dir)
        from build_model import prob_model  # type: ignore
        import jax.numpy as jnp
        from gigalens_research.plotting.labels import flatten_params
    except Exception as e:  # noqa: BLE001
        return "skipped_no_bijector", None, f"{type(e).__name__}: {e}"

    try:
        lens_model = prob_model.model
        unique = getattr(lens_model, "_unique", None)
        if not unique:
            joint = getattr(prob_model, "prior", None)
            unique = dict(getattr(joint, "model", {})) if joint is not None else None
        if not unique:
            raise AttributeError(
                "could not locate the per-parameter prior dict "
                "(LensModel._unique / ProbModel.prior.model)")
    except Exception as e:  # noqa: BLE001
        return "skipped_no_bijector", None, f"prior dict unavailable: {type(e).__name__}: {e}"

    try:
        theta_struct = prob_model.bij.forward(list(jnp.asarray(post_flat).T))
        theta_flat = {k: np.asarray(v) for k, v in flatten_params(theta_struct).items()}
    except Exception as e:  # noqa: BLE001
        return "skipped_no_bijector", None, f"bijector forward failed: {type(e).__name__}: {e}"

    results = []
    for i, name in enumerate(names):
        dist = unique.get(name)
        if dist is None:
            results.append({"index": i, "param": name, "status": "no_matching_prior_entry",
                             "crowding_frac": None, "bounds": None})
            continue
        low = getattr(dist, "low", None)
        high = getattr(dist, "high", None)
        if low is None or high is None:
            results.append({"index": i, "param": name, "status": "unbounded_prior_no_box",
                             "crowding_frac": None, "bounds": None})
            continue
        lo = float(np.asarray(low)); hi = float(np.asarray(high))
        width = hi - lo
        if not (width > 0):
            results.append({"index": i, "param": name, "status": "degenerate_box",
                             "crowding_frac": None, "bounds": [lo, hi]})
            continue
        vals = theta_flat.get(name)
        if vals is None:
            results.append({"index": i, "param": name, "status": "no_matching_theta_column",
                             "crowding_frac": None, "bounds": [lo, hi]})
            continue
        vals = np.asarray(vals).reshape(-1)
        outer_lo = lo + 0.01 * width
        outer_hi = hi - 0.01 * width
        frac = float(np.mean((vals <= outer_lo) | (vals >= outer_hi)))
        results.append({"index": i, "param": name, "status": "ok",
                         "bounds": [lo, hi], "crowding_frac": frac})
    return "ok", results, None


def mad_proxy(post_flat):
    """z-space SUPPLEMENT (never the pre-registered metric): fraction of samples
    with |z_i - median_i| > 4*MAD_i (raw, unscaled MAD) per dimension. For
    sigmoid-type bijectors, theta-space bound-crowding shows up as heavy |z|
    tails, so this is a cheap proxy when the bijector cannot be loaded."""
    post_flat = np.asarray(post_flat, dtype=np.float64)
    med = np.median(post_flat, axis=0)
    mad = np.median(np.abs(post_flat - med), axis=0)
    mad_safe = np.where(mad > 0, mad, 1e-12)
    frac = np.mean(np.abs(post_flat - med) > 4.0 * mad_safe, axis=0)
    return frac


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def make_plots(post_flat, names, ess, out_dir, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import corner

    dim = post_flat.shape[1]
    order = np.argsort(ess)  # ascending: worst (lowest) ESS first

    # -- cornerplots, <=4x4 panels/figure, worst-ESS-first chunking (repo rule) --
    chunks = [order[i:i + 4] for i in range(0, dim, 4)]
    corner_paths = []
    for c_idx, chunk in enumerate(chunks):
        cols = post_flat[:, chunk]
        labels = [names[i] for i in chunk]
        fig = corner.corner(cols, labels=labels, show_titles=True, title_fmt=".2g",
                             label_kwargs={"fontsize": 9})
        fig.suptitle(
            f"T2 z-space corner [{tag}] chunk {c_idx} (worst-ESS-first; "
            f"params {list(chunk)})", fontsize=10, y=1.02)
        path = os.path.join(out_dir, f"t2_corner_{tag}_chunk{c_idx:02d}.png")
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        corner_paths.append(path)

    # -- 1-D z-marginal histograms, <=16 panels/figure (4x4 grid), same order --
    hist_chunks = [order[i:i + 16] for i in range(0, dim, 16)]
    hist_paths = []
    for h_idx, chunk in enumerate(hist_chunks):
        n = len(chunk)
        ncols = 4
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
        axes = axes.ravel()
        for ax, i in zip(axes, chunk):
            ax.hist(post_flat[:, i], bins=60, color="steelblue")
            ax.set_title(f"{names[i]}\nESS={ess[i]:.0f}", fontsize=8)
            ax.tick_params(labelsize=7)
        for ax in axes[len(chunk):]:
            ax.axis("off")
        fig.suptitle(f"T2 z-marginal histograms [{tag}] chunk {h_idx} (worst-ESS-first)")
        fig.tight_layout()
        path = os.path.join(out_dir, f"t2_zhist_{tag}_chunk{h_idx:02d}.png")
        fig.savefig(path, dpi=110)
        plt.close(fig)
        hist_paths.append(path)

    return corner_paths, hist_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True,
                    help="path to a run npz (e.g. run_over.npz) OR a pipeline stage "
                         "directory (manifest.json + arrays.npz)")
    p.add_argument("--num-results", type=int, default=None,
                    help="override/ required if the npz has no 'nr' key")
    p.add_argument("--out-dir", required=True, help="output directory (created if missing)")
    p.add_argument("--data-dir", default=None,
                    help="the _h1h2_diag dir, for bijector/name recovery (optional)")
    p.add_argument("--worst-k", type=int, default=8,
                    help="number of worst-ESS directions to save for T3 (default 8)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(os.path.normpath(args.run)))[0]

    d, position, post, nr = load_run(args.run, args.num_results)
    n_chains, n_used, dim = post.shape
    post_flat = post.reshape(-1, dim)
    print(f"[t2] run={args.run}")
    print(f"[t2] position shape={position.shape}  post-burn-in slice: nr={nr} "
          f"-> post shape={post.shape}  flat n={post_flat.shape[0]}")

    # -- 1. per-parameter ESS / Rhat in z-space ------------------------------
    ess = np.asarray(bulk_ess(post), dtype=float)
    rhat = np.asarray(rank_rhat(post), dtype=float)

    names, name_source, live_err = recover_param_names(dim, args.data_dir)
    print_name_warning(name_source, live_err)

    worst_k = max(1, min(args.worst_k, dim))
    worst_idx = np.argsort(ess)[:worst_k]
    worst_directions = [
        {"index": int(i), "param": names[i], "ess": float(ess[i]), "rhat": float(rhat[i])}
        for i in worst_idx
    ]
    imin, imax = int(np.argmin(ess)), int(np.argmax(rhat))
    print(f"[t2] min bulk-ESS = {ess[imin]:.3g}  (param {names[imin]!r}, idx {imin})")
    print(f"[t2] max rank-Rhat = {rhat[imax]:.4f}  (param {names[imax]!r}, idx {imax})")
    print(f"[t2] worst-{worst_k}-ESS z-directions (feed T3 transects):")
    for row in worst_directions:
        print(f"    idx={row['index']:3d}  {row['param']:32s}  ESS={row['ess']:8.2f}  "
              f"Rhat={row['rhat']:.4f}")

    worst_json_path = os.path.join(args.out_dir, f"t2_worst_ess_directions_{tag}.json")
    with open(worst_json_path, "w") as f:
        json.dump({"run": os.path.abspath(args.run), "name_source": name_source,
                    "worst_ess_directions": worst_directions}, f, indent=2)
    print(f"[t2] saved worst-ESS directions -> {worst_json_path}")

    # -- 2. plots -------------------------------------------------------------
    corner_paths, hist_paths = make_plots(post_flat, names, ess, args.out_dir, tag)
    print(f"[t2] saved {len(corner_paths)} cornerplot chunk(s), "
          f"{len(hist_paths)} z-marginal histogram chunk(s) -> {args.out_dir}")

    # -- 3. bound-crowding (theta-space, needs bijector) + z-space proxy -----
    crowding_status, crowding_rows, crowding_note = compute_bound_crowding(
        args.data_dir, post_flat, names, dim)
    if crowding_status != "ok":
        print("=" * 78)
        print(f"[t2] PART 3 SKIPPED (bound-crowding, theta-space): status={crowding_status}")
        print(f"[t2]   reason: {crowding_note}")
        print("[t2]   No theta-space bound-crowding fraction is reported; NOT fabricated.")
        print("=" * 78)

    proxy_frac = mad_proxy(post_flat)

    # -- Gaussianity of z-marginals (always available); theta-side only if
    #    the bijector loaded, for the falsifier comparison.
    from scipy import stats
    skew_z = np.asarray(stats.skew(post_flat, axis=0))
    kurt_z = np.asarray(stats.kurtosis(post_flat, axis=0, fisher=True))  # excess; 0=Gaussian

    theta_gauss = None
    if crowding_status == "ok":
        try:
            data_dir_abs = os.path.abspath(args.data_dir)
            if data_dir_abs not in sys.path:
                sys.path.insert(0, data_dir_abs)
            import jax.numpy as jnp  # noqa: F401 (already imported successfully above)
            from build_model import prob_model  # type: ignore
            from gigalens_research.plotting.labels import flatten_params
            theta_struct = prob_model.bij.forward(list(jnp.asarray(post_flat).T))
            theta_flat = {k: np.asarray(v).reshape(-1) for k, v in
                          flatten_params(theta_struct).items()}
            skew_t, kurt_t = [], []
            for name in names:
                v = theta_flat.get(name)
                if v is None:
                    skew_t.append(None); kurt_t.append(None)
                else:
                    skew_t.append(float(stats.skew(v)))
                    kurt_t.append(float(stats.kurtosis(v, fisher=True)))
            theta_gauss = {"skew": skew_t, "kurtosis_excess": kurt_t}
        except Exception as e:  # noqa: BLE001
            theta_gauss = None
            print(f"[t2] NOTE: theta-marginal Gaussianity comparison failed: "
                  f"{type(e).__name__}: {e}")

    # -- 4. table + JSON --------------------------------------------------------
    table = []
    for i in range(dim):
        row = {
            "index": i,
            "param": names[i],
            "z_ess": float(ess[i]),
            "z_rhat": float(rhat[i]),
            "z_skew": float(skew_z[i]),
            "z_kurtosis_excess": float(kurt_z[i]),
            "z_mad_tail_frac_proxy": float(proxy_frac[i]),
            "theta_crowding_frac": (crowding_rows[i]["crowding_frac"]
                                     if crowding_rows is not None else None),
            "theta_crowding_status": (crowding_rows[i]["status"]
                                       if crowding_rows is not None else crowding_status),
            "theta_skew": (theta_gauss["skew"][i] if theta_gauss is not None else None),
            "theta_kurtosis_excess": (theta_gauss["kurtosis_excess"][i]
                                       if theta_gauss is not None else None),
        }
        table.append(row)

    print("\n[t2] per-parameter table (worst-ESS first):")
    header = (f"{'idx':>4} {'param':32s} {'zESS':>9} {'zRhat':>7} {'zSkew':>7} "
              f"{'zKurt':>7} {'MADtail':>8} {'crowdFrac':>10} {'thSkew':>7} {'thKurt':>7}")
    print(header)
    order_full = np.argsort(ess)
    for i in order_full:
        r = table[i]
        cf = r["theta_crowding_frac"]
        ts = r["theta_skew"]
        tk = r["theta_kurtosis_excess"]
        print(f"{r['index']:4d} {r['param']:32s} {r['z_ess']:9.2f} {r['z_rhat']:7.4f} "
              f"{r['z_skew']:7.2f} {r['z_kurtosis_excess']:7.2f} "
              f"{r['z_mad_tail_frac_proxy']:8.4f} "
              f"{('%.4f' % cf) if cf is not None else 'N/A':>10} "
              f"{('%.2f' % ts) if ts is not None else 'N/A':>7} "
              f"{('%.2f' % tk) if tk is not None else 'N/A':>7}")

    falsifier_text = (
        "z-marginals as Gaussian as theta-marginals and bound-crowding ~ 0 for the "
        "worst-ESS parameters (threshold: a posterior comfortably inside the box "
        "puts ~0 mass in the outer 1%; anything >~ a few % is crowding -- derive the "
        "exact number from the actual histograms before judging, and record it).")
    prediction_text = (
        "H3 predicts: worst-ESS parameters are bound-crowded and/or z-marginals show "
        "heavy tails/funnel shapes absent from their theta-marginals.")

    out = {
        "experiment": "T2",
        "run": os.path.abspath(args.run),
        "run_tag": tag,
        "num_chains": int(n_chains),
        "num_results_used": int(nr),
        "dim": int(dim),
        "name_source": name_source,
        "name_source_live_import_error": live_err,
        "min_ess": float(ess[imin]), "min_ess_param": names[imin],
        "max_rhat": float(rhat[imax]), "max_rhat_param": names[imax],
        "worst_ess_directions": worst_directions,
        "bound_crowding_status": crowding_status,
        "bound_crowding_note": crowding_note,
        "corner_plot_paths": corner_paths,
        "zmarginal_hist_paths": hist_paths,
        "table": table,
        "prediction": prediction_text,
        "falsifier": falsifier_text,
        "verdict": "proposed (UNCERTIFIED) -- see per-param table; this script does not "
                   "adjudicate the falsifier, a grader must inspect the plots/numbers.",
    }
    json_path = os.path.join(args.out_dir, f"t2_zspace_diagnostics_{tag}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[t2] saved full report -> {json_path}")
    print(f"[t2] falsifier check: {falsifier_text}")
    print("[t2] verdict: proposed (UNCERTIFIED) -- inspect the table/plots above; "
          "this script does not certify a result.")


if __name__ == "__main__":
    main()
