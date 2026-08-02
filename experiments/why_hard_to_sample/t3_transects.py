"""T3 -- multi-scale transect scan of the EXACT log-density (H1 probe).

Pre-registered in docs/logs/why-hard-to-sample.md (T3 section + 2026-07-02 log
entries). Context: on sys60 T1 showed the Gaussian clone samples ~130x faster
than the real posterior (H2 insufficient), T2 falsified H3, so H1 -- micro
roughness of the true log-density -- is the surviving suspect. T3 probes it
directly with 1-D transects of logp(z) at geometrically decreasing spacings h,
through a typical-set point, along ~7 directions.

This script PRODUCES artifacts and a PROPOSED (UNCERTIFIED) verdict; it does NOT
adjudicate. The grader inspects the plots/JSON, not this summary.

Design (verbatim from the checkpoint):
  1. Typical-set point x0: one seeded random draw from the pooled post-burn-in
     reference samples (clone_source.npz); provenance recorded.
  2. Directions (unit z-vectors): 3 seeded random; stiffest + softest clone-cov
     eigenvectors; coordinate axes of the two worst-ESS params from T0
     (planes/1/light/0/center_y, planes/0/mass/0/gamma).
  3. Scale ladder per direction: sigma_dir = sqrt(e^T Sigma e); ~36 log-spaced h
     from 2*sigma_dir down to 1e-6*sigma_dir. Sampler-step reference scale:
     eps_step * sigma_dir / sqrt(dim) (see ratio derivation in the report block).
  4. Per (dir, h): second-difference curvature D2(h), central-FD gradient G(h)
     vs AD reference g.e, macro f(t) curve on t in [-2,+2] sigma_dir.
  5. CRITICAL roundoff-floor overlay: a SMOOTH f still gives D2 ~ 4*eps_f/h^2
     from roundoff (eps_f ~ |f0|*eps_mach). D2 divergence is evidence of
     roughness ONLY clearly ABOVE that model at h well above machine noise.
     eps_f is ALSO estimated empirically (discreteness of returned f).
  6. float32 positive control (--allow-float32): same analysis, reduced set
     (1 random + stiffest), x64 OFF -- O5's known noise floor MUST blow up.
  7. Efficiency: one vmap'd batch per direction; one jax.grad per precision.
  8. Outputs: JSON (raw f, D2/G tables, eps estimates, provenance, proposed
     verdict fields marked UNCERTIFIED) + plots (D2-vs-h, FD-AD, macro f).

jax is imported ONLY inside functions so this module imports cleanly under a
plain-numpy python for offline smoke tests of the pure-numpy machinery.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Worst-ESS coordinate axes from T0 (sys60).
WORST_ESS_PARAMS = ["planes/1/light/0/center_y", "planes/0/mass/0/gamma"]

TINY = 1e-300  # guard for FD/AD relative-error denominator


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ===========================================================================
# Pure-numpy machinery (no jax) -- offline-testable
# ===========================================================================

def build_h_ladder(sigma_dir, n=36, hi_factor=2.0, lo_factor=1e-6):
    """Geometric ladder of spacings h for one direction.

    From hi_factor*sigma_dir down to lo_factor*sigma_dir (descending), n points.
    Default spans 2*sigma -> 1e-6*sigma = 6.3 decades (>= the required 6)."""
    if sigma_dir <= 0:
        raise ValueError(f"sigma_dir must be > 0; got {sigma_dir}")
    return np.geomspace(hi_factor * sigma_dir, lo_factor * sigma_dir, n)


def second_difference(f_plus, f_minus, f0, h):
    """D2(h) = [f(h) - 2 f(0) + f(-h)] / h^2  (elementwise over h)."""
    f_plus = np.asarray(f_plus, dtype=np.float64)
    f_minus = np.asarray(f_minus, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    return (f_plus - 2.0 * float(f0) + f_minus) / (h * h)


def central_grad(f_plus, f_minus, h):
    """G(h) = [f(h) - f(-h)] / (2h)  (elementwise over h)."""
    f_plus = np.asarray(f_plus, dtype=np.float64)
    f_minus = np.asarray(f_minus, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    return (f_plus - f_minus) / (2.0 * h)


def roundoff_model_D2(h, f0, eps_mach):
    """Roundoff-noise model for |D2| of a SMOOTH f: ~ 4*eps_f / h^2, with
    eps_f ~ |f0| * eps_mach. This is the interpretive trap: a perfectly smooth
    surface still makes D2 diverge as h->0 purely from float roundoff. Roughness
    is only credible where measured |D2| sits clearly ABOVE this curve."""
    h = np.asarray(h, dtype=np.float64)
    eps_f = abs(float(f0)) * float(eps_mach)
    return 4.0 * eps_f / (h * h)


def _onset_from_top(h_desc, dev_mask, min_run=2):
    """Given h sorted DESCENDING and a boolean deviation mask, return the
    largest h at which deviation begins and persists for >= min_run consecutive
    steps (scanning toward smaller h). None if never. Robust to single spikes."""
    dev = np.asarray(dev_mask, dtype=bool)
    n = len(dev)
    for i in range(n):
        if dev[i] and np.all(dev[i:min(i + min_run, n)]):
            return float(h_desc[i])
    return None


def summarize_direction(h, D2, fd_ad_rel, plateau_top_decade=10.0):
    """Proposed (UNCERTIFIED) scalar reads for the per-direction table.

    h            : (n,) spacings DESCENDING (as build_h_ladder returns)
    D2           : (n,) second-difference curvature
    fd_ad_rel    : (n,) |G(h)-g.e|/(|g.e|+tiny)

    Returns dict:
      plateau_D2            : median D2 over the top decade of h (macro read)
      h_D2_deviate          : largest h where |D2/plateau| leaves [1/3,3] and
                              persists (blow-up onset); 'stable' if never
      fd_ad_min             : min FD-AD rel error over the ladder
      h_fd_ad_min           : h at that minimum
      h_fd_ad_deviate       : largest h where rel error > 3x its top-decade
                              median and persists; 'stable' if never
    """
    h = np.asarray(h, dtype=np.float64)
    D2 = np.asarray(D2, dtype=np.float64)
    fd = np.asarray(fd_ad_rel, dtype=np.float64)

    hmax = h.max()
    top = h >= hmax / plateau_top_decade
    plateau = float(np.nanmedian(D2[top])) if np.any(top) else float("nan")

    if plateau != 0 and np.isfinite(plateau):
        ratio = D2 / plateau
        dev = (~np.isfinite(ratio)) | (ratio < 1.0 / 3.0) | (ratio > 3.0)
    else:
        dev = ~np.isfinite(D2)
    h_dev = _onset_from_top(h, dev)

    if np.all(~np.isfinite(fd)):
        fd_min = float("nan"); h_fd_min = float("nan")
    else:
        fd_min = float(np.nanmin(fd))
        h_fd_min = float(h[int(np.nanargmin(fd))])
    fd_top_med = float(np.nanmedian(fd[top])) if np.any(top) else float("nan")
    thr = 3.0 * fd_top_med if np.isfinite(fd_top_med) and fd_top_med > 0 else np.inf
    fd_dev = (~np.isfinite(fd)) | (fd > thr)
    h_fd_dev = _onset_from_top(h, fd_dev)

    return {
        "plateau_D2": plateau,
        "h_D2_deviate": h_dev if h_dev is not None else "stable",
        "fd_ad_min": fd_min,
        "h_fd_ad_min": h_fd_min,
        "h_fd_ad_deviate": h_fd_dev if h_fd_dev is not None else "stable",
    }


def empirical_eps_f(probe_f):
    """Empirical eps_f (returned-value discreteness quantum) from f evaluated at
    x0 + j*h_min*e for j=0..K along a direction. Returns the smallest nonzero
    absolute successive difference and the count of distinct values."""
    probe_f = np.asarray(probe_f, dtype=np.float64)
    diffs = np.abs(np.diff(probe_f))
    nz = diffs[diffs > 0]
    quantum = float(nz.min()) if nz.size else 0.0
    return {
        "observed_quantum": quantum,
        "n_distinct_values": int(np.unique(probe_f).size),
        "n_probe_points": int(probe_f.size),
    }


# ===========================================================================
# Direction construction (numpy)
# ===========================================================================

def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("zero-length direction")
    return v / n


def sigma_along(cov, e):
    """sigma_dir = sqrt(e^T Sigma e) for unit e."""
    e = np.asarray(e, dtype=np.float64)
    return float(np.sqrt(max(e @ (np.asarray(cov, dtype=np.float64) @ e), 0.0)))


def build_directions(cov, dim, param_names, rng, n_random=3, reduced=False):
    """Construct the transect direction set.

    Full: 3 random + stiffest(min-eig) + softest(max-eig) + 2 worst-ESS axes.
    Reduced (float32 control): random_0 + stiffest only.

    Returns list of dicts: {name, kind, e (unit np array), meta}.
    """
    cov = np.asarray(cov, dtype=np.float64)
    w, V = np.linalg.eigh(cov)          # ascending eigenvalues
    stiff = _unit(V[:, 0])             # smallest eigenvalue = stiffest
    soft = _unit(V[:, -1])            # largest eigenvalue  = softest

    dirs = []
    rand_dirs = []
    for i in range(n_random):
        rand_dirs.append(_unit(rng.standard_normal(dim)))

    dirs.append({"name": "random_0", "kind": "random", "e": rand_dirs[0],
                 "meta": {"index": 0}})
    if not reduced:
        for i in range(1, n_random):
            dirs.append({"name": f"random_{i}", "kind": "random",
                         "e": rand_dirs[i], "meta": {"index": i}})

    dirs.append({"name": "stiffest", "kind": "eig_min",
                 "e": stiff, "meta": {"eigval": float(w[0])}})
    if not reduced:
        dirs.append({"name": "softest", "kind": "eig_max",
                     "e": soft, "meta": {"eigval": float(w[-1])}})
        # coordinate axes of the two worst-ESS params
        if param_names is None:
            raise ValueError("param_names required to build worst-ESS axes")
        for pname in WORST_ESS_PARAMS:
            if pname not in param_names:
                raise AssertionError(
                    f"worst-ESS param {pname!r} not in param_names {param_names}")
            j = param_names.index(pname)
            e = np.zeros(dim); e[j] = 1.0
            dirs.append({"name": f"axis[{pname}]", "kind": "axis",
                         "e": e, "meta": {"param": pname, "index": int(j)}})
    return dirs


# ===========================================================================
# Offset set per direction (for one batched evaluation)
# ===========================================================================

def build_offsets(h_ladder, macro_t, n_probe=16):
    """Concatenate all t-offsets to evaluate for one direction, plus an index
    map to slice results back apart.

    Order: [0.0] + [+h,-h per ladder rung] + macro_t + probe(j*h_min, j=1..K).
    """
    h_ladder = np.asarray(h_ladder, dtype=np.float64)
    macro_t = np.asarray(macro_t, dtype=np.float64)
    h_min = float(h_ladder[-1])

    parts = [np.array([0.0])]
    idx = {"zero": (0, 1)}
    cur = 1

    plus = h_ladder.copy()
    minus = -h_ladder.copy()
    parts.append(plus); idx["plus"] = (cur, cur + len(plus)); cur += len(plus)
    parts.append(minus); idx["minus"] = (cur, cur + len(minus)); cur += len(minus)
    parts.append(macro_t); idx["macro"] = (cur, cur + len(macro_t)); cur += len(macro_t)

    probe = np.arange(1, n_probe + 1, dtype=np.float64) * h_min
    parts.append(probe); idx["probe"] = (cur, cur + len(probe)); cur += len(probe)

    t = np.concatenate(parts)
    return t, idx


def slice_results(fvals, idx):
    fvals = np.asarray(fvals, dtype=np.float64)
    a, b = idx["zero"]; f0 = float(fvals[a])
    a, b = idx["plus"]; f_plus = fvals[a:b]
    a, b = idx["minus"]; f_minus = fvals[a:b]
    a, b = idx["macro"]; f_macro = fvals[a:b]
    a, b = idx["probe"]; f_probe = fvals[a:b]
    return f0, f_plus, f_minus, f_macro, f_probe


# ===========================================================================
# jax-backed evaluator (imported lazily)
# ===========================================================================

def make_f_eval(prob_model):
    """Return (f_eval, grad_fn):
      f_eval(Z) -> np.ndarray of logp for stacked z rows (one vmap'd jit call)
      grad_fn(x0) -> np.ndarray gradient of logp at x0 (one jax.grad call)
    """
    import jax
    import jax.numpy as jnp

    def lp(z):
        return prob_model.log_prob(z)[0]

    batched = jax.jit(jax.vmap(lp))
    grad_lp = jax.jit(jax.grad(lp))

    def f_eval(Z, _chunk=16):  # 16 pts x ~52MB basis stack keeps peak <~2GiB
        # chunked to bound memory: heavy per-point models (carousel 300x300
        # render x 73-basis lstsq) OOM a 40GB card on the all-points batch.
        # Identical numbers to the single batched call.
        Z = np.asarray(Z)
        out = np.empty(Z.shape[0], dtype=np.float64)
        for s in range(0, Z.shape[0], _chunk):
            e = min(s + _chunk, Z.shape[0])
            out[s:e] = np.asarray(batched(jnp.asarray(Z[s:e]))).reshape(-1)
        return out

    def grad_fn(x0):
        return np.asarray(grad_lp(jnp.asarray(x0)))

    return f_eval, grad_fn


# ===========================================================================
# Core analysis (pure numpy given an f_eval callable) -- offline-testable
# ===========================================================================

def run_transects(f_eval, grad_vec, x0, directions, cov, dim, eps_mach,
                  eps_step, L_ref, *, n_ladder=36, n_macro=41, n_probe=16):
    """Evaluate + analyze all transects. f_eval: callable(Z)->array. grad_vec:
    AD gradient at x0 (np array) or None. Returns a results dict (JSON-able)."""
    x0 = np.asarray(x0, dtype=np.float64)
    cov = np.asarray(cov, dtype=np.float64)
    sqrt_dim = float(np.sqrt(dim))

    out = []
    for d in directions:
        e = _unit(d["e"])
        sigma_dir = sigma_along(cov, e)
        if sigma_dir <= 0:
            raise ValueError(f"sigma_dir<=0 for direction {d['name']}")
        h_ladder = build_h_ladder(sigma_dir, n=n_ladder)
        macro_t = np.linspace(-2.0 * sigma_dir, 2.0 * sigma_dir, n_macro)
        t, idx = build_offsets(h_ladder, macro_t, n_probe=n_probe)

        Z = x0[None, :] + t[:, None] * e[None, :]   # (n_pts, dim)
        fvals = f_eval(Z)                           # one batched call
        f0, f_plus, f_minus, f_macro, f_probe = slice_results(fvals, idx)

        D2 = second_difference(f_plus, f_minus, f0, h_ladder)
        G = central_grad(f_plus, f_minus, h_ladder)
        g_dot_e = float(grad_vec @ e) if grad_vec is not None else float("nan")
        fd_ad_rel = np.abs(G - g_dot_e) / (abs(g_dot_e) + TINY)

        roundoff = roundoff_model_D2(h_ladder, f0, eps_mach)

        # sampler-step reference scale on the h-axis (z-move per step along e)
        eps_step_move = float(eps_step) * sigma_dir / sqrt_dim
        traj_move = float(L_ref) * sigma_dir / sqrt_dim
        falsifier_floor = eps_step_move / 100.0

        eps_emp = empirical_eps_f(np.concatenate([[f0], f_probe]))
        summ = summarize_direction(h_ladder, D2, fd_ad_rel)

        out.append({
            "name": d["name"],
            "kind": d["kind"],
            "meta": d["meta"],
            "sigma_dir": sigma_dir,
            "f0": f0,
            "g_dot_e": g_dot_e,
            "eps_step_move": eps_step_move,
            "traj_move": traj_move,
            "falsifier_floor": falsifier_floor,
            "eps_f_theory": abs(f0) * float(eps_mach),
            "eps_f_empirical": eps_emp,
            # arrays (as lists for JSON)
            "h": h_ladder.tolist(),
            "D2": D2.tolist(),
            "G": G.tolist(),
            "fd_ad_rel": fd_ad_rel.tolist(),
            "roundoff_model_D2": roundoff.tolist(),
            "macro_t": macro_t.tolist(),
            "macro_f": f_macro.tolist(),
            "f_plus": f_plus.tolist(),
            "f_minus": f_minus.tolist(),
            "probe_t": (np.arange(1, n_probe + 1) * h_ladder[-1]).tolist(),
            "probe_f": f_probe.tolist(),
            "summary": summ,
        })
    return out


# ===========================================================================
# Plots
# ===========================================================================

def _grid(n):
    import math
    nc = int(math.ceil(math.sqrt(n)))
    nr = int(math.ceil(n / nc))
    nc = min(nc, 4); nr = min(nr, 4)
    return nr, nc


def plot_D2(results, precision, out_path):
    """|D2| vs h log-log per direction + roundoff model + sampler-step band."""
    n = len(results)
    nr, nc = _grid(n)
    fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.6 * nr), squeeze=False)
    axes = axes.ravel()
    for k, r in enumerate(results):
        ax = axes[k]
        h = np.array(r["h"])
        D2 = np.array(r["D2"])
        ro = np.array(r["roundoff_model_D2"])
        ax.loglog(h, np.abs(D2), ".-", ms=3, lw=1, label="|D2(h)| measured")
        ax.loglog(h, ro, "--", color="crimson", lw=1,
                  label="roundoff model 4*eps_f/h^2")
        # sampler-relevant band: [floor, eps_step_move]
        floor = r["falsifier_floor"]; step = r["eps_step_move"]
        ax.axvspan(floor, step, color="0.85", alpha=0.6, zorder=0)
        ax.axvline(step, color="green", lw=0.8, ls=":")
        ax.axvline(floor, color="green", lw=0.8, ls=":")
        pl = r["summary"]["plateau_D2"]
        if np.isfinite(pl):
            ax.axhline(abs(pl), color="0.4", lw=0.6, ls="-.")
        ax.set_title(f"{r['name']}\nsigma={r['sigma_dir']:.2g} "
                     f"|plateau D2|={abs(pl):.2g}", fontsize=8)
        ax.set_xlabel("h (z-move along e)"); ax.set_ylabel("|D2|")
        ax.tick_params(labelsize=7)
        if k == 0:
            ax.legend(fontsize=6, loc="best")
    for k in range(n, len(axes)):
        axes[k].axis("off")
    fig.suptitle(f"T3 second-difference curvature vs scale [{precision}] "
                 f"(shaded = sampler band [eps_step/100, eps_step])", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_fd_ad(results, precision, out_path):
    """FD-AD relative gradient error vs h (all directions overlaid)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in results:
        h = np.array(r["h"])
        ax.loglog(h, np.array(r["fd_ad_rel"]) + TINY, ".-", ms=3, lw=1,
                  label=r["name"])
    # sampler band from the first direction (each differs; draw union roughly)
    floors = [r["falsifier_floor"] for r in results]
    steps = [r["eps_step_move"] for r in results]
    ax.axvspan(min(floors), max(steps), color="0.9", alpha=0.5, zorder=0)
    ax.set_xlabel("h (z-move along e)")
    ax.set_ylabel("|G(h) - g.e| / (|g.e| + tiny)")
    ax.set_title(f"T3 central-FD vs autodiff gradient disagreement [{precision}]",
                 fontsize=10)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_macro(results, precision, out_path):
    """Macro log-density curves f(t)-f(0) vs t/sigma_dir (all directions)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in results:
        t = np.array(r["macro_t"]) / r["sigma_dir"]
        f = np.array(r["macro_f"]) - r["f0"]
        ax.plot(t, f, ".-", ms=3, lw=1, label=r["name"])
    ax.set_xlabel("t / sigma_dir")
    ax.set_ylabel("logp(x0 + t e) - logp(x0)")
    ax.set_title(f"T3 macro log-density transects [{precision}] "
                 f"(quadratic bowl => H0/curvature read)", fontsize=10)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ===========================================================================
# Provenance / x0 draw
# ===========================================================================

def draw_typical_point(samples_path, seed):
    """Seeded single draw from pooled post-burn-in reference samples. Returns
    (x0, provenance-dict). position expected (chains, steps, dim)."""
    d = np.load(samples_path)
    if "position" not in d:
        raise KeyError(f"'position' not in {samples_path}; keys={d.files}")
    pos = np.asarray(d["position"], dtype=np.float64)
    if pos.ndim != 3:
        raise ValueError(f"position must be 3-D (chains,steps,dim); got {pos.shape}")
    n_chains, n_steps, dim = pos.shape
    flat = pos.reshape(-1, dim)
    rng = np.random.default_rng(seed)
    idx = int(rng.integers(flat.shape[0]))
    x0 = flat[idx]
    prov = {
        "samples_path": os.path.abspath(samples_path),
        "position_shape": [n_chains, n_steps, dim],
        "seed": int(seed),
        "flat_index": idx,
        "chain_index": idx // n_steps,
        "step_index": idx % n_steps,
    }
    return x0, prov, dim


def read_ref_step_size(diag_path):
    """Final adapted step size eps and final L from the reference diagnostics.
    step_size / L expected shape (chains, steps). Returns dict; asserts nothing
    beyond shape (values are inspected/recorded, not assumed equal)."""
    d = np.load(diag_path)
    if "step_size" not in d:
        raise KeyError(f"'step_size' not in {diag_path}; keys={d.files}")
    ss = np.asarray(d["step_size"], dtype=np.float64)
    if ss.ndim != 2:
        raise ValueError(f"step_size expected (chains,steps); got {ss.shape}")
    ss_final = ss[:, -1]
    L_final = None
    if "L" in d:
        Larr = np.asarray(d["L"], dtype=np.float64)
        L_final = Larr[:, -1]
    return {
        "diag_path": os.path.abspath(diag_path),
        "step_size_shape": list(ss.shape),
        "step_size_final_per_chain": ss_final.tolist(),
        "eps_step": float(np.median(ss_final)),
        "L_final_per_chain": (L_final.tolist() if L_final is not None else None),
        "L_ref": (float(np.median(L_final)) if L_final is not None else float("nan")),
    }


# ===========================================================================
# main
# ===========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="T3 multi-scale transect scan")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--samples", required=True,
                   help="clone_source.npz (post-burn-in reference samples)")
    p.add_argument("--clone", required=True,
                   help="clone.npz with mean/cov (eigen + sigma_dir source)")
    p.add_argument("--ref-diagnostics", required=True,
                   help="reference mclmc diagnostics.npz (step_size, L)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--allow-float32", action="store_true",
                   help="float32 positive control: skip x64 assert, reduced dir "
                        "set (random_0 + stiffest), eps_mach=1.2e-7, label f32")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    # precision handling
    import jax  # lazy
    if args.allow_float32:
        if jax.config.read("jax_enable_x64"):
            raise RuntimeError(
                "--allow-float32 given but jax_enable_x64 is ON. Launch this "
                "arm with JAX_ENABLE_X64=0 so the log-density runs in float32.")
        precision = "float32"
        eps_mach = 1.2e-7
        reduced = True
    else:
        # assert float64 exactly like common.assert_x64
        from common import assert_x64
        assert_x64()
        precision = "float64"
        eps_mach = 2.2e-16
        reduced = False

    print(f"[t3] precision={precision} eps_mach={eps_mach} reduced={reduced}")

    # inputs
    x0, x0_prov, dim = draw_typical_point(args.samples, args.seed)
    clone = np.load(args.clone)
    cov = np.asarray(clone["cov"], dtype=np.float64)
    if cov.shape != (dim, dim):
        raise ValueError(f"clone cov shape {cov.shape} != ({dim},{dim})")
    ref = read_ref_step_size(args.ref_diagnostics)
    eps_step = ref["eps_step"]; L_ref = ref["L_ref"]
    print(f"[t3] eps_step={eps_step:.6g} L_ref={L_ref:.6g} dim={dim}")

    # target (for param names + evaluator)
    from common import load_target
    prob_model, qz, z_center, dim2, param_names = load_target(args.data_dir)
    if dim2 != dim:
        raise ValueError(f"data-dir dim {dim2} != samples dim {dim}")

    # CONFLICT GUARD: systems/*/system.py:load_target() unconditionally calls
    # jax.config.update("jax_enable_x64", True). For the float32 positive control
    # that would silently promote the energy back to float64 and make the arm a
    # useless duplicate of the float64 run. We cannot modify system.py (task
    # scope), so force x64 back OFF here and record the ACTUALLY achieved energy
    # dtype empirically below -- the label is not trusted, the measured dtype and
    # eps_f_empirical are. Blind spot flagged in the report: forward-model
    # constants (image/PSF) were built as float64 arrays before this flip, so
    # this is an IMPERFECT float32 control; verify energy_dtype/eps_f_empirical.
    x64_forced_off = False
    if args.allow_float32 and jax.config.read("jax_enable_x64"):
        jax.config.update("jax_enable_x64", False)
        x64_forced_off = True
        print("[t3] WARNING: load_target() enabled x64; forced back OFF for the "
              "float32 control. Energy dtype recorded empirically.")

    rng = np.random.default_rng(args.seed)
    directions = build_directions(cov, dim, param_names, rng, reduced=reduced)
    print(f"[t3] {len(directions)} directions: "
          f"{[d['name'] for d in directions]}")

    f_eval, grad_fn = make_f_eval(prob_model)
    grad_vec = grad_fn(x0)   # one jax.grad call for this precision

    # record logp at x0 and the ACTUALLY achieved energy dtype
    _f0_arr = f_eval(x0[None, :])
    energy_dtype = str(np.asarray(_f0_arr).dtype)
    x0_prov["logp_x0"] = float(_f0_arr[0])
    print(f"[t3] logp(x0)={x0_prov['logp_x0']:.6g} energy_dtype={energy_dtype}")
    if precision == "float32" and energy_dtype == "float64":
        print("[t3] WARNING: float32 control still returned float64 energy "
              "(forward-model float64 constants dominate). eps_f_empirical will "
              "reveal the true floor; flagged in JSON.")

    results = run_transects(f_eval, grad_vec, x0, directions, cov, dim,
                            eps_mach, eps_step, L_ref)

    # plots
    tag = precision
    plot_D2(results, tag, os.path.join(args.out_dir, f"t3_D2_vs_h_{tag}.png"))
    plot_fd_ad(results, tag, os.path.join(args.out_dir, f"t3_fd_ad_{tag}.png"))
    plot_macro(results, tag, os.path.join(args.out_dir, f"t3_macro_{tag}.png"))

    # JSON
    doc = {
        "experiment": "T3 multi-scale transect scan",
        "status": "proposed (UNCERTIFIED) -- grader inspects artifacts",
        "precision": precision,
        "eps_mach": eps_mach,
        "energy_dtype_achieved": energy_dtype,
        "x64_forced_off_after_load": x64_forced_off,
        "float32_control_caveat": (
            "systems/*/system.py forces jax_enable_x64=True in load_target(); "
            "for the float32 arm x64 is forced back off afterward, but forward "
            "constants (image/PSF) were built float64 -> imperfect control. "
            "Trust energy_dtype_achieved + per-direction eps_f_empirical, not "
            "the 'float32' label."),
        "timestamp_utc": _now(),
        "args": vars(args),
        "dim": dim,
        "x0_provenance": x0_prov,
        "grad_ad": grad_vec.tolist(),
        "ref_step_size": ref,
        "eps_to_zmove_mapping": (
            "per-step z-move along unit e ~= eps_step * sigma_dir / sqrt(dim); "
            "full-trajectory ~= L * sigma_dir / sqrt(dim). MCLMC isokinetic "
            "velocity is unit-norm in the Sigma-metric (preconditioned) space, "
            "so one step of length eps moves eps in whitened norm; a single "
            "whitened axis carries ~eps/sqrt(dim), mapped to z along e by "
            "sigma_dir. Falsifier floor = eps_step_move/100."),
        "param_names": param_names,
        "verdict_fields": {
            "H1_roughness": "proposed (UNCERTIFIED): see per-direction "
                            "h_D2_deviate vs falsifier_floor and roundoff model",
            "H0_macro_curvature": "proposed (UNCERTIFIED): read macro_f curves "
                                  "and plateau_D2",
            "note": "script does NOT adjudicate; divergence counts only ABOVE "
                    "the roundoff model at h >> machine noise, and roughness is "
                    "sampler-relevant only for h >= falsifier_floor.",
        },
        "directions": results,
    }
    json_path = os.path.join(args.out_dir, f"t3_results_{tag}.json")
    with open(json_path, "w") as fh:
        json.dump(doc, fh, indent=2)
    print(f"[t3] wrote {json_path}")

    # compact per-direction summary table
    print(f"\n=== T3 per-direction summary [{precision}] "
          f"(PROPOSED / UNCERTIFIED) ===")
    print(f"{'direction':<28} {'sigma_dir':>10} {'plateauD2':>11} "
          f"{'h_D2_dev':>11} {'flr':>9} {'fd_min':>9} {'h_fd_dev':>11}")
    for r in results:
        s = r["summary"]
        def fmt(x):
            return (f"{x:.2e}" if isinstance(x, (int, float)) else str(x))
        print(f"{r['name']:<28} {r['sigma_dir']:>10.3e} "
              f"{s['plateau_D2']:>11.3e} {fmt(s['h_D2_deviate']):>11} "
              f"{r['falsifier_floor']:>9.2e} {s['fd_ad_min']:>9.2e} "
              f"{fmt(s['h_fd_ad_deviate']):>11}")
    print("\n[t3] done.")


if __name__ == "__main__":
    main()
