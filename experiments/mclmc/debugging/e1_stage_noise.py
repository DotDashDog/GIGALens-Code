"""E1 — stage-wise float32 noise attribution in the lstsq logp pipeline.

DIAGNOSIS ONLY. Replicates ``BackwardProbModel.log_prob`` stage-by-stage
OUTSIDE the library (importing its building blocks, NOT editing them) so we can
extract intermediates and compare float32 vs float64 at each stage.

Pipeline stages (lstsq path; see gigalens/jax/model.py::BackwardProbModel and
gigalens/jax/simulator.py::lstsq_simulate):
  s0_x            : bij.forward(z)          (bijector -> physical params)
  s1_basis        : build_basis -> img      (raw shapelet/sersic basis tensor)
  s2_design       : conv_pool -> X          (the design matrix A, weighted)
  s3_gram         : Xt @ X (+regularize)    (normal-equations gram)
  s4_rhs          : Xt @ Y                  (normal-equations RHS)
  s5_coeffs       : solve(gram, rhs)        (the SOLVE -- hypothesised culprit)
  s6_model_image  : sum(X * coeffs)         (reconstructed model image)
  s7_chi2         : mean(((im-obs)/err)^2)  (reduced chi^2 summary)
  s8_logp         : log_like + log_prior    (final scalar)

For a given precision we evaluate ALL stages at an anchor and at K perturbations
z+delta*u, dump every intermediate to an .npz.  A separate analysis pass loads
the f32 and f64 dumps and computes per-stage relative discrepancy
||f32-f64||/||f64|| (logp also reported as absolute error).

Usage:
  # dump intermediates (run once per precision; x64 flag is process-global):
  python e1_stage_noise.py dump --anchor-class run_a_late --n-max 25 --precision float32
  python e1_stage_noise.py dump --anchor-class run_a_late --n-max 25 --precision float64
  # ... repeat for bootstrap, run_b_late, prior_far ...
  # then analyze + plot:
  python e1_stage_noise.py analyze --out-dir <e1 dir>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

home = os.path.expanduser("~/")
for _p in [
    os.path.join(home, "sidecar_jax_upgrade"),
    os.path.join(home, "gigalens/src"),
    os.path.join(home, "GIGALens-Code/src"),
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

DIM = 17
DESIRED_ENERGY_VAR = 5e-4
TARGET_DE_RMS = (DIM * DESIRED_ENERGY_VAR) ** 0.5  # ~0.092
ULP_LOGP_SCALE = 0.016  # float32 ulp of |logp|~1.5e5 (from D3/REPORT)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIAG_DIR = os.path.join(SCRIPT_DIR, "diagnosis_2026-06")
E1_DIR = os.path.join(DIAG_DIR, "e1")

STAGE_ORDER = [
    "s0_x", "s1_basis", "s2_design", "s3_gram", "s4_rhs",
    "s5_coeffs", "s6_model_image", "s7_chi2", "s8_logp",
]


# ---------------------------------------------------------------------------
# Model construction (verbatim-style reuse of d3_eevpd_vs_eps.py)
# ---------------------------------------------------------------------------
def build_model(n_max):
    import gigalens.jax.simulator as sim
    from gigalens_research.simtests.system import from_vela_dir
    from gigalens_research.simtests.experiments import vela_shapelets  # noqa: F401
    from gigalens_research.simtests.registry import get_inference_builder

    data_dir = os.path.join(home, "GIGALens-Code/data")
    system_name = "vela01_cam12_rep03_a0.500_f814w"
    system = from_vela_dir(
        system_dir=os.path.join(data_dir, "vela_sim_systems", system_name),
        source_dir=os.path.join(data_dir, "vela_sources", system_name),
        system_id="vela01_cam12_rep03",
        delta_pix=0.03, num_pix=200, supersample=1,
        background_rms=0.002, exp_time=2000.0,
    )
    model_seq = get_inference_builder("epl_shear_sersic_shapelets")(system, n_max=n_max)
    lens_sim = sim.LensSimulator(model_seq.phys_model, model_seq.sim_config, bs=1)
    return model_seq, lens_sim


# ---------------------------------------------------------------------------
# Stage-wise replication of BackwardProbModel.log_prob.
# Returns a dict of intermediates (jax arrays) plus the library logp for cross-check.
# ---------------------------------------------------------------------------
def make_stage_fn(model_seq, lens_sim):
    import jax
    import jax.numpy as jnp
    from gigalens.jax.simulator import (
        _shared_kernel_component_conv, _regularize_gram,
    )

    pm = model_seq.prob_model           # BackwardProbModel
    s = lens_sim

    def stages(z):
        """z: (1,17). Returns dict of stage intermediates + scalar logp."""
        out = {}
        # ----- s0: bijector forward -----
        zl = list(z.T)
        x = pm.bij.forward(zl)
        # x is nested list of dict params; capture a flat numeric summary by
        # concatenating all leaf arrays.
        leaves = jax.tree_util.tree_leaves(x)
        out["s0_x"] = jnp.concatenate([jnp.ravel(l).astype(jnp.float64) for l in leaves])

        # ----- replicate lstsq_simulate internals -----
        lens_params = x[0]
        if len(s.phys_model.lens_light) > 0:
            lens_light_params, source_light_params = x[1], x[2]
        else:
            source_light_params = x[1]
            lens_light_params = []
        beta_x, beta_y = s._beta(lens_params)

        # ----- s1: build basis -----
        img = jnp.zeros((0, *s.img_X.shape))
        for lightModel, p in zip(s.phys_model.lens_light, lens_light_params):
            img = jnp.concatenate((img, lightModel.light(s.img_X, s.img_Y, **p)), axis=0)
        for lightModel, p in zip(s.phys_model.source_light, source_light_params):
            img = jnp.concatenate((img, lightModel.light(beta_x, beta_y, **p)), axis=0)
        img = jnp.nan_to_num(img)
        img = jnp.transpose(img, (3, 0, 1, 2))  # bs, n_comp, h, w
        out["s1_basis"] = img

        # ----- s2: conv/pool -> design tensor ret (bs,h,w,n_comp) -----
        ret = (_shared_kernel_component_conv(img, s.flat_kernel)
               if s.flat_kernel is not None else img)
        from objax.functional import average_pool_2d
        ret = (average_pool_2d(ret, size=(s.supersample, s.supersample), padding="SAME")
               if s.supersample != 1 else ret)
        ret = jnp.transpose(ret, (0, 2, 3, 1))  # bs, h, w, n_comp

        # weighted design matrix X and RHS Y (mirrors _weighted_lstsq_reconstruct)
        observed_image = pm.observed_image
        err_map = pm.err_map
        W = (1 / err_map)[..., jnp.newaxis]
        Y = jnp.reshape(observed_image * jnp.squeeze(W), (1, -1, 1))
        X = jnp.reshape((ret * W), (ret.shape[0], -1, ret.shape[-1]))
        out["s2_design"] = X  # (1, npix, n_comp)

        Xt = jnp.transpose(X, (0, 2, 1))
        gram = Xt @ X
        out["s3_gram"] = gram                  # (1, n_comp, n_comp)
        rhs = Xt @ Y
        out["s4_rhs"] = rhs                    # (1, n_comp, 1)

        # ----- s5: the SOLVE -----
        gram_reg = _regularize_gram(gram[0])
        coeffs = jnp.linalg.solve(gram_reg, rhs[0])[jnp.newaxis, ...][..., 0]
        out["s5_coeffs"] = coeffs              # (1, n_comp)

        # ----- s6: recon model image -----
        recon = jnp.sum(ret * coeffs[:, jnp.newaxis, jnp.newaxis, :], axis=-1)
        im_sim = jnp.squeeze(recon)
        out["s6_model_image"] = im_sim

        # ----- s7: chi^2 summary -----
        chi2 = jnp.mean(((im_sim - observed_image) / err_map) ** 2)
        out["s7_chi2"] = chi2

        # ----- s8: logp (log_like + log_prior) -----
        log_like = pm.observed_dist.log_prob(im_sim)
        log_prior = pm.prior.log_prob(x) + pm.bij.forward_log_det_jacobian(zl)
        logp = log_like + log_prior
        out["s8_logp"] = logp
        return out

    def library_logp(z):
        return pm.log_prob(s, z)[0]

    return stages, library_logp


# ---------------------------------------------------------------------------
# Anchor loading (same provenance as D3)
# ---------------------------------------------------------------------------
def load_anchors(anchor_class, n_max, seed=0):
    run_a_dir = os.path.join(DIAG_DIR, "run_a_nmax25_ds1e-8_nb2000")
    run_b_dir = os.path.join(DIAG_DIR, "run_b_nmax10_ds1e-8_nb2000")
    pos_a = np.load(os.path.join(run_a_dir, "hist_position.npy"))  # (8,2500,17)
    pos_b = np.load(os.path.join(run_b_dir, "hist_position.npy"))

    if anchor_class == "bootstrap":
        # step-0 position proxy for qz.mean() (matches D3 --skip-bootstrap)
        anchors = (pos_a if n_max == 25 else pos_b)[[0], 0, :]
    elif anchor_class == "run_a_late":
        anchors = pos_a[[0, 2, 4, 6], -1, :]
    elif anchor_class == "run_b_late":
        anchors = pos_b[[0, 2, 4, 6], -1, :]
    elif anchor_class == "prior_far":
        # far-field controls: prior draws in z-space ~ N(0,1) (z is the
        # unconstrained / standardized space the sampler moves in).
        rng = np.random.default_rng(seed)
        anchors = rng.standard_normal((2, DIM))
    else:
        raise ValueError(anchor_class)
    return np.asarray(anchors, dtype=np.float64)


# ---------------------------------------------------------------------------
# DUMP: evaluate all stages at anchor + perturbations, save npz
# ---------------------------------------------------------------------------
def cmd_dump(args):
    if args.precision == "float64":
        import jax
        jax.config.update("jax_enable_x64", True)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_platform_name", "cpu")
    print(f"[E1] JAX {jax.__version__} devices={jax.devices()} precision={args.precision}", flush=True)

    model_seq, lens_sim = build_model(args.n_max)
    stages_fn, lib_logp = make_stage_fn(model_seq, lens_sim)
    print(f"[E1] model built n_max={args.n_max}", flush=True)

    anchors = load_anchors(args.anchor_class, args.n_max, seed=args.seed)
    print(f"[E1] anchor_class={args.anchor_class}  n_anchors={anchors.shape[0]}", flush=True)

    deltas = [1e-7, 1e-6, 1e-5]
    K = args.K
    rng = np.random.default_rng(args.seed + 1000)

    out_dir = args.out_dir or os.path.join(E1_DIR, "dumps")
    os.makedirs(out_dir, exist_ok=True)

    def to_z(zvec):
        # route through float64 numpy so active default dtype wins
        return jnp.asarray(np.asarray(zvec, dtype=np.float64)).reshape(1, DIM)

    stages_jit = jax.jit(stages_fn)
    lib_jit = jax.jit(lib_logp)

    # Compact summary: we DO NOT persist full stage arrays (design matrix X is
    # ~100 MB/eval). Instead, per stage per eval we store (a) the full float64
    # L2 norm and (b) a fixed PROJ_DIM random projection. The projection matrix
    # for a given flat length is seeded ONLY by that length, so float32 and
    # float64 runs build identical matrices => ||P(f32)-P(f64)||/||P(f64)|| is an
    # unbiased faithful estimator of the full relative discrepancy.
    PROJ_DIM = 64

    def get_proj(flat_len):
        rg = np.random.default_rng(987654321 + flat_len)  # deterministic in length
        return rg.standard_normal((flat_len, PROJ_DIM)) / np.sqrt(flat_len)

    proj_cache = {}

    def summarize(arr_np):
        a = arr_np.ravel().astype(np.float64)
        norm = float(np.linalg.norm(a))
        n = a.shape[0]
        if n not in proj_cache:
            proj_cache[n] = get_proj(n)
        proj = a @ proj_cache[n]  # (PROJ_DIM,)
        return norm, proj

    for ai, anchor in enumerate(anchors):
        t0 = time.perf_counter()
        u = rng.standard_normal((K, DIM))
        u = u / np.linalg.norm(u, axis=1, keepdims=True)  # unit vectors in z-space

        zlist = [anchor.copy()]  # index 0 = anchor itself
        meta = [("anchor", 0.0, -1)]
        for d in deltas:
            for k in range(K):
                zlist.append(anchor + d * u[k])
                meta.append((f"d{d:.0e}", d, k))
        zlist = np.stack(zlist, axis=0)

        per_stage_norm = {s: [] for s in STAGE_ORDER}
        per_stage_proj = {s: [] for s in STAGE_ORDER}
        lib_logps = []
        harness_logps = []
        for zi in range(zlist.shape[0]):
            zj = to_z(zlist[zi])
            res = stages_jit(zj)
            for s in STAGE_ORDER:
                nrm, prj = summarize(np.asarray(res[s], dtype=np.float64))
                per_stage_norm[s].append(nrm)
                per_stage_proj[s].append(prj)
            harness_logps.append(float(np.asarray(res["s8_logp"]).reshape(-1)[0]))
            lib_logps.append(float(np.asarray(lib_jit(zj)).reshape(-1)[0]))

        hl = np.array(harness_logps)
        ll = np.array(lib_logps)
        max_abs = float(np.max(np.abs(hl - ll)))
        max_rel = float(np.max(np.abs(hl - ll) / (np.abs(ll) + 1e-30)))
        print(f"[E1] anchor {ai}: harness-vs-library logp max_abs={max_abs:.3e} "
              f"max_rel={max_rel:.3e}  (anchor logp={ll[0]:.6g})", flush=True)

        norm_arrs = {s: np.array(per_stage_norm[s]) for s in STAGE_ORDER}      # (n_eval,)
        proj_arrs = {s: np.stack(per_stage_proj[s], 0) for s in STAGE_ORDER}   # (n_eval,PROJ_DIM)

        fname = os.path.join(
            out_dir,
            f"dump_{args.anchor_class}_nmax{args.n_max}_anchor{ai}_{args.precision}.npz",
        )
        np.savez(
            fname,
            anchor=anchor, u=u, deltas=np.array(deltas), K=K, proj_dim=PROJ_DIM,
            meta_label=np.array([m[0] for m in meta]),
            meta_delta=np.array([m[1] for m in meta]),
            meta_k=np.array([m[2] for m in meta]),
            lib_logp=ll, harness_logp=hl,
            harness_vs_lib_max_abs=max_abs, harness_vs_lib_max_rel=max_rel,
            **{f"norm_{s}": norm_arrs[s] for s in STAGE_ORDER},
            **{f"proj_{s}": proj_arrs[s] for s in STAGE_ORDER},
        )
        print(f"[E1] anchor {ai} dumped -> {fname}  ({time.perf_counter()-t0:.1f}s)", flush=True)

    print("[E1] dump complete", flush=True)
    return 0


# ---------------------------------------------------------------------------
# ANALYZE: load f32/f64 dumps, compute per-stage relative discrepancy, plot
# ---------------------------------------------------------------------------
def cmd_analyze(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dump_dir = os.path.join(args.out_dir or E1_DIR, "dumps") if not args.dump_dir else args.dump_dir
    out_dir = args.out_dir or E1_DIR
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(dump_dir) if f.startswith("dump_") and f.endswith("_float32.npz"))
    summary = {"stages": STAGE_ORDER, "anchors": {}}

    # group: per (anchor_class, n_max, anchor_idx)
    waterfall = {}  # key -> per-stage relative error array (over perturbations, excl anchor row 0)
    for f32name in files:
        f64name = f32name.replace("_float32.npz", "_float64.npz")
        f64path = os.path.join(dump_dir, f64name)
        if not os.path.exists(f64path):
            print(f"[E1] WARN no float64 match for {f32name}; skip", flush=True)
            continue
        d32 = np.load(os.path.join(dump_dir, f32name), allow_pickle=True)
        d64 = np.load(f64path, allow_pickle=True)
        key = f32name[len("dump_"):-len("_float32.npz")]

        # harness validity check
        hl_abs32 = float(d32["harness_vs_lib_max_abs"])
        hl_abs64 = float(d64["harness_vs_lib_max_abs"])

        meta_delta = d32["meta_delta"]
        # use perturbation rows only (exclude anchor row 0) for the noise stat,
        # over ALL deltas pooled (small deltas -> f64 locally smooth).
        pert_mask = np.arange(len(meta_delta)) > 0

        per_stage_rel = {}
        per_stage_abs = {}
        for s in STAGE_ORDER:
            # projection-based estimator of ||f32-f64||/||f64||. The same seeded
            # projection is used in both precisions, so differences are faithful.
            p32 = d32[f"proj_{s}"]  # (n_eval, PROJ_DIM)
            p64 = d64[f"proj_{s}"]
            num_proj = np.linalg.norm(p32 - p64, axis=1)
            den_proj = np.linalg.norm(p64, axis=1) + 1e-300
            rel = num_proj / den_proj
            # absolute error in true units: use full L2 norm * rel (rel from proj)
            full_norm64 = d64[f"norm_{s}"]
            abs_err = rel * full_norm64
            per_stage_rel[s] = float(np.median(rel[pert_mask]))
            per_stage_abs[s] = float(np.median(abs_err[pert_mask]))

        # logp absolute error: logp is scalar, so proj==value; use direct norms.
        # norm_s8_logp == |logp|; abs err in logp = |f32_logp - f64_logp|, but we
        # only have norms here. Recover via projection diff which for a scalar is
        # the signed value times the (1-d) projection — use the harness logp arrays.
        lp32 = d32["harness_logp"]; lp64 = d64["harness_logp"]
        logp_abs = float(np.median(np.abs(lp32 - lp64)[pert_mask]))
        per_stage_abs["s8_logp"] = logp_abs
        waterfall[key] = per_stage_rel
        summary["anchors"][key] = {
            "per_stage_rel_median": per_stage_rel,
            "per_stage_abs_median": per_stage_abs,
            "logp_abs_err_median": logp_abs,
            "logp_abs_err_vs_ulp": logp_abs / ULP_LOGP_SCALE,
            "logp_abs_err_vs_dE_target": logp_abs / TARGET_DE_RMS,
            "anchor_logp_f64": float(d64["lib_logp"][0]),
            "harness_vs_lib_max_abs_f32": hl_abs32,
            "harness_vs_lib_max_abs_f64": hl_abs64,
        }

    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[E1] summary -> {os.path.join(out_dir, 'summary.json')}", flush=True)

    # ---- aggregate per anchor CLASS (median across anchors of that class) ----
    # key looks like "<class>_nmax<NN>_anchor<i>"; class+nmax is the group.
    import re
    groups = {}
    for key, rels in waterfall.items():
        m = re.match(r"(.+_nmax\d+)_anchor\d+", key)
        gkey = m.group(1) if m else key
        groups.setdefault(gkey, []).append(rels)
    group_med = {}
    for gkey, lst in groups.items():
        group_med[gkey] = {s: float(np.median([r[s] for r in lst])) for s in STAGE_ORDER}
    summary["class_median_rel"] = group_med
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- Plot 1: noise waterfall (per anchor class, median over anchors) ----
    fig, ax = plt.subplots(figsize=(11, 6))
    xs = np.arange(len(STAGE_ORDER))
    for gkey, rels in sorted(group_med.items()):
        ys = [max(rels[s], 1e-20) for s in STAGE_ORDER]
        ax.plot(xs, ys, "o-", lw=2, ms=7, label=gkey, alpha=0.85)
    # faint per-anchor lines underneath
    for key, rels in sorted(waterfall.items()):
        ys = [max(rels[s], 1e-20) for s in STAGE_ORDER]
        ax.plot(xs, ys, "-", lw=0.5, color="gray", alpha=0.25)
    # mark the solve stage
    solve_idx = STAGE_ORDER.index("s5_coeffs")
    ax.axvline(solve_idx, color="k", ls="--", alpha=0.4)
    ax.text(solve_idx, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
            " SOLVE", va="top", fontsize=9)
    ax.axhline(np.finfo(np.float32).eps, color="gray", ls=":", alpha=0.6, label="f32 eps")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(STAGE_ORDER, rotation=35, ha="right")
    ax.set_ylabel("median relative f32-vs-f64 error  ||f32-f64||/||f64||")
    ax.set_title("E1 noise waterfall: per-stage float32 discrepancy (vs float64)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, which="both", alpha=0.3)
    p1 = os.path.join(out_dir, "noise_waterfall.png")
    fig.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[E1] plot -> {p1}", flush=True)

    print("[E1] analyze complete", flush=True)
    return 0


# ---------------------------------------------------------------------------
# RAY: logp along a ray through an anchor, f32 vs f64 (run per precision)
# ---------------------------------------------------------------------------
def cmd_ray(args):
    if args.precision == "float64":
        import jax
        jax.config.update("jax_enable_x64", True)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_platform_name", "cpu")

    model_seq, lens_sim = build_model(args.n_max)
    _, lib_logp = make_stage_fn(model_seq, lens_sim)
    lib_jit = jax.jit(lib_logp)

    anchors = load_anchors(args.anchor_class, args.n_max, seed=args.seed)
    anchor = anchors[args.anchor_idx]
    rng = np.random.default_rng(args.seed + 7)
    u = rng.standard_normal(DIM); u = u / np.linalg.norm(u)

    # ray scale in z-space: span +-1e-5 with fine resolution to expose raggedness
    ts = np.linspace(-args.ray_span, args.ray_span, args.ray_points)
    logps = []
    for t in ts:
        zj = jnp.asarray((anchor + t * u).astype(np.float64)).reshape(1, DIM)
        logps.append(float(np.asarray(lib_jit(zj))))
    logps = np.array(logps)

    out_dir = args.out_dir or E1_DIR
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"ray_{args.anchor_class}_anchor{args.anchor_idx}_{args.precision}.npz")
    np.savez(fname, ts=ts, logp=logps, anchor=anchor, u=u, span=args.ray_span)
    print(f"[E1] ray dumped -> {fname}  (logp range {logps.min():.6g}..{logps.max():.6g})", flush=True)
    return 0


def cmd_ray_plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = args.out_dir or E1_DIR
    # find ray pairs
    files = sorted(f for f in os.listdir(out_dir) if f.startswith("ray_") and f.endswith("_float32.npz"))
    for f32 in files:
        f64 = f32.replace("_float32.npz", "_float64.npz")
        if not os.path.exists(os.path.join(out_dir, f64)):
            continue
        d32 = np.load(os.path.join(out_dir, f32))
        d64 = np.load(os.path.join(out_dir, f64))
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        axes[0].plot(d64["ts"], d64["logp"], "g-", lw=1.2, label="float64")
        axes[0].plot(d32["ts"], d32["logp"], "b.-", ms=3, lw=0.6, alpha=0.8, label="float32")
        axes[0].set_ylabel("logp"); axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[0].set_title(f"E1 logp along z-ray: {f32[len('ray_'):-len('_float32.npz')]}")
        resid = d32["logp"] - np.interp(d32["ts"], d64["ts"], d64["logp"])
        axes[1].plot(d32["ts"], resid, "r.-", ms=3, lw=0.6)
        axes[1].axhline(0, color="k", lw=0.5)
        axes[1].set_ylabel("f32 - f64"); axes[1].set_xlabel("displacement t along unit ray (z-space)")
        axes[1].grid(alpha=0.3)
        p = os.path.join(out_dir, f32.replace("_float32.npz", "_overlay.png"))
        fig.savefig(p, dpi=120, bbox_inches="tight"); plt.close(fig)
        print(f"[E1] ray plot -> {p}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    pd = sub.add_parser("dump")
    pd.add_argument("--anchor-class", required=True,
                    choices=["bootstrap", "run_a_late", "run_b_late", "prior_far"])
    pd.add_argument("--n-max", type=int, default=25)
    pd.add_argument("--precision", choices=["float32", "float64"], required=True)
    pd.add_argument("--K", type=int, default=16)
    pd.add_argument("--seed", type=int, default=42)
    pd.add_argument("--out-dir", default=None)
    pd.set_defaults(func=cmd_dump)

    pa = sub.add_parser("analyze")
    pa.add_argument("--out-dir", default=None)
    pa.add_argument("--dump-dir", default=None)
    pa.set_defaults(func=cmd_analyze)

    pr = sub.add_parser("ray")
    pr.add_argument("--anchor-class", required=True,
                    choices=["bootstrap", "run_a_late", "run_b_late", "prior_far"])
    pr.add_argument("--anchor-idx", type=int, default=0)
    pr.add_argument("--n-max", type=int, default=25)
    pr.add_argument("--precision", choices=["float32", "float64"], required=True)
    pr.add_argument("--ray-span", type=float, default=1e-5)
    pr.add_argument("--ray-points", type=int, default=801)
    pr.add_argument("--seed", type=int, default=42)
    pr.add_argument("--out-dir", default=None)
    pr.set_defaults(func=cmd_ray)

    prp = sub.add_parser("ray-plot")
    prp.add_argument("--out-dir", default=None)
    prp.set_defaults(func=cmd_ray_plot)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
