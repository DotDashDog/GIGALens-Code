#!/usr/bin/env python
"""D-fix1 follow-up: WHERE does the float32-forward jitter live — the ill-conditioned
normal-equations solve, or the basis/conv?

Builds the real vela n_max=25 lstsq model + bootstrap anchor, then measures the
likelihood-logp jitter along random rays through the frozen anchor in three precision modes:
  f32all   : float32 basis + float32 gram/solve + float32 reduction   (= current mixed forward)
  f64solve : float32 basis + float64 gram/solve + float64 reduction   (CHEAP candidate)
  f64all   : float64 basis + float64 gram/solve + float64 reduction   (proven fix)

If f64solve kills the jitter (~ f64all) -> float64 solve-only is the near-free fix (heavy
basis/conv stay float32). If f64solve ~ f32all -> basis/conv float32 is the dominant noise
source -> full float64 forward is required.

Replication is validated against the real simulator.lstsq_simulate at the f32 and f64 endpoints
before any conclusion is drawn.

Card. Hypothesis: the gram (cond ~1e16 at n_max=25) is the float32-killer; float64 solve removes
the jitter. Prediction: f64solve jitter ~ f64all (<< dE_target 0.092), >= ~1e3x below f32all.
Falsifier: f64solve jitter ~ f32all (basis/conv float32 dominates) -> solve-only insufficient.
"""
import os, sys, argparse, json
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gigalens.jax.simulator as gsim
from objax.functional import average_pool_2d

TWO_PI = 2.0 * np.pi


def build_basis(lens_sim, x, dtype):
    """Replicate lstsq_simulate's basis build (gigalens/jax/simulator.py:172-195) at `dtype`."""
    imgX = lens_sim.img_X.astype(dtype)
    imgY = lens_sim.img_Y.astype(dtype)

    def cast(p):
        return {k: jnp.asarray(v).astype(dtype) for k, v in p.items()}

    lens_params = [cast(p) for p in x[0]]
    ll_params = [cast(p) for p in x[1]]
    sl_params = [cast(p) for p in x[2]]

    bx, by = imgX, imgY
    for lens, p in zip(lens_sim.phys_model.lenses, lens_params):
        fx, fy = lens.deriv(imgX, imgY, **p)
        bx, by = bx - fx, by - fy

    img = jnp.zeros((0, *imgX.shape), dtype)
    for lm, p in zip(lens_sim.phys_model.lens_light, ll_params):
        img = jnp.concatenate((img, lm.light(imgX, imgY, **p).astype(dtype)), axis=0)
    for lm, p in zip(lens_sim.phys_model.source_light, sl_params):
        img = jnp.concatenate((img, lm.light(bx, by, **p).astype(dtype)), axis=0)

    img = jnp.nan_to_num(img)
    img = jnp.transpose(img, (3, 0, 1, 2))  # bs, ncomp, h, w
    if lens_sim.flat_kernel is not None:
        ret = gsim._shared_kernel_component_conv(img, lens_sim.flat_kernel.astype(dtype))
    else:
        ret = img
    if lens_sim.supersample != 1:
        ret = average_pool_2d(ret, size=(lens_sim.supersample, lens_sim.supersample),
                              padding="SAME")
    ret = jnp.transpose(ret, (0, 2, 3, 1))  # bs, h, w, ncomp
    return ret


def reconstruct(ret, obs, err, solve_dtype):
    """Replicate _weighted_lstsq_reconstruct with the gram/solve forced to solve_dtype."""
    W = (1.0 / err.astype(solve_dtype))[..., jnp.newaxis]
    Y = jnp.reshape(obs.astype(solve_dtype) * jnp.squeeze(W), (1, -1, 1))
    X = jnp.reshape(ret.astype(solve_dtype) * W, (ret.shape[0], -1, ret.shape[-1]))
    Xt = jnp.transpose(X, (0, 2, 1))
    coeffs = gsim._solve_normal_eq_with_fallback(Xt @ X, Xt @ Y)[..., 0]
    recon = jnp.sum(ret.astype(solve_dtype) * coeffs[:, jnp.newaxis, jnp.newaxis, :], axis=-1)
    return jnp.squeeze(recon)


def log_like(recon, obs, err, dtype):
    resid = (recon.astype(dtype) - obs.astype(dtype)) / err.astype(dtype)
    return jnp.sum(-0.5 * resid ** 2 - jnp.log(err.astype(dtype)) - 0.5 * np.log(TWO_PI))


def make_logp(lens_sim, bij, obs, err, mode):
    f32, f64 = jnp.float32, jnp.float64

    def logp(z):
        x = bij.forward(list(z.T))  # mirror BackwardProbModel.log_prob: z -> list of per-param leaves
        if mode == "f32all":
            ret = build_basis(lens_sim, x, f32)
            recon = reconstruct(ret, obs, err, f32)
            return log_like(recon, obs, err, f32)
        if mode == "f64solve":
            ret = build_basis(lens_sim, x, f32)
            recon = reconstruct(ret, obs, err, f64)
            return log_like(recon, obs, err, f64)
        if mode == "f64all":
            ret = build_basis(lens_sim, x, f64)
            recon = reconstruct(ret, obs, err, f64)
            return log_like(recon, obs, err, f64)
        raise ValueError(mode)
    return jax.jit(logp)


def jitter_on_ray(logp_fn, z0, d, window, n):
    ts = jnp.linspace(-window, window, n)
    zs = z0[None, :] + ts[:, None] * d[None, :]
    vals = jax.lax.map(logp_fn, zs)
    vals = np.asarray(vals, dtype=np.float64)
    t = np.asarray(ts, dtype=np.float64)
    good = np.isfinite(vals)
    if good.sum() < 10:
        return dict(jitter=np.nan, span=np.nan, n_unique=0, vals=vals, t=t)
    # smooth local model: degree-4 poly; residual RMS = jitter amplitude
    c = np.polyfit(t[good], vals[good], 4)
    resid = vals[good] - np.polyval(c, t[good])
    return dict(
        jitter=float(np.sqrt(np.mean(resid ** 2))),
        span=float(np.nanmax(vals) - np.nanmin(vals)),
        n_unique=int(np.unique(np.round(vals[good], 6)).size),
        vals=vals, t=t,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-max", type=int, default=25)
    ap.add_argument("--windows", type=float, nargs="+", default=[1e-1, 1e-2, 1e-3, 1e-4])
    ap.add_argument("--n-ray", type=int, default=400)
    ap.add_argument("--n-dirs", type=int, default=3)
    ap.add_argument("--map-samples", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str,
                    default="diagnosis_2026-06/fix1/solve_probe")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[probe] jax {jax.__version__} x64={jax.config.jax_enable_x64} dev={jax.devices()}",
          flush=True)

    # ---- build the real model + bootstrap anchor (mirrors repro_vela_shapelets.py) ----
    from gigalens_research.simtests.system import from_vela_dir
    from gigalens_research.simtests.experiments import vela_shapelets  # noqa: F401  (registration)
    from gigalens_research.simtests.registry import get_inference_builder
    from gigalens_research.simtests.pipelines import VelaBootstrapQzStage
    from gigalens_research.inference_utils.pipeline import InferenceContext
    import gigalens.jax.simulator as sim

    home = os.path.expanduser("~")
    data_dir = os.path.join(home, "GIGALens-Code/data")
    system_name = "vela01_cam12_rep03_a0.500_f814w"
    system = from_vela_dir(
        system_dir=os.path.join(data_dir, "vela_sim_systems", system_name),
        source_dir=os.path.join(data_dir, "vela_sources", system_name),
        system_id="vela01_cam12_rep03",
        delta_pix=0.03, num_pix=200, supersample=1,
        background_rms=0.002, exp_time=2000.0,
    )
    model_seq = get_inference_builder("epl_shear_sersic_shapelets")(system, n_max=args.n_max)
    print(f"[probe] model built n_max={args.n_max} system={system.system_id}", flush=True)

    bootstrap = VelaBootstrapQzStage(system=system, n_max=args.n_max,
                                     map_num_steps=200, map_n_samples=args.map_samples,
                                     diag_scale=1e-8)
    ctx = InferenceContext.from_modelling_sequence(model_seq)
    res = bootstrap.run(ctx, {}, seed=args.seed)
    qz = bootstrap.derive_artifacts(res.arrays)["qz"]
    z0 = jnp.asarray(qz.mean()).astype(jnp.float64)
    print(f"[probe] anchor z0[:4]={np.asarray(z0)[:4]} dim={z0.shape[-1]}", flush=True)

    lens_sim = sim.LensSimulator(model_seq.phys_model, model_seq.sim_config, bs=1)
    pm = model_seq.prob_model
    obs = jnp.asarray(pm.observed_image)
    err = jnp.asarray(pm.err_map)
    bij = pm.bij

    logps = {m: make_logp(lens_sim, bij, obs, err, m)
             for m in ["f32all", "f64solve", "f64all"]}

    # ---- replication validation against the real simulator at the anchor ----
    x0_list = bij.forward(list(z0.T))
    x0_f32 = jax.tree_util.tree_map(lambda a: jnp.asarray(a).astype(jnp.float32), x0_list)
    x0_f64 = x0_list
    recon_real_f32 = jnp.squeeze(lens_sim.lstsq_simulate(x0_f32, obs, err)[0])
    recon_real_f64 = jnp.squeeze(lens_sim.lstsq_simulate(x0_f64, obs, err)[0])
    recon_my_f32 = reconstruct(build_basis(lens_sim, x0_f32, jnp.float32), obs, err, jnp.float32)
    recon_my_f64 = reconstruct(build_basis(lens_sim, x0_f64, jnp.float64), obs, err, jnp.float64)
    val = {
        "recon_f32_maxabsdiff": float(jnp.max(jnp.abs(recon_my_f32 - recon_real_f32))),
        "recon_f32_scale": float(jnp.max(jnp.abs(recon_real_f32))),
        "recon_f64_maxabsdiff": float(jnp.max(jnp.abs(recon_my_f64 - recon_real_f64))),
        "recon_f64_scale": float(jnp.max(jnp.abs(recon_real_f64))),
    }
    print(f"[probe] REPLICATION CHECK: f32 maxdiff={val['recon_f32_maxabsdiff']:.3e} "
          f"(scale {val['recon_f32_scale']:.3e}); f64 maxdiff={val['recon_f64_maxabsdiff']:.3e} "
          f"(scale {val['recon_f64_scale']:.3e})", flush=True)

    dE_target = float(np.sqrt(z0.shape[-1] * 5e-4))
    print(f"[probe] dE_target (xi=1) = {dE_target:.4f}", flush=True)

    rng = np.random.default_rng(args.seed)
    results = {"validation": val, "dE_target": dE_target, "rays": {}}
    fig, axes = plt.subplots(len(args.windows), 1, figsize=(9, 3 * len(args.windows)))
    if len(args.windows) == 1:
        axes = [axes]

    for wi, window in enumerate(args.windows):
        per_mode = {m: [] for m in logps}
        sample_curves = {}
        for di in range(args.n_dirs):
            d = rng.standard_normal(z0.shape[-1])
            d = jnp.asarray(d / np.linalg.norm(d), dtype=jnp.float64)
            for m, fn in logps.items():
                r = jitter_on_ray(fn, z0, d, window, args.n_ray)
                per_mode[m].append(r["jitter"])
                if di == 0:
                    sample_curves[m] = r
        summary = {m: float(np.nanmedian(per_mode[m])) for m in logps}
        results["rays"][f"window_{window:.0e}"] = {
            "jitter_median": summary,
            "n_unique_dir0": {m: sample_curves[m]["n_unique"] for m in logps},
            "span_dir0": {m: sample_curves[m]["span"] for m in logps},
        }
        print(f"\n[probe] window={window:.0e}  jitter(median over {args.n_dirs} dirs):", flush=True)
        for m in logps:
            print(f"    {m:9s} jitter={summary[m]:.3e}  "
                  f"({summary[m]/dE_target:.2e} x dE_target)  "
                  f"n_unique={sample_curves[m]['n_unique']}", flush=True)

        ax = axes[wi]
        for m in logps:
            c = sample_curves[m]
            v = c["vals"] - np.nanmedian(c["vals"])
            ax.plot(c["t"], v, lw=0.8, label=f"{m} (jit={summary[m]:.1e})")
        ax.set_title(f"logp along ray (dir 0), window={window:.0e}")
        ax.set_xlabel("t"); ax.set_ylabel("logp - median"); ax.legend(fontsize=7)
    fig.tight_layout()
    pngp = os.path.join(args.out_dir, "solve_probe_rays.png")
    fig.savefig(pngp, dpi=120)
    print(f"\n[probe] wrote {pngp}", flush=True)

    with open(os.path.join(args.out_dir, "solve_probe_summary.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"[probe] wrote {os.path.join(args.out_dir, 'solve_probe_summary.json')}", flush=True)


if __name__ == "__main__":
    main()
