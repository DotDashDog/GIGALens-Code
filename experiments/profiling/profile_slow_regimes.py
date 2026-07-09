#!/usr/bin/env python3
"""Phase-2 profiling: the scene-API lstsq likelihood gradient at the SLOW regimes.

Extends gigalens/wip/profile_scene_likelihood.py (Phase 0/1, June 2026) to the
regimes the user actually reports as slow: high n_max (30/40), supersample 4,
300-px cutouts, 2 datasets, MAP-scale batch. Pre-registered as the
"Slow-regime profiling matrix (Phase 2)" design checkpoint in
GIGALens-Code/docs/logs/compute-profiling.md (2026-07-09): hypotheses H-A..H-E.

Measurement harness only — asserts nothing about correctness.
Run inside the canonical Shifter container (JAX 0.10) on a GPU node:

  shifter --module=gpu,nccl-plugin --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 \
    /usr/bin/python3 experiments/profiling/profile_slow_regimes.py --part all

The model family matches the June harness (EPL+Shear mass, lstsq Sersic lens
light, lstsq Shapelets source, 21-px Gaussian PSF, bg_rms=0.01/exp_time=1000,
conv_precision=float32, likelihood float64) EXCEPT EPL niter: the matrix runs at
niter=18 (user directive 2026-07-09 — representative of production use), while
the June harness ran EPL(50). June comparability is provided by the H-C niter
sweep's (200px, ss2, n_max=15, niter=50) cell.
"""
import os
# float64 pipeline + be a good citizen on a shared GPU (no 75% preallocation).
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse
import json
import statistics
import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import random as jr

import tensorflow_probability.substrates.jax as tfp

tfd = tfp.distributions

from gigalens.simulator import SimulatorConfig
from gigalens.jax.profiles.mass.epl import EPL
from gigalens.jax.profiles.mass.shear import Shear
from gigalens.jax.profiles.light.sersic import SersicEllipse
from gigalens.jax.profiles.light.shapelets import Shapelets
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.scene_prob_model import Dataset, ProbModel
from gigalens.jax.scene_simulator import SceneSimulator


# ----------------------------------------------------------------------------- model
def gaussian_psf(npix=21, fwhm_pix=3.0):
    assert npix % 2 == 1
    sigma = fwhm_pix / 2.3548
    ax = np.arange(npix) - (npix - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).astype(np.float64)


def build_problem(num_pix=200, supersample=2, n_max=15, delta_pix=0.05,
                  niter=18, n_datasets=1, seed=0):
    """Same representative fit as the June harness, with EPL niter and an
    optional identical-grid second dataset as knobs (H-C / H-D)."""
    epl = Component(EPL(niter), dict(
        theta_E=tfd.LogNormal(np.log(1.1), 0.3), gamma=tfd.Normal(2.05, 0.1),
        e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
        center_x=tfd.Normal(0.0, 0.05), center_y=tfd.Normal(0.0, 0.05)))
    shear = Component(Shear(), dict(gamma1=tfd.Normal(0.0, 0.05),
                                    gamma2=tfd.Normal(0.0, 0.05)))
    lens_light = Component(SersicEllipse(use_lstsq=True), dict(
        R_sersic=tfd.LogNormal(np.log(1.2), 0.3), n_sersic=tfd.Uniform(0.5, 6.0),
        e1=tfd.Normal(0.0, 0.1), e2=tfd.Normal(0.0, 0.1),
        center_x=tfd.Normal(0.0, 0.05), center_y=tfd.Normal(0.0, 0.05)))
    source = Component(Shapelets(n_max=n_max, use_lstsq=True), dict(
        beta=tfd.LogNormal(np.log(0.1), 0.3),
        center_x=tfd.Normal(0.05, 0.1), center_y=tfd.Normal(-0.04, 0.1)))

    model = LensModel([
        Plane(mass=[epl, shear], light=[lens_light]),
        Plane(deflection_ratio=1.0, light=[source]),
    ])

    cfg = SimulatorConfig(delta_pix=delta_pix, num_pix=num_pix, supersample=supersample,
                          kernel=gaussian_psf(), likelihood_precision="float64",
                          conv_precision="float32", basis_precision=None)

    z = model.prior.sample(seed=jr.PRNGKey(seed))
    z_vec = jnp.stack(model.bijector.inverse(z))
    bg_rms, exp_time = 0.01, 1000.0
    sim0 = SceneSimulator(model, cfg)
    x = model.bijector.forward(list(z_vec.T))
    params = model.to_params(x)
    stacked = np.asarray(sim0.lstsq_simulate(
        params, jnp.zeros((num_pix, num_pix)), jnp.ones((num_pix, num_pix)),
        return_stacked=True))
    clean = stacked.sum(axis=-1).squeeze().astype(np.float64)
    err = np.sqrt(bg_rms ** 2 + np.clip(clean, 0, None) / exp_time)
    # Each dataset gets an INDEPENDENT noise realization: byte-identical twin
    # datasets let XLA CSE merge the per-dataset graphs (first H-D attempt showed
    # 2-dataset min time == 1-dataset min time), which real multi-band data cannot.
    datasets = [
        Dataset(clean + np.random.default_rng(seed + 1000 * k).normal(
                    size=clean.shape) * err,
                cfg, background_rms=bg_rms, exp_time=exp_time, sees="all")
        for k in range(n_datasets)]
    prob = ProbModel(model, datasets if n_datasets > 1 else datasets[0],
                     mode="lstsq")
    return model, prob, cfg, z_vec, params


# ----------------------------------------------------------------------------- timing
def time_fn(fn, *args, n_warmup=3, n_iter=30):
    """(median_ms, min_ms) per call, device-synchronized."""
    out = fn(*args)
    jax.block_until_ready(out)
    for _ in range(n_warmup - 1):
        jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), min(ts)


def xla_peak_mb(compiled):
    """Compile-time peak memory model (temp+output) for one executable.

    Used instead of device ``peak_bytes_in_use``, which is process-cumulative and
    therefore censored by whichever earlier cell peaked highest (grader fix,
    2026-07-09). Works even for cells that OOM at runtime."""
    try:
        ma = compiled.memory_analysis()
        return (ma.temp_size_in_bytes + ma.output_size_in_bytes) / 1e6
    except Exception:
        return float("nan")


# ----------------------------------------------------------------------------- stages
def stage_breakdown(num_pix, supersample, n_max, niter=18, n_datasets=1, n_iter=30):
    """Phase-1-style differenced stage decomposition of one bs=1 grad eval."""
    model, prob, cfg, z_vec, params = build_problem(
        num_pix=num_pix, supersample=supersample, n_max=n_max,
        niter=niter, n_datasets=n_datasets)
    sim = prob.simulators[0]
    ds = prob.datasets[0]
    img, err, mask = ds.image, ds.error_map, ds.mask

    f_stack_nd = jax.jit(lambda p: sim.lstsq_simulate(
        p, img, err, mask, return_stacked=True, no_deflection=True))
    f_stack = jax.jit(lambda p: sim.lstsq_simulate(
        p, img, err, mask, return_stacked=True))
    f_lstsq = jax.jit(lambda p: sim.lstsq_simulate(p, img, err, mask))
    f_fwd = jax.jit(lambda z: prob.log_prob(z)[0])
    f_grad = jax.jit(lambda z: jax.grad(lambda zz: prob.log_prob(zz)[0])(z))

    t = {}
    t["stack_nd"], t["stack_nd_min"] = time_fn(f_stack_nd, params, n_iter=n_iter)
    t["stack"], _ = time_fn(f_stack, params, n_iter=n_iter)
    t["lstsq"], _ = time_fn(f_lstsq, params, n_iter=n_iter)
    t["fwd"], _ = time_fn(f_fwd, z_vec, n_iter=n_iter)
    t["grad"], t["grad_min"] = time_fn(f_grad, z_vec, n_iter=n_iter)

    # Static cost/memory model for the grad (free; H-A cross-check).
    comp = f_grad.lower(z_vec).compile()
    ca = comp.cost_analysis()
    if isinstance(ca, (list, tuple)):
        ca = ca[0]
    flops = float(ca.get("flops", float("nan"))) if isinstance(ca, dict) else float("nan")
    gbytes = (float(ca.get("bytes accessed", float("nan"))) / 1e9
              if isinstance(ca, dict) else float("nan"))
    peak_mb = xla_peak_mb(comp)

    row = dict(num_pix=num_pix, ss=supersample, n_max=n_max, niter=niter,
               n_datasets=n_datasets, depth=sim.depth,
               basis_conv_pool=t["stack_nd"],
               deflection=t["stack"] - t["stack_nd"],
               gram_solve=t["lstsq"] - t["stack"],
               reduction=t["fwd"] - t["lstsq"],
               backward=t["grad"] - t["fwd"],
               fwd_total=t["fwd"], grad_total=t["grad"], grad_min=t["grad_min"],
               grad_gflop=flops / 1e9, grad_gb=gbytes, xla_peak_mb=peak_mb,
               contention=(t["grad"] - t["grad_min"]) / max(t["grad_min"], 1e-9))
    return row


def print_stage_row(r):
    print(f"  npix={r['num_pix']:>3} ss={r['ss']} n_max={r['n_max']:>2} "
          f"niter={r['niter']:>2} nds={r['n_datasets']} depth={r['depth']:>3} | "
          f"basis+conv+pool {r['basis_conv_pool']:7.2f}  defl {r['deflection']:7.2f}  "
          f"gram {r['gram_solve']:7.2f}  red {r['reduction']:6.2f}  "
          f"bwd {r['backward']:7.2f} | grad {r['grad_total']:8.2f} ms "
          f"(min {r['grad_min']:.2f}, GFLOP {r['grad_gflop']:.0f}, "
          f"peakMB {r['xla_peak_mb']:.0f})")


# ----------------------------------------------------------------------------- bs sweep
def bs_sweep(num_pix, supersample, n_max, bs_list, n_iter=10):
    """H-E: MAP-loss grad per-sample cost + peak memory vs batch size."""
    model, prob, cfg, z_vec, params = build_problem(
        num_pix=num_pix, supersample=supersample, n_max=n_max)
    npx = prob.num_pixels

    def loss(z):
        lp, _ = prob.log_prob(z)
        return -jnp.mean(lp) / npx

    rows = []
    for bs in bs_list:
        z = jnp.tile(z_vec[jnp.newaxis, :], (bs, 1))
        # jitter so samples are not identical (no accidental constant-folding)
        z = z + 1e-3 * jr.normal(jr.PRNGKey(bs), z.shape)
        zin = z[0] if bs == 1 else z
        f_grad = jax.jit(jax.grad(loss))
        # Compile + static memory model first: delivered even if the run OOMs.
        peak = float("nan")
        try:
            comp = f_grad.lower(zin).compile()
            peak = xla_peak_mb(comp)
        except Exception as e:
            msg = str(e).splitlines()[0][:160]
            rows.append(dict(num_pix=num_pix, ss=supersample, n_max=n_max, bs=bs,
                             oom=True, stage="compile", error=msg))
            print(f"  npix={num_pix:>3} ss={supersample} n_max={n_max:>2} bs={bs:>5} | "
                  f"COMPILE FAILED: {msg}")
            break
        try:
            med, mn = time_fn(f_grad, zin, n_iter=n_iter)
            rows.append(dict(num_pix=num_pix, ss=supersample, n_max=n_max, bs=bs,
                             grad_ms=med, grad_ms_min=mn, per_sample_ms=med / bs,
                             xla_peak_mb=peak, oom=False))
            print(f"  npix={num_pix:>3} ss={supersample} n_max={n_max:>2} bs={bs:>5} | "
                  f"grad {med:9.2f} ms  per-sample {med / bs:8.3f} ms  "
                  f"xla-peak {peak:9.0f} MB")
        except Exception as e:  # XlaRuntimeError RESOURCE_EXHAUSTED etc.
            msg = str(e).splitlines()[0][:160]
            rows.append(dict(num_pix=num_pix, ss=supersample, n_max=n_max, bs=bs,
                             oom=True, stage="run", xla_peak_mb=peak, error=msg))
            print(f"  npix={num_pix:>3} ss={supersample} n_max={n_max:>2} bs={bs:>5} | "
                  f"RUN FAILED (xla-peak {peak:.0f} MB): {msg}")
            break  # larger bs will also fail
    return rows


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all",
                    choices=["all", "matrix", "niter", "datasets", "bs"])
    ap.add_argument("--n-iter", type=int, default=30)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("device:", jax.devices()[0])
    results = dict(device=str(jax.devices()[0]), jax=jax.__version__, parts={})

    if args.part in ("all", "matrix"):
        print("\n== H-A/H-B stage matrix (bs=1 grad decomposition) ==")
        rows = []
        for (npix, ss, nm) in [(200, 2, 15),   # anchor (June comparable)
                               (200, 2, 30), (200, 2, 40),      # depth axis
                               (200, 4, 15), (200, 4, 30),      # ss axis (vela)
                               (300, 2, 15), (300, 2, 30)]:     # cutout axis
            try:
                r = stage_breakdown(npix, ss, nm, n_iter=args.n_iter)
                rows.append(r)
                print_stage_row(r)
            except Exception as e:
                msg = str(e).splitlines()[0][:160]
                rows.append(dict(num_pix=npix, ss=ss, n_max=nm, failed=msg))
                print(f"  npix={npix} ss={ss} n_max={nm} FAILED: {msg}")
        results["parts"]["matrix"] = rows

    if args.part in ("all", "niter"):
        print("\n== H-C niter sweep (anchor config) ==")
        rows = []
        for niter in [5, 18, 50]:
            r = stage_breakdown(200, 2, 15, niter=niter, n_iter=args.n_iter)
            rows.append(r)
            print_stage_row(r)
        results["parts"]["niter"] = rows

    if args.part in ("all", "datasets"):
        print("\n== H-D dataset scaling (anchor config) ==")
        rows = []
        for nds in [1, 2]:
            r = stage_breakdown(200, 2, 15, n_datasets=nds, n_iter=args.n_iter)
            rows.append(r)
            print_stage_row(r)
        results["parts"]["datasets"] = rows

    if args.part in ("all", "bs"):
        print("\n== H-E MAP-scale batch sweep ==")
        rows = bs_sweep(200, 2, 15, [1, 8, 32, 128, 500])
        rows += bs_sweep(80, 2, 15, [1, 32, 128, 512, 2000])
        results["parts"]["bs"] = rows

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results_slow_regimes.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nresults written to {out}")


if __name__ == "__main__":
    main()
