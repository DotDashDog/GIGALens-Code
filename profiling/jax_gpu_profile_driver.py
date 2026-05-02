import argparse
import json
import os
import pickle
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorflow_probability.substrates.jax as tfp


def block_tree(x):
    def _block(y):
        if hasattr(y, "block_until_ready"):
            y.block_until_ready()
        return y

    return jax.tree_util.tree_map(_block, x)


def to_python(x: Any):
    if isinstance(x, dict):
        return {k: to_python(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_python(v) for v in x]
    if hasattr(x, "item"):
        try:
            return x.item()
        except Exception:
            return str(x)
    return x


def time_callable(fn: Callable[[], Any], warm_repeats: int):
    t0 = time.perf_counter()
    first = fn()
    block_tree(first)
    first_s = time.perf_counter() - t0

    warm_times = []
    for _ in range(warm_repeats):
        t0 = time.perf_counter()
        out = fn()
        block_tree(out)
        warm_times.append(time.perf_counter() - t0)

    return {
        "compile_plus_execute_s": first_s,
        "warm_min_s": min(warm_times),
        "warm_avg_s": sum(warm_times) / len(warm_times),
        "warm_repeats": warm_repeats,
    }


@dataclass
class ProfileContext:
    true_z: jax.Array
    true_params_shp: Any
    qz: Any
    model_seq: Any
    prob_model: Any
    lens_sim: Any
    map_optimizer: Any


def build_sim_system_complex_context(seed: int, initialize_distributed: bool):
    home = os.path.expanduser("~/")
    sys.path.insert(0, os.path.join(home, "gigalens", "src"))
    sys.path.insert(0, os.path.join(home, "GIGALens-Code"))

    if initialize_distributed and (not jax.distributed.is_initialized()):
        jax.distributed.initialize()

    if not hasattr(jax.experimental, "shard_map"):
        jax.experimental.shard_map = types.SimpleNamespace(shard_map=jax.shard_map)

    from gigalens.jax.inference import ModellingSequence
    from gigalens.jax.model import BackwardProbModel
    from gigalens.jax.profiles.light import sersic, shapelets
    from gigalens.jax.profiles.mass import epl, shear
    from gigalens.jax.simulator import LensSimulator
    from gigalens.model import PhysicalModel
    from gigalens.simulator import SimulatorConfig

    tfd = tfp.distributions

    save_dir = os.path.join(home, "GIGALens-Code", "alternate_inference", "vela_sim_systems")
    source_plane_dir = os.path.join(save_dir, "vela10_cam00_a0.500_f814w")

    psf = np.load(os.path.join(source_plane_dir, "psf.npy")).astype(np.float32)
    observed_img_np = np.load(os.path.join(save_dir, "lens_img_no_subhalo.npy")).astype(np.float32)
    with open(os.path.join(save_dir, "true_params"), "rb") as fh:
        true_params = pickle.load(fh)
    with open(os.path.join(source_plane_dir, "metadata.json")) as fh:
        metadata = json.load(fh)

    delta_pix = metadata["instrument_pixel_scale_arcsec"]
    background_rms = 0.002
    exp_time = 2000.0
    num_pix = 200

    lens_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    theta_E=tfd.LogNormal(jnp.log(1.25), 0.4),
                    gamma=tfd.TruncatedNormal(2, 0.5, 1, 3),
                    e1=tfd.TruncatedNormal(0, 0.2, -0.5, 0.5),
                    e2=tfd.TruncatedNormal(0, 0.2, -0.5, 0.5),
                    center_x=tfd.Normal(0, 0.2),
                    center_y=tfd.Normal(0, 0.2),
                )
            ),
            tfd.JointDistributionNamed(
                dict(
                    gamma1=tfd.TruncatedNormal(0, 0.1, -0.5, 0.5),
                    gamma2=tfd.Normal(0, 0.1, -0.5, 0.5),
                )
            ),
        ]
    )
    lens_light_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    R_sersic=tfd.LogNormal(jnp.log(1.6), 0.25),
                    n_sersic=tfd.Uniform(0.5, 8),
                    e1=tfd.TruncatedNormal(0, 0.1, -0.2, 0.2),
                    e2=tfd.TruncatedNormal(0, 0.1, -0.2, 0.2),
                    center_x=tfd.Normal(0, 0.2),
                    center_y=tfd.Normal(0, 0.2),
                )
            )
        ]
    )
    source_light_prior = tfd.JointDistributionSequential(
        [
            tfd.JointDistributionNamed(
                dict(
                    beta=tfd.LogNormal(jnp.log(0.7), 0.4),
                    center_x=tfd.Normal(0, 0.01),
                    center_y=tfd.Normal(0, 0.01),
                )
            )
        ]
    )
    prior = tfd.JointDistributionSequential([lens_prior, lens_light_prior, source_light_prior])

    n_max = 10
    with jax.default_device(jax.devices("cpu")[0]):
        observed_img = jnp.array(observed_img_np)
        phys_model = PhysicalModel(
            [epl.EPL(50), shear.Shear()],
            [sersic.SersicEllipse(use_lstsq=True)],
            [shapelets.ShapeletsFast(n_max=n_max, use_lstsq=True, interpolate=True)],
        )
        sim_config = SimulatorConfig(delta_pix=delta_pix, num_pix=num_pix, supersample=1, kernel=psf)
        prob_model = BackwardProbModel(prior, observed_img, background_rms=background_rms, exp_time=exp_time)
        model_seq = ModellingSequence(phys_model, prob_model, sim_config)
        lens_sim = LensSimulator(phys_model, sim_config, bs=1)

        lens_light_no_Ie = true_params[1][0].copy()
        del lens_light_no_Ie["Ie"]
        shp_true = {
            "beta": jnp.array([0.1676427], dtype=jnp.float32),
            "center_x": jnp.array([-0.03878419], dtype=jnp.float32),
            "center_y": jnp.array([-0.10895219], dtype=jnp.float32),
        }
        true_params_shp = [true_params[0], [lens_light_no_Ie], [shp_true]]
        true_z = jnp.stack(prob_model.bij.inverse(true_params_shp)).T.astype(jnp.float32)

        default_start = jnp.diag(jnp.ones((true_z.shape[-1],), dtype=jnp.float32)) * 1e-3
        qz = tfd.MultivariateNormalTriL(loc=jnp.squeeze(true_z), scale_tril=default_start)
    map_optimizer = optax.adabelief(1e-2, b1=0.95, b2=0.99)

    return ProfileContext(
        true_z=true_z,
        true_params_shp=true_params_shp,
        qz=qz,
        model_seq=model_seq,
        prob_model=prob_model,
        lens_sim=lens_sim,
        map_optimizer=map_optimizer,
    )


def make_targets(ctx: ProfileContext, args):
    from alternate_inference.mclmc_alt import MCLMC_JIT

    log_prob = jax.jit(lambda z: ctx.prob_model.log_prob(ctx.lens_sim, z)[0])
    value_and_grad = jax.jit(
        jax.value_and_grad(lambda z: jnp.sum(ctx.prob_model.log_prob(ctx.lens_sim, z)[0]))
    )

    def run_map():
        return ctx.model_seq.MAP(
            ctx.map_optimizer,
            n_samples=args.map_samples,
            num_steps=args.map_steps,
            seed=args.seed,
            pbar_interval=0,
        )

    def run_mclmc():
        return MCLMC_JIT(
            ctx.model_seq,
            ctx.qz,
            n_hmc=args.mclmc_chains,
            num_burnin_steps=args.mclmc_burnin,
            num_results=args.mclmc_results,
            desired_energy_variance=5e-4,
            frac_tune1=0.2,
            frac_tune2=0.6,
            frac_tune3=0.2,
            seed=args.seed,
            debug_output=True,
            step_size_adapt_use_psmile=True,
            use_shard_map=args.use_shard_map,
            progress_bar=False,
        )

    return {
        "backward_log_prob": lambda: log_prob(ctx.true_z),
        "value_and_grad": lambda: value_and_grad(ctx.true_z),
        "map": run_map,
        "mclmc": run_mclmc,
    }


def simulator_stage_breakdown(ctx: ProfileContext, repeats: int):
    lens_sim = ctx.lens_sim
    observed_image = ctx.prob_model.observed_image
    err_map = ctx.prob_model.err_map
    params = ctx.true_params_shp

    @jax.jit
    def beta_only(params):
        return lens_sim._beta(params[0])

    @jax.jit
    def build_basis(params):
        lens_params = params[0]
        lens_light_params, source_light_params = params[1], params[2]
        beta_x, beta_y = lens_sim._beta(lens_params)
        img = jnp.zeros((0, *lens_sim.img_X.shape))
        for light_model, p in zip(lens_sim.phys_model.lens_light, lens_light_params):
            img = jnp.concatenate((img, light_model.light(lens_sim.img_X, lens_sim.img_Y, **p)), axis=0)
        for light_model, p in zip(lens_sim.phys_model.source_light, source_light_params):
            img = jnp.concatenate((img, light_model.light(beta_x, beta_y, **p)), axis=0)
        return jnp.nan_to_num(img)

    @jax.jit
    def conv_pool(img):
        img = jnp.transpose(img, (3, 0, 1, 2))
        if lens_sim.flat_kernel is not None:
            bs, depth, height, width = img.shape
            folded = jnp.reshape(img, (bs * depth, 1, height, width))
            ret = jax.lax.conv(folded, lens_sim.flat_kernel, (1, 1), "SAME")
            ret = jnp.reshape(ret, (bs, depth, height, width))
        else:
            ret = img
        if lens_sim.supersample != 1:
            from objax.functional import average_pool_2d
            ret = average_pool_2d(ret, size=(lens_sim.supersample, lens_sim.supersample), padding="SAME")
        return jnp.transpose(ret, (0, 2, 3, 1))

    @jax.jit
    def linear_solve(ret):
        W = (1 / err_map)[..., jnp.newaxis]
        Y = jnp.reshape(observed_image * jnp.squeeze(W), (1, -1, 1))
        X = jnp.reshape((ret * W), (lens_sim.bs, -1, lens_sim.depth))
        Xt = jnp.transpose(X, (0, 2, 1))
        return _solve_normal_eq_with_fallback(Xt @ X, Xt @ Y)[..., 0]

    @jax.jit
    def recombine(ret, coeffs):
        return jnp.sum(ret * coeffs[:, jnp.newaxis, jnp.newaxis, :], axis=-1)

    stage_results = {
        "beta_only": time_callable(lambda: beta_only(params), repeats),
        "build_basis": time_callable(lambda: build_basis(params), repeats),
    }

    basis = build_basis(params)
    block_tree(basis)
    stage_results["conv_pool"] = time_callable(lambda: conv_pool(basis), repeats)

    pooled = conv_pool(basis)
    block_tree(pooled)
    stage_results["linear_solve"] = time_callable(lambda: linear_solve(pooled), repeats)

    coeffs = linear_solve(pooled)
    block_tree(coeffs)
    stage_results["recombine"] = time_callable(lambda: recombine(pooled, coeffs), repeats)
    return stage_results


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def run_baseline(ctx: ProfileContext, args):
    targets = make_targets(ctx, args)
    results = {
        "devices": [str(device) for device in jax.devices()],
        "jax_version": jax.__version__,
        "targets": {},
    }
    for name, fn in targets.items():
        repeats = args.low_level_repeats if name in {"backward_log_prob", "value_and_grad"} else args.high_level_repeats
        results["targets"][name] = time_callable(fn, repeats)
    results["simulator_stages"] = simulator_stage_breakdown(ctx, args.low_level_repeats)
    return results


def run_trace(ctx: ProfileContext, args):
    targets = make_targets(ctx, args)
    trace_target = targets[args.trace_target]

    ensure_dir(args.trace_dir)
    trace_target()
    with jax.profiler.trace(
        args.trace_dir,
        create_perfetto_trace=True,
        create_perfetto_link=False,
    ):
        with jax.profiler.TraceAnnotation(f"profile_{args.trace_target}"):
            result = trace_target()
            block_tree(result)

    return {
        "trace_target": args.trace_target,
        "trace_dir": args.trace_dir,
        "devices": [str(device) for device in jax.devices()],
    }


def write_json(output_path: str, payload: dict):
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w") as fh:
        json.dump(to_python(payload), fh, indent=2, sort_keys=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Profile MAP and MCLMC for the sim_system_complex workload.")
    parser.add_argument("--mode", choices=["baseline", "trace"], required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initialize-distributed", action="store_true")

    parser.add_argument("--map-samples", type=int, default=16)
    parser.add_argument("--map-steps", type=int, default=25)
    parser.add_argument("--mclmc-chains", type=int, default=8)
    parser.add_argument("--mclmc-burnin", type=int, default=150)
    parser.add_argument("--mclmc-results", type=int, default=150)
    parser.add_argument("--use-shard-map", action="store_true")

    parser.add_argument("--low-level-repeats", type=int, default=5)
    parser.add_argument("--high-level-repeats", type=int, default=2)

    parser.add_argument(
        "--trace-target",
        choices=["backward_log_prob", "value_and_grad", "map", "mclmc"],
        default="mclmc",
    )
    parser.add_argument("--trace-dir", default=os.path.expanduser("~/GIGALens-Code/profiling_outputs/jax-trace"))
    return parser.parse_args()


def main():
    args = parse_args()
    ctx = build_sim_system_complex_context(args.seed, args.initialize_distributed)

    if args.mode == "baseline":
        payload = run_baseline(ctx, args)
    else:
        payload = run_trace(ctx, args)

    if args.output_json is not None:
        write_json(args.output_json, payload)
    print(json.dumps(to_python(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
