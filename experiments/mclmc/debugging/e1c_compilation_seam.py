"""E1c — confirm the compilation-seam dE-offset mechanism.

DIAGNOSIS ONLY. Reuses d3_eevpd_vs_eps.py's model construction and anchor
loading VERBATIM. Does NOT modify any library/sampler code.

Pre-registered card (E1c):
  Hypothesis: the four frozen-anchor dE offsets (+3.859/+4.203/+2.672/-3581)
  and the bootstrap +0.078125 are produced by the
  init-logdensity-vs-kernel-recomputed-logdensity compilation seam, ON GPU.
  Predictions:
   (i)  On GPU, logdensity_kernel_path(x) - logdensity_standalone(x) reproduces
        those five constants to a few ulps (sign+magnitude), per anchor.
        NOTE on sign: the kernel forms energy_error = kinetic - ld_kernel +
        state.logdensity(standalone). At eps->0, kinetic->0 so
        dE -> ld_standalone - ld_kernel == -(c), with c = ld_kernel - ld_std.
        So the archived dE equals (ld_standalone - ld_kernel). We report both.
   (ii) Offset is eps-independent and momentum-independent.
   (iii)On CPU the difference is <= a few ulps (~0.02) at ALL anchors incl. #3.
   (iv) Honest dE (init ld set by kernel's own compiled path) collapses
        small-eps dE to ~ulp.

What we measure, per anchor:
  A   = standalone jax.value_and_grad(logdensity_fn)(x)[0]   (d3 init path)
  B   = kernel-recomputed logdensity, read directly from info.logdensity at
        tiny eps (eps->0 => kernel ld of ~anchor; mclachlan does two half
        position updates but at eps=1e-9 the position barely moves, so
        info.logdensity ~ ld_kernel(anchor)). We ALSO read the eps->0 dE
        (info.extras.energy_change_raw), which by construction = A - B'
        where B' is the kernel ld at the (eps-perturbed) endpoint.
  raw_dE(eps->0)  = the d3 quantity = energy_change_raw at tiny eps.
  honest_dE       = dE of a SECOND step whose input state.logdensity came from
                    the first (warm-up) kernel step's output (kernel's own
                    compiled ld). Should be ~ulp.
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


def _parse():
    p = argparse.ArgumentParser()
    p.add_argument("--n-max", type=int, default=25)
    p.add_argument("--precision", choices=["float32", "float64"], default="float32")
    p.add_argument("--anchor-class", choices=["run_a_late", "bootstrap", "both"],
                   default="both")
    p.add_argument("--device", choices=["cpu", "gpu"], default="gpu",
                   help="Informational tag only; actual device set by env (JAX_PLATFORMS).")
    p.add_argument("--K", type=int, default=8, help="momentum draws (>=8 for momentum-independence)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = _parse()

    if args.precision == "float64":
        import jax
        jax.config.update("jax_enable_x64", True)
        print("[E1c] float64 enabled", flush=True)

    import jax
    import jax.numpy as jnp
    print(f"[E1c] JAX {jax.__version__} devices: {jax.devices()}", flush=True)
    print(f"[E1c] n_max={args.n_max} precision={args.precision} device_tag={args.device} K={args.K}", flush=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_out = args.out_dir or os.path.join(script_dir, "diagnosis_2026-06", "e1c")
    out_dir = base_out
    os.makedirs(out_dir, exist_ok=True)
    print(f"[E1c] out_dir: {out_dir}", flush=True)

    # --- anchors: load run_a positions VERBATIM as d3 does ---
    diag_dir = os.path.join(script_dir, "diagnosis_2026-06")
    run_a_dir = os.path.join(diag_dir, "run_a_nmax25_ds1e-8_nb2000")
    pos_a = np.load(os.path.join(run_a_dir, "hist_position.npy"))  # (8,2500,17)
    anchors_a_late = pos_a[[0, 2, 4, 6], -1, :]  # (4,17) — d3 line 214
    bootstrap_proxy = pos_a[0, 0, :]             # d3 --skip-bootstrap proxy

    # --- model construction VERBATIM from d3 ---
    import tensorflow_probability.substrates.jax as tfp  # noqa: F401
    from gigalens_research.simtests.system import from_vela_dir
    from gigalens_research.simtests.experiments import vela_shapelets  # noqa: F401
    from gigalens_research.simtests.registry import get_inference_builder
    import gigalens.jax.simulator as sim

    data_dir = os.path.join(home, "GIGALens-Code/data")
    system_name = "vela01_cam12_rep03_a0.500_f814w"
    system = from_vela_dir(
        system_dir=os.path.join(data_dir, "vela_sim_systems", system_name),
        source_dir=os.path.join(data_dir, "vela_sources", system_name),
        system_id="vela01_cam12_rep03",
        delta_pix=0.03, num_pix=200, supersample=1,
        background_rms=0.002, exp_time=2000.0,
    )
    prob_model = get_inference_builder("epl_shear_sersic_shapelets")(system, n_max=args.n_max)
    lens_sim = sim.LensSimulator(prob_model.model, prob_model.datasets[0].sim_config, bs=1)

    def logdensity_fn(z):
        return prob_model.log_prob(lens_sim, z)[0]

    print(f"[E1c] model built n_max={args.n_max}", flush=True)

    from gigalens_research.inference.blackjax_updated_utils import (
        _build_kernel_shardmap, isokinetic_mclachlan_smart,
    )
    from blackjax.mcmc.integrators import IntegratorState
    from blackjax.util import generate_unit_vector

    L_val = float(np.sqrt(DIM))

    # metric: d3's MANDATORY frozen-anchor combo uses diag_1e-8_I for run_a_late
    # (the combo that produced the archived offsets). bootstrap also diag_1e-8.
    metric_np = np.eye(DIM, dtype=np.float64) * 1e-8
    metric_jnp = jnp.asarray(metric_np)
    kernel = _build_kernel_shardmap(logdensity_fn, metric_jnp, isokinetic_mclachlan_smart)

    # ulp of |logp| in current precision for reporting
    def ulp_of(x):
        return float(np.spacing(np.float32(abs(x)))) if args.precision == "float32" \
            else float(np.spacing(np.float64(abs(x))))

    eps_list = [1e-9, 1e-7, 1e-5]

    def probe_anchor(anchor_z, label):
        anchor = jnp.asarray(np.asarray(anchor_z, dtype=np.float64))

        # ---- Path A: standalone value_and_grad (d3 init, line 260) ----
        ld_A, ldg_A = jax.value_and_grad(logdensity_fn)(anchor)
        ld_A = float(ld_A)
        gnorm_A = float(jnp.linalg.norm(ldg_A))
        ldg_A_dev = jax.value_and_grad(logdensity_fn)  # keep callable

        # momenta
        rng = jax.random.PRNGKey(args.seed)
        mom_keys = jax.random.split(rng, args.K)
        momenta = jax.vmap(lambda k: generate_unit_vector(k, anchor))(mom_keys)  # (K,17)

        # ---- single d3-style step: returns (new_state, info) ----
        ld_j = jax.value_and_grad(logdensity_fn)(anchor)[0]
        ldg_j = jax.value_and_grad(logdensity_fn)(anchor)[1]

        @jax.jit
        def one_step(momentum, eps):
            state = IntegratorState(position=anchor, momentum=momentum,
                                    logdensity=ld_j, logdensity_grad=ldg_j)
            skey = jax.random.PRNGKey(0)
            new_state, info = kernel(skey, state, jnp.array(L_val), jnp.array(eps))
            # info.logdensity = kernel-recomputed ld at the (perturbed) endpoint
            # info.extras.energy_change_raw = raw dE (the d3 quantity)
            return (info.extras.energy_change_raw, info.logdensity,
                    new_state.logdensity, new_state.position)

        # sweep eps x momentum
        raw_dE = np.zeros((len(eps_list), args.K))
        ld_kernel = np.zeros((len(eps_list), args.K))
        for ie, eps in enumerate(eps_list):
            for k in range(args.K):
                rd, ldk, _, _ = one_step(momenta[k], float(eps))
                raw_dE[ie, k] = float(rd)
                ld_kernel[ie, k] = float(ldk)

        # eps->0 quantities (smallest eps)
        dE0 = raw_dE[0]            # (K,)
        ldk0 = ld_kernel[0]        # (K,) kernel ld at eps->0 endpoint
        # c(x) = ld_kernel - ld_standalone ; at eps->0 dE = ld_std - ld_kernel = -c
        # measured "standalone - kernel" discrepancy == eps->0 dE
        disc_std_minus_kernel = float(np.median(dE0))   # = ld_A - ld_kernel(endpoint)
        # also a direct ld difference (kernel ld read directly minus standalone)
        ld_kernel_direct = float(np.median(ldk0))
        direct_diff = ld_A - ld_kernel_direct  # standalone - kernel(direct read)

        # ---- (iv) HONEST dE: warm-up step, then measure NEXT step ----
        # First kernel step produces an output state whose .logdensity came from
        # the kernel's OWN compiled path. Feed that as the init of a 2nd step.
        @jax.jit
        def honest_step(momentum, eps):
            state0 = IntegratorState(position=anchor, momentum=momentum,
                                     logdensity=ld_j, logdensity_grad=ldg_j)
            skey = jax.random.PRNGKey(0)
            warm_state, _ = kernel(skey, state0, jnp.array(L_val), jnp.array(eps))
            # second step from the warm (kernel-self-consistent) state
            skey2 = jax.random.PRNGKey(1)
            _, info2 = kernel(skey2, warm_state, jnp.array(L_val), jnp.array(eps))
            return info2.extras.energy_change_raw

        honest_dE0 = np.array([float(honest_step(momenta[k], 1e-9)) for k in range(args.K)])

        u = ulp_of(ld_A)
        print(f"[E1c] [{label}] A(std logp)={ld_A:.6f}  gnorm_A={gnorm_A:.4g}  ulp={u:.4g}", flush=True)
        print(f"        kernel ld direct (eps->0) median={ld_kernel_direct:.6f}  "
              f"standalone-kernel(direct)={direct_diff:.6g} ({direct_diff/u:.2g} ulp)", flush=True)
        print(f"        eps->0 raw dE: median={disc_std_minus_kernel:.6f}  "
              f"std_over_K={float(np.std(dE0)):.3e}  ({disc_std_minus_kernel/u:.2g} ulp)", flush=True)
        for ie, eps in enumerate(eps_list):
            print(f"        eps={eps:.0e}: dE median={float(np.median(raw_dE[ie])):.6f} "
                  f"std_over_K={float(np.std(raw_dE[ie])):.3e}", flush=True)
        print(f"        HONEST dE (eps->0): median={float(np.median(honest_dE0)):.6g} "
              f"std={float(np.std(honest_dE0)):.3e}  ({float(np.median(honest_dE0))/u:.2g} ulp)", flush=True)

        return {
            "label": label,
            "anchor_logp_standalone_A": ld_A,
            "grad_norm_A": gnorm_A,
            "ulp": u,
            "kernel_ld_direct_eps0": ld_kernel_direct,
            "standalone_minus_kernel_direct": direct_diff,
            "raw_dE_eps0_median": disc_std_minus_kernel,
            "raw_dE_eps0_std_over_K": float(np.std(dE0)),
            "eps_list": eps_list,
            "raw_dE_per_eps_median": [float(np.median(raw_dE[ie])) for ie in range(len(eps_list))],
            "raw_dE_per_eps_std": [float(np.std(raw_dE[ie])) for ie in range(len(eps_list))],
            "raw_dE_full": raw_dE.tolist(),
            "ld_kernel_per_eps": ld_kernel.tolist(),
            "honest_dE_eps0_median": float(np.median(honest_dE0)),
            "honest_dE_eps0_std": float(np.std(honest_dE0)),
            "honest_dE_full": honest_dE0.tolist(),
        }

    # archived GPU constants (from d3 nmax25_float32 raw npz, eps=1e-7)
    archived = {
        "run_a_late_0": 3.859375,
        "run_a_late_1": 4.203125,
        "run_a_late_2": 2.671875,
        "run_a_late_3": -3580.78125,
        "bootstrap_0": 0.078125,
    }

    results = []
    if args.anchor_class in ("run_a_late", "both"):
        for ai in range(4):
            results.append(probe_anchor(anchors_a_late[ai], f"run_a_late_{ai}"))
    if args.anchor_class in ("bootstrap", "both"):
        results.append(probe_anchor(bootstrap_proxy, "bootstrap_0"))

    # attach archived expectation
    for r in results:
        r["archived_dE_offset_GPU"] = archived.get(r["label"], None)

    summary = {
        "n_max": args.n_max,
        "precision": args.precision,
        "device_tag": args.device,
        "jax_devices": [str(d) for d in jax.devices()],
        "metric": "diag_1e-8_I",
        "L": L_val,
        "K": args.K,
        "archived_GPU_constants": archived,
        "anchors": results,
    }

    tag = f"{args.device}_{args.precision}_nmax{args.n_max}_{args.anchor_class}"
    summary_path = os.path.join(out_dir, f"summary_{tag}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[E1c] summary written: {summary_path}", flush=True)

    # raw npz
    npz_path = os.path.join(out_dir, f"raw_{tag}.npz")
    np.savez(npz_path,
             labels=np.array([r["label"] for r in results]),
             A_logp=np.array([r["anchor_logp_standalone_A"] for r in results]),
             raw_dE_eps0=np.array([r["raw_dE_eps0_median"] for r in results]),
             direct_diff=np.array([r["standalone_minus_kernel_direct"] for r in results]),
             honest_dE_eps0=np.array([r["honest_dE_eps0_median"] for r in results]),
             archived=np.array([archived.get(r["label"], np.nan) for r in results]),
             eps_list=np.array(eps_list))
    print(f"[E1c] raw npz written: {npz_path}", flush=True)

    # console table
    print("\n" + "=" * 78, flush=True)
    print(f"E1c CONSTANTS TABLE  ({tag})", flush=True)
    print("=" * 78, flush=True)
    print(f"{'anchor':<14}{'A(logp)':>14}{'archived dE':>14}{'measured dE0':>14}{'honest dE0':>14}", flush=True)
    for r in results:
        arc = r["archived_dE_offset_GPU"]
        arc_s = f"{arc:+.4f}" if arc is not None else "   n/a"
        print(f"{r['label']:<14}{r['anchor_logp_standalone_A']:>14.2f}"
              f"{arc_s:>14}{r['raw_dE_eps0_median']:>+14.4f}{r['honest_dE_eps0_median']:>+14.6f}", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
