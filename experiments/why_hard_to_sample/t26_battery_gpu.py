"""T26 -- acceptance battery on a reparameterization route (GPU stage).

Pre-registered 2026-07-03 (docs/logs/why-hard-to-sample.md, "T25 ... + T26 ...").
Scope: NEW arm only. Per route (A or B; --route): 3 seeds of STANDARD MCLMC from
the SAME physical T21 typical-set init point, MAPPED into the route coordinates,
plus a matched Gaussian clone fitted to that route's OWN post-burn-in samples and
run at the same init. Census (displaced-chain, T16) + xi-tail stats exactly as
T21. Outputs use the SAME schema as the T21 npz.

WHY "same physical init": the funnel fix must be compared against the T21 new-arm
baseline (eps 0.354, min-ESS ~143) at the SAME starting point, so the ONLY thing
that changed is the coordinate. We recompute z_init with T21's exact selection
(select_typical_init, seed 20260703, closest-to-median logp evaluated in BASELINE
z-coords -- coordinate-dependent, so it MUST use the baseline prob_model) and map
it through the route leaf: z_init_route = leaf.inverse(z_init) on the Rs column.

Outputs (HARDCODED dir; never a CLI arg):
  results_carousel/phaseC/t26/route{A,B}/t0_seed{1,2,3}.npz  (+ manifests, census)
  results_carousel/phaseC/t26/route{A,B}/clone.npz           (fit to own samples)
  results_carousel/phaseC/t26/route{A,B}/t1_clone_typical_seed1.npz
  results_carousel/phaseC/t26/route{A,B}/summary.json
  --limit smoke -> short chains (200/200), *_smoke outputs.
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from common import (  # noqa: E402
    assert_x64, compute_diagnostics, git_commit, load_target,
    print_diagnostics, run_standard_mclmc,
)
from exp_config import STANDARD  # noqa: E402
from t18_map_arm import _displaced_census, CENSUS_SIGMA  # noqa: E402
from run_t1_clone import build_clone_target  # noqa: E402
from build_clone import _spd_cholesky  # noqa: E402
from t21_typical_init import (  # noqa: E402
    select_typical_init, xi_stats, Z_INIT_SEED, QZ_SCALE, ARMS as T21_ARMS,
)

SEEDS = [1, 2, 3]
CLONE_SEED = 1
T26_OUT = os.path.join(HERE, "results_carousel", "phaseC", "t26")
BASE_SYS = os.path.join(HERE, "systems", "carousel_min_new")

# T21 new-arm baselines (pre-reg; for the manifest's side-by-side context)
T21_NEW_BASELINE = {"eps": 0.354, "min_ess": 143, "clone_eps": 1.22,
                    "clone_min_ess": 1008, "frac_xi_gt10": [0.077, 0.087]}


def _route_module(route):
    mod_dir = "carousel_min_newA" if route == "A" else "carousel_min_newB"
    spec = importlib.util.spec_from_file_location(
        f"_route_{route}_sys", os.path.join(HERE, "systems", mod_dir, "system.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fit_clone(pooled, out_npz):
    """Fit a full-cov Gaussian to pooled (N,dim) samples and save a clone.npz in
    build_clone's format (mean, cov, cholesky). Same math as build_clone.py."""
    mean = pooled.mean(axis=0)
    cov = np.cov(pooled, rowvar=False).astype(np.float64)
    cov = 0.5 * (cov + cov.T)
    L, cov_used, min_eig, jitter = _spd_cholesky(cov)
    os.makedirs(os.path.dirname(os.path.abspath(out_npz)) or ".", exist_ok=True)
    np.savez(out_npz, mean=mean, cov=cov_used, cholesky=L,
             min_eig=min_eig, jitter=jitter, dim=pooled.shape[1])
    return min_eig, jitter


def run_route(route, stage, limit):
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions

    assert_x64()
    cfg = STANDARD if not limit else dataclasses.replace(
        STANDARD, num_burnin_steps=int(limit), num_results=int(limit))
    nr = cfg.num_results
    suffix = "_smoke" if limit else ""
    out_dir = os.path.join(T26_OUT, f"route{route}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n========== T26 route {route} (stage={stage}, "
          f"config={cfg.num_burnin_steps}/{cfg.num_results}) ==========", flush=True)

    # --- baseline new system: recover the EXACT T21 z_init (baseline coords) --
    ms_b, _qz_b, z_center, dim, param_names = load_target(BASE_SYS)
    pm_b = ms_b.prob_model
    ref_path = T21_ARMS["new"]["ref"]
    chains = T21_ARMS["new"]["chains"]
    z_init, ledger, sel_dim = select_typical_init(pm_b, ref_path, chains, Z_INIT_SEED)
    if sel_dim != dim:
        raise ValueError(f"selection dim {sel_dim} != dim {dim}")
    print(f"[T26 {route}] z_init = T21 maximally-typical draw (chain "
          f"{ledger['median_draw_used_as_init']['chain']}, t "
          f"{ledger['median_draw_used_as_init']['t']}, logp "
          f"{ledger['median_draw_used_as_init']['logp']:.3f})", flush=True)

    # --- route system + mapped init ------------------------------------------
    rmod = _route_module(route)
    rb = rmod.build_route_target()
    ms_r = rb["model_seq"]; leaf = rb["leaf"]; col = rb["col"]
    if rb["dim"] != dim or list(rb["param_names"]) != list(param_names):
        raise ValueError("route system dim/param mismatch vs baseline new arm")
    u_init = rmod.map_z_to_u(z_init, leaf, col)
    print(f"[T26 {route}] mapped init: z_Rs {z_init[col]:.4f} -> u_Rs {u_init[col]:.4f} "
          f"(other coords unchanged; max|dz|={np.max(np.abs(u_init-z_init)):.3g})",
          flush=True)

    qz_prime = tfd.MultivariateNormalDiag(
        loc=jnp.asarray(u_init), scale_diag=jnp.full(dim, QZ_SCALE))

    result = {"route": route, "dim": dim, "qz_scale": QZ_SCALE,
              "artifact": rb["artifact"], "artifact_sha256": rb["sha256"],
              "z_init_ledger": ledger, "config": cfg.to_dict(),
              "rs_col": int(col), "param_names": list(param_names),
              "u_init_Rs": float(u_init[col]), "z_init_Rs": float(z_init[col])}

    # --- real runs (seeds 1,2,3) ---------------------------------------------
    if stage in ("real", "both"):
        per_seed = []
        for seed in SEEDS:
            out_npz = os.path.join(out_dir, f"t0_seed{seed}{suffix}.npz")
            pos = run_standard_mclmc(
                ms_r, qz_prime, cfg, seed, out_npz,
                target_desc=f"carousel_min_new route {route} (reparam), T26 typical init",
                provenance={"route": route, "artifact": rb["artifact"],
                            "artifact_sha256": rb["sha256"],
                            "qz": f"MVNDiag(loc=u_init, scale_diag={QZ_SCALE})",
                            "z_init_median_draw": ledger["median_draw_used_as_init"]},
            )
            diag = compute_diagnostics(pos, nr, param_names)
            print_diagnostics(diag, header=f"T26 {route} real seed {seed}")
            census = _displaced_census(pos, nr)
            xis = xi_stats(out_npz, num_results=nr)
            print(f"[T26 {route}] seed {seed} persistent chains="
                  f"{census['persistent_chains']}; frac(xi>10)={xis['frac_gt10']:.4f} "
                  f"(p99={xis['p99']:.3g} max={xis['max']:.3g})", flush=True)
            with open(os.path.splitext(out_npz)[0] + ".census.json", "w") as f:
                json.dump(census, f, indent=2)
            per_seed.append({
                "seed": seed, "npz": os.path.abspath(out_npz),
                "min_ess": diag["min_ess"], "min_ess_param": diag["min_ess_param"],
                "max_rhat": diag["max_rhat"], "max_rhat_param": diag["max_rhat_param"],
                "ess_per_param": {n: float(e) for n, e in zip(diag["names"], diag["ess"])},
                "n_persistent": census["n_persistent"],
                "persistent_chains": census["persistent_chains"], "xi": xis,
            })
        result["real"] = {"per_seed": per_seed}

    # --- matched clone (fit to route's OWN post-burn-in samples) --------------
    if stage in ("clone", "both"):
        # pool the real runs just produced (or on disk) for this route
        pooled = []
        for seed in SEEDS:
            p = os.path.join(out_dir, f"t0_seed{seed}{suffix}.npz")
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"clone stage needs real run {p}; run --stage real/both first.")
            pos = np.asarray(np.load(p)["position"], np.float64)
            pooled.append(pos[:, -nr:, :].reshape(-1, dim))
        pooled = np.concatenate(pooled, axis=0)
        clone_npz = os.path.join(out_dir, f"clone{suffix}.npz")
        min_eig, jitter = _fit_clone(pooled, clone_npz)
        print(f"[T26 {route}] clone fit to {pooled.shape[0]} own samples "
              f"(min_eig={min_eig:.3e}, jitter={jitter:.3e}) -> {clone_npz}", flush=True)

        clone_stub, clone_dim = build_clone_target(clone_npz)
        if clone_dim != dim:
            raise ValueError(f"clone dim {clone_dim} != dim {dim}")
        out_npz = os.path.join(out_dir, f"t1_clone_typical_seed{CLONE_SEED}{suffix}.npz")
        pos = run_standard_mclmc(
            clone_stub, qz_prime, cfg, CLONE_SEED, out_npz,
            target_desc=f"Gaussian clone of route {route} own samples, T26 typical init",
            provenance={"route": route, "clone_npz": os.path.abspath(clone_npz),
                        "qz": f"SAME qz' as real: MVNDiag(loc=u_init, {QZ_SCALE})"},
        )
        diag = compute_diagnostics(pos, nr, param_names)
        print_diagnostics(diag, header=f"T26 {route} clone seed {CLONE_SEED}")
        xis = xi_stats(out_npz, num_results=nr)
        result["clone"] = {
            "seed": CLONE_SEED, "npz": os.path.abspath(out_npz),
            "clone_npz": os.path.abspath(clone_npz),
            "min_ess": diag["min_ess"], "min_ess_param": diag["min_ess_param"],
            "max_rhat": diag["max_rhat"], "max_rhat_param": diag["max_rhat_param"],
            "xi": xis,
        }

    # --- summary -------------------------------------------------------------
    summary = {
        "experiment": f"T26 acceptance battery route {route}",
        "status": "proposed (UNCERTIFIED)",
        "stage": stage, "smoke_limit": (int(limit) if limit else None),
        "T21_new_baseline": T21_NEW_BASELINE,
        "census_sigma": CENSUS_SIGMA,
        "conv_precision_env": os.environ.get("WHTS_CONV_PRECISION", "float32(default)"),
        "git_commit": git_commit(),
        "result": result,
    }
    sfile = os.path.join(out_dir, f"summary{suffix}.json")
    with open(sfile, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"[T26 {route}] wrote {sfile} (status: proposed (UNCERTIFIED))", flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="T26 acceptance battery (per route)")
    ap.add_argument("--route", choices=["A", "B"], required=True)
    ap.add_argument("--stage", choices=["real", "clone", "both"], default="both")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke: short chains of N burnin/N results; writes *_smoke")
    args = ap.parse_args(argv)

    assert_x64()
    print(f"[T26] float64 asserted; WHTS_CONV_PRECISION="
          f"{os.environ.get('WHTS_CONV_PRECISION')}; route={args.route} "
          f"stage={args.stage} limit={args.limit or 'none'}", flush=True)
    run_route(args.route, args.stage, args.limit)


if __name__ == "__main__":
    main()
