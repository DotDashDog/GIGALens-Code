"""P-10 pre-registered analysis (docs/logs/point-source-sbc.md, P-10 checkpoint).

Inputs: runs/<sid>/mc1 (P-10, multiplicity constraint in-likelihood) and the
P-7 baselines runs/<sid>/default + diagnostics/double_filter/*.npz
(1024 thinned draws, lenstronomy image counts, kept = count==2).

Parts (pre-registered):
  A: per-system frac(N_eff~2) for ALL 100 systems on 2000 thinned mc1 draws,
     scored with the FINE-config operator (grid 768, eps 0.05, window 12",
     fp32) — deliberately not the in-likelihood config (384/0.1/4*theta_E),
     so a shared-quadrature artifact cannot fully self-confirm.
     Prediction (1): all 12 zero-kept systems > 0.9; falsifier (a): any < 0.5.
  B: healthy systems (P-7 kept_frac >= 0.5, not in the 12): per-parameter
     |mean_mc1 - mean_P7filtered| / sd_P7filtered; prediction (3): median
     across systems < 0.3 for every parameter.
  C: lenstronomy LensEquationSolver cross-check on the 12 target systems:
     500 thinned mc1 draws each, search_window 12, min_distance ladder
     {0.05, 0.025, 0.01}; count = n positions returned (dataset-generator
     convention, no magnification cut). Reported next to A's frac.
  D: loglik_rank collection across the 100 mc1 run.json files (the one
     Holm rejection in the aggregate) — rank histogram + direction.

Code: gigalens pinned worktree d321d3c, GIGALens-Code 4358ebb.
"""
import json
import multiprocessing as mp
import os
import sys

import numpy as np

os.environ.setdefault("GIGALENS_ALLOW_LEGACY_JAX", "1")
# Pinned at the commits this analysis ran against (GIGALens-Code 4358ebb,
# gigalens d321d3c; see module docstring). At analysis time ~/GIGALens-Code
# was at 4358ebb; the live checkout has since migrated off ModellingSequence,
# which changes the inference builder's return type, so the pin now points at
# a frozen worktree of the same commit.
sys.path.insert(0, os.path.expanduser("~/GIGALens-Code-worktree-p10/src"))
sys.path.insert(0, os.path.expanduser("~/gigalens-worktree-p10/src"))

BASE = os.path.expanduser(
    "~/GIGALens-Code/simtests_results/hundred_point_sources_v2_double")
DS = os.path.join(BASE, "dataset")
OUT = os.path.join(BASE, "diagnostics", "p10_analysis")

TARGET12 = ["sys_05", "sys_25", "sys_39", "sys_43", "sys_48", "sys_57",
            "sys_62", "sys_63", "sys_65", "sys_73", "sys_76", "sys_88"]

MC = dict(mu_min=0.1, eps=0.1, lam=10.0, grid_n=384, n_tiles=16,
          window_scale=4.0)
BUILDER_KW = dict(fit_flux=True, fit_td=True, newton_steps=10,
                  trust_region_frac=0.4, src_parameterization="absolute",
                  priors="from_dataset", src_anchor_sigma=0.004,
                  multiplicity_constraint=dict(MC))

COLS = ["planes/0/mass/0/theta_E", "planes/0/mass/0/gamma",
        "planes/0/mass/0/e1", "planes/0/mass/0/e2",
        "planes/0/mass/0/center_x", "planes/0/mass/0/center_y",
        "planes/0/mass/1/gamma1", "planes/0/mass/1/gamma2",
        "planes/1/light/0/center_x", "planes/1/light/0/center_y"]

N_NEFF_DRAWS = 2000
N_LT_DRAWS = 500
LT_LADDER = [0.05, 0.025, 0.01]
LT_WINDOW = 12.0
Z_LENS, Z_SOURCE, OM0 = 0.5, 1.5, 0.3


# --- fine-config N_eff (verbatim from the validated tiled prototype) --------
def make_neff(grid_n, n_tiles, window=None, window_scale=None, fp32=True):
    import jax
    import jax.numpy as jnp
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.mass.shear import Shear

    assert (window is None) != (window_scale is None)
    epl, shear = EPL(niter=18), Shear()
    u1 = (jnp.arange(grid_n) + 0.5) / grid_n - 0.5
    ux, uy = jnp.meshgrid(u1, u1)
    upts = jnp.stack([ux.ravel(), uy.ravel()], -1).reshape(n_tiles, -1, 2)
    dt = jnp.float32 if fp32 else jnp.float64
    e0 = jnp.array([1.0, 0.0], dt)
    e1v = jnp.array([0.0, 1.0], dt)

    def neff(row, eps, mu_min):
        row = row.astype(dt)
        eps = jnp.asarray(eps, dt)
        mu = jnp.asarray(mu_min, dt)
        tE, gam, ee1, ee2, cx, cy, g1, g2, sx, sy = row
        win = jnp.asarray(window, dt) if window is not None \
            else window_scale * tE
        cell = (win / grid_n) ** 2

        def f(pt):
            ax_, ay_ = epl.deriv(pt[0], pt[1], tE, gam, ee1, ee2, cx, cy)
            bx_, by_ = shear.deriv(pt[0], pt[1], g1, g2)
            return jnp.stack([pt[0] - ax_ - bx_, pt[1] - ay_ - by_])

        def per_pt(upt):
            pt = jnp.stack([cx, cy]) + win * upt
            b, t1 = jax.jvp(f, (pt,), (e0,))
            _, t2 = jax.jvp(f, (pt,), (e1v,))
            det = jnp.abs(t1[0] * t2[1] - t2[0] * t1[1])
            w = 1.0 / (1.0 + (mu * det) ** 4)
            d2 = (b[0] - sx) ** 2 + (b[1] - sy) ** 2
            return jnp.exp(-0.5 * d2 / eps**2) * det * w

        @jax.checkpoint
        def tile_sum(carry, tile):
            return carry + jnp.sum(jax.vmap(per_pt)(tile.astype(dt))), None

        tot, _ = jax.lax.scan(tile_sum, jnp.zeros((), dt), upts)
        return (tot * cell / (2.0 * jnp.pi * eps**2)).astype(jnp.float64)

    return neff


def thin_rows(sid, n_draws):
    """Thinned mc1 draws as (n_draws, 10) rows in COLS order + full params."""
    import jax.numpy as jnp
    import jax
    a = np.load(os.path.join(BASE, "runs", sid, "mc1", "mclmc", "arrays.npz"))
    sz = a["samples_z"].reshape(-1, a["samples_z"].shape[-1])
    sel = np.linspace(0, len(sz) - 1, n_draws).astype(int)
    prob = _PROBS[sid]
    params = jax.tree_util.tree_map(
        np.asarray, prob.bij.forward(jnp.asarray(sz[sel])))
    rows = np.stack([np.asarray([params[c][j] for c in COLS])
                     for j in range(len(sel))])
    return rows, params


def part_a(sids):
    import jax
    import jax.numpy as jnp
    neff_fn = jax.jit(make_neff(768, 64, window=12.0, fp32=True))
    res = {}
    for sid in sids:
        rows, _ = thin_rows(sid, N_NEFF_DRAWS)
        ne = np.array([float(neff_fn(jnp.asarray(r), 0.05, MC["mu_min"]))
                       for r in rows])
        res[sid] = {
            "frac_neff_2": float(np.mean(np.abs(ne - 2.0) < 0.35)),
            "neff_q10_50_90": [float(np.percentile(ne, q))
                               for q in (10, 50, 90)],
        }
        np.save(os.path.join(OUT, f"neff_{sid}.npy"), ne)
        print(f"[A] {sid}: frac(N_eff~2) = {res[sid]['frac_neff_2']:.3f}  "
              f"q10/50/90 = {res[sid]['neff_q10_50_90']}", flush=True)
    return res


def part_b(kept_frac_per_system):
    filt_dir = os.path.join(BASE, "diagnostics", "double_filter")
    healthy = [s for s, f in sorted(kept_frac_per_system.items())
               if f >= 0.5 and s not in TARGET12]
    deltas = {}          # param -> list of |dmean|/sd across systems
    for sid in healthy:
        d = np.load(os.path.join(filt_dir, f"{sid}.npz"))
        keys = [str(k) for k in d["keys"]]
        kept = d["counts"] == 2
        if kept.sum() < 10:
            continue
        p7_mean = d["mat"][kept].mean(axis=0)
        p7_sd = d["mat"][kept].std(axis=0)
        _, params = thin_rows(sid, 1024)
        for j, k in enumerate(keys):
            m1 = float(np.mean(params[k]))
            z = abs(m1 - p7_mean[j]) / max(p7_sd[j], 1e-30)
            deltas.setdefault(k, []).append(z)
        print(f"[B] {sid}: max|dmean|/sd = "
              f"{max(deltas[k][-1] for k in keys):.3f}", flush=True)
    summary = {k: {"median": float(np.median(v)),
                   "q90": float(np.percentile(v, 90)),
                   "max": float(np.max(v)), "n": len(v)}
               for k, v in deltas.items()}
    return {"n_healthy_systems": len(healthy), "per_param": summary}


def _count_one(args):
    (theta_E, gamma, e1, e2, cx, cy, g1, g2, sx, sy) = args
    from astropy.cosmology import FlatLambdaCDM
    from lenstronomy.LensModel.lens_model import LensModel
    from lenstronomy.LensModel.Solver.lens_equation_solver import (
        LensEquationSolver)
    cosmo = FlatLambdaCDM(H0=70.0, Om0=OM0, Ob0=0.0)
    lm = LensModel(["EPL", "SHEAR"], z_lens=Z_LENS, z_source=Z_SOURCE,
                   cosmo=cosmo)
    kw = [dict(theta_E=theta_E, gamma=gamma, e1=e1, e2=e2,
               center_x=cx, center_y=cy),
          dict(gamma1=g1, gamma2=g2, ra_0=0.0, dec_0=0.0)]
    counts = []
    for md in LT_LADDER:
        try:
            x, y = LensEquationSolver(lm).image_position_from_source(
                sx, sy, kw, min_distance=md, search_window=LT_WINDOW,
                precision_limit=1e-10)
            counts.append(int(np.asarray(x).size))
        except Exception:
            counts.append(-1)
    return counts


def part_c(sids):
    res = {}
    ctx = mp.get_context("spawn")     # no fork of the CUDA-initialized parent
    with ctx.Pool(min(8, os.cpu_count() or 8)) as pool:
        for sid in sids:
            rows, _ = thin_rows(sid, N_LT_DRAWS)
            counts = np.array(pool.map(_count_one, [tuple(r) for r in rows]))
            fin = counts[:, -1]                      # finest ladder rung
            stable = (counts[:, 1] == counts[:, 2])
            valid = fin >= 0
            res[sid] = {
                "frac_count2": float(np.mean(fin[valid] == 2)),
                "frac_valid": float(np.mean(valid)),
                "frac_ladder_stable": float(np.mean(stable[valid])),
                "count_hist": {int(k): int(n) for k, n in
                               zip(*np.unique(fin[valid], return_counts=True))},
            }
            np.save(os.path.join(OUT, f"ltcounts_{sid}.npy"), counts)
            print(f"[C] {sid}: frac(count==2) = {res[sid]['frac_count2']:.3f}"
                  f"  hist = {res[sid]['count_hist']}", flush=True)
    return res


def part_d(sids):
    ranks, L = [], None
    for sid in sids:
        with open(os.path.join(BASE, "runs", sid, "mc1", "run.json")) as fh:
            m = json.load(fh)["metrics"]["loglik_rank"]
        ranks.append(m["rank"])
        L = m["_L"]
    ranks = np.asarray(ranks, float)
    pit = ranks / L
    return {"L": L, "ranks": ranks.tolist(),
            "mean_pit": float(pit.mean()),
            "pit_deciles": np.histogram(pit, bins=10, range=(0, 1))[0].tolist()}


def main():
    os.makedirs(OUT, exist_ok=True)
    from gigalens_research.simtests.system import System
    from gigalens_research.simtests.experiments.lenstronomy_point_source import (
        build_epl_shear_point_source_obs)
    import jax
    print("devices:", jax.devices(), flush=True)

    with open(os.path.join(BASE, "diagnostics",
                           "double_filtered_sbc.json")) as fh:
        p7 = json.load(fh)

    sids = sorted(os.path.basename(p)
                  for p in os.listdir(os.path.join(BASE, "runs")))
    global _PROBS
    _PROBS = {}
    for sid in sids:
        s = System.load(DS, sid)
        _PROBS[sid] = build_epl_shear_point_source_obs(
            s, **BUILDER_KW).prob_model
    print(f"[setup] {len(_PROBS)} prob models built", flush=True)

    out = {"target12": TARGET12,
           "p7_kept_frac": {s: p7["kept_frac_per_system"].get(s)
                            for s in TARGET12}}
    out["D_loglik"] = part_d(sids)
    print("[D]", json.dumps(out["D_loglik"]["pit_deciles"]), flush=True)
    out["B_healthy_deltas"] = part_b(p7["kept_frac_per_system"])
    out["A_neff"] = part_a(sids)
    out["C_lenstronomy"] = part_c(TARGET12)

    with open(os.path.join(OUT, "p10_analysis.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("P10 ANALYSIS DONE", flush=True)


if __name__ == "__main__":
    main()
