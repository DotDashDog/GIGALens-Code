"""P-11 pre-registered analysis (docs/logs/point-source-sbc.md, P-11 checkpoint).

Remedy audit for the P-10 falsifier: the in-likelihood coarse operator
(eps 0.1"/grid 384) undercounts a genuine 3rd image on sys_57 (+6 partially
affected), so the lam=10 penalty never engaged. The mc2 arm reruns those 7
systems with the fine operator (eps 0.05"/grid 768/64 tiles) in the
likelihood; this script scores the reruns and the two side deliverables.

Subcommands (driven by p11_remedy.sbatch; each writes its own JSON so a
partial job still leaves scored pieces):

  recount — CPU. lenstronomy truth image counts for ALL 100 systems at
            search_window 12 (the generator counted within 6 — P-10 found 5
            mislabeled doubles among the 14 systems it spot-checked), ladder
            {0.05, 0.025}. Output: corrected label table for future arms.
  rerank  — GPU. Unpenalized loglik rank for all 100 mc1 posteriors (L=512,
            same thinning as the run.json metric) — the pre-registered fix
            for the loglik Holm rejection, which P-10 traced to truth paying
            the coarse operator's penalty error (Spearman rho=0.51 with
            pen(truth)). Also recomputes pen_coarse(truth) and the
            Spearman(pen_truth, PIT_unpenalized) that should now collapse.
  audit   — GPU+CPU, after the mc2 reruns finish. Per rerun system: fine
            AUDIT-config N_eff (768/eps 0.05/window 12", 2000 draws — note
            this now matches the in-likelihood resolution, hence the other
            two checks), a 2x-resolution convergence rung (1536/eps 0.025/
            256 tiles, 500 draws, sys_57), the independent lenstronomy count
            cross-check (500 draws, window 12), the mc2 unpenalized loglik
            rank, and the convergence-gate fields from run.json.

Code: gigalens pinned worktree d321d3c, GIGALens-Code (see checkpoint entry).
"""
import json
import multiprocessing as mp
import os
import sys
import types

import numpy as np

os.environ.setdefault("GIGALENS_ALLOW_LEGACY_JAX", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
# Runtime deps are PINNED at the P-10 campaign/analysis commits (GIGALens-Code
# 4358ebb, gigalens d321d3c): both live checkouts have since moved (the
# ModellingSequence migration changed the builder's return type), and the
# remedy comparison is only controlled if nothing but the operator resolution
# differs from mc1.
sys.path.insert(0, os.path.expanduser("~/GIGALens-Code-worktree-p10/src"))
sys.path.insert(0, os.path.expanduser("~/gigalens-worktree-p10/src"))

from p10_analysis import (BASE, BUILDER_KW, COLS, DS, MC, _count_one,
                          make_neff)

OUT = os.path.join(BASE, "diagnostics", "p11_analysis")

TARGET7 = ["sys_35", "sys_38", "sys_57", "sys_67", "sys_69", "sys_86",
           "sys_99"]
MC_FINE = dict(mu_min=0.1, eps=0.05, lam=10.0, grid_n=768, n_tiles=64,
               window_scale=4.0)
N_NEFF_DRAWS = 2000
N_LT_DRAWS = 500
LAM = 10.0

_SYSTEMS = {}
_PROBS = {}      # (sid, "pen"|"unpen") -> prob model


def _system(sid):
    if sid not in _SYSTEMS:
        from gigalens_research.simtests.system import System
        _SYSTEMS[sid] = System.load(DS, sid)
    return _SYSTEMS[sid]


def _prob(sid, penalized):
    key = (sid, "pen" if penalized else "unpen")
    if key not in _PROBS:
        from gigalens_research.simtests.experiments.lenstronomy_point_source \
            import build_epl_shear_point_source_obs
        kw = dict(BUILDER_KW)
        kw["multiplicity_constraint"] = dict(MC) if penalized else None
        _PROBS[key] = build_epl_shear_point_source_obs(
            _system(sid), **kw).prob_model
    return _PROBS[key]


def _truth_unique_z(sid):
    """Truth as {z_param_name: value} dict + z vector, via the scene mapping
    (same route as the run.json loglik_rank metric)."""
    from gigalens_research.simtests.metrics import _scene_truth
    from gigalens_research.simtests.experiments.lenstronomy_point_source \
        import _truth_unique
    prob = _prob(sid, penalized=False)
    shim = types.SimpleNamespace(scene=prob.model)
    unique = _truth_unique(prob.model, _scene_truth(shim, _system(sid)))
    if unique is None:
        raise RuntimeError(f"{sid}: truth->scene mapping failed")
    z = np.asarray(prob.bij.inverse(unique)).reshape(-1)
    return unique, z


def _thin_z(sid, variant, target):
    from gigalens_research.simtests.experiments.lenstronomy_point_source \
        import _thin_samples
    a = np.load(os.path.join(BASE, "runs", sid, variant, "mclmc",
                             "arrays.npz"))
    return _thin_samples(np.asarray(a["samples_z"]), target)


def _rows(sid, variant, n_draws):
    """(n, 10) draws in COLS order (linspace thinning, as p10 part A/C)."""
    import jax
    import jax.numpy as jnp
    a = np.load(os.path.join(BASE, "runs", sid, variant, "mclmc",
                             "arrays.npz"))
    sz = a["samples_z"].reshape(-1, a["samples_z"].shape[-1])
    sel = np.linspace(0, len(sz) - 1, n_draws).astype(int)
    prob = _prob(sid, penalized=False)
    params = jax.tree_util.tree_map(
        np.asarray, prob.bij.forward(jnp.asarray(sz[sel])))
    return np.stack([np.asarray([params[c][j] for c in COLS])
                     for j in range(len(sel))])


def _unpen_rank(sid, variant):
    """Rank of truth's UNPENALIZED loglik among 512 thinned draws'."""
    import jax.numpy as jnp
    prob = _prob(sid, penalized=False)
    _, z_truth = _truth_unique_z(sid)
    thin = _thin_z(sid, variant, 512)
    lls = []
    for start in range(0, thin.shape[0], 512):
        ll, _ = prob.log_like(jnp.asarray(thin[start:start + 512]))
        lls.append(np.asarray(ll).reshape(-1))
    ll_draws = np.concatenate(lls)
    ll_truth = float(np.asarray(
        prob.log_like(jnp.asarray(z_truth)[None, :])[0]).reshape(-1)[0])
    finite = np.isfinite(ll_draws)
    return {"rank": int(np.sum(ll_draws[finite] < ll_truth)),
            "_L": int(finite.sum()), "ll_truth": ll_truth}


def cmd_rerank():
    import jax
    import jax.numpy as jnp
    from scipy import stats
    print("devices:", jax.devices(), flush=True)
    sids = sorted(os.path.basename(p)
                  for p in os.listdir(os.path.join(BASE, "runs")))
    neff_coarse = jax.jit(make_neff(MC["grid_n"], MC["n_tiles"],
                                    window_scale=MC["window_scale"],
                                    fp32=True))
    neff_fine = jax.jit(make_neff(MC_FINE["grid_n"], MC_FINE["n_tiles"],
                                  window=12.0, fp32=True))
    out = {}
    for sid in sids:
        unique, _ = _truth_unique_z(sid)
        row = jnp.asarray([float(np.asarray(unique[c])) for c in COLS])
        ne_t = float(neff_coarse(row, MC["eps"], MC["mu_min"]))
        ne_ft = float(neff_fine(row, MC_FINE["eps"], MC["mu_min"]))
        pen_t = -LAM * (ne_t - 2.0) ** 2
        r = _unpen_rank(sid, "mc1")
        out[sid] = {**r, "neff_coarse_truth": ne_t, "pen_truth": pen_t,
                    "neff_fine_truth": ne_ft,
                    "pen_fine_truth": -LAM * (ne_ft - 2.0) ** 2}
        # free the per-system compiled log_like
        _PROBS.pop((sid, "unpen"), None)
        print(f"[R] {sid}: rank={r['rank']}/{r['_L']}  "
              f"pen(truth) coarse={pen_t:.3f} "
              f"fine={out[sid]['pen_fine_truth']:.3f}", flush=True)
    pit = np.array([out[s]["rank"] / out[s]["_L"] for s in sids])
    pen = np.array([out[s]["pen_truth"] for s in sids])
    pen_f = np.array([out[s]["pen_fine_truth"] for s in sids])
    ks = stats.kstest(pit, "uniform")
    rho = stats.spearmanr(pen, pit)
    summary = {
        "n": len(sids),
        "mean_pit": float(pit.mean()),
        "pit_deciles": np.histogram(pit, bins=10, range=(0, 1))[0].tolist(),
        "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
        "spearman_pen_pit": float(rho.statistic),
        "spearman_p": float(rho.pvalue),
        # population incidence of intrinsic near-caustic smoothed-count
        # excess: systems whose TRUTH pays a non-trivial fine-config penalty
        # (these are the ones the constraint distorts at any resolution).
        "n_pen_fine_truth_below_05": int(np.sum(pen_f < -0.5)),
        "n_pen_fine_truth_below_2": int(np.sum(pen_f < -2.0)),
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "p11_rerank.json"), "w") as fh:
        json.dump({"summary": summary, "per_system": out}, fh, indent=1)
    print("[R] summary:", json.dumps(summary), flush=True)
    print("P11 RERANK DONE", flush=True)


def cmd_audit():
    import jax
    import jax.numpy as jnp
    print("devices:", jax.devices(), flush=True)
    os.makedirs(OUT, exist_ok=True)
    neff_fine = jax.jit(make_neff(768, 64, window=12.0, fp32=True))
    out = {}
    for sid in TARGET7:
        rows = _rows(sid, "mc2", N_NEFF_DRAWS)
        ne = np.array([float(neff_fine(jnp.asarray(r), 0.05, MC["mu_min"]))
                       for r in rows])
        np.save(os.path.join(OUT, f"neff_mc2_{sid}.npy"), ne)
        with open(os.path.join(BASE, "runs", sid, "mc2", "run.json")) as fh:
            m = json.load(fh)["metrics"]
        out[sid] = {
            "frac_neff_2": float(np.mean(np.abs(ne - 2.0) < 0.35)),
            "neff_q10_50_90": [float(np.percentile(ne, q))
                               for q in (10, 50, 90)],
            "max_rhat": m.get("max_rhat"), "min_ess": m.get("min_ess"),
            "unpen_loglik": _unpen_rank(sid, "mc2"),
        }
        print(f"[A] {sid}: frac(N_eff~2) = {out[sid]['frac_neff_2']:.3f}  "
              f"q10/50/90 = {out[sid]['neff_q10_50_90']}", flush=True)

    # 2x-resolution convergence rung on the falsifier system: does the audit
    # operator itself move when refined again? (|delta| < 0.05 on >= 90%)
    neff_2x = jax.jit(make_neff(1536, 256, window=12.0, fp32=True))
    rows57 = _rows("sys_57", "mc2", 500)
    ne_f = np.array([float(neff_fine(jnp.asarray(r), 0.05, MC["mu_min"]))
                     for r in rows57])
    ne_2x = np.array([float(neff_2x(jnp.asarray(r), 0.025, MC["mu_min"]))
                      for r in rows57])
    out["sys_57_convergence"] = {
        "frac_agree_005": float(np.mean(np.abs(ne_f - ne_2x) < 0.05)),
        "max_abs_delta": float(np.max(np.abs(ne_f - ne_2x))),
    }
    print("[A] sys_57 2x rung:", json.dumps(out["sys_57_convergence"]),
          flush=True)

    # independent cross-check: lenstronomy image counts on mc2 draws
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(8, os.cpu_count() or 8)) as pool:
        for sid in TARGET7:
            rows = _rows(sid, "mc2", N_LT_DRAWS)
            counts = np.array(pool.map(_count_one, [tuple(r) for r in rows]))
            fin = counts[:, -1]
            valid = fin >= 0
            out[sid]["lt_frac_count2"] = float(np.mean(fin[valid] == 2))
            out[sid]["lt_count_hist"] = {
                int(k): int(n) for k, n in
                zip(*np.unique(fin[valid], return_counts=True))}
            np.save(os.path.join(OUT, f"ltcounts_mc2_{sid}.npy"), counts)
            print(f"[C] {sid}: frac(count==2) = "
                  f"{out[sid]['lt_frac_count2']:.3f}  "
                  f"hist = {out[sid]['lt_count_hist']}", flush=True)

    with open(os.path.join(OUT, "p11_audit.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print("P11 AUDIT DONE", flush=True)


def _recount_one(args):
    # window-12 ladder {0.05, 0.025}: P-10's spot check showed the two rungs
    # agree on truth configs; drop the 0.01 rung for speed.
    from p10_analysis import _count_one as c1
    import p10_analysis as p10
    old = p10.LT_LADDER
    p10.LT_LADDER = [0.05, 0.025]
    try:
        return c1(args)
    finally:
        p10.LT_LADDER = old


def cmd_recount():
    sids = sorted(os.path.basename(p)
                  for p in os.listdir(os.path.join(BASE, "runs")))
    rows = {}
    for sid in sids:
        unique, _ = _truth_unique_z(sid)
        rows[sid] = tuple(float(np.asarray(unique[c])) for c in COLS)
        _PROBS.pop((sid, "unpen"), None)
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(4, os.cpu_count() or 4)) as pool:
        counts = pool.map(_recount_one, [rows[s] for s in sids])
    out, mislabeled = {}, []
    for sid, c in zip(sids, counts):
        stable = (len(set(x for x in c if x >= 0)) == 1)
        n = c[-1]
        out[sid] = {"counts": c, "ladder_stable": bool(stable), "n_images": n}
        if n != 2:
            mislabeled.append(sid)
        print(f"[T] {sid}: counts={c}" + ("  MISLABELED" if n != 2 else ""),
              flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "p11_truth_recount_w12.json"), "w") as fh:
        json.dump({"search_window": 12.0, "ladder": [0.05, 0.025],
                   "mislabeled": mislabeled, "per_system": out}, fh, indent=1)
    print(f"[T] mislabeled ({len(mislabeled)}): {mislabeled}", flush=True)
    print("P11 RECOUNT DONE", flush=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "rerank":
        cmd_rerank()
    elif cmd == "audit":
        cmd_audit()
    elif cmd == "recount":
        cmd_recount()
    else:
        raise SystemExit("usage: p11_analysis.py {rerank|audit|recount}")


if __name__ == "__main__":
    main()
