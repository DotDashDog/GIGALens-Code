"""T23 -- momentum-space xi accounting (GPU stage).

Pre-registered 2026-07-03 (docs/logs/why-hard-to-sample.md). Reads the SAVED
T21 chains (results phase only) and, per arm, builds the momentum-space census
that discriminates the competing cause hypotheses for the stationary xi tail:

  HW (end-wall)     : spikes generated at a hard prior wall (bijector saturation)
                      -> spike steps sit against a bounded-param wall (large |z_k|)
                      -> reversal-like turn angles on that coordinate.
  HS (stiff-dir)    : spikes = integrator-stability events where eps*sqrt(c_t)~O(1)
                      -> median c_t(spike)/c_t(calm) large.
  H0 (neither)      : spikes at unremarkable local geometry -> per-step probing
                      closes; remaining candidate is trajectory-level accumulation.

REGISTERED PREDICTION (orchestrator): HW favored -- spike median |z_Rs| (new;
old: Rs or alpha_Rs) at >= 0.80 quantile of calm |z_Rs|, reversal turn angles on
the same coordinate; classified STRUCTURAL. Thresholds live in t23_t24_analyze.py
(adjudication is a SEPARATE, login-node step); this stage only PRODUCES the
UNCERTIFIED census arrays.

xi<->displacement alignment and the Sigma metric are traced from source in
t23_t24_common.py (module docstring, with file:line citations) and echoed into
every output json. Curvature along motion:
    c_t      = Dz'(-H(z_t))Dz / (Dz' Sigma^{-1} Dz)
    c_t_eucl = Dz'(-H(z_t))Dz / ||Dz||^2
with Dz_t = position[t]-position[t-1] (INCOMING; generates xi[t]), H via HVP
(float64, WHTS_CONV_PRECISION=float64). CLONE control: analytic -H = clone
precision (no HVPs).

Outputs (HARDCODED dir; never a CLI arg -- see run_t21.sh "0" mishap):
  results_carousel/phaseC/t23/t23_{arm}.npz  + t23_{arm}.manifest.json
  (--limit N -> t23_{arm}_smoke.npz, caps spike & calm at N each.)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np

import t23_t24_common as C
from t23_t24_common import ARMS


def _now():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# jax-backed HVP / Hessian machinery (lazy)
# ---------------------------------------------------------------------------

def build_jax_ops(model_seq, hvp_chunk):
    """Return (f_np, hvp_chunked, dense_hessian). float64 assumed."""
    import jax
    import jax.numpy as jnp

    def f(z):
        return model_seq.prob_model.log_prob(z[None])[0][0]

    gradf = jax.grad(f)

    def _hvp_one(z, v):
        return jax.jvp(gradf, (z,), (v,))[1]

    hvp_batch = jax.jit(jax.vmap(_hvp_one))
    hess = jax.jit(jax.hessian(f))
    f_batch = jax.jit(jax.vmap(f))

    def f_np(Z):
        Z = np.asarray(Z, np.float64)
        out = np.empty(Z.shape[0], np.float64)
        for s in range(0, Z.shape[0], 16):
            e = min(s + 16, Z.shape[0])
            out[s:e] = np.asarray(f_batch(jnp.asarray(Z[s:e]))).reshape(-1)
        return out

    def hvp_chunked(Z, V, tag=""):
        """H(z_i) @ v_i for aligned (Z,V) of shape (N,14). Chunked; float64."""
        Z = np.asarray(Z, np.float64); V = np.asarray(V, np.float64)
        n = Z.shape[0]
        out = np.empty_like(Z)
        t0 = time.perf_counter()
        nch = (n + hvp_chunk - 1) // hvp_chunk
        for ci, s in enumerate(range(0, n, hvp_chunk)):
            e = min(s + hvp_chunk, n)
            out[s:e] = np.asarray(hvp_batch(jnp.asarray(Z[s:e]), jnp.asarray(V[s:e])))
            if ci % 10 == 0 or e == n:
                print(f"    [hvp {tag}] chunk {ci+1}/{nch} rows {e}/{n} "
                      f"({time.perf_counter()-t0:.1f}s)", flush=True)
        return out

    def dense_hessian(z):
        return np.asarray(hess(jnp.asarray(np.asarray(z, np.float64))), np.float64)

    return f_np, hvp_chunked, dense_hessian


# ---------------------------------------------------------------------------
# per-step column assembly (numpy, given a curvature-numerator provider)
# ---------------------------------------------------------------------------

def assemble_columns(sel, runs_by_seed, sigma_inv_by_seed, numer_fn, eigvecs, arm, grp):
    """Build all per-step T23 columns for a selected set `sel`
    (dict seed/chain/step/xi). `numer_fn(Zt, Dz, tag) -> num` returns the
    Dz'(-H(z_t))Dz numerator array (real: HVP; clone: analytic quad).
    `runs_by_seed[s]` -> run dict; `sigma_inv_by_seed[s]` -> Sigma^{-1}.
    Returns a dict of arrays."""
    n = len(sel["step"])
    dim = eigvecs.shape[0]
    Zt = np.empty((n, dim)); Dz = np.empty((n, dim))
    turn_prev = np.full(n, np.nan); turn_next = np.full(n, np.nan)
    denom_pre = np.empty(n); denom_euc = np.empty(n)
    for k in range(n):
        s = int(sel["seed"][k]); c = int(sel["chain"][k]); t = int(sel["step"][k])
        pos = runs_by_seed[s]["position"]           # (C,4000,14)
        z_t = pos[c, t]
        dz = pos[c, t] - pos[c, t - 1]              # incoming (generates xi[t])
        Zt[k] = z_t; Dz[k] = dz
        # turn angles: guard segment edges (need t-2 and t+1 to exist)
        if t - 2 >= 0:
            dz_prev = pos[c, t - 1] - pos[c, t - 2]
            turn_prev[k] = _cos(dz_prev, dz)
        if t + 1 < pos.shape[1]:
            dz_next = pos[c, t + 1] - pos[c, t]
            turn_next[k] = _cos(dz, dz_next)
        Si = sigma_inv_by_seed[s]
        denom_pre[k] = float(dz @ Si @ dz)
        denom_euc[k] = float(dz @ dz)
    num = numer_fn(Zt, Dz, f"{arm}:{grp}")          # Dz'(-H)Dz
    with np.errstate(divide="ignore", invalid="ignore"):
        c_t = num / denom_pre
        c_t_eucl = num / denom_euc
    # C3 loadings of the UNIT incoming displacement on the fixed eigenbasis
    norms = np.linalg.norm(Dz, axis=1, keepdims=True)
    Dz_hat = np.where(norms > 0, Dz / norms, 0.0)
    loadings = Dz_hat @ eigvecs                       # (n,dim); col j = load on eigvec j
    eps = np.array([runs_by_seed[int(sel["seed"][k])]["step_size"]
                    [int(sel["chain"][k]), int(sel["step"][k])] for k in range(n)])
    return {
        "z": Zt, "Dz": Dz, "c_t": c_t, "c_t_eucl": c_t_eucl,
        "num_negH_quad": num, "denom_precond": denom_pre, "denom_eucl": denom_euc,
        "eps": eps, "turn_prev": turn_prev, "turn_next": turn_next,
        "loadings": loadings,
        "seed": sel["seed"], "chain": sel["chain"], "step": sel["step"], "xi": sel["xi"],
    }


def _cos(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else np.nan


# ---------------------------------------------------------------------------
# per-arm driver
# ---------------------------------------------------------------------------

def run_arm(arm, hvp_chunk, limit):
    import jax.numpy as jnp
    from common import load_target

    print(f"\n========== T23 arm={arm} ==========", flush=True)
    model_seq, qz, z_center, dim, param_names = load_target(ARMS[arm]["system_dir"])
    bnd = C.derive_bounded(param_names, arm)
    f_np, hvp_chunked, dense_hessian = build_jax_ops(model_seq, hvp_chunk)

    # --- reference point z_ref (recover the exact typical-set init draw) ------
    chain, tstep = ARMS[arm]["z_init_ct"]
    ref_sz = np.load(ARMS[arm]["ref_samples"])["samples_z"]
    z_ref = np.asarray(ref_sz[chain, tstep], np.float64)
    lp_ref = float(f_np(z_ref[None])[0])
    lp_expected = ARMS[arm]["z_init_logp"]
    lp_gap = abs(lp_ref - lp_expected)
    z_ref_source = "z_init_recovered(ref_samples[chain,t])"
    print(f"[t23:{arm}] z_ref = ref_samples[{chain},{tstep}]; logp={lp_ref:.4f} "
          f"expected={lp_expected:.4f} gap={lp_gap:.4g}", flush=True)
    if lp_gap > 5.0:
        print(f"[t23:{arm}] *** WARN z_ref logp gap {lp_gap:.3g} > 5 nats; "
              f"falling back to pooled results-phase median as reference point.")
        runs_tmp = [(s, C.load_run(C.seed_npz(arm, s))) for s in C.SEEDS]
        pooled = np.concatenate([r["position"][:, C.RESULTS_LO:C.RESULTS_HI]
                                 .reshape(-1, dim) for _, r in runs_tmp], axis=0)
        z_ref = np.median(pooled, axis=0)
        lp_ref = float(f_np(z_ref[None])[0])
        z_ref_source = "pooled_results_median(FALLBACK)"

    # --- fixed reference eigenbasis of -H(z_ref) -----------------------------
    H_ref = dense_hessian(z_ref)                       # (14,14)
    negH = -0.5 * (H_ref + H_ref.T)                    # symmetrize
    ref_eigvals, ref_eigvecs = np.linalg.eigh(negH)    # ascending
    order = np.argsort(ref_eigvals)[::-1]              # descending (stiffest first)
    ref_eigvals = ref_eigvals[order]; ref_eigvecs = ref_eigvecs[:, order]
    print(f"[t23:{arm}] eig(-H(z_ref)) range [{ref_eigvals.min():.4g}, "
          f"{ref_eigvals.max():.4g}] (n_neg={int((ref_eigvals<0).sum())})", flush=True)

    # =====================================================================
    # REAL arm (seeds 1,2,3)
    # =====================================================================
    runs = {s: C.load_run(C.seed_npz(arm, s)) for s in C.SEEDS}
    sig_inv = {}
    for s in C.SEEDS:
        Sigma, dmax = C.frozen_sigma(runs[s], arm, s)
        sig_inv[s] = C.sigma_inv(Sigma)
        print(f"[t23:{arm}] seed{s} Sigma frozen (max diff {dmax:.1e})", flush=True)

    n_cap = limit if limit else C.N_SPIKE_MAX
    spike, calm, sel_meta = C.select_spike_calm(
        [(s, runs[s]) for s in C.SEEDS], arm, n_spike=n_cap, n_calm=n_cap)
    print(f"[t23:{arm}] REAL spike n={sel_meta['n_spike']} "
          f"(top{C.XI_TOP_FRAC} always={sel_meta['n_always_top']}, "
          f"thr={sel_meta['xi_top_thresh']:.2f})  calm n={sel_meta['n_calm']}", flush=True)

    def real_numer(Zt, Dz, tag):
        Hv = hvp_chunked(Zt, Dz, tag)                  # H(z_t) @ Dz
        return -np.einsum("ij,ij->i", Dz, Hv)          # Dz'(-H)Dz

    real_spike = assemble_columns(spike, runs, sig_inv, real_numer, ref_eigvecs, arm, "spike")
    real_calm = assemble_columns(calm, runs, sig_inv, real_numer, ref_eigvecs, arm, "calm")
    print(f"[t23:{arm}] REAL median c_t spike={np.nanmedian(real_spike['c_t']):.4g} "
          f"calm={np.nanmedian(real_calm['c_t']):.4g}", flush=True)

    # =====================================================================
    # CLONE control (seed 1 only; analytic -H = clone precision)
    # =====================================================================
    clone_run = C.load_run(C.clone_run_npz(arm))
    Sig_c, dmax_c = C.frozen_sigma(clone_run, arm, "clone")
    sig_inv_c = {C.CLONE_SEED: C.sigma_inv(Sig_c)}
    P_clone, _L = C.clone_precision(arm)               # -H = precision (constant)
    # clone reference eigenbasis = eig(-H) = eig(P_clone) (constant everywhere)
    cl_eigvals, cl_eigvecs = np.linalg.eigh(0.5 * (P_clone + P_clone.T))
    corder = np.argsort(cl_eigvals)[::-1]
    cl_eigvals = cl_eigvals[corder]; cl_eigvecs = cl_eigvecs[:, corder]

    cl_spike, cl_calm, cl_meta = C.select_spike_calm(
        [(C.CLONE_SEED, clone_run)], arm, n_spike=n_cap, n_calm=n_cap)
    print(f"[t23:{arm}] CLONE spike n={cl_meta['n_spike']} calm n={cl_meta['n_calm']} "
          f"(clone Sigma frozen max diff {dmax_c:.1e})", flush=True)

    def clone_numer(Zt, Dz, tag):
        return C.quad(P_clone, Dz)                      # Dz' P Dz = Dz'(-H)Dz

    clone_runs = {C.CLONE_SEED: clone_run}
    clone_spike = assemble_columns(cl_spike, clone_runs, sig_inv_c, clone_numer,
                                   cl_eigvecs, arm, "clone_spike")
    clone_calm = assemble_columns(cl_calm, clone_runs, sig_inv_c, clone_numer,
                                  cl_eigvecs, arm, "clone_calm")

    # =====================================================================
    # save npz + manifest
    # =====================================================================
    npz = {}
    for grp, d in [("real_spike", real_spike), ("real_calm", real_calm),
                   ("clone_spike", clone_spike), ("clone_calm", clone_calm)]:
        for k, v in d.items():
            npz[f"{grp}_{k}"] = np.asarray(v)
    npz["z_ref"] = z_ref
    npz["ref_eigvals"] = ref_eigvals
    npz["ref_eigvecs"] = ref_eigvecs
    npz["clone_eigvals"] = cl_eigvals
    npz["clone_eigvecs"] = cl_eigvecs
    npz["clone_precision"] = P_clone
    npz["param_names"] = np.array(list(param_names))
    npz["bounded_idx"] = np.array(bnd["bounded_idx"], dtype=np.int64)

    os.makedirs(C.T23_OUT, exist_ok=True)
    suffix = "_smoke" if limit else ""
    npz_path = os.path.join(C.T23_OUT, f"t23_{arm}{suffix}.npz")
    np.savez(npz_path, **npz)
    print(f"[t23:{arm}] wrote {npz_path}", flush=True)

    manifest = {
        "experiment": "T23 momentum-space xi accounting (carousel)",
        "arm": arm,
        "status": "proposed (UNCERTIFIED)",
        "timestamp_utc": _now(),
        "smoke_limit": (int(limit) if limit else None),
        "hvp_chunk": int(hvp_chunk),
        "xi_alignment": C.XI_ALIGNMENT,
        "metric_note": C.METRIC_NOTE,
        "seed_subsample": C.SEED,
        "registered_constants": {
            "XI_SPIKE_THRESH": C.XI_SPIKE_THRESH, "XI_TOP_FRAC": C.XI_TOP_FRAC,
            "N_SPIKE_MAX": C.N_SPIKE_MAX, "N_CALM_MAX": C.N_CALM_MAX,
            "SEEDS": list(C.SEEDS), "results_phase": [C.RESULTS_LO, C.RESULTS_HI],
        },
        "param_names": list(param_names),
        "bounded": {
            "idx": bnd["bounded_idx"], "names": bnd["bounded_names"],
            "tokens": bnd["bounded_tokens"], "rs_index": bnd["rs_index"],
            "alpha_rs_index": bnd["alpha_rs_index"],
        },
        "z_ref": {"source": z_ref_source, "chain": int(chain), "t": int(tstep),
                  "logp": lp_ref, "logp_expected": lp_expected, "logp_gap": lp_gap},
        "ref_eig_range": [float(ref_eigvals.min()), float(ref_eigvals.max())],
        "real_selection_meta": sel_meta,
        "clone_selection_meta": cl_meta,
        "counts": {"real_spike": len(real_spike["step"]),
                   "real_calm": len(real_calm["step"]),
                   "clone_spike": len(clone_spike["step"]),
                   "clone_calm": len(clone_calm["step"])},
    }
    man_path = os.path.join(C.T23_OUT, f"t23_{arm}{suffix}.manifest.json")
    with open(man_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[t23:{arm}] wrote {man_path}", flush=True)
    n_hvp = len(real_spike["step"]) + len(real_calm["step"]) + dim   # +dim for dense Hessian
    return n_hvp


def main(argv=None):
    ap = argparse.ArgumentParser(description="T23 momentum-space xi accounting (GPU)")
    ap.add_argument("--arm", choices=["old", "new", "both"], default="both")
    ap.add_argument("--hvp-chunk", type=int, default=C.HVP_CHUNK_DEFAULT)
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke: cap spike & calm at N each; writes *_smoke.npz")
    args = ap.parse_args(argv)

    from common import assert_x64
    assert_x64()
    print(f"[t23] float64 asserted; WHTS_CONV_PRECISION="
          f"{os.environ.get('WHTS_CONV_PRECISION')}; hvp_chunk={args.hvp_chunk} "
          f"limit={args.limit or 'none'}", flush=True)

    arms = ["old", "new"] if args.arm == "both" else [args.arm]
    t0 = time.perf_counter()
    total_hvp = 0
    for arm in arms:
        total_hvp += run_arm(arm, args.hvp_chunk, args.limit)
    wall = time.perf_counter() - t0
    print(f"\n[t23] DONE arms={arms} total_HVPs~{total_hvp} wall={wall:.1f}s "
          f"(PROPOSED / UNCERTIFIED)", flush=True)


if __name__ == "__main__":
    main()
