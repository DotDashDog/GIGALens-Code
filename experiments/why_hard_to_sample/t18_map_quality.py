"""T18 -- REUSABLE MAP-convergence diagnostic (docs/logs/why-hard-to-sample.md, T18).

Answers the operator question: "how do I tell a converged from an unconverged
MAP when all I have is my (badly-sampled) posterior?" WITHOUT ground truth.

Three independent tests, each printing value / threshold / PASS-WARN-FAIL:

  D1  trajectory-slope   -- is the best-so-far log-posterior still visibly
                            climbing at the end of the optimization?
  D2  Newton decrement   -- exact local test: build g = grad logp and H = hess
                            logp at z_best; if -H is positive-definite the
                            expected one-Newton-step gain is lambda^2/2 with
                            lambda = sqrt(g^T (-H)^{-1} g). A converged MAP has
                            ~0 predicted gain. An indefinite Hessian means
                            z_best is not even a local max -> FAIL outright.
  D3  mode-vs-typical gap-- loglike(z_best) - median(per-sample loglike). A real
                            mode sits ABOVE the typical set by ~dim/2 nats
                            (Gaussian ballpark). A gap <= 0 means the "MAP" is
                            below its own posterior bulk -> under-converged init.

Each test is designed so that D2/D3 are SUFFICIENT to catch the failure the T15-
T17 line diagnosed (the old carousel run's MAP sat 6.7 nats BELOW the sample
median). D1 is NECESSARY-not-sufficient (the reference MAP plateaued at step 421
yet sat 7 nats low) and is documented as such.

`lp_hist` semantics (verified against gigalens/jax/inference.py:MAP with
output_type="best_step", + reference map/arrays.npz shape (num_steps,)):
  lp_hist[k] = max over the n_samples particles of the log-posterior at step k
  (the per-step best particle). It is NOT best-so-far and NOT per-particle, so
  D1 takes np.maximum.accumulate to form the best-so-far envelope.

Library + CLI. Import `run_quality(...)` from a driver (T18 arm does), or run
standalone:
  python t18_map_quality.py --data-dir systems/carousel_min_old \
      --map-arrays <map arrays.npz> [--samples <mclmc arrays.npz>] --label old500
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HARNESS = os.path.dirname(os.path.abspath(__file__))
if HARNESS not in sys.path:
    sys.path.insert(0, HARNESS)

# ---------------------------------------------------------------------------
# REGISTERED thresholds (pre-registered; do NOT tune)
# ---------------------------------------------------------------------------
D1_STILL_CLIMBING_NAT = 0.5     # WARN if best-so-far improves > this over final 10%
D2_FAIL_NAT = 0.5               # FAIL if predicted Newton gain lambda^2/2 > this
D2_PASS_NAT = 0.1               # PASS if lambda^2/2 <= this; WARN in between
D2_EIG_PD_TOL = 0.0             # -H positive-definite iff min eigenvalue > this
D3_PASS_NAT = 1.0               # PASS if gap >= this
D3_FAIL_NAT = 0.0               # FAIL if gap < this; WARN in [FAIL, PASS)
D3_N_SUB = 512                  # subsampled posterior draws for the typical set
D3_BATCH = 64                   # log_like batch size
D3_RNG_SEED = 20260703

BATCH_LL = 64


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _loglike_batched(prob_model, z, batch=BATCH_LL):
    """Per-sample log-likelihood for z (N, dim), batched to bound memory."""
    import jax.numpy as jnp
    out = np.empty(z.shape[0], dtype=np.float64)
    for s in range(0, z.shape[0], batch):
        e = min(s + batch, z.shape[0])
        ll, _ = prob_model.log_like(jnp.asarray(z[s:e]))
        out[s:e] = np.asarray(ll, dtype=np.float64).reshape(-1)
    return out


def _load_samples_z(path):
    """Load posterior draws from either an mclmc arrays.npz ('samples_z',
    shape (chains, draws, dim)) or a harness run npz ('position', shape
    (chains, total, dim)). Returns a flat (N, dim) array, or None if unusable."""
    if not path or not os.path.isfile(path):
        return None
    a = np.load(path)
    if "samples_z" in a.files:
        sz = np.asarray(a["samples_z"], dtype=np.float64)  # post-burn-in only
    elif "position" in a.files:
        sz = np.asarray(a["position"], dtype=np.float64)
        # harness 'position' spans burnin+results; keep the trailing half so
        # burn-in transients don't drag D3's typical-set median down
        sz = sz[:, sz.shape[1] // 2:, :]
    else:
        return None
    return sz.reshape(-1, sz.shape[-1])


# ---------------------------------------------------------------------------
# D1 -- trajectory slope
# ---------------------------------------------------------------------------

def test_d1_trajectory(lp_hist):
    """best-so-far envelope of the per-step best-particle logp; WARN if it is
    still climbing (> D1_STILL_CLIMBING_NAT over the final 10% of steps)."""
    lp = np.asarray(lp_hist, dtype=np.float64).reshape(-1)
    n = lp.size
    bsf = np.maximum.accumulate(lp)
    k = max(0, int(np.floor(0.9 * n)) - 1)
    tail_gain = float(bsf[-1] - bsf[k])
    status = "WARN" if tail_gain > D1_STILL_CLIMBING_NAT else "PASS"
    return {
        "test": "D1_trajectory_slope",
        "value": tail_gain,
        "value_desc": "best-so-far logp gain over final 10% of steps (nats)",
        "threshold": D1_STILL_CLIMBING_NAT,
        "status": status,
        "counts_as_fail": False,   # D1 can only PASS/WARN
        "note": ("NECESSARY not sufficient: a plateau does NOT prove convergence "
                 "(reference old MAP plateaued at step 421 yet sat 7 nats low)."),
        "n_steps": int(n),
        "best_so_far_final": float(bsf[-1]),
    }


# ---------------------------------------------------------------------------
# D2 -- Newton decrement (exact local curvature test)
# ---------------------------------------------------------------------------

def test_d2_newton(prob_model, z_best):
    """Exact g/H of the scalar logp at z_best. If -H is PD, predicted one-step
    Newton gain is lambda^2/2. Indefinite -H => not a local max => FAIL."""
    import jax
    import jax.numpy as jnp

    z0 = jnp.asarray(np.asarray(z_best, dtype=np.float64))

    def f(z):
        logp, _ = prob_model.log_prob(z[None, :])
        return logp[0]

    g = np.asarray(jax.grad(f)(z0), dtype=np.float64)
    H = np.asarray(jax.hessian(f)(z0), dtype=np.float64)
    negH = -0.5 * (H + H.T)  # symmetrize before eig (kill float asymmetry)
    eig = np.linalg.eigvalsh(negH)
    min_eig = float(eig.min())
    grad_norm = float(np.linalg.norm(g))

    if min_eig > D2_EIG_PD_TOL:
        # solve (-H) x = g  ->  lambda^2 = g^T (-H)^{-1} g
        x = np.linalg.solve(negH, g)
        lam2 = float(g @ x)
        lam2 = max(lam2, 0.0)  # guard tiny negative from float noise
        predicted_gain = 0.5 * lam2
        lam = float(np.sqrt(lam2))
        if predicted_gain > D2_FAIL_NAT:
            status, reason = "FAIL", "predicted Newton gain exceeds threshold"
        elif predicted_gain <= D2_PASS_NAT:
            status, reason = "PASS", "at a sharp local maximum"
        else:
            status, reason = "WARN", "small but non-negligible predicted gain"
        return {
            "test": "D2_newton_decrement",
            "value": predicted_gain,
            "value_desc": "predicted one-Newton-step logp gain lambda^2/2 (nats)",
            "threshold_fail": D2_FAIL_NAT,
            "threshold_pass": D2_PASS_NAT,
            "status": status,
            "counts_as_fail": status == "FAIL",
            "lambda": lam,
            "grad_norm": grad_norm,
            "min_eig_negH": min_eig,
            "hessian_pd": True,
            "reason": reason,
        }
    # indefinite -H: z_best is not a local max -- that IS the diagnostic
    neg_eigs = [float(e) for e in eig if e <= D2_EIG_PD_TOL]
    return {
        "test": "D2_newton_decrement",
        "value": float("nan"),
        "value_desc": "predicted Newton gain (undefined: -H not PD)",
        "threshold_fail": D2_FAIL_NAT,
        "threshold_pass": D2_PASS_NAT,
        "status": "FAIL",
        "counts_as_fail": True,
        "grad_norm": grad_norm,
        "min_eig_negH": min_eig,
        "hessian_pd": False,
        "negative_eigenvalues": neg_eigs,
        "reason": "z_best is not at a local max (indefinite Hessian)",
    }


# ---------------------------------------------------------------------------
# D3 -- mode-vs-typical-set gap (zero extra compute; needs posterior draws)
# ---------------------------------------------------------------------------

def test_d3_mode_gap(prob_model, z_best, samples_z, dim):
    """gap = loglike(z_best) - median(per-sample loglike) over ~D3_N_SUB draws.
    Gaussian ballpark: gap ~ +dim/2. gap<0 => mode below its own bulk => FAIL."""
    ballpark = 0.5 * dim
    if samples_z is None:
        return {
            "test": "D3_mode_vs_typical_gap",
            "value": None,
            "value_desc": "loglike(z_best) - median sample loglike (nats)",
            "threshold_pass": D3_PASS_NAT,
            "threshold_fail": D3_FAIL_NAT,
            "status": "N/A",
            "counts_as_fail": False,
            "ballpark_gap": ballpark,
            "reason": "no posterior samples supplied (D3 skipped)",
        }
    import jax.numpy as jnp
    z0 = jnp.asarray(np.asarray(z_best, dtype=np.float64))[None, :]
    ll_map = float(np.asarray(prob_model.log_like(z0)[0]).reshape(-1)[0])

    N = samples_z.shape[0]
    rng = np.random.default_rng(D3_RNG_SEED)
    n_sub = min(D3_N_SUB, N)
    idx = rng.choice(N, n_sub, replace=False)
    ll_samp = _loglike_batched(prob_model, samples_z[idx], batch=D3_BATCH)
    med = float(np.median(ll_samp))
    gap = ll_map - med

    if gap < D3_FAIL_NAT:
        status = "FAIL"
    elif gap < D3_PASS_NAT:
        status = "WARN"
    else:
        status = "PASS"
    return {
        "test": "D3_mode_vs_typical_gap",
        "value": gap,
        "value_desc": "loglike(z_best) - median sample loglike (nats)",
        "threshold_pass": D3_PASS_NAT,
        "threshold_fail": D3_FAIL_NAT,
        "status": status,
        "counts_as_fail": status == "FAIL",
        "ballpark_gap": ballpark,
        "loglike_z_best": ll_map,
        "median_sample_loglike": med,
        "n_sub": int(n_sub),
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

def run_quality(prob_model, z_best, lp_hist, samples_z, dim, label, out_dir=None):
    """Run D1/D2/D3 on one MAP; print a block, optionally write JSON. Returns a
    dict with the three test results + n_fail (FAIL count, N/A does not count)."""
    d1 = test_d1_trajectory(lp_hist)
    d2 = test_d2_newton(prob_model, z_best)
    d3 = test_d3_mode_gap(prob_model, z_best, samples_z, dim)
    tests = [d1, d2, d3]
    n_fail = sum(1 for t in tests if t.get("counts_as_fail"))

    print(f"\n----- MAP quality: {label} -----")
    for t in tests:
        v = t["value"]
        vs = "  N/A" if v is None else (f"{v:+.4g}" if np.isfinite(v) else "  nan")
        print(f"  {t['test']:24s} value={vs:>10}  -> {t['status']}"
              + (f"   ({t.get('reason')})" if t.get("reason") else ""))
    print(f"  => FAIL count = {n_fail}"
          + ("  [D1 is necessary-not-sufficient; PASS there proves nothing]"
             if d1["status"] == "PASS" else ""))

    result = {
        "label": label,
        "dim": int(dim),
        "n_fail": int(n_fail),
        "D1": d1, "D2": d2, "D3": d3,
        "status": "proposed (UNCERTIFIED)",
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, f"quality_{label}.json")
        with open(p, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  wrote {p}")
        result["json_path"] = os.path.abspath(p)
    return result


def run_quality_from_paths(data_dir, map_arrays, samples, label, out_dir):
    """CLI convenience: load the system + a map arrays.npz (+ optional samples)
    and run the trio. Kept separate so drivers can call run_quality directly
    with an already-loaded prob_model (avoids re-importing the model)."""
    from common import assert_x64, load_target
    assert_x64()
    prob_model, _qz, _zc, dim, _names = load_target(data_dir)
    a = np.load(map_arrays)
    z_best = np.asarray(a["z_best"], dtype=np.float64)
    lp_hist = np.asarray(a["lp_hist"], dtype=np.float64)
    samples_z = _load_samples_z(samples)
    return run_quality(prob_model, z_best, lp_hist, samples_z, dim,
                       label, out_dir=out_dir)


def main():
    ap = argparse.ArgumentParser(description="Reusable MAP-convergence diagnostic (D1/D2/D3).")
    ap.add_argument("--data-dir", required=True, help="system module dir (has system.py)")
    ap.add_argument("--map-arrays", required=True, help="npz with z_best + lp_hist")
    ap.add_argument("--samples", default=None,
                    help="optional mclmc arrays.npz (samples_z) or run npz (position) for D3")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out-dir", default=os.path.join(HARNESS, "results_carousel", "old", "t18"))
    args = ap.parse_args()
    run_quality_from_paths(args.data_dir, args.map_arrays, args.samples,
                           args.label, args.out_dir)


if __name__ == "__main__":
    main()
