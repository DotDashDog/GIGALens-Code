"""T25a -- Rs conditional-width profile lambda_Rs(z_Rs) (GPU stage).

Pre-registered 2026-07-03 (docs/logs/why-hard-to-sample.md, section "T25 ... +
T26 ..."). Scope: the NEW arm only (NFW_ELLIPSE_EINSTEIN, theta_E~N(13,1)) -- the
better baseline (7.2x clone gap, tight seeds).

WHAT THIS MEASURES
------------------
The conditional Rs precision along the unconstrained Rs coordinate z_Rs:

    lambda_Rs(z) = e_Rs' (-H(z)) e_Rs        (diagonal conditional precision)

where H = Hessian of f(z) = log_prob(z[None])[0][0] and e_Rs is the one-hot at the
z_Rs sampler column. 1 HVP/point (v = e_Rs), conv float64. The MCLMC tuner
suppresses eps ~3.45x because lambda_Rs explodes at low z_Rs (the Rs funnel);
T25b/T26 flatten it. This stage ONLY produces the UNCERTIFIED profile.

Two point groups (pre-reg T25a):
  (i)  ON-FLOOR: pooled T21 new-arm results-phase draws (seeds 1,2,3), z_Rs binned
       into 12 bins over the visited range, 8 draws/bin (deterministic,
       RandomState(20260703)) -> ~96 pts, 1 HVP each.
  (ii) BELOW-FLOOR (never visited): from the 8 lowest-z_Rs pooled draws, transects
       in z_Rs ONLY (all other coords frozen) down to Rs~22, 16 pts each -> ~128.
       Named blind spot (pre-reg): frozen-others slices below the floor measure
       SLICE curvature -> upper envelope of the true conditional there.

Also computed:
  * theta_E* = median theta_E (THETA space, via bij.forward) over ~2000 subsampled
    pooled draws -- the Route-B reference arc radius.
  * lambda at the T21 typical-set init point z_init: recovered EXACTLY as t23 did
    (ref_samples[chain,t] with (chain,t)=ARMS['new']['z_init_ct'], gated on the
    recorded logp ARMS['new']['z_init_logp']).

HVP machinery is the T23 pattern (jax.jvp over grad of the scalar log_prob;
vmapped, chunk<=8) -- imported from t23_momentum_gpu to avoid divergence.

Output (HARDCODED dir; never a CLI arg):
  results_carousel/phaseC/t25/profile.npz + profile.manifest.json
  status: proposed (UNCERTIFIED).
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

HERE = os.path.dirname(os.path.abspath(__file__))
T25_OUT = os.path.join(HERE, "results_carousel", "phaseC", "t25")

# registered constants (NEVER tune)
SELECT_SEED = 20260703          # RandomState for the 8-draws/bin selection
N_BINS = 12                     # z_Rs bins over the visited range
DRAWS_PER_BIN = 8               # on-floor draws per bin
N_LOWEST = 8                    # below-floor transect seed draws (lowest z_Rs)
TRANSECT_PTS = 16               # points per below-floor transect
RS_FLOOR_TARGET = 22.0          # transect down to Rs ~ 22
N_THETAE_SUB = 2000             # subsample for theta_E* median
ARM = "new"                     # pre-reg scope


def _now():
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# HVP with a FIXED direction v = e_Rs (lambda_Rs = -(H e_Rs)[rs])
# ---------------------------------------------------------------------------

def build_lambda_op(model_seq, rs_index, dim, hvp_chunk=8):
    """Return lambda_batch(Z) -> lambda_Rs at each row of Z (N,dim), chunked."""
    import jax
    import jax.numpy as jnp

    def f(z):
        return model_seq.prob_model.log_prob(z[None])[0][0]

    gradf = jax.grad(f)
    e_rs = jnp.asarray(np.eye(dim, dtype=np.float64)[rs_index])

    def _hvp_e(z):
        return jax.jvp(gradf, (z,), (e_rs,))[1]        # H(z) @ e_Rs

    hvp_batch = jax.jit(jax.vmap(_hvp_e))

    def lambda_batch(Z, tag=""):
        Z = np.asarray(Z, np.float64)
        n = Z.shape[0]
        out = np.empty(n, np.float64)
        t0 = time.perf_counter()
        nch = (n + hvp_chunk - 1) // hvp_chunk
        for ci, s in enumerate(range(0, n, hvp_chunk)):
            e = min(s + hvp_chunk, n)
            Hv = np.asarray(hvp_batch(jnp.asarray(Z[s:e])))     # (chunk,dim)
            out[s:e] = -Hv[:, rs_index]                          # e_Rs'(-H)e_Rs
            if ci % 5 == 0 or e == n:
                print(f"    [hvp {tag}] chunk {ci+1}/{nch} rows {e}/{n} "
                      f"({time.perf_counter()-t0:.1f}s)", flush=True)
        return out

    return lambda_batch


def build_rs_map(model_seq, rs_index, dim):
    """Return Rs(z_rs) and its inverse via the REAL Uniform(20,100) leaf (probe
    the baseline bijector on a z-grid varying only the rs column). Exact through
    the actual sigmoid leaf -- no closed-form assumption."""
    import jax.numpy as jnp
    grid = np.linspace(-12.0, 12.0, 4001)
    cols = [np.zeros_like(grid) for _ in range(dim)]
    cols[rs_index] = grid
    out = model_seq.prob_model.bij.forward([jnp.asarray(c) for c in cols])
    # find the Rs key (token 'Rs', not 'alpha_Rs')
    rs_key = _find_key(out, "Rs")
    Rs_grid = np.asarray(out[rs_key], np.float64)
    order = np.argsort(Rs_grid)
    Rs_s, z_s = Rs_grid[order], grid[order]

    def Rs_of_z(z):
        return np.interp(np.asarray(z, np.float64), grid, Rs_grid)

    def z_of_Rs(Rs):
        return np.interp(np.asarray(Rs, np.float64), Rs_s, z_s)

    return Rs_of_z, z_of_Rs, rs_key, grid, Rs_grid


def _find_key(d, token):
    cand = [k for k in d.keys() if str(k) == token or str(k).endswith(token)]
    # longest-suffix wins so 'alpha_Rs' does not shadow 'Rs'
    cand = [k for k in cand if not (token == "Rs" and str(k).endswith("alpha_Rs"))]
    if len(cand) != 1:
        raise ValueError(f"could not uniquely resolve token {token!r}: {cand}")
    return cand[0]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description="T25a Rs conditional-width profile (GPU)")
    ap.add_argument("--limit", type=int, default=0,
                    help="smoke: cap on-floor bins and below-floor seeds; writes *_smoke")
    ap.add_argument("--hvp-chunk", type=int, default=C.HVP_CHUNK_DEFAULT)
    args = ap.parse_args(argv)

    from common import assert_x64, load_target
    assert_x64()
    print(f"[t25] float64 asserted; WHTS_CONV_PRECISION="
          f"{os.environ.get('WHTS_CONV_PRECISION')}; hvp_chunk={args.hvp_chunk} "
          f"limit={args.limit or 'none'}", flush=True)

    t_start = time.perf_counter()
    sys_dir = ARMS[ARM]["system_dir"]
    model_seq, qz, z_center, dim, param_names = load_target(sys_dir)
    bnd = C.derive_bounded(param_names, ARM)
    rs_index = bnd["rs_index"]
    if rs_index is None:
        raise RuntimeError("could not locate Rs column in the new arm")
    print(f"[t25] rs_index = {rs_index} (param {param_names[rs_index]})", flush=True)

    lambda_batch = build_lambda_op(model_seq, rs_index, dim, args.hvp_chunk)
    Rs_of_z, z_of_Rs, rs_key, rsmap_z, rsmap_Rs = build_rs_map(model_seq, rs_index, dim)

    # --- pool T21 new results-phase draws (seeds 1,2,3) ----------------------
    pool = []
    for s in C.SEEDS:
        run = C.load_run(C.seed_npz(ARM, s))
        pool.append(run["position"][:, C.RESULTS_LO:C.RESULTS_HI].reshape(-1, dim))
    pool = np.concatenate(pool, axis=0)                # (~48000, 14)
    zrs = pool[:, rs_index]
    lo, hi = float(zrs.min()), float(zrs.max())
    print(f"[t25] pooled draws={pool.shape[0]}; visited z_Rs range "
          f"[{lo:.4f}, {hi:.4f}]  (Rs [{Rs_of_z(lo):.2f}, {Rs_of_z(hi):.2f}])",
          flush=True)

    rng = np.random.RandomState(SELECT_SEED)

    # --- (i) ON-FLOOR: 12 bins over visited range, 8 draws/bin ---------------
    n_bins = args.limit if args.limit else N_BINS
    edges = np.linspace(lo, hi, n_bins + 1)
    onfloor_Z, onfloor_bin = [], []
    for b in range(n_bins):
        a, c = edges[b], edges[b + 1]
        sel = np.where((zrs >= a) & (zrs <= c if b == n_bins - 1 else zrs < c))[0]
        if len(sel) == 0:
            continue
        take = DRAWS_PER_BIN if not args.limit else min(2, DRAWS_PER_BIN)
        if len(sel) > take:
            sel = np.sort(rng.choice(sel, size=take, replace=False))
        onfloor_Z.append(pool[sel])
        onfloor_bin.append(np.full(len(sel), b))
    onfloor_Z = np.concatenate(onfloor_Z, axis=0)
    onfloor_bin = np.concatenate(onfloor_bin, axis=0)
    print(f"[t25] on-floor points = {onfloor_Z.shape[0]} ({n_bins} bins)", flush=True)
    onfloor_lam = lambda_batch(onfloor_Z, "onfloor")
    onfloor_zrs = onfloor_Z[:, rs_index]
    onfloor_Rs = Rs_of_z(onfloor_zrs)

    # bin medians (log-space representative used by Route A's lambda_hat)
    bin_med_z, bin_med_lam, bin_med_Rs = [], [], []
    for b in range(n_bins):
        m = onfloor_bin == b
        if not np.any(m):
            continue
        bin_med_z.append(float(np.median(onfloor_zrs[m])))
        bin_med_lam.append(float(np.median(onfloor_lam[m])))
        bin_med_Rs.append(float(np.median(onfloor_Rs[m])))
    bin_med_z = np.array(bin_med_z); bin_med_lam = np.array(bin_med_lam)
    bin_med_Rs = np.array(bin_med_Rs)

    # --- (ii) BELOW-FLOOR transects: 8 lowest-z_Rs draws, down to Rs~22 -------
    n_low = args.limit if args.limit else N_LOWEST
    n_tp = TRANSECT_PTS if not args.limit else 4
    low_order = np.argsort(zrs)[:n_low]
    z_target = float(z_of_Rs(RS_FLOOR_TARGET))
    bf_Z, bf_seed, bf_pt = [], [], []
    for si, ridx in enumerate(low_order):
        base = pool[ridx].copy()
        z_start = float(base[rs_index])
        z_end = min(z_target, z_start)             # go DOWN from the draw's z_Rs
        zline = np.linspace(z_start, z_end, n_tp)
        for pj, zv in enumerate(zline):
            row = base.copy(); row[rs_index] = zv
            bf_Z.append(row); bf_seed.append(si); bf_pt.append(pj)
    bf_Z = np.asarray(bf_Z); bf_seed = np.asarray(bf_seed); bf_pt = np.asarray(bf_pt)
    print(f"[t25] below-floor points = {bf_Z.shape[0]} ({n_low} transects x {n_tp}; "
          f"down to Rs~{RS_FLOOR_TARGET} at z_Rs~{z_target:.3f})", flush=True)
    bf_lam = lambda_batch(bf_Z, "belowfloor")
    bf_zrs = bf_Z[:, rs_index]
    bf_Rs = Rs_of_z(bf_zrs)
    # below-floor transect medians per z_Rs point index (extend Route A's low end)
    # bin the below-floor points onto a coarse z grid and take medians
    bf_grid_edges = np.linspace(bf_zrs.min(), lo, 9)  # from lowest up to the floor
    bf_med_z, bf_med_lam = [], []
    for b in range(len(bf_grid_edges) - 1):
        a, c = bf_grid_edges[b], bf_grid_edges[b + 1]
        m = (bf_zrs >= a) & (bf_zrs < c if b < len(bf_grid_edges) - 2 else bf_zrs <= c)
        if np.any(m):
            bf_med_z.append(float(np.median(bf_zrs[m])))
            bf_med_lam.append(float(np.median(bf_lam[m])))
    bf_med_z = np.array(bf_med_z); bf_med_lam = np.array(bf_med_lam)

    # --- theta_E* = median theta_E over ~2000 subsampled pooled draws --------
    import jax.numpy as jnp
    n_sub = min(N_THETAE_SUB, pool.shape[0])
    sub = np.sort(rng.choice(pool.shape[0], size=n_sub, replace=False))
    theta_out = model_seq.prob_model.bij.forward([jnp.asarray(pool[sub][:, j])
                                                  for j in range(dim)])
    te_key = _find_key(theta_out, "theta_E")
    theta_E_samples = np.asarray(theta_out[te_key], np.float64)
    theta_E_star = float(np.median(theta_E_samples))
    print(f"[t25] theta_E* = median theta_E over {n_sub} draws = {theta_E_star:.4f} "
          f"(CV={np.std(theta_E_samples)/theta_E_star:.2e})", flush=True)

    # --- lambda at z_init (recovered EXACTLY as t23) -------------------------
    chain, tstep = ARMS[ARM]["z_init_ct"]
    ref_sz = np.load(ARMS[ARM]["ref_samples"])["samples_z"]
    z_init = np.asarray(ref_sz[chain, tstep], np.float64)

    def _f_np(Z):
        out = np.empty(Z.shape[0])
        for s in range(0, Z.shape[0], 16):
            e = min(s + 16, Z.shape[0])
            lp, _ = model_seq.prob_model.log_prob(jnp.asarray(Z[s:e]))
            out[s:e] = np.asarray(lp).reshape(-1)
        return out

    lp_init = float(_f_np(z_init[None])[0])
    lp_expected = ARMS[ARM]["z_init_logp"]
    lp_gap = abs(lp_init - lp_expected)
    print(f"[t25] z_init = ref_samples[{chain},{tstep}]; logp={lp_init:.4f} "
          f"expected={lp_expected:.4f} gap={lp_gap:.4g}", flush=True)
    if lp_gap > 5.0:
        raise RuntimeError(
            f"z_init logp gap {lp_gap:.3g} > 5 nats -- the recorded typical-set init "
            "point does not reproduce; refuse to derive transforms off a wrong point.")
    lam_init = float(lambda_batch(z_init[None], "z_init")[0])
    zrs_init = float(z_init[rs_index])
    Rs_init = float(Rs_of_z(zrs_init))
    print(f"[t25] lambda_Rs(z_init) = {lam_init:.4g} at z_Rs={zrs_init:.4f} "
          f"(Rs={Rs_init:.3f})", flush=True)

    # --- P-T25a direction/magnitude read (printed; adjudicated in analyzer) --
    def _lam_at(zq):
        # nearest on-floor bin median in z for a coarse read
        i = int(np.argmin(np.abs(bin_med_z - zq)))
        return bin_med_lam[i]
    if len(bin_med_z) >= 2:
        lam_hi = _lam_at(3.0); lam_lo = _lam_at(lo)
        tv = (max(bin_med_lam) / max(min(bin_med_lam), 1e-30))
        print(f"[t25] P-T25a read: lambda(z~3)={lam_hi:.4g} lambda(z~{lo:.2f})={lam_lo:.4g}; "
              f"total variation over bins ~{tv:.1f}x (F-T25a fires if <5x)", flush=True)

    # --- save ----------------------------------------------------------------
    os.makedirs(T25_OUT, exist_ok=True)
    suffix = "_smoke" if args.limit else ""
    npz_path = os.path.join(T25_OUT, f"profile{suffix}.npz")
    np.savez(
        npz_path,
        # on-floor
        onfloor_zrs=onfloor_zrs, onfloor_lambda=onfloor_lam, onfloor_bin=onfloor_bin,
        onfloor_Rs=onfloor_Rs,
        bin_med_z=bin_med_z, bin_med_lambda=bin_med_lam, bin_med_Rs=bin_med_Rs,
        # below-floor
        belowfloor_zrs=bf_zrs, belowfloor_lambda=bf_lam, belowfloor_Rs=bf_Rs,
        belowfloor_seed=bf_seed, belowfloor_pt=bf_pt,
        belowfloor_med_z=bf_med_z, belowfloor_med_lambda=bf_med_lam,
        # references
        theta_E_star=np.float64(theta_E_star),
        theta_E_samples=theta_E_samples,
        z_init=z_init, zrs_init=np.float64(zrs_init), Rs_init=np.float64(Rs_init),
        lambda_init=np.float64(lam_init), logp_init=np.float64(lp_init),
        # Rs(z) mapping grid through the REAL Uniform leaf (Route B needs Rs(z))
        rsmap_z=rsmap_z, rsmap_Rs=rsmap_Rs,
        # meta
        rs_index=np.int64(rs_index), dim=np.int64(dim),
        param_names=np.array(list(param_names)),
        visited_zrs_lo=np.float64(lo), visited_zrs_hi=np.float64(hi),
    )
    print(f"[t25] wrote {npz_path}", flush=True)

    manifest = {
        "experiment": "T25a Rs conditional-width profile (carousel NEW arm)",
        "status": "proposed (UNCERTIFIED)",
        "timestamp_utc": _now(),
        "arm": ARM, "smoke_limit": (int(args.limit) if args.limit else None),
        "rs_index": int(rs_index), "rs_key": str(rs_key),
        "theta_E_key": str(te_key), "theta_E_star": theta_E_star,
        "theta_E_CV": float(np.std(theta_E_samples) / theta_E_star),
        "n_onfloor": int(onfloor_Z.shape[0]), "n_belowfloor": int(bf_Z.shape[0]),
        "n_bins": int(n_bins), "draws_per_bin": DRAWS_PER_BIN,
        "select_seed": SELECT_SEED,
        "visited_zrs": [lo, hi], "Rs_floor_target": RS_FLOOR_TARGET,
        "z_init": {"chain": int(chain), "t": int(tstep), "logp": lp_init,
                   "logp_expected": lp_expected, "logp_gap": lp_gap,
                   "zrs": zrs_init, "Rs": Rs_init, "lambda": lam_init},
        "hvp_chunk": int(args.hvp_chunk),
        "conv_precision": os.environ.get("WHTS_CONV_PRECISION"),
        "note": ("below-floor transects freeze all non-Rs coords -> SLICE curvature, "
                 "treated as an UPPER ENVELOPE of the conditional (pre-reg blind spot)."),
    }
    man_path = os.path.join(T25_OUT, f"profile{suffix}.manifest.json")
    with open(man_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[t25] wrote {man_path}", flush=True)
    n_hvp = onfloor_Z.shape[0] + bf_Z.shape[0] + 1
    wall = time.perf_counter() - t_start
    print(f"[t25] DONE ~{n_hvp} HVPs, wall={wall:.1f}s (PROPOSED / UNCERTIFIED)",
          flush=True)


if __name__ == "__main__":
    main()
