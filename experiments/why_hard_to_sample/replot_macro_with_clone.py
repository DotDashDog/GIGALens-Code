"""Post-processing for T3: replot the macro transects with the T1 Gaussian
clone's EXACT transect along the same lines as the null reference.

Why: the raw macro plot's nat-drops are expected even for a Gaussian (a straight
line exits a rotated stiff ridge long before covering the marginal width
sigma_dir; the marginal/conditional width ratio is ~65 here, so ~8000-nat drops
at 2*sigma_dir are the Gaussian NULL, not a finding). The finding is deviation
from the clone's parabola: shoulders, asymmetry, non-monotonicity.

Direction vectors are not stored in the T3 JSON (only meta.index), so they are
reconstructed exactly (same rng stream for randoms, eigh for eig directions,
axes from param_names) and VERIFIED against the stored per-direction g_dot_e
(sign-sensitive, uses stored grad_ad) and sigma_dir before plotting; eigenvector
signs are flipped to match g_dot_e when LAPACK sign conventions differ. Raises
on any mismatch -- never plots an unverified overlay.

Usage:
  python replot_macro_with_clone.py --t3-json results_t0t1/sys60/t3/float64/t3_results_float64.json \
      --clone results_t0t1/sys60/clone.npz --out results_t0t1/sys60/t3/float64/t3_macro_with_clone_ref.png
"""
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular


def _unit(v):
    return v / np.linalg.norm(v)


def rebuild_directions(j, cov):
    dim = j["dim"]
    names = j["param_names"]
    seed = j["args"]["seed"]
    rng = np.random.default_rng(seed)
    rand = [_unit(rng.standard_normal(dim)) for _ in range(3)]
    w, V = np.linalg.eigh(cov)
    by_name = {
        "random_0": rand[0], "random_1": rand[1], "random_2": rand[2],
        "stiffest": _unit(V[:, 0]), "softest": _unit(V[:, -1]),
    }
    for d in j["directions"]:
        if d["kind"] == "axis":
            pname = d["name"][len("axis["):-1]
            e = np.zeros(dim)
            e[names.index(pname)] = 1.0
            by_name[d["name"]] = e
    return by_name


def verify_direction(d, e, grad_ad, cov):
    g_dot_e = float(np.dot(grad_ad, e))
    stored = d["g_dot_e"]
    if not np.isclose(g_dot_e, stored, rtol=1e-6, atol=1e-12):
        if np.isclose(-g_dot_e, stored, rtol=1e-6, atol=1e-12):
            e = -e  # eigh sign convention differs; transect direction flips
        else:
            raise ValueError(
                f"{d['name']}: reconstructed g.e={g_dot_e:.9g} does not match "
                f"stored {stored:.9g} (either sign) -- reconstruction invalid.")
    sigma = float(np.sqrt(e @ cov @ e))
    if not np.isclose(sigma, d["sigma_dir"], rtol=1e-6):
        raise ValueError(
            f"{d['name']}: reconstructed sigma_dir={sigma:.9g} != stored "
            f"{d['sigma_dir']:.9g} -- reconstruction invalid.")
    return e


def clone_delta_logp(x0, e, ts, mean, chol):
    """Delta logp of the clone Gaussian along x0 + t*e, exact via Cholesky."""
    r0 = solve_triangular(chol, x0 - mean, lower=True)
    re = solve_triangular(chol, e, lower=True)
    q0 = r0 @ r0
    out = []
    for t in ts:
        r = r0 + t * re
        out.append(-0.5 * (r @ r - q0))
    return np.asarray(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t3-json", required=True)
    ap.add_argument("--clone", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--x-norm", choices=["marginal", "conditional"], default="marginal",
                    help="x-axis units: 'marginal' = sigma_dir = sqrt(e'Se) (the distance a "
                         "chain must traverse to decorrelate); 'conditional' = 1/sqrt(e'S^-1 e) "
                         "(local valley width -- makes the Gaussian null the universal -t^2/2 "
                         "parabola in every panel; the per-direction range in these units "
                         "equals 2x the marginal/conditional ratio = the rotation amplification)")
    args = ap.parse_args()

    j = json.load(open(args.t3_json))
    c = np.load(args.clone)
    mean, cov, chol = c["mean"], c["cov"], c["cholesky"]
    grad_ad = np.asarray(j["grad_ad"], dtype=np.float64)

    src = np.load(j["x0_provenance"]["samples_path"])
    flat = np.asarray(src["position"], dtype=np.float64).reshape(-1, j["dim"])
    x0 = flat[j["x0_provenance"]["flat_index"]]

    by_name = rebuild_directions(j, cov)

    n = len(j["directions"])
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.4 * nrows),
                             squeeze=False)
    axes = axes.ravel()
    for ax, d in zip(axes, j["directions"]):
        e = verify_direction(d, by_name[d["name"]], grad_ad, cov)
        ts = np.asarray(d["macro_t"])
        f = np.asarray(d["macro_f"]) - d["f0"]
        ref = clone_delta_logp(x0, e, ts, mean, chol)
        if args.x_norm == "marginal":
            tnorm = ts / d["sigma_dir"]
            xlabel = "t / sigma_dir (marginal width)"
            title_extra = f"sigma_dir={d['sigma_dir']:.3g}"
        else:
            # conditional width along the line: 1/sqrt(e' S^-1 e), via Cholesky
            re_ = solve_triangular(chol, e, lower=True)
            cond_w = 1.0 / np.sqrt(re_ @ re_)
            tnorm = ts / cond_w
            ratio = d["sigma_dir"] / cond_w
            xlabel = "t / conditional width (null = -t^2/2)"
            title_extra = f"marg/cond ratio={ratio:.3g}"
        ax.plot(tnorm, f, "-o", ms=2.5, lw=1.2, color="#1f5f7a", label="real logp")
        ax.plot(tnorm, ref, "--", lw=1.4, color="#8a3b3b",
                label="T1 clone (Gaussian null)")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(f"{d['name']}  ({title_extra})", fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel("dlogp", fontsize=8)
        ax.tick_params(labelsize=7)
    axes[0].legend(fontsize=7)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("T3 macro transects vs the T1 clone's exact transect "
                 "(deviation from dashed = the finding)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=120)
    print(f"[replot] all {n} directions verified against stored g_dot_e/sigma_dir")
    print(f"[replot] wrote {args.out}")


if __name__ == "__main__":
    main()
