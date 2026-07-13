#!/usr/bin/env python
"""GATE PT-4 link L: the PT-0b equal-swap-cost PT ladder recipe as importable code.

Codifies, for a NEW posterior, the certified carousel pipeline:
probe u-draws -> per-beta sd(u) [+ se] -> log-log-interpolated cost integral
int sd(u) dbeta -> equal-cost knots over [beta_min, 1]; target nats/pair from
desired adjacent acceptance via the validated Gaussian swap model a = erfc(s/2);
beta_min from the pinned ln(100) discovery rule on measured basin-class leakage;
basin classifier = nearest known mode center in pooled-cov whitened distance.

Reference implementation ported EXACTLY (same math, same 4000-point geomspace
grid): pt0_ladder_design.py; archived output ladder_design_power.json.
Login-node safe (numpy + stdlib only), float64. Diagnostic/design tool.
"""
import argparse
import json
import math

import numpy as np

NGRID = 4000  # pt0_ladder_design.py interpolation grid (pinned)


def measure_sd_u(u, discard):
    """Per-beta sd(u) over the retained window, plus a standard error of sd.

    u: (nbeta, ngroups, nsteps, nchain). sd EXACTLY as pt0_ladder_design.py:
    np.std (ddof=0) over the flattened retained window u[k, :, discard:, :].

    se(sd) estimator (batch means): chains/groups are independent; steps are
    autocorrelated. For each (group, chain) series of length T, IAT is
    estimated by batch means with batch length b = floor(sqrt(T)):
    IAT = b * var(batch means, ddof=1) / var(series, ddof=1), clipped to >= 1.
    N_eff_total = sum over the ngroups*nchain series of T/IAT, and
    se(sd) ~= sd / sqrt(2 * N_eff_total) -- the delta method on the Gaussian
    sd estimator (var(sd_hat) ~ sigma^2 / (2 N) for N effectively independent
    draws). Returns (sd, se_sd), each (nbeta,).
    """
    u = np.asarray(u, np.float64)
    nb = u.shape[0]
    sd, se = np.empty(nb), np.empty(nb)
    for k in range(nb):
        w = u[k, :, discard:, :]                       # (ngroups, T, nchain)
        sd[k] = np.std(w.reshape(-1))
        T = w.shape[1]
        s = np.transpose(w, (0, 2, 1)).reshape(-1, T)  # (nseries, T)
        b = max(int(math.sqrt(T)), 2)
        nbat = T // b
        bm = s[:, :nbat * b].reshape(-1, nbat, b).mean(axis=2)
        iat = b * bm.var(axis=1, ddof=1) / np.maximum(s.var(axis=1, ddof=1), 1e-300)
        neff = float((T / np.maximum(iat, 1.0)).sum())
        se[k] = sd[k] / math.sqrt(2.0 * neff)
    return sd, se


def cost_curve(betas, sd):
    """(bgrid, cumulative swap cost in nats): IDENTICAL math to
    pt0_ladder_design.py -- geomspace(betas[0], 1, NGRID) grid, log-log
    interpolation of sd(u), trapezoid cumsum of int sd(u) dbeta."""
    betas = np.asarray(betas, np.float64)
    sd = np.asarray(sd, np.float64)
    bgrid = np.geomspace(betas[0], 1.0, NGRID)
    sdg = np.exp(np.interp(np.log(bgrid), np.log(betas), np.log(np.maximum(sd, 1e-12))))
    cost = np.concatenate([[0.0], np.cumsum(0.5 * (sdg[1:] + sdg[:-1]) * np.diff(bgrid))])
    return bgrid, cost


def _knots_fixed_n(betas, sd, beta_min, n_rungs):
    """Equal-cost knots over [beta_min, 1] at a FIXED rung count."""
    bgrid, cost = cost_curve(betas, sd)
    c0 = float(np.interp(beta_min, bgrid, cost))   # cost measured from beta_min
    knots = np.interp(np.linspace(c0, cost[-1], n_rungs), cost, bgrid)
    knots[0], knots[-1] = beta_min, 1.0
    return knots, float(cost[-1] - c0)


def design_ladder(betas, sd, beta_min, target_nats):
    """Equal-swap-cost ladder over [beta_min, 1]: restrict the cost curve
    (cost_r(beta) = C(beta) - C(beta_min), C(beta_min) by linear interp on the
    NGRID polyline), n_rungs = ceil(total_restricted/target)+1, knots at equal
    cost fractions via np.interp, endpoints pinned to beta_min and 1.0.
    Returns dict(knots, nats_per_pair, n_rungs, total_cost). With
    beta_min = betas[0] this is bitwise pt0_ladder_design.py's full design."""
    bgrid, cost = cost_curve(betas, sd)
    c0 = float(np.interp(beta_min, bgrid, cost))
    total = float(cost[-1] - c0)
    n_rungs = int(np.ceil(total / target_nats)) + 1
    knots = np.interp(np.linspace(c0, cost[-1], n_rungs), cost, bgrid)
    knots[0], knots[-1] = beta_min, 1.0
    return dict(knots=knots, nats_per_pair=total / (n_rungs - 1),
                n_rungs=n_rungs, total_cost=total)


def target_from_acceptance(acc):
    """Desired nats/pair s from desired adjacent swap acceptance acc, inverting
    the validated Gaussian swap model a = erfc(s/2): s = 2*erfcinv(acc).
    erfcinv scipy-free: bisection-safeguarded Newton on math.erfc, float64."""
    if not 0.0 < acc < 2.0:
        raise ValueError(f"acceptance must be in (0, 2), got {acc}")
    lo, hi, x = -10.0, 10.0, 0.0
    for _ in range(200):
        f = math.erfc(x) - acc
        if f > 0.0:
            lo = x          # erfc(x) too large -> x too small
        else:
            hi = x
        xn = x + f * math.sqrt(math.pi) / 2.0 * math.exp(x * x)  # Newton
        if not lo < xn < hi:
            xn = 0.5 * (lo + hi)
        if xn == x:
            break
        x = xn
    return 2.0 * x


def beta_min_rule(betas, leak_rates, nsys, budget_steps, probe_steps, level=0.99):
    """Coldest (largest) probe beta certified for >= `level` discovery within
    budget: nsys * p_hat * (budget_steps/probe_steps) >= ln(1/(1-level)).

    leak_rates: per-beta conservative-DIRECTION (smaller of the two basin-class
    directions) leakage fraction per probe_steps, aligned with betas.

    CONSERVATIVE leak estimate p_hat: leakage falls toward beta = 1, so a
    candidate's own retained-window rate is the optimistic bracket of its
    crossing capability; the pinned rule scores each candidate with
    p_hat = min(own rate, next-colder grid point's rate) [provenance: PT-0b
    grader advisory 7, "the beta = 0.6 figure is the conservative one", and the
    PT-4 design checkpoint's carousel check, which quotes BOTH figures for
    beta_min = 0.3594: 16*0.256*11 = 45 at beta = 0.3594 AND the conservative
    beta = 0.6 figure 0.176 -> 31. NOTE: scoring candidates by their own rate
    alone would return 0.5995 on the carousel table (0.176 -> 31 >= 4.6); only
    the neighbor-conservative form reproduces the certified beta_min = 0.3594].
    The coldest grid point uses its own rate (no colder neighbor)."""
    betas = np.asarray(betas, np.float64)
    p = np.asarray(leak_rates, np.float64)
    order = np.argsort(betas)
    betas, p = betas[order], p[order]
    need = math.log(1.0 / (1.0 - level))
    scale = nsys * (budget_steps / probe_steps)
    best = None
    for i in range(len(betas)):
        p_hat = p[i] if i == len(betas) - 1 else min(p[i], p[i + 1])
        if scale * p_hat >= need:
            best = float(betas[i])      # ascending scan -> keeps the coldest
    if best is None:
        raise ValueError("no probe beta satisfies the discovery rule")
    return best


def knot_se(betas, sd, se_sd, beta_min, target_nats):
    """Per-knot position standard error by the delta method: perturb each sd_i
    by +/- se_i ONE AT A TIME, recompute the equal-cost knots at the FIXED
    unperturbed rung count (n_rungs jumps are design changes, not position
    noise), take the central half-difference (knots(+) - knots(-))/2 as the
    per-input sensitivity, and sum in quadrature over i. Feeds the
    pre-registered L2 criterion (fresh-probe knots within 3x this se)."""
    sd = np.asarray(sd, np.float64)
    se_sd = np.asarray(se_sd, np.float64)
    n_rungs = design_ladder(betas, sd, beta_min, target_nats)["n_rungs"]
    var = np.zeros(n_rungs)
    for i in range(len(sd)):
        sp, sm = sd.copy(), sd.copy()
        sp[i] += se_sd[i]
        sm[i] = max(sm[i] - se_sd[i], 1e-12)
        kp, _ = _knots_fixed_n(betas, sp, beta_min, n_rungs)
        km, _ = _knots_fixed_n(betas, sm, beta_min, n_rungs)
        var += (0.5 * (kp - km)) ** 2
    return np.sqrt(var)


def classify_nearest_mode(z, mode_centers, cov):
    """Index (n,) of the nearest mode center in whitened Mahalanobis distance:
    whiten with the Cholesky factor L of cov (d = |L^-1 (z - c_k)|_2)."""
    z = np.asarray(z, np.float64)
    centers = np.asarray(mode_centers, np.float64)
    L = np.linalg.cholesky(np.asarray(cov, np.float64))
    d2 = np.empty((z.shape[0], centers.shape[0]))
    for k in range(centers.shape[0]):
        w = np.linalg.solve(L, (z - centers[k]).T)
        d2[:, k] = np.einsum("dn,dn->n", w, w)
    return d2.argmin(axis=1)


def main():
    ap = argparse.ArgumentParser(description="PT ladder design from probe u draws")
    ap.add_argument("npz", help="probe arrays npz with keys u (nbeta,ngroups,steps,nchain), betas")
    ap.add_argument("--discard", type=int, default=1500)
    ap.add_argument("--acc", type=float, default=None,
                    help="desired adjacent acceptance -> target nats via erfcinv")
    ap.add_argument("--target", type=float, default=1.0, help="target nats/pair")
    ap.add_argument("--beta-min", type=float, default=None,
                    help="ladder bottom (default betas[0]; use beta_min_rule for discovery)")
    a = ap.parse_args()
    d = np.load(a.npz)
    betas = np.asarray(d["betas"], np.float64)
    sd, se = measure_sd_u(np.asarray(d["u"]), a.discard)
    for b, s, e in zip(betas, sd, se):
        print(f"  beta={b:8.4f}  sd(u)={s:12.3f}  se(sd)={e:10.3f}")
    target = a.target if a.acc is None else target_from_acceptance(a.acc)
    bmin = betas[0] if a.beta_min is None else a.beta_min
    des = design_ladder(betas, sd, bmin, target)
    print(f"[design] beta_min={bmin:.6g} target={target:.4g} nats/pair -> "
          f"{des['n_rungs']} rungs, total {des['total_cost']:.3f} nats, "
          f"realized {des['nats_per_pair']:.4f} nats/pair")
    print("  " + ",".join(f"{b:.6g}" for b in des["knots"]))


if __name__ == "__main__":
    main()
