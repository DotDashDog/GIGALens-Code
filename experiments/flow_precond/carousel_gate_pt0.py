#!/usr/bin/env python
"""GATE PT-0: tempering-path diagnosis + instrumented PT-MCLMC pilot (pre-reg 2026-07-10).

Design checkpoint: docs/logs/carousel-mclmc-sampling.md "carousel GATE PT-0",
incl. GRADER AMENDMENTS round 1 (items i-iii, viii adopted here).
One arm per process (orchestrator assigns GPUs):

  control  Arm 0 (amendment ii, BLOCKING; runs before any dPIE arm): (0a)
           known-answer weight gate on the June-28 CPU-era Gaussian-mixture
           target (tempering/pt_weight.py: D=10, modes +/-5, weights 0.7/0.3,
           equal cov), PASS = |cold occ_+ - 0.70| <= max(2*se_across_systems,
           0.025); (0b) transport calibration c_rw = observed round trips per
           ladder / (ROUNDS*abar/R^2), needs >=20 total round trips, sanity
           band [0.1, 3]. Routed through the SAME PT harness (run_pt: rung
           kernels + adapt_one + host swaps + walker tracking) as B1-B3 --
           the point is validating the code path that will run on dPIE.
  A_power  confined tempered-mass profile, power path      u(z) = log_prob(z),
           logdensity_beta = beta * log_prob(z)
  A_lik    confined tempered-mass profile, likelihood path u(z) = log_like(z),
           logdensity_beta = log_prior(z) + beta*log_like(z)
                           = (1-beta)*log_prior(z) + beta*log_prob(z)
  B1/B2/B3 instrumented PT-MCLMC pilot: B1 power/balanced init, B2 lik/balanced,
           B3 lik/all-main init (hot rung from prior on B2/B3).
  smoke    all six arms, SHAPE-FAITHFUL (amendment iii, the GATE L attempt-1
           lesson): FULL production compile widths/counts (all Arm-A betas at
           32-wide, both hot-end checks, B at R=12 x NSYS=8 incl. swap sync +
           incremental npz, control at R=10 x NSYS=16); only steps/rounds cut.

Separable target components (gigalens scene API):
  log_prior(z) = ProbModel.log_prior(z)      [scene_prob_model.py:576]
  log_like(z)  = ProbModel.log_like(z)[0]    [scene_prob_model.py:538]
  identity log_prior + log_like == log_prob asserted at startup (8 points, <=1e-6).
Prior z-space draws = prob_model.bij.inverse(prob_model.prior.sample(n, seed=key)),
the exact ModellingSequence.MAP(start=None) code path [inference.py:133,137].

MAMS64 draws are POSITION POOLS + METRICS only (never weights; human directive).

Run (4-GPU node, shifter jax container, float64), one arm per GPU:
  srun --overlap --jobid=$JOB shifter --module=gpu,nccl-plugin \
    --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 bash -c '
      export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages:<repo>/src:<repo>/experiments/flow_precond
      CUDA_VISIBLE_DEVICES=0 /usr/bin/python3 carousel_gate_pt0.py --arm A_power'
Smoke first: GATE_PT0_SMOKE=1 ... --arm smoke   (~10 min, 1 GPU).
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "src")))
import carousel_model
from gigalens_research.inference.blackjax_updated_utils import (
    _build_kernel_shardmap, isokinetic_mclachlan_smart, _single_init, handle_nans)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ shared config
OUT = os.path.join(HERE, "carousel_gate_pt0_out")
DPIE = os.path.join(HERE, "..", "sim_carousel", "messy_tests", "dpie")
MAMS_NPZ = os.path.join(DPIE, "mams", "arrays.npz")
SVI_NPZ = os.path.join(DPIE, "svi", "arrays.npz")   # fallback metric only
POCKET_COL, POCKET_THR = 6, -22.35
DIM = 33
L_MCLMC = math.sqrt(DIM)            # fixed L = sqrt(33), both arms (recorded)
DEVAR, TRUST = 5e-4, 1.5            # EEVPD adapter (carousel_pt.py lineage)
DECAY = (150.0 - 1.0) / (150.0 + 1.0)
SS_INIT, SS_MAX0 = 0.05, 1.0        # init step 0.05; adapter converges from there
JITTER = 1e-3                       # N(0,1) jitter on pool-drawn init positions
SEP_TOL = 1e-6                      # |log_prior + log_like - log_prob| startup gate
U_REL_TOL = 1e-6                    # RELATIVE u-recovery gate (audit fixes 2+3):
                                    # max |u - u_direct| / (1 + |u_direct|)
MIN_CLASS_N = 50                    # Arm A: min retained samples per (beta, class)
                                    # cell; below -> E missing, Delta truncated (fix 4)
ARM_SEED = dict(A_power=0, A_lik=0, B1=0, B2=1, B3=2)   # checkpoint seeds
# Arm A
BETAS_A = np.geomspace(0.01, 1.0, 10)
NCH_A = 16                          # chains per basin group
STEPS_A, DISCARD_A = 3000, 1500
N_HOT_CHECK = 2                     # unconfined consistency runs at 2 hottest betas
N_PRIOR = 4096                      # prior draws for m_prior (A_lik)
DELTA_REFS = (-2.0, -4.2)           # viability floor / unworkable refs (amendment i)
STAT_FLAG_SIGMA = 3.0               # split-half stationarity flag (advisory viii)
# Arm B
R_B, NSYS_B, K_B, ROUNDS_B = 12, 8, 10, 2000
THIN_B, SAVE_EVERY_B, PRINT_EVERY_B = 5, 100, 50
LAST_OCC, RHAT_LAST = 500, 1000
EEVPD_BAND = (1e-4, 2e-3)           # W-4 health band
# Arm 0 control (amendment ii): June-28 pt_weight.py known-answer target
CTRL_D, CTRL_M = 10, 5.0            # D=10 Gaussian mixture, modes at +/-5
CTRL_W = (0.7, 0.3)                 # weights (+mode 0.7 / -mode 0.3), equal cov
CTRL_R, CTRL_NSYS, CTRL_K = 10, 16, 20
CTRL_ROUNDS, CTRL_BURN = 3000, 600
CTRL_SEED = 0
CTRL_OCC_TRUTH = CTRL_W[0]          # cold occ_+ target 0.70
CTRL_TOL_FLOOR = 0.025              # PASS = |occ - 0.70| <= max(2*se, 0.025)
CTRL_MIN_RT = 20                    # >=20 round trips -> c_rw Poisson err <=~25%
CRW_BAND = (0.1, 3.0)               # transport-constant sanity band

SMOKE = False


def apply_smoke():
    """SHAPE-FAITHFUL smoke (amendment iii, BLOCKING; the GATE L attempt-1 XLA
    lesson): FULL production vmap/compile shapes -- all 10 Arm-A beta compiles at
    full 32-wide, both hot-end checks at full width, Arm B at full R=12 x NSYS=8
    exercising the jitted swap-sync round loop AND the incremental-npz save path,
    control at full R=10 x NSYS=16 -- ONLY step/round counts are reduced.
    Numbers are NOT the pre-registered measurement."""
    global SMOKE, STEPS_A, DISCARD_A
    global ROUNDS_B, SAVE_EVERY_B, PRINT_EVERY_B, LAST_OCC, RHAT_LAST
    global CTRL_ROUNDS, CTRL_BURN
    SMOKE = True
    STEPS_A, DISCARD_A = 60, 30
    ROUNDS_B = 20
    SAVE_EVERY_B, PRINT_EVERY_B = 10, 5   # SAVE_EVERY=10 so the npz path fires
    LAST_OCC, RHAT_LAST = 10, 20
    CTRL_ROUNDS, CTRL_BURN = 30, 10


def pr(*a):
    print(*a, flush=True)


def _json_default(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)


def _f3(v):
    """Format possibly-missing (None) estimates for prints (audit fix 4)."""
    return "MISSING" if v is None else f"{v:.3f}"


# --------------------------------------------------------------- model + pools
M = {}  # module namespace filled by setup_model()


def setup_model():
    model_seq, prob_model = carousel_model.build()
    assert prob_model.z_param_names[POCKET_COL] == "planes/0/mass/1/center_x"
    lp_fn = lambda z: prob_model.log_prob(z)[0]
    lpri_fn = lambda z: prob_model.log_prior(z)
    llik_fn = lambda z: prob_model.log_like(z)[0]
    lpri_b = jax.jit(jax.vmap(lpri_fn))
    _lp_vec = jax.jit(jax.vmap(lp_fn))

    def lp_batch(z, chunk=64):
        z = np.asarray(z, dtype=np.float64)
        out = np.empty(len(z))
        for i in range(0, len(z), chunk):
            out[i:i + chunk] = np.asarray(_lp_vec(jnp.asarray(z[i:i + chunk])))
        return out

    mams = np.load(MAMS_NPZ)["samples_z"]
    assert mams.shape == (64, 1000, DIM), mams.shape
    draws = mams.reshape(-1, DIM).astype(np.float64)
    is_pocket = draws[:, POCKET_COL] > POCKET_THR
    pool_M, pool_P = draws[~is_pocket], draws[is_pocket]
    cov_pool = np.cov(draws.T)
    cov_M, cov_P = np.cov(pool_M.T), np.cov(pool_P.T)

    # startup gate: log_prior + log_like == log_prob on 8 test points (single jit,
    # so the shared log_like subgraph is CSE'd -- checks the API split, not FP noise)
    @jax.jit
    def _sep(z):
        return lp_fn(z), lpri_fn(z), llik_fn(z)

    trng = np.random.default_rng(12345)
    tests = np.concatenate([pool_M[trng.integers(0, len(pool_M), 4)],
                            pool_P[trng.integers(0, len(pool_P), 4)]])
    sep_max = 0.0
    for z in tests:
        lp, lpri, ll = (float(v) for v in _sep(jnp.asarray(z)))
        sep_max = max(sep_max, abs(lpri + ll - lp))
    assert sep_max <= SEP_TOL, (
        f"log_prior + log_like != log_prob: max |diff| = {sep_max:.3e} > {SEP_TOL}")
    pr(f"[setup] separability check OK: max |log_prior+log_like-log_prob| "
       f"= {sep_max:.3e} over 8 points")

    M.update(prob_model=prob_model, lp_fn=lp_fn, lpri_fn=lpri_fn, llik_fn=llik_fn,
             lpri_b=lpri_b, lp_batch=lp_batch, draws=draws, is_pocket=is_pocket,
             pool_M=pool_M, pool_P=pool_P, cov_pool=cov_pool, cov_M=cov_M,
             cov_P=cov_P, sep_max=sep_max)


def prior_z_samples(n, seed):
    """Prior draws in z-space: the exact MAP(start=None) path (inference.py:133,137)."""
    x = M["prob_model"].prior.sample(n, seed=jax.random.PRNGKey(seed))
    z = np.asarray(M["prob_model"].bij.inverse(x), dtype=np.float64)
    assert z.shape == (n, DIM), z.shape
    return z


def make_tempered(path_kind, beta):
    """logdensity_beta. power: beta*lp. lik: log_prior + beta*log_like, computed as
    (1-beta)*log_prior + beta*log_prob (identical by the verified split; one sim)."""
    b = float(beta)
    lp_fn, lpri_fn = M["lp_fn"], M["lpri_fn"]
    if path_kind == "power":
        return lambda z: b * lp_fn(z)
    return lambda z: (1.0 - b) * lpri_fn(z) + b * lp_fn(z)


def u_from_state(path_kind, beta, logdensity, positions):
    """Path's u recovered from the kernel's tempered logdensity (beta >= 0.01)."""
    if path_kind == "power":
        return logdensity / beta
    return (logdensity - M["lpri_b"](positions)) / beta


# ----------------------------------------------------- EEVPD adapter + runners
def adapt_one(prev_state, next_state, info, step_size, adapt_state, nan_key,
              dim=DIM):
    """carousel_pt.py adapt_one, math EXACT (parameterized by D=dim, DEVAR=5e-4);
    also returns the handle_nans success flag so reverts can be counted
    (checkpoint NaN policy). dim=33 for dPIE arms, 10 for the Arm-0 control."""
    time_, x_avg, ss_max = adapt_state
    success, state, ss_max, ec = handle_nans(
        prev_state, next_state, step_size, ss_max, info.energy_change, nan_key)
    xi = jnp.square(ec) / (dim * DEVAR) + 1e-8
    weight = jnp.exp(-0.5 * jnp.square(jnp.log(xi) / (6.0 * TRUST)))
    x_avg = DECAY * x_avg + weight * (xi / jnp.power(step_size, 6.0))
    time_ = DECAY * time_ + weight
    new_step = jnp.minimum(jnp.power(x_avg / time_, -1.0 / 6.0), ss_max)
    return state, new_step, (time_, x_avg, ss_max), jnp.square(info.energy_change), success


def fresh_adapt(nch):
    return (jnp.zeros(nch), jnp.zeros(nch), jnp.full(nch, SS_MAX0))


def make_runner_a(tf, path_kind, beta, invmasses, n_steps):
    """One jitted runner per beta (GATE L per-chunk-compile style): kernels for each
    metric group inside a single jit; init (momentum) + full n_steps scan; per-step
    traces of u (fp64), indicator, step size, energy_change^2, nan-success."""
    kerns = [_build_kernel_shardmap(logdensity_fn=tf,
                                    inverse_mass_matrix=jnp.asarray(c),
                                    integrator=isokinetic_mclachlan_smart)
             for c in invmasses]
    n_g = len(kerns)
    b = float(beta)

    @jax.jit
    def run(pos_groups, ss_groups, ast_groups, key):
        kinit, kscan = jax.random.split(key)
        states = []
        for gi in range(n_g):
            pg = jnp.asarray(pos_groups[gi])
            ks = jax.random.split(jax.random.fold_in(kinit, gi), pg.shape[0])
            states.append(jax.vmap(lambda p, k: _single_init(p, tf, k))(pg, ks))
        states = tuple(states)

        def sstep(carry, k):
            sts, sss, asts = carry
            new_s, new_ss, new_a, outs = [], [], [], []
            for gi in range(n_g):
                kern = kerns[gi]

                def per(s, step, a, kk, kern=kern):
                    k1, k2 = jax.random.split(kk)
                    ns, info = kern(rng_key=k1, state=s, L=L_MCLMC, step_size=step)
                    return adapt_one(s, ns, info, step, a, k2)

                keys_g = jax.random.split(jax.random.fold_in(k, gi),
                                          sss[gi].shape[0])
                s2, ss2, a2, ec2, ok = jax.vmap(per)(sts[gi], sss[gi], asts[gi],
                                                     keys_g)
                u = u_from_state(path_kind, b, s2.logdensity, s2.position)
                ind = s2.position[:, POCKET_COL] > POCKET_THR
                new_s.append(s2)
                new_ss.append(ss2)
                new_a.append(a2)
                outs.append((u, ind, ss2, ec2, ok))
            return (tuple(new_s), tuple(new_ss), tuple(new_a)), tuple(outs)

        keys = jax.random.split(kscan, n_steps)
        (states, sss, asts), traces = jax.lax.scan(
            sstep, (states, tuple(ss_groups), tuple(ast_groups)), keys)
        finals = tuple(s.position for s in states)
        return finals, sss, asts, traces

    return run


def make_runner_b(tf, kern, dim, L):
    """Per-rung jitted K-step runner (carousel_pt.py _mk pattern), with the per-round
    momentum refresh (init_level pattern) folded into the jit: states are rebuilt
    from positions each call, so swaps only move positions/u; adapt stays with rung.
    Shared by ALL PT arms (B1/B2/B3 dPIE and the Arm-0 control; amendment ii)."""

    @jax.jit
    def run(positions, ss, ast, init_keys, step_keys):  # step_keys (K, NSYS)
        states = jax.vmap(lambda p, k: _single_init(p, tf, k))(
            jnp.asarray(positions), init_keys)

        def sstep(carry, kt):
            sts, sss, asts = carry

            def per(s, step, a, k):
                k1, k2 = jax.random.split(k)
                ns, info = kern(rng_key=k1, state=s, L=L, step_size=step)
                return adapt_one(s, ns, info, step, a, k2, dim)

            s2, ss2, a2, ec2, ok = jax.vmap(per)(sts, sss, asts, kt)
            return (s2, ss2, a2), (ec2, ok)

        (states, sss, asts), (ec2, ok) = jax.lax.scan(
            sstep, (states, ss, ast), step_keys)
        return states.position, states.logdensity, sss, asts, ec2, ok

    return run


# ------------------------------------------------------------- Arm A analysis
def basin_estimates(u, ind):
    """u, ind: (2 init-groups [M,P], steps, nch). Classify retained samples by
    CURRENT indicator; pool both init groups per class. Chain-clustered se
    (32 chains as clusters). Returns per-class stats + per-init-group leak.
    Empty-class guard (audit fix 4): a cell with < MIN_CLASS_N retained samples
    reports E = None (missing; null in JSON) -- it never enters the trapezoid."""
    ur, ir = u[:, DISCARD_A:, :], ind[:, DISCARD_A:, :]
    out = {}
    for cls, mask in (("M", ~ir), ("P", ir)):
        vals = ur[mask]
        cms = []
        for g in range(ur.shape[0]):
            for c in range(ur.shape[2]):
                mgc = mask[g, :, c]
                if mgc.any():
                    cms.append(ur[g, mgc, c].mean())
        cms = np.asarray(cms)
        if len(vals) < MIN_CLASS_N:
            out[cls] = dict(E=None, se=None, n=int(len(vals)),
                            n_chains=int(len(cms)))
        else:
            se = (float(cms.std(ddof=1) / math.sqrt(len(cms)))
                  if len(cms) > 1 else float("inf"))
            out[cls] = dict(E=float(vals.mean()), se=se, n=int(len(vals)),
                            n_chains=int(len(cms)))
    out["leak_Minit"] = float(ir[0].mean())
    out["leak_Pinit"] = float((~ir[1]).mean())
    return out


def iat_1d(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    v = float((x * x).mean())
    if v == 0:
        return 1.0
    n = len(x)
    f = np.correlate(x, x, "full")[n - 1:] / (v * n)
    tau = 1.0
    for t in range(1, n // 2):
        if f[t] < 0.05:
            break
        tau += 2.0 * float(f[t])
    return tau


def group_ess_proxy(u_g):
    """u_g (steps, nch) retained; ESS proxy = n_meas*nch/(2*IAT), IAT chain-mean."""
    taus = [iat_1d(u_g[:, c]) for c in range(u_g.shape[1])]
    tau = float(np.mean(taus))
    return u_g.shape[0] * u_g.shape[1] / (2.0 * tau), tau


def split_half_report(u_g):
    """Advisory viii: split-half stationarity of E[u] within the measurement
    window. u_g (meas_steps, nch) = one (beta, basin-init) config's retained u.
    E over each half with chain-clustered se; z = (E2-E1)/hypot(se1, se2)."""
    h = u_g.shape[0] // 2
    out = {}
    for name, x in (("half1", u_g[:h]), ("half2", u_g[h:2 * h])):
        cm = x.mean(axis=0)                       # per-chain half means
        out[f"E_{name}"] = float(x.mean())
        out[f"se_{name}"] = (float(cm.std(ddof=1) / math.sqrt(x.shape[1]))
                             if x.shape[1] > 1 else float("inf"))
    se_d = math.hypot(out["se_half1"], out["se_half2"])
    out["z_diff"] = (float((out["E_half2"] - out["E_half1"]) / se_d)
                     if se_d > 0 else float("inf"))
    out["flag_3se"] = bool(abs(out["z_diff"]) > STAT_FLAG_SIGMA)
    return out


def cum_trapz_from_cold(betas, d, var_d):
    """Delta(beta_k) = trapz of d over [beta_k, 1] (cold-end anchored); se propagated
    through the trapezoid weights, points treated as independent."""
    n = len(betas)
    delta, se = np.zeros(n), np.zeros(n)
    for k in range(n - 1):
        w = np.zeros(n)
        for j in range(k, n - 1):
            h = betas[j + 1] - betas[j]
            w[j] += 0.5 * h
            w[j + 1] += 0.5 * h
        delta[k] = float(np.sum(w * d))
        se[k] = float(math.sqrt(np.sum(w ** 2 * var_d)))
    return delta, se


def run_arm_a(path_kind, tag):
    seed = ARM_SEED["A_power" if path_kind == "power" else "A_lik"]
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)
    betas = BETAS_A
    card = dict(arm=tag, path=path_kind, seed=seed, smoke=SMOKE,
                script=os.path.abspath(__file__), jax=jax.__version__,
                devices=[str(d) for d in jax.devices()],
                x64=bool(jax.config.jax_enable_x64), dim=DIM,
                betas=betas.tolist(), n_chains_per_basin=NCH_A,
                steps=STEPS_A, discard=DISCARD_A, L=L_MCLMC,
                ss_init=SS_INIT, devar=DEVAR, jitter=JITTER,
                metric="per-basin empirical cov (confined) / pooled cov (hot-end "
                       "unconfined), MAMS64 positions only, never weights",
                mams64=MAMS_NPZ, n_pool_M=int(len(M["pool_M"])),
                n_pool_P=int(len(M["pool_P"])),
                pocket_indicator=f"z[{POCKET_COL}] > {POCKET_THR}",
                sep_check_max_abs=M["sep_max"], n_hot_check=N_HOT_CHECK,
                u_def="log_prob" if path_kind == "power" else "log_like")
    pr("MODEL CARD:", json.dumps(card, indent=1, default=_json_default))
    summary = dict(model_card=card)
    t0_all = time.time()

    def draw_init(pool, n):
        idx = rng.integers(0, len(pool), size=n)
        return pool[idx] + JITTER * rng.standard_normal((n, DIM))

    n_b = len(betas)
    u_tr = np.zeros((n_b, 2, STEPS_A, NCH_A))
    ind_tr = np.zeros((n_b, 2, STEPS_A, NCH_A), dtype=bool)
    ss_tr = np.zeros((n_b, 2, STEPS_A, NCH_A))
    ev_tr = np.zeros((n_b, 2, STEPS_A))
    n_revert = np.zeros((n_b, 2), dtype=int)
    table = []
    for bi, b in enumerate(betas):
        t0 = time.time()
        tf = make_tempered(path_kind, b)
        runner = make_runner_a(tf, path_kind, b, [M["cov_M"], M["cov_P"]], STEPS_A)
        pos = (draw_init(M["pool_M"], NCH_A), draw_init(M["pool_P"], NCH_A))
        ss = (jnp.full(NCH_A, SS_INIT), jnp.full(NCH_A, SS_INIT))
        ast = (fresh_adapt(NCH_A), fresh_adapt(NCH_A))
        key, kr = jax.random.split(key)
        finals, _, _, traces = runner(pos, ss, ast, kr)
        for g in range(2):
            u, ind, sst, ec2, ok = traces[g]
            u_tr[bi, g] = np.asarray(u)          # (steps, nch), fp64
            ind_tr[bi, g] = np.asarray(ind)
            ss_tr[bi, g] = np.asarray(sst)
            ev_tr[bi, g] = np.asarray(ec2).mean(axis=1) / DIM
            n_revert[bi, g] = int((~np.asarray(ok)).sum())
        # u-recovery identity check per (beta, basin) config (audit fix 3): the
        # runner is one jitted call, so the check runs right after it, comparing
        # the recorded in-scan u at the FINAL step against a direct eval on the
        # final positions (the only step whose positions are returned). Same
        # RELATIVE 1e-6 gate as the Arm-B round-0 check (audit fix 2 rationale).
        u_check_rel = []
        for g in range(2):
            zfin = np.asarray(finals[g])
            if path_kind == "power":
                u_dir = M["lp_batch"](zfin)
            else:
                u_dir = (M["lp_batch"](zfin)
                         - np.asarray(M["lpri_b"](jnp.asarray(zfin))))
            rel = float(np.max(np.abs(u_tr[bi, g, -1] - u_dir)
                               / (1.0 + np.abs(u_dir))))
            u_check_rel.append(rel)
            if rel > U_REL_TOL:
                raise RuntimeError(
                    f"Arm A u-recovery check failed at beta={b:.4f} group "
                    f"{'MP'[g]}: rel {rel:.3e} > {U_REL_TOL}")
        est = basin_estimates(u_tr[bi], ind_tr[bi])
        essM, tauM = group_ess_proxy(u_tr[bi, 0, DISCARD_A:])
        essP, tauP = group_ess_proxy(u_tr[bi, 1, DISCARD_A:])
        stat_M = split_half_report(u_tr[bi, 0, DISCARD_A:])
        stat_P = split_half_report(u_tr[bi, 1, DISCARD_A:])
        row = dict(beta=float(b), E_M=est["M"]["E"], se_M=est["M"]["se"],
                   n_M=est["M"]["n"], n_chains_M=est["M"]["n_chains"],
                   E_P=est["P"]["E"], se_P=est["P"]["se"], n_P=est["P"]["n"],
                   n_chains_P=est["P"]["n_chains"],
                   leak_Minit=est["leak_Minit"], leak_Pinit=est["leak_Pinit"],
                   leak_flag=bool(max(est["leak_Minit"], est["leak_Pinit"]) > 0.10),
                   ess_proxy_Minit=essM, iat_Minit=tauM,
                   ess_proxy_Pinit=essP, iat_Pinit=tauP,
                   stationarity_Minit=stat_M, stationarity_Pinit=stat_P,
                   u_check_rel=u_check_rel,
                   eevpd_last=[float(ev_tr[bi, g, -min(500, STEPS_A // 2):].mean())
                               for g in range(2)],
                   step_final=[float(ss_tr[bi, g, -1].mean()) for g in range(2)],
                   n_revert=[int(n_revert[bi, g]) for g in range(2)],
                   wall_s=time.time() - t0)
        table.append(row)
        pr(f"[A {tag}] beta={b:.4f}: E_M={_f3(row['E_M'])}+-{_f3(row['se_M'])} "
           f"(n={row['n_M']}) E_P={_f3(row['E_P'])}+-{_f3(row['se_P'])} "
           f"(n={row['n_P']}) "
           f"leak M/P={row['leak_Minit']:.3f}/{row['leak_Pinit']:.3f}"
           f"{' LEAK>10%' if row['leak_flag'] else ''} "
           f"u_check_rel={u_check_rel[0]:.1e}/{u_check_rel[1]:.1e} "
           f"eevpd={row['eevpd_last'][0]:.1e}/{row['eevpd_last'][1]:.1e} "
           f"reverts={row['n_revert']} ({row['wall_s']:.0f}s)")
        if row["E_M"] is None or row["E_P"] is None:
            pr(f"[A {tag}]   EMPTY-CLASS FLAG beta={b:.4f}: "
               f"n_M={row['n_M']}, n_P={row['n_P']} (< {MIN_CLASS_N} in a cell) "
               f"-> E missing; Delta will be truncated here")
        for gnm, st in (("Minit", stat_M), ("Pinit", stat_P)):
            if st["flag_3se"]:
                pr(f"[A {tag}]   STATIONARITY FLAG {gnm} beta={b:.4f}: "
                   f"E half1/2 = {st['E_half1']:.3f}/{st['E_half2']:.3f}, "
                   f"|z| = {abs(st['z_diff']):.2f} > {STAT_FLAG_SIGMA}")

    # Truncation rule (audit fix 4; reporting, not interpolation): Delta(beta) is
    # computed only over the maximal COMPLETE suffix ending at the cold anchor --
    # down to the hottest beta with BOTH classes complete; hotter betas get null.
    complete = [(r["E_M"] is not None) and (r["E_P"] is not None) for r in table]
    k_min = n_b
    for k in range(n_b - 1, -1, -1):
        if complete[k]:
            k_min = k
        else:
            break
    d = np.full(n_b, np.nan)
    var_d = np.full(n_b, np.nan)
    for i, r in enumerate(table):
        if complete[i]:
            d[i] = r["E_M"] - r["E_P"]
            var_d[i] = r["se_M"] ** 2 + r["se_P"] ** 2
    delta = np.full(n_b, np.nan)
    delta_se = np.full(n_b, np.nan)
    if k_min < n_b:
        dl, ds = cum_trapz_from_cold(betas[k_min:], d[k_min:], var_d[k_min:])
        delta[k_min:], delta_se[k_min:] = dl, ds
        delta_truncated_at_beta = float(betas[k_min]) if k_min > 0 else None
    else:
        delta_truncated_at_beta = "no_complete_config"
    if k_min > 0:
        pr(f"[A {tag}] DELTA TRUNCATED: complete suffix starts at "
           f"beta={betas[k_min]:.4f} (k_min={k_min})" if k_min < n_b
           else f"[A {tag}] DELTA TRUNCATED: NO complete (beta, class) config")
    pr(f"[A {tag}] Delta(beta): " +
       " ".join(f"{b:.3f}:{v:+.2f}+-{s:.2f}" if np.isfinite(v)
                else f"{b:.3f}:null"
                for b, v, s in zip(betas, delta, delta_se)))

    # hot-end consistency: unconfined runs at the N_HOT_CHECK hottest betas,
    # 50/50 main/pocket init, pooled-cov metric -> direct pocket occupancy
    hot = {}
    for bi in range(N_HOT_CHECK):
        b = betas[bi]
        t0 = time.time()
        tf = make_tempered(path_kind, b)
        runner = make_runner_a(tf, path_kind, b, [M["cov_pool"]], STEPS_A)
        pos0 = np.concatenate([draw_init(M["pool_M"], NCH_A),
                               draw_init(M["pool_P"], NCH_A)])
        key, kr = jax.random.split(key)
        _, _, _, traces = runner((pos0,), (jnp.full(2 * NCH_A, SS_INIT),),
                                 (fresh_adapt(2 * NCH_A),), kr)
        ind_h = np.asarray(traces[0][1])[DISCARD_A:]      # (meas, 32)
        occ_chain = ind_h.mean(axis=0)
        occ = float(ind_h.mean())
        se = (float(occ_chain.std(ddof=1) / math.sqrt(len(occ_chain)))
              if len(occ_chain) > 1 else float("inf"))
        # convenience prediction from Delta + ASSUMED 0.1 cold pocket anchor
        if np.isfinite(delta[bi]):
            lo = math.log(0.1 / 0.9) + float(delta[bi])
            occ_pred = 1.0 / (1.0 + math.exp(-lo))
            # consistency flag (audit fix 5): direct occupancy vs prediction
            consistent = (bool(abs(occ - occ_pred) <= 2.0 * se)
                          if math.isfinite(se) else None)
        else:
            occ_pred, consistent = None, None   # Delta truncated at this beta
        hot[f"beta_{b:.4f}"] = dict(
            beta=float(b), occ_pocket=occ, se_chain_clustered=se,
            occ_pred_assuming_cold_anchor_0p1=occ_pred,
            consistency_within_2se=consistent,
            wall_s=time.time() - t0)
        hot[f"beta_{b:.4f}"]["ind_frac_by_chain"] = occ_chain.tolist()
        pr(f"[A {tag}] hot-end unconfined beta={b:.4f}: pocket occ={occ:.4f}"
           f"+-{se:.4f} (pred w/ 0.1 anchor: {_f3(occ_pred)}; "
           f"consistent within 2se: {consistent}) "
           f"({hot[f'beta_{b:.4f}']['wall_s']:.0f}s)")

    summary["per_beta"] = table
    _null = lambda arr: [float(v) if np.isfinite(v) else None for v in arr]
    summary["delta"] = dict(betas=betas.tolist(), d_integrand=_null(d),
                            delta=_null(delta), delta_se=_null(delta_se),
                            delta_truncated_at_beta=delta_truncated_at_beta,
                            min_class_n=MIN_CLASS_N)
    summary["hot_end_consistency"] = hot
    summary["leak_flags"] = [r["beta"] for r in table if r["leak_flag"]]

    if path_kind == "lik":  # m_prior: prior indicator split (cheap, A_lik only)
        zpri = prior_z_samples(N_PRIOR, seed)
        mp = float((zpri[:, POCKET_COL] > POCKET_THR).mean())
        se_mp = math.sqrt(max(mp * (1 - mp), 1e-12) / N_PRIOR)
        summary["m_prior"] = dict(n=N_PRIOR, m_prior=mp, se_binomial=se_mp)
        pr(f"[A {tag}] m_prior = {mp:.4f} +- {se_mp:.4f} (n={N_PRIOR})")
    else:
        zpri = None

    np.savez(os.path.join(OUT, f"arrays_{tag}.npz"),
             betas=betas, u=u_tr, ind=ind_tr.astype(np.uint8), step=ss_tr,
             eevpd=ev_tr, n_revert=n_revert, d_integrand=d, delta=delta,
             delta_se=delta_se,
             **({"prior_z6": zpri[:, POCKET_COL]} if zpri is not None else {}))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].errorbar(betas[:-1], delta[:-1], yerr=delta_se[:-1], fmt="o-",
                   capsize=3, label=f"Delta(beta), {path_kind} path")
    ax[0].fill_between(betas[:-1], delta[:-1] - delta_se[:-1],
                       delta[:-1] + delta_se[:-1], alpha=0.2)
    for ref in DELTA_REFS:
        ax[0].axhline(ref, ls="--", lw=0.8, color="k")
        ax[0].text(betas[-2], ref, f"{ref} nats", fontsize=7, va="bottom")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("beta")
    ax[0].set_ylabel("Delta(beta) = log[wP/wM](beta) - log[wP/wM](1)  [nats]")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"{tag}: tempered-mass relative profile")
    EM = np.array([np.nan if r["E_M"] is None else r["E_M"] for r in table],
                  dtype=float)
    EP = np.array([np.nan if r["E_P"] is None else r["E_P"] for r in table],
                  dtype=float)
    ax[1].plot(betas, EM, "o-", label="E_M[u]")
    ax[1].plot(betas, EP, "s-", label="E_P[u]")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("beta")
    ax[1].set_ylabel("E[u]")
    ax2 = ax[1].twinx()
    ax2.plot(betas, d, "^:", color="C3", label="E_M - E_P")
    ax2.set_ylabel("E_M - E_P [nats]", color="C3")
    ax[1].legend(fontsize=8, loc="upper left")
    ax[1].set_title("per-basin confined expectations")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"pt0_{tag}_profile.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for bi in range(len(betas)):
        ax[0].plot(ss_tr[bi, 0].mean(axis=1), lw=0.7,
                   label=f"b={betas[bi]:.3f}" if bi % 3 == 0 else None)
        w = max(1, STEPS_A // 300)
        ev_sm = ev_tr[bi, 0][:STEPS_A // w * w].reshape(-1, w).mean(axis=1)
        ax[1].plot(np.arange(len(ev_sm)) * w, ev_sm, lw=0.7)
    ax[0].set_yscale("log")
    ax[0].set_title("mean step size (M-init group) per beta")
    ax[0].set_xlabel("step")
    ax[0].legend(fontsize=6)
    ax[1].set_yscale("log")
    ax[1].axhline(DEVAR, ls="--", color="k", lw=0.8)
    ax[1].set_title("EEVPD (M-init group) per beta; target 5e-4")
    ax[1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"pt0_{tag}_adapt.png"), dpi=120)
    plt.close(fig)

    summary["total_wall_s"] = time.time() - t0_all
    with open(os.path.join(OUT, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1, default=_json_default)
    pr(f"[A {tag}] DONE in {summary['total_wall_s']:.0f}s -> {OUT}")


# ------------------------------------------------------------------ Arm B pilot
def split_rhat_binary(x):
    """Plain split-Rhat: x (n_systems, n_rounds) binary; halves -> 2*n chains."""
    n = x.shape[1] // 2
    if n < 2:
        return float("nan")
    halves = np.concatenate([x[:, :n], x[:, n:2 * n]], axis=0).astype(np.float64)
    nn = halves.shape[1]
    cm, cv = halves.mean(axis=1), halves.var(axis=1, ddof=1)
    W, B = float(cv.mean()), float(nn * cm.var(ddof=1))
    if W == 0.0:
        return float("nan") if B == 0.0 else float("inf")
    return float(math.sqrt(((nn - 1) / nn * W + B / nn) / W))


def run_pt(tag, seed, spec):
    """Shared PT-MCLMC harness — the SINGLE code path used by ALL PT arms:
    B1/B2/B3 (dPIE) AND the Arm-0 control (amendment ii). Per-rung
    _build_kernel_shardmap kernels + make_runner_b (adapt_one EEVPD, momentum
    refresh) + host even/odd swaps + walker round-trip tracking + recording.
    spec: dim, L, betas, NSYS, K, ROUNDS, inv_mass, make_tf(beta)->logdensity_fn,
    u_from(logd, pos)->u (R,NSYS), u_direct(pos_flat)->u (round-0 verify),
    indicator(pos)->bool, init_pos (R,NSYS,dim), card, ind_label, occ_truth (opt).
    Returns (summary, data-dict); writes summary/npz/plots for the tag."""
    dim, L = spec["dim"], spec["L"]
    betas = np.asarray(spec["betas"], dtype=np.float64)  # rung 0 = hottest
    R = len(betas)
    NSYS, K, ROUNDS = spec["NSYS"], spec["K"], spec["ROUNDS"]
    indicator = spec["indicator"]
    ind_label = spec.get("ind_label", "pocket")
    swap_rng = np.random.default_rng(seed + 10_000)
    key = jax.random.key(seed)
    card = spec["card"]
    pr("MODEL CARD:", json.dumps(card, indent=1, default=_json_default))
    summary = dict(model_card=card)
    t0_all = time.time()

    # kernels + per-rung jitted runners (one metric for all rungs)
    inv_mass = jnp.asarray(spec["inv_mass"])
    runners = []
    for r in range(R):
        tf = spec["make_tf"](betas[r])
        kern = _build_kernel_shardmap(logdensity_fn=tf, inverse_mass_matrix=inv_mass,
                                      integrator=isokinetic_mclachlan_smart)
        runners.append(make_runner_b(tf, kern, dim, L))

    pos = np.array(spec["init_pos"], dtype=np.float64)
    assert pos.shape == (R, NSYS, dim), pos.shape
    init_cold_occ = float(indicator(pos[R - 1]).mean())
    pr(f"[B {tag}] init cold occ = {init_cold_occ:.3f}; "
       f"init occ per rung = "
       f"{np.round(indicator(pos).mean(axis=1), 2).tolist()}")

    steps = [jnp.full(NSYS, SS_INIT) for _ in range(R)]
    adapt = [fresh_adapt(NSYS) for _ in range(R)]
    wid = np.tile(np.arange(R)[:, None], (1, NSYS))     # walker ids
    # wflag [walker_id, sys] (audit fix 1): 0 neutral, 1 tagged-hot,
    # 2 down-traversed arriving MAIN-classified, 3 down-traversed arriving
    # POCKET-classified -- the arrival class rides in the flag so round trips
    # split by class on return to rung 0 (W-2 scores the pocket-classified count).
    wflag = np.zeros((R, NSYS), dtype=np.int8)
    down_traverses = np.zeros(NSYS, dtype=int)
    round_trips = np.zeros(NSYS, dtype=int)
    round_trips_main = np.zeros(NSYS, dtype=int)
    round_trips_pocket = np.zeros(NSYS, dtype=int)      # the W-2 statistic
    pocket_cold_arr = np.zeros(NSYS, dtype=int)
    main_cold_arr = np.zeros(NSYS, dtype=int)

    n_thin = (ROUNDS + THIN_B - 1) // THIN_B
    ind_thin = np.zeros((n_thin, R, NSYS), dtype=np.uint8)
    cold_ind = np.zeros((ROUNDS, NSYS), dtype=np.uint8)
    ev = np.zeros((ROUNDS, R))
    ssm = np.zeros((ROUNDS, R))
    att = np.zeros((R - 1, 2), dtype=int)   # [pair, 0=same 1=cross] attempts
    acc = np.zeros((R - 1, 2), dtype=int)
    n_revert = 0
    u0_verify, u0_rel = None, None
    npz_path = os.path.join(OUT, f"arrays_{tag}.npz")

    def save_npz(t):
        np.savez(npz_path, betas=betas, ind_thin=ind_thin[:t // THIN_B + 1],
                 cold_ind=cold_ind[:t + 1], eevpd=ev[:t + 1], step_mean=ssm[:t + 1],
                 swap_attempts=att, swap_accepts=acc,
                 walker_id=wid, walker_flag=wflag,   # machine reconstructible (fix 1)
                 down_traverses=down_traverses, round_trips=round_trips,
                 round_trips_main=round_trips_main,
                 round_trips_pocket=round_trips_pocket,
                 pocket_cold_arrivals=pocket_cold_arr,
                 main_cold_arrivals=main_cold_arr, n_revert=np.int64(n_revert),
                 init_cold_occ=init_cold_occ, rounds_done=np.int64(t + 1))

    t0 = time.time()
    logd = np.zeros((R, NSYS))
    for t in range(ROUNDS):
        key, *lks = jax.random.split(key, R + 1)
        for r in range(R):
            ik = jax.random.split(lks[r], NSYS)
            sk = jax.random.split(jax.random.fold_in(lks[r], 1),
                                  K * NSYS).reshape(K, NSYS)
            p2, ld, steps[r], adapt[r], ec2, ok = runners[r](
                pos[r], steps[r], adapt[r], ik, sk)
            pos[r] = np.asarray(p2)
            logd[r] = np.asarray(ld)
            ev[t, r] = float(np.mean(np.asarray(ec2)) / dim)
            ssm[t, r] = float(np.mean(np.asarray(steps[r])))
            n_revert += int((~np.asarray(ok)).sum())
        # path's u recovered from the tempered logdensity (spec-provided; the lik
        # path does its log_prior batch eval inside u_from)
        u = spec["u_from"](logd, pos)
        if t == 0:   # round-0 identity verification against direct evals
            u_direct = np.asarray(
                spec["u_direct"](pos.reshape(-1, dim))).reshape(R, NSYS)
            # RELATIVE tolerance (audit fix 2): absolute 1e-6 on |u|~3e5 is a
            # ~1e-11 relative reproducibility demand across jit compiles that
            # non-bitwise lstsq log_like can spuriously trip (and log_prior
            # cross-compile mismatch is amplified by 1/beta on the lik path);
            # 1e-6 RELATIVE (~0.3 nats here) still catches real algebra bugs.
            u0_verify = float(np.max(np.abs(u - u_direct)))
            u0_rel = float(np.max(np.abs(u - u_direct) / (1.0 + np.abs(u_direct))))
            pr(f"[B {tag}] round-0 u identity: rel = {u0_rel:.3e} "
               f"(abs = {u0_verify:.3e})")
            if u0_rel > U_REL_TOL:
                raise RuntimeError(
                    f"round-0 u verification failed: rel {u0_rel:.3e} "
                    f"> {U_REL_TOL}")
        ind_pre = indicator(pos)
        # even/odd adjacent swaps (host numpy, vectorized over NSYS)
        parity = t % 2
        for r in range(parity, R - 1, 2):
            same = ind_pre[r] == ind_pre[r + 1]
            la = (betas[r] - betas[r + 1]) * (u[r + 1] - u[r])
            a = np.log(swap_rng.random(NSYS)) < la
            att[r, 0] += int(same.sum())
            att[r, 1] += int((~same).sum())
            acc[r, 0] += int((a & same).sum())
            acc[r, 1] += int((a & ~same).sum())
            pr_, pr1 = pos[r].copy(), pos[r + 1].copy()
            pos[r] = np.where(a[:, None], pr1, pr_)
            pos[r + 1] = np.where(a[:, None], pr_, pr1)
            u_, u1 = u[r].copy(), u[r + 1].copy()
            u[r] = np.where(a, u1, u_)
            u[r + 1] = np.where(a, u_, u1)
            w_, w1 = wid[r].copy(), wid[r + 1].copy()
            wid[r] = np.where(a, w1, w_)
            wid[r + 1] = np.where(a, w_, w1)
        ind_post = indicator(pos)
        # walker round-trip state machine (audit fix 1): neutral(0) -> tagged-hot(1)
        # at rung 0; tagged-hot reaching rung R-1 = DOWN-traverse, arrival class
        # carried in the flag (2 = arrived MAIN-classified, 3 = arrived
        # POCKET-classified); back at rung 0 = round trip, split by carried class.
        for s in range(NSYS):
            wh = wid[0, s]
            if wflag[wh, s] in (2, 3):
                round_trips[s] += 1
                if wflag[wh, s] == 3:
                    round_trips_pocket[s] += 1   # W-2 statistic
                else:
                    round_trips_main[s] += 1
                wflag[wh, s] = 1
            elif wflag[wh, s] == 0:
                wflag[wh, s] = 1
            wc = wid[R - 1, s]
            if wflag[wc, s] == 1:
                down_traverses[s] += 1
                if ind_post[R - 1, s]:
                    pocket_cold_arr[s] += 1
                    wflag[wc, s] = 3             # arrived POCKET-classified
                else:
                    main_cold_arr[s] += 1
                    wflag[wc, s] = 2             # arrived MAIN-classified
        cold_ind[t] = ind_post[R - 1]
        if t % THIN_B == 0:
            ind_thin[t // THIN_B] = ind_post
        if t % PRINT_EVERY_B == 0 or t == ROUNDS - 1:
            pr(f"[B {tag}] round {t:5d}/{ROUNDS} cold occ={cold_ind[t].mean():.3f} "
               f"hot occ={ind_post[0].mean():.3f} RT={int(round_trips.sum())} "
               f"pocketCA={int(pocket_cold_arr.sum())} "
               f"EEVPD[h/m/c]={ev[t, 0]:.1e}/{ev[t, R // 2]:.1e}/{ev[t, -1]:.1e} "
               f"reverts={n_revert} wall={time.time() - t0:.0f}s")
        if t % SAVE_EVERY_B == 0 or t == ROUNDS - 1:
            save_npz(t)

    # ------------------------------------------------------------- end-of-arm stats
    last = min(LAST_OCC, ROUNDS)
    occ_sys = cold_ind[-last:].mean(axis=0)
    occ_mean, occ_sd = float(occ_sys.mean()), float(occ_sys.std(ddof=1))
    rl = min(RHAT_LAST, ROUNDS)
    rhat = split_rhat_binary(cold_ind[-rl:].T)
    ev_last = ev[-last:].mean(axis=0)
    ev_in_band = [(EEVPD_BAND[0] <= v <= EEVPD_BAND[1]) for v in ev_last]
    with np.errstate(invalid="ignore"):
        acc_frac = np.where(att > 0, acc / np.maximum(att, 1), np.nan)
    summary["end_stats"] = dict(
        indicator_label=ind_label,
        last_rounds=last,
        cold_occ_per_system=occ_sys.tolist(),
        cold_occ_mean=occ_mean, cold_occ_sd=occ_sd,
        init_cold_occ=init_cold_occ,
        round_trips_per_system=round_trips.tolist(),
        round_trips_total=int(round_trips.sum()),
        round_trips_median_per_system=float(np.median(round_trips)),
        round_trips_main_per_system=round_trips_main.tolist(),
        round_trips_main_total=int(round_trips_main.sum()),
        round_trips_pocket_per_system=round_trips_pocket.tolist(),
        round_trips_pocket_total=int(round_trips_pocket.sum()),  # W-2 scores this
        down_traverses_per_system=down_traverses.tolist(),
        pocket_cold_arrivals_per_system=pocket_cold_arr.tolist(),
        pocket_cold_arrivals_total=int(pocket_cold_arr.sum()),
        main_cold_arrivals_total=int(main_cold_arr.sum()),
        swap_attempts_same=att[:, 0].tolist(), swap_attempts_cross=att[:, 1].tolist(),
        swap_acc_same=acc_frac[:, 0].tolist(), swap_acc_cross=acc_frac[:, 1].tolist(),
        cold_indicator_split_rhat=rhat, rhat_last_rounds=rl,
        eevpd_last_per_rung=ev_last.tolist(), eevpd_in_band=ev_in_band,
        eevpd_band=list(EEVPD_BAND),
        step_mean_last_per_rung=ssm[-last:].mean(axis=0).tolist(),
        n_nan_reverts=int(n_revert),
        u0_verify_rel=u0_rel,
        u0_verify_max_abs=u0_verify,
    )
    pr(f"\n[B {tag}] ===== END-OF-ARM (PROPOSED/UNCERTIFIED) =====")
    pr(f"  cold occ last {last} rounds: {occ_mean:.4f} +- {occ_sd:.4f} "
       f"(per-sys {np.round(occ_sys, 3).tolist()}; init {init_cold_occ:.3f})")
    pr(f"  round trips total {int(round_trips.sum())} "
       f"(median/sys {float(np.median(round_trips)):.1f}; "
       f"pocket-classified {int(round_trips_pocket.sum())} [W-2], "
       f"main-classified {int(round_trips_main.sum())}); "
       f"pocket cold-arrivals {int(pocket_cold_arr.sum())}, "
       f"main {int(main_cold_arr.sum())}")
    pr(f"  swap acc same={np.round(acc_frac[:, 0], 3).tolist()}")
    pr(f"  swap acc cross={np.round(acc_frac[:, 1], 3).tolist()} "
       f"(cross attempts {att[:, 1].tolist()})")
    pr(f"  cold split-Rhat = {rhat}; EEVPD in [1e-4,2e-3]: {ev_in_band}; "
       f"NaN reverts {n_revert}")
    save_npz(ROUNDS - 1)

    # ---------------------------------------------------------------------- plots
    xs = np.arange(0, ROUNDS, THIN_B)[:ind_thin.shape[0]]
    n_fig = (NSYS + 3) // 4                     # <=4 panels per figure
    for fi in range(n_fig):
        systems = list(range(fi * 4, min(fi * 4 + 4, NSYS)))
        fig, axes = plt.subplots(len(systems), 1,
                                 figsize=(10, 2.2 * len(systems)), squeeze=False)
        for i, s in enumerate(systems):
            axm = axes[i, 0]
            axm.imshow(ind_thin[:, :, s].T, aspect="auto", origin="lower",
                       interpolation="nearest", cmap="coolwarm",
                       extent=[0, xs[-1] if len(xs) else 1, -0.5, R - 0.5])
            axm.set_ylabel(f"sys {s}\nrung (0=hot)")
        axes[-1, 0].set_xlabel("round")
        fig.suptitle(f"{tag}: basin identity (red={ind_label}) rung x round")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"pt0_{tag}_worms{fi + 1}.png"), dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    w = max(1, ROUNDS // 200)
    for s in range(NSYS):
        sm = cold_ind[:ROUNDS // w * w, s].reshape(-1, w).mean(axis=1)
        ax.plot(np.arange(len(sm)) * w, sm, lw=0.6, alpha=0.6)
    smm = cold_ind[:ROUNDS // w * w].mean(axis=1).reshape(-1, w).mean(axis=1)
    ax.plot(np.arange(len(smm)) * w, smm, "k-", lw=2, label="mean over systems")
    ax.axhline(init_cold_occ, ls="--", color="r", lw=1,
               label=f"init occ {init_cold_occ:.2f}")
    if spec.get("occ_truth") is not None:
        ax.axhline(spec["occ_truth"], ls="--", color="g", lw=1,
                   label=f"truth {spec['occ_truth']:.2f}")
    ax.set_xlabel("round")
    ax.set_ylabel(f"cold-rung {ind_label} occupancy")
    ax.set_title(f"{tag}: cold occupancy (rolling mean, w={w})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"pt0_{tag}_coldocc.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    thin = max(1, ROUNDS // 400)
    for r in range(R):
        ax.plot(np.arange(0, ROUNDS, thin), ev[::thin, r], lw=0.7,
                label=f"rung {r} (b={betas[r]:.3f})" if r % 3 == 0 else None)
    ax.axhline(DEVAR, ls="--", color="k", lw=0.8, label="target 5e-4")
    for b in EEVPD_BAND:
        ax.axhline(b, ls=":", color="gray", lw=0.6)
    ax.set_yscale("log")
    ax.set_xlabel("round")
    ax.set_ylabel("EEVPD")
    ax.set_title(f"{tag}: per-rung EEVPD")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"pt0_{tag}_eevpd.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(R - 1)
    ax.bar(x - 0.2, np.nan_to_num(acc_frac[:, 0]), 0.4, label="same-basin")
    ax.bar(x + 0.2, np.nan_to_num(acc_frac[:, 1]), 0.4, label="cross-basin")
    for i in range(R - 1):
        ax.text(i + 0.2, 0.02, str(att[i, 1]), ha="center", fontsize=6,
                rotation=90)
    ax.set_xlabel("adjacent pair (hot side index)")
    ax.set_ylabel("swap acceptance")
    ax.set_title(f"{tag}: swap acceptance per pair, same vs cross "
                 f"(cross attempt counts annotated)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"pt0_{tag}_swaps.png"), dpi=120)
    plt.close(fig)

    summary["total_wall_s"] = time.time() - t0_all
    with open(os.path.join(OUT, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1, default=_json_default)
    pr(f"[B {tag}] DONE in {summary['total_wall_s']:.0f}s -> {OUT}")
    data = dict(cold_ind=cold_ind, ind_thin=ind_thin, ev=ev, ssm=ssm,
                att=att, acc=acc, round_trips=round_trips,
                round_trips_main=round_trips_main,
                round_trips_pocket=round_trips_pocket,
                down_traverses=down_traverses,
                pocket_cold_arr=pocket_cold_arr, main_cold_arr=main_cold_arr,
                init_cold_occ=init_cold_occ, n_revert=n_revert)
    return summary, data


def run_arm_b(arm, tag):
    """dPIE PT arms B1/B2/B3: build the target spec and route through run_pt."""
    seed = ARM_SEED[arm]
    path_kind = "power" if arm == "B1" else "lik"
    rng = np.random.default_rng(seed)
    betas = np.geomspace(0.01, 1.0, R_B)   # rung 0 = hottest, R-1 = cold

    def draw_pool(pool, n):
        idx = rng.integers(0, len(pool), size=n)
        return pool[idx] + JITTER * rng.standard_normal((n, DIM))

    pos = np.zeros((R_B, NSYS_B, DIM))
    for r in range(R_B):
        if arm == "B3":
            pos[r] = draw_pool(M["pool_M"], NSYS_B)
        else:
            pick_p = rng.random(NSYS_B) < 0.5
            zm, zp = draw_pool(M["pool_M"], NSYS_B), draw_pool(M["pool_P"], NSYS_B)
            pos[r] = np.where(pick_p[:, None], zp, zm)
    if arm in ("B2", "B3"):   # hot rung from PRIOR on the likelihood arms
        pos[0] = prior_z_samples(NSYS_B, seed + 777)

    if path_kind == "power":
        u_from = lambda logd, p: logd / betas[:, None]
        u_direct = lambda pf: M["lp_batch"](pf)
    else:
        def u_from(logd, p):
            lpri = np.asarray(M["lpri_b"](jnp.asarray(p.reshape(-1, DIM)))
                              ).reshape(logd.shape)
            return (logd - lpri) / betas[:, None]

        def u_direct(pf):
            return M["lp_batch"](pf) - np.asarray(M["lpri_b"](jnp.asarray(pf)))

    card = dict(arm=tag, path=path_kind, seed=seed, swap_seed=seed + 10_000,
                prior_init_seed=seed + 777 if arm in ("B2", "B3") else None,
                smoke=SMOKE, script=os.path.abspath(__file__), jax=jax.__version__,
                devices=[str(d) for d in jax.devices()],
                x64=bool(jax.config.jax_enable_x64), dim=DIM,
                R=R_B, NSYS=NSYS_B, K=K_B, ROUNDS=ROUNDS_B, thin=THIN_B,
                betas=betas.tolist(), L=L_MCLMC, ss_init=SS_INIT, devar=DEVAR,
                metric="pooled MAMS64 empirical cov, all rungs (positions only)",
                mams64=MAMS_NPZ, pocket_indicator=f"z[{POCKET_COL}] > {POCKET_THR}",
                init={"B1": "all rungs Bernoulli(0.5) main/pocket pool",
                      "B2": "rungs 1..R-1 Bernoulli(0.5); rung 0 prior draws",
                      "B3": "all main pool; rung 0 prior draws"}[arm],
                u_def="log_prob = logdensity/beta" if path_kind == "power"
                      else "log_like = (logdensity - log_prior)/beta",
                harness="run_pt (shared with Arm-0 control, amendment ii)",
                sep_check_max_abs=M["sep_max"])
    spec = dict(dim=DIM, L=L_MCLMC, betas=betas, NSYS=NSYS_B, K=K_B,
                ROUNDS=ROUNDS_B, inv_mass=M["cov_pool"],
                make_tf=lambda b: make_tempered(path_kind, b),
                u_from=u_from, u_direct=u_direct,
                indicator=lambda p: p[..., POCKET_COL] > POCKET_THR,
                init_pos=pos, card=card, ind_label="pocket")
    return run_pt(tag, seed, spec)


def run_control(tag):
    """Arm 0 (amendment ii, BLOCKING): (0a) known-answer weight gate on the
    June-28 pt_weight.py Gaussian-mixture target + (0b) transport calibration
    c_rw, BOTH through the SAME run_pt harness the dPIE B arms use."""
    seed = CTRL_SEED
    betas = np.geomspace(0.03, 1.0, CTRL_R)
    logW = jnp.log(jnp.asarray(np.array(CTRL_W)))
    MU = jnp.asarray(np.array([+CTRL_M, -CTRL_M]))
    cst = -0.5 * CTRL_D * jnp.log(2 * jnp.pi)

    def mix_logpdf(z):   # pt_weight.py logdensity_fn, math verbatim
        z0 = z[0]
        qr = jnp.sum(z[1:] ** 2)
        c0 = logW[0] + cst - 0.5 * ((z0 - MU[0]) ** 2 + qr)
        c1 = logW[1] + cst - 0.5 * ((z0 - MU[1]) ** 2 + qr)
        return jax.scipy.special.logsumexp(jnp.stack([c0, c1]))

    mix_v = jax.jit(jax.vmap(mix_logpdf))
    rng = np.random.default_rng(seed)
    # ALL replicas of ALL systems in the WRONG (-5, weight 0.3) basin, as in
    # pt_weight.py: tests discovery + unbiased weight through the new harness.
    init = rng.standard_normal((CTRL_R, CTRL_NSYS, CTRL_D))
    init[:, :, 0] += float(MU[1])

    card = dict(arm=tag, path="power", seed=seed, swap_seed=seed + 10_000,
                smoke=SMOKE, script=os.path.abspath(__file__), jax=jax.__version__,
                devices=[str(d) for d in jax.devices()],
                x64=bool(jax.config.jax_enable_x64), dim=CTRL_D,
                target="June-28 pt_weight.py known-answer: D=10 Gaussian mixture,"
                       " modes +/-5, weights 0.7/0.3, equal covariance",
                R=CTRL_R, NSYS=CTRL_NSYS, K=CTRL_K, ROUNDS=CTRL_ROUNDS,
                burn=CTRL_BURN, thin=THIN_B, betas=betas.tolist(),
                L=math.sqrt(CTRL_D), ss_init=SS_INIT, devar=DEVAR,
                metric="identity", indicator="z[0] > 0 (occ_+, +mode weight 0.70)",
                init="all rungs all systems in wrong (-5) basin (N(0,1) spread)",
                u_def="log_prob = logdensity/beta (power path)",
                harness="run_pt (SAME code path as B1/B2/B3)")
    spec = dict(dim=CTRL_D, L=math.sqrt(CTRL_D), betas=betas, NSYS=CTRL_NSYS,
                K=CTRL_K, ROUNDS=CTRL_ROUNDS, inv_mass=np.eye(CTRL_D),
                make_tf=lambda b: (lambda z, b=float(b): b * mix_logpdf(z)),
                u_from=lambda logd, p: logd / betas[:, None],
                u_direct=lambda pf: np.asarray(mix_v(jnp.asarray(pf))),
                indicator=lambda p: p[..., 0] > 0.0,
                init_pos=init, card=card, ind_label="plus-mode",
                occ_truth=CTRL_OCC_TRUTH)
    summary, data = run_pt(tag, seed, spec)

    # ---- 0a: known-answer weight gate --------------------------------------
    burn = min(CTRL_BURN, max(1, CTRL_ROUNDS // 3))
    per_sys = data["cold_ind"][burn:].astype(np.float64).mean(axis=0)
    occ = float(per_sys.mean())
    se = (float(per_sys.std(ddof=1) / math.sqrt(len(per_sys)))
          if len(per_sys) > 1 else float("inf"))
    tol = max(2.0 * se, CTRL_TOL_FLOOR)
    gate_pass = bool(abs(occ - CTRL_OCC_TRUTH) <= tol)
    # ---- 0b: transport calibration ------------------------------------------
    att, acc = data["att"], data["acc"]
    abar = float(acc.sum() / max(att.sum(), 1))
    rt_total = int(data["round_trips"].sum())
    rt_per_ladder = rt_total / CTRL_NSYS
    denom = CTRL_ROUNDS * abar / CTRL_R ** 2
    c_rw = float(rt_per_ladder / denom) if denom > 0 else float("nan")
    c_rw_in_band = bool(np.isfinite(c_rw)
                        and CRW_BAND[0] <= c_rw <= CRW_BAND[1])
    # exact tempered profile of the equal-cov mixture (minority mass per rung)
    w_minus = (CTRL_W[1] ** betas) / (CTRL_W[1] ** betas + CTRL_W[0] ** betas)
    ctrl = dict(
        burn=burn, occ_plus=occ, se_across_systems=se,
        occ_plus_per_system=per_sys.tolist(), truth=CTRL_OCC_TRUTH,
        tolerance=tol, tolerance_floor=CTRL_TOL_FLOOR,
        gate_0a_pass=gate_pass,
        abar_measured=abar,
        round_trips_total=rt_total, min_round_trips=CTRL_MIN_RT,
        rt_count_sufficient=bool(rt_total >= CTRL_MIN_RT),
        round_trips_per_system=data["round_trips"].tolist(),
        round_trips_per_ladder=rt_per_ladder,
        c_rw=c_rw, c_rw_band=list(CRW_BAND), c_rw_in_band=c_rw_in_band,
        w_minus_exact_profile=w_minus.tolist(),
        june28_reference_occ="0.6986 +/- 0.0122 (pt_weight lineage)")
    summary["control_gate"] = ctrl
    with open(os.path.join(OUT, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1, default=_json_default)
    pr(f"\n[0 {tag}] ===== CONTROL GATE (PROPOSED/UNCERTIFIED) =====")
    pr(f"  0a weight gate: cold occ_+ = {occ:.4f} +- {se:.4f} (across-system se); "
       f"|occ - 0.70| = {abs(occ - CTRL_OCC_TRUTH):.4f} vs tol {tol:.4f} "
       f"-> {'PASS' if gate_pass else 'FAIL'}")
    pr(f"  0b calibration: abar = {abar:.3f}; round trips total = {rt_total} "
       f"(need >= {CTRL_MIN_RT}: "
       f"{'OK' if rt_total >= CTRL_MIN_RT else 'INSUFFICIENT'}); "
       f"c_rw = {c_rw:.3f} (band [{CRW_BAND[0]}, {CRW_BAND[1]}]: "
       f"{'OK' if c_rw_in_band else 'OUT -> re-derive W-2/W-5 before Arm B'})")
    return summary, data


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="GATE PT-0 (one arm per process)")
    ap.add_argument("--arm", required=True,
                    choices=["smoke", "control", "A_power", "A_lik",
                             "B1", "B2", "B3"])
    args = ap.parse_args()

    if args.arm == "smoke" or os.environ.get("GATE_PT0_SMOKE", "0") == "1":
        apply_smoke()
    os.makedirs(OUT, exist_ok=True)
    if args.arm != "control":   # Arm 0 is analytic: no dPIE model build needed
        setup_model()

    def tag_of(arm):
        return f"{arm}_smoke" if SMOKE else arm

    if args.arm == "smoke":
        run_control(tag_of("control"))          # Arm 0 first (amendment ii)
        run_arm_a("power", tag_of("A_power"))
        run_arm_a("lik", tag_of("A_lik"))
        for arm in ("B1", "B2", "B3"):
            run_arm_b(arm, tag_of(arm))
    elif args.arm == "control":
        run_control(tag_of("control"))
    elif args.arm == "A_power":
        run_arm_a("power", tag_of("A_power"))
    elif args.arm == "A_lik":
        run_arm_a("lik", tag_of("A_lik"))
    else:
        run_arm_b(args.arm, tag_of(args.arm))


if __name__ == "__main__":
    main()
