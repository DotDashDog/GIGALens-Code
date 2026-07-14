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
  ST       GATE PT-5a two-phase self-tuning ladder mode (checkpoint "carousel
           GATE PT-5a", rd-3 CERTIFY-RECOMMENDED): phase 0 = swaps-OFF
           measurement leg on the pinned arm-A probe grid with a per-rung
           u-stationarity trigger + re-space through ladder_recipe; phase 1 =
           run_pt production leg on the re-spaced ladder (nearest-log-beta
           position carry-over, log-beta-interpolated metric seeds). MAP +
           1e-6*I entry as D2. NOT part of the composite --arm smoke; smoke it
           with GATE_PT0_SMOKE=1 --arm ST. --selftest-st = numpy-only unit
           checks of the ST numeric helpers (no model build).
  PR       GATE PT-5a-r2 probe->production mode (checkpoint "carousel GATE
           PT-5a-r2", rd-2 CERTIFY-RECOMMENDED): phase P (probe) = swaps-OFF
           BROAD-init (both MAMS basins, the arm-A draw_init convention) leg
           on the arm-A probe grid, fixed pooled metric; readiness = the
           rd-1 CUMULATIVE-CONVERGENCE re-derivation of (ladder knots,
           beta_min) on the trailing window [D0, t), comparing t vs t/1.5
           (replaces PT-5a's falsified in-run u-stationarity trigger).
           Phase Q (production) = run_pt on the probe's ladder with a FRESH
           MAP + 1e-6*I entry (D2 convention; NOT carried from the probe).
           NOT part of the composite --arm smoke; smoke it with
           GATE_PT0_SMOKE=1 --arm PR. --selftest-pr = numpy-only unit check
           of the readiness convergence logic against arrays_A_power.npz.
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
Smoke first: GATE_PT0_SMOKE=1 with per-arm --arm flags in parallel across GPUs
(~15 min wall; single-process --arm smoke is ~25-60 min: ~60 dPIE compiles).
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
import ladder_recipe   # PT-5a ST re-space (numpy-only; validated by pt4_recipe_validate)
from gigalens_research.inference.blackjax_updated_utils import (
    _build_kernel_shardmap, isokinetic_mclachlan_smart, _single_init, handle_nans)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ shared config
OUT = os.path.join(HERE, "carousel_gate_pt0_out")
DPIE = os.path.join(HERE, "..", "sim_carousel", "messy_tests", "dpie")
MAMS_NPZ = os.path.join(DPIE, "mams", "arrays.npz")
SVI_NPZ = os.path.join(DPIE, "svi", "arrays.npz")   # fallback metric only  # (B5: PRIMARY metric — comment updated rd-2)
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
ARM_SEED = dict(A_power=0, A_lik=0, B1=0, B2=1, B3=2, B4=3, B5=4, D1=5, D2=6,
                ST=7, PR=8)
                # checkpoint seeds (PT-0b/PT-1/PT-2 arms override via
                # GATE_PT0_SEED_B: 10-13 PT-0b, 20/21 PT-1, 30-33 PT-2;
                # PT-5a ST production arms 55/56/57; PT-5a-r2 PR production
                # arms via the same env override, seeds pinned at launch)
# PT-2 windowed mass-matrix adaptation (D arms; checkpoint "GATE PT-2")
METRIC_WINDOWS = (100, 250, 500)    # expanding boundaries; metric FROZEN after last
GENEIG_EVERY = 100                  # gen-eig ratio trace cadence (rounds)
MAP_NPZ = os.path.join(DPIE, "map", "arrays.npz")   # D2 entry point (z_best)
D2_INIT_SCALE = 1e-3                # stock no-SVI diagonal: init_scales=1e-3 STD
                                    # in z (ModellingSequence.SVI default,
                                    # gigalens/jax/inference.py:254,282-284;
                                    # pipeline.py:1440) => seed cov = 1e-6 * I
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
# GATE PT-5a ST (self-tune) two-phase mode; checkpoint "carousel GATE PT-5a"
ST_GRID_N = 10                      # phase-0 grid = arm-A probe grid,
                                    # geomspace(0.01, 1, 10) (PINNED)
ST_NSYS = 16                        # phase NSYS (checkpoint pin; env-overridable
                                    # via GATE_PT0_NSYS_B, card-recorded)
ST_ROUNDS0_MAX = 400                # phase-0 deadline (F-never past this)
ST_TRIG_FLOOR = 200                 # earliest trigger evaluation round (B8)
ST_TRIG_EVERY = 10                  # host-side trigger evaluation cadence
ST_TRIG_WIN = 100                   # trigger window (split-halves 50/50)
ST_RSTAR_FLOOR = 300                # B2' exposure floor: R* = max(trigger, 300)
ST_WINDOW0 = 100                    # phase-0 single metric boundary (scale-
                                    # finding); also the flip-count boundary (B8)
ST_MEAS_WIN = 100                   # sd(u)/tau window: rounds (R*-100, R*]
ST_ROUNDS1 = 1000                   # phase-1 production rounds (PT-2 frontier
                                    # precedent; GATE_PT0_ROUNDS_B overrides,
                                    # smoke > env)
ST_TARGET_NATS = 1.0                # design_ladder target (PT-0b convention)
ST_PHASE0_CHUNK = 6                  # phase-0 rung-batch: the 10-rung x NSYS
                                     # fused vmap (160-wide) OOMs a 40 GB A100;
                                     # swaps-OFF => rungs independent => chunk
                                     # of 6 (96-wide, the proven-fitting size)
                                     # is EQUIVALENT up to XLA vmap-width
                                     # FP-reorder (NOT bitwise; measured 3.8e-8
                                     # chunk-2-vs-6 — a distinct-but-valid
                                     # realization, same per-rung kernel/keys/
                                     # target; GATE_PT0_ST_CHUNK overrides;
                                     # smoke-discovered OOM fix)
ST_RULE_NSYS = 16                   # beta_min discovery-rule constants: the
ST_RULE_BUDGET = 16500              # pinned ln(100)/(16*11) threshold
ST_RULE_PROBE = 1500                # (pt4_recipe_validate L1b convention)
ST_MIN_COUNT = 2                    # B2' k >= 2 minimum-count guard
ST_LEAK_UNIT = 1500.0               # leakage rates normalized per 1500 kernel
                                    # steps; exposure (R*-100)*K*NSYS is the
                                    # SHARED runner/scorer denominator (rd-3
                                    # pin 1)

# GATE PT-5a-r2 PR (probe -> production) two-phase mode; checkpoint "carousel
# GATE PT-5a-r2". rd-1 READINESS REDESIGN (CUMULATIVE-CONVERGENCE re-
# derivation of the ladder/beta_min itself, replacing the falsified crossing-
# count/occupancy-stationarity proxy) + rd-2 4 pre-run items are BINDING over
# the original "Scheme"/"Win conditions" text. Phase P (probe): run_arm_a-
# style BROAD init (both MAMS basins), the arm-A probe grid, swaps OFF, a
# FIXED (non-adaptive) pooled metric; readiness = periodic re-derivation of
# (ladder knots, beta_min) on the cumulative trailing window [D0, t), fixed
# D0, comparing t vs t/1.5. Phase Q (production): the D2/C-24 production leg
# (FRESH MAP + 1e-6*I entry -- NOT carried from the probe) run through run_pt
# on the probe's ladder.
PR_GRID_N = 10                      # phase-P grid = arm-A probe grid (PINNED)
PR_NSYS = 16                         # phase-P NSYS (matches ST_NSYS convention;
                                     # half M-pool / half P-pool per rung, 8+8,
                                     # mirroring run_arm_a's draw_init)
PR_D0 = 10                           # readiness fixed discard, ROUNDS (rd-1:
                                     # "D0 = 100 STEPS" at K=10 steps/round)
PR_READY_EVERY = 10                  # readiness re-derivation cadence, ROUNDS
                                     # (rd-1: "every READINESS_EVERY rounds
                                     # (e.g. 10)")
PR_READY_FLOOR = 30                  # earliest readiness evaluation round (a
                                     # "small min"): derived, not guessed --
                                     # st_tau_rounds needs window >= 4 rounds
                                     # for BOTH the t and t/1.5 windows, so
                                     # floor/1.5 - PR_D0 >= 4 => floor >=
                                     # 1.5*(PR_D0+4) = 21; 30 gives margin
PR_CAP_ROUNDS = 200                  # readiness CAP: 2000 steps / K=10 = 200
                                     # rounds (rd-2 item 2, pinned above
                                     # seed-A's 1200-step stabilization with
                                     # cross-seed margin)
PR_PHASE0_CHUNK = ST_PHASE0_CHUNK    # reuse the 96-wide OOM fix chunk width
PR_RULE_NSYS = ST_RULE_NSYS          # pinned ln(100) discovery-rule constants,
PR_RULE_BUDGET = ST_RULE_BUDGET      # REUSED unchanged (same threshold as the
                                     # certified table, not re-derived)
PR_TARGET_NATS = ST_TARGET_NATS      # design_ladder target (PT-0b convention)

SMOKE = False
# Wall-clock fix (2026-07-11): the PT round is ONE fused jitted call vmapped over
# rungs with beta as a traced argument (the per-rung dispatch loop was fixed-cost
# dominated: ~34 s/round in smoke => ~19 h/arm). GATE_PT0_LEGACY_RUNNERS=1
# selects the old per-rung path (kept for --equiv-check, not for production).
LEGACY_RUNNERS = os.environ.get("GATE_PT0_LEGACY_RUNNERS", "0") == "1"


def apply_smoke():
    """SHAPE-FAITHFUL smoke (amendment iii, BLOCKING; the GATE L attempt-1 XLA
    lesson): FULL production vmap/compile shapes -- all 10 Arm-A beta compiles at
    full 32-wide, both hot-end checks at full width, Arm B at full R=12 x NSYS=8
    exercising the jitted swap-sync round loop AND the incremental-npz save path,
    control at full R=10 x NSYS=16 -- ONLY step/round counts are reduced.
    Numbers are NOT the pre-registered measurement."""
    global SMOKE, STEPS_A, DISCARD_A
    global ROUNDS_B, SAVE_EVERY_B, PRINT_EVERY_B, LAST_OCC, RHAT_LAST
    global CTRL_ROUNDS, CTRL_BURN, METRIC_WINDOWS, GENEIG_EVERY
    global ST_ROUNDS0_MAX, ST_TRIG_FLOOR, ST_TRIG_EVERY, ST_TRIG_WIN
    global ST_RSTAR_FLOOR, ST_WINDOW0, ST_MEAS_WIN, ST_ROUNDS1
    SMOKE = True
    STEPS_A, DISCARD_A = 60, 30
    ROUNDS_B = 20
    SAVE_EVERY_B, PRINT_EVERY_B = 10, 5   # SAVE_EVERY=10 so the npz path fires
    LAST_OCC, RHAT_LAST = 10, 20
    CTRL_ROUNDS, CTRL_BURN = 30, 10
    # PT-2 D-arm smoke: boundaries scaled purely to exercise the window/freeze/
    # save code paths at ROUNDS=20 (recorded in the model card)
    METRIC_WINDOWS, GENEIG_EVERY = (5, 10, 15), 5
    # PT-5a ST smoke: tiny two-phase schedule exercising swaps-off phase 0,
    # trigger firing (FORCED at the floor if the natural test fails), re-space
    # + handoff save, restart into phase 1, and all new npz keys. Exposure
    # floor waived (LOUD print at R* selection); phase-0 metric boundary at
    # round 2 keeps floor = boundary + window (production 200 = 100 + 100) so
    # the trigger window sits fully post-boundary (B8); phase-1 windows are
    # the (5, 10, 15) METRIC_WINDOWS above. Numbers are NOT the measurement.
    ST_ROUNDS0_MAX, ST_TRIG_FLOOR, ST_TRIG_EVERY = 24, 12, 2
    ST_TRIG_WIN, ST_RSTAR_FLOOR = 10, 16
    ST_WINDOW0, ST_MEAS_WIN, ST_ROUNDS1 = 2, 10, 20
    # GATE PT-5a-r2 PR smoke: tiny readiness schedule exercising the
    # cumulative-window re-derivation, a forced-readiness fallback (mirrors
    # the ST forced-trigger pattern) if the natural test hasn't converged by
    # the floor, the probe->handoff->phase-Q path, and all new npz keys.
    # PR_READY_FLOOR=10 keeps BOTH the t and t/1.5=6 windows >= 4 rounds
    # (PR_D0=2 -> window_prev=6-2=4, the st_tau_rounds minimum). Numbers are
    # NOT the measurement.
    global PR_D0, PR_READY_EVERY, PR_READY_FLOOR, PR_CAP_ROUNDS
    PR_D0, PR_READY_EVERY, PR_READY_FLOOR, PR_CAP_ROUNDS = 2, 2, 10, 20


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

    # C-8 duty (PT-3): every run log carries the z column -> parameter name
    # map (the C-8 trap: column names were once mis-assigned by assuming
    # insertion order; print the authoritative sorted-key mapping, never assume)
    names = [str(n) for n in prob_model.z_param_names]
    pr(f"[setup] z_param_names ({len(names)} columns; C-8 column->name map):")
    for i, nm in enumerate(names):
        pr(f"  z[{i:2d}] = {nm}")

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
    (1-beta)*log_prior + beta*log_prob (identical by the verified split; one sim).
    `beta` may be a Python/numpy float (legacy per-rung path, Arm A) OR a traced
    jax scalar (fused runner, vmapped over rungs): no coercion, and the Python
    branch is on path_kind only, never on beta."""
    b = beta
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
              dim=DIM, devar=DEVAR):
    """carousel_pt.py adapt_one, math EXACT (parameterized by D=dim and the EEVPD
    target devar, default 5e-4); also returns the handle_nans success flag so
    reverts can be counted (checkpoint NaN policy). dim=33 for dPIE arms, 10 for
    the Arm-0 control. devar is overridable for B arms ONLY via GATE_PT0_DEVAR
    (PT-1 C2 bias probe); Arm A and the control always use the default."""
    time_, x_avg, ss_max = adapt_state
    success, state, ss_max, ec = handle_nans(
        prev_state, next_state, step_size, ss_max, info.energy_change, nan_key)
    xi = jnp.square(ec) / (dim * devar) + 1e-8
    weight = jnp.exp(-0.5 * jnp.square(jnp.log(xi) / (6.0 * TRUST)))
    x_avg = DECAY * x_avg + weight * (xi / jnp.power(step_size, 6.0))
    time_ = DECAY * time_ + weight
    new_step = jnp.minimum(jnp.power(x_avg / time_, -1.0 / 6.0), ss_max)
    return state, new_step, (time_, x_avg, ss_max), jnp.square(info.energy_change), success


def fresh_adapt(nch, ss_max=SS_MAX0):
    """Initial adapt state; ss_max is the initial step-size cap consumed by
    handle_nans (0.8x shrink on NaN) and adapt_one's min(new_step, ss_max).
    PT-0b overrides it for B arms via GATE_PT0_SSMAX (control stays 1.0)."""
    return (jnp.zeros(nch), jnp.zeros(nch), jnp.full(nch, ss_max))


def fresh_adapt2(n_rungs, nch, ss_max=SS_MAX0):
    """(R, NSYS)-shaped adapt state for the fused round runner."""
    return (jnp.zeros((n_rungs, nch)), jnp.zeros((n_rungs, nch)),
            jnp.full((n_rungs, nch), ss_max))


def round_all_chunked(fn, pos, ss, ast, ik, sk, betas, inv_mass, chunk):
    """Rung-chunked round_all for PT-5a phase-0 memory. round_all is
    jax.vmap(per_rung) over the rung axis with NO cross-rung coupling, and
    phase-0 runs SWAPS-OFF, so splitting the R rungs into chunks of `chunk`
    and concatenating every output on axis 0 feeds each rung the EXACT same
    inputs (positions/step-sizes/adapt-state/keys/betas/inv_mass SLICED on
    axis 0, NOT re-split) it would get in the fused call. NOT bitwise: XLA
    fuses different vmap widths with different matmul reduction order, so the
    chunked result differs from the single R-wide call at the FP-reorder level
    (measured chunk-2-vs-6 max|dpos| 3.8e-8, then chaotically amplified) — a
    distinct-but-valid realization of the identical per-rung kernel, the same
    phenomenon accepted for the fused-vs-legacy production runner (op-6). Its
    validity for the distributional phase-0 estimands (sd(u), leakage) was
    grader-ruled VALID-WITH-CONDITION (chunk=6-vs-5 statistic-invariance check
    2026-07-14). Motivation: the 10-rung x NSYS=160-wide fused vmap overflows
    a 40 GB A100 on the 90k-pixel forward model; chunk=6 (96-wide, the
    proven-fitting production width) fits. chunk >= R is the identity."""
    R = int(pos.shape[0])
    if chunk >= R:
        return fn(pos, ss, ast, ik, sk, betas, inv_mass)
    P, L, S, A0, A1, A2, E, O = ([] for _ in range(8))
    for c in range(0, R, chunk):
        sl = slice(c, min(c + chunk, R))
        p2, ld, s2, ad2, ec2, ok = fn(
            pos[sl], ss[sl], (ast[0][sl], ast[1][sl], ast[2][sl]),
            ik[sl], sk[sl], betas[sl], inv_mass[sl])
        P.append(p2); L.append(ld); S.append(s2)
        A0.append(ad2[0]); A1.append(ad2[1]); A2.append(ad2[2])
        E.append(ec2); O.append(ok)
    cat = lambda xs: jnp.concatenate(xs, axis=0)
    return (cat(P), cat(L), cat(S), (cat(A0), cat(A1), cat(A2)),
            cat(E), cat(O))


def regularize_cov_np(cov, n):
    """Host copy of the PRODUCTION _regularize_cov math (mclmc.py:154-164),
    verbatim in numpy: symmetrize; Stan window shrinkage n/(n+5) with
    1e-3*(5/(n+5))*I; PSD eigenvalue floor at the shrink level. (The production
    code applies the shrinkage to the covariance directly -- no correlation
    scaling -- and that code is the copied authority.)"""
    cov = 0.5 * (cov + cov.T)
    n = float(n)
    eye = np.eye(cov.shape[-1])
    shrink = 1e-3 * (5.0 / (n + 5.0))
    reg = (n / (n + 5.0)) * cov + shrink * eye
    w, V = np.linalg.eigh(reg)
    w = np.clip(w, shrink, None)
    return (V * w[None, :]) @ V.T


def geneig_ratios(adapted, ref_chol):
    """Generalized-eigenvalue ratios of (adapted vs reference) covariance:
    eigvalsh of Lref^-1 adapted Lref^-T -- the solve(reference, adapted)-style
    pencil spectrum (PT-2 internals instrument)."""
    m = np.linalg.solve(ref_chol, np.linalg.solve(ref_chol, adapted).T).T
    return np.linalg.eigvalsh(0.5 * (m + m.T))


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


def make_runner_b(tf, kern, dim, L, devar=DEVAR):
    """Per-rung jitted K-step runner (carousel_pt.py _mk pattern), with the per-round
    momentum refresh (init_level pattern) folded into the jit: states are rebuilt
    from positions each call, so swaps only move positions/u; adapt stays with rung.
    Shared by ALL PT arms (B1..B5 dPIE and the Arm-0 control; amendment ii)."""

    @jax.jit
    def run(positions, ss, ast, init_keys, step_keys):  # step_keys (K, NSYS)
        states = jax.vmap(lambda p, k: _single_init(p, tf, k))(
            jnp.asarray(positions), init_keys)

        def sstep(carry, kt):
            sts, sss, asts = carry

            def per(s, step, a, k):
                k1, k2 = jax.random.split(k)
                ns, info = kern(rng_key=k1, state=s, L=L, step_size=step)
                return adapt_one(s, ns, info, step, a, k2, dim, devar)

            s2, ss2, a2, ec2, ok = jax.vmap(per)(sts, sss, asts, kt)
            return (s2, ss2, a2), (ec2, ok)

        (states, sss, asts), (ec2, ok) = jax.lax.scan(
            sstep, (states, ss, ast), step_keys)
        return states.position, states.logdensity, sss, asts, ec2, ok

    return run


def make_legacy_runners(spec):
    """Old per-rung dispatch path (one jitted runner per rung, concrete beta).
    Kept ONLY for the --equiv-check guard and GATE_PT0_LEGACY_RUNNERS=1."""
    imm = jnp.asarray(spec["inv_mass"])
    runners = []
    for b in np.asarray(spec["betas"], dtype=np.float64):
        tf = spec["make_tf"](float(b))
        kern = _build_kernel_shardmap(logdensity_fn=tf, inverse_mass_matrix=imm,
                                      integrator=isokinetic_mclachlan_smart)
        runners.append(make_runner_b(tf, kern, spec["dim"], spec["L"],
                                     float(spec.get("devar", DEVAR))))
    return runners


def make_fused_round_runner(spec):
    """Wall-clock fix (2026-07-11): ONE jitted call per PT round, vmapped over
    rungs with beta as a mapped (traced) argument -- removes the R sequential
    per-rung dispatches that dominated the round cost. tf and the kernel are
    built INSIDE the traced per_rung context (all construction is pure jax:
    _build_kernel_shardmap composes jnp.where-based steps; the integrator's
    cholesky of the inverse-mass matrix runs in-trace; beta and -- since PT-2 --
    the per-rung inv_mass are traced arguments, so host-side metric updates
    change only VALUES, never the jit cache key). Inner chain vmap is
    unchanged, so the fused batch is (R, NSYS). Math and per-rung/chain/step
    key semantics identical to make_runner_b (verified by --equiv-check up to
    FP reordering)."""
    dim, L, K = spec["dim"], spec["L"], spec["K"]
    devar = float(spec.get("devar", DEVAR))
    make_tf = spec["make_tf"]

    @jax.jit
    def round_all(positions, ss, ast, init_keys, step_keys, betas, inv_mass):
        # positions (R,NSYS,dim); ss (R,NSYS); ast tuple of (R,NSYS);
        # init_keys (R,NSYS); step_keys (R,K,NSYS); betas (R,);
        # inv_mass (R,dim,dim) TRACED, vmapped over rungs (PT-2: host-side
        # windowed metric adaptation updates its VALUE between rounds; the
        # shape/dtype are static, so jit's cache key is unchanged and no
        # recompile occurs -- the Cholesky runs inside the trace).
        def per_rung(pos_r, ss_r, ast_r, ik_r, sk_r, beta, im_r):
            tf = make_tf(beta)                    # beta TRACED here
            kern = _build_kernel_shardmap(
                logdensity_fn=tf, inverse_mass_matrix=im_r,
                integrator=isokinetic_mclachlan_smart)
            states = jax.vmap(lambda p, k: _single_init(p, tf, k))(pos_r, ik_r)

            def sstep(carry, kt):
                sts, sss, asts = carry

                def per(s, step, a, k):
                    k1, k2 = jax.random.split(k)
                    ns, info = kern(rng_key=k1, state=s, L=L, step_size=step)
                    return adapt_one(s, ns, info, step, a, k2, dim, devar)

                s2, ss2, a2, ec2, ok = jax.vmap(per)(sts, sss, asts, kt)
                return (s2, ss2, a2), (ec2, ok)

            (states, sss, asts), (ec2, ok) = jax.lax.scan(
                sstep, (states, ss_r, ast_r), sk_r)
            return states.position, states.logdensity, sss, asts, ec2, ok

        return jax.vmap(per_rung)(positions, ss, ast, init_keys, step_keys,
                                  betas, inv_mass)

    return round_all


def _round_keys(lks, R, NSYS, K):
    """Per-round key arrays with the SAME split structure per rung/chain/step as
    the legacy loop (ik = split(lks[r], NSYS); sk = split(fold_in(lks[r],1),
    K*NSYS).reshape(K, NSYS)) -- trajectories comparable up to FP reordering."""
    ik_all = jnp.stack([jax.random.split(lks[r], NSYS) for r in range(R)])
    sk_all = jnp.stack([jax.random.split(jax.random.fold_in(lks[r], 1),
                                         K * NSYS).reshape(K, NSYS)
                        for r in range(R)])
    return ik_all, sk_all


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
    seed_source = "ARM_SEED default"
    env_seed = os.environ.get("GATE_PT0_SEED_A", "").strip()
    if env_seed:   # PT-4 probe: seed 54 per checkpoint (L2 fresh-probe arm)
        seed = int(env_seed)
        seed_source = f"env override GATE_PT0_SEED_A={env_seed}"
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)
    betas = BETAS_A
    card = dict(arm=tag, path=path_kind, seed=seed, seed_source=seed_source,
                smoke=SMOKE,
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
    fused = not LEGACY_RUNNERS
    card = dict(spec["card"],
                runner_impl="fused-vmap-over-rungs" if fused
                            else "legacy-per-rung-dispatch")
    pr("MODEL CARD:", json.dumps(card, indent=1, default=_json_default))
    summary = dict(model_card=card)
    t0_all = time.time()

    # runner: ONE fused jitted call per round (default; wall-clock fix) or the
    # legacy per-rung dispatch list (GATE_PT0_LEGACY_RUNNERS=1 / equiv guard)
    adapt_metric = bool(spec.get("adapt_metric", False))
    if adapt_metric and not fused:
        raise RuntimeError("adaptive metric (PT-2 D arms) requires the fused "
                           "runner; unset GATE_PT0_LEGACY_RUNNERS")
    if fused:
        round_all = make_fused_round_runner(spec)
        betas_j = jnp.asarray(betas)
    else:
        runners = make_legacy_runners(spec)
    # per-rung metric stack (R, dim, dim); constant for non-adaptive arms,
    # host-updated at window boundaries for adapt_metric arms (PT-2 M1/M2)
    inv_rungs = np.broadcast_to(
        np.asarray(spec["inv_mass"], dtype=np.float64), (R, dim, dim)).copy()
    inv_rungs_j = jnp.asarray(inv_rungs)
    windows = tuple(spec.get("metric_windows") or METRIC_WINDOWS)
    n0 = float(spec.get("metric_n0") or (10 * NSYS))
    metric_ref = spec.get("metric_ref")
    ref_chol = (np.linalg.cholesky(np.asarray(metric_ref, dtype=np.float64))
                if (adapt_metric and metric_ref is not None) else None)
    metric_est = str(spec.get("metric_estimator") or "pooled")
    assert metric_est in ("pooled", "within"), metric_est
    w_n = 0                                       # Welford over ROUND-END pos
    w_mean = np.zeros((R, dim))
    w_M2 = np.zeros((R, dim, dim))
    # PT-4 within/between decomposition: s_C sums the per-round cross-chain
    # ddof-1 cov C_t; bm_* = Welford over the round ensemble-means. Recorded
    # at every boundary in BOTH estimator modes.
    s_C = np.zeros((R, dim, dim))
    bm_n = 0
    bm_mean = np.zeros((R, dim))
    bm_M2 = np.zeros((R, dim, dim))
    metric_frozen_flag = False
    boundary_covs, geneig_trace, geneig_rounds = [], [], []
    within_covs, between_covs = [], []

    pos = np.array(spec["init_pos"], dtype=np.float64)
    assert pos.shape == (R, NSYS, dim), pos.shape
    init_cold_occ = float(indicator(pos[R - 1]).mean())
    pr(f"[B {tag}] init cold occ = {init_cold_occ:.3f}; "
       f"init occ per rung = "
       f"{np.round(indicator(pos).mean(axis=1), 2).tolist()}")

    ss_max0 = float(spec.get("ss_max", SS_MAX0))   # B arms may override (PT-0b)
    if fused:
        steps_all = jnp.full((R, NSYS), SS_INIT)
        adapt_all = fresh_adapt2(R, NSYS, ss_max0)
    else:
        steps = [jnp.full(NSYS, SS_INIT) for _ in range(R)]
        adapt = [fresh_adapt(NSYS, ss_max0) for _ in range(R)]
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
    wid_thin = np.zeros((n_thin, R, NSYS), dtype=np.int16)  # rd-2 A4: F-2 forensics
    cold_ind = np.zeros((ROUNDS, NSYS), dtype=np.uint8)
    ev = np.zeros((ROUNDS, R))
    ssm = np.zeros((ROUNDS, R))
    att = np.zeros((R - 1, 2), dtype=int)   # [pair, 0=same 1=cross] attempts
    acc = np.zeros((R - 1, 2), dtype=int)
    n_revert = 0
    u0_verify, u0_rel = None, None
    npz_path = os.path.join(OUT, f"arrays_{tag}.npz")

    def save_npz(t):
        extra = {}
        if adapt_metric:   # PT-2 metric-adaptation artifacts
            extra = dict(
                metric_frozen=inv_rungs,                    # per rung, current
                metric_boundary_covs=(np.stack(boundary_covs)
                                      if boundary_covs else np.zeros((0,))),
                metric_within_covs=(np.stack(within_covs)
                                    if within_covs else np.zeros((0,))),
                metric_between_covs=(np.stack(between_covs)
                                     if between_covs else np.zeros((0,))),
                metric_geneig_trace=(np.stack(geneig_trace)
                                     if geneig_trace else np.zeros((0,))),
                metric_geneig_rounds=np.asarray(geneig_rounds, dtype=np.int64),
                metric_windows=np.asarray(windows, dtype=np.int64),
                metric_n0=n0,
                metric_estimator=np.array(metric_est),
            )
        np.savez(npz_path, betas=betas, ind_thin=ind_thin[:t // THIN_B + 1],
                 wid_thin=wid_thin[:t // THIN_B + 1],
                 cold_ind=cold_ind[:t + 1], eevpd=ev[:t + 1], step_mean=ssm[:t + 1],
                 swap_attempts=att, swap_accepts=acc,
                 walker_id=wid, walker_flag=wflag,   # machine reconstructible (fix 1)
                 down_traverses=down_traverses, round_trips=round_trips,
                 round_trips_main=round_trips_main,
                 round_trips_pocket=round_trips_pocket,
                 pocket_cold_arrivals=pocket_cold_arr,
                 main_cold_arrivals=main_cold_arr, n_revert=np.int64(n_revert),
                 init_cold_occ=init_cold_occ, rounds_done=np.int64(t + 1),
                 **extra)

    t0 = time.time()
    logd = np.zeros((R, NSYS))
    for t in range(ROUNDS):
        key, *lks = jax.random.split(key, R + 1)
        if fused:   # one jitted (R, NSYS)-wide call per round
            ik_all, sk_all = _round_keys(lks, R, NSYS, K)
            p2, ld, steps_all, adapt_all, ec2, ok = round_all(
                jnp.asarray(pos), steps_all, adapt_all, ik_all, sk_all,
                betas_j, inv_rungs_j)
            pos = np.array(p2)              # writable copy (host swaps mutate)
            logd = np.asarray(ld)
            ev[t] = np.asarray(ec2).mean(axis=(1, 2)) / dim
            ssm[t] = np.asarray(steps_all).mean(axis=1)
            n_revert += int((~np.asarray(ok)).sum())
        else:       # legacy per-rung dispatch (equiv guard only)
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
            wid_thin[t // THIN_B] = wid
        if adapt_metric and not metric_frozen_flag:
            # per-rung Welford over ROUND-END (post-swap) positions
            for r in range(R):
                batch = pos[r]                            # (NSYS, dim)
                bmean = batch.mean(axis=0)
                bctr = batch - bmean
                bM2 = bctr.T @ bctr
                if w_n == 0:
                    w_mean[r], w_M2[r] = bmean, bM2
                else:
                    delta = bmean - w_mean[r]
                    tot = w_n + NSYS
                    w_M2[r] += bM2 + np.outer(delta, delta) * w_n * NSYS / tot
                    w_mean[r] += delta * NSYS / tot
                # PT-4: within (C_t reuses bM2) + between (Welford over bmean)
                s_C[r] += bM2 / (NSYS - 1)
                delta_b = bmean - bm_mean[r]
                bm_mean[r] += delta_b / (bm_n + 1)
                bm_M2[r] += np.outer(delta_b, bmean - bm_mean[r])
            w_n += NSYS
            bm_n += 1
            if (t + 1) in windows:
                # window boundary: previous metric = prior at weight n0; the
                # window estimate REPLACES the metric (production-style),
                # regularized by the copied _regularize_cov rule; Welford resets
                n_w = w_n
                T = bm_n                    # rounds this window (= w_n // NSYS)
                if T < 2:
                    raise RuntimeError(
                        f"metric window boundary at round {t + 1}: only T={T} "
                        f"round(s) accumulated; B needs T >= 2 (never default)")
                W_win = s_C / T             # PT-4 within: mean cross-chain C_t
                B_win = bm_M2 / (T - 1)     # PT-4 between: ddof-1 over bmean
                for r in range(R):
                    cov_w = (W_win[r] if metric_est == "within"
                             else w_M2[r] / max(n_w - 1, 1))
                    comb = (n0 * inv_rungs[r] + n_w * cov_w) / (n0 + n_w)
                    inv_rungs[r] = regularize_cov_np(comb, n0 + n_w)
                boundary_covs.append(inv_rungs.copy())
                within_covs.append(W_win)   # recorded in BOTH estimator modes
                between_covs.append(B_win)
                inv_rungs_j = jnp.asarray(inv_rungs)
                # step-size adapt running averages RESET (production
                # behaviour); ss_max carried
                adapt_all = (jnp.zeros_like(adapt_all[0]),
                             jnp.zeros_like(adapt_all[1]), adapt_all[2])
                w_n, w_mean[:], w_M2[:] = 0, 0.0, 0.0
                bm_n, s_C[:], bm_mean[:], bm_M2[:] = 0, 0.0, 0.0, 0.0
                if (t + 1) == windows[-1]:
                    metric_frozen_flag = True     # FROZEN from here on
                pr(f"[B {tag}] metric window boundary at round {t + 1}: "
                   f"n_w={n_w}, n0={n0:.0f}, est={metric_est}"
                   f"{' (METRIC FROZEN)' if metric_frozen_flag else ''}")
        if adapt_metric and ref_chol is not None and (
                t % GENEIG_EVERY == 0 or t == ROUNDS - 1):
            geneig_rounds.append(t)
            geneig_trace.append(np.stack(
                [geneig_ratios(inv_rungs[r], ref_chol) for r in range(R)]))
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
    if adapt_metric:
        gfin = geneig_trace[-1] if geneig_trace else None   # (R, dim) ratios
        summary["metric_adapt"] = dict(
            windows=list(windows), n0=n0, estimator=metric_est,
            frozen=bool(metric_frozen_flag),
            n_boundaries_applied=len(boundary_covs),
            reference=("pooled MAMS64 cov — DIAGNOSTIC ONLY, NOT the scored clause "
                       "instrument)" if metric_ref is not None else None),
            geneig_final_cold_min=(float(gfin[R - 1].min())
                                   if gfin is not None else None),
            geneig_final_cold_max=(float(gfin[R - 1].max())
                                   if gfin is not None else None),
            geneig_final_min_per_rung=(gfin.min(axis=1).tolist()
                                       if gfin is not None else None),
            geneig_final_max_per_rung=(gfin.max(axis=1).tolist()
                                       if gfin is not None else None),
            geneig_pooledref_DIAGNOSTIC_in_band=(   # NOT scored: scored clause = Sigma_ref(w-hat), pt2 scorer
                bool(np.all((gfin >= 1.0 / 3.0) & (gfin <= 3.0)))
                if gfin is not None else None),
        )
        if gfin is not None:
            pr(f"  metric adapt: {len(boundary_covs)} boundaries applied, "
               f"frozen={metric_frozen_flag}; final gen-eig cold rung "
               f"[{gfin[R - 1].min():.3f}, {gfin[R - 1].max():.3f}] "
               f"(band [1/3, 3] all rungs: "
               f"{summary['metric_adapt']['geneig_pooledref_DIAGNOSTIC_in_band']} "
               f"[DIAGNOSTIC vs pooled ref — scored clause is vs Sigma_ref(w-hat), pt2 scorer])")
    save_npz(ROUNDS - 1)

    if adapt_metric and geneig_trace:
        fig, ax = plt.subplots(figsize=(9, 4))
        gt = np.stack(geneig_trace)                 # (n_snap, R, dim)
        for a in range(dim):
            ax.plot(geneig_rounds, gt[:, R - 1, a], lw=0.6, alpha=0.7)
        for b in (1.0 / 3.0, 3.0):
            ax.axhline(b, ls="--", color="k", lw=0.9)
        for wb in windows:
            ax.axvline(wb, ls=":", color="gray", lw=0.7)
        ax.set_yscale("log")
        ax.set_xlabel("round")
        ax.set_ylabel("gen-eig ratio (adapted vs reference)")
        ax.set_title(f"{tag}: cold-rung per-axis gen-eig ratios vs POOLED ref — "
                     f"DIAGNOSTIC ONLY (scored clause: vs Sigma_ref(w-hat), pt2 scorer)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"pt0_{tag}_geneig.png"), dpi=120)
        plt.close(fig)

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


def run_equiv_check(tag, seed, spec, n_rounds=3):
    """MANDATORY equivalence guard for the fused runner (2026-07-11): run
    n_rounds with BOTH implementations from identical inits/keys (SMOKE config)
    and report max |dpos|, max |du| per round plus both round-0 u-identity
    rels. Kernel path only: the host swap/walker logic is shared code operating
    on these outputs, so kernel equivalence implies run equivalence up to FP
    reordering."""
    dim = spec["dim"]
    betas = np.asarray(spec["betas"], dtype=np.float64)
    R, NSYS, K = len(betas), spec["NSYS"], spec["K"]
    pr(f"[EQUIV {tag}] fused vs legacy, {n_rounds} rounds, R={R} NSYS={NSYS} "
       f"K={K} dim={dim} seed={seed}")
    round_all = make_fused_round_runner(spec)
    runners = make_legacy_runners(spec)
    betas_j = jnp.asarray(betas)

    pos_l = np.array(spec["init_pos"], dtype=np.float64)
    pos_f = pos_l.copy()
    ss_max0 = float(spec.get("ss_max", SS_MAX0))
    steps_l = [jnp.full(NSYS, SS_INIT) for _ in range(R)]
    adapt_l = [fresh_adapt(NSYS, ss_max0) for _ in range(R)]
    steps_f = jnp.full((R, NSYS), SS_INIT)
    adapt_f = fresh_adapt2(R, NSYS, ss_max0)
    key = jax.random.key(seed)
    logd_l = np.zeros((R, NSYS))
    rows = []
    for t in range(n_rounds):
        key, *lks = jax.random.split(key, R + 1)
        ik_all, sk_all = _round_keys(lks, R, NSYS, K)   # SAME keys for both
        for r in range(R):
            p2, ld, steps_l[r], adapt_l[r], _, _ = runners[r](
                pos_l[r], steps_l[r], adapt_l[r], ik_all[r], sk_all[r])
            pos_l[r] = np.asarray(p2)
            logd_l[r] = np.asarray(ld)
        pf, ldf, steps_f, adapt_f, _, _ = round_all(
            jnp.asarray(pos_f), steps_f, adapt_f, ik_all, sk_all, betas_j,
            jnp.asarray(np.broadcast_to(
                np.asarray(spec["inv_mass"], dtype=np.float64),
                (R, dim, dim))))
        pos_f = np.array(pf)
        logd_f = np.asarray(ldf)
        u_l = spec["u_from"](logd_l, pos_l)
        u_f = spec["u_from"](logd_f, pos_f)
        row = dict(round=t,
                   max_dpos=float(np.max(np.abs(pos_l - pos_f))),
                   max_du=float(np.max(np.abs(u_l - u_f))))
        if t == 0:
            for nm, uu, pp in (("legacy", u_l, pos_l), ("fused", u_f, pos_f)):
                ud = np.asarray(spec["u_direct"](pp.reshape(-1, dim))
                                ).reshape(R, NSYS)
                row[f"u0_rel_{nm}"] = float(
                    np.max(np.abs(uu - ud) / (1.0 + np.abs(ud))))
        rows.append(row)
        pr(f"[EQUIV {tag}] round {t}: max|dpos| = {row['max_dpos']:.3e}, "
           f"max|du| = {row['max_du']:.3e}"
           + (f", u0_rel legacy/fused = {row['u0_rel_legacy']:.3e}/"
              f"{row['u0_rel_fused']:.3e}" if t == 0 else ""))
    out = dict(tag=tag, seed=seed, smoke=SMOKE, R=R, NSYS=NSYS, K=K, dim=dim,
               n_rounds=n_rounds, rounds=rows)
    with open(os.path.join(OUT, f"summary_equiv_{tag}.json"), "w") as f:
        json.dump(out, f, indent=1, default=_json_default)
    pr(f"[EQUIV {tag}] written -> summary_equiv_{tag}.json (UNCERTIFIED; "
       f"grader reads the numbers, not this print)")
    return out


def run_arm_b(arm, tag, equiv=False):
    """dPIE PT arms B1..B5 + D1/D2: build the target spec, route through run_pt.
    B4 (PT-0b) = POWER path, ALL-MAIN init at every rung/system, no prior draws.
    B5 (PT-1 C1, production composition) = POWER path, identical to B4 EXCEPT
    metric = SVI covariance (dpie/svi qz_scale_tril -> cov) and init = SVI draws
    at every rung/system (the true point-and-go input state).
    D1/D2 (PT-2 M1/M2): POWER path with host-side WINDOWED mass-matrix
    adaptation (adapt_metric=True). D1: seed cov = SVI cov, inits = SVI draws
    all rungs. D2: seed cov = the repo-discoverable stock no-SVI diagonal
    (init_scales=1e-3 std => 1e-6*I), inits = MAP z_best + diagonal draws."""
    seed = ARM_SEED[arm]
    seed_source = "ARM_SEED default"
    env_seed = os.environ.get("GATE_PT0_SEED_B", "").strip()
    if env_seed:   # PT-0b 10-13; PT-1 20/21; PT-2 30-33 per checkpoints
        seed = int(env_seed)
        seed_source = f"env override GATE_PT0_SEED_B={env_seed}"
    path_kind = "power" if arm in ("B1", "B4", "B5", "D1", "D2") else "lik"
    adaptive = arm in ("D1", "D2")
    rng = np.random.default_rng(seed)

    # PT-0b/PT-1 env overrides (B arms ONLY; control/Arm A never read these).
    def _env_num(name, default, cast):
        v = os.environ.get(name, "").strip()
        if v:
            return cast(v), f"env override {name}={v}"
        return default, "default"

    k_b, k_src = _env_num("GATE_PT0_K_B", K_B, int)
    nsys, nsys_src = _env_num("GATE_PT0_NSYS_B", NSYS_B, int)
    rounds, rounds_src = _env_num("GATE_PT0_ROUNDS_B", ROUNDS_B, int)
    if SMOKE and rounds != ROUNDS_B:
        # smoke > env > default for ROUNDS too (2026-07-12 incident: an env
        # ROUNDS silently overrode the smoke reduction and ran full-length
        # mislabeled as smoke — second occurrence of the class; precedence is
        # now uniform and the ignored env value is recorded loudly)
        rounds_src = (f"SMOKE override {ROUNDS_B} [env GATE_PT0_ROUNDS_B "
                      f"IGNORED; precedence smoke > env > default]")
        rounds = ROUNDS_B
    ss_max, ssmax_src = _env_num("GATE_PT0_SSMAX", SS_MAX0, float)
    devar, devar_src = _env_num("GATE_PT0_DEVAR", DEVAR, float)

    # PT-3: metric window/freeze schedule for ADAPTIVE arms only.
    # Precedence (explicit): SMOKE > env GATE_PT0_METRIC_WINDOWS > default --
    # the smoke schedule exists purely to exercise the window/freeze/save code
    # paths at ROUNDS=20 and must win; which source applied is recorded.
    env_mw = os.environ.get("GATE_PT0_METRIC_WINDOWS", "").strip()
    if not adaptive:
        metric_windows, mw_src = None, "n/a (non-adaptive arm)"
    elif SMOKE:
        metric_windows = tuple(METRIC_WINDOWS)   # apply_smoke's (5, 10, 15)
        mw_src = (f"SMOKE override {metric_windows} "
                  + (f"[env GATE_PT0_METRIC_WINDOWS={env_mw} IGNORED; "
                     f"precedence smoke > env > default]" if env_mw
                     else "[precedence smoke > env > default]"))
    elif env_mw:
        metric_windows = tuple(int(x) for x in env_mw.split(","))
        assert len(metric_windows) >= 1 and all(
            w > 0 for w in metric_windows), \
            "GATE_PT0_METRIC_WINDOWS entries must be positive ints"
        assert all(b > a for a, b in
                   zip(metric_windows, metric_windows[1:])), \
            "GATE_PT0_METRIC_WINDOWS must be strictly increasing"
        mw_src = (f"env override GATE_PT0_METRIC_WINDOWS={env_mw} "
                  f"(freeze = last entry, round {metric_windows[-1]})")
    else:
        metric_windows = tuple(METRIC_WINDOWS)
        mw_src = f"default {metric_windows}"
    if adaptive and metric_windows[-1] >= rounds:
        pr(f"[B {tag}] WARNING: freeze boundary {metric_windows[-1]} >= "
           f"ROUNDS {rounds} -- the metric will NEVER freeze in this run")

    # PT-4: window-covariance estimator. NOT smoke-overridden (smoke may set
    # it freely to exercise the within path); pooled = bitwise status quo.
    env_est = os.environ.get("GATE_PT0_METRIC_EST", "").strip()
    metric_est = env_est or "pooled"
    assert metric_est in ("pooled", "within"), \
        f"GATE_PT0_METRIC_EST must be 'pooled' or 'within', got {metric_est!r}"
    est_src = (f"env override GATE_PT0_METRIC_EST={env_est}" if env_est
               else "default")

    # metric: pooled MAMS64 cov (B1-B4), production SVI cov (B5, D1 seed), or
    # the stock no-SVI diagonal (D2 seed); D arms ADAPT from their seed.
    d2_diag_provenance = None
    if arm in ("B5", "D1"):
        svi = np.load(SVI_NPZ)
        svi_loc = np.asarray(svi["qz_loc"], dtype=np.float64).reshape(-1)
        svi_tril = np.asarray(svi["qz_scale_tril"],
                              dtype=np.float64).reshape(DIM, DIM)
        assert svi_loc.shape == (DIM,), svi_loc.shape
        inv_mass = svi_tril @ svi_tril.T
        metric_desc = ("ADAPTIVE (windowed); seed = SVI cov (dpie/svi)"
                       if arm == "D1" else "SVI cov (dpie/svi)")
    elif arm == "D2":
        map_npz = np.load(MAP_NPZ)
        map_zbest = np.asarray(map_npz["z_best"],
                               dtype=np.float64).reshape(-1)
        assert map_zbest.shape == (DIM,), map_zbest.shape
        inv_mass = (D2_INIT_SCALE ** 2) * np.eye(DIM)
        d2_diag_provenance = (
            f"stock no-SVI diagonal convention: init_scales={D2_INIT_SCALE} "
            f"(STD in z) from ModellingSequence.SVI default "
            f"(gigalens/src/gigalens/jax/inference.py:254,282-284; mirrored "
            f"gigalens_research/inference_utils/pipeline.py:1440) => seed cov "
            f"= {D2_INIT_SCALE}^2 * I = {D2_INIT_SCALE**2:g}*I")
        metric_desc = (f"ADAPTIVE (windowed); seed = diagonal "
                       f"{D2_INIT_SCALE**2:g}*I (stock init_scales convention)")
    else:
        inv_mass = M["cov_pool"]
        metric_desc = "pooled MAMS64 empirical cov, all rungs (positions only)"
    # B-arm ladder override for the PT-0b continuation (mid-run finding: the
    # geometric ladder is disconnected on the likelihood path -- hottest-pair
    # swap acc 0.000, coldest 0.008-0.048 in B2 -- so continuation runs pass a
    # MEASURED ladder). Applies to B arms ONLY, never Arm A or the control.
    env_betas = os.environ.get("GATE_PT0_BETAS_B", "")
    if env_betas:
        betas = np.array([float(x) for x in env_betas.split(",")],
                         dtype=np.float64)   # rung 0 = hottest, R-1 = cold
        assert len(betas) >= 2, "GATE_PT0_BETAS_B needs >= 2 rungs"
        assert np.all(np.diff(betas) > 0), \
            "GATE_PT0_BETAS_B must be strictly increasing"
        assert np.all((betas > 0.0) & (betas <= 1.0)), \
            "GATE_PT0_BETAS_B values must lie in (0, 1]"
        assert betas[-1] == 1.0, "GATE_PT0_BETAS_B must end at exactly 1.0"
        betas_source = (f"env override GATE_PT0_BETAS_B "
                        f"({len(betas)} rungs: {env_betas})")
    else:
        betas = np.geomspace(0.01, 1.0, R_B)   # rung 0 = hottest, R-1 = cold
        betas_source = "geomspace default"
    R = len(betas)

    def draw_pool(pool, n):
        idx = rng.integers(0, len(pool), size=n)
        return pool[idx] + JITTER * rng.standard_normal((n, DIM))

    pos = np.zeros((R, nsys, DIM))
    for r in range(R):
        if arm in ("B5", "D1"):   # SVI draws at EVERY rung/system
            pos[r] = svi_loc + rng.standard_normal((nsys, DIM)) @ svi_tril.T
        elif arm == "D2":         # MAP z_best + stock-diagonal draws (PT-2 M2)
            pos[r] = (map_zbest
                      + D2_INIT_SCALE * rng.standard_normal((nsys, DIM)))
        elif arm in ("B3", "B4"):  # ALL-MAIN init (B4: power path, no prior)
            pos[r] = draw_pool(M["pool_M"], nsys)
        else:
            pick_p = rng.random(nsys) < 0.5
            zm, zp = draw_pool(M["pool_M"], nsys), draw_pool(M["pool_P"], nsys)
            pos[r] = np.where(pick_p[:, None], zp, zm)
    if arm in ("B2", "B3"):   # hot rung from PRIOR on the likelihood arms only
        pos[0] = prior_z_samples(nsys, seed + 777)

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

    card = dict(arm=tag, path=path_kind, seed=seed, seed_source=seed_source,
                swap_seed=seed + 10_000,
                prior_init_seed=seed + 777 if arm in ("B2", "B3") else None,
                smoke=SMOKE, script=os.path.abspath(__file__), jax=jax.__version__,
                devices=[str(d) for d in jax.devices()],
                x64=bool(jax.config.jax_enable_x64), dim=DIM,
                R=R, NSYS=nsys, K=k_b, ROUNDS=rounds, thin=THIN_B,
                betas=betas.tolist(), betas_source=betas_source,
                L=L_MCLMC, ss_init=SS_INIT, ss_max=ss_max, devar=devar,
                env_overrides=dict(K=k_src, NSYS=nsys_src, ROUNDS=rounds_src,
                                   ss_max=ssmax_src, devar=devar_src,
                                   metric_windows=mw_src),
                metric=metric_desc,
                adapt_metric=adaptive,
                metric_windows=(list(metric_windows) if adaptive else None),
                metric_windows_source=mw_src,
                metric_estimator=metric_est,
                metric_estimator_source=est_src,
                metric_n0=(10 * nsys if adaptive else None),
                metric_reference=("pooled MAMS64 cov (reference-ONLY)"
                                  if adaptive else None),
                metric_smoke_boundaries_note=(
                    "SMOKE boundaries scaled to exercise window/freeze/save "
                    "paths only" if (adaptive and SMOKE) else None),
                d2_diag_provenance=d2_diag_provenance,
                svi_npz=SVI_NPZ if arm in ("B5", "D1") else None,
                map_npz=MAP_NPZ if arm == "D2" else None,
                mams64=MAMS_NPZ, pocket_indicator=f"z[{POCKET_COL}] > {POCKET_THR}",
                init={"B1": "all rungs Bernoulli(0.5) main/pocket pool",
                      "B2": "rungs 1..R-1 Bernoulli(0.5); rung 0 prior draws",
                      "B3": "all main pool; rung 0 prior draws",
                      "B4": "ALL rungs/systems main pool (power; no prior draws)",
                      "B5": "SVI draws all rungs",
                      "D1": "SVI draws all rungs (adaptive metric)",
                      "D2": "MAP z_best + stock-diagonal draws all rungs "
                            "(adaptive metric)",
                      }[arm],
                u_def="log_prob = logdensity/beta" if path_kind == "power"
                      else "log_like = (logdensity - log_prior)/beta",
                harness="run_pt (shared with Arm-0 control, amendment ii)",
                sep_check_max_abs=M["sep_max"])
    spec = dict(dim=DIM, L=L_MCLMC, betas=betas, NSYS=nsys, K=k_b,
                ROUNDS=rounds, inv_mass=inv_mass, ss_max=ss_max, devar=devar,
                make_tf=lambda b: make_tempered(path_kind, b),
                u_from=u_from, u_direct=u_direct,
                indicator=lambda p: p[..., POCKET_COL] > POCKET_THR,
                init_pos=pos, card=card, ind_label="pocket",
                adapt_metric=adaptive,
                metric_windows=metric_windows if adaptive else None,
                metric_n0=(10 * nsys if adaptive else None),
                metric_estimator=metric_est,
                metric_ref=(M["cov_pool"] if adaptive else None))
    if equiv:
        return run_equiv_check(tag, seed, spec)
    return run_pt(tag, seed, spec)


def run_control(tag, equiv=False):
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
                L=math.sqrt(CTRL_D), ss_init=SS_INIT, ss_max=SS_MAX0, devar=DEVAR,
                metric="identity", indicator="z[0] > 0 (occ_+, +mode weight 0.70)",
                init="all rungs all systems in wrong (-5) basin (N(0,1) spread)",
                u_def="log_prob = logdensity/beta (power path)",
                harness="run_pt (SAME code path as B1/B2/B3)")
    spec = dict(dim=CTRL_D, L=math.sqrt(CTRL_D), betas=betas, NSYS=CTRL_NSYS,
                K=CTRL_K, ROUNDS=CTRL_ROUNDS, inv_mass=np.eye(CTRL_D),
                # b bound as lambda parameter; works for float AND traced beta
                make_tf=lambda b: (lambda z: b * mix_logpdf(z)),
                u_from=lambda logd, p: logd / betas[:, None],
                u_direct=lambda pf: np.asarray(mix_v(jnp.asarray(pf))),
                indicator=lambda p: p[..., 0] > 0.0,
                init_pos=init, card=card, ind_label="plus-mode",
                occ_truth=CTRL_OCC_TRUTH)
    if equiv:
        return run_equiv_check(tag, seed, spec)
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


# ---------------------------------------------- GATE PT-5a ST: numpy helpers
# Pure-numpy ST numeric core (no jax use in this block): unit-tested by
# --selftest-st and login-node testable by ast-extraction of these defs.
def st_tau_rounds(x):
    """Per-rung u IAT in ROUNDS by batch means over a (W, nch) round window:
    per chain, b = max(floor(sqrt(W)), 2), IAT = b * var(batch means, ddof=1)
    / var(series, ddof=1) (measure_sd_u convention, series variance over the
    full window), chain-AVERAGED, then floored at 1 (checkpoint: 'per chain,
    then averaged; floor tau_round at 1')."""
    x = np.asarray(x, np.float64)
    W, nch = x.shape
    b = max(int(math.sqrt(W)), 2)
    nbat = W // b
    if nbat < 2:
        raise ValueError(f"st_tau_rounds: window W={W} too short for batch means")
    s = x.T                                              # (nch, W)
    bm = s[:, :nbat * b].reshape(nch, nbat, b).mean(axis=2)
    iat = b * bm.var(axis=1, ddof=1) / np.maximum(s.var(axis=1, ddof=1), 1e-300)
    return max(float(iat.mean()), 1.0)


def st_trigger_test(u_win):
    """Split-halves u-stationarity gate on one evaluation window (derived
    trigger, checkpoint 'TRIGGER'): u_win (W, R, nch), W even. Per rung, over
    the POOLED per-rung u sample: |mean(half2) - mean(half1)| <= 2*se with
    se = sd(u_window) * sqrt(1/N_eff,1 + 1/N_eff,2), per-half N_eff =
    nch * (W/2) / tau_round, tau_round from st_tau_rounds on the full window;
    sd(u_window) = pooled np.std (ddof 0) over the full window. ALL rungs must
    pass simultaneously. Returns (all_pass, diagnostics dict)."""
    u_win = np.asarray(u_win, np.float64)
    W, R, nch = u_win.shape
    h = W // 2
    ok = np.zeros(R, dtype=bool)
    dmean, thr2se, tau = np.zeros(R), np.zeros(R), np.zeros(R)
    for r in range(R):
        x = u_win[:, r, :]
        tau[r] = st_tau_rounds(x)
        neff_half = nch * h / tau[r]
        se = float(np.std(x)) * math.sqrt(2.0 / neff_half)
        dmean[r] = abs(float(x[h:2 * h].mean()) - float(x[:h].mean()))
        thr2se[r] = 2.0 * se
        ok[r] = dmean[r] <= thr2se[r]
    return bool(ok.all()), dict(ok=ok, dmean=dmean, thr2se=thr2se, tau=tau)


def st_flip_counts(ind, boundary, r_star):
    """Basin-class flip counts over rounds (boundary, r_star] ONLY
    (post-boundary counting, B8). ind[(i, r, c)] = pocket indicator at the END
    of round i+1; the flip event at round n compares rounds n and n-1, counted
    for n = boundary+1 .. r_star -> exactly r_star - boundary comparisons (the
    rd-3 pin-1 exposure token (R*-100) rounds). Returns (counts (R,) int64,
    flips (r_star-boundary, R, nch) uint8)."""
    if boundary < 1 or r_star <= boundary:
        raise ValueError(f"st_flip_counts: need 1 <= boundary < r_star, got "
                         f"boundary={boundary}, r_star={r_star}")
    ind = np.asarray(ind[:r_star], bool)
    flips = ind[boundary:r_star] != ind[boundary - 1:r_star - 1]
    return flips.sum(axis=(0, 2)).astype(np.int64), flips.astype(np.uint8)


def st_sd_u_window(u, r_star, meas_win):
    """Per-rung sd(u) over rounds (r_star - meas_win, r_star]: POOLED across
    chains and rounds, np.std ddof-0, NO per-chain mean removal (recipe
    convention, ladder_recipe.measure_sd_u lineage); se(sd) = sd/sqrt(2*N_eff)
    with N_eff = nch * meas_win / tau_round (checkpoint B4 pin: the in-script
    computation). Returns (sd, se, tau_rounds), each (R,)."""
    w = np.asarray(u[r_star - meas_win:r_star], np.float64)
    if w.shape[0] != meas_win:
        raise ValueError(f"st_sd_u_window: u has {w.shape[0]} rounds in the "
                         f"window, expected {meas_win}")
    R, nch = w.shape[1], w.shape[2]
    sd = np.array([float(np.std(w[:, r, :])) for r in range(R)])
    tau = np.array([st_tau_rounds(w[:, r, :]) for r in range(R)])
    neff = nch * meas_win / tau
    se = sd / np.sqrt(2.0 * neff)
    return sd, se, tau


def st_carryover_map(old_betas, new_betas):
    """A5 pinned carry-over rule: each new rung takes the OLD rung nearest in
    |log beta| (general rule; on the certified knots this reproduces the
    pinned map {0.3594->0.3594, 0.4388->0.3594, 0.5373->0.5995,
    0.6598->0.5995, 0.8116->1.0, 1.0->1.0}). Returns old-rung indices (R1,)."""
    lb_old = np.log(np.asarray(old_betas, np.float64))
    return np.array([int(np.argmin(np.abs(math.log(float(b)) - lb_old)))
                     for b in np.asarray(new_betas, np.float64)], dtype=np.int64)


def st_interp_metrics(old_betas, old_mets, new_betas):
    """Phase-1 metric seeds: log-beta-linear ELEMENTWISE interpolation between
    the two bracketing phase-0 frozen metrics (a convex combination of PSD
    matrices, hence PSD); an exact-match beta (rtol 1e-12 -- beta_min_rule
    returns exact grid floats, so the first and last knots match exactly) uses
    that rung's metric directly. Extrapolation outside the grid raises.
    Returns (mets (R1, d, d), brackets (R1, 2) int, weights (R1,))."""
    ob = np.asarray(old_betas, np.float64)
    old_mets = np.asarray(old_mets, np.float64)
    nb = np.asarray(new_betas, np.float64)
    lb = np.log(ob)
    d = old_mets.shape[-1]
    mets = np.empty((len(nb), d, d))
    brackets = np.zeros((len(nb), 2), dtype=np.int64)
    weights = np.zeros(len(nb))
    for j, b in enumerate(nb):
        exact = np.where(np.isclose(ob, b, rtol=1e-12, atol=0.0))[0]
        if len(exact):
            i = int(exact[0])
            mets[j], brackets[j] = old_mets[i], (i, i)
            continue
        hi = int(np.searchsorted(lb, math.log(float(b))))
        if hi <= 0 or hi >= len(lb):
            raise ValueError(f"st_interp_metrics: beta {b} outside the "
                             f"phase-0 grid [{ob[0]}, {ob[-1]}]")
        lo = hi - 1
        w = (math.log(float(b)) - lb[lo]) / (lb[hi] - lb[lo])
        mets[j] = (1.0 - w) * old_mets[lo] + w * old_mets[hi]
        brackets[j], weights[j] = (lo, hi), w
    return mets, brackets, weights


def st_interp_geneig(mets, brackets, old_mets):
    """A1 offline interpolation-quality check (F-H discriminating pair, leg a):
    gen-eig extremes of each interpolated seed metric vs EACH of its two
    bracketing phase-0 metrics. Returns (R1, 2, 2) = [rung, bracket lo/hi,
    (min, max)]; exact-match rungs give identically 1."""
    out = np.zeros((len(mets), 2, 2))
    for j in range(len(mets)):
        for s in range(2):
            ch = np.linalg.cholesky(old_mets[brackets[j, s]])
            ev = geneig_ratios(mets[j], ch)
            out[j, s] = (float(ev.min()), float(ev.max()))
    return out


def st_selftest():
    """--selftest-st: numpy-only unit checks of the ST numeric helpers on
    synthetic data (batch-means/trigger, flip counting, sd-window convention,
    beta_min count guard, carry-over map, metric interpolation + A1 gen-eig).
    Raises AssertionError on failure; no jax use, no model build."""
    from ladder_recipe import beta_min_rule
    rng = np.random.default_rng(0)
    # 1) batch-means IAT: white noise ~ 1 (floored); AR(1) rho=0.9 within 2x
    t_white = st_tau_rounds(rng.standard_normal((100, 16)))
    assert 1.0 <= t_white < 1.6, t_white
    rho, n = 0.9, 4000
    e = rng.standard_normal((n, 8))
    y = np.empty_like(e)
    y[0] = e[0]
    for i in range(1, n):
        y[i] = rho * y[i - 1] + math.sqrt(1.0 - rho ** 2) * e[i]
    t_ar, t_true = st_tau_rounds(y), (1 + rho) / (1 - rho)
    assert t_true / 2.0 < t_ar < t_true * 2.0, (t_ar, t_true)
    pr(f"[selftest] tau: white={t_white:.2f} (in [1, 1.6)); AR1(0.9)={t_ar:.1f}"
       f" vs true {t_true:.1f} (within 2x) PASS")
    # 2) trigger: stationary window passes; one drifting rung blocks it.
    # A half-mean-matched palindrome (base + reversed base) makes the two
    # split-halves means IDENTICAL per rung (dmean == 0 << 2se), so all 10
    # rungs pass DETERMINISTICALLY. A raw random draw is fragile: dmean/2se ~
    # |Z|/2, so P(one rung within 2se) = 0.9545 independent of N and the all-10
    # conjunction is only ~0.63 (fails ~37% incl. at seed 0) — audit B1 fix.
    base = rng.standard_normal((50, 10, 16))
    u = np.concatenate([base, base[::-1]])
    ok, diag = st_trigger_test(u)
    assert ok, (diag["dmean"] / np.maximum(diag["thr2se"], 1e-300)).round(2)
    u2 = u.copy()
    u2[:, 3, :] += np.linspace(0.0, 3.0, 100)[:, None]   # 3-sd descent on rung 3
    ok2, diag2 = st_trigger_test(u2)
    assert (not ok2) and (not diag2["ok"][3]) and diag2["ok"].sum() == 9, diag2
    pr(f"[selftest] trigger: stationary all-pass; drifting rung 3 blocks "
       f"(|d|={diag2['dmean'][3]:.2f} > 2se={diag2['thr2se'][3]:.2f}) PASS")
    # 3) flip counting: post-boundary only, r_star - boundary comparisons
    ind = np.zeros((12, 2, 3), dtype=np.uint8)
    ind[5, 0, 1] = 1        # flips at rounds 6 and 7 (rung 0)
    ind[1:3, 1, 0] = 1      # flips at rounds 2 and 4 (rung 1, pre-boundary)
    counts, flips = st_flip_counts(ind, 4, 10)
    assert flips.shape == (6, 2, 3), flips.shape
    assert counts.tolist() == [2, 0], counts
    pr(f"[selftest] flips: counts={counts.tolist()} over (4, 10] "
       f"({flips.shape[0]} comparisons); pre-boundary flips excluded PASS")
    # 4) sd-window convention: pooled ddof-0, NO per-chain mean removal
    u3 = np.zeros((20, 1, 4))
    u3[:, 0, :] = np.array([0.0, 1.0, 2.0, 3.0])[None, :]  # chain offsets only
    sd, se, tau = st_sd_u_window(u3, 20, 10)
    assert abs(sd[0] - math.sqrt(1.25)) < 1e-12, sd   # demeaning would give 0
    pr(f"[selftest] sd window: sd={sd[0]:.6f} = sqrt(1.25) (pooled, no "
       f"per-chain mean removal) PASS")
    # 5) beta_min B2' count guard + counts=None bitwise default
    betas10 = np.geomspace(0.01, 1.0, 10)
    leak = np.array([.9, .8, .7, .6, .5, .45, .35, .26, .18, .008])
    b0 = beta_min_rule(betas10, leak, ST_RULE_NSYS, ST_RULE_BUDGET, ST_RULE_PROBE)
    assert round(b0, 4) == 0.3594, b0
    counts = np.array([50, 50, 50, 50, 50, 50, 40, 30, 1, 0])
    b1 = beta_min_rule(betas10, leak, ST_RULE_NSYS, ST_RULE_BUDGET,
                       ST_RULE_PROBE, counts=counts, min_count=ST_MIN_COUNT)
    assert round(b1, 4) == 0.2154, b1   # k=1 at 0.5995 blocks 0.3594 (false-cold)
    counts2 = counts.copy()
    counts2[8] = 2
    b2 = beta_min_rule(betas10, leak, ST_RULE_NSYS, ST_RULE_BUDGET,
                       ST_RULE_PROBE, counts=counts2, min_count=ST_MIN_COUNT)
    assert round(b2, 4) == 0.3594, b2   # k=2 re-admits it
    assert beta_min_rule(betas10, leak, ST_RULE_NSYS, ST_RULE_BUDGET,
                         ST_RULE_PROBE, counts=None) == b0
    pr(f"[selftest] beta_min guard: no-counts {b0:.4f}; k=1@0.5995 -> {b1:.4f}"
       f" (blocked colder); k=2 -> {b2:.4f}; counts=None bitwise PASS")
    # 6) carry-over: A5 pinned map on the certified knots
    knots = np.array([0.3594, 0.4388, 0.5373, 0.6598, 0.8116, 1.0])
    cmap = st_carryover_map(betas10, knots)
    assert cmap.tolist() == [7, 7, 8, 8, 9, 9], cmap
    pr(f"[selftest] carry-over map on certified knots = {cmap.tolist()} "
       f"(A5 pinned {{0.3594,0.4388->0.3594; 0.5373,0.6598->0.5995; "
       f"0.8116,1.0->1.0}}) PASS")
    # 7) interpolation: exact match, log-midpoint convexity, PSD, A1 gen-eig
    dm = 4
    A = rng.standard_normal((dm, dm))
    A = A @ A.T + dm * np.eye(dm)
    B = rng.standard_normal((dm, dm))
    B = B @ B.T + dm * np.eye(dm)
    ob, om = np.array([0.25, 1.0]), np.stack([A, B])
    mets, br, wt = st_interp_metrics(ob, om, np.array([0.25, 0.5, 1.0]))
    assert np.array_equal(mets[0], A) and np.array_equal(mets[2], B)
    assert abs(wt[1] - 0.5) < 1e-12 and np.allclose(
        mets[1], 0.5 * (A + B), rtol=0, atol=1e-12)   # log-beta midpoint
    assert np.linalg.eigvalsh(mets[1]).min() > 0.0    # convex comb of PSD
    ge = st_interp_geneig(mets, br, om)
    assert ge.shape == (3, 2, 2) and np.allclose(ge[0], 1.0) \
        and np.allclose(ge[2], 1.0), ge
    assert ge[1, 0, 0] <= 1.0 <= ge[1, 0, 1], ge[1]
    pr(f"[selftest] interp: exact-match endpoints; log-midpoint = (A+B)/2; "
       f"PSD; A1 gen-eig vs lo bracket [{ge[1, 0, 0]:.3f}, {ge[1, 0, 1]:.3f}]"
       f" straddles 1 PASS")
    pr("[selftest] ALL ST helper checks PASS")


# ------------------------------------------------- GATE PT-5a ST: runner mode
def _st_env_num(name, default, cast):
    """ST copy of the run_arm_b env-override pattern (that one is nested)."""
    v = os.environ.get(name, "").strip()
    if v:
        return cast(v), f"env override {name}={v}"
    return default, "default"


def run_arm_st(tag):
    """GATE PT-5a two-phase self-tuning runner (checkpoint 'carousel GATE
    PT-5a', grader rd-3 CERTIFY-RECOMMENDED). Phase 0 = swaps-OFF measurement
    leg (B1 pin: kernel-only leakage semantics) on the pinned arm-A probe grid,
    MAP + 1e-6*I entry exactly as D2, single scale-finding metric boundary at
    round 100 then FROZEN; per-rung split-halves u-stationarity trigger (floor
    200, deadline 400 -> F-never exit); RE-SPACE at R* = max(trigger, 300)
    through ladder_recipe (amended B2' beta_min rule); handoff npz saved.
    Phase 1 = run_pt production leg on the re-spaced ladder: positions carried
    per new rung from the nearest-log-beta old rung (A5), per-rung metric
    seeded by log-beta-linear interpolation of the phase-0 frozen metrics
    (A1 gen-eig quality check printed BEFORE the leg runs), EEVPD/ss reset,
    swaps ON, windows 100/250/500 freeze-500, ROUNDS 1000.
    GATE_PT0_ST_PHASE: auto (default) = phase 0 then, iff trigger <= the R*
    floor, phase 1 in-process (trigger above the floor -> the pre-committed
    B3 split-allocation contingency: save + exit, relaunch phase 1);
    0 = phase 0 only; 1 = phase 1 standalone from the saved handoff."""
    if LEGACY_RUNNERS:
        raise RuntimeError("arm ST requires the fused runner; unset "
                           "GATE_PT0_LEGACY_RUNNERS")
    for bad in ("GATE_PT0_BETAS_B", "GATE_PT0_METRIC_WINDOWS"):
        if os.environ.get(bad, "").strip():
            raise RuntimeError(
                f"{bad} is set, but arm ST pins its grid/windows (phase-0 "
                f"grid = arm-A probe grid, windows ({ST_WINDOW0},) phase 0 / "
                f"{tuple(METRIC_WINDOWS)} phase 1); unset it "
                f"(launch-discipline: NO conflicting env)")
    seed = ARM_SEED["ST"]
    seed_source = "ARM_SEED default"
    env_seed = os.environ.get("GATE_PT0_SEED_B", "").strip()
    if env_seed:   # PT-5a production arms: seeds 55/56/57 per checkpoint
        seed = int(env_seed)
        seed_source = f"env override GATE_PT0_SEED_B={env_seed}"
    nsys, nsys_src = _st_env_num("GATE_PT0_NSYS_B", ST_NSYS, int)
    k_b, k_src = _st_env_num("GATE_PT0_K_B", K_B, int)
    ss_max, ssmax_src = _st_env_num("GATE_PT0_SSMAX", SS_MAX0, float)
    devar, devar_src = _st_env_num("GATE_PT0_DEVAR", DEVAR, float)
    rounds1, rounds1_src = _st_env_num("GATE_PT0_ROUNDS_B", ST_ROUNDS1, int)
    if SMOKE and rounds1 != ST_ROUNDS1:
        # smoke > env > default (standing precedence; mirrors run_arm_b ROUNDS)
        rounds1_src = (f"SMOKE override {ST_ROUNDS1} [env GATE_PT0_ROUNDS_B "
                       f"IGNORED; precedence smoke > env > default]")
        rounds1 = ST_ROUNDS1
    env_est = os.environ.get("GATE_PT0_METRIC_EST", "").strip()
    # ST arm defaults to WITHIN (the PT-5a gate's pinned config + the scorer's
    # asserted estimator), NOT pooled — a launch that omits the env still runs
    # the correct estimator instead of failing the scorer after 3.4 h (audit A1)
    metric_est = env_est or "within"
    assert metric_est in ("pooled", "within"), \
        f"GATE_PT0_METRIC_EST must be 'pooled' or 'within', got {metric_est!r}"
    est_src = (f"env override GATE_PT0_METRIC_EST={env_est}" if env_est
               else "default")
    phase = os.environ.get("GATE_PT0_ST_PHASE", "").strip() or "auto"
    if phase not in ("auto", "0", "1"):
        raise RuntimeError(
            f"GATE_PT0_ST_PHASE must be auto/0/1, got {phase!r}")
    kn = dict(seed=seed, seed_source=seed_source, nsys=nsys, nsys_src=nsys_src,
              k_b=k_b, k_src=k_src, ss_max=ss_max, ssmax_src=ssmax_src,
              devar=devar, devar_src=devar_src, rounds1=rounds1,
              rounds1_src=rounds1_src, metric_est=metric_est, est_src=est_src,
              phase=phase)
    handoff_path = os.path.join(OUT, f"handoff_{tag}.npz")
    if phase in ("auto", "0"):
        trigger = run_st_phase0(tag, handoff_path, kn)
        if trigger is None:
            return                          # F-never (A4 explicit exit path)
        if phase == "0":
            pr(f"[ST {tag}] GATE_PT0_ST_PHASE=0: phase 0 complete, handoff "
               f"saved -> {handoff_path}; run phase 1 with GATE_PT0_ST_PHASE=1")
            return
        if trigger > ST_RSTAR_FLOOR:
            pr(f"[ST {tag}] PRE-COMMITTED SPLIT-ALLOCATION CONTINGENCY (B3): "
               f"trigger {trigger} > {ST_RSTAR_FLOOR} -> phase 1 NOT run in "
               f"this allocation; relaunch with GATE_PT0_ST_PHASE=1 "
               f"(handoff: {handoff_path})")
            return
    run_st_phase1(tag, handoff_path, kn)


def run_st_phase0(tag, handoff_path, kn):
    """PT-5a phase 0: swaps-OFF measurement leg. Returns the trigger round, or
    None on F-never (phase-0 arrays saved either way)."""
    betas = np.geomspace(0.01, 1.0, ST_GRID_N)   # arm-A probe grid (PINNED)
    R = len(betas)
    st_chunk, st_chunk_src = _st_env_num("GATE_PT0_ST_CHUNK", ST_PHASE0_CHUNK, int)
    seed, nsys, k_b = kn["seed"], kn["nsys"], kn["k_b"]
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)
    map_npz = np.load(MAP_NPZ)
    map_zbest = np.asarray(map_npz["z_best"], dtype=np.float64).reshape(-1)
    assert map_zbest.shape == (DIM,), map_zbest.shape
    inv_mass = (D2_INIT_SCALE ** 2) * np.eye(DIM)
    d2_diag_provenance = (   # same provenance pin as D2 (run_arm_b)
        f"stock no-SVI diagonal convention: init_scales={D2_INIT_SCALE} "
        f"(STD in z) from ModellingSequence.SVI default "
        f"(gigalens/src/gigalens/jax/inference.py:254,282-284; mirrored "
        f"gigalens_research/inference_utils/pipeline.py:1440) => seed cov "
        f"= {D2_INIT_SCALE}^2 * I = {D2_INIT_SCALE**2:g}*I")
    pos = np.zeros((R, nsys, DIM))
    for r in range(R):   # MAP z_best + stock-diagonal draws (D2 entry, PT-2 M2)
        pos[r] = map_zbest + D2_INIT_SCALE * rng.standard_normal((nsys, DIM))
    windows0 = (ST_WINDOW0,)
    trig_cfg = dict(floor=ST_TRIG_FLOOR, every=ST_TRIG_EVERY,
                    window=ST_TRIG_WIN, deadline=ST_ROUNDS0_MAX,
                    rstar_rule=f"max(trigger, {ST_RSTAR_FLOOR})",
                    rstar_floor=ST_RSTAR_FLOOR, meas_window=ST_MEAS_WIN,
                    flip_boundary=ST_WINDOW0)
    b2prime = dict(min_count=ST_MIN_COUNT,
                   threshold_rate=math.log(100.0)
                   / (ST_RULE_NSYS * (ST_RULE_BUDGET / ST_RULE_PROBE)),
                   rule_nsys=ST_RULE_NSYS, rule_budget_steps=ST_RULE_BUDGET,
                   rule_probe_steps=ST_RULE_PROBE,
                   leak_norm_per_steps=ST_LEAK_UNIT)
    card = dict(arm=tag, gate="PT-5a", st_phase=0, path="power", seed=seed,
                seed_source=kn["seed_source"], smoke=SMOKE,
                script=os.path.abspath(__file__), jax=jax.__version__,
                devices=[str(d) for d in jax.devices()],
                x64=bool(jax.config.jax_enable_x64), dim=DIM,
                R=R, NSYS=nsys, K=k_b, rounds_deadline=ST_ROUNDS0_MAX,
                betas=betas.tolist(),
                betas_source="arm-A probe grid geomspace(0.01, 1, 10) (PINNED)",
                L=L_MCLMC, ss_init=SS_INIT, ss_max=kn["ss_max"],
                devar=kn["devar"],
                env_overrides=dict(NSYS=kn["nsys_src"], K=kn["k_src"],
                                   ss_max=kn["ssmax_src"],
                                   devar=kn["devar_src"]),
                swaps_off_phase0=True,
                phases_run=("0 only (GATE_PT0_ST_PHASE=0)" if kn["phase"] == "0"
                            else "0 (+1 in-process iff trigger <= R* floor)"),
                metric=("ADAPTIVE (single scale-finding window, then FROZEN); "
                        "seed = diagonal 1e-6*I (stock init_scales convention)"),
                adapt_metric=True, metric_windows=list(windows0),
                metric_estimator=kn["metric_est"],
                metric_estimator_source=kn["est_src"], metric_n0=10 * nsys,
                d2_diag_provenance=d2_diag_provenance, map_npz=MAP_NPZ,
                pocket_indicator=f"z[{POCKET_COL}] > {POCKET_THR}",
                init="MAP z_best + stock-diagonal draws all rungs (as D2)",
                u_def=("log_prob = logdensity/beta (power path; the u the "
                       "swap logic would use -- swaps OFF, B1 pin; recorded "
                       "per (round, rung, chain), float64)"),
                trigger_config=trig_cfg, b2prime=b2prime,
                leak_direction=("M->P-dominated measured rate as-is (chains "
                                "start all-main; the probe's conservative "
                                "direction)"),
                harness=("ST phase-0 fused kernel rounds, NO swap attempts "
                         "(swap arrays saved present-and-all-zero)"),
                sep_check_max_abs=M["sep_max"])
    pr("MODEL CARD:", json.dumps(card, indent=1, default=_json_default))
    summary = dict(model_card=card)
    t0_all = time.time()

    spec_kernel = dict(dim=DIM, L=L_MCLMC, K=k_b, devar=kn["devar"],
                       make_tf=lambda b: make_tempered("power", b))
    round_all = make_fused_round_runner(spec_kernel)
    betas_j = jnp.asarray(betas)
    inv_rungs = np.broadcast_to(inv_mass, (R, DIM, DIM)).copy()
    inv_rungs_j = jnp.asarray(inv_rungs)
    steps_all = jnp.full((R, nsys), SS_INIT)
    adapt_all = fresh_adapt2(R, nsys, kn["ss_max"])
    n0 = float(10 * nsys)
    # Welford accumulators over ROUND-END positions (run_pt lineage; within/
    # between decomposition recorded at the boundary in BOTH estimator modes)
    w_n = 0
    w_mean = np.zeros((R, DIM))
    w_M2 = np.zeros((R, DIM, DIM))
    s_C = np.zeros((R, DIM, DIM))
    bm_n = 0
    bm_mean = np.zeros((R, DIM))
    bm_M2 = np.zeros((R, DIM, DIM))
    metric_frozen_flag = False
    boundary_covs, within_covs, between_covs = [], [], []
    u_all = np.zeros((ST_ROUNDS0_MAX, R, nsys))          # float64
    ind_all = np.zeros((ST_ROUNDS0_MAX, R, nsys), dtype=np.uint8)
    ev = np.zeros((ST_ROUNDS0_MAX, R))
    ssm = np.zeros((ST_ROUNDS0_MAX, R))
    n_revert = 0
    u0_rel = None
    trig_rounds, trig_pass, trig_dmean, trig_thr, trig_tau = [], [], [], [], []
    trigger, forced, r_star = None, False, None
    ph0_npz = os.path.join(OUT, f"arrays_{tag}_phase0.npz")

    def save_phase0(t):
        np.savez(
            ph0_npz, betas=betas, u=u_all[:t + 1], ind=ind_all[:t + 1],
            eevpd=ev[:t + 1], step_mean=ssm[:t + 1],
            # swaps OFF (B1): swap arrays PRESENT AND ALL-ZERO (documented
            # choice -- keeps pt4-class loaders working), plus explicit flag
            swap_attempts=np.zeros((R - 1, 2), dtype=np.int64),
            swap_accepts=np.zeros((R - 1, 2), dtype=np.int64),
            swaps_off=np.bool_(True),
            metric_frozen=inv_rungs,
            metric_boundary_covs=(np.stack(boundary_covs)
                                  if boundary_covs else np.zeros((0,))),
            metric_within_covs=(np.stack(within_covs)
                                if within_covs else np.zeros((0,))),
            metric_between_covs=(np.stack(between_covs)
                                 if between_covs else np.zeros((0,))),
            metric_windows=np.asarray(windows0, dtype=np.int64),
            metric_n0=n0, metric_estimator=np.array(kn["metric_est"]),
            n_revert=np.int64(n_revert), rounds_done=np.int64(t + 1),
            trigger=np.int64(-1 if trigger is None else trigger),
            trigger_forced=np.bool_(forced),
            r_star=np.int64(-1 if r_star is None else r_star),
            trig_eval_rounds=np.asarray(trig_rounds, dtype=np.int64),
            trig_eval_pass=(np.stack(trig_pass) if trig_pass
                            else np.zeros((0, R), dtype=np.uint8)),
            trig_eval_dmean=(np.stack(trig_dmean) if trig_dmean
                             else np.zeros((0, R))),
            trig_eval_thr2se=(np.stack(trig_thr) if trig_thr
                              else np.zeros((0, R))),
            trig_eval_tau=(np.stack(trig_tau) if trig_tau
                           else np.zeros((0, R))),
            # model-card fields (npz-embedded copies; full card in the json)
            arm=np.array(tag), gate=np.array("PT-5a"),
            seed=np.int64(seed), nsys=np.int64(nsys), K=np.int64(k_b),
            ss_max=np.float64(kn["ss_max"]), devar=np.float64(kn["devar"]),
            smoke=np.bool_(SMOKE),
            trig_floor=np.int64(ST_TRIG_FLOOR),
            trig_every=np.int64(ST_TRIG_EVERY),
            trig_window=np.int64(ST_TRIG_WIN),
            deadline=np.int64(ST_ROUNDS0_MAX),
            rstar_floor=np.int64(ST_RSTAR_FLOOR),
            meas_window=np.int64(ST_MEAS_WIN),
            flip_boundary=np.int64(ST_WINDOW0),
            min_count=np.int64(ST_MIN_COUNT),
            rule_nsys=np.int64(ST_RULE_NSYS),
            rule_budget_steps=np.int64(ST_RULE_BUDGET),
            rule_probe_steps=np.int64(ST_RULE_PROBE))

    pr(f"[ST {tag}] phase 0: swaps OFF, {R}x{nsys} = {R * nsys} chains "
       f"(s/round below is the {R * nsys}-wide GO/NO-GO timing), K={k_b}, "
       f"trigger floor {ST_TRIG_FLOOR} every {ST_TRIG_EVERY}, deadline "
       f"{ST_ROUNDS0_MAX}, R* = max(trigger, {ST_RSTAR_FLOOR})")
    t0 = time.time()
    t_block, last_block_rd = t0, 0
    for t in range(ST_ROUNDS0_MAX):
        key, *lks = jax.random.split(key, R + 1)
        ik_all, sk_all = _round_keys(lks, R, nsys, k_b)
        p2, ld, steps_all, adapt_all, ec2, ok = round_all_chunked(
            round_all, jnp.asarray(pos), steps_all, adapt_all, ik_all,
            sk_all, betas_j, inv_rungs_j, st_chunk)
        if t == 0 and os.environ.get("GATE_PT0_ST_VERIFY_CHUNK") == "1" \
                and st_chunk < R:
            # one-time STRUCTURAL check on a rung-subset that fits unchunked
            # (<= 96-wide): chunked vs direct agree up to FP-reorder (NOT 0 ULP
            # — XLA vmap-width reduction-order; an O(1) diff = rung mis-slice)
            nv = min(6, R)
            d = round_all(jnp.asarray(pos)[:nv], steps_all[:nv],
                          (adapt_all[0][:nv], adapt_all[1][:nv],
                           adapt_all[2][:nv]), ik_all[:nv], sk_all[:nv],
                          betas_j[:nv], inv_rungs_j[:nv])
            cw = round_all_chunked(round_all, jnp.asarray(pos)[:nv],
                                   steps_all[:nv], (adapt_all[0][:nv],
                                   adapt_all[1][:nv], adapt_all[2][:nv]),
                                   ik_all[:nv], sk_all[:nv], betas_j[:nv],
                                   inv_rungs_j[:nv], 2)
            a, b = np.asarray(d[0]), np.asarray(cw[0])
            dp = float(np.max(np.abs(a - b)))
            drel = float(np.max(np.abs(a - b) / (1.0 + np.abs(a))))
            # STRUCTURAL check (not bitwise): chunking is a pure per-rung vmap
            # slice, so a mis-assigned rung would differ by O(1); XLA fuses
            # 2-wide vs 6-wide vmaps with different matmul reduction order, so
            # a valid chunking still differs at the FP-reorder level (the same
            # cross-compile phenomenon as the B-arm fused-vs-legacy equiv-check)
            # -> tolerate FP-reorder, catch mis-slicing (rel > 1e-3)
            pr(f"[ST {tag}] CHUNK-EQUIV (nv={nv}, chunk 2 vs direct): "
               f"max|dpos| = {dp:.3e}, rel = {drel:.3e} (FP-reorder expected; "
               f"rel > 1e-3 would mean a rung mis-assignment)")
            if drel > 1e-3:
                raise RuntimeError(f"chunk STRUCTURAL check FAILED: rel "
                                   f"{drel:.3e} > 1e-3 (rung mis-assignment?)")
        pos = np.array(p2)
        logd = np.asarray(ld)
        ev[t] = np.asarray(ec2).mean(axis=(1, 2)) / DIM
        ssm[t] = np.asarray(steps_all).mean(axis=1)
        n_revert += int((~np.asarray(ok)).sum())
        u = logd / betas[:, None]   # power-path u (what the swap logic uses)
        if t == 0:   # round-0 identity verification (run_pt convention)
            u_direct = M["lp_batch"](pos.reshape(-1, DIM)).reshape(R, nsys)
            u0_rel = float(np.max(np.abs(u - u_direct)
                                  / (1.0 + np.abs(u_direct))))
            pr(f"[ST {tag}] round-0 u identity: rel = {u0_rel:.3e}")
            if u0_rel > U_REL_TOL:
                raise RuntimeError(f"round-0 u verification failed: rel "
                                   f"{u0_rel:.3e} > {U_REL_TOL}")
        u_all[t] = u
        ind_all[t] = pos[:, :, POCKET_COL] > POCKET_THR   # round-end indicator
        # NO swap attempts (B1 pin): kernel rounds only in phase 0
        if not metric_frozen_flag:
            for r in range(R):
                batch = pos[r]                            # (nsys, dim)
                bmean = batch.mean(axis=0)
                bctr = batch - bmean
                bM2 = bctr.T @ bctr
                if w_n == 0:
                    w_mean[r], w_M2[r] = bmean, bM2
                else:
                    delta = bmean - w_mean[r]
                    tot = w_n + nsys
                    w_M2[r] += bM2 + np.outer(delta, delta) * w_n * nsys / tot
                    w_mean[r] += delta * nsys / tot
                s_C[r] += bM2 / (nsys - 1)
                delta_b = bmean - bm_mean[r]
                bm_mean[r] += delta_b / (bm_n + 1)
                bm_M2[r] += np.outer(delta_b, bmean - bm_mean[r])
            w_n += nsys
            bm_n += 1
            if (t + 1) in windows0:
                n_w = w_n
                T = bm_n
                if T < 2:
                    raise RuntimeError(
                        f"metric window boundary at round {t + 1}: only T={T} "
                        f"round(s) accumulated; needs T >= 2 (never default)")
                W_win = s_C / T
                B_win = bm_M2 / (T - 1)
                for r in range(R):
                    cov_w = (W_win[r] if kn["metric_est"] == "within"
                             else w_M2[r] / max(n_w - 1, 1))
                    comb = (n0 * inv_rungs[r] + n_w * cov_w) / (n0 + n_w)
                    inv_rungs[r] = regularize_cov_np(comb, n0 + n_w)
                boundary_covs.append(inv_rungs.copy())
                within_covs.append(W_win.copy())
                between_covs.append(B_win.copy())
                inv_rungs_j = jnp.asarray(inv_rungs)
                adapt_all = (jnp.zeros_like(adapt_all[0]),
                             jnp.zeros_like(adapt_all[1]), adapt_all[2])
                w_n, w_mean[:], w_M2[:] = 0, 0.0, 0.0
                bm_n, s_C[:], bm_mean[:], bm_M2[:] = 0, 0.0, 0.0, 0.0
                if (t + 1) == windows0[-1]:
                    metric_frozen_flag = True   # FROZEN (single-window path)
                pr(f"[ST {tag}] metric window boundary at round {t + 1}: "
                   f"n_w={n_w}, n0={n0:.0f}, est={kn['metric_est']}"
                   f"{' (METRIC FROZEN)' if metric_frozen_flag else ''}")
        rd = t + 1
        if (trigger is None and rd >= ST_TRIG_FLOOR
                and (rd - ST_TRIG_FLOOR) % ST_TRIG_EVERY == 0):
            ok_all, diag = st_trigger_test(u_all[rd - ST_TRIG_WIN:rd])
            trig_rounds.append(rd)
            trig_pass.append(diag["ok"].astype(np.uint8))
            trig_dmean.append(diag["dmean"])
            trig_thr.append(diag["thr2se"])
            trig_tau.append(diag["tau"])
            n_pass = int(diag["ok"].sum())
            pr(f"[ST {tag}] trigger eval @ round {rd}: {n_pass}/{R} rungs "
               f"pass (worst |d|/2se = "
               f"{float(np.max(diag['dmean'] / np.maximum(diag['thr2se'], 1e-300))):.2f})"
               f"{' -> TRIGGER FIRED' if ok_all else ''}")
            if ok_all:
                trigger = rd
            elif SMOKE and rd == ST_TRIG_FLOOR:
                trigger, forced = rd, True
                pr(f"[ST {tag}] SMOKE: trigger FORCED at round {rd} (natural "
                   f"test: {n_pass}/{R} rungs passed) -- forced-trigger smoke "
                   f"per checkpoint; NOT a measurement")
            if trigger is not None:
                r_star = max(trigger, ST_RSTAR_FLOOR)
                if SMOKE:
                    pr(f"[ST {tag}] SMOKE: exposure floor waived (R* floor "
                       f"{ST_RSTAR_FLOOR}; production 300)")
                pr(f"[ST {tag}] trigger={trigger} -> RE-SPACE at R* = "
                   f"max({trigger}, {ST_RSTAR_FLOOR}) = {r_star}")
        if (rd % PRINT_EVERY_B == 0 or rd == ST_ROUNDS0_MAX
                or (r_star is not None and rd == r_star)):
            spr = (time.time() - t_block) / max(rd - last_block_rd, 1)
            t_block, last_block_rd = time.time(), rd
            pr(f"[ST {tag}] phase0 round {rd:4d} "
               f"cold occ={ind_all[t, -1].mean():.3f} "
               f"hot occ={ind_all[t, 0].mean():.3f} "
               f"EEVPD[h/m/c]={ev[t, 0]:.1e}/{ev[t, R // 2]:.1e}/{ev[t, -1]:.1e} "
               f"reverts={n_revert} block {spr:.2f} s/round "
               f"({R * nsys}-wide) wall={time.time() - t0:.0f}s")
        if rd % SAVE_EVERY_B == 0:
            save_phase0(t)
        if r_star is not None and rd == r_star:
            break

    if trigger is None:   # F-never (A4): save, print loudly, EXIT -- no
        save_phase0(ST_ROUNDS0_MAX - 1)   # re-space, no phase 1
        summary["f_never"] = True
        summary["trigger"] = None
        summary["trig_eval_rounds"] = trig_rounds
        summary["trig_eval_pass_counts"] = [int(p.sum()) for p in trig_pass]
        summary["total_wall_s"] = time.time() - t0_all
        with open(os.path.join(OUT, f"summary_{tag}_phase0.json"), "w") as f:
            json.dump(summary, f, indent=1, default=_json_default)
        pr(f"\n[ST {tag}] " + "!" * 66)
        pr(f"[ST {tag}] F-NEVER: NO trigger by the round-{ST_ROUNDS0_MAX} "
           f"deadline -- u equilibrates slower than hypothesized (the APS-lag "
           f"caution materialized).")
        pr(f"[ST {tag}] Phase-0 arrays saved -> {ph0_npz}. NO re-space, NO "
           f"phase 1, nothing downstream attempted (A4 exit path). W-S is "
           f"still adjudicated on the phase-0 data by the scorer.")
        pr(f"[ST {tag}] " + "!" * 66)
        return None

    # ---------------- R* measurement: sd(u), leakage, re-spaced ladder ------
    if r_star - ST_MEAS_WIN < ST_WINDOW0:
        raise RuntimeError(
            f"sd window (R*-{ST_MEAS_WIN}, R*] crosses the metric boundary "
            f"{ST_WINDOW0} (R*={r_star}) -- config invalid (never default)")
    sd_u, sd_u_se, tau_rounds = st_sd_u_window(u_all, r_star, ST_MEAS_WIN)
    flip_counts, flips = st_flip_counts(ind_all, ST_WINDOW0, r_star)
    exposure_steps = (r_star - ST_WINDOW0) * k_b * nsys   # rd-3 pin 1 (SHARED
    exposure_per1500 = exposure_steps / ST_LEAK_UNIT      # runner/scorer token)
    flip_rates = flip_counts / exposure_per1500           # per 1500 kernel steps
    pr(f"[ST {tag}] R*={r_star} measurement (sd window ({r_star - ST_MEAS_WIN},"
       f" {r_star}], flips ({ST_WINDOW0}, {r_star}], exposure "
       f"{exposure_steps} steps = {exposure_per1500:.2f} per-1500 units):")
    for r in range(R):
        pr(f"  beta={betas[r]:8.4f}  sd(u)={sd_u[r]:12.3f} +- {sd_u_se[r]:8.3f}"
           f"  tau={tau_rounds[r]:6.2f} rounds  flips k={int(flip_counts[r]):4d}"
           f"  rate={flip_rates[r]:.4f}/1500")
    if flip_counts[0] == 0:
        msg = (f"F-flip: ZERO flips at the hottest rung (beta={betas[0]:.4f}) "
               f"post-boundary -- this posterior is known to cross (probe "
               f"class 0.24-0.66 per 1500 steps); in-run leakage counting "
               f"broken (code class)")
        if SMOKE:
            pr(f"[ST {tag}] SMOKE: F-flip zero-flip check WAIVED "
               f"(exposure {exposure_per1500:.2f} units is sub-unit in smoke)"
               f" -- production raises: {msg}")
        else:
            raise RuntimeError(msg)
    bmin_fallback = False
    try:
        beta_min = ladder_recipe.beta_min_rule(
            betas, flip_rates, ST_RULE_NSYS, ST_RULE_BUDGET, ST_RULE_PROBE,
            counts=flip_counts, min_count=ST_MIN_COUNT)
    except ValueError:
        if not SMOKE:
            raise
        beta_min = float(betas[7])   # 0.35938-class grid point
        bmin_fallback = True
        pr(f"[ST {tag}] SMOKE: beta_min_rule inadmissible on smoke-length "
           f"exposure -- FALLING BACK to beta_min = {beta_min:.6g} grid point "
           f"(smoke-only path exercise; production raises)")
    des = ladder_recipe.design_ladder(betas, sd_u, beta_min, ST_TARGET_NATS)
    knots = np.asarray(des["knots"], np.float64)
    kse = ladder_recipe.knot_se(betas, sd_u, sd_u_se, beta_min, ST_TARGET_NATS)
    pr(f"[ST {tag}] beta_min = {beta_min:.6g} (B2' rule, min_count "
       f"{ST_MIN_COUNT}{', SMOKE FALLBACK' if bmin_fallback else ''}); "
       f"re-spaced ladder: {des['n_rungs']} rungs, total "
       f"{des['total_cost']:.3f} nats, {des['nats_per_pair']:.4f} nats/pair")
    pr(f"[ST {tag}] knots = "
       + ", ".join(f"{k:.6g}(+-{s:.2e})" for k, s in zip(knots, kse)))
    carry = st_carryover_map(betas, knots)
    pr(f"[ST {tag}] carry-over map (A5 nearest-log-beta): "
       + ", ".join(f"{knots[j]:.4f}<-{betas[carry[j]]:.4f}"
                   for j in range(len(knots))))
    mets_interp, brackets, weights = st_interp_metrics(betas, inv_rungs, knots)
    interp_geneig = st_interp_geneig(mets_interp, brackets, inv_rungs)
    pr(f"[ST {tag}] A1 interpolation-quality check (gen-eig of each "
       f"interpolated seed metric vs its two bracketing phase-0 metrics):")
    for j in range(len(knots)):
        pr(f"  rung {j} beta={knots[j]:.4f} "
           f"(brackets {betas[brackets[j, 0]]:.4f}/{betas[brackets[j, 1]]:.4f},"
           f" w={weights[j]:.3f}): vs lo [{interp_geneig[j, 0, 0]:.3f}, "
           f"{interp_geneig[j, 0, 1]:.3f}], vs hi [{interp_geneig[j, 1, 0]:.3f},"
           f" {interp_geneig[j, 1, 1]:.3f}]")
    u_level = np.array([float(u_all[r_star - ST_MEAS_WIN:r_star, r, :].mean())
                        for r in range(R)])
    np.savez(
        handoff_path, betas_grid=betas, final_pos=pos,
        metric_frozen=inv_rungs, sd_u=sd_u, sd_u_se=sd_u_se,
        tau_rounds=tau_rounds, flip_counts=flip_counts, flip_rates=flip_rates,
        exposure_steps=np.int64(exposure_steps),
        exposure_per1500=np.float64(exposure_per1500),
        trigger=np.int64(trigger), trigger_forced=np.bool_(forced),
        r_star=np.int64(r_star), knots=knots,
        beta_min=np.float64(beta_min),
        beta_min_fallback_smoke=np.bool_(bmin_fallback), knot_se=kse,
        nats_per_pair=np.float64(des["nats_per_pair"]),
        total_cost=np.float64(des["total_cost"]), carryover_map=carry,
        metric_interp=mets_interp, interp_brackets=brackets,
        interp_weights=weights, interp_geneig=interp_geneig,
        u_level_phase0=u_level, u=u_all[:r_star], ind=ind_all[:r_star],
        flips=flips, jax_key_data=np.asarray(jax.random.key_data(key)),
        seed=np.int64(seed), nsys=np.int64(nsys), K=np.int64(k_b),
        ss_max=np.float64(kn["ss_max"]), devar=np.float64(kn["devar"]),
        metric_estimator=np.array(kn["metric_est"]), smoke=np.bool_(SMOKE),
        flip_boundary=np.int64(ST_WINDOW0), meas_window=np.int64(ST_MEAS_WIN),
        trig_floor=np.int64(ST_TRIG_FLOOR), trig_every=np.int64(ST_TRIG_EVERY),
        trig_window=np.int64(ST_TRIG_WIN), deadline=np.int64(ST_ROUNDS0_MAX),
        rstar_floor=np.int64(ST_RSTAR_FLOOR), min_count=np.int64(ST_MIN_COUNT),
        rule_nsys=np.int64(ST_RULE_NSYS),
        rule_budget_steps=np.int64(ST_RULE_BUDGET),
        rule_probe_steps=np.int64(ST_RULE_PROBE),
        phase0_npz=np.array(os.path.basename(ph0_npz)))
    save_phase0(r_star - 1)   # final phase-0 arrays through R*
    pr(f"[ST {tag}] handoff saved -> {handoff_path}")

    # pre-committed plot: per-rung u traces with the trigger round marked
    fig, ax = plt.subplots(figsize=(9, 4))
    for r in range(R):
        um = u_all[:r_star, r, :].mean(axis=1)
        ax.plot(np.arange(1, r_star + 1), (um - u_level[r]) / max(sd_u[r], 1e-12),
                lw=0.7, label=f"b={betas[r]:.3f}" if r % 3 == 0 else None)
    ax.axvline(ST_WINDOW0, ls=":", color="gray", lw=0.8, label="metric boundary")
    ax.axvline(trigger, ls="--", color="r", lw=1.0,
               label=f"trigger {trigger}{' (SMOKE-FORCED)' if forced else ''}")
    ax.axvline(r_star, ls="--", color="k", lw=1.0, label=f"R* {r_star}")
    ax.set_xlabel("round")
    ax.set_ylabel("per-rung mean u, offset by R*-window level [sd(u) units]")
    ax.set_title(f"{tag}: phase-0 u traces (swaps OFF) with trigger marked")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"pt0_{tag}_phase0_utrace.png"), dpi=120)
    plt.close(fig)

    summary["trigger"] = int(trigger)
    summary["trigger_forced"] = bool(forced)
    summary["r_star"] = int(r_star)
    summary["trig_eval_rounds"] = trig_rounds
    summary["trig_eval_pass_counts"] = [int(p.sum()) for p in trig_pass]
    summary["sd_u"] = sd_u.tolist()
    summary["sd_u_se"] = sd_u_se.tolist()
    summary["tau_rounds"] = tau_rounds.tolist()
    summary["flip_counts"] = flip_counts.tolist()
    summary["flip_rates_per1500"] = flip_rates.tolist()
    summary["exposure_steps"] = int(exposure_steps)
    summary["exposure_per1500"] = float(exposure_per1500)
    summary["beta_min"] = float(beta_min)
    summary["beta_min_fallback_smoke"] = bool(bmin_fallback)
    summary["knots"] = knots.tolist()
    summary["knot_se"] = kse.tolist()
    summary["n_rungs"] = int(des["n_rungs"])
    summary["nats_per_pair"] = float(des["nats_per_pair"])
    summary["carryover_map"] = carry.tolist()
    summary["interp_geneig_minmax"] = interp_geneig.tolist()
    summary["u_level_phase0"] = u_level.tolist()
    summary["u0_verify_rel"] = u0_rel
    summary["n_nan_reverts"] = int(n_revert)
    summary["total_wall_s"] = time.time() - t0_all
    with open(os.path.join(OUT, f"summary_{tag}_phase0.json"), "w") as f:
        json.dump(summary, f, indent=1, default=_json_default)
    pr(f"[ST {tag}] phase 0 DONE in {summary['total_wall_s']:.0f}s -> {OUT}")
    return int(trigger)


def run_st_phase1(tag, handoff_path, kn):
    """PT-5a phase 1 (checkpoint 'carousel GATE PT-5a', Scheme (pinned),
    2nd paragraph; rd-3 CERTIFY-RECOMMENDED): production leg on the re-spaced
    ladder, run through run_pt -- the SAME shared harness B1-B3/control/D1/D2
    use (swaps ON, adapt_metric ON). Standalone entry point for
    GATE_PT0_ST_PHASE=1 (loads the saved handoff fresh from disk; never
    re-runs phase 0). Positions carried per new rung from the nearest-log-beta
    old rung (A5 pin, st_carryover_map, cross-checked against the handoff's
    stored map -- fail-closed on a stale/foreign handoff); per-rung metric
    seeded by log-beta-linear interpolation of the phase-0 FROZEN metrics
    (st_interp_metrics); the A1 offline interpolation-quality gen-eig
    (st_interp_geneig) is computed, PRINTED, and archived BEFORE the leg runs
    (F-H discriminating pair, leg a). run_pt has no hook to record raw u
    externally, so u is recorded via a side-effect closure passed as
    spec['u_from'] -- run_pt itself is not touched. run_pt cannot accept extra
    npz keys, so the ST/scorer keys are added by loading + re-saving
    arrays_<tag>.npz after run_pt returns (same load+resave pattern as
    elsewhere in this file); model card + summary json record phase=1,
    entry=carried, interpolated-metric, estimator, windows, seeds."""
    hd = np.load(handoff_path)
    betas0 = np.asarray(hd["betas_grid"], np.float64)          # phase-0 grid (10)
    knots = np.asarray(hd["knots"], np.float64)                # re-spaced ladder
    R1 = len(knots)                                            # NEVER hardcode 6
    final_pos = np.asarray(hd["final_pos"], np.float64)        # (R0, nsys0, DIM)
    metric_frozen = np.asarray(hd["metric_frozen"], np.float64)  # (R0, DIM, DIM)
    sd_u0 = np.asarray(hd["sd_u"], np.float64)
    sd_u0_se = np.asarray(hd["sd_u_se"], np.float64)
    tau_rounds0 = np.asarray(hd["tau_rounds"], np.float64)
    flip_counts0 = np.asarray(hd["flip_counts"], np.int64)
    flip_rates0 = np.asarray(hd["flip_rates"], np.float64)
    u_level0 = np.asarray(hd["u_level_phase0"], np.float64)
    trigger = int(np.asarray(hd["trigger"]).item())
    trigger_forced = bool(np.asarray(hd["trigger_forced"]).item())
    r_star = int(np.asarray(hd["r_star"]).item())
    beta_min = float(np.asarray(hd["beta_min"]).item())
    exposure_steps = int(np.asarray(hd["exposure_steps"]).item())
    exposure_per1500 = float(np.asarray(hd["exposure_per1500"]).item())
    phase0_npz_name = (str(np.asarray(hd["phase0_npz"])) if "phase0_npz"
                       in hd.files else os.path.basename(
                           os.path.join(OUT, f"arrays_{tag}_phase0.npz")))

    # A5 carry-over (pinned mapping): trust the handoff's map ONLY if it
    # reproduces the general rule exactly -- never silently trust a
    # stale/foreign handoff npz (fail-closed, never default)
    carry_check = st_carryover_map(betas0, knots)
    if "carryover_map" in hd.files:
        carry = np.asarray(hd["carryover_map"], dtype=np.int64)
        if not np.array_equal(carry, carry_check):
            raise RuntimeError(
                f"[ST {tag}] handoff carryover_map {carry.tolist()} != "
                f"st_carryover_map recompute {carry_check.tolist()} on "
                f"(betas_grid, knots) -- stale or foreign handoff; refusing "
                f"phase 1 (never silently trust)")
    else:
        carry = carry_check
    pr(f"[ST {tag}] phase 1: R1={R1} rungs (knots="
       f"{np.round(knots, 4).tolist()}); carry-over (A5 nearest-log-beta) = "
       + ", ".join(f"{knots[j]:.4f}<-{betas0[carry[j]]:.4f}"
                   for j in range(R1)))

    nsys, k_b = int(kn["nsys"]), int(kn["k_b"])
    if final_pos.shape[1] != nsys:
        raise RuntimeError(
            f"[ST {tag}] handoff final_pos NSYS {final_pos.shape[1]} != "
            f"phase-1 NSYS {nsys} (kn['nsys']) -- position carry-over "
            f"requires matching chain counts across phases (never default)")
    init_pos = final_pos[carry].copy()
    assert init_pos.shape == (R1, nsys, DIM), init_pos.shape

    # metric seeds: log-beta-linear interpolation of the phase-0 FROZEN
    # metrics; A1 offline interpolation-quality gen-eig BEFORE the leg runs
    mets_interp, brackets, weights = st_interp_metrics(betas0, metric_frozen,
                                                       knots)
    if "metric_interp" in hd.files:
        mi = np.asarray(hd["metric_interp"], np.float64)
        dev = float(np.max(np.abs(mets_interp - mi)))
        if dev > 1e-9:
            raise RuntimeError(
                f"[ST {tag}] recomputed st_interp_metrics deviates from the "
                f"handoff's saved metric_interp by {dev:.3e} -- stale/foreign "
                f"handoff or non-reproducible interpolation; refusing "
                f"phase 1 (never silently trust)")
    interp_geneig = st_interp_geneig(mets_interp, brackets, metric_frozen)
    pr(f"[ST {tag}] A1 offline interpolation-quality check (recomputed at "
       f"phase-1 entry; gen-eig of each seed metric vs its two bracketing "
       f"phase-0 metrics):")
    for j in range(R1):
        pr(f"  rung {j} beta={knots[j]:.4f} (brackets "
           f"{betas0[brackets[j, 0]]:.4f}/{betas0[brackets[j, 1]]:.4f}, "
           f"w={weights[j]:.3f}): vs lo [{interp_geneig[j, 0, 0]:.3f}, "
           f"{interp_geneig[j, 0, 1]:.3f}], vs hi [{interp_geneig[j, 1, 0]:.3f},"
           f" {interp_geneig[j, 1, 1]:.3f}]")

    rounds1 = int(kn["rounds1"])
    metric_windows = tuple(METRIC_WINDOWS)   # PT-2 pinned (100,250,500);
                                              # smoke-overridden globally
    metric_n0 = float(10 * nsys)             # default convention (D1/D2), not
                                              # pinned otherwise by the checkpoint
    if metric_windows[-1] >= rounds1:
        pr(f"[ST {tag}] WARNING: freeze boundary {metric_windows[-1]} >= "
           f"ROUNDS {rounds1} -- the metric will NEVER freeze in this run")

    # u recorded per phase-1 round via a side-effect closure passed as
    # spec['u_from'] -- run_pt calls this exactly once per round, in round
    # order, for every arm; recording here is purely a property of the
    # closure I supply, run_pt itself is untouched (A1 leg-b input)
    u_buf = np.zeros((rounds1, R1, nsys), np.float64)
    _u_ctr = [0]

    def u_from(logd, p):
        u = logd / knots[:, None]
        t = _u_ctr[0]
        if t < rounds1:
            u_buf[t] = u
        _u_ctr[0] = t + 1
        return u

    def u_direct(pf):
        return M["lp_batch"](pf)

    card = dict(
        arm=tag, gate="PT-5a", st_phase=1, path="power", seed=kn["seed"],
        seed_source=kn["seed_source"], smoke=SMOKE,
        script=os.path.abspath(__file__), jax=jax.__version__,
        devices=[str(d) for d in jax.devices()],
        x64=bool(jax.config.jax_enable_x64), dim=DIM,
        R=R1, NSYS=nsys, K=k_b, ROUNDS=rounds1, thin=THIN_B,
        betas=knots.tolist(),
        betas_source=(f"ladder_recipe re-space at R*={r_star} "
                     f"(trigger={trigger}{', SMOKE-FORCED' if trigger_forced else ''})"
                     f", beta_min={beta_min:.6g}; handoff "
                     f"{os.path.abspath(handoff_path)}"),
        L=L_MCLMC, ss_init=SS_INIT, ss_max=kn["ss_max"], devar=kn["devar"],
        env_overrides=dict(NSYS=kn["nsys_src"], K=kn["k_src"],
                           ss_max=kn["ssmax_src"], devar=kn["devar_src"],
                           ROUNDS=kn["rounds1_src"]),
        metric=("ADAPTIVE (windowed); seed = log-beta-linear interpolation "
                "of the phase-0 FROZEN metrics (st_interp_metrics; A1 "
                "gen-eig printed + archived above/in npz)"),
        adapt_metric=True, metric_windows=list(metric_windows),
        metric_windows_source=f"default {metric_windows} (PT-2 pinned)",
        metric_estimator=kn["metric_est"], metric_estimator_source=kn["est_src"],
        metric_n0=metric_n0,
        metric_reference="pooled MAMS64 cov (reference-ONLY, DIAGNOSTIC)",
        entry="carried (A5 nearest-log-beta position carry-over from the "
              "phase-0 final positions); metric = interpolated (NOT a fresh "
              "MAP/SVI draw or seed)",
        map_npz=MAP_NPZ, pocket_indicator=f"z[{POCKET_COL}] > {POCKET_THR}",
        init="carried: final_pos[carryover_map] from phase 0 (A5)",
        u_def="log_prob = logdensity/beta (power path)",
        harness="run_pt (shared production harness; phase-1 leg of the ST "
                "two-phase self-tuning runner)",
        sep_check_max_abs=M["sep_max"],
        st_phase0=dict(trigger=trigger, trigger_forced=trigger_forced,
                      r_star=r_star, beta_min=beta_min,
                      exposure_steps=exposure_steps,
                      exposure_per1500=exposure_per1500,
                      handoff_npz=os.path.abspath(handoff_path),
                      phase0_npz=phase0_npz_name))
    spec = dict(dim=DIM, L=L_MCLMC, betas=knots, NSYS=nsys, K=k_b,
               ROUNDS=rounds1, inv_mass=mets_interp, ss_max=kn["ss_max"],
               devar=kn["devar"], make_tf=lambda b: make_tempered("power", b),
               u_from=u_from, u_direct=u_direct,
               indicator=lambda p: p[..., POCKET_COL] > POCKET_THR,
               init_pos=init_pos, card=card, ind_label="pocket",
               adapt_metric=True, metric_windows=metric_windows,
               metric_n0=metric_n0, metric_estimator=kn["metric_est"],
               metric_ref=M["cov_pool"])

    summary, data = run_pt(tag, kn["seed"], spec)

    # ---- A1 leg (b): first-100-round phase-1 u vs the phase-0 stationary
    # level, printed for a live check (the scorer recomputes this from the
    # archived arrays; this print is the in-run corroboration) ------------
    if rounds1 >= 100:
        m1 = u_buf[:100].mean(axis=(0, 2))
        lgrid, lknot = np.log(betas0), np.log(knots)
        lvl_interp = np.interp(lknot, lgrid, u_level0)
        sd_interp = np.exp(np.interp(lknot, lgrid, np.log(sd_u0)))
        pr(f"[ST {tag}] A1 leg (b): phase-1 first-100-round u vs phase-0 "
           f"stationary level (units of phase-0 sd(u)):")
        for j in range(R1):
            s = int(carry[j])
            d_src = (m1[j] - u_level0[s]) / sd_u0[s]
            d_int = (m1[j] - lvl_interp[j]) / sd_interp[j]
            pr(f"  rung {j} beta={knots[j]:.4f}: vs source rung "
               f"(beta={betas0[s]:.4f}) {d_src:+.2f} sd; vs log-beta interp "
               f"{d_int:+.2f} sd")
    else:
        pr(f"[ST {tag}] WARNING: rounds1={rounds1} < 100 -- A1 leg (b) "
           f"first-100-round check skipped (smoke-scale run)")

    # ---- run_pt cannot accept extra npz keys: load + re-save (matches the
    # load+resave pattern already used elsewhere in this file) -------------
    npz_path = os.path.join(OUT, f"arrays_{tag}.npz")
    merged = dict(np.load(npz_path))
    extra_st = dict(
        st_trigger=np.int64(trigger), st_trigger_forced=np.bool_(trigger_forced),
        st_R_star=np.int64(r_star), st_beta_min=np.float64(beta_min),
        st_knots=knots, st_sd_u=sd_u0, st_sd_u_se=sd_u0_se,
        st_tau_rounds=tau_rounds0, st_flip_counts=flip_counts0,
        st_flip_rates=flip_rates0, st_exposure_steps=np.int64(exposure_steps),
        st_exposure_per1500=np.float64(exposure_per1500),
        st_carryover_map=carry, st_interp_geneig=interp_geneig,
        st_interp_brackets=brackets, st_interp_weights=weights,
        u_phase1=u_buf, st_phase0_npz=np.array(phase0_npz_name),
        st_handoff_npz=np.array(os.path.abspath(handoff_path)))
    collide = sorted(set(merged) & set(extra_st))
    if collide:
        raise RuntimeError(
            f"[ST {tag}] ST scorer key(s) {collide} collide with existing "
            f"run_pt npz keys in {npz_path} -- refusing to silently "
            f"overwrite (never default)")
    merged.update(extra_st)
    np.savez(npz_path, **merged)
    pr(f"[ST {tag}] phase-1 npz enriched with ST scorer keys -> {npz_path}")

    summary["st_phase1"] = dict(
        handoff_npz=os.path.abspath(handoff_path), phase0_npz=phase0_npz_name,
        trigger=trigger, trigger_forced=trigger_forced, r_star=r_star,
        beta_min=beta_min, knots=knots.tolist(), carryover_map=carry.tolist(),
        interp_geneig=interp_geneig.tolist(), sd_u_phase0=sd_u0.tolist(),
        tau_rounds_phase0=tau_rounds0.tolist(),
        flip_counts_phase0=flip_counts0.tolist(),
        exposure_per1500=exposure_per1500)
    with open(os.path.join(OUT, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1, default=_json_default)
    pr(f"[ST {tag}] phase 1 DONE -> {OUT}")


# --------------------------------------------------- GATE PT-5a-r2 PR: readiness
def pr_leak_window(ind, mask_m, mask_p, d0, t):
    """GATE PT-5a-r2 readiness leak measure: per-rung MIN(M-init-now-pocket,
    P-init-now-main) occupancy FRACTION, pooled over rounds AND chains in the
    cumulative window [d0, t) (basin_estimates leak_Minit/leak_Pinit
    convention -- the PT-4 L1b conservative-direction "min of the two basin-
    class directions" form, NOT the ST B2' flip-count rate). ind: (n_rounds,
    R, nsys) bool/uint8 pocket indicator; mask_m/mask_p: (nsys,) bool,
    disjoint, selecting the M-init / P-init chains. Returns (R,) float64."""
    if t <= d0:
        raise ValueError(f"pr_leak_window: need t > d0, got d0={d0}, t={t}")
    w = np.asarray(ind[d0:t], dtype=bool)
    leak_m = w[:, :, mask_m].mean(axis=(0, 2))
    leak_p = (~w[:, :, mask_p]).mean(axis=(0, 2))
    return np.minimum(leak_m, leak_p)


def pr_ladder_snapshot(betas, u, ind, mask_m, mask_p, d0, t):
    """One cumulative-window [d0, t) ladder/beta_min re-derivation (GATE
    PT-5a-r2 rd-1 READINESS REDESIGN, binding over the original crossing-
    count/occupancy-stationarity proxy). sd(u) via st_sd_u_window (r_star=t,
    meas_win=t-d0 -- EXACTLY the window [d0, t), pooled ddof-0, no per-chain
    mean removal, recipe convention -- REUSED, not reimplemented); leak via
    pr_leak_window over the SAME window; beta_min via the ORIGINAL (pre-B2')
    ln(100) discovery rule -- beta_min_rule(..., counts=None) -- with
    probe_steps = the window's OWN length (the task-pinned call form:
    occupancy-fraction leak pooled over nsys*window_rounds samples does not
    have the sparse-flip-count problem the B2' min-count guard targeted, so
    the plain rule applies here, unlike ST's flip-rate leak). GUARD (rd-2
    IMPLEMENTATION NOTE): design_ladder divides by (n_rungs-1); beta_min ==
    1.0 (single-rung collapse, incl. beta_min_rule finding no admissible
    beta at all) or n_rungs <= 1 -> NOT CONVERGED, beta_min/n_rungs/knots/
    knot_se all None (never crash). Returns a dict."""
    w = t - d0
    if w < 1:
        raise ValueError(f"pr_ladder_snapshot: window [{d0}:{t}) empty")
    sd, se, tau = st_sd_u_window(u, t, w)
    leak = pr_leak_window(ind, mask_m, mask_p, d0, t)
    try:
        beta_min = ladder_recipe.beta_min_rule(betas, leak, PR_RULE_NSYS,
                                               PR_RULE_BUDGET, w)
    except ValueError:
        beta_min = None
    n_rungs, knots, kse = None, None, None
    if beta_min is not None and beta_min < 1.0:
        des = ladder_recipe.design_ladder(betas, sd, beta_min, PR_TARGET_NATS)
        n_rungs = int(des["n_rungs"])
        if n_rungs > 1:
            knots = np.asarray(des["knots"], np.float64)
            kse = ladder_recipe.knot_se(betas, sd, se, beta_min,
                                        PR_TARGET_NATS)
        else:
            beta_min = None       # rd-2 guard: n_rungs <= 1
    else:
        beta_min = None            # rd-2 guard: beta_min == 1.0 / inadmissible
    return dict(t=int(t), d0=int(d0), window=int(w), sd_u=sd, sd_u_se=se,
               tau_rounds=tau, leak=leak, beta_min=beta_min, n_rungs=n_rungs,
               knots=knots, knot_se=kse)


def pr_readiness_ready(snap_t, snap_prev):
    """rd-1/rd-2 pinned READY test: compare the cumulative-window derivation
    at t vs t/1.5 (snap_t, snap_prev, from pr_ladder_snapshot). READY iff
    (i) beta_min is the SAME grid point AND non-trivial (< 1.0, i.e. not
    None under the rd-2 collapse guard) at BOTH windows, with the SAME
    n_rungs (a ladder-shape change makes the knot arrays position-
    incomparable -- treated as not converged, never crash), AND (ii)
    max|interior knot(t) - interior knot(t/1.5)| < T_TOL, T_TOL = 3x the max
    INTERIOR delta-method knot se (ladder_recipe.knot_se) at the CURRENT
    (larger) window t -- the derived W-T tolerance, not an invented
    constant. n_rungs <= 2 (no interior knots to compare) trivially
    satisfies (ii). Returns (ready: bool, max_knot_dev: float or None,
    t_tol: float or None)."""
    bt, bp = snap_t["beta_min"], snap_prev["beta_min"]
    if bt is None or bp is None:
        return False, None, None
    if snap_t["n_rungs"] != snap_prev["n_rungs"]:
        return False, None, None
    if not math.isclose(bt, bp, rel_tol=0.0, abs_tol=1e-9):
        return False, None, None
    n = snap_t["n_rungs"]
    if n <= 2:
        return True, 0.0, 0.0
    max_dev = float(np.max(np.abs(snap_t["knots"][1:-1]
                                  - snap_prev["knots"][1:-1])))
    t_tol = 3.0 * float(np.max(snap_t["knot_se"][1:-1]))
    return (max_dev < t_tol), max_dev, t_tol


def pr_selftest():
    """--selftest-pr: numpy-only unit test of the GATE PT-5a-r2 readiness
    convergence logic (pr_ladder_snapshot / pr_readiness_ready, the rd-1
    READINESS REDESIGN, rd-2 CERTIFY-RECOMMENDED) against the ARCHIVED arm-A
    probe data (arrays_A_power.npz, broad MAMS init, step-resolved u/ind) --
    reproduces the checkpoint's verified table (readiness FALSE at
    cumulative t=300..1100, fires at t=1200) as the core correctness check.
    Here "t"/"d0" are raw MCLMC STEPS (arm-A's per-step archived trace,
    D0=100 per rd-1); the live PR runner calls the SAME functions with
    t/d0 in ROUNDS (PR_D0=10 rounds = 100 steps at K=10) -- the functions are
    generic over the trace's time unit. No jax use, no model build."""
    npz = os.path.join(OUT, "arrays_A_power.npz")
    if not os.path.exists(npz):
        raise RuntimeError(f"--selftest-pr needs {npz} (archived arm-A "
                           f"broad-init probe data); not found")
    d = np.load(npz)
    u_g, ind_g, betas = d["u"], d["ind"], d["betas"]   # (10, 2, 3000, 16)
    R, ng, T, nch = u_g.shape
    assert ng == 2, ng
    # combine the two init-groups (0=M-init, 1=P-init; arm-A's basin_
    # estimates convention) into one chain axis, group-major so mask_m/
    # mask_p are contiguous halves
    u_cum = np.transpose(u_g, (2, 0, 1, 3)).reshape(T, R, ng * nch)
    ind_cum = np.transpose(ind_g, (2, 0, 1, 3)).reshape(T, R, ng * nch
                                                        ).astype(bool)
    mask_m = np.zeros(ng * nch, dtype=bool)
    mask_m[:nch] = True
    mask_p = np.zeros(ng * nch, dtype=bool)
    mask_p[nch:] = True
    d0 = 100

    def snap(t):
        return pr_ladder_snapshot(betas, u_cum, ind_cum, mask_m, mask_p, d0, t)

    s300, s200 = snap(300), snap(200)
    ready300, dev300, tol300 = pr_readiness_ready(s300, s200)
    assert s300["beta_min"] is None, s300["beta_min"]          # collapsed (1.0)
    assert round(s200["beta_min"], 4) == 0.3594, s200["beta_min"]
    assert not ready300, (ready300, dev300, tol300)
    pr(f"[selftest-pr] t=300: beta_min collapsed (1.0 -> None under the rd-2 "
       f"guard) vs t/1.5=200's {s200['beta_min']:.4f} -> mismatch, NOT READY "
       f"PASS")
    s1200, s800 = snap(1200), snap(800)
    ready1200, dev1200, tol1200 = pr_readiness_ready(s1200, s800)
    assert round(s1200["beta_min"], 4) == 0.3594, s1200["beta_min"]
    assert round(s800["beta_min"], 4) == 0.3594, s800["beta_min"]
    assert s1200["n_rungs"] == s800["n_rungs"] == 6, \
        (s1200["n_rungs"], s800["n_rungs"])
    assert dev1200 < tol1200, (dev1200, tol1200)
    assert ready1200, (ready1200, dev1200, tol1200)
    pr(f"[selftest-pr] t=1200: beta_min stable at 0.3594 vs t/1.5=800 (both "
       f"6 rungs); max|Delta knot|={dev1200:.4g} < T_TOL={tol1200:.4g} (3x "
       f"max interior knot se) -> READY PASS")
    # intermediate cumulative rounds 300..1100 all NOT ready (reproduces the
    # checkpoint's "FALSE at 300/400/.../1000/1100" verified table row)
    for t in (300, 400, 500, 600, 700, 800, 900, 1000, 1100):
        tprev = int(t / 1.5)
        if tprev <= d0:
            continue
        ready, _, _ = pr_readiness_ready(snap(t), snap(tprev))
        assert not ready, (t, ready)
    pr("[selftest-pr] readiness FALSE at cumulative t=300..1100 (checkpoint's "
       "verified table) PASS")
    pr("[selftest-pr] ALL PR readiness checks PASS")


def run_pr_phase_p(tag, handoff_path, kn):
    """GATE PT-5a-r2 phase P (probe): BROAD-init measurement leg (run_arm_a's
    draw_init convention -- half M-pool, half P-pool per rung, PR_NSYS=16
    total), the arm-A probe grid, swaps OFF (kernel-only leakage semantics,
    the ST B1 pin carried over -- swap-relabeling would corrupt the per-
    init-group leak measurement), a FIXED (non-adaptive) pooled metric
    (M["cov_pool"], the arm-A hot-end unconfined convention -- the checkpoint
    does not pin a windowed metric adaptation for phase P), chunked via
    round_all_chunked at PR_PHASE0_CHUNK (96-wide OOM fix, reused unchanged).
    Readiness = the rd-1 CUMULATIVE-CONVERGENCE redesign (pr_ladder_snapshot
    / pr_readiness_ready), NOT a per-round u-stationarity trigger. Returns
    True if readiness fired (handoff + probe npz saved), False on F-R (cap
    reached without convergence; probe npz still saved, no handoff)."""
    betas = np.geomspace(0.01, 1.0, PR_GRID_N)   # arm-A probe grid (PINNED)
    R = len(betas)
    chunk, chunk_src = _st_env_num("GATE_PT0_ST_CHUNK", PR_PHASE0_CHUNK, int)
    seed, nsys, k_b = kn["seed"], kn["nsys"], kn["k_b"]
    if nsys % 2 != 0:
        raise RuntimeError(f"PR phase P needs an even NSYS for the half-M/"
                           f"half-P broad init, got {nsys}")
    half = nsys // 2
    rng = np.random.default_rng(seed)
    key = jax.random.key(seed)

    def draw_init(pool, n):     # run_arm_a's draw_init, verbatim convention
        idx = rng.integers(0, len(pool), size=n)
        return pool[idx] + JITTER * rng.standard_normal((n, DIM))

    pos = np.zeros((R, nsys, DIM))
    for r in range(R):
        pos[r] = np.concatenate([draw_init(M["pool_M"], half),
                                 draw_init(M["pool_P"], half)])
    mask_m = np.zeros(nsys, dtype=bool)
    mask_m[:half] = True
    mask_p = np.zeros(nsys, dtype=bool)
    mask_p[half:] = True

    inv_mass = np.asarray(M["cov_pool"], dtype=np.float64)
    inv_rungs_j = jnp.asarray(np.broadcast_to(inv_mass, (R, DIM, DIM)))
    betas_j = jnp.asarray(betas)
    steps_all = jnp.full((R, nsys), SS_INIT)
    adapt_all = fresh_adapt2(R, nsys, kn["ss_max"])

    card = dict(
        arm=tag, gate="PT-5a-r2", pr_phase="P", path="power", seed=seed,
        seed_source=kn["seed_source"], smoke=SMOKE,
        script=os.path.abspath(__file__), jax=jax.__version__,
        devices=[str(d) for d in jax.devices()],
        x64=bool(jax.config.jax_enable_x64), dim=DIM,
        R=R, NSYS=nsys, K=k_b, cap_rounds=PR_CAP_ROUNDS,
        betas=betas.tolist(),
        betas_source="arm-A probe grid geomspace(0.01, 1, 10) (PINNED)",
        L=L_MCLMC, ss_init=SS_INIT, ss_max=kn["ss_max"], devar=kn["devar"],
        env_overrides=dict(NSYS=kn["nsys_src"], K=kn["k_src"],
                           ss_max=kn["ssmax_src"], devar=kn["devar_src"],
                           chunk=chunk_src),
        swaps_off_phase_P=True,
        init=(f"BROAD: {half} chains draw_init(pool_M) + {half} chains "
             f"draw_init(pool_P) per rung (run_arm_a convention, "
             f"JITTER={JITTER})"),
        metric="FIXED (non-adaptive) pooled MAMS64 empirical cov, all rungs "
               "(arm-A hot-end unconfined convention)",
        pocket_indicator=f"z[{POCKET_COL}] > {POCKET_THR}",
        u_def="log_prob = logdensity/beta (power path; the u the readiness "
              "re-derivation uses -- swaps OFF, B1-class pin)",
        readiness=dict(
            rule="rd-1 CUMULATIVE-CONVERGENCE (supersedes the crossing-"
                 "count/occupancy-stationarity proxy)",
            d0_rounds=PR_D0, every_rounds=PR_READY_EVERY,
            floor_round=PR_READY_FLOOR, cap_rounds=PR_CAP_ROUNDS,
            growth_factor=1.5, rule_nsys=PR_RULE_NSYS,
            rule_budget_steps=PR_RULE_BUDGET, target_nats=PR_TARGET_NATS,
            t_tol="3x max interior ladder_recipe.knot_se at the current "
                  "(larger) window -- derived, not invented"),
        harness=(f"PR phase-P fused kernel rounds (round_all_chunked, "
                f"chunk={chunk}), NO swap attempts"),
        sep_check_max_abs=M["sep_max"])
    pr("MODEL CARD:", json.dumps(card, indent=1, default=_json_default))
    summary = dict(model_card=card)
    t0_all = time.time()

    spec_kernel = dict(dim=DIM, L=L_MCLMC, K=k_b, devar=kn["devar"],
                       make_tf=lambda b: make_tempered("power", b))
    round_all = make_fused_round_runner(spec_kernel)

    u_all = np.zeros((PR_CAP_ROUNDS, R, nsys), np.float64)
    ind_all = np.zeros((PR_CAP_ROUNDS, R, nsys), dtype=np.uint8)
    ev = np.zeros((PR_CAP_ROUNDS, R))
    ssm = np.zeros((PR_CAP_ROUNDS, R))
    n_revert = 0
    u0_rel = None
    ready_t, ready_pass = [], []      # readiness_trace parallel arrays
    ready_bmin, ready_dev, ready_tol = [], [], []
    t_ready, readiness_fired = None, False
    final_snap = None
    ph_npz = os.path.join(OUT, f"arrays_PR_{tag}_probe.npz")

    def save_probe(t):
        np.savez(
            ph_npz, betas_grid=betas, u=u_all[:t + 1], ind=ind_all[:t + 1],
            eevpd=ev[:t + 1], step_mean=ssm[:t + 1],
            swap_attempts=np.zeros((R - 1, 2), dtype=np.int64),
            swap_accepts=np.zeros((R - 1, 2), dtype=np.int64),
            swaps_off=np.bool_(True), metric_fixed=inv_mass,
            n_revert=np.int64(n_revert), rounds_done=np.int64(t + 1),
            readiness_fired=np.bool_(readiness_fired),
            t_ready_steps=np.int64(-1 if t_ready is None else t_ready),
            readiness_trace_t=np.asarray(ready_t, dtype=np.int64),
            readiness_trace_pass=np.asarray(ready_pass, dtype=np.bool_),
            readiness_trace_beta_min=np.asarray(
                [np.nan if b is None else b for b in ready_bmin], np.float64),
            readiness_trace_maxdev=np.asarray(
                [np.nan if v is None else v for v in ready_dev], np.float64),
            readiness_trace_ttol=np.asarray(
                [np.nan if v is None else v for v in ready_tol], np.float64),
            mask_m=mask_m, mask_p=mask_p,
            arm=np.array(tag), gate=np.array("PT-5a-r2"),
            seed=np.int64(seed), nsys=np.int64(nsys), K=np.int64(k_b),
            ss_max=np.float64(kn["ss_max"]), devar=np.float64(kn["devar"]),
            smoke=np.bool_(SMOKE), d0_rounds=np.int64(PR_D0),
            every_rounds=np.int64(PR_READY_EVERY),
            floor_round=np.int64(PR_READY_FLOOR),
            cap_rounds=np.int64(PR_CAP_ROUNDS))

    pr(f"[PR {tag}] phase P: swaps OFF, {R}x{nsys} = {R * nsys} chains "
       f"(broad init, half M-pool/half P-pool), K={k_b}, readiness every "
       f"{PR_READY_EVERY} from round {PR_READY_FLOOR}, D0={PR_D0} rounds, "
       f"cap {PR_CAP_ROUNDS} rounds")
    t0 = time.time()
    t_block, last_block_rd = t0, 0
    for t in range(PR_CAP_ROUNDS):
        key, *lks = jax.random.split(key, R + 1)
        ik_all, sk_all = _round_keys(lks, R, nsys, k_b)
        p2, ld, steps_all, adapt_all, ec2, ok = round_all_chunked(
            round_all, jnp.asarray(pos), steps_all, adapt_all, ik_all,
            sk_all, betas_j, inv_rungs_j, chunk)
        pos = np.array(p2)
        logd = np.asarray(ld)
        ev[t] = np.asarray(ec2).mean(axis=(1, 2)) / DIM
        ssm[t] = np.asarray(steps_all).mean(axis=1)
        n_revert += int((~np.asarray(ok)).sum())
        u = logd / betas[:, None]   # power-path u (readiness reads this)
        if t == 0:   # round-0 identity verification (run_pt convention)
            u_direct = M["lp_batch"](pos.reshape(-1, DIM)).reshape(R, nsys)
            u0_rel = float(np.max(np.abs(u - u_direct)
                                  / (1.0 + np.abs(u_direct))))
            pr(f"[PR {tag}] round-0 u identity: rel = {u0_rel:.3e}")
            if u0_rel > U_REL_TOL:
                raise RuntimeError(f"round-0 u verification failed: rel "
                                   f"{u0_rel:.3e} > {U_REL_TOL}")
        u_all[t] = u
        ind_all[t] = pos[:, :, POCKET_COL] > POCKET_THR   # round-end indicator
        # NO swap attempts (B1-class pin): kernel rounds only in phase P
        rd = t + 1
        if (rd >= PR_READY_FLOOR
                and (rd - PR_READY_FLOOR) % PR_READY_EVERY == 0):
            tprev = int(rd / 1.5)
            snap_t = ok_ready = maxdev = ttol = None
            if tprev > PR_D0:
                try:
                    snap_t = pr_ladder_snapshot(betas, u_all[:rd],
                                               ind_all[:rd], mask_m, mask_p,
                                               PR_D0, rd)
                    snap_p = pr_ladder_snapshot(betas, u_all[:rd],
                                               ind_all[:rd], mask_m, mask_p,
                                               PR_D0, tprev)
                    ok_ready, maxdev, ttol = pr_readiness_ready(snap_t,
                                                                snap_p)
                except ValueError as e:   # transient IAT-window underflow;
                    ok_ready = False      # never crash the probe on this
                    pr(f"[PR {tag}] readiness eval @ round {rd}: SKIPPED "
                       f"({e}) -- not ready")
            else:
                ok_ready = False
            ready_t.append(rd)
            ready_pass.append(bool(ok_ready))
            ready_bmin.append(None if snap_t is None else snap_t["beta_min"])
            ready_dev.append(maxdev)
            ready_tol.append(ttol)
            bm_str = ("n/a" if snap_t is None or snap_t["beta_min"] is None
                      else f"{snap_t['beta_min']:.4f}")
            pr(f"[PR {tag}] readiness eval @ round {rd} (tprev={tprev}): "
               f"beta_min={bm_str} maxdev={_f3(maxdev)} T_TOL={_f3(ttol)}"
               f"{' -> READY' if ok_ready else ''}")
            if ok_ready:
                t_ready, readiness_fired, final_snap = rd, True, snap_t
            elif SMOKE and rd == PR_READY_FLOOR:
                # SMOKE forced-readiness (mirrors the ST forced-trigger
                # pattern): exercise the handoff/phase-Q path even if the
                # natural convergence test cannot fire in a tiny smoke
                # budget -- NOT a measurement; production never forces.
                fb = snap_t if (snap_t is not None
                                and snap_t["beta_min"] is not None) else None
                if fb is None:
                    bm_fb = float(betas[7])       # 0.35938-class grid point
                    sd_fb, se_fb, tau_fb = st_sd_u_window(u_all[:rd], rd,
                                                         rd - PR_D0)
                    des_fb = ladder_recipe.design_ladder(
                        betas, sd_fb, bm_fb, PR_TARGET_NATS)
                    kse_fb = ladder_recipe.knot_se(betas, sd_fb, se_fb,
                                                   bm_fb, PR_TARGET_NATS)
                    fb = dict(
                        t=rd, d0=PR_D0, window=rd - PR_D0, sd_u=sd_fb,
                        sd_u_se=se_fb, tau_rounds=tau_fb,
                        leak=pr_leak_window(ind_all[:rd], mask_m, mask_p,
                                            PR_D0, rd),
                        beta_min=bm_fb, n_rungs=int(des_fb["n_rungs"]),
                        knots=np.asarray(des_fb["knots"], np.float64),
                        knot_se=kse_fb)
                t_ready, readiness_fired, final_snap = rd, True, fb
                pr(f"[PR {tag}] SMOKE: readiness FORCED at round {rd} "
                   f"(natural test not converged) -- forced-readiness smoke "
                   f"per checkpoint; NOT a measurement")
        if (rd % PRINT_EVERY_B == 0 or rd == PR_CAP_ROUNDS or t_ready == rd):
            spr = (time.time() - t_block) / max(rd - last_block_rd, 1)
            t_block, last_block_rd = time.time(), rd
            pr(f"[PR {tag}] phaseP round {rd:4d} "
               f"cold occ={ind_all[t, -1].mean():.3f} "
               f"hot occ={ind_all[t, 0].mean():.3f} "
               f"EEVPD[h/m/c]={ev[t, 0]:.1e}/{ev[t, R // 2]:.1e}/"
               f"{ev[t, -1]:.1e} reverts={n_revert} block {spr:.2f} "
               f"s/round ({R * nsys}-wide) wall={time.time() - t0:.0f}s")
        if rd % SAVE_EVERY_B == 0:
            save_probe(t)
        if t_ready == rd:
            break

    if not readiness_fired:
        save_probe(PR_CAP_ROUNDS - 1)
        summary["f_readiness_never"] = True
        summary["readiness_trace_t"] = ready_t
        summary["total_wall_s"] = time.time() - t0_all
        with open(os.path.join(OUT, f"summary_PR_{tag}_probe.json"), "w") as f:
            json.dump(summary, f, indent=1, default=_json_default)
        pr(f"\n[PR {tag}] " + "!" * 66)
        pr(f"[PR {tag}] F-R: readiness did NOT fire by the round-"
           f"{PR_CAP_ROUNDS} cap -- broad-init occupancy/ladder is not "
           f"converging as the arm-A finding predicted.")
        pr(f"[PR {tag}] Probe arrays saved -> {ph_npz}. NO handoff, NO "
           f"phase Q attempted (A4-class exit path).")
        pr(f"[PR {tag}] " + "!" * 66)
        return False

    # ---------------- final measurement + crossing-count sanity floor ------
    sd_u, sd_u_se = final_snap["sd_u"], final_snap["sd_u_se"]
    tau_rounds, leak = final_snap["tau_rounds"], final_snap["leak"]
    beta_min, knots, kse = (final_snap["beta_min"], final_snap["knots"],
                            final_snap["knot_se"])
    n_rungs = final_snap["n_rungs"]
    crossing_counts, _ = st_flip_counts(ind_all[:t_ready], PR_D0, t_ready)
    pr(f"[PR {tag}] READY at round {t_ready} (D0={PR_D0}): beta_min="
       f"{beta_min:.6g}, {n_rungs} rungs, knots="
       + ", ".join(f"{k:.6g}(+-{s:.2e})" for k, s in zip(knots, kse)))
    pr(f"[PR {tag}] crossing-count sanity floor (report-only, NOT the gate; "
       f"cumulative basin crossings over ({PR_D0}, {t_ready}]): "
       + ", ".join(f"b={betas[r]:.4f}:{int(crossing_counts[r])}"
                   for r in range(R)))

    np.savez(
        handoff_path, betas_grid=betas, readiness_fired=np.bool_(True),
        t_ready_steps=np.int64(t_ready),
        readiness_trace_t=np.asarray(ready_t, dtype=np.int64),
        readiness_trace_pass=np.asarray(ready_pass, dtype=np.bool_),
        readiness_trace_beta_min=np.asarray(
            [np.nan if b is None else b for b in ready_bmin], np.float64),
        readiness_trace_maxdev=np.asarray(
            [np.nan if v is None else v for v in ready_dev], np.float64),
        readiness_trace_ttol=np.asarray(
            [np.nan if v is None else v for v in ready_tol], np.float64),
        sd_u=sd_u, sd_u_se=sd_u_se, tau_rounds=tau_rounds, leak=leak,
        knots=knots, beta_min=np.float64(beta_min), knot_se=kse,
        n_rungs=np.int64(n_rungs), crossing_counts=crossing_counts,
        d0_rounds=np.int64(PR_D0), seed=np.int64(seed), nsys=np.int64(nsys),
        K=np.int64(k_b), smoke=np.bool_(SMOKE),
        probe_npz=np.array(os.path.basename(ph_npz)))
    save_probe(t_ready - 1)
    pr(f"[PR {tag}] handoff saved -> {handoff_path}")

    # pre-committed plot (rd-1): beta_min(t) and max|Delta knot|(t) vs
    # cumulative round t, fire-round marked
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    tt = np.asarray(ready_t)
    bmv = np.asarray([np.nan if b is None else b for b in ready_bmin])
    dv = np.asarray([np.nan if v is None else v for v in ready_dev])
    tolv = np.asarray([np.nan if v is None else v for v in ready_tol])
    ax[0].plot(tt, bmv, "o-")
    ax[0].axhline(beta_min, ls="--", color="k", lw=0.8,
                  label=f"final {beta_min:.4f}")
    ax[0].axvline(t_ready, ls="--", color="r", lw=1.0, label="READY")
    ax[0].set_xlabel("cumulative round t")
    ax[0].set_ylabel("beta_min(t)")
    ax[0].legend(fontsize=8)
    ax[0].set_title(f"{tag}: readiness beta_min(t)")
    ax[1].plot(tt, dv, "o-", label="max|Delta knot| (t vs t/1.5)")
    ax[1].plot(tt, tolv, "s--", label="T_TOL (3x max interior knot se)")
    ax[1].axvline(t_ready, ls="--", color="r", lw=1.0, label="READY")
    ax[1].set_xlabel("cumulative round t")
    ax[1].set_ylabel("beta units")
    ax[1].legend(fontsize=8)
    ax[1].set_title(f"{tag}: readiness knot deviation vs T_TOL")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"pt0_PR_{tag}_readiness.png"), dpi=120)
    plt.close(fig)

    summary["readiness_fired"] = True
    summary["t_ready_steps"] = int(t_ready)
    summary["readiness_trace_t"] = ready_t
    summary["beta_min"] = float(beta_min)
    summary["n_rungs"] = int(n_rungs)
    summary["knots"] = knots.tolist()
    summary["knot_se"] = kse.tolist()
    summary["sd_u"] = sd_u.tolist()
    summary["leak"] = leak.tolist()
    summary["crossing_counts"] = crossing_counts.tolist()
    summary["u0_verify_rel"] = u0_rel
    summary["n_nan_reverts"] = int(n_revert)
    summary["total_wall_s"] = time.time() - t0_all
    with open(os.path.join(OUT, f"summary_PR_{tag}_probe.json"), "w") as f:
        json.dump(summary, f, indent=1, default=_json_default)
    pr(f"[PR {tag}] phase P DONE in {summary['total_wall_s']:.0f}s -> {OUT}")
    return True


def run_pr_phase_q(tag, handoff_path, kn):
    """GATE PT-5a-r2 phase Q (production): the D2/C-24 production leg --
    FRESH MAP + 1e-6*I entry, ALL rungs (NOT carried from the phase-P
    broad-init probe: production needs only TRANSPORT on the probe's ladder,
    not the probe's positions/metric -- rd-1 crux-1) -- run through run_pt,
    the SAME shared harness B1-B3/control/D1/D2/ST-phase1 use. The ladder
    (knots from the probe's handoff) is the ONLY thing that changes vs a
    normal D2 run; swaps ON, adapt_metric ON (run_pt default behaviour)."""
    hd = np.load(handoff_path)
    knots = np.asarray(hd["knots"], np.float64)
    beta_min = float(np.asarray(hd["beta_min"]).item())
    t_ready = int(np.asarray(hd["t_ready_steps"]).item())
    n_rungs = int(np.asarray(hd["n_rungs"]).item())
    probe_npz_name = str(np.asarray(hd["probe_npz"]))
    R1 = len(knots)
    if R1 != n_rungs:
        raise RuntimeError(f"[PR {tag}] handoff n_rungs {n_rungs} != "
                           f"len(knots) {R1} -- stale/foreign handoff, "
                           f"refusing phase Q (never silently trust)")

    seed, nsys, k_b = kn["seed"], kn["nsys"], kn["k_b"]
    rng = np.random.default_rng(seed)
    map_npz = np.load(MAP_NPZ)
    map_zbest = np.asarray(map_npz["z_best"], dtype=np.float64).reshape(-1)
    assert map_zbest.shape == (DIM,), map_zbest.shape
    pos = np.zeros((R1, nsys, DIM))
    for r in range(R1):    # fresh MAP z_best + stock-diagonal draws (D2 entry)
        pos[r] = map_zbest + D2_INIT_SCALE * rng.standard_normal((nsys, DIM))
    inv_mass = (D2_INIT_SCALE ** 2) * np.eye(DIM)
    d2_diag_provenance = (   # same provenance pin as D2 (run_arm_b)
        f"stock no-SVI diagonal convention: init_scales={D2_INIT_SCALE} "
        f"(STD in z) from ModellingSequence.SVI default "
        f"(gigalens/src/gigalens/jax/inference.py:254,282-284; mirrored "
        f"gigalens_research/inference_utils/pipeline.py:1440) => seed cov "
        f"= {D2_INIT_SCALE}^2 * I = {D2_INIT_SCALE**2:g}*I")

    rounds1, rounds1_src = _st_env_num("GATE_PT0_ROUNDS_B", ST_ROUNDS1, int)
    if SMOKE and rounds1 != ST_ROUNDS1:
        rounds1_src = (f"SMOKE override {ST_ROUNDS1} [env GATE_PT0_ROUNDS_B "
                       f"IGNORED; precedence smoke > env > default]")
        rounds1 = ST_ROUNDS1
    metric_windows = tuple(METRIC_WINDOWS)   # PT-2 pinned (100,250,500);
                                              # smoke-overridden globally
    metric_n0 = float(10 * nsys)
    if metric_windows[-1] >= rounds1:
        pr(f"[PR {tag}] WARNING: freeze boundary {metric_windows[-1]} >= "
           f"ROUNDS {rounds1} -- the metric will NEVER freeze in this run")

    def u_from(logd, p):
        return logd / knots[:, None]

    def u_direct(pf):
        return M["lp_batch"](pf)

    card = dict(
        arm=tag, gate="PT-5a-r2", pr_phase="Q", path="power", seed=seed,
        seed_source=kn["seed_source"], smoke=SMOKE,
        script=os.path.abspath(__file__), jax=jax.__version__,
        devices=[str(d) for d in jax.devices()],
        x64=bool(jax.config.jax_enable_x64), dim=DIM,
        R=R1, NSYS=nsys, K=k_b, ROUNDS=rounds1, thin=THIN_B,
        betas=knots.tolist(),
        betas_source=(f"PR phase-P ladder (readiness fired at cumulative "
                     f"round {t_ready}, beta_min={beta_min:.6g}); handoff "
                     f"{os.path.abspath(handoff_path)}"),
        L=L_MCLMC, ss_init=SS_INIT, ss_max=kn["ss_max"], devar=kn["devar"],
        env_overrides=dict(NSYS=kn["nsys_src"], K=kn["k_src"],
                           ss_max=kn["ssmax_src"], devar=kn["devar_src"],
                           ROUNDS=rounds1_src),
        metric=(f"ADAPTIVE (windowed); seed = diagonal "
               f"{D2_INIT_SCALE**2:g}*I (D2 stock init_scales convention)"),
        adapt_metric=True, metric_windows=list(metric_windows),
        metric_windows_source=f"default {metric_windows} (PT-2 pinned)",
        metric_estimator=kn["metric_est"],
        metric_estimator_source=kn["est_src"], metric_n0=metric_n0,
        metric_reference="pooled MAMS64 cov (reference-ONLY, DIAGNOSTIC)",
        d2_diag_provenance=d2_diag_provenance,
        entry=("FRESH MAP z_best + stock-diagonal draws, ALL rungs (D2 "
              "convention; NOT carried from the phase-P broad-init probe -- "
              "production needs only TRANSPORT on the probe's ladder)"),
        map_npz=MAP_NPZ, pocket_indicator=f"z[{POCKET_COL}] > {POCKET_THR}",
        init="MAP z_best + stock-diagonal draws all rungs (as D2)",
        u_def="log_prob = logdensity/beta (power path)",
        harness=("run_pt (shared production harness; phase-Q leg of the "
                "PT-5a-r2 probe->production runner)"),
        sep_check_max_abs=M["sep_max"],
        pr_phase_p=dict(t_ready_steps=t_ready, beta_min=beta_min,
                        n_rungs=n_rungs,
                        handoff_npz=os.path.abspath(handoff_path),
                        probe_npz=probe_npz_name))
    spec = dict(dim=DIM, L=L_MCLMC, betas=knots, NSYS=nsys, K=k_b,
               ROUNDS=rounds1, inv_mass=inv_mass, ss_max=kn["ss_max"],
               devar=kn["devar"], make_tf=lambda b: make_tempered("power", b),
               u_from=u_from, u_direct=u_direct,
               indicator=lambda p: p[..., POCKET_COL] > POCKET_THR,
               init_pos=pos, card=card, ind_label="pocket",
               adapt_metric=True, metric_windows=metric_windows,
               metric_n0=metric_n0, metric_estimator=kn["metric_est"],
               metric_ref=M["cov_pool"])

    summary, data = run_pt(tag, seed, spec)

    # ---- run_pt cannot accept extra npz keys: load + re-save (matches the
    # load+resave pattern already used by run_st_phase1) --------------------
    npz_path = os.path.join(OUT, f"arrays_{tag}.npz")
    merged = dict(np.load(npz_path))
    extra_pr = dict(
        pr_t_ready_steps=np.int64(t_ready), pr_beta_min=np.float64(beta_min),
        pr_knots=knots, pr_readiness_fired=np.bool_(True),
        pr_readiness_trace_t=hd["readiness_trace_t"],
        pr_readiness_trace_beta_min=hd["readiness_trace_beta_min"],
        pr_readiness_trace_maxdev=hd["readiness_trace_maxdev"],
        pr_crossing_counts=hd["crossing_counts"],
        pr_probe_npz=np.array(probe_npz_name),
        pr_handoff_npz=np.array(os.path.abspath(handoff_path)))
    collide = sorted(set(merged) & set(extra_pr))
    if collide:
        raise RuntimeError(f"[PR {tag}] PR scorer key(s) {collide} collide "
                           f"with existing run_pt npz keys in {npz_path} -- "
                           f"refusing to silently overwrite (never default)")
    merged.update(extra_pr)
    np.savez(npz_path, **merged)
    pr(f"[PR {tag}] phase-Q npz enriched with PR probe cross-reference keys "
       f"-> {npz_path}")

    summary["pr_phase_q"] = dict(
        handoff_npz=os.path.abspath(handoff_path), probe_npz=probe_npz_name,
        t_ready_steps=t_ready, beta_min=beta_min, n_rungs=n_rungs,
        knots=knots.tolist())
    with open(os.path.join(OUT, f"summary_{tag}.json"), "w") as f:
        json.dump(summary, f, indent=1, default=_json_default)
    pr(f"[PR {tag}] phase Q DONE -> {OUT}")


def run_arm_pr(tag):
    """GATE PT-5a-r2 probe->production runner (checkpoint 'carousel GATE
    PT-5a-r2'; rd-1 READINESS REDESIGN + rd-2 4 items BINDING). Phase P =
    broad-init measurement leg (run_pr_phase_p); readiness = the cumulative-
    convergence re-derivation (NOT the falsified crossing-count/occupancy-
    stationarity proxy PT-5a-r2 replaced). Phase Q = the D2/C-24 production
    leg on the probe's ladder (run_pr_phase_q), FRESH MAP entry (NOT carried
    from the probe). GATE_PT0_PR_PHASE: auto (default) = phase P then, iff
    readiness fired, phase Q in-process; 0 = phase P only; 1 = phase Q
    standalone from a saved handoff (debugging/restart use -- the
    checkpoint's cost section assumes ONE allocation for both phases, no
    split-alloc contingency, unlike ST/PT-5a)."""
    if LEGACY_RUNNERS:
        raise RuntimeError("arm PR requires the fused runner; unset "
                           "GATE_PT0_LEGACY_RUNNERS")
    for bad in ("GATE_PT0_BETAS_B", "GATE_PT0_METRIC_WINDOWS"):
        if os.environ.get(bad, "").strip():
            raise RuntimeError(
                f"{bad} is set, but arm PR pins its grid/windows (phase-P "
                f"grid = arm-A probe grid; phase-Q windows "
                f"{tuple(METRIC_WINDOWS)}); unset it (launch-discipline: NO "
                f"conflicting env)")
    seed = ARM_SEED["PR"]
    seed_source = "ARM_SEED default"
    env_seed = os.environ.get("GATE_PT0_SEED_B", "").strip()
    if env_seed:   # PT-5a-r2 PR production arms: seeds pinned at launch
        seed = int(env_seed)
        seed_source = f"env override GATE_PT0_SEED_B={env_seed}"
    nsys, nsys_src = _st_env_num("GATE_PT0_NSYS_B", PR_NSYS, int)
    k_b, k_src = _st_env_num("GATE_PT0_K_B", K_B, int)
    ss_max, ssmax_src = _st_env_num("GATE_PT0_SSMAX", SS_MAX0, float)
    devar, devar_src = _st_env_num("GATE_PT0_DEVAR", DEVAR, float)
    env_est = os.environ.get("GATE_PT0_METRIC_EST", "").strip()
    # phase Q defaults to WITHIN (the A1 default, reused) -- a launch that
    # omits the env still runs the correct production estimator
    metric_est = env_est or "within"
    assert metric_est in ("pooled", "within"), \
        f"GATE_PT0_METRIC_EST must be 'pooled' or 'within', got {metric_est!r}"
    est_src = (f"env override GATE_PT0_METRIC_EST={env_est}" if env_est
               else "default")
    phase = os.environ.get("GATE_PT0_PR_PHASE", "").strip() or "auto"
    if phase not in ("auto", "0", "1"):
        raise RuntimeError(
            f"GATE_PT0_PR_PHASE must be auto/0/1, got {phase!r}")
    kn = dict(seed=seed, seed_source=seed_source, nsys=nsys, nsys_src=nsys_src,
              k_b=k_b, k_src=k_src, ss_max=ss_max, ssmax_src=ssmax_src,
              devar=devar, devar_src=devar_src, metric_est=metric_est,
              est_src=est_src, phase=phase)
    handoff_path = os.path.join(OUT, f"handoff_PR_{tag}.npz")
    if phase in ("auto", "0"):
        ready = run_pr_phase_p(tag, handoff_path, kn)
        if not ready:
            return                          # F-R (no handoff, no phase Q)
        if phase == "0":
            pr(f"[PR {tag}] GATE_PT0_PR_PHASE=0: phase P complete, handoff "
               f"saved -> {handoff_path}; run phase Q with "
               f"GATE_PT0_PR_PHASE=1")
            return
    run_pr_phase_q(tag, handoff_path, kn)


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="GATE PT-0 (one arm per process)")
    ap.add_argument("--arm", required=True,
                    choices=["smoke", "control", "A_power", "A_lik",
                             "B1", "B2", "B3", "B4", "B5", "D1", "D2", "ST",
                             "PR"])
    ap.add_argument("--equiv-check", action="store_true",
                    help="fused-vs-legacy runner equivalence guard: 3 rounds, "
                         "both implementations, identical inits/keys, SMOKE "
                         "config; requires a PT arm (control/B1/B2/B3)")
    ap.add_argument("--selftest-st", action="store_true",
                    help="numpy-only unit checks of the GATE PT-5a ST numeric "
                         "helpers (st_tau_rounds/st_trigger_test/"
                         "st_flip_counts/st_sd_u_window/st_carryover_map/"
                         "st_interp_metrics/st_interp_geneig); no model "
                         "build, no jax use -- runs before setup_model()")
    ap.add_argument("--selftest-pr", action="store_true",
                    help="numpy-only unit checks of the GATE PT-5a-r2 PR "
                         "readiness convergence logic (pr_ladder_snapshot/"
                         "pr_readiness_ready) against the ARCHIVED arm-A "
                         "broad-init probe data (arrays_A_power.npz); no "
                         "model build, no jax use -- runs before "
                         "setup_model()")
    args = ap.parse_args()
    if args.selftest_st:
        st_selftest()
        return
    if args.selftest_pr:
        pr_selftest()
        return
    equiv = args.equiv_check or os.environ.get("GATE_PT0_EQUIV", "0") == "1"

    if equiv or args.arm == "smoke" \
            or os.environ.get("GATE_PT0_SMOKE", "0") == "1":
        apply_smoke()
    os.makedirs(OUT, exist_ok=True)
    if args.arm != "control":   # Arm 0 is analytic: no dPIE model build needed
        setup_model()

    def tag_of(arm):
        # GATE_PT0_TAG_SUFFIX: continuation runs (e.g. PT-0b) tag their outputs
        # separately (B2 -> B2_pt0b) so PT-0 artifacts are never clobbered.
        suffix = os.environ.get("GATE_PT0_TAG_SUFFIX", "").strip()
        if suffix:
            arm = f"{arm}_{suffix.lstrip('_')}"
        return f"{arm}_smoke" if SMOKE else arm

    if equiv:
        if args.arm not in ("control", "B1", "B2", "B3", "B4", "B5",
                            "D1", "D2"):
            ap.error("--equiv-check requires a PT arm (control/B1..B5/D1/D2)")
        if args.arm == "control":
            run_control(tag_of("control"), equiv=True)
        else:
            run_arm_b(args.arm, tag_of(args.arm), equiv=True)
        return

    if args.arm == "smoke":
        run_control(tag_of("control"))          # Arm 0 first (amendment ii)
        run_arm_a("power", tag_of("A_power"))
        run_arm_a("lik", tag_of("A_lik"))
        for arm in ("B1", "B2", "B3", "B4", "B5", "D1", "D2"):
            run_arm_b(arm, tag_of(arm))
    elif args.arm == "control":
        run_control(tag_of("control"))
    elif args.arm == "A_power":
        run_arm_a("power", tag_of("A_power"))
    elif args.arm == "A_lik":
        run_arm_a("lik", tag_of("A_lik"))
    elif args.arm == "ST":
        run_arm_st(tag_of("ST"))
    elif args.arm == "PR":
        run_arm_pr(tag_of("PR"))
    else:
        run_arm_b(args.arm, tag_of(args.arm))


if __name__ == "__main__":
    main()
