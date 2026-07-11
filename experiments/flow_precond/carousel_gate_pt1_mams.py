#!/usr/bin/env python
"""GATE PT-1 C3/C4: MH-exact cross-method bracket on the dPIE carousel.

Design checkpoint: docs/logs/carousel-mclmc-sampling.md "carousel GATE PT-1"
(L3 link). Thin wrapper around the PRODUCTION MAMS
(gigalens_research.inference.mams.MAMS_JIT — adjusted, hence unbiased in law;
MAMS itself untouched) with a qz-adapter (PoolQZ) whose .sample returns
basin-mixture pool draws — the bracket INSTRUMENT — and whose .mean()/
.covariance() are the PRODUCTION SVI artifacts qz_loc / qz_scale_tril@T
(L3-rerun amendment, grader-approved: the original pooled-cov mass seeding made
MAMS ~5x slower than the SVI-seeded baseline and C3/C4 were clipped; the seed
metric is production-like, the init mixture is the instrument). Two arms
initialized from OPPOSITE-side basin mixtures bracket the pocket occupancy:

  C3: --p-pocket 0.25 --seed 22 --tag C3   (main-heavy init)
  C4: --p-pocket 0.75 --seed 23 --tag C4   (pocket-heavy init)

MAMS64 draws are POSITION POOLS only (never weights; human directive); metric
seeding comes from dpie/svi. Outputs land in carousel_gate_pt0_out/ tagged
*_pt1 (names UNCHANGED so the certified scorer runs unmodified).

MAMS_JIT qz usage matched (src/gigalens_research/inference/mams.py):
  line ~76: qz.sample((n_chains,), seed=init_key)   [init positions, jax key]
  line ~83: qz.covariance()                          [starting inverse mass]
  line ~95: qz.mean()                                [svi_mean regularizer]

Run (1 GPU, shifter jax container, float64):
  ... /usr/bin/python3 carousel_gate_pt1_mams.py --p-pocket 0.25 --seed 22 --tag C3
Smoke: add --smoke (n_hmc=8, burnin 100, results 200; same code path).
Probe: add --probe (n_hmc=64, burnin 100, results 100; prints wall + s/step
INCLUDING compile — conservative, grader condition — and writes
probe_<tag>.json; no npz/plot).
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
from gigalens_research.inference.mams import MAMS_JIT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------ config
OUT = os.path.join(HERE, "carousel_gate_pt0_out")
MAMS_NPZ = os.path.join(HERE, "..", "sim_carousel", "messy_tests", "dpie",
                        "mams", "arrays.npz")
SVI_NPZ = os.path.join(HERE, "..", "sim_carousel", "messy_tests", "dpie",
                       "svi", "arrays.npz")   # production seed metric (L3 rerun)
POCKET_COL, POCKET_THR = 6, -22.35
DIM = 33
N_HMC, BURNIN, RESULTS = 64, 2000, 4000     # checkpoint config (C3/C4)
TARGET_ACC = 0.9
THIN = 4                                    # samples_z stored thinned x4
LAST_KEEP = 4000                            # pinned W-3 window: ALL kept draws (rd-2)
JITTER = 1e-3                               # pool-draw init jitter


def pr(*a):
    print(*a, flush=True)


class PoolQZ:
    """qz adapter for MAMS_JIT (interface matched to mams.py lines ~76/83/95).

    .sample(shape, seed) — shape=(n_chains,), seed is a JAX key (MAMS passes
      seed=init_key). The key is folded deterministically to a numpy seed via
      jax.random.key_data(seed)[-1]; draws are per-chain Bernoulli(p_pocket)
      pocket-vs-main pool picks + 1e-3 N(0,1) jitter. Returns jnp float64
      (n_chains, DIM). The realized pocket fraction is recorded.
    .mean() / .covariance() — PRODUCTION SVI artifacts (qz_loc, qz_scale_tril
      @ .T; jnp float64), consumed by MAMS as the svi_mean regularizer /
      starting inverse mass (L3-rerun amendment; the seed metric is NOT the
      bracket instrument — the init mixture is).
    """

    def __init__(self, pool_main, pool_pocket, p_pocket, mean, cov, seed):
        self.pool_M = np.asarray(pool_main, dtype=np.float64)
        self.pool_P = np.asarray(pool_pocket, dtype=np.float64)
        self.p_pocket = float(p_pocket)
        self._mean = jnp.asarray(mean, dtype=jnp.float64)
        self._cov = jnp.asarray(cov, dtype=jnp.float64)
        self._seed = int(seed)
        self.realized_pocket_frac = None

    def sample(self, shape, seed):
        n = int(np.prod(shape))
        folded = int(np.asarray(jax.random.key_data(seed))[-1])
        rng = np.random.default_rng([self._seed, folded])
        pick = rng.random(n) < self.p_pocket
        zm = self.pool_M[rng.integers(0, len(self.pool_M), n)]
        zp = self.pool_P[rng.integers(0, len(self.pool_P), n)]
        z = np.where(pick[:, None], zp, zm)
        z = z + JITTER * rng.standard_normal(z.shape)
        self.realized_pocket_frac = float(pick.mean())
        return jnp.asarray(z.reshape(tuple(shape) + (z.shape[-1],)),
                           dtype=jnp.float64)

    def mean(self):
        return self._mean

    def covariance(self):
        return self._cov


def iat_1d(x):
    """Rough IAT: 1 + 2*sum(rho_t) until rho < 0.05 (same estimator family as
    the PT-0 occ-ESS proxy)."""
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


def main():
    global N_HMC, BURNIN, RESULTS, LAST_KEEP
    ap = argparse.ArgumentParser(
        description="GATE PT-1 MH-exact MAMS bracket arm (C3/C4)")
    ap.add_argument("--p-pocket", type=float, required=True,
                    choices=[0.25, 0.75],
                    help="Bernoulli pocket probability of the init mixture")
    ap.add_argument("--seed", type=int, required=True,
                    help="arm seed (checkpoint: C3=22, C4=23)")
    ap.add_argument("--tag", required=True, help="output tag, e.g. C3 or C4")
    ap.add_argument("--smoke", action="store_true",
                    help="plumbing check: n_hmc=8, burnin 100, results 200 "
                         "(same code path; numbers NOT the measurement)")
    ap.add_argument("--probe", action="store_true",
                    help="wall-time probe: n_hmc=64, burnin 100, results 100; "
                         "prints wall + s/step INCLUDING compile and writes "
                         "probe_<tag>.json; no npz/plot")
    args = ap.parse_args()
    if args.smoke and args.probe:
        ap.error("--smoke and --probe are mutually exclusive")
    if args.smoke:
        N_HMC, BURNIN, RESULTS = 8, 100, 200
        LAST_KEEP = 200   # smoke: all kept
    if args.probe:
        _ps = int(os.environ.get("GATE_PT1_PROBE_STEPS", "100"))  # diagnostic knob
        N_HMC, BURNIN, RESULTS = 64, _ps, _ps
        LAST_KEEP = 100   # probe: all kept (console diagnostics only)
    tag = (f"{args.tag}_smoke" if args.smoke
           else f"{args.tag}_probe" if args.probe else args.tag)
    os.makedirs(OUT, exist_ok=True)
    t0_all = time.time()

    model_seq, prob_model = carousel_model.build()
    assert prob_model.z_param_names[POCKET_COL] == "planes/0/mass/1/center_x"

    mams = np.load(MAMS_NPZ)["samples_z"]
    assert mams.shape == (64, 1000, DIM), mams.shape
    draws = mams.reshape(-1, DIM).astype(np.float64)
    is_pocket = draws[:, POCKET_COL] > POCKET_THR
    pool_M, pool_P = draws[~is_pocket], draws[is_pocket]

    # PRODUCTION-like seed metric (L3-rerun amendment): SVI qz_loc / cov.
    svi = np.load(SVI_NPZ)
    svi_loc = np.asarray(svi["qz_loc"], dtype=np.float64).reshape(-1)
    svi_tril = np.asarray(svi["qz_scale_tril"],
                          dtype=np.float64).reshape(DIM, DIM)
    assert svi_loc.shape == (DIM,), svi_loc.shape
    svi_cov = svi_tril @ svi_tril.T

    qz = PoolQZ(pool_M, pool_P, args.p_pocket, svi_loc, svi_cov, args.seed)

    card = dict(
        script=os.path.abspath(__file__), tag=tag, arm="MAMS MH-exact bracket",
        sampler="gigalens_research.inference.mams.MAMS_JIT (production, "
                "untouched)",
        jax=jax.__version__, devices=[str(d) for d in jax.devices()],
        x64=bool(jax.config.jax_enable_x64), dim=DIM, smoke=bool(args.smoke),
        seed=args.seed, p_pocket=args.p_pocket,
        n_hmc=N_HMC, num_burnin_steps=BURNIN, num_results=RESULTS,
        target_acceptance=TARGET_ACC, thin_stored=THIN, last_keep=LAST_KEEP,
        mams64=os.path.abspath(MAMS_NPZ), svi_npz=os.path.abspath(SVI_NPZ),
        seed_metric="SVI cov (dpie/svi)",
        seed_mean="SVI qz_loc (dpie/svi)",
        pools=dict(n_main=int(len(pool_M)), n_pocket=int(len(pool_P)),
                   use="POSITION pools for INITS ONLY, never weights (human "
                       "directive); seed metric is SVI, L3-rerun amendment"),
        pocket_indicator=f"z[{POCKET_COL}] > {POCKET_THR}",
        qz_adapter="PoolQZ: sample = Bernoulli(p_pocket) pool-mixture draws "
                   "+ 1e-3 jitter (UNCHANGED bracket instrument); "
                   "mean/covariance = SVI qz_loc / qz_scale_tril@T",
        probe=bool(args.probe), jitter=JITTER,
    )
    pr("MODEL CARD:", json.dumps(card, indent=1,
                                 default=lambda o: o.tolist()
                                 if hasattr(o, "tolist") else str(o)))

    t0_mams = time.time()
    samples = MAMS_JIT(model_seq, qz, n_hmc=N_HMC, num_burnin_steps=BURNIN,
                       num_results=RESULTS, target_acceptance=TARGET_ACC,
                       seed=args.seed, debug_output=False)
    samples = np.asarray(samples)               # (N_HMC, RESULTS, DIM)
    wall_mams = time.time() - t0_mams           # INCLUDES compile (conservative)
    assert samples.shape == (N_HMC, RESULTS, DIM), samples.shape
    pr(f"[{tag}] realized init pocket fraction = {qz.realized_pocket_frac} "
       f"(requested Bernoulli p = {args.p_pocket})")

    if args.probe:   # wall-time probe (grader condition): timing only, no
        # npz/plot -- artifact names of the real C3/C4 runs stay untouched
        steps = BURNIN + RESULTS
        probe = dict(wall_s=wall_mams, steps=steps,
                     s_per_step=wall_mams / steps)
        with open(os.path.join(OUT, f"probe_{args.tag}.json"), "w") as f:
            json.dump(probe, f, indent=1)
        pr(f"[{tag}] PROBE: wall {wall_mams:.1f}s over {steps} sampler steps "
           f"(burnin {BURNIN} + results {RESULTS}) = "
           f"{wall_mams / steps:.3f} s/step INCLUDING compile (conservative)")
        pr(f"[{tag}] PROBE written -> probe_{args.tag}.json; total wall "
           f"{time.time() - t0_all:.0f}s")
        return

    ind = samples[:, :, POCKET_COL] > POCKET_THR        # (N_HMC, RESULTS) bool
    occ_chain = ind[:, -LAST_KEEP:].mean(axis=1)        # per-chain, ALL-kept window
    iat_chain = np.array([iat_1d(ind[c, -LAST_KEEP:].astype(np.float64))
                          for c in range(N_HMC)])
    ess_chain = LAST_KEEP / (2.0 * iat_chain)   # IAT proxy — NON-SCORING diagnostic;
    # the PINNED W-3 estimator (moment-matching, p-hat = two-arm pooled mean) needs
    # BOTH arms and is computed by the cross-arm scorer below / pt1_score.
    occ_mean = float(occ_chain.mean())
    se_clust = (float(occ_chain.std(ddof=1) / math.sqrt(N_HMC))
                if N_HMC > 1 else float("inf"))
    # rd-2 A2-mirror drift check: first vs second half of the kept window
    h = RESULTS // 2
    occ_h1, occ_h2 = float(ind[:, :h].mean()), float(ind[:, h:].mean())

    np.savez(
        os.path.join(OUT, f"arrays_{tag}_pt1.npz"),
        samples_z=samples[:, ::THIN].astype(np.float32),   # (N, RESULTS/4, DIM)
        indicator=ind,                                      # full-res bools
        occ_chain=occ_chain, iat_chain=iat_chain, occ_ess_chain=ess_chain,
        occ_mean=occ_mean, se_chain_clustered=se_clust,
        occ_first_half=occ_h1, occ_second_half=occ_h2,
        realized_init_pocket_frac=(np.nan if qz.realized_pocket_frac is None
                                   else qz.realized_pocket_frac),
        p_pocket=args.p_pocket, seed=np.int64(args.seed),
        n_hmc=np.int64(N_HMC), num_burnin_steps=np.int64(BURNIN),
        num_results=np.int64(RESULTS), last_keep=np.int64(LAST_KEEP),
        pocket_col=np.int64(POCKET_COL), pocket_thr=POCKET_THR,
    )

    fig, ax = plt.subplots(figsize=(10, 4.5))
    w = max(1, RESULTS // 200)
    nkeep = RESULTS // w * w
    for c in range(N_HMC):
        sm = ind[c, :nkeep].reshape(-1, w).mean(axis=1)
        ax.plot(np.arange(len(sm)) * w, sm, lw=0.4, alpha=0.4)
    smm = ind[:, :nkeep].mean(axis=0).reshape(-1, w).mean(axis=1)
    ax.plot(np.arange(len(smm)) * w, smm, "k-", lw=2,
            label=f"mean over {N_HMC} chains")
    ax.axvline(RESULTS // 2, color="r", ls=":", lw=1,
               label="drift-check halves (scoring = ALL kept)")
    ax.axhline(args.p_pocket, color="g", ls="--", lw=0.8,
               label=f"init p_pocket {args.p_pocket}")
    ax.set_xlabel("kept draw")
    ax.set_ylabel("pocket occupancy (rolling mean)")
    ax.set_title(f"{tag}: MAMS MH-exact per-chain occupancy "
                 f"(p_pocket={args.p_pocket}, seed={args.seed})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"pt1_mams_{tag}_occ.png"), dpi=120)
    plt.close(fig)

    pr(f"\n[{tag}] ===== MH-EXACT BRACKET ARM (PROPOSED/UNCERTIFIED) =====")
    pr(f"  ALL-kept ({LAST_KEEP}) occupancy = {occ_mean:.4f} +- {se_clust:.4f} "
       f"(chain-clustered se, {N_HMC} chains)")
    pr(f"  drift check (A2 mirror): first-half {occ_h1:.4f} vs second-half "
       f"{occ_h2:.4f} (delta {occ_h2-occ_h1:+.4f})")
    pr(f"  occ-ESS IAT-proxy per chain (NON-SCORING diagnostic): min "
       f"{ess_chain.min():.2f}, median {float(np.median(ess_chain)):.2f}, max "
       f"{ess_chain.max():.2f} (IAT median {float(np.median(iat_chain)):.1f}); "
       f"the SCORING occ-ESS is the pinned two-arm moment-matching estimator "
       f"(p-hat(1-p-hat)/sd_chains^2, p-hat = pooled C3+C4 mean) computed "
       f"cross-arm; UNDERPOWERED gate (occ-ESS < 4) applies to THAT")
    pr(f"  per-chain occ range [{occ_chain.min():.3f}, {occ_chain.max():.3f}]")
    pr(f"  wall {time.time() - t0_all:.0f}s -> {OUT} (arrays_{tag}_pt1.npz, "
       f"pt1_mams_{tag}_occ.png)")


if __name__ == "__main__":
    main()
