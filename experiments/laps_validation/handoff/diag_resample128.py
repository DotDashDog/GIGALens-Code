#!/usr/bin/env python
# coding: utf-8
r"""DC-7.4 validation driver: efficient operation at M=128 chains (with
512-chain controls), building on DC-7.3's resample-rescue mechanism.

Spec: docs/logs/laps_prior_init_investigation.md, "DC-7.4 -- Efficiency +
128-chain validation" section. Claims under test:
  Q1: does the resampled prior-start pipeline (DC-7.3's mid-Phase-2 straggler
      resample, see the "DC-7.3 RESAMPLE step" block in laps_late_adjusted.py)
      still work at M=128 chains (vs the 512-chain DC-7.3 validation)?
  Q2: does CHANGE-B thinned collection (p2_keep_per_chain=4, p2_thin=5)
      recover estimator precision at 128 chains via 4x more posterior draws
      (512 samples from 128 chains) without extra chains?
  Q3 (exploratory): can Phase-1 be trimmed (num_unadjusted_steps 300 -> 200)?
      D1 showed the ensemble reaches the plateau band by ~step 200 of Phase 1,
      so arrivals mostly happen in Phase 2 regardless.

Arms (all init via LAPS_late_adjusted_JIT, early_stop=False, track_chains=True,
num_adjusted_steps=248 unless noted; p2_resample_at_chunk=13 = T2a, per DC-7.3's
D1-derived ~30% core fraction by chunk 13):
  R128a/b/c  prior-init, num_chains=128, seeds 0/1/2, p2_resample_at_chunk=13,
             p2_resample_min_survivors=24 (128-chain guard: D1 core fraction
             ~30% at chunk 13 predicts ~38 survivors; 24 = floor at which a
             diag-Var metric estimate has ~15%/dim relative error --
             acceptable; if survivors < 24 the guard skips and the arm fails
             informatively).
  W128       warm-init (SVI surrogate), num_chains=128, seed 0, same budget
             (small-M yardstick; no resample kwargs -- the lever is a no-op
             at defaults).
  R128k      as R128a plus p2_keep_per_chain=4, p2_thin=5 (CHANGE-B: 512
             samples from 128 chains via thinned post-freeze collection;
             res.samples is (128, 4, dim) -- width/offset stats use the
             FLATTENED (512, dim) view; split-R-hat + a crude ESS are
             additionally reported over the (128 chains, 4 samples) shape,
             since a thinned within-chain series this short cannot be treated
             as 512 independent draws without a decorrelation check).
  Rtrim      prior-init, num_chains=512, num_unadjusted_steps=200 (Phase-1
             trim probe, Q3), p2_resample_at_chunk=13, default min_survivors
             (32) -- 512 chains comfortably clear the guard.
  A0r128     (v2 amendment) prior-init, num_chains=128, seed 0, resampling
             OFF (no resample kwargs) -- same-M negative control: shows the
             mixture the resampler is meant to fix, at the SAME T2 budget and
             the same M=128 as the R arms.

Adjudication bands (derived, from the spec; this driver reports numbers/plots
only, no verdict -- adjudication happens elsewhere per the pre-registration).
At M=128 the width-estimate noise dominates: with n_eff ~ 38 ancestors,
std-ratio SE ~ 1/sqrt(2*37) = 0.117; with only 128 final samples the estimate
itself adds SE ~ 1/sqrt(2*127) = 0.063. SUCCESS(R128) = every mass-param ratio
within W128's envelope widened by 2*SE_combined (~0.27): ratios in [0.7, 1.5]
AND |offset| <= 0.5 sigma in >= 2/3 seeds, AND corner overlay viewed with no
visible offset/shape anomaly (load-bearing at this M). SUCCESS(R128k) = same
bands with the sample-noise term dropped (512 samples): ratios in
[0.75, 1.45]. FALSIFIED = any ratio outside [0.6, 1.9] or |offset| > 0.8 sigma
in 2+ seeds, or guard-skip (survivors < 24) in 2+ seeds. Rtrim is exploratory
(Q3): report against the DC-7.3 bands, no hard falsifier.

v2 PRIMARY quantitative gate at M=128 (arm-to-arm, cancels the shared
HMC-denominator and small-M estimation structure): for each of R128a/b/c and
R128k, the per-param ratio std(arm mass)/std(W128 mass) -- summary fields
"ratio_vs_w128" (list of 8) + "ratio_vs_w128_min"/"_max"; bands [0.73, 1.27]
CONDITIONAL on dup_decorr saturation. R128k uses its flattened (512, dim)
samples, same as its other width stats.

Outputs (diag_resample128/): per-arm mass/z samples, summary.json (std ratios,
mean offsets in HMC sigma, core fraction, final-ensemble logp, Phase-2
eps/accept, resample_info scalars, n_unique_ancestors / ratio_noise_bar_neff /
dup_decorr_saturated for the resample arms, R128k's extra split-R-hat/ESS
block, ratio_vs_w128 gate fields for R128a/b/c + R128k), the R128a/b/c +
Rtrim duplicate-decorrelation curves (dup_decorr128.png; n/a for A0r128 --
no resample, no pairs), corner overlays (corner128_full.png /
corner128_zoom.png, HMC black + R128a (C3) + W128 (C0) + R128k (C4) +
Rtrim (C5) + A0r128 (C1)), and the Phase-2 eps trajectory (eps128.png,
R128a vs W128 vs Rtrim). PRODUCES NUMBERS AND PLOTS, no verdict. Model
build + HMC-ref load + to_mass/MASS_ORDER verbatim from
``diag_resample.py`` / ``diag_levers.py``.
"""
import os, json
import jax
jax.config.update("jax_enable_x64", True)
from gigalens.jax.inference import ModellingSequence
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.scene_prob_model import Dataset, ProbModel
from gigalens.simulator import SimulatorConfig
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.mass import epl, shear
import tensorflow_probability.substrates.jax as tfp
import numpy as np, optax
from jax import numpy as jnp
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from corner import corner
tfd = tfp.distributions
from gigalens_research.inference.laps_late_adjusted import LAPS_late_adjusted_JIT
print("jax devices:", jax.devices())

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "diag_resample128"); os.makedirs(OUT, exist_ok=True)

epl_p = dict(theta_E=tfd.LogNormal(jnp.log(1.25), 0.25), gamma=tfd.TruncatedNormal(2, 0.25, 1, 3),
    e1=tfd.Normal(0, 0.1), e2=tfd.Normal(0, 0.1), center_x=tfd.Normal(0, 0.05), center_y=tfd.Normal(0, 0.05))
shear_p = dict(gamma1=tfd.Normal(0, 0.05), gamma2=tfd.Normal(0, 0.05))
lens_light_p = dict(R_sersic=tfd.LogNormal(jnp.log(1.0), 0.15), n_sersic=tfd.Uniform(2, 6),
    e1=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3), e2=tfd.TruncatedNormal(0, 0.1, -0.3, 0.3),
    center_x=tfd.Normal(0, 0.05), center_y=tfd.Normal(0, 0.05), Ie=tfd.LogNormal(jnp.log(500.0), 0.3))
source_light_p = dict(R_sersic=tfd.LogNormal(jnp.log(0.25), 0.15), n_sersic=tfd.Uniform(0.5, 4),
    e1=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5), e2=tfd.TruncatedNormal(0, 0.15, -0.5, 0.5),
    center_x=tfd.Normal(0, 0.25), center_y=tfd.Normal(0, 0.25), Ie=tfd.LogNormal(jnp.log(150.0), 0.5))

ASSETS = "/global/u1/l/linusu/gigalens/src/gigalens/assets"
kernel = np.load(f"{ASSETS}/psf.npy").astype(np.float32)
sim_config = SimulatorConfig(delta_pix=0.065, num_pix=60, supersample=2, kernel=kernel)
lens_light = Component(sersic.SersicEllipse(use_lstsq=False), lens_light_p)
source_light = Component(sersic.SersicEllipse(use_lstsq=False), source_light_p)
model = LensModel([
    Plane(mass=[Component(epl.EPL(50), epl_p), Component(shear.Shear(), shear_p)], light=[lens_light]),
    Plane(deflection_ratio=1.0, light=[source_light])])
observed_img = np.load(f"{ASSETS}/demo.npy")
ds = Dataset(jnp.asarray(observed_img), sim_config, background_rms=0.2, exp_time=100, sees="all")
prob_model = ProbModel(model, ds, mode="forward")
model_seq = ModellingSequence(prob_model)
DIM = int(model.num_free_params); print("dim:", DIM)
opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
best, best_lp, best_chisq = model_seq.MAP(opt, seed=0)
print("MAP best_chisq (min):", float(np.min(np.asarray(best_chisq))))
opt = optax.adabelief(1e-4, b1=0.95, b2=0.99)
qz, _ = model_seq.SVI(best, opt, n_vi=1000, num_steps=1500); print("SVI done.", flush=True)

# ---- HMC reference + truth markers (from the already-computed artifacts) ----
MASS_ORDER = [
    "planes/0/mass/0/theta_E", "planes/0/mass/0/gamma",
    "planes/0/mass/0/e1", "planes/0/mass/0/e2",
    "planes/0/mass/0/center_y", "planes/0/mass/0/center_x",
    "planes/0/mass/1/gamma2", "planes/0/mass/1/gamma1",
]
labels = [r'$\theta_E$', r'$\gamma$', r'$\epsilon_2$', r'$\epsilon_1$',
          r'$y$', r'$x$', r'$\gamma_{2,ext}$', r'$\gamma_{1,ext}$']
hmc_mass = np.load(os.path.join(HERE, "hmc_ref", "hmc_mass.npy"))
markers = json.load(open(os.path.join(HERE, "hmc_ref", "overlay_summary.json")))["truth"]
hs = hmc_mass.std(0)
hm = hmc_mass.mean(0)
print(f"HMC ref: {hmc_mass.shape}, std={hs}")


def to_mass(samples_z):
    """Flattens any leading (chain, [collection], ...) shape to (-1, dim) and
    maps through the bijector to physical mass params, ordered MASS_ORDER."""
    smp = np.asarray(samples_z).reshape((-1, DIM))
    phys = prob_model.bij.forward(list(jnp.asarray(smp).T))
    return np.stack([np.asarray(phys[k]) for k in MASS_ORDER], axis=1)


def to_mass_by_chain(samples_z):
    """Same map as to_mass, but PRESERVES the leading shape (e.g. (M, keep,
    dim) -> (M, keep, n_mass)) instead of flattening -- needed for the R128k
    split-R-hat/ESS block, which needs per-chain structure."""
    arr = np.asarray(samples_z)
    lead_shape = arr.shape[:-1]
    flat = arr.reshape((-1, DIM))
    phys = prob_model.bij.forward(list(jnp.asarray(flat).T))
    mass_flat = np.stack([np.asarray(phys[k]) for k in MASS_ORDER], axis=1)
    return mass_flat.reshape(lead_shape + (mass_flat.shape[-1],))


logp_one = jax.jit(jax.vmap(lambda z: prob_model.log_prob(z)[0]))

# DC-7.3/7.4 default survivor cut (mirrors the sampler's own p2_resample_dlogp
# default: dim/2 + 4*sqrt(dim/2)); reused here ONLY to define "final-core"
# membership for the duplicate-decorrelation SATURATION reference level. This
# is independent of num_chains (a function of DIM only), so it is valid
# across the mixed 128/512-chain arms in this driver.
DEFAULT_DLOGP = DIM / 2.0 + 4.0 * np.sqrt(DIM / 2.0)

NUM_ADJ = 248          # T2a=13 + T2b=18 chunks @ p2_chunk_size=8 default
T2A_CHUNK = 13
MIN_SURV_128 = 24

ARMS = {
    "R128a": dict(init_mode="prior", num_chains=128, seed=0,
                  num_adjusted_steps=NUM_ADJ, p2_resample_at_chunk=T2A_CHUNK,
                  p2_resample_min_survivors=MIN_SURV_128),
    "R128b": dict(init_mode="prior", num_chains=128, seed=1,
                  num_adjusted_steps=NUM_ADJ, p2_resample_at_chunk=T2A_CHUNK,
                  p2_resample_min_survivors=MIN_SURV_128),
    "R128c": dict(init_mode="prior", num_chains=128, seed=2,
                  num_adjusted_steps=NUM_ADJ, p2_resample_at_chunk=T2A_CHUNK,
                  p2_resample_min_survivors=MIN_SURV_128),
    "W128":  dict(init_mode="warm", num_chains=128, seed=0,
                  num_adjusted_steps=NUM_ADJ),
    "R128k": dict(init_mode="prior", num_chains=128, seed=0,
                  num_adjusted_steps=NUM_ADJ, p2_resample_at_chunk=T2A_CHUNK,
                  p2_resample_min_survivors=MIN_SURV_128,
                  p2_keep_per_chain=4, p2_thin=5),
    "Rtrim": dict(init_mode="prior", num_chains=512, seed=0,
                  num_unadjusted_steps=200, num_adjusted_steps=NUM_ADJ,
                  p2_resample_at_chunk=T2A_CHUNK),
    # v2 amendment: same-M negative control -- prior-init, resampling OFF
    # (no resample kwargs), same T2 budget. Not in DUP_ARMS (no resample ->
    # no straggler/donor pairs, dup-decorr n/a).
    "A0r128": dict(init_mode="prior", num_chains=128, seed=0,
                   num_adjusted_steps=NUM_ADJ),
}
DUP_ARMS = ["R128a", "R128b", "R128c", "Rtrim"]
# v2 gate arms: per-param std(arm)/std(W128) -- the PRIMARY quantitative gate
# at M=128 (bands [0.73, 1.27] conditional on dup_decorr saturation).
VS_W128_ARMS = ["R128a", "R128b", "R128c", "R128k"]
CORNER_ARMS = ["R128a", "W128", "R128k", "Rtrim", "A0r128"]
CORNER_COLORS = ["C3", "C0", "C4", "C5", "C1"]
CORNER_LABELS = ["R128a resampled prior (M=128)", "W128 warm-init (M=128)",
                 "R128k resampled+thinned (M=128, keep=4)",
                 "Rtrim prior, Phase-1 trimmed (M=512)",
                 "A0r128 prior, no resample (M=128)"]
EPS_ARMS = ["R128a", "W128", "Rtrim"]
EPS_COLORS = ["C3", "C0", "C5"]


def n_unique_ancestors(res, num_chains):
    """Distinct chains contributing to the final ensemble (same rule as
    DC-7.3, parameterized by num_chains since arms here mix M=128 and 512).
    Only an ACTIVE "replace" resample collapses ancestry (stragglers inherit
    donors' states): survivors keep themselves, stragglers map to donors, so
    n_unique = |survivors UNION donors|. A skipped / absent resample (or W128,
    which never sets p2_resample_at_chunk) replaces nothing -> all distinct."""
    info = res.resample_info
    if (info is None or bool(info.get("skipped", True))
            or info.get("mode", "replace") != "replace"):
        return num_chains
    strag = np.asarray(info["stragglers"])
    donors = np.asarray(info["donors"])
    survivors = set(range(num_chains)) - set(strag.tolist())
    return len(survivors | set(donors.tolist()))


def rms_pairwise(X):
    """RMS pairwise Euclidean distance among rows of X (n, d); nan if n < 2."""
    n = X.shape[0]
    if n < 2:
        return float("nan")
    diffs = X[:, None, :] - X[None, :, :]
    d2 = np.sum(diffs ** 2, axis=-1)
    iu = np.triu_indices(n, k=1)
    return float(np.sqrt(np.mean(d2[iu])))


def dup_decorr_curve(res):
    """Per-snapshot RMS distance between each (straggler, donor) pair, from
    the post-resample snapshot (snap_index, where the curve is exactly 0 by
    construction -- duplicated states) to the LAST recorded Phase-2 snapshot.
    Returns None if the arm's resample was skipped (no pairs). NOTE (R128k):
    p2_keep_per_chain>1 collection sub-chunks are NOT snapshotted, so this
    curve (adaptation-phase chain_traj only) is unaffected by CHANGE-B; R128k
    is not in DUP_ARMS so this distinction doesn't otherwise arise here."""
    info = res.resample_info
    if info is None or bool(info.get("skipped", True)):
        return None
    strag = np.asarray(info["stragglers"])
    donors = np.asarray(info["donors"])
    snap = int(info["snap_index"])
    pos = np.asarray(res.chain_traj["p2_pos"])          # (n_ck2, M, dim)
    sub = pos[snap:]                                    # (n_off, M, dim)
    diffs = sub[:, strag, :] - sub[:, donors, :]         # (n_off, n_pairs, dim)
    rms = np.sqrt(np.mean(np.sum(diffs ** 2, axis=-1), axis=1))  # (n_off,)
    return rms


def saturation_value(res):
    """RMS pairwise distance among the FINAL ensemble's core chains (logp >
    max - DEFAULT_DLOGP on the last Phase-2 snapshot) -- the intrinsic
    ensemble-spread scale the duplicate-decorrelation curve is expected to
    rise to and saturate at."""
    lp_final = np.asarray(res.chain_traj["p2_logp"][-1])
    cut = float(np.nanmax(lp_final)) - DEFAULT_DLOGP
    core_idx = np.where(lp_final > cut)[0]
    pos_final = np.asarray(res.chain_traj["p2_pos"][-1])[core_idx]
    return rms_pairwise(pos_final)


def rhat_ess_128x4(mass_by_chain):
    """Split-R-hat + a CRUDE ESS over an (M chains, K draws-per-chain, n_mass)
    per-mass-param array (R128k's thinned CHANGE-B collection: M=128, K=4).

    R-hat: the standard multi-chain Gelman-Rubin formula, applied DIRECTLY to
    the M=128 chains x K=4 draws (m=128, n=4) rather than to split half-
    chains -- K=4 is too short to split further, so this is the un-split
    variant (plain R-hat = sqrt((W*(n-1)/n + B/n)/W)) rather than the
    rank-normalized/split estimator.
        chain_mean_j, chain_var_j (ddof=1) per chain j
        W        = mean_j(chain_var_j)                       (within-chain)
        B/n      = var_j(chain_mean_j, ddof=1)                (between-chain)
        var_hat  = (n-1)/n * W + B/n
        Rhat     = sqrt(var_hat / W)

    ESS: CRUDE, not the autocorrelation/rank-normalized estimator (K=4 is far
    too short to estimate an autocorrelation function) -- uses the standard
    "variance-inflation" heuristic implied by R-hat itself:
        ess = M*K / Rhat^2   (= M*K * W/var_hat)
    This is a quick approximation, not a rigorous ESS; treat as indicative
    only, flagged as such in the summary.

    Returns (rhat_list, rhat_max, ess_list, ess_min) each length n_mass.
    """
    M, K, P = mass_by_chain.shape
    chain_mean = mass_by_chain.mean(axis=1)                  # (M, P)
    chain_var = mass_by_chain.var(axis=1, ddof=1)            # (M, P)
    W = chain_var.mean(axis=0)                                # (P,)
    B_over_n = chain_mean.var(axis=0, ddof=1)                 # (P,)
    var_hat = (K - 1) / K * W + B_over_n
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(var_hat / W)
        ess = M * K * (W / var_hat)
    rhat = np.where(np.isfinite(rhat), rhat, np.nan)
    ess = np.where(np.isfinite(ess), ess, np.nan)
    return rhat.tolist(), float(np.nanmax(rhat)), ess.tolist(), float(np.nanmin(ess))


results, summary = {}, {}
for name, kw in ARMS.items():
    n_chains_arm = int(kw["num_chains"])
    print(f"\n=== {name}: num_chains={n_chains_arm}, "
          f"num_adjusted_steps={kw.get('num_adjusted_steps', NUM_ADJ)}, "
          f"early_stop=False, track_chains=True, {kw} ===", flush=True)
    res = LAPS_late_adjusted_JIT(
        model_seq, qz, early_stop=False, track_chains=True, **kw)

    mass = to_mass(res.samples)   # flattens (M,[keep],dim) -> (M*keep, n_mass)
    smp_flat = np.asarray(res.samples).reshape((-1, DIM))
    np.save(os.path.join(OUT, f"{name}_mass.npy"), mass)
    np.save(os.path.join(OUT, f"{name}_samples_z.npy"), smp_flat)

    ratio = mass.std(0) / hs
    offset = (mass.mean(0) - hm) / hs
    in_box = np.all(np.abs(mass - hm) <= 6 * hs, axis=1)
    core_frac = float(in_box.mean())

    lp = np.asarray(logp_one(jnp.asarray(smp_flat)))

    info = res.resample_info
    resample_summary = None
    if info is not None:
        resample_summary = dict(
            chunk=int(info["chunk"]), skipped=bool(info["skipped"]),
            mode=info.get("mode"),
            n_survivors=int(info["n_survivors"]),
            n_stragglers=int(info["n_stragglers"]),
            cut=float(info["cut"]),
            eps0_rs=(float(info["eps0_rs"]) if "eps0_rs" in info else None),
        )

    # v2 grader gate (carried over from DC-7.3): n_eff-adjusted reporting --
    # duplicate-induced ancestry collapse means the R arms' std ratios carry
    # MORE Monte-Carlo noise than n_chains_arm independent chains would imply.
    n_unique = n_unique_ancestors(res, n_chains_arm)
    ratio_noise = 1.0 / np.sqrt(2.0 * (n_unique - 1))

    results[name] = dict(mass=mass, res=res)
    summary[name] = dict(
        num_chains=n_chains_arm,
        kwargs={k: str(v) for k, v in kw.items()},
        ratio=ratio.tolist(), ratio_median=float(np.median(ratio)),
        ratio_min=float(np.min(ratio)), ratio_max=float(np.max(ratio)),
        offset=offset.tolist(), offset_max_abs=float(np.max(np.abs(offset))),
        core_frac=core_frac,
        logp_median=float(np.median(lp)), logp_max=float(np.max(lp)),
        p2_final_step_size=float(res.p2_final_step_size),
        p2_accept_last=float(np.asarray(res.p2_accept)[-1]),
        phase1_len=int(res.phase1_len), switched=bool(res.switched),
        resample_info=resample_summary,
        n_unique_ancestors=int(n_unique),
        ratio_noise_bar_neff=float(ratio_noise),
    )
    s = summary[name]
    n_surv_print = resample_summary["n_survivors"] if resample_summary is not None else None
    print(f"[{name}] ratio med/min/max={s['ratio_median']:.3f}/{s['ratio_min']:.3f}/"
          f"{s['ratio_max']:.3f}  |offset|max={s['offset_max_abs']:.3f}  "
          f"core_frac={s['core_frac']:.3f}  n_survivors={n_surv_print}  "
          f"eps={s['p2_final_step_size']:.3e}  accept={s['p2_accept_last']:.3f}  "
          f"n_unique={n_unique} (ratio_noise~{ratio_noise:.3f})  "
          f"resample={resample_summary}", flush=True)

# --------------------------------------------------------------------------- #
# R128k EXTRA: split-R-hat + crude ESS over the (128 chains, 4 draws) array   #
# (only meaningful/available shape when p2_keep_per_chain > 1)                #
# --------------------------------------------------------------------------- #
r128k_res = results["R128k"]["res"]
mass_by_chain = to_mass_by_chain(r128k_res.samples)   # (128, 4, n_mass)
print(f"R128k res.samples shape: {np.asarray(r128k_res.samples).shape}  "
      f"(mass_by_chain: {mass_by_chain.shape})", flush=True)
rhat_list, rhat_max, ess_list, ess_min = rhat_ess_128x4(mass_by_chain)
summary["R128k"]["rhat_128x4"] = dict(
    labels=labels, rhat_per_param=rhat_list, rhat_max=rhat_max,
    ess_per_param=ess_list, ess_min=ess_min,
    note=("Split-R-hat (un-split variant, m=128 chains x n=4 draws; K=4 is "
          "too short to split further) and a CRUDE ESS (= M*K/Rhat^2, a "
          "variance-inflation heuristic, NOT the autocorrelation/rank-"
          "normalized estimator) over the CHANGE-B thinned collection "
          "(p2_keep_per_chain=4, p2_thin=5). Reported as a complement to the "
          "flattened (512, dim) width/offset stats above, since within-chain "
          "thinned samples may still be correlated at K=4."),
)
print(f"[R128k] split-Rhat (m=128,n=4): max={rhat_max:.4f}  "
      f"per-param={['%.4f' % r for r in rhat_list]}", flush=True)
print(f"[R128k] crude ESS (=M*K/Rhat^2): min={ess_min:.1f}  "
      f"per-param={['%.1f' % e for e in ess_list]}", flush=True)

# --------------------------------------------------------------------------- #
# v2 PRIMARY gate at M=128: arm-to-arm per-param width ratio vs W128          #
# std(arm mass)/std(W128 mass); bands [0.73, 1.27] CONDITIONAL on dup_decorr  #
# saturation (adjudicated elsewhere -- reported here, no verdict). For R128k  #
# the arm std comes from its flattened (512, dim)->(512, n_mass) samples,     #
# same array as its other width stats.                                        #
# --------------------------------------------------------------------------- #
w128_std = results["W128"]["mass"].std(0)
for name in VS_W128_ARMS:
    r_w = results[name]["mass"].std(0) / w128_std
    summary[name]["ratio_vs_w128"] = r_w.tolist()
    summary[name]["ratio_vs_w128_min"] = float(np.min(r_w))
    summary[name]["ratio_vs_w128_max"] = float(np.max(r_w))
    print(f"[{name}]", flush=True)
    print(f"   vs_W128: min={float(np.min(r_w)):.3f} max={float(np.max(r_w)):.3f} "
          f"{['%.3f' % v for v in r_w]}", flush=True)

# --------------------------------------------------------------------------- #
# R128a/b/c + Rtrim duplicate-decorrelation curves + saturation reference     #
# --------------------------------------------------------------------------- #
dup_curves = {}
saturations = {}
for name in DUP_ARMS:
    curve = dup_decorr_curve(results[name]["res"])
    sat = saturation_value(results[name]["res"])
    dup_curves[name] = curve
    saturations[name] = sat
    if curve is not None:
        np.save(os.path.join(OUT, f"{name}_dup_decorr_curve.npy"), curve)
    summary[name]["dup_decorr_saturation"] = sat
    # v2 grader gate: has the duplicate-separation curve actually risen to the
    # core's intrinsic spread (>= 80% of the saturation reference)?
    saturated = (curve is not None and np.isfinite(sat)
                 and float(curve[-1]) >= 0.8 * sat)
    summary[name]["dup_decorr_saturated"] = bool(saturated)
    print(f"[{name}] dup_decorr: "
          f"{'skipped (no pairs)' if curve is None else f'{len(curve)} pts, final={curve[-1]:.4g}'}"
          f"  saturation_ref={sat:.4g}  saturated={saturated}", flush=True)

fig, ax = plt.subplots(figsize=(7, 5))
colors_dup = {"R128a": "C3", "R128b": "C4", "R128c": "C5", "Rtrim": "C6"}
for name in DUP_ARMS:
    curve = dup_curves[name]
    if curve is None:
        continue
    c = colors_dup[name]
    ax.plot(np.arange(len(curve)), curve, color=c, lw=1.5, label=f"{name} straggler-donor rms")
    ax.axhline(saturations[name], color=c, ls=":", lw=1.0,
               label=f"{name} final-core saturation")
ax.set_xlabel("Phase-2 chunks since resample")
ax.set_ylabel("rms distance (unconstrained z)")
ax.set_title("DC-7.4 duplicate decorrelation: R128a/b/c (M=128) + Rtrim (M=512)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "dup_decorr128.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved dup_decorr128.png", flush=True)

# --------------------------------------------------------------------------- #
# corner overlays: shared-range AND HMC-zoomed                                #
# HMC black + R128a (C3) + W128 (C0) + R128k (C4) + Rtrim (C5) + A0r128 (C1)  #
# --------------------------------------------------------------------------- #
for tag, rng in (
    ("full", None),
    ("zoom", [(m - 6 * s, m + 6 * s) for m, s in zip(hmc_mass.mean(0), hs)]),
):
    if rng is None:
        allcat = np.concatenate([hmc_mass] + [results[a]["mass"] for a in CORNER_ARMS])
        lo, hi = allcat.min(0), allcat.max(0)
        pad = np.where((hi - lo) > 0, 0.05 * (hi - lo), 1e-6)
        rng = list(zip(lo - pad, hi + pad))
    ckw = dict(labels=labels, range=rng, bins=30, plot_datapoints=False,
               smooth=1.0, hist_kwargs=dict(density=True))
    fig = corner(hmc_mass, color="black", truths=markers, truth_color="green", **ckw)
    for a, col in zip(CORNER_ARMS, CORNER_COLORS):
        m = results[a]["mass"]
        clip = np.all([(m[:, i] >= rng[i][0]) & (m[:, i] <= rng[i][1])
                       for i in range(len(rng))], axis=0)
        if clip.sum() >= 10:
            corner(m[clip], color=col, fig=fig, **ckw)
        else:
            print(f"(zoom) {a}: only {int(clip.sum())} samples inside HMC±6σ — "
                  f"omitted from zoom corner", flush=True)
    handles = ([Line2D([0], [0], color="black", lw=2, label="HMC reference")] +
               [Line2D([0], [0], color=c, lw=2, label=l)
                for c, l in zip(CORNER_COLORS, CORNER_LABELS)] +
               [Line2D([0], [0], color="green", lw=2, label="truth")])
    fig.legend(handles=handles, loc="upper right", fontsize=11, frameon=False)
    fig.suptitle(f"DC-7.4 M=128 validation ({tag}): "
                 f"HMC vs R128a/W128/R128k/Rtrim/A0r128", fontsize=13)
    fig.savefig(os.path.join(OUT, f"corner128_{tag}.png"), dpi=110,
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved corner128_{tag}.png", flush=True)

# --------------------------------------------------------------------------- #
# eps trajectory: R128a vs W128 vs Rtrim (log-y), re-tune jump                #
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(8, 5))
for a, col in zip(EPS_ARMS, EPS_COLORS):
    ss = np.asarray(results[a]["res"].p2_step_size)
    ax.plot(np.arange(len(ss)), ss, color=col, lw=1.2, label=a)
r128a_info = results["R128a"]["res"].resample_info
if r128a_info is not None and not bool(r128a_info.get("skipped", True)):
    resample_step = int(r128a_info["chunk"]) * 8    # p2_chunk_size default
    ax.axvline(resample_step, color="k", ls="--", lw=1.0, label="R128a resample")
ax.set_yscale("log")
ax.set_xlabel("Phase-2 step")
ax.set_ylabel("eps (step size)")
ax.set_title("DC-7.4: Phase-2 step-size trajectory, R128a (M=128) vs W128 (M=128) vs Rtrim (M=512)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "eps128.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved eps128.png", flush=True)

# --------------------------------------------------------------------------- #
# summary.json                                                                #
# --------------------------------------------------------------------------- #
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
print("\nSUMMARY:", json.dumps(summary, indent=2))
print("DIAG DONE", flush=True)
