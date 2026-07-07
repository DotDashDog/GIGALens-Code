#!/usr/bin/env python
# coding: utf-8
r"""DC-7.3 validation driver: does mid-Phase-2 straggler resampling rescue
prior-seeded LAPS on the demo lens to the warm-init class?

Mechanism under test (flag-gated, default off; see the "DC-7.3 RESAMPLE step"
block in ``laps_late_adjusted.py`` and the DC-7.3 section of
``docs/logs/laps_prior_init_investigation.md``): after T2a=13 Phase-2
adaptation chunks, straggler chains (logp <= ensemble max - dlogp) have their
STATE replaced by a uniform-with-replacement draw from survivor states, the
diagonal preconditioner + eps bisection are rebuilt from the resampled
ensemble, and the remaining T2b=18 chunks run to a fresh freeze. Pre-resample
states are burn-in (never averaged); validity = MH re-initialization, the
same class as warm-init (SVI-surrogate) init, which is CERTIFIED at
0.92-1.04x vs HMC.

Arms (512 chains, 300 unadj + 248 adj (T2a=13 + T2b=18 chunks @ p2_chunk_size=8
default) ~= 248 total adjusted traj vs the 200-traj baseline; early_stop=False,
track_chains=True throughout):
  R1/R1b/R1c  resampled prior-init, seeds 0/1/2, p2_resample_at_chunk=13
              (T2a, per DC-7.3's D1-derived ~30% core fraction by chunk 13).
  M1          prior-init, seed 0, p2_resample_at_chunk=13,
              p2_resample_mode="retune_only" (v2 control: keeps ALL positions,
              rebuilds metric+eps from the SURVIVOR subset only -- isolates
              the poisoned-metric fix from position replacement, i.e. the
              pre-committed "retuned adjusted phase" alternative).
  W           warm-init (SVI surrogate), seed 0, same code+budget (fresh
              same-day yardstick; no resample kwargs -- the lever is a no-op
              at defaults).
  A0r         prior-init, resampling OFF, seed 0 (same-code control: shows
              the mixture the resampler is meant to fix, at the SAME T2
              budget as the R arms).

n_eff-adjusted reporting (v2 grader gate): each arm's summary carries
"n_unique_ancestors" (distinct chains contributing to the final ensemble,
= |survivors UNION donors| from the resample map; survivors keep themselves,
stragglers map to donors -- 512 for W/A0r/M1 and skipped resamples) and
"ratio_noise_bar_neff" = 1/sqrt(2*(n_unique-1)), the Gaussian std SE at the
effective ancestor count. R arms additionally carry "dup_decorr_saturated"
(final decorrelation value >= 0.8x the final-core saturation reference).

Outputs (diag_resample/): per-arm mass/z samples, summary.json (std ratios,
mean offsets in HMC sigma, core fraction, final-ensemble logp, Phase-2
eps/accept, resample_info scalars, n_eff block), the R-arm duplicate-
decorrelation curves (dup_decorr.png), corner overlays
(resample_corner_full.png / _zoom.png, HMC + R1/M1/W/A0r), and the Phase-2
eps trajectory (resample_eps.png, R1 vs M1 vs W vs A0r, showing the
post-resample re-tune jump). PRODUCES NUMBERS AND PLOTS, no verdict
(adjudication happens elsewhere per the pre-registration triplet in the
DC-7.3 log section). Model build + HMC-ref load + to_mass/MASS_ORDER
verbatim from ``diag_levers.py``.
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
OUT = os.path.join(HERE, "diag_resample"); os.makedirs(OUT, exist_ok=True)

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
    smp = np.asarray(samples_z).reshape((-1, DIM))
    phys = prob_model.bij.forward(list(jnp.asarray(smp).T))
    return np.stack([np.asarray(phys[k]) for k in MASS_ORDER], axis=1)


logp_one = jax.jit(jax.vmap(lambda z: prob_model.log_prob(z)[0]))

# DC-7.3 default survivor cut (mirrors the sampler's own p2_resample_dlogp
# default: dim/2 + 4*sqrt(dim/2)); reused here ONLY to define "final-core"
# membership for the duplicate-decorrelation SATURATION reference level
# (judgment call: no separate core definition is specified by the design doc,
# so we reuse the resampler's own self-consistent rule on the FINAL ensemble).
DEFAULT_DLOGP = DIM / 2.0 + 4.0 * np.sqrt(DIM / 2.0)

NUM_CHAINS = 512
NUM_UNADJ = 300
NUM_ADJ = 248          # T2a=13 + T2b=18 chunks @ p2_chunk_size=8 (default)
T2A_CHUNK = 13

ARMS = {
    "R1":   dict(init_mode="prior", seed=0, p2_resample_at_chunk=T2A_CHUNK),
    "R1b":  dict(init_mode="prior", seed=1, p2_resample_at_chunk=T2A_CHUNK),
    "R1c":  dict(init_mode="prior", seed=2, p2_resample_at_chunk=T2A_CHUNK),
    "M1":   dict(init_mode="prior", seed=0, p2_resample_at_chunk=T2A_CHUNK,
                 p2_resample_mode="retune_only"),
    "W":    dict(init_mode="warm", seed=0),
    "A0r":  dict(init_mode="prior", seed=0),
}
R_ARMS = ["R1", "R1b", "R1c"]


def n_unique_ancestors(res):
    """Distinct chains contributing to the final ensemble. Only an ACTIVE
    "replace" resample collapses ancestry (stragglers inherit their donors'
    states): survivors keep themselves, stragglers map to donors, so
    n_unique = |survivors UNION donors|. retune_only keeps all positions and
    a skipped / absent resample replaces nothing -> all chains distinct."""
    info = res.resample_info
    if (info is None or bool(info.get("skipped", True))
            or info.get("mode", "replace") != "replace"):
        return NUM_CHAINS
    strag = np.asarray(info["stragglers"])
    donors = np.asarray(info["donors"])
    survivors = set(range(NUM_CHAINS)) - set(strag.tolist())
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
    construction -- duplicated states) to the LAST recorded Phase-2
    snapshot. Returns None if the arm's resample was skipped (no pairs)."""
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


results, summary = {}, {}
for name, kw in ARMS.items():
    print(f"\n=== {name}: 512 chains, {NUM_UNADJ} unadj + {NUM_ADJ} adj, "
          f"early_stop=False, track_chains=True, {kw} ===", flush=True)
    res = LAPS_late_adjusted_JIT(
        model_seq, qz, num_chains=NUM_CHAINS, early_stop=False,
        track_chains=True, num_unadjusted_steps=NUM_UNADJ,
        num_adjusted_steps=NUM_ADJ, **kw)
    smp = np.asarray(res.samples).reshape((-1, DIM))
    mass = to_mass(smp)
    np.save(os.path.join(OUT, f"{name}_mass.npy"), mass)
    np.save(os.path.join(OUT, f"{name}_samples_z.npy"), smp)

    ratio = mass.std(0) / hs
    offset = (mass.mean(0) - hm) / hs
    in_box = np.all(np.abs(mass - hm) <= 6 * hs, axis=1)
    core_frac = float(in_box.mean())

    lp = np.asarray(logp_one(jnp.asarray(smp)))

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

    # v2 grader gate: n_eff-adjusted reporting (duplicate-induced ancestry
    # collapse means the R arms' std ratios carry MORE Monte-Carlo noise than
    # 512 independent chains would suggest).
    n_unique = n_unique_ancestors(res)
    ratio_noise = 1.0 / np.sqrt(2.0 * (n_unique - 1))

    results[name] = dict(mass=mass, res=res)
    summary[name] = dict(
        kwargs={k: str(v) for k, v in kw.items()},
        ratio=ratio.tolist(), ratio_median=float(np.median(ratio)),
        ratio_max=float(np.max(ratio)),
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
    print(f"[{name}] ratio median={s['ratio_median']:.3f} max={s['ratio_max']:.3f}  "
          f"offset_max={s['offset_max_abs']:.3f}  core_frac={s['core_frac']:.3f}  "
          f"eps={s['p2_final_step_size']:.3e}  acc_last={s['p2_accept_last']:.3f}  "
          f"n_unique={n_unique} (ratio_noise~{ratio_noise:.3f})  "
          f"resample={resample_summary}", flush=True)

# --------------------------------------------------------------------------- #
# R-arm duplicate-decorrelation curves + saturation reference                 #
# --------------------------------------------------------------------------- #
dup_curves = {}
saturations = {}
for name in R_ARMS:
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
colors_r = {"R1": "C3", "R1b": "C4", "R1c": "C5"}
for name in R_ARMS:
    curve = dup_curves[name]
    if curve is None:
        continue
    c = colors_r[name]
    ax.plot(np.arange(len(curve)), curve, color=c, lw=1.5, label=f"{name} straggler-donor rms")
    ax.axhline(saturations[name], color=c, ls=":", lw=1.0,
               label=f"{name} final-core saturation")
ax.set_xlabel("Phase-2 chunks since resample")
ax.set_ylabel("rms distance (unconstrained z)")
ax.set_title("DC-7.3 duplicate decorrelation: straggler-donor separation vs core spread")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "dup_decorr.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved dup_decorr.png", flush=True)

# --------------------------------------------------------------------------- #
# corner overlays: shared-range AND HMC-zoomed (diag_levers.py style)          #
# --------------------------------------------------------------------------- #
plot_arms = ["R1", "M1", "W", "A0r"]
colors = ["C3", "C2", "C0", "C1"]
arm_labels = ["R1 resampled prior", "M1 retune-only prior", "W warm-init",
              "A0r prior (no resample)"]
for tag, rng in (
    ("full", None),
    ("zoom", [(m - 6 * s, m + 6 * s) for m, s in zip(hmc_mass.mean(0), hs)]),
):
    if rng is None:
        allcat = np.concatenate([hmc_mass] + [results[a]["mass"] for a in plot_arms])
        lo, hi = allcat.min(0), allcat.max(0)
        pad = np.where((hi - lo) > 0, 0.05 * (hi - lo), 1e-6)
        rng = list(zip(lo - pad, hi + pad))
    ckw = dict(labels=labels, range=rng, bins=30, plot_datapoints=False,
               smooth=1.0, hist_kwargs=dict(density=True))
    fig = corner(hmc_mass, color="black", truths=markers, truth_color="green", **ckw)
    for a, col in zip(plot_arms, colors):
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
                for c, l in zip(colors, arm_labels)] +
               [Line2D([0], [0], color="green", lw=2, label="truth")])
    fig.legend(handles=handles, loc="upper right", fontsize=12, frameon=False)
    fig.suptitle(f"DC-7.3 resample test ({tag}): HMC vs R1/M1/W/A0r", fontsize=14)
    fig.savefig(os.path.join(OUT, f"resample_corner_{tag}.png"), dpi=110,
                bbox_inches="tight")
    plt.close(fig)
    print(f"saved resample_corner_{tag}.png", flush=True)

# --------------------------------------------------------------------------- #
# levers-style eps trajectory: R1 vs M1 vs W vs A0r (log-y), re-tune jump     #
# --------------------------------------------------------------------------- #
fig, ax = plt.subplots(figsize=(8, 5))
for a, col in zip(plot_arms, colors):
    ss = np.asarray(results[a]["res"].p2_step_size)
    ax.plot(np.arange(len(ss)), ss, color=col, lw=1.2, label=a)
r1_info = results["R1"]["res"].resample_info
if r1_info is not None and not bool(r1_info.get("skipped", True)):
    resample_step = int(r1_info["chunk"]) * 8    # p2_chunk_size default
    ax.axvline(resample_step, color="k", ls="--", lw=1.0, label="R1/M1 resample")
ax.set_yscale("log")
ax.set_xlabel("Phase-2 step")
ax.set_ylabel("eps (step size)")
ax.set_title("DC-7.3: Phase-2 step-size trajectory (R1/M1 re-tune jump vs W / A0r)")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "resample_eps.png"), dpi=110, bbox_inches="tight")
plt.close(fig)
print("saved resample_eps.png", flush=True)

# --------------------------------------------------------------------------- #
# summary.json                                                                #
# --------------------------------------------------------------------------- #
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
print("\nSUMMARY:", json.dumps(summary, indent=2))
print("DIAG DONE", flush=True)
