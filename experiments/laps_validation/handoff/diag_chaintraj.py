#!/usr/bin/env python
# coding: utf-8
r"""DC-7.2 D1 v2 (grader-amended): per-chain logp trajectories on the demo lens
(H_S frozen tail vs H_R slowly-relaxing tail).

TWO seeds: A0 config (prior-init, 512 chains, 300 unadj + 200 adj,
early_stop=False, seeds 0 and 1) with track_chains=True. Membership uses
HYSTERESIS (leave-core at logp<0, enter-core at logp>152; measured gap:
straggler p75=-738 vs core p5=152) to kill typical-set boundary jitter, plus a
POSITIVE CONTROL (plain crossing at logp>160, core median) that must show high
late-Phase-2 crossing to prove the estimator sees churn when churn exists. The
plain logp>150 membership (DC-7.1 separator) is KEPT for the core-fraction
curve only. The decisive TIE-BREAKER block compares tail vs core chain
DYNAMICS (Delta-logp drift, per-chunk z-space displacement) over the last
third of Phase 2, split by final hysteresis membership. Outputs
(diag_chaintraj/): raw per-seed snapshot arrays, membership/crossing/churn
stats (plain + hysteresis + positive control), tie-breaker stats, Spearman
rank stability, tail logp evolution, per-seed spaghetti + rate plots,
summary.json (one file, per-seed entries). PRODUCES NUMBERS AND PLOTS, no
verdict. Model build verbatim from diag_levers.py.
"""
import os, json
import jax
jax.config.update("jax_enable_x64", True)
from gigalens.jax.inference import MAP, SVI
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
tfd = tfp.distributions
from gigalens_research.inference.laps_late_adjusted import LAPS_late_adjusted_JIT
print("jax devices:", jax.devices())

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "diag_chaintraj"); os.makedirs(OUT, exist_ok=True)

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
DIM = int(model.num_free_params); print("dim:", DIM)
opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
best, best_lp, best_chisq = MAP(prob_model, opt, seed=0)
print("MAP best_chisq (min):", float(np.min(np.asarray(best_chisq))))
opt = optax.adabelief(1e-4, b1=0.95, b2=0.99)
qz, _ = SVI(prob_model, best, opt, n_vi=1000, num_steps=1500); print("SVI done.", flush=True)

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
print(f"HMC ref: {hmc_mass.shape}, std={hs}")

# ---------------------------------------------------------------------------
# Spearman rank correlation: scipy if importable in the container, else a
# manual rank + np.corrcoef fallback (logp is continuous -> ties negligible).
# ---------------------------------------------------------------------------
try:
    from scipy.stats import spearmanr as _spearmanr

    def spearman(a, b):
        return float(_spearmanr(a, b)[0])
except Exception:
    def _rank(x):
        r = np.empty(len(x))
        r[np.argsort(x)] = np.arange(len(x))
        return r

    def spearman(a, b):
        return float(np.corrcoef(_rank(np.asarray(a)), _rank(np.asarray(b)))[0, 1])

CORE_LOGP = 150.0            # DC-7.1 near-clean separator; PLAIN membership,
                              # kept only for the core-fraction curve (v2 D1-2)
HYST_ENTER = 152.0           # hysteresis enter-core threshold (core p5)
HYST_EXIT = 0.0              # hysteresis leave-core threshold
POS_CONTROL_LOGP = 160.0     # positive-control plain threshold (core median)


def hysteresis_membership(lp, enter=HYST_ENTER, exit_=HYST_EXIT):
    """Per-chain state machine over concatenated snapshots. Start in-core iff
    logp[0] > enter; leave core only when logp < exit_; enter core only when
    logp > enter. In the dead zone [exit_, enter] the state is unchanged
    (this IS the hysteresis: it kills boundary jitter). Returns (T, M) bool.
    """
    T, M = lp.shape
    cur = lp[0] > enter
    states = np.zeros((T, M), dtype=bool)
    states[0] = cur
    for t in range(1, T):
        lpt = lp[t]
        leave = cur & (lpt < exit_)
        enter_m = (~cur) & (lpt > enter)
        cur = cur & ~leave
        cur = cur | enter_m
        states[t] = cur
    return states


def membership_stats(in_core, n1, n2, T):
    """Plain/hysteresis/positive-control crossing + core-fraction stats
    (same recipe for all three membership defs). in_core: (T, M) bool."""
    core_frac = in_core.mean(axis=1)
    changed = in_core[1:] != in_core[:-1]
    cross = changed.mean(axis=1)
    entered = (in_core[1:] & ~in_core[:-1]).mean(axis=1)
    exited = (~in_core[1:] & in_core[:-1]).mean(axis=1)
    p2_snap_idx = np.arange(n1, T)
    late_start = int(p2_snap_idx[(2 * n2) // 3])
    late_snaps = np.arange(late_start, T)
    late_trans = late_snaps[late_snaps >= 1] - 1
    late_changed = changed[late_trans]
    late_core = in_core[late_snaps]
    p2_mid = int(p2_snap_idx[n2 // 2])
    final_core = in_core[-1]
    n_fc = int(final_core.sum())
    frac_mid = (float(in_core[p2_mid][final_core].mean()) if n_fc > 0
                else float("nan"))
    stats = dict(
        final_core_frac=float(final_core.mean()),
        core_frac_per_snapshot=[float(v) for v in core_frac],
        cross_rate_per_transition=[float(v) for v in cross],
        entered_per_transition=[float(v) for v in entered],
        exited_per_transition=[float(v) for v in exited],
        late_phase=dict(
            window_first_snapshot=late_start, window_last_snapshot=int(T - 1),
            n_transitions=int(len(late_trans)),
            mean_cross_per_chunk=float(cross[late_trans].mean()),
            mean_entered_per_chunk=float(entered[late_trans].mean()),
            mean_exited_per_chunk=float(exited[late_trans].mean()),
            churn_frac_chains_changed=float(late_changed.any(axis=0).mean()),
            core_union_frac=float(late_core.any(axis=0).mean()),
            core_mean_frac=float(late_core.mean()),
            p2_mid_snapshot=p2_mid,
            frac_final_core_already_core_at_p2_mid=frac_mid,
        ),
    )
    return stats, final_core, late_trans


def run_one(seed):
    print(f"\n=== D1 v2: prior-init 512 chains, 300/200, early_stop=False, "
          f"seed {seed}, track_chains=True ===", flush=True)
    res = LAPS_late_adjusted_JIT(prob_model, qz, init_mode="prior", num_chains=512,
                                 early_stop=False, seed=seed, track_chains=True)
    traj = res.chain_traj
    p1_pos, p1_lp = np.asarray(traj["p1_pos"]), np.asarray(traj["p1_logp"])
    p2_pos, p2_lp = np.asarray(traj["p2_pos"]), np.asarray(traj["p2_logp"])
    n1, n2 = p1_lp.shape[0], p2_lp.shape[0]
    M = p1_lp.shape[1]
    print(f"[seed{seed}] snapshots: phase1 {p1_lp.shape} (incl. initial), "
          f"phase2 {p2_lp.shape}", flush=True)

    pos = np.concatenate([p1_pos, p2_pos], axis=0)       # (T, M, d)
    lp = np.concatenate([p1_lp, p2_lp], axis=0)          # (T, M)
    T = lp.shape[0]
    BOUNDARY = n1 - 1

    in_core_plain = lp > CORE_LOGP
    in_core_hyst = hysteresis_membership(lp)
    in_core_pos = lp > POS_CONTROL_LOGP     # positive control: plain crossing

    stats_plain, final_core_plain, _ = membership_stats(in_core_plain, n1, n2, T)
    stats_hyst, final_core_hyst, late_trans = membership_stats(in_core_hyst, n1, n2, T)
    stats_pos, _, _ = membership_stats(in_core_pos, n1, n2, T)
    positive_control_crossing_late = stats_pos["late_phase"]["mean_cross_per_chunk"]

    tail_hyst = ~final_core_hyst
    core_hyst = final_core_hyst
    n_final_core = int(final_core_hyst.sum())

    np.savez(os.path.join(OUT, f"chaintraj_lens_seed{seed}.npz"),
             p1_logp=p1_lp, p2_logp=p2_lp, p1_pos=p1_pos, p2_pos=p2_pos,
             in_core_plain=in_core_plain, in_core_hysteresis=in_core_hyst,
             in_core_positive_control=in_core_pos,
             core_logp_threshold=np.array(CORE_LOGP),
             hyst_enter=np.array(HYST_ENTER), hyst_exit=np.array(HYST_EXIT),
             pos_control_logp=np.array(POS_CONTROL_LOGP))

    # ---- rank stability: Spearman of per-chain logp, consecutive snapshots ----
    rho = np.array([spearman(lp[t], lp[t + 1]) for t in range(T - 1)])
    rho_late_tail = ([spearman(lp[t][tail_hyst], lp[t + 1][tail_hyst]) for t in late_trans]
                     if int(tail_hyst.sum()) >= 3 else [])
    spearman_stats = dict(
        per_transition=[float(r) for r in rho],
        late_mean=float(rho[late_trans].mean()), late_min=float(rho[late_trans].min()),
        late_tail_mean=(float(np.mean(rho_late_tail)) if rho_late_tail else None),
    )

    # ---- tail evolution over Phase 2 (hysteresis-tail chains) ----
    dlp_tail = lp[-1, tail_hyst] - lp[BOUNDARY, tail_hyst]
    qs = [5, 25, 50, 75, 95]
    tail_stats = dict(
        n_tail=int(tail_hyst.sum()),
        dlogp_quantiles={f"q{q:02d}": float(v)
                         for q, v in zip(qs, np.percentile(dlp_tail, qs))}
        if tail_hyst.sum() else {},
        dlogp_mean=(float(dlp_tail.mean()) if tail_hyst.sum() else None),
        frac_abs_dlogp_lt1=(float((np.abs(dlp_tail) < 1.0).mean())
                            if tail_hyst.sum() else None),
    )

    # ---- TIE-BREAKER (decisive): tail vs core dynamics, last third of Phase 2 ----
    dlp_trans = lp[late_trans + 1] - lp[late_trans]              # (n_late, M)
    per_chain_drift = np.median(dlp_trans, axis=0)               # (M,)
    disp_trans = np.linalg.norm(pos[late_trans + 1] - pos[late_trans], axis=-1)  # (n_late, M)

    def _grp_stats(mask):
        if int(mask.sum()) == 0:
            return dict(n=0, drift_median=None, drift_iqr=None, disp_median=None)
        dd = per_chain_drift[mask]
        disp = disp_trans[:, mask].ravel()
        q25, q75 = np.percentile(dd, [25, 75])
        return dict(n=int(mask.sum()), drift_median=float(np.median(dd)),
                    drift_iqr=float(q75 - q25), disp_median=float(np.median(disp)))

    tie_tail, tie_core = _grp_stats(tail_hyst), _grp_stats(core_hyst)
    disp_ratio = (tie_tail["disp_median"] / tie_core["disp_median"]
                  if (tie_tail["disp_median"] is not None
                      and tie_core["disp_median"]) else None)
    tie_breaker = dict(tail=tie_tail, core=tie_core,
                        tail_over_core_disp_ratio=(float(disp_ratio)
                            if disp_ratio is not None else None),
                        n_late_transitions=int(len(late_trans)))

    p2_mid = stats_hyst["late_phase"]["p2_mid_snapshot"]
    frac_final_core_at_mid = (float(in_core_hyst[p2_mid][final_core_hyst].mean())
                              if n_final_core > 0 else float("nan"))

    # ---- plot (i): spaghetti of per-chain logp trajectories (hyst grouping) ----
    x = np.arange(T)
    fig, ax = plt.subplots(figsize=(11, 6))
    for m in range(M):
        ax.plot(x, lp[:, m], color=("C0" if final_core_hyst[m] else "C3"),
                alpha=0.1, lw=0.7)
    ax.set_yscale("symlog", linthresh=100)
    ax.axvline(BOUNDARY, color="k", ls="--", lw=1, label="phase boundary")
    ax.axhline(CORE_LOGP, color="green", ls=":", lw=1.5,
               label=f"plain core threshold logp={CORE_LOGP:g}")
    ax.axhline(HYST_ENTER, color="darkgreen", ls="-.", lw=1.2,
               label=f"hyst enter logp={HYST_ENTER:g}")
    ax.axhline(HYST_EXIT, color="orange", ls="-.", lw=1.2,
               label=f"hyst exit logp={HYST_EXIT:g}")
    ax.set_xlabel("chunk snapshot index (0 = initial ensemble)")
    ax.set_ylabel("per-chain logp (symlog)")
    ax.set_title(f"D1 lens seed{seed}: per-chain logp trajectories "
                 f"(blue = final-core[hyst] n={n_final_core}, red = final-tail "
                 f"n={int(tail_hyst.sum())})")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"lens_spaghetti_seed{seed}.png"), dpi=110)
    plt.close(fig)
    print(f"saved lens_spaghetti_seed{seed}.png", flush=True)

    # ---- plot (ii): core fraction (plain) + crossing rate (plain & hyst) ----
    core_frac_plain = in_core_plain.mean(axis=1)
    changed_plain = in_core_plain[1:] != in_core_plain[:-1]
    cross_plain = changed_plain.mean(axis=1)
    entered_plain = (in_core_plain[1:] & ~in_core_plain[:-1]).mean(axis=1)
    exited_plain = (~in_core_plain[1:] & in_core_plain[:-1]).mean(axis=1)
    changed_hyst = in_core_hyst[1:] != in_core_hyst[:-1]
    cross_hyst = changed_hyst.mean(axis=1)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, core_frac_plain, "C0-o", ms=3, label="core fraction (plain, logp>150)")
    ax.set_xlabel("chunk snapshot index")
    ax.set_ylabel("core fraction", color="C0")
    ax.axvline(BOUNDARY, color="k", ls="--", lw=1)
    ax2 = ax.twinx()
    ax2.plot(x[1:], cross_plain, "C3-", lw=1.2, label="crossing rate (plain, both)")
    ax2.plot(x[1:], entered_plain, "C2--", lw=1, label="entered (plain)")
    ax2.plot(x[1:], exited_plain, "C1--", lw=1, label="exited (plain)")
    ax2.plot(x[1:], cross_hyst, "C4-", lw=1.6, label="crossing rate (hysteresis)")
    ax2.set_ylabel("crossing rate per chunk", color="C3")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    ax.set_title(f"D1 lens seed{seed}: core fraction and membership crossing rate")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"lens_rates_seed{seed}.png"), dpi=110)
    plt.close(fig)
    print(f"saved lens_rates_seed{seed}.png", flush=True)

    seed_summary = dict(
        config=dict(init_mode="prior", num_chains=M, num_unadjusted_steps=300,
                    num_adjusted_steps=200, early_stop=False, seed=seed,
                    core_logp_threshold=CORE_LOGP, hyst_enter=HYST_ENTER,
                    hyst_exit=HYST_EXIT, pos_control_logp=POS_CONTROL_LOGP),
        snapshots=dict(n_phase1=int(n1), n_phase2=int(n2), total=int(T),
                       phase_boundary_index=int(BOUNDARY)),
        run=dict(phase1_len=int(res.phase1_len), switched=bool(res.switched),
                 switch_index=int(res.switch_index),
                 switch_index_paper=int(res.switch_index_paper),
                 p2_final_eps=float(res.p2_final_step_size),
                 p2_accept_last=float(np.asarray(res.p2_accept)[-1])),
        membership_plain=stats_plain,
        membership_hysteresis=stats_hyst,
        positive_control_crossing_late=float(positive_control_crossing_late),
        frac_final_core_hyst_already_core_at_p2_mid=frac_final_core_at_mid,
        tie_breaker=tie_breaker,
        spearman=spearman_stats,
        tail_evolution=tail_stats,
    )
    print(f"[seed{seed}] final_core plain={stats_plain['final_core_frac']:.3f} "
          f"hyst={stats_hyst['final_core_frac']:.3f}  "
          f"pos_control_late_cross={positive_control_crossing_late:.4f}  "
          f"tie_breaker tail_disp={tie_tail['disp_median']} "
          f"core_disp={tie_core['disp_median']} "
          f"ratio={tie_breaker['tail_over_core_disp_ratio']}", flush=True)
    print(f"DIAG DONE seed {seed}", flush=True)
    return seed_summary


# =====================================================================
# TWO seeds: A0 config + track_chains
# =====================================================================
summary = {}
for s in (0, 1):
    summary[f"seed{s}"] = run_one(s)

summary["thresholds"] = dict(core_logp_plain=CORE_LOGP, hyst_enter=HYST_ENTER,
                             hyst_exit=HYST_EXIT, pos_control_logp=POS_CONTROL_LOGP)
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)


def _compact(v):
    if isinstance(v, dict):
        return {k: _compact(x) for k, x in v.items()
                if not (isinstance(x, list) and len(x) > 8)}
    return v


print("\nSUMMARY:", json.dumps(_compact(summary), indent=2), flush=True)
print("DIAG DONE", flush=True)
