#!/usr/bin/env python
# coding: utf-8
r"""DC-7.2 D2 v2 (grader-amended): controlled stranding bed (exploratory
known-answer testbed hunt).

Target: d=22 anisotropic Gaussian, per-dim sigma = geomspace(1e-3, 0.5, 22)
(lens-z-like span), logp(x) = -0.5*sum((x/sig)^2). Init = N(0,1) draws
("prior" 2x-1000x overdispersed per dim). 512 chains, 300 unadj + 200 adj
(lens-like budget), early_stop=False, seeds 0/1, track_chains=True.

Per seed: final core fraction (box def: all |x_i| < 6*sig_i; plus a logp-based
alternative, plus the D1-v2 box-hysteresis analog: leave-box at any
|x_i|>9*sig_i, enter-box at all |x_i|<6*sig_i), per-dim std ratio final/true,
the D1 membership-churn stats (all three membership defs), the D1-v2 decisive
TIE-BREAKER block (tail vs core Delta-logp drift + per-chunk z-space
displacement over the last third of Phase 2, split by box-hysteresis
membership), pre-pinned outcome-adjudication numbers (final core fractions,
tail rms-z-beyond-6 stats, a logp bimodality gap statistic), spaghetti
(sum((x/sig)^2) distance, log-y) + rate plots. Outcome A (bed strands like the
lens) => cheap known-answer testbed; Outcome B (bed converges) => stranding
needs lens-specific structure. PRODUCES NUMBERS AND PLOTS, no verdict. Bed
template: aniso_bed_f1.py.
"""
import os, json
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from gigalens_research.inference.laps_late_adjusted import LAPS_late_adjusted
print("devices", jax.devices()[:1], "...", len(jax.devices()), "total")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "diag_chaintraj"); os.makedirs(OUT, exist_ok=True)

d = 22
sig = np.geomspace(1e-3, 0.5, d)          # per-dim true std (lens-z-like span)
sig_j = jnp.asarray(sig)


def logp(x):
    return -0.5 * jnp.sum((x / sig_j) ** 2, axis=-1)


NUM_CHAINS = 512
# logp-based membership: logp > E[logp] - (d/2 + 2*sqrt(d/2)); logp = -chi2_d/2
# so E[logp] = -d/2 and std[logp] = sqrt(d/2).
LOGP_TYPICAL = -d / 2.0
LOGP_THRESH = LOGP_TYPICAL - (d / 2.0 + 2.0 * np.sqrt(d / 2.0))
R2_THRESH = -2.0 * LOGP_THRESH            # same threshold in sum((x/sig)^2)


def box_membership(pos):
    """all(|x_i| < 6*sig_i) per chain; pos (..., M, d) -> (..., M) bool."""
    return np.all(np.abs(pos) < 6.0 * sig, axis=-1)


def box_hysteresis_membership(pos, enter_k=6.0, exit_k=9.0):
    """Hysteresis analog of box_membership (D1 v2 recipe): per-chain state
    machine over concatenated snapshots. Start in-box iff all(|x_i| <
    enter_k*sig_i) at t=0; leave box only when any(|x_i| > exit_k*sig_i);
    enter box only when all(|x_i| < enter_k*sig_i). pos (T, M, d) -> (T, M)."""
    ratio = np.abs(pos) / sig                       # (T, M, d)
    all_in = np.all(ratio < enter_k, axis=-1)        # (T, M) enter condition
    any_out = np.any(ratio > exit_k, axis=-1)        # (T, M) leave condition
    T, M = all_in.shape
    cur = all_in[0].copy()
    states = np.zeros((T, M), dtype=bool)
    states[0] = cur
    for t in range(1, T):
        leave = cur & any_out[t]
        enter_m = (~cur) & all_in[t]
        cur = cur & ~leave
        cur = cur | enter_m
        states[t] = cur
    return states


def late_window(n1, n2, T):
    """Global transition indices (i -> i+1) for the last third of Phase 2."""
    p2_snap_idx = np.arange(n1, T)
    late_start = int(p2_snap_idx[(2 * n2) // 3])
    late_snaps = np.arange(late_start, T)
    return late_snaps[late_snaps >= 1] - 1


def tie_breaker_stats(lp, pos, late_trans, tail_mask, core_mask):
    """D1-v2-style decisive tie-breaker: tail vs core Delta-logp drift and
    per-chunk z-space displacement over the last third of Phase 2."""
    dlp_trans = lp[late_trans + 1] - lp[late_trans]              # (n_late, M)
    per_chain_drift = np.median(dlp_trans, axis=0)               # (M,)
    disp_trans = np.linalg.norm(pos[late_trans + 1] - pos[late_trans], axis=-1)

    def _grp(mask):
        if int(mask.sum()) == 0:
            return dict(n=0, drift_median=None, drift_iqr=None, disp_median=None)
        dd = per_chain_drift[mask]
        disp = disp_trans[:, mask].ravel()
        q25, q75 = np.percentile(dd, [25, 75])
        return dict(n=int(mask.sum()), drift_median=float(np.median(dd)),
                    drift_iqr=float(q75 - q25), disp_median=float(np.median(disp)))

    tail_s, core_s = _grp(tail_mask), _grp(core_mask)
    ratio = (tail_s["disp_median"] / core_s["disp_median"]
             if (tail_s["disp_median"] is not None and core_s["disp_median"])
             else None)
    return dict(tail=tail_s, core=core_s,
                tail_over_core_disp_ratio=(float(ratio) if ratio is not None else None),
                n_late_transitions=int(len(late_trans)))


def churn_stats(in_core, n1):
    """The D1 membership-churn stats. in_core (T, M); snapshots 0..n1-1 = P1."""
    T, M = in_core.shape
    n2 = T - n1
    core_frac = in_core.mean(axis=1)
    changed = in_core[1:] != in_core[:-1]
    cross = changed.mean(axis=1)
    entered = (in_core[1:] & ~in_core[:-1]).mean(axis=1)
    exited = (~in_core[1:] & in_core[:-1]).mean(axis=1)
    p2_snap_idx = np.arange(n1, T)
    late_start = int(p2_snap_idx[(2 * n2) // 3])      # last third of P2 snaps
    late_snaps = np.arange(late_start, T)
    late_trans = late_snaps[late_snaps >= 1] - 1      # cross[i] = i -> i+1
    late_changed = changed[late_trans]
    late_core = in_core[late_snaps]
    p2_mid = int(p2_snap_idx[n2 // 2])
    final_core = in_core[-1]
    n_fc = int(final_core.sum())
    frac_mid = (float(in_core[p2_mid][final_core].mean()) if n_fc > 0
                else float("nan"))
    return dict(
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
    ), final_core


summary = {}
for s in (0, 1):
    print(f"\n=== D2 seed {s}: N(0,1)-init {NUM_CHAINS} chains, 300/200, "
          "early_stop=False, track_chains=True ===", flush=True)
    init = jax.random.normal(jax.random.key(100 + s), (NUM_CHAINS, d))
    res = LAPS_late_adjusted(logp, dim=d, num_chains=NUM_CHAINS,
                             num_unadjusted_steps=300, num_adjusted_steps=200,
                             init_positions=init, early_stop=False, seed=s,
                             track_chains=True)
    traj = res.chain_traj
    p1_pos, p1_lp = np.asarray(traj["p1_pos"]), np.asarray(traj["p1_logp"])
    p2_pos, p2_lp = np.asarray(traj["p2_pos"]), np.asarray(traj["p2_logp"])
    n1, n2 = p1_pos.shape[0], p2_pos.shape[0]
    pos = np.concatenate([p1_pos, p2_pos], axis=0)    # (T, M, d)
    lp = np.concatenate([p1_lp, p2_lp], axis=0)       # (T, M)
    T, M = lp.shape
    BOUNDARY = n1 - 1        # last P1 snapshot == the P2 start state
    print(f"snapshots: phase1 {p1_pos.shape} (incl. initial), "
          f"phase2 {p2_pos.shape}", flush=True)

    # final Phase-2 ensemble
    smp = np.asarray(res.samples).reshape((-1, d))    # (M, d)
    final_std = smp.std(axis=0)
    ratio = final_std / sig
    final_core_box = box_membership(smp)
    lp_final = -0.5 * np.sum((smp / sig) ** 2, axis=1)
    final_core_lp = lp_final > LOGP_THRESH

    # membership per snapshot: box def (primary) + logp def (alternative)
    in_core_box = box_membership(pos)                 # (T, M)
    in_core_lp = lp > LOGP_THRESH                     # (T, M)
    stats_box, fc_box = churn_stats(in_core_box, n1)
    stats_lp, _ = churn_stats(in_core_lp, n1)

    # ---- hysteresis analog of the box membership (D1 v2 D2-1) ----
    in_core_box_hyst = box_hysteresis_membership(pos)  # (T, M)
    stats_box_hyst, fc_box_hyst = churn_stats(in_core_box_hyst, n1)
    late_trans = late_window(n1, n2, T)
    tail_box_hyst, core_box_hyst = ~fc_box_hyst, fc_box_hyst
    tie_breaker = tie_breaker_stats(lp, pos, late_trans, tail_box_hyst, core_box_hyst)

    # ---- pre-pinned outcome-adjudication fields (D2-2): numbers only ----
    ratio_final_abs = np.abs(smp) / sig                     # (M, d)
    max_ratio_final = ratio_final_abs.max(axis=-1)           # (M,)
    rms_ratio_final = np.sqrt((ratio_final_abs ** 2).mean(axis=-1))
    tail_mask_final = tail_box_hyst                          # final-snapshot tail (box hysteresis)
    tail_rms_beyond_6 = dict(
        n_tail=int(tail_mask_final.sum()),
        frac_tail_max_ratio_gt6=(float((max_ratio_final[tail_mask_final] > 6.0).mean())
                                 if tail_mask_final.sum() else None),
        median_tail_max_ratio=(float(np.median(max_ratio_final[tail_mask_final]))
                               if tail_mask_final.sum() else None),
        median_tail_rms_ratio=(float(np.median(rms_ratio_final[tail_mask_final]))
                               if tail_mask_final.sum() else None),
    )
    # logp bimodality gap statistic: fraction of chains with final logp
    # strictly between the "in-core" cutoff (reuses LOGP_THRESH, the
    # script's existing analytic 2-sigma-in-chi2 entry threshold, ~-28.6
    # for d=22 here, rather than a hardcoded illustrative number) and a
    # deep "clearly stranded" floor (-200, per spec). A thin bridge
    # (frac_in_gap near 0) indicates bimodal core/tail separation.
    GAP_UPPER, GAP_LOWER = float(LOGP_THRESH), -200.0
    gap_frac = float(np.mean((lp_final < GAP_UPPER) & (lp_final > GAP_LOWER)))
    bimodality = dict(gap_upper=GAP_UPPER, gap_lower=GAP_LOWER,
                       frac_in_gap=gap_frac,
                       note="thin bridge (frac_in_gap near 0) => bimodal core/tail separation")

    r2 = np.sum((pos / sig) ** 2, axis=-1)            # (T, M) "distance"

    np.savez(os.path.join(OUT, f"strand_bed_seed{s}.npz"),
             p1_pos=p1_pos, p2_pos=p2_pos, p1_logp=p1_lp, p2_logp=p2_lp,
             samples=smp, sig=sig, r2=r2,
             in_core_box=in_core_box, in_core_logp=in_core_lp,
             in_core_box_hysteresis=in_core_box_hyst,
             logp_thresh=np.array(LOGP_THRESH))

    # ---- plot (i): spaghetti of sum((x/sig)^2) distance, log-y ----
    x = np.arange(T)
    fig, ax = plt.subplots(figsize=(11, 6))
    for m in range(M):
        ax.plot(x, np.maximum(r2[:, m], 1e-12),
                color=("C0" if fc_box[m] else "C3"), alpha=0.1, lw=0.7)
    ax.set_yscale("log")
    ax.axvline(BOUNDARY, color="k", ls="--", lw=1, label="phase boundary")
    ax.axhline(R2_THRESH, color="green", ls=":", lw=1.5,
               label=rf"logp-core threshold $\sum (x/\sigma)^2$={R2_THRESH:.1f}")
    ax.set_xlabel("chunk snapshot index (0 = initial ensemble)")
    ax.set_ylabel(r"$\sum_i (x_i/\sigma_i)^2$ (log)")
    ax.set_title(f"D2 bed seed {s}: per-chain distance trajectories "
                 f"(blue = final box-core n={int(fc_box.sum())}, red = tail "
                 f"n={int((~fc_box).sum())})")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"strand_bed_seed{s}_spaghetti.png"), dpi=110)
    plt.close(fig)

    # ---- plot (ii): core fraction + crossing rate (box def) ----
    core_frac = in_core_box.mean(axis=1)
    changed = in_core_box[1:] != in_core_box[:-1]
    cross = changed.mean(axis=1)
    entered = (in_core_box[1:] & ~in_core_box[:-1]).mean(axis=1)
    exited = (~in_core_box[1:] & in_core_box[:-1]).mean(axis=1)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(x, core_frac, "C0-o", ms=3, label="core fraction (box)")
    ax.plot(x, in_core_lp.mean(axis=1), "C4-s", ms=3, alpha=0.6,
            label="core fraction (logp)")
    ax.set_xlabel("chunk snapshot index")
    ax.set_ylabel("core fraction", color="C0")
    ax.axvline(BOUNDARY, color="k", ls="--", lw=1)
    ax2 = ax.twinx()
    ax2.plot(x[1:], cross, "C3-", lw=1.2, label="crossing rate (box, both)")
    ax2.plot(x[1:], entered, "C2--", lw=1, label="entered")
    ax2.plot(x[1:], exited, "C1--", lw=1, label="exited")
    ax2.set_ylabel("crossing rate per chunk", color="C3")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")
    ax.set_title(f"D2 bed seed {s}: core fraction and membership crossing rate")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"strand_bed_seed{s}_rates.png"), dpi=110)
    plt.close(fig)
    print(f"saved strand_bed_seed{s}_spaghetti.png / _rates.png", flush=True)

    summary[f"seed{s}"] = dict(
        snapshots=dict(n_phase1=int(n1), n_phase2=int(n2), total=int(T),
                       phase_boundary_index=int(BOUNDARY)),
        run=dict(phase1_len=int(res.phase1_len), switched=bool(res.switched),
                 switch_index_paper=int(res.switch_index_paper),
                 p2_final_eps=float(res.p2_final_step_size),
                 p2_accept_last=float(np.asarray(res.p2_accept)[-1])),
        final_core_frac_box=float(final_core_box.mean()),
        final_core_frac_logp=float(final_core_lp.mean()),
        final_core_frac_box_hysteresis=float(fc_box_hyst.mean()),
        std_ratio_per_dim=[float(v) for v in ratio],
        std_ratio_median=float(np.median(ratio)),
        std_ratio_max=float(np.max(ratio)),
        membership_box=stats_box,
        membership_logp=stats_lp,
        membership_box_hysteresis=stats_box_hyst,
        tie_breaker=tie_breaker,
        tail_rms_beyond_6=tail_rms_beyond_6,
        bimodality=bimodality,
    )
    sd = summary[f"seed{s}"]
    print(f"[seed{s}] core_frac box={sd['final_core_frac_box']:.3f} "
          f"logp={sd['final_core_frac_logp']:.3f} "
          f"box_hyst={sd['final_core_frac_box_hysteresis']:.3f}  "
          f"std ratio median={sd['std_ratio_median']:.2f} "
          f"max={sd['std_ratio_max']:.2f}  "
          f"late cross box={stats_box['late_phase']['mean_cross_per_chunk']:.4f} "
          f"tie_breaker disp_ratio={tie_breaker['tail_over_core_disp_ratio']} "
          f"bimodality frac_in_gap={bimodality['frac_in_gap']:.4f}",
          flush=True)
    print(f"BED DONE seed {s}", flush=True)

summary["thresholds"] = dict(logp_typical=LOGP_TYPICAL, logp_thresh=LOGP_THRESH,
                             r2_thresh=R2_THRESH, box_k_sigma=6.0)
json.dump(summary, open(os.path.join(OUT, "strand_bed_summary.json"), "w"),
          indent=2)


def _compact(v):
    if isinstance(v, dict):
        return {k: _compact(x) for k, x in v.items()
                if not (isinstance(x, list) and len(x) > 8)}
    return v


print("\nSUMMARY:", json.dumps(_compact(summary), indent=2), flush=True)
print("BED DONE", flush=True)
