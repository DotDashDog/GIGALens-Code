r"""plot_run: a 5-panel internals+results figure for one LAPS validation run.

Panels (spec method-discipline 5: LOOK before trusting the metric):
  (a) D_tilde(t), log-y, with the finite-M equilibrium floor (~2/M ref + the
      run's own late-time median) and the switch step.
  (b) EEVPD_obs / EEVPD_wanted (t) -> 1 (the Phase-1 controller tracking check).
  (c) max_i delta_i/floor_i (t) vs the firing level k, with the switch step
      (the switch-ripeness check: did Phase 1 fire near its noise floor?).
  (d) Ehat[x_i^2] vs E_true[x_i^2] scatter on the y=x line, colored by b^2_i
      (the result check: structured second-moment disagreement, per dim).
  (e) Phase-2 acceptance(t) with the target +/-3% band and the freeze step.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _arr(x):
    return np.asarray(x, dtype=np.float64)


def plot_run(record, out_png):
    I = record["internals"]
    R = record["results"]
    cfg = record["config"]
    M = int(cfg.get("num_chains", 0)) or 1

    D = _arr(I["D_tilde"])
    eo = _arr(I["eevpd_obs"])
    ew = _arr(I["eevpd_wanted"])
    ratio_max = _arr(I["delta_over_floor_max"])
    switch_idx = int(I["switch_index"])
    phase1_len = int(I["phase1_len"])
    k = float(I["switch_k"])
    acc = _arr(I["p2_accept"])
    frozen = np.asarray(I["p2_frozen"], dtype=bool)
    a_target = float(I["target_accept"])

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    title = (f"LAPS validation: {cfg.get('target')} | schedule={cfg.get('schedule','paper')} "
             f"switch={cfg.get('switch','paper')}/{cfg.get('switch_mode','self_calibrated')} "
             f"init={cfg.get('init_mode','warm')} qz={cfg.get('qz_mode')} M={M}")
    fig.suptitle(title, fontsize=11)

    def mark_switch(ax):
        if switch_idx < phase1_len:
            ax.axvline(switch_idx, color="red", ls="--", lw=1.2,
                       label=f"switch @ {switch_idx}")
        else:
            ax.axvline(phase1_len - 1, color="grey", ls=":", lw=1.0,
                       label="no switch (budget)")

    # (a) D_tilde with equilibrium floor
    ax = axes[0, 0]
    t = np.arange(len(D))
    ax.semilogy(t, np.maximum(D, 1e-30), color="C0", label=r"$\tilde D(t)$")
    floor_ref = 2.0 / M
    ax.axhline(floor_ref, color="green", ls=":", lw=1.0, label=f"~2/M={floor_ref:.2e}")
    if len(D) >= 4:
        late = np.median(D[max(0, len(D) - max(2, len(D) // 5)):])
        ax.axhline(late, color="orange", ls="--", lw=1.0, label=f"late median={late:.2e}")
    mark_switch(ax)
    ax.set_xlabel("Phase-1 step"); ax.set_ylabel(r"$\tilde D$ (equipartition)")
    ax.set_title("(a) equipartition divergence -> floor"); ax.legend(fontsize=8)

    # (b) EEVPD obs/wanted -> 1
    ax = axes[0, 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = eo / ew
    ax.plot(t, r, color="C1", label=r"EEVPD$_{obs}$/EEVPD$_{wanted}$")
    ax.axhline(1.0, color="black", ls=":", lw=1.0, label="target ratio 1")
    mark_switch(ax)
    ax.set_yscale("log")
    ax.set_xlabel("Phase-1 step"); ax.set_ylabel("obs / wanted")
    ax.set_title("(b) EEVPD controller tracking"); ax.legend(fontsize=8)

    # (c) max delta/floor with firing level k
    ax = axes[0, 2]
    ax.plot(t, ratio_max, color="C2", label=r"$\max_i \delta_i/\mathrm{floor}_i$")
    ax.axhline(k, color="red", ls="--", lw=1.0, label=f"fire level k={k}")
    ax.axhline(1.0, color="green", ls=":", lw=1.0, label="floor (=1)")
    mark_switch(ax)
    ax.set_yscale("log")
    ax.set_xlabel("Phase-1 step"); ax.set_ylabel(r"$\max_i \delta_i/\mathrm{floor}_i$")
    ax.set_title("(c) switch ripeness vs noise floor"); ax.legend(fontsize=8)

    # (d) Ehat[x^2] vs E_true[x^2] colored by b2_i
    ax = axes[1, 0]
    ex2_true = _arr(R["ex2_true"]); ex2_hat = _arr(R["ex2_hat"])
    b2_i = _arr(R["b2_per_dim"])
    lim_lo = min(ex2_true.min(), ex2_hat.min())
    lim_hi = max(ex2_true.max(), ex2_hat.max())
    pad = 0.05 * (lim_hi - lim_lo + 1e-12)
    sc = ax.scatter(ex2_true, ex2_hat, c=b2_i, cmap="viridis", s=60,
                    edgecolor="k", linewidth=0.5, zorder=3)
    ax.plot([lim_lo - pad, lim_hi + pad], [lim_lo - pad, lim_hi + pad],
            color="grey", ls="--", lw=1.0, label="y = x")
    plt.colorbar(sc, ax=ax, label=r"$b^2_i$")
    ax.set_xlabel(r"$E_{true}[x_i^2]$"); ax.set_ylabel(r"$\hat E[x_i^2]$")
    ax.set_title(f"(d) 2nd-moment recovery  b2_max={R['b2_max']:.2e}")
    ax.legend(fontsize=8)

    # (e) Phase-2 acceptance with target band and freeze step
    ax = axes[1, 1]
    t2 = np.arange(len(acc))
    ax.plot(t2, acc, color="C3", label="acceptance(t)")
    ax.axhline(a_target, color="black", ls=":", lw=1.0, label=f"target {a_target}")
    ax.axhspan(a_target - 0.03, a_target + 0.03, color="green", alpha=0.15,
               label="+/-3% band")
    if frozen.any():
        fz = int(np.argmax(frozen))
        ax.axvline(fz, color="purple", ls="--", lw=1.2, label=f"freeze @ {fz}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Phase-2 step"); ax.set_ylabel("acceptance")
    ax.set_title("(e) Phase-2 acceptance -> target"); ax.legend(fontsize=8)

    # (f) summary text
    ax = axes[1, 2]; ax.axis("off")
    lines = [
        f"target        : {cfg.get('target')}  (d={cfg.get('dim')})",
        f"schedule/switch: {cfg.get('schedule','paper')} / {cfg.get('switch','paper')}",
        f"switch_mode   : {cfg.get('switch_mode','self_calibrated')} (k={k})",
        f"init / qz     : {cfg.get('init_mode','warm')} / {cfg.get('qz_mode')}",
        f"M chains      : {M}",
        "",
        f"switch_index  : {switch_idx}  (phase1_len={phase1_len}, switched={bool(I['switched'])})",
        f"  paper post  : {int(I['switch_index_paper'])}   emaus post: {int(I['switch_index_emaus'])}",
        f"integ order   : {int(I['integrator_order'])}  a_target={a_target}",
        f"frozen final  : {bool(frozen[-1]) if len(frozen) else False}",
        "",
        f"b2_max        : {R['b2_max']:.3e}   (floor 1/M={R['b2_floor']:.3e})",
        f"b2_avg        : {R['b2_avg']:.3e}",
        f"b2 success<0.01: {bool(R['b2_success'])}",
        f"max var relerr: {R['max_var_rel_err']:.3e}",
        f"mean abs err  : {R['mean_abs_err_max']:.3e}",
        f"nan_free      : {bool(record.get('nan_free', False))}",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace",
            fontsize=9, transform=ax.transAxes)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png
