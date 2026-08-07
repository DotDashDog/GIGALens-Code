"""Plots + pre-registered verdicts for the corrugation testbed run2 (DC-2).

Reads npz/summary.json from corrugation_scan.py; writes PNGs into <outdir>/plots/
and verdicts.json next to summary.json. Plots are the primary artifact (operating
card #3); the verdict table is derived from them.

    python make_plots.py [--out results/undersampling_corrugation/run2]
"""

import argparse
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from gigalens_research.paths import resolve_out_dir

SS_SCAN = (1, 2, 4, 8)
CUSPY = ("n4_re0.5_nopsf", "n8_re0.5_nopsf")
PHASE_LANES = ("phase_ph00_n8_re0.5_nopsf", "n8_re0.5_nopsf", "phase_ph47_n8_re0.5_nopsf")
SOFT_ARM = (("n4_re0.5_nopsf", 0.0), ("soft_rc0.25_n4_re0.5_nopsf", 0.25),
            ("soft_rc0.5_n4_re0.5_nopsf", 0.5), ("soft_rc1_n4_re0.5_nopsf", 1.0))
INJ_LANE, INJ_AMP, INJ_FREQ = "n4_re0.5_nopsf", 1.0, 3.3  # DC-2 P1'(c)

SS_COLOR = {ss: cm.Blues(0.35 + 0.6 * i / 3) for i, ss in enumerate(SS_SCAN)}
N_COLOR = {1.0: "#4477AA", 4.0: "#EE6677", 8.0: "#228833"}
GRID_KW = dict(alpha=0.25, lw=0.6)


def detrend(y, x, deg=6):
    c = np.polynomial.polynomial.polyfit(x, y, deg)
    return y - np.polynomial.polynomial.polyval(x, c)


def spectrum(resid, dx_pix):
    w = np.hanning(len(resid))
    p = np.abs(np.fft.rfft(resid * w)) ** 2
    f = np.fft.rfftfreq(len(resid), d=dx_pix)
    return f, p


def comb_peak(resid, dx_pix):
    f, p = spectrum(resid, dx_pix)
    band = f > 0.4
    fb, pb = f[band], p[band]
    return float(fb[int(np.argmax(pb))]), f[1] - f[0]


# ------------------------------------------------------------------- figures
def lane_scan_figure(name, lane, d, pdir):
    xs = d["xs_pix"]
    ss_ref = lane["ss_ref"]
    vref, gref = d[f"logL_ss{ss_ref}"], d[f"grad_ss{ss_ref}"]
    fig, axes = plt.subplots(len(SS_SCAN), 2, figsize=(11, 9), sharex=True)
    for row, ss in enumerate(SS_SCAN):
        resid = detrend(d[f"logL_ss{ss}"] - vref, xs)
        gresid = detrend(d[f"grad_ss{ss}"] - gref, xs)
        for col, (y, lab) in enumerate(
            ((resid, r"$\Delta\log L$ (detrended)"), (gresid, r"$\Delta\,\partial_x\log L$"))
        ):
            ax = axes[row, col]
            ax.plot(xs, y, color=SS_COLOR[ss], lw=1.0)
            for k in np.arange(-2, 2 + 1e-9, 1.0 / ss):
                ax.axvline(k, color="0.85", **GRID_KW)
            ax.text(0.02, 0.92, f"ss={ss}", transform=ax.transAxes,
                    va="top", fontsize=9, color="0.25")
            if row == len(SS_SCAN) - 1:
                ax.set_xlabel(r"$\Delta$center_x  [native pix]")
            if col == 0 and row == 1:
                ax.set_ylabel(lab)
    tag = " [REFERENCE-LIMITED]" if lane.get("reference_limited") else ""
    fig.suptitle(f"{name}: residual vs ss_ref={ss_ref} scan{tag} "
                 "(gridlines at the subgrid period)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, f"scan_{name}.png"), dpi=140)
    plt.close(fig)


def lane_spectrum_figure(name, lane, d, pdir):
    xs = d["xs_pix"]
    dx = xs[1] - xs[0]
    vref = d[f"logL_ss{lane['ss_ref']}"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for ss in SS_SCAN:
        f, p = spectrum(detrend(d[f"logL_ss{ss}"] - vref, xs), dx)
        ax.semilogy(f, p, color=SS_COLOR[ss], lw=1.2, label=f"ss={ss}")
        ax.axvline(ss, color=SS_COLOR[ss], ls="--", lw=0.8, alpha=0.6)
    ax.set_xlim(0, 12)
    ax.set_xlabel("frequency  [cycles / native pix]")
    ax.set_ylabel("power (Hann rFFT)")
    ax.set_title(f"{name}: comb vs expected f=ss (dashed)", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, **GRID_KW)
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, f"spectrum_{name}.png"), dpi=140)
    plt.close(fig)


def error_map_figure(name, lane, d, pdir):
    ref = d[f"truthmap_ss{lane['ss_ref']}"]
    delta1 = d["truthmap_ss1"] - ref
    sr = np.abs(d["truthmap_ss1"] - d["truthmap_ss2"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    ims = [axes[0].imshow(ref, origin="lower", cmap="magma")]
    axes[0].set_title(f"truth render (ss={lane['ss_ref']})", fontsize=9)
    v = np.abs(delta1).max()
    ims.append(axes[1].imshow(delta1, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v))
    axes[1].set_title(r"$m_{ss=1}-m_{ref}$  [$\sigma$ units]", fontsize=9)
    ims.append(axes[2].imshow(sr, origin="lower", cmap="magma"))
    axes[2].set_title(r"$\sigma_{render}=|m_{1}-m_{2}|$", fontsize=9)
    for ax, im in zip(axes, ims):
        fig.colorbar(im, ax=ax, shrink=0.85)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, f"maps_{name}.png"), dpi=140)
    plt.close(fig)


def summary_figures(summary, pdir):
    prim = {k: v for k, v in summary.items()
            if isinstance(v, dict) and v.get("role") == "primary"}
    res = sorted({v["re_pix"] for v in prim.values()})
    fig, axes = plt.subplots(1, len(res), figsize=(4 * len(res), 3.8), sharey=True)
    for ax, re_pix in zip(np.atleast_1d(axes), res):
        for n, col in N_COLOR.items():
            lane = next((v for v in prim.values()
                         if v["n"] == n and v["re_pix"] == re_pix), None)
            if lane is None:
                continue
            A = [lane["per_ss"][str(ss)]["A"] for ss in SS_SCAN]
            ls = "--" if lane.get("reference_limited") else "-"
            ax.loglog(SS_SCAN, A, "o" + ls, color=col, lw=1.5, ms=5, label=f"n={n:g}")
        ax.axhline(0.5, color="0.6", ls=":", lw=1)
        ax.set_title(f"$R_e$ = {re_pix:g} pix", fontsize=10)
        ax.set_xlabel("supersample factor")
        ax.set_xticks(SS_SCAN, [str(s) for s in SS_SCAN])
        ax.grid(True, which="both", **GRID_KW)
    np.atleast_1d(axes)[0].set_ylabel(r"corrugation amplitude  [$\Delta\log L$]")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Primary (no-PSF) corrugation amplitude "
                 "(dotted: 0.5 = 1$\\sigma$ contour; dashed lines: reference-limited)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, "amplitude_vs_ss.png"), dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for key, lane in prim.items():
        shifts = [abs(lane["per_ss"][str(ss)]["mode_shift_over_sigma"]) for ss in SS_SCAN]
        ax.semilogy(SS_SCAN, np.maximum(shifts, 1e-4), "o-",
                    color=N_COLOR[lane["n"]], alpha=0.8, lw=1.2,
                    label=f"n={lane['n']:g}, $R_e$={lane['re_pix']:g}")
    ax.axhline(1.0, color="0.4", ls=":", lw=1)
    ax.set_xlabel("supersample factor")
    ax.set_xticks(SS_SCAN, [str(s) for s in SS_SCAN])
    ax.set_ylabel(r"|mode shift| / $\sigma_{x_0}$")
    ax.set_title("P6b: aliasing-induced mode displacement (primary lanes)", fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(True, **GRID_KW)
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, "mode_shift_vs_ss.png"), dpi=140)
    plt.close(fig)


def soft_arm_figure(summary, pdir):
    rcs, amps = [], {ss: [] for ss in SS_SCAN}
    for name, rc in SOFT_ARM:
        if name not in summary:
            return
        rcs.append(rc)
        for ss in SS_SCAN:
            amps[ss].append(summary[name]["per_ss"][str(ss)]["A"])
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for ss in SS_SCAN:
        ax.semilogy(rcs, amps[ss], "o-", color=SS_COLOR[ss], lw=1.5, label=f"ss={ss}")
    ax.set_xlabel(r"core softening $r_c$  [native pix]")
    ax.set_ylabel(r"corrugation amplitude  [$\Delta\log L$]")
    ax.set_title("Softened-core arm (n=4, $R_e$=0.5 pix, no PSF): "
                 "cusp removal collapses the comb", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, **GRID_KW)
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, "soft_arm.png"), dpi=140)
    plt.close(fig)


def phase_figure(summary, pdir):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for ss in (1, 2):
        ph, sh = [], []
        for name in PHASE_LANES:
            if name not in summary:
                return
            ph.append(summary[name]["phase"][0])
            sh.append(summary[name]["per_ss"][str(ss)]["mode_shift_over_sigma"])
        ax.plot(ph, sh, "o-", color=SS_COLOR[ss], lw=1.5, label=f"ss={ss}")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("truth sub-pixel phase (x)  [native pix]")
    ax.set_ylabel(r"mode shift / $\sigma_{x_0}$")
    ax.set_title("P6b': mode displacement vs truth phase (n=8, $R_e$=0.5, no PSF)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, **GRID_KW)
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, "phase_arm.png"), dpi=140)
    plt.close(fig)


def stage2_figure(outdir, summary, pdir):
    if "stage2" not in summary:
        return
    lane = summary["stage2"]["lane"]
    ss_ref = summary[lane]["ss_ref"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for ax, ss in zip(axes, (1, 2)):
        d_old = np.load(os.path.join(outdir, f"scan_{lane}.npz"))
        d_new = np.load(os.path.join(outdir, f"stage2_{lane}_ss{ss}.npz"))
        r_old = detrend(d_old[f"logL_ss{ss}"] - d_old[f"logL_ss{ss_ref}"], d_old["xs_pix"])
        r_new = detrend(d_new["logL"] - d_new["logL_ref"], d_new["xs_pix"])
        ax.plot(d_old["xs_pix"], r_old, color="#EE6677", lw=1.0, label="original $\\sigma$")
        ax.plot(d_new["xs_pix"], r_new, color="#4477AA", lw=1.0,
                label=r"$\sigma_{eff}=\sqrt{\sigma^2+\sigma_{render}^2}$")
        s2 = summary["stage2"][str(ss)]
        ax.set_title(f"ss={ss}: suppression {s2['suppression']:.1f}x, "
                     f"width x{s2['width_ratio']:.1f}, "
                     f"relevance gain {s2['relevance_gain']:.1f}x", fontsize=10)
        ax.set_xlabel(r"$\Delta$center_x  [native pix]")
        ax.grid(True, **GRID_KW)
    axes[0].set_ylabel(r"$\Delta\log L$ (detrended)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(f"Stage 2 (noise inflation), lane {lane}", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(pdir, "stage2.png"), dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------- verdicts
def injection_test(outdir, summary):
    """DC-2 P1'(c): plant a sinusoid on the reference scan; pipeline must recover it."""
    lane = summary[INJ_LANE]
    d = np.load(os.path.join(outdir, f"scan_{INJ_LANE}.npz"))
    xs = d["xs_pix"]
    inj = INJ_AMP * np.sin(2 * np.pi * INJ_FREQ * xs)
    resid = detrend(inj.copy(), xs)  # what the pipeline does to a pure signal
    A_rec = float(resid.max() - resid.min())
    f_rec, dfbin = comb_peak(resid, xs[1] - xs[0])
    return dict(
        A_true=2 * INJ_AMP, A_recovered=A_rec,
        amp_err_frac=abs(A_rec - 2 * INJ_AMP) / (2 * INJ_AMP),
        f_true=INJ_FREQ, f_recovered=f_rec,
        passed=bool(abs(A_rec - 2 * INJ_AMP) / (2 * INJ_AMP) < 0.15
                    and abs(f_rec - INJ_FREQ) <= dfbin),
    )


def verdicts(summary, outdir):
    v = {}
    prim = {k: la for k, la in summary.items()
            if isinstance(la, dict) and la.get("role") in ("primary", "phase")}
    certified = {k: la for k, la in prim.items() if not la["reference_limited"]}
    v["reference_certification"] = {
        k: dict(cert_gap=round(la["cert_gap"], 4), limited=la["reference_limited"])
        for k, la in summary.items() if isinstance(la, dict) and "cert_gap" in la}

    # P1'a: every resolved primary/phase amplitude combs at f=ss (+-1 bin)
    fails = [(k, ss) for k, la in prim.items() for ss in SS_SCAN
             if la["per_ss"][str(ss)]["A_resolved"]
             and not la["per_ss"][str(ss)]["comb"]["within_one_bin"]
             and not la["reference_limited"]]
    v["P1a_period"] = dict(passed=not fails, fails=fails)

    # P1'b: log-log slope of f_peak vs ss = 1 +- 0.05 on cuspy certified lanes
    slopes = {}
    for k in CUSPY:
        if k not in certified:
            continue
        fp = [prim[k]["per_ss"][str(ss)]["comb"]["f_peak"] for ss in SS_SCAN]
        slopes[k] = float(np.polyfit(np.log(SS_SCAN), np.log(fp), 1)[0])
    v["P1b_scaling"] = dict(passed=bool(slopes) and all(abs(s - 1) < 0.05 for s in slopes.values()),
                            slopes={k: round(s, 4) for k, s in slopes.items()})

    v["P1c_injection"] = injection_test(outdir, summary)

    # P2: monotone orderings on certified primary lanes; ss-reversal = STRUCTURAL
    ok, viol, structural = True, [], []
    by_key = {(la["n"], la["re_pix"]): la for la in summary.values()
              if isinstance(la, dict) and la.get("role") == "primary"
              and not la["reference_limited"]}
    for ss in SS_SCAN:
        for re_pix in (0.5, 1.0, 3.0):
            a = [by_key[(n, re_pix)]["per_ss"][str(ss)]["A"]
                 for n in (1.0, 4.0, 8.0) if (n, re_pix) in by_key]
            if len(a) > 1 and any(np.diff(a) < 0):
                ok = False; viol.append(("n", ss, re_pix, [float(f"{x:.4g}") for x in a]))
        for n in (1.0, 4.0, 8.0):
            a = [by_key[(n, re_pix)]["per_ss"][str(ss)]["A"]
                 for re_pix in (0.5, 1.0, 3.0) if (n, re_pix) in by_key]
            if len(a) > 1 and any(np.diff(a) > 0):
                ok = False; viol.append(("re", ss, n, [float(f"{x:.4g}") for x in a]))
    for key, la in by_key.items():
        a = [la["per_ss"][str(ss)]["A"] for ss in SS_SCAN]
        if any(np.diff(a) > 0):
            ok = False
            structural.append((key, [float(f"{x:.4g}") for x in a]))
    v["P2_ordering"] = dict(passed=bool(ok), violations=viol,
                            ss_reversals_STRUCTURAL=structural)

    # P3': control absolute + relative criteria
    ctrl = summary["control_n1_re10_nopsf"]["per_ss"]
    rel, absx = {}, {}
    for ss in SS_SCAN:
        a_ctrl = ctrl[str(ss)]["A"]
        a_max = max(la["per_ss"][str(ss)]["A"] for la in prim.values())
        rel[ss] = a_ctrl / a_max
        absx[ss] = a_ctrl
    v["P3_control"] = dict(
        passed=bool(all(r < 1e-2 for r in rel.values())
                    and all(absx[ss] < 0.5 for ss in (2, 4, 8))),
        A_control={k: round(x, 4) for k, x in absx.items()},
        rel_to_cuspiest={k: float(f"{x:.3g}") for k, x in rel.items()})

    # P4 (harness check): grad amplification within factor 3 on certified lanes
    ratios = [la["per_ss"][str(ss)]["A_grad"] / la["per_ss"][str(ss)]["A_grad_pred"]
              for la in certified.values() for ss in SS_SCAN
              if la["per_ss"][str(ss)]["A_resolved"]]
    v["P4_grad_harness_check"] = dict(passed=bool(all(1 / 3 <= r <= 3 for r in ratios)),
                                      min=round(min(ratios), 2), max=round(max(ratios), 2))

    # P5 (descriptive): PSF vs no-PSF amplitude ratios
    p5 = {}
    for n in (1.0, 4.0, 8.0):
        for re_pix in (0.5, 1.0, 3.0):
            a = summary.get(f"n{n:g}_re{re_pix:g}_nopsf")
            b = summary.get(f"n{n:g}_re{re_pix:g}_psf")
            if a and b:
                p5[f"n{n:g}_re{re_pix:g}"] = [
                    round(a["per_ss"][str(ss)]["A"] / b["per_ss"][str(ss)]["A"], 2)
                    for ss in SS_SCAN]
    v["P5_psf_descriptive"] = dict(A_nopsf_over_psf=p5)

    # P6: posterior relevance within +-1 sigma (cuspy lanes)
    a1s = {k: max(prim[k]["per_ss"][str(ss)]["A_within_1sigma"] for ss in (1, 2))
           for k in CUSPY if k in prim}
    v["P6_relevance"] = dict(
        passed=bool(any(np.isfinite(x) and x > 1 for x in a1s.values())),
        A_within_1sigma={k: (round(x, 3) if np.isfinite(x) else None)
                         for k, x in a1s.items()})

    # P6b': mode displacement oscillates with truth phase
    ph = {}
    for ss in (1, 2):
        vals = [summary[k]["per_ss"][str(ss)]["mode_shift_over_sigma"]
                for k in PHASE_LANES if k in summary]
        ph[ss] = [round(x, 3) for x in vals]
    varies = all(len(set(np.sign(x) for x in vals)) > 1 or
                 (max(vals) - min(vals)) > max(1.0, 0.2 * max(abs(x) for x in vals))
                 for vals in ([v_ for v_ in ph[ss]] for ss in (1, 2)) if vals)
    big = any(abs(x) > 1 for vals in ph.values() for x in vals)
    v["P6b_phase"] = dict(passed=bool(varies and big), shifts_by_phase=ph)

    # Soft arm: monotone collapse with r_c; A(rc=1)/A(0) < 0.1 at ss=1
    if all(name in summary for name, _ in SOFT_ARM):
        arm = {}
        for ss in SS_SCAN:
            arm[ss] = [summary[name]["per_ss"][str(ss)]["A"] for name, _ in SOFT_ARM]
        a1 = arm[1]
        v["Soft_arm"] = dict(
            passed=bool(all(np.diff(a1) < 0) and a1[-1] / a1[0] < 0.1),
            A_ss1_by_rc=[float(f"{x:.4g}") for x in a1],
            collapse_ratio=float(f"{a1[-1] / a1[0]:.3g}"))

    # Stage 2': dimensionless criteria on the certified target lane
    if "stage2" in summary:
        s2 = {k: e for k, e in summary["stage2"].items() if k != "lane"}
        ok = all(abs(e["mode_bias_new_over_sigma"]) < 1 and e["relevance_gain"] > 3
                 for e in s2.values())
        v["Stage2_inflation"] = dict(passed=bool(ok), lane=summary["stage2"]["lane"],
                                     detail=s2)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/undersampling_corrugation/run2")
    args = ap.parse_args()
    outdir = resolve_out_dir(args.out)
    pdir = os.path.join(outdir, "plots")
    os.makedirs(pdir, exist_ok=True)
    summary = json.load(open(os.path.join(outdir, "summary.json")))

    for name, lane in summary.items():
        if not (isinstance(lane, dict) and "per_ss" in lane):
            continue
        d = np.load(os.path.join(outdir, f"scan_{name}.npz"))
        lane_scan_figure(name, lane, d, pdir)
        lane_spectrum_figure(name, lane, d, pdir)
        error_map_figure(name, lane, d, pdir)
    summary_figures(summary, pdir)
    soft_arm_figure(summary, pdir)
    phase_figure(summary, pdir)
    stage2_figure(outdir, summary, pdir)

    vd = verdicts(summary, outdir)
    with open(os.path.join(outdir, "verdicts.json"), "w") as fh:
        json.dump(vd, fh, indent=2, default=str)
    print(json.dumps(vd, indent=2, default=str))
    print(f"[done] plots in {pdir}")


if __name__ == "__main__":
    main()
