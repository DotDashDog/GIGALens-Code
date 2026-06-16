#!/usr/bin/env python
"""Analyze the D-fix1 diag_scale sweep dumps: per-phase eps/step_norm/xi and per-window
mass-matrix eigenspectra (incl. first non-PSD step). Plots before metrics.

Run on login node (CPU, numpy only): no GPU/JAX needed.
"""
import os, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SWEEP = os.path.join(os.path.dirname(__file__), "diagnosis_2026-06/fix1/diag_sweep")

# phase boundaries for nb=2000 (frac 0.2/0.6/0.2), nr=500
NB, NR = 2000, 500
T1 = round(NB * 0.2)            # 400  end of tune1
T2 = T1 + round(NB * 0.6)       # 1600 end of tune2
T3 = NB                         # 2000 end of tune3
# mass-matrix windows inside tune2
num_mm = round(0.67 * round(NB * 0.6))   # 804
mm_start = T1
ratios = [1, 2, 4]; tot = sum(ratios)
wsz = [max(1, round(num_mm * r / tot)) for r in ratios]; wsz[-1] = num_mm - sum(wsz[:-1])
wends, p = [], mm_start
for w in wsz:
    p += w; wends.append(p - 1)
# wends ~ [514, 744, 1203]

def phase_slices(n):
    return {
        "tune1": slice(0, T1),
        "tune2": slice(T1, T2),
        "tune3": slice(T2, T3),
        "sampling": slice(T3, n),
    }

def _dump_dir(tag):
    # dumps land in SWEEP/tag/nmax..._timestamp/hist_*.npy
    cands = glob.glob(os.path.join(SWEEP, tag, "*", "hist_step_size.npy"))
    if cands:
        return os.path.dirname(sorted(cands)[-1])
    # fallback: directly under tag
    if os.path.exists(os.path.join(SWEEP, tag, "hist_step_size.npy")):
        return os.path.join(SWEEP, tag)
    return None

def load(tag):
    d = _dump_dir(tag)
    out = {}
    for f in ["step_size", "step_norm", "xi", "inverse_mass_matrix", "kernel_nonan", "energy_change_raw"]:
        p = os.path.join(d, f"hist_{f}.npy") if d else None
        out[f] = np.load(p) if (p and os.path.exists(p)) else None
    return out

def metric_eigs(imm):
    # imm: (chains, steps, dim, dim); chains share the metric -> use chain 0
    M = imm[0]  # (steps, dim, dim)
    eigs = np.linalg.eigvalsh(M.astype(np.float64))  # ascending
    return eigs  # (steps, dim)

def summarize(tag):
    h = load(tag)
    if h["step_size"] is None:
        return None
    eps = h["step_size"]       # (chains, steps)
    sn = h["step_norm"]
    xi = h["xi"]
    knan = h["kernel_nonan"]
    n = eps.shape[1]
    ps = phase_slices(n)
    rec = {"tag": tag, "n_steps": n}
    for name, sl in ps.items():
        e = eps[:, sl]
        rec[name] = {
            "eps_med": float(np.nanmedian(e)),
            "eps_end": float(np.nanmedian(eps[:, sl.stop - 1])) if sl.stop and sl.stop <= n else None,
            "step_norm_med": float(np.nanmedian(sn[:, sl])) if sn is not None else None,
            "xi_q50": float(np.nanmedian(xi[:, sl])) if xi is not None else None,
            "xi_q90": float(np.nanpercentile(xi[:, sl], 90)) if xi is not None else None,
            "kernel_reject_frac": float(1.0 - np.nanmean(knan[:, sl])) if knan is not None else None,
        }
    # mass matrix eigenspectra at window boundaries
    if h["inverse_mass_matrix"] is not None:
        eigs = metric_eigs(h["inverse_mass_matrix"])  # (steps, dim)
        rec["windows"] = {}
        for k, we in enumerate(wends):
            s = min(we + 1, n - 1)
            ev = eigs[s]
            rec["windows"][f"w{k+1}_end{we}"] = {
                "min_eig": float(ev.min()), "max_eig": float(ev.max()),
                "cond": float(ev.max() / max(ev.min(), 1e-300)),
                "n_neg": int((ev < 0).sum()),
            }
        # first non-PSD step overall
        neg_steps = np.where((eigs < 0).any(axis=1))[0]
        rec["first_nonpsd_step"] = int(neg_steps[0]) if neg_steps.size else None
        rec["min_eig_ever"] = float(eigs.min())
        rec["_eigs"] = eigs  # for plotting, stripped before json
    return rec

def plot_tag(tag, rec, h):
    eps, sn, xi = h["step_size"], h["step_norm"], h["xi"]
    eigs = rec.get("_eigs")
    n = eps.shape[1]
    fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
    for c in range(eps.shape[0]):
        ax[0].semilogy(eps[c], lw=0.7, alpha=0.7)
    ax[0].set_ylabel("step size"); ax[0].set_title(f"{tag}: step size")
    for c in range(sn.shape[0]):
        ax[1].semilogy(np.maximum(sn[c], 1e-30), lw=0.7, alpha=0.7)
    ax[1].set_ylabel("step_norm"); ax[1].set_title("position step norm")
    ax[2].semilogy(np.nanmedian(np.abs(xi), axis=0), 'k', lw=1)
    ax[2].axhline(1.0, color='r', ls='--', lw=0.8)
    ax[2].set_ylabel("median |xi|"); ax[2].set_title("energy-error ratio")
    if eigs is not None:
        ax[3].semilogy(np.maximum(eigs.min(axis=1), 1e-30), label="min")
        ax[3].semilogy(eigs.mean(axis=1), 'k', label="mean")
        ax[3].semilogy(eigs.max(axis=1), 'r', label="max")
        npsd = rec.get("first_nonpsd_step")
        if npsd is not None:
            ax[3].axvline(npsd, color='magenta', ls=':', label=f"non-PSD@{npsd}")
        ax[3].legend(fontsize=7)
    ax[3].set_ylabel("M^-1 eig"); ax[3].set_title("inverse mass-matrix eigenvalues")
    for a in ax:
        for we in wends:
            a.axvline(we, color='gray', ls=':', lw=0.5)
        a.axvline(T1, color='brown', ls='--', lw=0.6)
        a.axvline(T2, color='blue', ls='--', lw=0.6)
    ax[-1].set_xlabel("step")
    fig.tight_layout()
    out = os.path.join(SWEEP, f"{tag}_panels.png")
    fig.savefig(out, dpi=110); plt.close(fig)
    return out

def main():
    tags = sorted([os.path.basename(d) for d in glob.glob(os.path.join(SWEEP, "*")) if os.path.isdir(d)])
    recs = {}
    for tag in tags:
        rec = summarize(tag)
        if rec is None:
            print(f"[skip] {tag}: no dump")
            continue
        h = load(tag)
        plot_tag(tag, rec, h)
        rec.pop("_eigs", None)
        recs[tag] = rec
        w = rec.get("windows", {})
        print(f"\n=== {tag} ===")
        for ph in ["tune1", "tune2", "tune3", "sampling"]:
            r = rec[ph]
            print(f"  {ph:9s} eps_med={r['eps_med']:.2e} step_norm={r['step_norm_med']:.2e} "
                  f"xi50={r['xi_q50']:.2e} xi90={r['xi_q90']:.2e} rej={r['kernel_reject_frac']:.3f}")
        for wn, wv in w.items():
            print(f"  {wn}: min_eig={wv['min_eig']:.2e} cond={wv['cond']:.2e} n_neg={wv['n_neg']}")
        print(f"  first_nonpsd_step={rec.get('first_nonpsd_step')} min_eig_ever={rec.get('min_eig_ever'):.2e}")
    with open(os.path.join(SWEEP, "sweep_summary.json"), "w") as f:
        json.dump(recs, f, indent=2)
    print(f"\nwrote {os.path.join(SWEEP, 'sweep_summary.json')}")

if __name__ == "__main__":
    main()
