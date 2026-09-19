"""Lens-light cusp convergence ladder (2026-09-15).

For the central 8 x 8 pixels of the 120 px frame, integrate the lens Sersic per
pixel with the midpoint rule at supersample s = 4 ... 4096 on exactly the
LensWCS sub-pixel grid (coordinate (m - (N s - 1)/2) * delta / s), using
gigalens' own SersicEllipse.light so the profile definition matches. The 4096
rung is the reference (self-checked against 2048). The error patch of each rung
is convolved with the system PSF and expressed in units of the DESI-238
background rms, which is the quantity that matters for the observed data.
"""
import os, sys, json, pickle, time
import numpy as np
import jax, jax.numpy as jnp
from scipy.signal import fftconvolve
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from gigalens.jax.profiles.light.sersic import SersicEllipse

W = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DS = "/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset/systems"
N, DELTA, SIG, EXP = 120, 0.065, 0.0076, 1197.7
PATCH = 8; J0 = N // 2 - PATCH // 2          # pixels 56..63
RUNGS = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
light = SersicEllipse(use_lstsq=False)

@jax.jit
def _pix_mean(x, y, R, n, e1, e2, cx, cy, Ie):
    return jnp.mean(light.light(x, y, R, n, e1, e2, cx, cy, Ie))

def patch_at(s, p):
    """Per-pixel mean of the profile over s x s midpoints, times pixel area -> cps/px."""
    out = np.zeros((PATCH, PATCH))
    k = (np.arange(s) + 0.5) * DELTA / s
    for a in range(PATCH):
        for b in range(PATCH):
            xs = (J0 + b - N / 2) * DELTA + k         # x along columns
            ys = (J0 + a - N / 2) * DELTA + k         # y along rows
            X, Y = np.meshgrid(xs, ys)
            out[a, b] = float(_pix_mean(jnp.asarray(X), jnp.asarray(Y), p["R_sersic"], p["n_sersic"],
                                        p["e1"], p["e2"], p["center_x"], p["center_y"], p["Ie"]))
    return out * DELTA ** 2

def full_frame_lens(p, s=32):
    """Unconvolved lens light on the whole frame at supersample s (cps/px), row-chunked."""
    out = np.zeros((N, N))
    k = (np.arange(s) + 0.5) * DELTA / s
    for a in range(N):
        ys = (a - N / 2) * DELTA + k
        xs = (np.arange(N)[:, None] - N / 2) * DELTA + k[None, :]      # (N, s)
        X = np.broadcast_to(xs[:, None, :], (N, s, s)); Y = np.broadcast_to(ys[None, :, None], (N, s, s))
        v = light.light(jnp.asarray(X), jnp.asarray(Y), p["R_sersic"], p["n_sersic"], p["e1"], p["e2"],
                        p["center_x"], p["center_y"], p["Ie"])
        out[a] = np.asarray(jnp.mean(v, axis=(1, 2)))
    return out * DELTA ** 2

def load_truth(sim):
    t = pickle.load(open(f"{DS}/vela{sim}_cam12_a0.400_rep00/truth_x.pkl", "rb"))[1][0]
    return {k: float(v) for k, v in t.items()}

psf = np.load(f"{DS}/vela08_cam12_a0.400_rep00/psf.npy")
cases = {f"vela{s}": load_truth(s) for s in ["08", "09", "23", "22"]}
noiseless = {f"vela{s}": np.load(f"{DS}/vela{s}_cam12_a0.400_rep00/noiseless_image.npy").astype(np.float64) for s in ["08", "09", "23", "22"]}
cases["n6_corner"] = dict(R_sersic=1.0, n_sersic=6.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0, Ie=20.0)
cases["n6_pixcentre"] = dict(R_sersic=1.0, n_sersic=6.0, e1=0.0, e2=0.0, center_x=DELTA / 2, center_y=DELTA / 2, Ie=20.0)
res = {}
for name, p in cases.items():
    t0 = time.time()
    if name not in noiseless:
        noiseless[name] = fftconvolve(full_frame_lens(p), psf, mode="same")
    sig_tot = np.sqrt(SIG ** 2 + np.clip(noiseless[name], 0, None) / EXP)
    print(f"{name}: peak {noiseless[name].max():.2f} cps/px -> sigma_tot at peak {sig_tot.max():.4f} = {sig_tot.max()/SIG:.1f} x bkg", flush=True)
    patches = {s: patch_at(s, p) for s in RUNGS}
    ref = patches[4096]
    row = {"n_sersic": p["n_sersic"], "R_sersic": p["R_sersic"], "center": [p["center_x"], p["center_y"]],
           "ref_self_check_prepsf_sigma": float(np.abs(patches[2048] - ref).max() / SIG), "rungs": {}}
    for s in RUNGS[:-1]:
        err = np.zeros((N, N)); err[J0:J0 + PATCH, J0:J0 + PATCH] = patches[s] - ref
        post_abs = fftconvolve(err, psf, mode="same")
        post = post_abs / SIG
        row["rungs"][s] = {"prepsf_max_sigma": float(np.abs(err).max() / SIG), "postpsf_max_sigma": float(np.abs(post).max()),
                           "postpsf_max_sigma_total": float(np.abs(post_abs / sig_tot).max()),
                           "postpsf_chi2": float(((post_abs / sig_tot) ** 2).sum()),
                           "flux_err_cps": float(err.sum())}
    row["peak_cps"] = float(noiseless[name].max()); row["sigma_tot_peak_over_bkg"] = float(sig_tot.max() / SIG)
    res[name] = row
    print(f"{name:13s} n={p['n_sersic']:.2f} Re={p['R_sersic']:.2f} ref self-check {row['ref_self_check_prepsf_sigma']:.4f}σ  "
          + "  ".join(f"s{s}:{row['rungs'][s]['postpsf_max_sigma']:.3f}/{row['rungs'][s]['postpsf_max_sigma_total']:.3f}" for s in [4, 16, 32, 64, 128, 256, 512, 1024])
          + f"   ({time.time()-t0:.0f}s)", flush=True)
json.dump(res, open(os.path.join(os.path.dirname(__file__), "cusp_ladder.json"), "w"), indent=1)
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
for ax, key, lab in [(axes[0], "postpsf_max_sigma", "background rms only (0.0076 cps/px)"), (axes[1], "postpsf_max_sigma_total", "total per-pixel noise (bkg + Poisson)")]:
    for name, row in res.items():
        ss = sorted(int(s) for s in row["rungs"]); ax.plot(ss, [row["rungs"][s][key] for s in ss], "o-", label=f"{name} (n={row['n_sersic']:.1f})")
    ax.axhline(0.1, color="k", ls="--", lw=0.8); ax.text(4.2, 0.11, "0.1 σ", fontsize=9)
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("supersample"); ax.set_ylabel("max |error| after PSF  [σ]")
    ax.set_title(f"Lens-light cusp error vs supersample — σ = {lab}", fontsize=11); ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(os.path.join(os.path.dirname(__file__), "cusp_ladder.png"), dpi=100)
