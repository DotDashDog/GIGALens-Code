"""Figure: three boundary treatments per source, same truth, same noise realization."""
import json, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.patches import Circle
from gigalens_research.simtests.experiments import vela_simulated as vs

T = os.path.dirname(os.path.abspath(__file__))  # reads boundary_treatments.json / .npz written next to this script
OUT = sys.argv[1]
D = json.load(open(os.path.join(T, "boundary_treatments.json")))
E = json.load(open(os.path.join(T, "boundary_edges.json"))) if os.path.exists(os.path.join(T, "boundary_edges.json")) else {}
Z = np.load(os.path.join(T, "boundary_treatments_images.npz"))
man = json.load(open("/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3/dataset/manifest.json"))["extra"]
dp, n = man["delta_pix"], man["num_pix"]
R0, R1, crop = D["R0"], D["R1"], D["crop"]
ids = list(D["rows"].keys())
cols = ["source (uncropped), crop / taper radii", f"lensed, hard crop {crop}\" + noise",
        f"lensed, cosine taper {R0}\"–{R1}\" + noise", "lensed, no crop + noise", "light the hard crop removes, in σ"]
fig, axes = plt.subplots(len(ids), 5, figsize=(17, 3.35 * len(ids)))
half = n * dp / 2
ext = [-half, half, -half, half]
for i, sid in enumerate(ids):
    r = D["rows"][sid]; e = E.get(sid, {})
    g = lambda k: Z[f"{sid}__{k}"]
    sig = g("sig"); rng = np.random.default_rng(100 + i)
    noise = rng.normal(0.0, 1.0, sig.shape) * sig
    full = g("full"); vmax = float(full.max())
    src = g("src_full"); s = r["fov"] / src.shape[0]
    cy, cx = vs._flux_centroid(src)
    zoom = 2.3
    hs = src.shape[0] * s / 2
    ax = axes[i, 0]
    ax.imshow(src, cmap="inferno", norm=PowerNorm(0.5, vmin=0, vmax=src.max()), origin="lower",
              extent=[-hs, hs, -hs, hs])
    c0 = ((cx - (src.shape[1] - 1) / 2) * s, (cy - (src.shape[0] - 1) / 2) * s)
    for rad, ls in ((crop, "-"), (R0, ":"), (R1, ":")):
        ax.add_patch(Circle(c0, rad, fill=False, ec="cyan", lw=0.8, ls=ls))
    ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom)
    ax.set_title(f"{sid}  θE={r['theta_E']:.2f}\"  amp={r['amp']:.2f}\nhard crop removes {100*r['f_removed_hard']:.1f}% of the source flux", fontsize=9)
    for j, key in enumerate(("hard", "taper", "full")):
        ax = axes[i, j + 1]
        ax.imshow(g(key) + noise, cmap="inferno", norm=PowerNorm(0.5, vmin=0, vmax=vmax), origin="lower", extent=ext)
        sn = {"hard": r["snr_removed_hard"], "taper": r["snr_removed_taper"], "full": 0.0}[key]
        pk = {"hard": r["peak_removed_hard"], "taper": r["peak_removed_taper"], "full": 0.0}[key]
        edge = e.get(key)
        t = f"removed light: S/N {sn:.1f}, peak {pk:.2f}σ/px" if key != "full" else f"frame-edge band: S/N {r['snr_band']:.1f}, peak {r['peak_band']:.2f}σ/px"
        if edge is not None:
            t += f"\nedge (high-pass on the lensed 1.5\" ring): S/N {edge:.1f}"
        ax.set_title(t, fontsize=8.5)
    ax = axes[i, 4]
    im = ax.imshow((full - g("hard")) / sig, cmap="inferno", vmin=0, vmax=3, origin="lower", extent=ext)
    ax.set_title("(no crop − hard crop) / σ per pixel, 0–3σ", fontsize=8.5)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([])
for j, c in enumerate(cols):
    axes[0, j].text(0.5, 1.28, c, transform=axes[0, j].transAxes, ha="center", fontsize=10, fontweight="bold")
fig.suptitle("Source-plane boundary treatments, v3 F140W review set (same truth, same noise realization across columns; lensed source only, sqrt stretch)", y=1.0, fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.995))
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print("wrote", OUT)
