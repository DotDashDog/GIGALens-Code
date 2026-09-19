"""Would a depleted core in the lens light be visible in these data? For each system's truth lens
Sersic (n, R_e, Ie, q, PA) render the central 41x41 px at ss=32, once as the pure Sersic and once
with a flat core inside r_b = {1, 2, 5}% R_e (I = I(r_b) for r < r_b: the maximal deficit for that
break radius; a core-Sersic with gamma > 0 loses less), bin, convolve with the system's PSF
(bin-first), and compare in units of the per-pixel sigma. Also the missing flux fraction and the
physical size of r_b."""
import os, json, numpy as np
from scipy.signal import fftconvolve
from gigalens_research.simtests.system import System
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = "/pscratch/sd/l/linusu/gigalens/simtests_results/vela_f140w_v3"
sids = json.load(open(BASE + "/dataset/manifest.json")); sids = sids["system_ids"] if isinstance(sids, dict) and "system_ids" in sids else [s if isinstance(s, str) else s["system_id"] for s in (sids["systems"] if isinstance(sids, dict) else sids)]
try:
    from astropy.cosmology import FlatLambdaCDM; cosmo = FlatLambdaCDM(70, 0.3); kpc = lambda z: cosmo.kpc_proper_per_arcmin(z).value / 60
except Exception: kpc = None
N, SS = 41, 32; dp = 0.065
def bn(n): return 2 * n - 1 / 3 + 4 / (405 * n) + 46 / (25515 * n ** 2)
out = {}
print(f"{'system':8s} {'n':>4s} {'R_e\"':>5s} {'q':>4s} | max |dI|/sigma at r_b = 1%, 2%, 5% R_e | chi2 of the difference (2%) | missing flux (2%) | r_b(2%) in px, and pc at z=0.5")
for sid in sids:
    sy = System.load(BASE + "/dataset", sid); LL = sy.truth_x[1][0]; n, Re, Ie = float(LL["n_sersic"]), float(LL["R_sersic"]), float(LL["Ie"])
    e1, e2 = float(LL["e1"]), float(LL["e2"]); e = np.hypot(e1, e2); q = (1 - e) / (1 + e); phi = 0.5 * np.arctan2(e2, e1)
    cx, cy = float(LL["center_x"]), float(LL["center_y"]); psf = np.asarray(sy.psf) if hasattr(sy, "psf") else np.load(f"{BASE}/dataset/systems/{sid}/psf.npy")
    # fine grid over the central N px (pixel centres at integer offsets from the frame centre), sub-pixel midpoints
    c = np.arange(N) - N // 2; sub = (np.arange(SS) + 0.5) / SS - 0.5
    xs = (c[:, None] + sub[None, :]).ravel() * dp - cx; ys = (c[:, None] + sub[None, :]).ravel() * dp - cy
    X, Y = np.meshgrid(xs, ys); xr = X * np.cos(phi) + Y * np.sin(phi); yr = -X * np.sin(phi) + Y * np.cos(phi)
    r = np.sqrt(q * xr ** 2 + yr ** 2 / q)   # elliptical radius, same area convention as gigalens' Sersic
    def render(rb):
        rr = np.maximum(r, rb) if rb > 0 else r
        im = Ie * dp ** 2 * np.exp(-bn(n) * ((rr / Re) ** (1 / n) - 1)); im = im.reshape(N, SS, N, SS).mean(axis=(1, 3))  # cps per pixel
        return fftconvolve(im, psf / psf.sum(), mode="same")
    base = render(0.0); sig = np.sqrt(sy.background_rms ** 2 + np.clip(base, 0, None) / sy.exp_time)
    res = {}
    for f in (0.01, 0.02, 0.05):
        im = render(f * Re); d = (im - base) / sig
        res[f] = dict(max_abs_dsig=float(np.abs(d).max()), chi2=float((d ** 2).sum()), missing_frac=float((base - im).sum() / (Ie * 2 * np.pi * n * Re ** 2 * q * np.exp(bn(n)) * bn(n) ** (-2 * n) * float(__import__("scipy.special", fromlist=["gamma"]).gamma(2 * n)))))
    rb2 = 0.02 * Re; out[sid] = dict(n=n, Re=Re, q=q, results={str(k): v for k, v in res.items()}, rb2_px=rb2 / dp, rb2_pc=(rb2 * kpc(0.5) * 1000 if kpc else None))
    print(f"{sid[:6]:8s} {n:4.2f} {Re:5.2f} {q:4.2f} | {res[0.01]['max_abs_dsig']:6.3f} {res[0.02]['max_abs_dsig']:6.3f} {res[0.05]['max_abs_dsig']:6.3f} | {res[0.02]['chi2']:7.3f} | {100*res[0.02]['missing_frac']:5.2f}% | {rb2/dp:5.2f} px, {out[sid]['rb2_pc'] or float('nan'):5.0f} pc")
json.dump(out, open(f"{HERE}/core_visibility.json", "w"), indent=1); print("done")
