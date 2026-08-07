"""Corrugation testbed: quadrature-aliasing scan on a single unlensed Sersic.

Design checkpoints DC-1/DC-2: docs/logs/undersampling-corrugation.md. DC-2 (run2)
implements the grader amendments: no-PSF lanes carry H1 (one rendering operator at
every ss), per-lane reference certification, softened-core discriminating arm,
truth-phase arm, and dimensionless stage-2 criteria.

Usage (login node, gigalens_env python, new-API gigalens on PYTHONPATH):
    python corrugation_scan.py            # full run2 grid (~10-20 min)
    python corrugation_scan.py --smoke    # 1 lane, short scan, API sanity only

Outputs land in $PSCRATCH/gigalens/results/undersampling_corrugation/<run>/ via
gigalens_research.paths.resolve_out_dir.
"""

import argparse
import copy
import functools
import json
import os
import subprocess
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from gigalens.simulator import SimulatorConfig
from gigalens.jax.profiles.light.sersic import Sersic
from gigalens.jax.scene import Component, Plane, LensModel
from gigalens.jax.scene_simulator import SceneSimulator
from gigalens_research.paths import resolve_out_dir

# ---------------------------------------------------------------- configuration
DELTA_PIX = 0.05          # arcsec / native pixel (HST-like)
NUM_PIX = 48
SS_SCAN = (1, 2, 4, 8)
SS_REF_NOPSF = 64         # DC-2: certified per lane via cert_gap (ss_ref vs ss_ref/2)
SS_REF_PSF = 32
PEAK_SNR = 50.0
SIGMA = 1.0
PSF_FWHM_PIX = 2.5
PHASE_DEFAULT = (0.30, 0.15)   # generic truth sub-pixel phase, native pixels
SCAN_HALF_PIX = 2.0            # DC-2 amendment 3: >=4 periods at ss=1
NSCAN = 1601
# stage-2 target: first reference-certified lane in this pre-declared order
STAGE2_PREFERENCE = ("n8_re0.5_nopsf", "n4_re0.5_nopsf", "n8_re1_nopsf", "n4_re1_nopsf")


class SoftenedSersic(Sersic):
    """Sersic with a softened center, r -> sqrt(r^2 + r_soft^2) (DC-2 arm 5).

    r_soft is a fixed (static) attribute, not a sampled parameter: each softening
    scale is its own profile instance. r_soft=0 must reproduce stock Sersic to
    float64 roundoff (gated in main()).
    """

    _name = "SERSIC_SOFT"

    def __init__(self, r_soft, use_lstsq=False, **kwargs):
        super().__init__(use_lstsq=use_lstsq, **kwargs)
        self.r_soft = float(r_soft)

    @functools.partial(jax.jit, static_argnums=(0,))
    def distance(self, x, y, cx, cy, e1=None, e2=None):
        r = super().distance(x, y, cx, cy, e1, e2)
        return jnp.sqrt(r * r + self.r_soft**2)


def lane_specs():
    lanes = []
    for n in (1, 4, 8):
        for re_pix in (0.5, 1.0, 3.0):
            lanes.append(dict(name=f"n{n}_re{re_pix:g}_nopsf", n=float(n), re_pix=re_pix,
                              psf=False, rc=0.0, phase=PHASE_DEFAULT,
                              ss_ref=SS_REF_NOPSF, role="primary"))
    lanes.append(dict(name="control_n1_re10_nopsf", n=1.0, re_pix=10.0, psf=False,
                      rc=0.0, phase=PHASE_DEFAULT, ss_ref=SS_REF_NOPSF, role="control"))
    for n in (1, 4, 8):
        for re_pix in (0.5, 1.0, 3.0):
            lanes.append(dict(name=f"n{n}_re{re_pix:g}_psf", n=float(n), re_pix=re_pix,
                              psf=True, rc=0.0, phase=PHASE_DEFAULT,
                              ss_ref=SS_REF_PSF, role="psf"))
    lanes.append(dict(name="control_n1_re10_psf", n=1.0, re_pix=10.0, psf=True,
                      rc=0.0, phase=PHASE_DEFAULT, ss_ref=SS_REF_PSF, role="control_psf"))
    for rc in (0.25, 0.5, 1.0):
        lanes.append(dict(name=f"soft_rc{rc:g}_n4_re0.5_nopsf", n=4.0, re_pix=0.5,
                          psf=False, rc=rc, phase=PHASE_DEFAULT,
                          ss_ref=SS_REF_NOPSF, role="soft"))
    for tag, ph in (("ph00", (0.0, 0.0)), ("ph47", (0.47, 0.31))):
        lanes.append(dict(name=f"phase_{tag}_n8_re0.5_nopsf", n=8.0, re_pix=0.5,
                          psf=False, rc=0.0, phase=ph,
                          ss_ref=SS_REF_NOPSF, role="phase"))
    return lanes


def gaussian_kernel(fwhm_pix: float, size: int = 17) -> np.ndarray:
    sig = fwhm_pix / 2.3548200450309493
    r = np.arange(size) - size // 2
    k = np.exp(-(r[:, None] ** 2 + r[None, :] ** 2) / (2.0 * sig**2))
    return k / k.sum()


# ------------------------------------------------------- model/simulator plumbing
def build_model(rc: float) -> LensModel:
    prof = Sersic(use_lstsq=False) if rc == 0.0 else SoftenedSersic(rc * DELTA_PIX)
    comp = Component(prof, dict(R_sersic=0.1, n_sersic=2.0, Ie=1.0,
                                center_x=0.0, center_y=0.0))
    return LensModel([Plane(light=[comp])])


def find_light_path(tree, path=()):
    if isinstance(tree, dict):
        if "center_x" in tree:
            return path
        for k, v in tree.items():
            p = find_light_path(v, path + (k,))
            if p is not None:
                return p
    return None


def set_light_params(params0, path, **vals):
    p = copy.deepcopy(params0)
    d = p
    for k in path:
        d = d[k]
    d.update(vals)
    return p


class SimBank:
    """Simulators + jitted scan/render fns, keyed (ss, psf_on, rc); built lazily."""

    def __init__(self, kernel):
        self.kernel = kernel
        self.models, self.paths0 = {}, {}
        self.scan, self.render = {}, {}

    def _model(self, rc):
        if rc not in self.models:
            m = build_model(rc)
            p0 = copy.deepcopy(m.constants)
            path = find_light_path(p0)
            assert path is not None
            self.models[rc] = (m, p0, path)
        return self.models[rc]

    def get(self, ss, psf_on, rc):
        key = (int(ss), bool(psf_on), float(rc))
        if key not in self.scan:
            model, params0, path = self._model(key[2])
            sim = SceneSimulator(model, SimulatorConfig(
                delta_pix=DELTA_PIX, num_pix=NUM_PIX, supersample=key[0],
                kernel=self.kernel if key[1] else None))

            def logL(cx, cy, R, n, Ie, data, sigma_map):
                p = set_light_params(params0, path, center_x=cx, center_y=cy,
                                     R_sersic=R, n_sersic=n, Ie=Ie)
                m = jnp.squeeze(sim.simulate(p))
                r = (m - data) / sigma_map
                return -0.5 * jnp.sum(r * r)

            def render(cx, cy, R, n, Ie):
                p = set_light_params(params0, path, center_x=cx, center_y=cy,
                                     R_sersic=R, n_sersic=n, Ie=Ie)
                return jnp.squeeze(sim.simulate(p))

            self.scan[key] = jax.jit(jax.value_and_grad(logL, argnums=0))
            self.render[key] = jax.jit(render)
        return self.scan[key], self.render[key]


# ------------------------------------------------------------------- analysis
def detrend(y, x, deg=6):
    c = np.polynomial.polynomial.polyfit(x, y, deg)
    return y - np.polynomial.polynomial.polyval(x, c)


def spectrum(resid, dx_pix):
    w = np.hanning(len(resid))
    p = np.abs(np.fft.rfft(resid * w)) ** 2
    f = np.fft.rfftfreq(len(resid), d=dx_pix)
    return f, p


def comb_stats(resid, dx_pix, f_expect):
    f, p = spectrum(resid, dx_pix)
    band = f > 0.4
    fb, pb = f[band], p[band]
    i = int(np.argmax(pb))
    med = float(np.median(pb))
    dfbin = f[1] - f[0]
    return dict(
        f_peak=float(fb[i]),
        peak_over_median=float(pb[i] / med) if med > 0 else np.inf,
        f_expect=float(f_expect),
        within_one_bin=bool(abs(fb[i] - f_expect) <= dfbin + 1e-12),
        df_bin=float(dfbin),
    )


def quad_fit_mode(x, y, half_win):
    i0 = int(np.argmax(y))
    m = np.abs(x - x[i0]) <= half_win
    c = np.polynomial.polynomial.polyfit(x[m], y[m], 2)
    if c[2] >= 0:
        return float(x[i0]), np.nan
    return float(-c[1] / (2 * c[2])), float(1.0 / np.sqrt(-2.0 * c[2]))


def scan_lane(fn, xs, cy, R, n, Ie, data, sigma_map):
    vals = np.empty(len(xs))
    grads = np.empty(len(xs))
    for i, x in enumerate(xs):
        v, g = fn(x, cy, R, n, Ie, data, sigma_map)
        vals[i], grads[i] = float(v), float(g)
    return vals, grads


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/undersampling_corrugation/run2")
    args = ap.parse_args()

    outdir = resolve_out_dir(args.out if not args.smoke else args.out + "_smoke")
    os.makedirs(outdir, exist_ok=True)
    print(f"[setup] outdir = {outdir}", flush=True)

    lanes = lane_specs()
    nscan = NSCAN
    if args.smoke:
        lanes = [la for la in lanes if la["name"] in ("n4_re0.5_nopsf",)]
        nscan = 41

    bank = SimBank(gaussian_kernel(PSF_FWHM_PIX))

    # ---- gate: SoftenedSersic(rc=0) == stock Sersic to float64 roundoff (DC-2 arm 5)
    _, r_stock = bank.get(1, False, 0.0)
    m_stock = build_model(0.0)
    soft0 = LensModel([Plane(light=[Component(
        SoftenedSersic(0.0), dict(R_sersic=0.1, n_sersic=2.0, Ie=1.0,
                                  center_x=0.0, center_y=0.0))])])
    sim0 = SceneSimulator(soft0, SimulatorConfig(
        delta_pix=DELTA_PIX, num_pix=NUM_PIX, supersample=1, kernel=None))
    pr0 = copy.deepcopy(soft0.constants)
    pth0 = find_light_path(pr0)
    a = np.asarray(jnp.squeeze(sim0.simulate(set_light_params(
        pr0, pth0, center_x=0.01, center_y=0.005, R_sersic=0.025, n_sersic=4.0, Ie=1.0))))
    b = np.asarray(r_stock(0.01, 0.005, 0.025, 4.0, 1.0))
    gate = float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))
    print(f"[gate] SoftenedSersic(0) vs Sersic max rel diff = {gate:.3e}", flush=True)
    assert gate < 1e-12, "softened-core gate failed; arm 5 invalid"

    meta = dict(
        delta_pix=DELTA_PIX, num_pix=NUM_PIX, ss_scan=list(SS_SCAN),
        ss_ref_nopsf=SS_REF_NOPSF, ss_ref_psf=SS_REF_PSF, peak_snr=PEAK_SNR,
        sigma=SIGMA, psf_fwhm_pix=PSF_FWHM_PIX, nscan=nscan,
        scan_half_pix=SCAN_HALF_PIX, jax=jax.__version__, cmd=" ".join(sys.argv),
        softened_gate_rel=gate,
    )
    for name, repo in (("gigalens_sha", os.path.expanduser("~/gigalens")),
                       ("gigalens_code_sha",
                        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))):
        try:
            meta[name] = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"],
                                        capture_output=True, text=True,
                                        check=True).stdout.strip()
        except Exception as e:  # noqa: BLE001
            meta[name] = f"unavailable: {e}"
    with open(os.path.join(outdir, "run_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    summary = {}
    for la in lanes:
        name, n_s, re_pix, psf_on, rc = la["name"], la["n"], la["re_pix"], la["psf"], la["rc"]
        ss_ref = la["ss_ref"]
        R = re_pix * DELTA_PIX
        cx_t, cy_t = la["phase"][0] * DELTA_PIX, la["phase"][1] * DELTA_PIX
        xs_pix = np.linspace(-SCAN_HALF_PIX, SCAN_HALF_PIX, nscan)
        xs = cx_t + xs_pix * DELTA_PIX
        dx_pix = xs_pix[1] - xs_pix[0]

        _, render_ref = bank.get(ss_ref, psf_on, rc)
        m1 = np.asarray(render_ref(cx_t, cy_t, R, n_s, 1.0))
        Ie = PEAK_SNR * SIGMA / float(m1.max())
        data = jnp.asarray(m1 * Ie)
        sigma_map = jnp.full_like(data, SIGMA)

        lane_res = dict(la, Ie=Ie)
        arrays = dict(xs_pix=xs_pix)

        truth_ss = sorted(set(SS_SCAN) | {2, 4, ss_ref // 2, ss_ref})
        truth_maps = {}
        for ss in truth_ss:
            _, rf = bank.get(ss, psf_on, rc)
            truth_maps[ss] = np.asarray(rf(cx_t, cy_t, R, n_s, Ie))
            arrays[f"truthmap_ss{ss}"] = truth_maps[ss]
        cert_gap = float(np.max(np.abs(truth_maps[ss_ref] - truth_maps[ss_ref // 2])) / SIGMA)
        lane_res["cert_gap"] = cert_gap
        lane_res["reference_limited"] = bool(cert_gap > 0.1)

        scans = {}
        for ss in list(SS_SCAN) + [ss_ref]:
            fn, _ = bank.get(ss, psf_on, rc)
            scans[ss] = scan_lane(fn, xs, cy_t, R, n_s, Ie, data, sigma_map)
            arrays[f"logL_ss{ss}"] = scans[ss][0]
            arrays[f"grad_ss{ss}"] = scans[ss][1]
        print(f"[scan] lane {name}: done (cert_gap={cert_gap:.3g}"
              f"{' REFERENCE-LIMITED' if lane_res['reference_limited'] else ''})", flush=True)

        vref, gref = scans[ss_ref]
        x_ref_hat, sig_ref = quad_fit_mode(xs_pix, vref, half_win=0.05)
        lane_res["x_hat_ref_pix"], lane_res["sigma_ref_pix"] = x_ref_hat, sig_ref
        lane_res["float64_floor"] = float(1e-13 * np.max(np.abs(vref)))
        per_ss = {}
        for ss in SS_SCAN:
            vals, grads = scans[ss]
            resid = detrend(vals - vref, xs_pix)
            gresid = detrend(grads - gref, xs_pix)
            st = comb_stats(resid, dx_pix, f_expect=float(ss))
            A = float(resid.max() - resid.min())
            Ag = float(gresid.max() - gresid.min())
            x_hat, _ = quad_fit_mode(xs_pix, vals, half_win=0.05)
            win = np.abs(xs_pix - x_ref_hat) <= (sig_ref if np.isfinite(sig_ref) else 0.05)
            A_win = float(resid[win].max() - resid[win].min()) if win.sum() > 2 else np.nan
            per_ss[ss] = dict(
                comb=st, A=A, A_grad=Ag,
                A_grad_pred=A * 2.0 * np.pi * ss / DELTA_PIX,
                A_within_1sigma=A_win,
                A_resolved=bool(A > 10 * lane_res["float64_floor"]),
                mode_shift_pix=float(x_hat - x_ref_hat),
                mode_shift_over_sigma=float((x_hat - x_ref_hat) / sig_ref)
                if np.isfinite(sig_ref) else np.nan,
            )
        lane_res["per_ss"] = per_ss
        summary[name] = lane_res
        np.savez_compressed(os.path.join(outdir, f"scan_{name}.npz"), **arrays)

    # ------------------------------------------------------------- stage 2 (DC-2 #7)
    stage2_lane = next((k for k in STAGE2_PREFERENCE
                        if k in summary and not summary[k]["reference_limited"]), None)
    if not args.smoke and stage2_lane is not None:
        print(f"[stage2] target lane: {stage2_lane}", flush=True)
        la = summary[stage2_lane]
        R, n_s, Ie, ss_ref = la["re_pix"] * DELTA_PIX, la["n"], la["Ie"], la["ss_ref"]
        cx_t, cy_t = la["phase"][0] * DELTA_PIX, la["phase"][1] * DELTA_PIX
        d = np.load(os.path.join(outdir, f"scan_{stage2_lane}.npz"))
        data = jnp.asarray(d[f"truthmap_ss{ss_ref}"])
        xs_pix = d["xs_pix"]
        xs = cx_t + xs_pix * DELTA_PIX
        stage2 = {}
        for ss in (1, 2):
            m_ss, m_2ss = d[f"truthmap_ss{ss}"], d[f"truthmap_ss{2*ss}"]
            sigma_render = np.abs(m_ss - m_2ss)
            sigma_eff = jnp.asarray(np.sqrt(SIGMA**2 + sigma_render**2))
            delta = m_ss - np.asarray(data)
            w_old = (delta / SIGMA) ** 2
            w_new = (delta / np.asarray(sigma_eff)) ** 2
            F_pred = float(w_new.sum() / w_old.sum()) if w_old.sum() > 0 else np.nan

            fn, _ = bank.get(ss, la["psf"], la["rc"])
            fr, _ = bank.get(ss_ref, la["psf"], la["rc"])
            vals, _ = scan_lane(fn, xs, cy_t, R, n_s, Ie, data, sigma_eff)
            vref2, _ = scan_lane(fr, xs, cy_t, R, n_s, Ie, data, sigma_eff)
            resid = detrend(vals - vref2, xs_pix)
            A_new = float(resid.max() - resid.min())
            A_old = la["per_ss"][ss]["A"]
            x_hat_new, _ = quad_fit_mode(xs_pix, vals, half_win=0.05)
            x_ref_new, sig_new = quad_fit_mode(xs_pix, vref2, half_win=0.05)
            sig_old = la["sigma_ref_pix"]
            stage2[ss] = dict(
                F_pred=F_pred, A_old=A_old, A_new=A_new,
                suppression=float(A_old / A_new) if A_new > 0 else np.inf,
                mode_bias_new_over_sigma=float((x_hat_new - x_ref_new) / sig_new)
                if np.isfinite(sig_new) else np.nan,
                mode_bias_old_over_sigma=la["per_ss"][ss]["mode_shift_over_sigma"],
                width_ratio=float(sig_new / sig_old),
                relevance_gain=float((A_old / A_new) / (sig_new / sig_old)),
                sigma_render_max=float(sigma_render.max()),
            )
            np.savez_compressed(
                os.path.join(outdir, f"stage2_{stage2_lane}_ss{ss}.npz"),
                xs_pix=xs_pix, logL=vals, logL_ref=vref2, sigma_render=sigma_render)
        summary["stage2"] = {str(k): v for k, v in stage2.items()}
        summary["stage2"]["lane"] = stage2_lane
        print("[stage2] done", flush=True)

    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"[done] wrote {outdir}/summary.json", flush=True)


if __name__ == "__main__":
    main()
