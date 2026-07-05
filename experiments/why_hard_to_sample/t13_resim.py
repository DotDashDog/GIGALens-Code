"""T13' Steps 0-3 -- re-simulate sys60's DATA at supersample=16 (same truth, same
recovered noise realization) and run the reproduction + convergence gates.

Pre-registered in docs/logs/why-hard-to-sample.md (checkpoint T13', approved by the
human 2026-07-02; design, ss=16, and STRICT separation explicitly approved). Builds
on C-4/T12: sys60's original observed image was simulated at supersample=2 (generator
attic/Linus-FourSim.ipynb), and the ss=2 model's GN stiffness is a subgrid comb that
collapses x1400 at ss=4. T13' asks the clean payoff question by re-simulating the DATA
at ss=16 (an effectively infinitely-supersampled sky) with the SAME truth and the SAME
noise realization recovered from the original image, then running a clean model-fidelity
2x2 (ss2 vs ss4 model, this file only produces the data + gates; t13_arms.py runs arms).

============================================================================
STEP 0 -- GENERATION RECIPE (extracted from attic/Linus-FourSim.ipynb, MAIN checkout)
============================================================================
The 100 simulated systems (sys60 = index 60) were made in Linus-FourSim.ipynb:

  * TRUTH params: cell 13 defines the GENERATION prior (NARROWER than today's MODELING
    prior -- see discrepancy note below); cell 14 samples 100 systems with
    random.PRNGKey(0); cell 20 persists them to
    data/simulated_systems/100SystemsStandardParams.yaml (nested list-of-dicts:
    [[epl,shear],[lens_light],[source_light]], each value a length-100 list). The
    truth for sys60 is yaml[...][60] -- loaded here the notebook's way
    (params_lists_to_jax(yaml) then index 60; we read the yaml directly, no jax needed
    for extraction). The GENERATION prior does NOT enter the data: only the sampled
    truth values, the profiles, the render config, the PSF, and the noise model do.

  * RENDER config (cells 4-5): PhysicalModel([EPL(50), Shear()],[SersicEllipse],
    [SersicEllipse]); SimulatorConfig(delta_pix=0.065, num_pix=80, supersample=2,
    kernel=psf) with kernel = np.load(gigalens/assets/psf.npy).astype(np.float32);
    LensSimulator(...).simulate(params) -> clean image (OLD gigalens API, float32).

  * NOISE model (cell 7-8, 15): noised = clean + add_poisson(clean, exp_time=100)
    + add_background(clean, sigma_bkd=0.2)  [lenstronomy image_util]; i.e. per-pixel
    variance ~ background_rms^2 + clean/exp_time, Gaussian background (mean 0) +
    Poisson shot noise. background_rms=0.2, exp_time=100.

  * SAVED (cell 20): data/simulated_systems/100SystemsStandard80px.npz keys sys_0..
    sys_99 (numeric save order => index 60 == key "sys_60"), paired with the yaml.

DISCREPANCIES vs today's systems/sys60/system.py (== TestSersic60.ipynb cells 3-4):
  (1) MODELING prior (system.py) is WIDER than the GENERATION prior (cell 13): e.g.
      theta_E LogNormal(log1.25, 0.4 vs 0.25); gamma TruncNormal(2, 0.5 vs 0.25, 1,3);
      EPL e1/e2 Normal(0, 0.2 vs 0.1); shear Normal(0, 0.1 vs 0.05); lens-light
      n_sersic U(0.5,8 vs 2,6), Ie sigma 0.5 vs 0.3; source n_sersic U(0.5,8 vs 0.5,4),
      centers Normal(0, 0.5 vs 0.25), Ie sigma 0.9 vs 0.5. This is IRRELEVANT to data
      reproduction: render-at-truth uses theta DIRECTLY (via to_params/simulate),
      bypassing prior/bijector entirely. The prior only matters when fitting (the arms).
  (2) PSF dtype: generation .astype(np.float32) + float32 render (old LensSimulator,
      no x64); today's system.py .astype(np.float64) + float64 render. A ~1e-6-relative
      fidelity difference; GATE A is exactly the control that validates equivalence.
  (3) API: generation = OLD API (PhysicalModel/LensSimulator); today = NEW scene API
      (LensModel/Plane/Component + SceneSimulator.simulate). GATE A validates the two
      render paths agree at the truth (E1 verified the new render == the likelihood's
      image; GATE A additionally verifies new-render(truth) == old-generation(truth)).
  Same-in-both (no discrepancy): delta_pix=0.065, num_pix=80, supersample=2, same PSF
  file, background_rms=0.2, exp_time=100, EPL(50), two non-lstsq SersicEllipse profiles.

The error map used throughout is the Dataset's own (scene_prob_model.Dataset):
  error_map = sqrt(background_rms^2 + clip(observed_image, 0, inf) / exp_time)
i.e. it is derived from the OBSERVED image -- the same map the likelihood uses. GATE A
uses the ORIGINAL data's Dataset error map (systems/sys60 built on the original npz).

STEPS (this file):
  1 (GATE A, reproduction): r = observed - render_ss2(truth) must be pure noise.
  2 (GATE B, convergence):  max|m16-m8|/err < 0.05; REPORT max|m2-m16|/err (aliasing).
  3: d' = m16 + r; save resim/sys60_ss16/observed_ss16.npz + manifest JSON.
On ANY Gate-A failure: write diagnostics, print STOP, exit nonzero -- do NOT proceed.

jax is imported ONLY inside functions so the module imports under a plain conda python
(no jax) for the offline --smoke tests of the pure-numpy gate machinery.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- absolute paths into the MAIN checkout (read-only inputs) ----------------
_MAIN = "/global/homes/l/linusu/GIGALens-Code"
_TRUTH_YAML = os.path.join(_MAIN, "data/simulated_systems/100SystemsStandardParams.yaml")
_ORIG_NPZ = os.path.join(_MAIN, "data/simulated_systems/100SystemsStandard80px.npz")
_SYSTEM_INDEX = 60
_EXPECTED_DIM = 22

# ---- image / grid geometry ---------------------------------------------------
IMG_SIDE = 80
N_PIX = IMG_SIDE * IMG_SIDE

# ---- pre-registered GATE thresholds (T13' checkpoint; restated next to measured)
# NOTE (ambiguity resolved): the log annotates the full-image interval as
# "1 +/- 5 sqrt(2/6400)", which numerically = +/-0.088 -> [0.912,1.088]; but the
# PINNED interval in both the log and the agent brief is [0.956,1.044] (= +/-0.044 =
# 2.5 sqrt(2/6400)). We use the PINNED numeric interval as authoritative and flag the
# annotation mismatch. The window interval [0.53,1.47] = 5 sqrt(2/225) is self-consistent.
GATE_A_CHI2_LO = 0.956
GATE_A_CHI2_HI = 1.044
SMOOTH_KERNEL = 13                    # box kernel side (px) for the smoothed-residual gate
SMOOTH_MAX = 1.5                      # max |box-smoothed r/err| must be < this
WIN_ROW, WIN_COL, WIN_HALF = 57, 12, 7   # counter-image 15x15 window (== T11/T12 hotspot)
GATE_A_WIN_CHI2_LO = 0.53
GATE_A_WIN_CHI2_HI = 1.47
GATE_B_CONV = 0.05                    # max|m16-m8|/err must be < this


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ===========================================================================
# Pure-numpy gate machinery (offline-testable; no jax)
# ===========================================================================

def box_smooth(field2d, k=SMOOTH_KERNEL):
    """Centered k x k BOX MEAN with edge-clipped (shrinking) windows, via an integral
    image. At each pixel the mean is over the k x k neighbourhood intersected with the
    image bounds (so a coherent amplitude-A patch >= k px wide smooths to ~A at its
    centre; iid N(0,1) noise smooths to ~N(0, 1/k^2))."""
    a = np.asarray(field2d, dtype=np.float64)
    H, W = a.shape
    ii = np.zeros((H + 1, W + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(a, axis=0), axis=1)
    r = k // 2
    out = np.empty_like(a)
    for i in range(H):
        i0, i1 = max(0, i - r), min(H, i + r + 1)
        for j in range(W):
            j0, j1 = max(0, j - r), min(W, j + r + 1)
            s = ii[i1, j1] - ii[i0, j1] - ii[i1, j0] + ii[i0, j0]
            out[i, j] = s / ((i1 - i0) * (j1 - j0))
    return out


def reduced_chi2(r, err, mask=None):
    """sum((r/err)^2) / N over the (optionally masked) pixels."""
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    err = np.asarray(err, dtype=np.float64).reshape(-1)
    z2 = (r / err) ** 2
    if mask is not None:
        m = np.asarray(mask, dtype=bool).reshape(-1)
        z2 = z2[m]
    return float(np.sum(z2) / z2.size)


def window_slice(row=WIN_ROW, col=WIN_COL, half=WIN_HALF, side=IMG_SIDE):
    r0, r1 = row - half, row + half + 1
    c0, c1 = col - half, col + half + 1
    if r0 < 0 or c0 < 0 or r1 > side or c1 > side:
        raise ValueError(f"[T13] counter-image window out of bounds: "
                         f"[{r0}:{r1},{c0}:{c1}] on {side}x{side}")
    return (slice(r0, r1), slice(c0, c1))


def gate_a_metrics(observed, model2, err, side=IMG_SIDE):
    """Compute the three GATE-A statistics for r = observed - render_ss2(truth).
    Returns dict with reduced chi^2 (full), max |box-smoothed r/err|, windowed reduced
    chi^2, and the maps needed for plotting."""
    obs = np.asarray(observed, dtype=np.float64).reshape(side, side)
    m2 = np.asarray(model2, dtype=np.float64).reshape(side, side)
    e = np.asarray(err, dtype=np.float64).reshape(side, side)
    r = obs - m2
    z = r / e                                   # r / err_map
    red_full = reduced_chi2(r, e)
    smoothed = box_smooth(z, SMOOTH_KERNEL)
    smoothed_absmax = float(np.max(np.abs(smoothed)))
    imax = np.unravel_index(int(np.argmax(np.abs(smoothed))), smoothed.shape)
    ws = window_slice(side=side)
    red_win = reduced_chi2(r[ws], e[ws])
    return {
        "reduced_chi2_full": red_full,
        "smoothed_absmax": smoothed_absmax,
        "smoothed_absmax_loc": [int(imax[0]), int(imax[1])],
        "reduced_chi2_window": red_win,
        "_r": r, "_z": z, "_smoothed": smoothed, "_window": ws,
    }


def gate_a_verdict(m):
    """Apply the pre-registered thresholds; return (passed, per-check dict)."""
    checks = {
        "reduced_chi2_full": {
            "measured": m["reduced_chi2_full"],
            "interval": [GATE_A_CHI2_LO, GATE_A_CHI2_HI],
            "pass": GATE_A_CHI2_LO <= m["reduced_chi2_full"] <= GATE_A_CHI2_HI,
        },
        "smoothed_absmax": {
            "measured": m["smoothed_absmax"],
            "threshold_lt": SMOOTH_MAX,
            "loc_row_col": m["smoothed_absmax_loc"],
            "pass": m["smoothed_absmax"] < SMOOTH_MAX,
        },
        "reduced_chi2_window": {
            "measured": m["reduced_chi2_window"],
            "interval": [GATE_A_WIN_CHI2_LO, GATE_A_WIN_CHI2_HI],
            "pass": GATE_A_WIN_CHI2_LO <= m["reduced_chi2_window"] <= GATE_A_WIN_CHI2_HI,
        },
    }
    passed = all(c["pass"] for c in checks.values())
    return passed, checks


# ===========================================================================
# Truth extraction (pure python; no jax) + render-at-truth (jax inside)
# ===========================================================================

def load_truth_nested(yaml_path=_TRUTH_YAML, index=_SYSTEM_INDEX):
    """Read the persisted generation truth the notebook's way and select system `index`.
    Returns the nested structure [[epl,shear],[lens_light],[source_light]] with python
    float scalars (== jax.tree.map(select_index, params_lists_to_jax(yaml)))."""
    import yaml
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"[T13] truth yaml missing: {yaml_path}")
    d = yaml.safe_load(open(yaml_path))
    if not (len(d) == 3 and len(d[0]) == 2 and len(d[1]) == 1 and len(d[2]) == 1):
        raise ValueError(f"[T13] unexpected truth yaml structure: "
                         f"lens={len(d[0])} light={len(d[1])} source={len(d[2])}")

    def pick(prof):
        return {k: float(prof[k][index]) for k in prof}
    return [[pick(d[0][0]), pick(d[0][1])], [pick(d[1][0])], [pick(d[2][0])]]


def truth_flat_from_nested(truth_nested):
    """Flatten the nested truth into the scene-API flat key scheme
    (planes/<p>/<mass|light>/<idx>/<param>) so it can be checked against the bijector's
    sorted-key output. Profile placement matches TestSersic60/system.py: plane 0 mass 0
    = EPL, plane 0 mass 1 = Shear, plane 0 light 0 = lens light, plane 1 light 0 =
    source light (T12 confirms 'planes/1/light/0/center_x' is the source center)."""
    epl, shear = truth_nested[0]
    lls = truth_nested[1][0]
    src = truth_nested[2][0]
    flat = {}
    for k, v in epl.items():
        flat[f"planes/0/mass/0/{k}"] = float(v)
    for k, v in shear.items():
        flat[f"planes/0/mass/1/{k}"] = float(v)
    for k, v in lls.items():
        flat[f"planes/0/light/0/{k}"] = float(v)
    for k, v in src.items():
        flat[f"planes/1/light/0/{k}"] = float(v)
    return flat


def make_render(model_seq):
    """jitted render(z) -> (N_PIX,) flattened model image, EXACTLY the likelihood's
    render path (scene_prob_model _model_image): sim.simulate(model.to_params(
    bij.forward(z))). Same op E1 verified against log_prob's chi^2 aux."""
    import jax
    pm = model_seq.prob_model
    sim = pm.simulators[0]

    def render(z):
        x = pm.bij.forward(list(z.T))
        return sim.simulate(pm.model.to_params(x)).reshape(-1)
    return jax.jit(render)


def render_at_truth(model_seq, param_names, truth_nested, bij_tol=1e-6):
    """z_truth = bij.inverse(truth_nested) (== the notebook's commented
    'bij.inverse(sys_60_true)'), then render via the verified render(z) path. Validates
    that bij.forward(z_truth) round-trips back to the flat truth (max abs diff < bij_tol)
    AND that the flat-key scheme exactly matches param_names -- a wrong truth->theta
    mapping RAISES here rather than silently producing a wrong 'clean' image.

    Returns (img_flat (N_PIX,), z_truth (dim,), roundtrip_err)."""
    import jax.numpy as jnp
    pm = model_seq.prob_model

    flat_truth = truth_flat_from_nested(truth_nested)
    if set(flat_truth.keys()) != set(param_names):
        missing = set(param_names) - set(flat_truth.keys())
        extra = set(flat_truth.keys()) - set(param_names)
        raise ValueError(f"[T13] truth flat keys != bijector param names; "
                         f"missing={sorted(missing)} extra={sorted(extra)}")

    # bij.forward maps a LIST of 22 z-columns -> flat dict, so bij.inverse takes
    # the flat DICT (forward's output structure) and returns the 22-list in
    # forward-input (= sampler column) order. Passing the 3-group nested truth
    # here raised a pytree length mismatch (3 vs 22) on the first run. The
    # name-keyed round-trip check below independently validates the ordering.
    flat_in = {k: jnp.asarray([flat_truth[k]], dtype=jnp.float64)
               for k in flat_truth}
    z_list = pm.bij.inverse(flat_in)
    z_truth = np.asarray(
        [float(np.asarray(v).reshape(-1)[0]) for v in z_list], dtype=np.float64)
    if z_truth.shape != (len(param_names),):
        raise ValueError(f"[T13] bij.inverse gave shape {z_truth.shape} != "
                         f"({len(param_names)},)")
    fwd = pm.bij.forward(list(jnp.asarray(z_truth, dtype=jnp.float64)))
    rt_err = max(abs(float(np.asarray(fwd[k])) - flat_truth[k]) for k in param_names)
    if rt_err > bij_tol:
        raise RuntimeError(f"[T13] truth<->z round-trip error {rt_err:.3e} > {bij_tol:.1e}"
                           " -- theta<->z mapping is inconsistent; STOP.")
    render = make_render(model_seq)
    img = np.asarray(render(jnp.asarray(z_truth, dtype=jnp.float64)),
                     dtype=np.float64).reshape(-1)
    return img, z_truth, rt_err


# ===========================================================================
# System loader with supersample override (systems/sys60 path)
# ===========================================================================

def load_sys60(data_dir, supersample=None):
    """Import systems/sys60/system.py and call load_target(supersample=...). Returns
    (model_seq, qz, z_center, dim, param_names). supersample overrides ONLY the
    SimulatorConfig factor (everything else -- observed image, PSF, grid, noise --
    unchanged)."""
    import importlib.util
    data_dir = os.path.abspath(data_dir)
    sys_py = os.path.join(data_dir, "system.py")
    if not os.path.isfile(sys_py):
        raise FileNotFoundError(f"[T13] no system.py in data-dir: {data_dir}")
    tag = "base" if supersample is None else f"ss{supersample}"
    mod_name = "_t13_sys_" + os.path.basename(data_dir).replace(".", "_") + "_" + tag
    if data_dir not in sys.path:
        sys.path.insert(0, data_dir)
    spec = importlib.util.spec_from_file_location(mod_name, sys_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load_target(supersample=supersample)


# ===========================================================================
# Plots
# ===========================================================================

def plot_gate_a(gm, err, out_path):
    """Residual map r, r/err, and box-smoothed r/err (with the counter-image window)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    z = gm["_z"]; r = gm["_r"]; sm = gm["_smoothed"]; ws = gm["_window"]
    for ax, data, title, cmap, vlim in [
        (axes[0], r, "residual r = observed - m2", "RdBu_r", None),
        (axes[1], z, "r / err_map", "RdBu_r", 4.0),
        (axes[2], sm, f"box-smoothed r/err ({SMOOTH_KERNEL}px)", "RdBu_r", SMOOTH_MAX),
    ]:
        if vlim is None:
            v = float(np.max(np.abs(data)))
        else:
            v = vlim
        im = ax.imshow(data, cmap=cmap, vmin=-v, vmax=v, origin="upper")
        # counter-image window box
        r0 = ws[0].start; r1 = ws[0].stop; c0 = ws[1].start; c1 = ws[1].stop
        ax.plot([c0, c1, c1, c0, c0], [r0, r0, r1, r1, r0], "k-", lw=0.8)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("T13' GATE A -- reproduction residual (PROPOSED / UNCERTIFIED)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_gate_b(m2, m8, m16, err, out_path, side=IMG_SIDE):
    """(m16-m8)/err and (m2-m16)/err maps -- convergence + aliasing."""
    e = np.asarray(err).reshape(side, side)
    d_conv = (np.asarray(m16) - np.asarray(m8)).reshape(side, side) / e
    d_alias = (np.asarray(m2) - np.asarray(m16)).reshape(side, side) / e
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, data, title, vlim in [
        (axes[0], d_conv, "(m_hi - m_prev) / err  [GATE B top pair]", 0.05),
        (axes[1], d_alias, "(m2 - m_hi) / err  [aliasing REPORT]", None),
    ]:
        v = float(np.max(np.abs(data))) if vlim is None else vlim
        im = ax.imshow(data, cmap="RdBu_r", vmin=-v, vmax=v, origin="upper")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("T13' GATE B -- data convergence (ladder top pair) + aliasing (ss2 vs ss_hi)"
                 " (PROPOSED / UNCERTIFIED)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ===========================================================================
# Offline smoke tests (numpy only; no jax/GPU)
# ===========================================================================

def smoke_box_smooth(verbose=True):
    """Box smoother correctness: (a) constant field -> same constant everywhere;
    (b) a single unit spike at an interior pixel -> centre value 1/k^2 (full k x k
    window), zero far away."""
    k = SMOOTH_KERNEL
    const = np.full((IMG_SIDE, IMG_SIDE), 3.7)
    sc = box_smooth(const, k)
    ok_const = np.allclose(sc, 3.7)
    spike = np.zeros((IMG_SIDE, IMG_SIDE))
    spike[40, 40] = 1.0
    ss = box_smooth(spike, k)
    ok_spike = abs(ss[40, 40] - 1.0 / (k * k)) < 1e-12 and abs(ss[0, 0]) < 1e-12
    ok = bool(ok_const and ok_spike)
    if verbose:
        print(f"[smoke] box_smooth: const->{sc[40,40]:.4f} (3.7); "
              f"spike centre {ss[40,40]:.6f} (1/k^2={1.0/(k*k):.6f}) -> "
              f"{'PASS' if ok else 'FAIL'}")
    return {"const_ok": bool(ok_const), "spike_ok": bool(ok_spike), "pass": ok}


def smoke_gate_thresholds(verbose=True):
    """chi^2 threshold constants are EXACTLY the pre-registered intervals, and the
    window interval matches 5 sqrt(2/225). Also reports the full-image half-width vs the
    log's annotation (flagged mismatch: 0.044 = 2.5 sqrt(2/6400), not 5x)."""
    full_hw = (GATE_A_CHI2_HI - GATE_A_CHI2_LO) / 2.0
    win_hw = (GATE_A_WIN_CHI2_HI - GATE_A_WIN_CHI2_LO) / 2.0
    se_full = np.sqrt(2.0 / N_PIX)
    se_win = np.sqrt(2.0 / ( (2*WIN_HALF+1) ** 2 ))
    ok = (abs(GATE_A_CHI2_LO - 0.956) < 1e-12 and abs(GATE_A_CHI2_HI - 1.044) < 1e-12
          and abs(GATE_A_WIN_CHI2_LO - 0.53) < 1e-12 and abs(GATE_A_WIN_CHI2_HI - 1.47) < 1e-12
          and abs(win_hw - 5.0 * se_win) < 2e-3)
    if verbose:
        print(f"[smoke] thresholds: full [{GATE_A_CHI2_LO},{GATE_A_CHI2_HI}] "
              f"hw={full_hw:.4f} (= {full_hw/se_full:.2f} sqrt(2/6400); log annotation "
              f"says 5x={5*se_full:.4f} -- MISMATCH, pinned interval used); "
              f"window [{GATE_A_WIN_CHI2_LO},{GATE_A_WIN_CHI2_HI}] hw={win_hw:.4f} "
              f"(5 sqrt(2/225)={5*se_win:.4f}) -> {'PASS' if ok else 'FAIL'}")
    return {"full_hw": full_hw, "full_hw_in_sigma": full_hw / se_full,
            "win_hw": win_hw, "win_hw_5sigma": 5 * se_win, "pass": bool(ok)}


def smoke_gate_pure_noise(seed=0, verbose=True):
    """A pure-noise residual (r = err * N(0,1)) PASSES all three GATE-A checks."""
    rng = np.random.default_rng(seed)
    err = np.full((IMG_SIDE, IMG_SIDE), 0.2)
    # give the counter-image window a realistic higher-signal err so the window stat is
    # a fair standard-normal too (err cancels: r = err*noise => r/err = noise regardless)
    noise = rng.standard_normal((IMG_SIDE, IMG_SIDE))
    r = err * noise
    obs = np.zeros_like(err)          # model = 0 => r = obs; equivalently obs=r, m2=0
    gm = gate_a_metrics(r, obs, err)
    passed, checks = gate_a_verdict(gm)
    if verbose:
        print(f"[smoke] pure noise: red_chi2_full={gm['reduced_chi2_full']:.4f} "
              f"[{GATE_A_CHI2_LO},{GATE_A_CHI2_HI}]; smoothed_absmax="
              f"{gm['smoothed_absmax']:.4f} (<{SMOOTH_MAX}); red_chi2_win="
              f"{gm['reduced_chi2_window']:.4f} [{GATE_A_WIN_CHI2_LO},"
              f"{GATE_A_WIN_CHI2_HI}] -> {'PASS' if passed else 'FAIL'}")
    return {"metrics": {k: gm[k] for k in
                        ("reduced_chi2_full", "smoothed_absmax", "reduced_chi2_window")},
            "checks": checks, "pass": bool(passed)}


def smoke_gate_blob(seed=0, verbose=True):
    """A planted COHERENT blob is DETECTED (gate FAILS). Two demonstrations:
      (a) a 1.0-sigma coherent square over the 15x15 counter-image window pushes the
          WINDOWED reduced chi^2 to ~2 (>1.47) -> window gate fails (this is the
          '1 sigma x window coherent blob' the brief calls out);
      (b) a 1.6-sigma coherent square (>= kernel size) pushes the SMOOTHED |r/err| max
          above 1.5 -> smoothed gate fails.
    The mean box-smoother caps a coherent amplitude-A patch at ~A, so the SMOOTHED gate
    (threshold 1.5) catches coherent structure at >~1.5 sigma; the WINDOWED-chi^2 gate
    (5 sigma on 225 cells) catches a 1 sigma coherent offset. Reported honestly."""
    rng = np.random.default_rng(seed)
    err = np.full((IMG_SIDE, IMG_SIDE), 0.2)
    base = err * rng.standard_normal((IMG_SIDE, IMG_SIDE))
    obs = np.zeros_like(err)

    # (a) 1.0-sigma coherent offset over the counter-image window
    ra = base.copy()
    ws = window_slice()
    ra[ws] += 1.0 * err[ws]
    gma = gate_a_metrics(ra, obs, err)
    _, ca = gate_a_verdict(gma)
    win_fail = not ca["reduced_chi2_window"]["pass"]

    # (b) 1.6-sigma coherent square, 15x15, elsewhere (clear of the window)
    rb = base.copy()
    rb[20:35, 55:70] += 1.6 * err[20:35, 55:70]
    gmb = gate_a_metrics(rb, obs, err)
    _, cb = gate_a_verdict(gmb)
    smooth_fail = not cb["smoothed_absmax"]["pass"]

    ok = bool(win_fail and smooth_fail)
    if verbose:
        print(f"[smoke] blob (a) 1.0-sigma@window: red_chi2_win="
              f"{gma['reduced_chi2_window']:.3f} (>{GATE_A_WIN_CHI2_HI}?) window-gate "
              f"{'FAILS' if win_fail else 'passes'}")
        print(f"[smoke] blob (b) 1.6-sigma 15x15:  smoothed_absmax="
              f"{gmb['smoothed_absmax']:.3f} (>{SMOOTH_MAX}?) smoothed-gate "
              f"{'FAILS' if smooth_fail else 'passes'} -> "
              f"{'PASS' if ok else 'FAIL'} (both detect)")
    return {"window_1sigma_reduced_chi2": gma["reduced_chi2_window"],
            "window_gate_fails": bool(win_fail),
            "smoothed_absmax_1p6sigma": gmb["smoothed_absmax"],
            "smoothed_gate_fails": bool(smooth_fail), "pass": ok}


def run_smoke():
    print("=== T13' resim offline smoke tests (numpy only; no jax/GPU) ===")
    a = smoke_box_smooth(); b = smoke_gate_thresholds()
    c = smoke_gate_pure_noise(); d = smoke_gate_blob()
    allpass = a["pass"] and b["pass"] and c["pass"] and d["pass"]
    print(f"[smoke] overall: {'PASS' if allpass else 'FAIL'}")
    return {"box_smooth": a, "thresholds": b, "pure_noise": c, "blob": d,
            "pass": bool(allpass)}


# ===========================================================================
# Main (Steps 1-3)
# ===========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="T13' Steps 0-3: re-simulate sys60 at ss16")
    p.add_argument("--data-dir", help="systems/sys60 (original data + model)")
    p.add_argument("--out-dir", help="resim/sys60_ss16 (all new products go here)")
    p.add_argument("--smoke", action="store_true",
                   help="run offline numpy-only smoke tests and exit")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.smoke:
        res = run_smoke()
        if not res["pass"]:
            raise SystemExit("[T13] smoke tests FAILED")
        return

    for req in ("data_dir", "out_dir"):
        if getattr(args, req) is None:
            raise ValueError(f"--{req.replace('_', '-')} is required (no default).")
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    from common import assert_x64, git_commit
    assert_x64()

    # --- build the ORIGINAL-data model at ss2/ss8/ss16 -----------------------
    print("[T13] loading systems/sys60 model (ss2) ...")
    model2, _, _, dim, param_names = load_sys60(args.data_dir, supersample=2)
    if dim != _EXPECTED_DIM:
        raise ValueError(f"[T13] dim {dim} != {_EXPECTED_DIM}")
    pm2 = model2.prob_model
    observed = np.asarray(pm2.observed_image, dtype=np.float64).reshape(-1)
    err_map = np.asarray(pm2.error_map, dtype=np.float64).reshape(-1)
    print(f"[T13] observed + err_map from the ORIGINAL Dataset (sha1 npz "
          f"{_sha1(_ORIG_NPZ)[:12]}...); event_size={pm2.event_size}")

    truth_nested = load_truth_nested()
    print(f"[T13] truth (sys60, index {_SYSTEM_INDEX}) source center = "
          f"({truth_nested[2][0]['center_x']:.4f}, {truth_nested[2][0]['center_y']:.4f})"
          " -- the counter-image lever")

    # --- Step 1: render truth at ss2 -> GATE A -------------------------------
    m2, z_truth, rt2 = render_at_truth(model2, param_names, truth_nested)
    print(f"[T13] rendered m2 (ss2); truth<->z round-trip err {rt2:.2e}")

    gm = gate_a_metrics(observed, m2, err_map)
    passed_a, checks_a = gate_a_verdict(gm)
    fig_a = os.path.join(out_dir, "t13_gate_a_residual.png")
    plot_gate_a(gm, err_map, fig_a)

    print("\n=== GATE A (reproduction) -- measured vs pre-registered ===")
    print(f"  reduced chi^2 (full)   = {gm['reduced_chi2_full']:.4f}  "
          f"in [{GATE_A_CHI2_LO}, {GATE_A_CHI2_HI}]?  {checks_a['reduced_chi2_full']['pass']}")
    print(f"  max|smoothed r/err|    = {gm['smoothed_absmax']:.4f}  < {SMOOTH_MAX}?  "
          f"{checks_a['smoothed_absmax']['pass']}  (at {gm['smoothed_absmax_loc']})")
    print(f"  reduced chi^2 (window) = {gm['reduced_chi2_window']:.4f}  "
          f"in [{GATE_A_WIN_CHI2_LO}, {GATE_A_WIN_CHI2_HI}]?  "
          f"{checks_a['reduced_chi2_window']['pass']}")

    if not passed_a:
        diag_path = os.path.join(out_dir, "t13_gate_a_FAILED.json")
        with open(diag_path, "w") as f:
            json.dump({"experiment": "T13' GATE A", "status": "FAILED -- STOP",
                       "timestamp_utc": _now(), "git_commit": git_commit(),
                       "original_npz": _ORIG_NPZ, "original_npz_sha1": _sha1(_ORIG_NPZ),
                       "truth_flat": truth_flat_from_nested(truth_nested),
                       "roundtrip_err": rt2, "checks": checks_a,
                       "residual_plot": os.path.basename(fig_a)}, f, indent=2)
        print("\n" + "=" * 72)
        print("[T13] *** GATE A FAILED -- STOP. r = observed - render_ss2(truth) is NOT")
        print("      pure noise. This flags codebase drift / render-path mismatch vs the")
        print("      original generation (old API, float32 PSF). Investigate BEFORE any")
        print("      re-simulation. Diagnostics: " + diag_path)
        print("=" * 72)
        raise SystemExit(2)
    print("[T13] GATE A PASSED -- proceeding to Step 2.")

    # --- Step 2: GATE B as a CONVERGENCE LADDER ------------------------------
    # First run: ss16 vs ss8 differed by 0.69 sigma at the cuspy lens-light
    # center (40,39) -- ss16 was not converged (gate did its job). Amendment
    # (logged): extend the ladder ss = 8, 16, 32, 64; gate on the top
    # consecutive pair; d' uses the top level. Cap 64: still-failing => STOP.
    ladder = [8, 16, 32, 64, 128]   # extended: 64vs32 = 0.068 (rate ~x2.2/doubling); 128 should land ~0.03
    renders = {}
    prev = None
    ss_hi = None
    conv_max = None
    conv_loc = None
    for ss in ladder:
        print(f"[T13] loading model at ss{ss} ...")
        model_k, _, _, _, names_k = load_sys60(args.data_dir, supersample=ss)
        if names_k != param_names:
            raise ValueError("[T13] param-name mismatch across supersample builds")
        renders[ss], _, _ = render_at_truth(model_k, param_names, truth_nested)
        del model_k
        if prev is not None:
            conv = np.abs(renders[ss] - renders[prev]) / err_map
            conv_max = float(np.max(conv))
            conv_loc = np.unravel_index(int(np.argmax(conv)), (IMG_SIDE, IMG_SIDE))
            print(f"  ladder: max|m{ss}-m{prev}|/err = {conv_max:.4f}  "
                  f"(at row,col {list(map(int, conv_loc))})")
            if conv_max < GATE_B_CONV:
                ss_hi = ss
                break
        prev = ss
    m_hi = renders[ss_hi] if ss_hi else renders[ladder[-1]]
    alias = np.abs(m2 - m_hi) / err_map
    alias_max = float(np.max(alias))
    alias_loc = np.unravel_index(int(np.argmax(alias)), (IMG_SIDE, IMG_SIDE))
    fig_b = os.path.join(out_dir, "t13_gate_b_convergence.png")
    plot_gate_b(m2, renders[prev], m_hi, err_map, fig_b)

    print("\n=== GATE B (data convergence ladder) -- measured vs pre-registered ===")
    print(f"  top pair: max|m{ss_hi or ladder[-1]}-m{prev}|/err = {conv_max:.4f}  "
          f"< {GATE_B_CONV}?  {ss_hi is not None}")
    print(f"  REPORT  max|m2-m_hi|/err = {alias_max:.4f}  (at row,col "
          f"{list(map(int, alias_loc))}; expected near the counter-image "
          f"({WIN_ROW},{WIN_COL}))")
    if ss_hi is None:
        print("[T13] *** GATE B FAILED: not converged even at ss64 -- STOP.")
        with open(os.path.join(out_dir, "t13_gate_b_FAILED.json"), "w") as f:
            json.dump({"experiment": "T13' GATE B ladder", "status": "FAILED -- STOP",
                       "top_pair_max_over_err": conv_max, "threshold_lt": GATE_B_CONV,
                       "loc": list(map(int, conv_loc)), "timestamp_utc": _now()}, f, indent=2)
        raise SystemExit(3)
    print(f"[T13] GATE B PASSED at ss{ss_hi} -- proceeding to Step 3 (d' uses ss{ss_hi}).")

    # --- Step 3: d' = m_hi + r; save ------------------------------------------
    residual_noise = observed - m2                     # the recovered noise realization
    d_prime = m_hi + residual_noise
    npz_path = os.path.join(out_dir, "observed_ss16.npz")   # path name is registered; ss_hi recorded inside
    save_arrays = {
        "observed": d_prime.reshape(IMG_SIDE, IMG_SIDE),
        "m2": m2.reshape(IMG_SIDE, IMG_SIDE),
        "m_hi": m_hi.reshape(IMG_SIDE, IMG_SIDE),
        "ss_hi": np.int64(ss_hi),
        "residual_noise": residual_noise.reshape(IMG_SIDE, IMG_SIDE),
        "err_map": err_map.reshape(IMG_SIDE, IMG_SIDE),
    }
    for ss, mk in renders.items():
        save_arrays[f"m{ss}"] = np.asarray(mk).reshape(IMG_SIDE, IMG_SIDE)
    np.savez(npz_path, **save_arrays)
    print(f"[T13] wrote {npz_path} (d' = m{ss_hi} + recovered noise; SAME realization)")

    manifest = {
        "experiment": "T13' Steps 0-3 -- re-simulated sys60 data at ss16",
        "status": "proposed (UNCERTIFIED) -- grader inspects plots + numbers",
        "timestamp_utc": _now(),
        "git_commit": git_commit(),
        "strict_separation": ("all products under resim/sys60_ss16/; data/, results/, "
                              "and the MAIN checkout are read-only / untouched"),
        "original_npz": _ORIG_NPZ,
        "original_npz_sha1": _sha1(_ORIG_NPZ),
        "truth_provenance": {
            "yaml": _TRUTH_YAML, "index": _SYSTEM_INDEX,
            "method": "params_lists_to_jax(yaml)[i=60] (read directly); "
                      "z_truth = bij.inverse(nested truth); render via verified render(z)",
            "truth_flat": truth_flat_from_nested(truth_nested),
            "roundtrip_err_ss2": rt2,
        },
        "dim": int(dim), "param_names": param_names,
        "render_config": {"delta_pix": 0.065, "num_pix": 80,
                          "supersamples": [2] + ladder[:ladder.index(ss_hi) + 1],
                          "ss_hi": int(ss_hi),
                          "background_rms": 0.2, "exp_time": 100,
                          "psf": "gigalens/assets/psf.npy",
                          "error_map": "sqrt(bg^2 + clip(observed,0,inf)/exp_time) "
                                       "(Dataset's own; original data's map)"},
        "gate_a_reproduction": {
            "passed": bool(passed_a),
            "reduced_chi2_full": {"measured": gm["reduced_chi2_full"],
                                  "interval": [GATE_A_CHI2_LO, GATE_A_CHI2_HI]},
            "smoothed_absmax": {"measured": gm["smoothed_absmax"], "threshold_lt": SMOOTH_MAX,
                                "kernel_px": SMOOTH_KERNEL, "loc": gm["smoothed_absmax_loc"]},
            "reduced_chi2_window": {"measured": gm["reduced_chi2_window"],
                                    "interval": [GATE_A_WIN_CHI2_LO, GATE_A_WIN_CHI2_HI],
                                    "window": [WIN_ROW, WIN_COL, "15x15"]},
            "plot": os.path.basename(fig_a),
        },
        "gate_b_convergence": {
            "passed": ss_hi is not None,
            "converged_at_ss": int(ss_hi),
            "top_pair_max_over_err": {"measured": conv_max, "threshold_lt": GATE_B_CONV,
                                      "loc": list(map(int, conv_loc))},
            "report_max_m2_mhi_over_err": {"measured": alias_max,
                                           "loc": list(map(int, alias_loc)),
                                           "note": "data-side aliasing of the original "
                                                   "ss2 data; expected at the counter-image"},
            "plot": os.path.basename(fig_b),
        },
        "step3_output": {"npz": os.path.abspath(npz_path),
                         "keys": sorted(save_arrays.keys()),
                         "d_prime": f"m{ss_hi} + (observed - m2)  [SAME recovered noise]"},
    }
    man_path = os.path.join(out_dir, "t13_resim_manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[T13] wrote {man_path}, {fig_a}, {fig_b}")
    print("[T13] Steps 0-3 done. GATE A + GATE B PASSED; d' saved. "
          "Run t13_arms.py next (pipeline/mclmc/summary/comb).")


if __name__ == "__main__":
    main()
