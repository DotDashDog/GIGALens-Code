#!/usr/bin/env python
"""GATE L: Laplace jump-proposal feasibility diagnostics (pre-registered 2026-07-09).

Design checkpoint: docs/logs/carousel-mclmc-sampling.md "GATE L". Three measurements,
no sampling runs:

  M1  Polish zM/zP (basin medians) to local stationary points z*M/z*P (Adam in
      SVI-whitened coords, lr ladder 1e-3/1e-4/1e-5 x 1000 steps, best-lp iterate),
      then Hessian H = -grad^2 lp at z*, eigen-spectrum, PSD + conditioning.
  M2  (a) Laplace N(z*, H^-1) vs empirical per-mode moments of the MH-exact MAMS64
      draws: KL(N_emp || N_Laplace) in nats + per-direction scale ratios
      sqrt(gen-eigvals(S_emp, S_L)); pocket-shape robustness check vs the benchmark
      8x4000 draws (MH-exact in law; trap-biased occupancy never used for weights).
      (b) MC jump acceptance for three pre-registered proposal configs:
        P1 plain Gaussian Laplace mixture, P2 armored (multivariate-t df=5,
        scale-tril x1.5), P3 pure translation z' = z +- (z*P - z*M).
      Mixture weights are pipeline-realistic Laplace weights
      logw_k = lp(z*_k) + 0.5 logdet(2 pi Sigma_k) (oracle 9.57% variant = diagnostic
      only). alpha = min(1, exp(lp(z') - lp(z) + logq(z) - logq(z'))), 1024 pairs per
      (state-basin x proposal-component) cell, states subsampled seed 0 from MAMS64.
      Thresholds (derived in checkpoint): cross-mode mean alpha >= 20% W1a-class;
      2-20% clear improvement over plain MAMS; < 0.5% unworkable.
  M3  1024-start multistart MAP through the production machinery (pipeline-realistic
      config: adabelief 1e-2 b1 .95 b2 .99 nesterov, 4000 steps; seed 0). Outcome:
      fraction of finals with z[6] > -22.35 AND lp >= lp(z*P) - 33.

Run (4-GPU node, shifter jax container, float64):
  srun --overlap --jobid=$JOB shifter --module=gpu,nccl-plugin \
    --image=docker:ghcr.io/nvidia/jax:jax-2026-04-13 bash -c '
      export PYTHONPATH=/global/homes/l/linusu/sidecar_jax_upgrade:$HOME/.conda/envs/gigalens_multinode_env/lib/python3.12/site-packages:<repo>/src:<repo>/experiments/flow_precond
      /usr/bin/python3 carousel_gate_l.py'
"""
import json
import math
import os
import sys
import time

import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
from jax import numpy as jnp
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import carousel_model

OUT = os.path.join(HERE, "carousel_gate_l_out")
os.makedirs(OUT, exist_ok=True)
DPIE = os.path.join(HERE, "..", "sim_carousel", "messy_tests", "dpie")
BASIN = "/pscratch/sd/l/linusu/carousel_diag/basin_slice/basin_slice.npz"
BENCH = os.path.join(HERE, "carousel_benchmark_out", "benchmark_arrays.npz")
POCKET_COL, POCKET_THR = 6, -22.35
SEED = 0
DIM = 33
N_PAIRS = 1024          # MC pairs per (state-basin x proposal-component) cell
T_DF, T_SCALE = 5.0, 1.5  # P2 armor
M3_N, M3_STEPS = 1024, 4000  # pipeline-realistic MAP config (manifest: 128x4000)
IN_BASIN_NATS = 33.0    # typical-set half-width 2*(d/2) below the polished peak
POLISH_STEPS = 1000     # per lr rung
SMOKE = os.environ.get("GATE_L_SMOKE", "0") == "1"
if SMOKE:  # plumbing check only; numbers are NOT the pre-registered measurement
    N_PAIRS, M3_N, M3_STEPS, POLISH_STEPS = 64, 8, 20, 10

t0_all = time.time()

# --------------------------------------------------------------------- model + lp
model_seq, prob_model = carousel_model.build()
assert prob_model.z_param_names[POCKET_COL] == "planes/0/mass/1/center_x"
lp_fn = lambda z: prob_model.log_prob(z)[0]
lp1 = jax.jit(lp_fn)
_lp_vec = jax.jit(jax.vmap(lp_fn))


def lp_batch(z, chunk=128):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty(len(z))
    for i in range(0, len(z), chunk):
        out[i:i + chunk] = np.asarray(_lp_vec(jnp.asarray(z[i:i + chunk])))
    return out


basin = np.load(BASIN)
zM0, zP0 = np.asarray(basin["zM"]), np.asarray(basin["zP"])
mams = np.load(os.path.join(DPIE, "mams", "arrays.npz"))["samples_z"]
draws = mams.reshape(-1, DIM).astype(np.float64)
is_pocket = draws[:, POCKET_COL] > POCKET_THR
svi = np.load(os.path.join(DPIE, "svi", "arrays.npz"))
L_svi = np.asarray(svi["qz_scale_tril"], dtype=np.float64)

card = dict(
    script=os.path.abspath(__file__), jax=jax.__version__,
    devices=[str(d) for d in jax.devices()], n_devices=len(jax.devices()),
    x64=bool(jax.config.jax_enable_x64), dim=DIM, seed=SEED, smoke=SMOKE,
    basin_slice=BASIN, mams64=os.path.join(DPIE, "mams", "arrays.npz"),
    n_draws=int(len(draws)), pocket_occupancy=float(is_pocket.mean()),
    pocket_indicator=f"z[{POCKET_COL}] > {POCKET_THR}",
    m2=dict(n_pairs=N_PAIRS, t_df=T_DF, t_scale=T_SCALE),
    m3=dict(n_samples=M3_N, num_steps=M3_STEPS,
            optimizer="adabelief_1e-2_b1_0.95_b2_0.99_nesterov",
            in_basin_nats=IN_BASIN_NATS),
)
print("MODEL CARD:", json.dumps(card, indent=1), flush=True)
summary = dict(model_card=card)

# --------------------------------------------------------------------------- M1
print("\n=== M1: polish + Hessian spectra ===", flush=True)
Lm = jnp.asarray(L_svi)
z0s = jnp.asarray(np.stack([zM0, zP0]))


@jax.jit
def _polish_val_grad(dws):
    def f(dw, z0):
        return -lp_fn(z0 + Lm @ dw)
    vals, grads = jax.vmap(jax.value_and_grad(f))(dws, z0s)
    return vals, grads


def polish():
    dws = jnp.zeros((2, DIM))
    vals, _ = _polish_val_grad(dws)
    best_lp = -np.asarray(vals)
    best_dw = np.zeros((2, DIM))
    lp_start = best_lp.copy()
    steps_done = 0
    for lr in (1e-3, 1e-4, 1e-5):
        opt = optax.adam(lr)
        state = opt.init(dws)

        @jax.jit
        def step(dws, state):
            vals, grads = _polish_val_grad(dws)
            upd, state = opt.update(grads, state)
            return optax.apply_updates(dws, upd), state, vals

        for _ in range(POLISH_STEPS):
            prev = np.asarray(dws)  # vals from step() belong to THIS iterate
            dws, state, vals = step(dws, state)
            lp_now = -np.asarray(vals)
            for k in range(2):
                if lp_now[k] > best_lp[k]:
                    best_lp[k] = lp_now[k]
                    best_dw[k] = prev[k]
            steps_done += 1
    # assess the final iterate too (never scored inside the loop)
    vals_fin, _ = _polish_val_grad(dws)
    lp_fin = -np.asarray(vals_fin)
    for k in range(2):
        if lp_fin[k] > best_lp[k]:
            best_lp[k] = lp_fin[k]
            best_dw[k] = np.asarray(dws[k])
    zstars = np.asarray(z0s) + best_dw @ L_svi.T
    return zstars, lp_start, best_lp, steps_done


t0 = time.time()
zstars, lp_start, lp_star, n_polish = polish()
zstarM, zstarP = zstars
grads = np.asarray(jax.vmap(jax.grad(lp_fn))(jnp.asarray(zstars)))
m1 = dict(
    lp_median=dict(M=float(lp_start[0]), P=float(lp_start[1])),
    lp_star=dict(M=float(lp_star[0]), P=float(lp_star[1])),
    polish_gain=dict(M=float(lp_star[0] - lp_start[0]),
                     P=float(lp_star[1] - lp_start[1])),
    polish_steps=n_polish,
    pocket_membership=dict(M=bool(zstarM[POCKET_COL] > POCKET_THR),
                           P=bool(zstarP[POCKET_COL] > POCKET_THR)),
    wall_s=time.time() - t0,
)

# empirical per-mode moments (needed for the pre-registered Sigma_L floor rule)
emp = {}
for nm, sel in (("M", ~is_pocket), ("P", is_pocket)):
    dd = draws[sel]
    emp[nm] = (dd.mean(0), np.cov(dd.T), sel)
emp_lam_max = {nm: float(np.linalg.eigvalsh(emp[nm][1]).max()) for nm in ("M", "P")}

t0 = time.time()
hess = jax.jit(jax.jacfwd(jax.grad(lp_fn)))
Hs, Sig_L, eigs = {}, {}, {}
for nm, zst, g in (("M", zstarM, grads[0]), ("P", zstarP, grads[1])):
    H = -np.asarray(hess(jnp.asarray(zst)))
    H = 0.5 * (H + H.T)
    lam, V = np.linalg.eigh(H)
    Hs[nm], eigs[nm] = H, lam
    psd = bool(lam.min() > -1e-8 * lam.max())
    gray_zone = bool(-1e-6 * lam.max() <= lam.min() <= -1e-8 * lam.max())
    # pre-registered floor (grader item i): no Sigma_L axis sd may exceed 2x the
    # longest empirical posterior axis of this mode
    lam_floor = 1.0 / (4.0 * emp_lam_max[nm])
    n_floored = int((lam < lam_floor).sum())
    lam_c = np.maximum(lam, lam_floor)
    Sig = (V / lam_c) @ V.T
    Sig_L[nm] = Sig
    nat_g = float(np.sqrt(g @ Sig @ g))
    m1[f"hessian_{nm}"] = dict(
        eig_min=float(lam.min()), eig_max=float(lam.max()),
        psd_pass=psd, gray_zone=gray_zone,
        cond=float(lam.max() / max(lam.min(), 1e-300)),
        n_nonpos=int((lam <= 0).sum()),
        lam_floor=lam_floor, n_floored=n_floored,
        natural_grad=nat_g, grad_norm=float(np.linalg.norm(g)),
    )
    print(f"[M1] {nm}: lp* {lp_star[0 if nm=='M' else 1]:.3f} "
          f"(gain {m1['polish_gain'][nm]:+.2f}), eig[min,max]=[{lam.min():.3e},"
          f" {lam.max():.3e}], PSD={psd}, nat|grad|={nat_g:.3f}", flush=True)
m1["hessian_wall_s"] = time.time() - t0
summary["M1"] = m1

# --------------------------------------------------------------------------- M2a
print("\n=== M2a: Laplace vs empirical per-mode moments ===", flush=True)


def gauss_kl(mu_e, S_e, mu_l, S_l):
    d = len(mu_e)
    A = np.linalg.cholesky(S_l)
    X = np.linalg.solve(A, S_e)
    tr = float(np.trace(np.linalg.solve(A.T, X)))
    dm = np.linalg.solve(A, mu_l - mu_e)
    logdet_l = 2 * float(np.log(np.diag(A)).sum())
    logdet_e = float(np.linalg.slogdet(S_e)[1])
    return 0.5 * (tr + float(dm @ dm) - d + logdet_l - logdet_e)


def scale_ratios(S_e, S_l):
    A = np.linalg.cholesky(S_l)
    M = np.linalg.solve(A, np.linalg.solve(A, S_e).T)
    return np.sqrt(np.clip(np.linalg.eigvalsh(M), 0, None))


m2a = {}
for nm in ("M", "P"):
    mu_e, S_e, sel = emp[nm]
    r = scale_ratios(S_e, Sig_L[nm])
    m2a[nm] = dict(
        n_draws=int(sel.sum()),
        kl_emp_vs_laplace_nats=gauss_kl(mu_e, S_e, zstars[0 if nm == "M" else 1],
                                        Sig_L[nm]),
        scale_ratio_min=float(r.min()), scale_ratio_max=float(r.max()),
        scale_ratios=r.tolist(),
        mean_offset_natural=float(np.sqrt(
            (mu_e - zstars[0 if nm == "M" else 1])
            @ np.linalg.solve(Sig_L[nm], mu_e - zstars[0 if nm == "M" else 1]))),
    )
    print(f"[M2a] {nm}: KL(emp||Laplace) = {m2a[nm]['kl_emp_vs_laplace_nats']:.2f} "
          f"nats, scale ratios [{r.min():.3f}, {r.max():.3f}], "
          f"mean offset {m2a[nm]['mean_offset_natural']:.2f} sigma", flush=True)

# pocket-shape robustness: benchmark 8x4000 decoded-z draws (exact in law).
# Pre-registered (grader item iii): exclude stuck chains (per-chain occ > 0.99).
bench_chains = np.load(BENCH)["z"].astype(np.float64)
b_occ = (bench_chains[:, :, POCKET_COL] > POCKET_THR).mean(1)
keep = b_occ <= 0.99
print(f"[M2a] benchmark chains kept {int(keep.sum())}/{len(keep)} "
      f"(per-chain occ {np.round(b_occ, 3).tolist()})", flush=True)
bench_z = bench_chains[keep].reshape(-1, DIM)
bsel = bench_z[:, POCKET_COL] > POCKET_THR
bd = bench_z[bsel]
r_b = scale_ratios(np.cov(bd.T), Sig_L["P"])
m2a["P_robustness_benchmark"] = dict(
    n_draws=int(bsel.sum()),
    kl_emp_vs_laplace_nats=gauss_kl(bd.mean(0), np.cov(bd.T), zstarP, Sig_L["P"]),
    scale_ratio_min=float(r_b.min()), scale_ratio_max=float(r_b.max()),
)
print(f"[M2a] P robustness (benchmark, n={int(bsel.sum())}): "
      f"KL {m2a['P_robustness_benchmark']['kl_emp_vs_laplace_nats']:.2f} nats, "
      f"ratios [{r_b.min():.3f}, {r_b.max():.3f}]", flush=True)
summary["M2a"] = m2a

# --------------------------------------------------------------------------- M2b
print("\n=== M2b: MC jump acceptance ===", flush=True)
rng = np.random.default_rng(SEED)
chol = {nm: np.linalg.cholesky(Sig_L[nm]) for nm in ("M", "P")}
logdet_sig = {nm: 2 * float(np.log(np.diag(chol[nm])).sum()) for nm in ("M", "P")}

# pipeline-realistic Laplace mixture weights (no posterior info)
logw_un = np.array([lp_star[0] + 0.5 * (DIM * math.log(2 * math.pi) + logdet_sig["M"]),
                    lp_star[1] + 0.5 * (DIM * math.log(2 * math.pi) + logdet_sig["P"])])
logw = logw_un - (logw_un.max() + math.log(np.exp(logw_un - logw_un.max()).sum()))
w = np.exp(logw)
summary["M2b_weights"] = dict(
    laplace_pocket_weight=float(w[1]), oracle_pocket_weight=float(is_pocket.mean()),
    prior_record_proxy=0.054)
print(f"[M2b] Laplace mixture pocket weight = {w[1]:.4f} "
      f"(truth 0.0957, prior-record proxy 0.054)", flush=True)


def log_gauss(z, mu, A, logdet):
    y = np.linalg.solve(A, (z - mu).T).T
    return -0.5 * (DIM * math.log(2 * math.pi) + logdet + (y * y).sum(-1))


def log_t(z, mu, A_t, logdet_t, df):
    y = np.linalg.solve(A_t, (z - mu).T).T
    q = (y * y).sum(-1)
    return (math.lgamma((df + DIM) / 2) - math.lgamma(df / 2)
            - 0.5 * DIM * math.log(df * math.pi) - 0.5 * logdet_t
            - 0.5 * (df + DIM) * np.log1p(q / df))


def logsumexp2(a, b):
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


mus = {"M": zstarM, "P": zstarP}
A_t = {nm: T_SCALE * chol[nm] for nm in ("M", "P")}
logdet_t = {nm: logdet_sig[nm] + 2 * DIM * math.log(T_SCALE) for nm in ("M", "P")}


occ = float(is_pocket.mean())
logw_oracle = np.log(np.array([1.0 - occ, occ]))  # diagnostic-only weight variant


def logq(cfg, z, lw=None):
    lw = logw if lw is None else lw
    if cfg == "P1":
        comp = {nm: log_gauss(z, mus[nm], chol[nm], logdet_sig[nm])
                for nm in ("M", "P")}
    else:
        comp = {nm: log_t(z, mus[nm], A_t[nm], logdet_t[nm], T_DF)
                for nm in ("M", "P")}
    return logsumexp2(lw[0] + comp["M"], lw[1] + comp["P"])


def draw(cfg, nm, n):
    eps = rng.standard_normal((n, DIM))
    if cfg == "P1":
        return mus[nm] + eps @ chol[nm].T
    g = rng.chisquare(T_DF, size=n)
    return mus[nm] + (eps @ A_t[nm].T) * np.sqrt(T_DF / g)[:, None]


idxM = rng.choice(np.flatnonzero(~is_pocket), N_PAIRS, replace=False)
idxP = rng.choice(np.flatnonzero(is_pocket), N_PAIRS, replace=False)
states = {"M": draws[idxM], "P": draws[idxP]}
state_chain = {"M": idxM // 1000, "P": idxP // 1000}  # MAMS64 is (64,1000,33) flat
t0 = time.time()
lp_states = {nm: lp_batch(states[nm]) for nm in ("M", "P")}


def clustered_se(alpha, chains):
    """Per-source-chain mean alpha; se = sd(chain means)/sqrt(n_chains) (item ii)."""
    uc = np.unique(chains)
    cm = np.array([alpha[chains == c].mean() for c in uc])
    se = float(cm.std(ddof=1) / math.sqrt(len(uc))) if len(uc) > 1 else float("inf")
    return se, int(len(uc))

m2b = {}
for cfg in ("P1", "P2"):
    lq_states = {nm: logq(cfg, states[nm]) for nm in ("M", "P")}
    lq_states_o = {nm: logq(cfg, states[nm], logw_oracle) for nm in ("M", "P")}
    for comp in ("M", "P"):
        zp = draw(cfg, comp, N_PAIRS)
        lp_zp = lp_batch(zp)
        lq_zp = logq(cfg, zp)
        lq_zp_o = logq(cfg, zp, logw_oracle)
        landed_pocket = zp[:, POCKET_COL] > POCKET_THR
        for frm in ("M", "P"):
            la = (lp_zp - lp_states[frm]) + (lq_states[frm] - lq_zp)
            alpha = np.exp(np.minimum(0.0, la))
            alpha[~np.isfinite(la)] = 0.0
            key = f"{cfg}_{frm}to{comp}"
            se_cl, n_ch = clustered_se(alpha, state_chain[frm])
            la_o = (lp_zp - lp_states[frm]) + (lq_states_o[frm] - lq_zp_o)
            alpha_o = np.exp(np.minimum(0.0, la_o))
            alpha_o[~np.isfinite(la_o)] = 0.0
            m2b[key] = dict(mean_alpha=float(alpha.mean()),
                            se_binomial=float(alpha.std() / math.sqrt(N_PAIRS)),
                            se_clustered=se_cl, n_chains=n_ch,
                            n_accept_ge_half=int((alpha > 0.5).sum()),
                            prop_landed_pocket_frac=float(landed_pocket.mean()),
                            oracle_weight_mean_alpha_diag=float(alpha_o.mean()))
            print(f"[M2b] {key}: mean alpha = {alpha.mean():.4f} "
                  f"(se_bin {alpha.std()/math.sqrt(N_PAIRS):.4f}, "
                  f"se_clust {se_cl:.4f}, {n_ch} chains; "
                  f"oracleW diag {alpha_o.mean():.4f})", flush=True)

# P3: pure translation (symmetric +-Delta pair; q-terms cancel)
delta = zstarP - zstarM
for frm, sgn, tag in (("M", +1, "MtoP"), ("P", -1, "PtoM"),
                      ("M", -1, "M_anti"), ("P", +1, "P_anti")):
    zp = states[frm] + sgn * delta
    la = lp_batch(zp) - lp_states[frm]
    alpha = np.exp(np.minimum(0.0, la))
    alpha[~np.isfinite(la)] = 0.0
    se_cl, n_ch = clustered_se(alpha, state_chain[frm])
    m2b[f"P3_{tag}"] = dict(mean_alpha=float(alpha.mean()),
                            se_binomial=float(alpha.std() / math.sqrt(N_PAIRS)),
                            se_clustered=se_cl, n_chains=n_ch)
    print(f"[M2b] P3_{tag}: mean alpha = {alpha.mean():.5f} "
          f"(se_clust {se_cl:.5f})", flush=True)
m2b["wall_s"] = time.time() - t0

# Pre-registered pass criterion (grader items ii+iv+x): P1/P2 only, realistic
# weights, cross-mode cells; the 0.5%/2% alpha lines were derived at w_P = 0.1, so
# they rescale by 0.1/w_P when the measured Laplace weight leaves [0.05, 0.2]
# (equivalently: decisions route on the per-step cross-jump rate w_P*alpha).
thr_scale = 1.0 if 0.05 <= w[1] <= 0.2 else 0.1 / w[1]
verdicts = {}
for key in ("P1_MtoP", "P1_PtoM", "P2_MtoP", "P2_PtoM"):
    c = m2b[key]
    a, se = c["mean_alpha"], c["se_clustered"]
    if a >= 0.02 * thr_scale and a - 2 * se > 0.005 * thr_scale:
        verdicts[key] = "PASS"
    elif a >= 0.02 * thr_scale:
        verdicts[key] = "INSUFFICIENT-STATE-DIVERSITY"
    else:
        verdicts[key] = "FAIL"
m2b["threshold_scale"] = float(thr_scale)
m2b["cell_verdicts"] = verdicts
print(f"[M2b] cell verdicts (thr_scale {thr_scale:.3f}):",
      json.dumps(verdicts), flush=True)
summary["M2b"] = m2b
with open(os.path.join(OUT, "gate_l_summary.json"), "w") as f:
    json.dump(summary, f, indent=1)  # early dump; final rewrite at end

# --------------------------------------------------------------------------- M3
print("\n=== M3: 1024-start multistart MAP ===", flush=True)
try:
    m3_opt = optax.adabelief(1e-2, b1=0.95, b2=0.99, nesterov=True)
except TypeError:
    m3_opt = optax.adabelief(1e-2, b1=0.95, b2=0.99)
t0 = time.time()
map_samples, map_lps, map_chisqs = model_seq.MAP(
    m3_opt, n_samples=M3_N, num_steps=M3_STEPS, seed=SEED,
    output_type="all", pbar_interval=0)
map_samples = np.asarray(map_samples)   # (n, steps, dim)
map_lps = np.asarray(map_lps)           # (n, steps)
m3_wall = time.time() - t0

final_z = map_samples[:, -1]
# Production MAP "all" output pairs post-update params with pre-update lp (one-step
# offset, grader item viii): recompute lp at final_z for classification; the library
# lp is kept as a diagnostic.
final_lp_lib = map_lps[:, -1]
final_lp = lp_batch(final_z)
best_step = np.nanargmax(np.where(np.isnan(map_lps), -np.inf, map_lps), axis=1)
best_lp = map_lps[np.arange(M3_N), best_step]
best_z = map_samples[np.arange(M3_N), best_step]


def classify(z, lp):
    pocket = (z[:, POCKET_COL] > POCKET_THR) & (lp >= lp_star[1] - IN_BASIN_NATS)
    main = (z[:, POCKET_COL] <= POCKET_THR) & (lp >= lp_star[0] - IN_BASIN_NATS)
    return pocket, main


pk_f, mn_f = classify(final_z, final_lp)
pk_b, mn_b = classify(best_z, best_lp)
summary["M3"] = dict(
    wall_s=m3_wall, n=M3_N, steps=M3_STEPS,
    n_nan_final=int(np.isnan(final_lp).sum()),
    lp_recompute_vs_lib_max_abs=float(np.nanmax(np.abs(final_lp - final_lp_lib))),
    final=dict(pocket=int(pk_f.sum()), main=int(mn_f.sum()),
               straggler=int(M3_N - pk_f.sum() - mn_f.sum())),
    best_step_diagnostic_only=dict(pocket=int(pk_b.sum()), main=int(mn_b.sum()),
                                   straggler=int(M3_N - pk_b.sum() - mn_b.sum())),
    best_lp_overall=float(np.nanmax(best_lp)),
    lp_star_ref=dict(M=float(lp_star[0]), P=float(lp_star[1])),
    pipeline_map_manifest_best_lp=-291336.70477837865,
)
print(f"[M3] final: pocket {pk_f.sum()}, main {mn_f.sum()}, "
      f"straggler {M3_N - pk_f.sum() - mn_f.sum()}; "
      f"best-step: pocket {pk_b.sum()}, main {mn_b.sum()}; "
      f"best lp overall {np.nanmax(best_lp):.3f}; wall {m3_wall:.0f}s", flush=True)
with open(os.path.join(OUT, "gate_l_summary.json"), "w") as f:
    json.dump(summary, f, indent=1)  # pre-plot dump; plots can fail without data loss

# ----------------------------------------------------------------------- persist
np.savez(
    os.path.join(OUT, "gate_l_arrays.npz"),
    zstarM=zstarM, zstarP=zstarP, H_M=Hs["M"], H_P=Hs["P"],
    eig_M=eigs["M"], eig_P=eigs["P"],
    m3_final_z=final_z, m3_final_lp=final_lp, m3_final_lp_lib=final_lp_lib,
    m3_best_lp=best_lp, m3_best_z6=best_z[:, POCKET_COL],
    idx_states_M=idxM, idx_states_P=idxP,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
for i, nm in enumerate(("M", "P")):
    lam = eigs[nm]
    ax[i].semilogy(np.sort(np.abs(lam))[::-1], ".-")
    ax[i].set_title(f"|eig| spectrum, mode {nm} "
                    f"(n_nonpos={int((lam <= 0).sum())})")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "gate_l_m1_spectra.png"), dpi=120)

fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
for i, nm in enumerate(("M", "P")):
    r = np.array(m2a[nm]["scale_ratios"])
    ax[i].semilogy(np.sort(r), ".-")
    ax[i].axhline(1.0, color="k", lw=0.5)
    ax[i].set_title(f"emp/Laplace scale ratios, mode {nm}")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "gate_l_m2_ratios.png"), dpi=120)

fig, ax = plt.subplots(figsize=(9, 3.5))
keys = [k for k in m2b if isinstance(m2b[k], dict) and "mean_alpha" in m2b[k]]
ax.bar(range(len(keys)), [max(m2b[k]["mean_alpha"], 1e-7) for k in keys])
ax.set_yscale("log")
ax.set_xticks(range(len(keys)))
ax.set_xticklabels(keys, rotation=60, ha="right", fontsize=7)
for thr, lab in ((0.20, "W1a-class"), (0.02, "improvement"), (0.005, "unworkable")):
    ax.axhline(thr * thr_scale, ls="--", lw=0.7)
    ax.text(len(keys) - 0.5, thr * thr_scale, lab, fontsize=7,
            va="bottom", ha="right")
ax.set_ylabel("mean alpha")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "gate_l_m2_accept.png"), dpi=120)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(final_z[:, POCKET_COL], final_lp, ".", ms=3, alpha=0.5)
ax.axvline(POCKET_THR, color="r", lw=0.7)
for v, c in ((lp_star[0], "C1"), (lp_star[1], "C2")):
    ax.axhline(v - IN_BASIN_NATS, color=c, lw=0.7, ls="--")
ax.set_xlabel(f"final z[{POCKET_COL}]")
ax.set_ylabel("final lp")
ax.set_ylim(lp_star[1] - 400, lp_star[1] + 20)
ax.set_title("M3 multistart finals")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "gate_l_m3.png"), dpi=120)

summary["total_wall_s"] = time.time() - t0_all
with open(os.path.join(OUT, "gate_l_summary.json"), "w") as f:
    json.dump(summary, f, indent=1)
print(f"\nDONE in {summary['total_wall_s']:.0f}s -> {OUT}", flush=True)
