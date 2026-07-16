"""PIVOTAL MEASUREMENT: does plain supersample=1 rendering UNDER-RESOLVE the src5
source at the two posterior modes of system 2 (1_2_3_4_5_9), band "4-5"?

Instrument: compare_to_reference (gigalens.jax.experimental.adaptive_supersample).
Falsifier construction: an AdaptiveGrid with factor=1.0 EVERYWHERE (verified via
tier histogram) IS plain supersample=1; compared against a uniform ss=8 reference.

CAUTION handled explicitly: carousel_model_s2 uses use_lstsq=True Shapelets for
src4/src5 (164 total linear amplitudes solved per-evaluation via lstsq_simulate).
compare_to_reference calls .simulate() (the plain forward, amplitude-SUMMING path),
not .lstsq_simulate(). This script checks compatibility explicitly and reports
exactly what happens rather than assuming either way.
"""
import traceback

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import carousel_model_s2 as S2
from gigalens.jax.scene_prob_model import ImageData
from gigalens.jax.experimental.adaptive_supersample import (
    AdaptiveGrid, AdaptiveSceneSimulator, compare_to_reference, ALLOWED_FACTORS)

OUT_DIR = "/global/u1/l/linusu/GIGALens-Code/.claude/worktrees/flow-precond/experiments/flow_precond/carousel_gate_pt0_out/diag_sys2"
ARCHIVE = "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/debug_carousel/1_2_3_4_5_9/mclmc/arrays.npz"

print("=" * 78)
print("STEP 1: build PLAIN (non-adaptive) model via S2.AdaptiveImageData patch")
print("=" * 78)
S2.AdaptiveImageData = ImageData
print("patched carousel_model_s2.AdaptiveImageData ->", S2.AdaptiveImageData.__name__)
model_seq, pm = S2.build()
for i, d in enumerate(pm.datasets):
    print(f"  dataset[{i}] type = {type(d).__name__}")
    assert type(d).__name__ == "ImageData", (
        f"dataset {i} is {type(d).__name__}, expected plain ImageData -- patch failed")

model = pm.model
BAND_IDX = 2  # d4_5 (ds("4-5", [src4, src5]))
ds4_5 = pm.datasets[BAND_IDX]
print(f"band index {BAND_IDX} image.shape = {tuple(ds4_5.image.shape)}, "
      f"sim_config = {ds4_5.sim_config}")

print()
print("=" * 78)
print("STEP 2: load MCLMC archive, build z_sharp / z_compact draw pools")
print("=" * 78)
z_all = np.load(ARCHIVE)["samples_z"]  # (8, 10000, 46)
print("samples_z shape:", z_all.shape)
THR = -3.0920
J = 37
print(f"z_param_names[{J}] = {pm.z_param_names[J]!r}")
assert pm.z_param_names[J] == "planes/3/light/1/beta", pm.z_param_names[J]

# z_sharp: literally all draws (any chain) with z[:,J] > THR.
sharp_mask = z_all[:, :, J] > THR
z_sharp_pool = z_all[sharp_mask]
print(f"z_sharp pool: {z_sharp_pool.shape[0]} draws (z[{J}] > {THR})")
per_chain_sharp = sharp_mask.sum(axis=1)
print("  per-chain sharp counts:", per_chain_sharp.tolist())

# z_compact: STATIONARY segment only. Chain 1 excluded entirely (arrives late
# ~draw 8300 and is still drifting in its own post-arrival window -- verified
# below). Chains 3, 5, 6 restricted to their post-arrival segment: arrival =
# first draw index after which z[:,J] < THR for the ENTIRE remainder of that
# chain (computed as last_index_above_THR + 1, which by construction absorbs
# any earlier transit excursions above/below THR before the final crossing).
below_mask = z_all[:, :, J] < THR
arrivals = {}
for c in range(8):
    above_idx = np.nonzero(~below_mask[c])[0]
    last_above = int(above_idx[-1]) if above_idx.size else -1
    arrivals[c] = last_above + 1
print("per-chain arrival index (draws after which z[J]<THR for the remainder):")
for c in range(8):
    print(f"  chain {c}: arrival={arrivals[c]:5d}  "
          f"n_below_total={int(below_mask[c].sum())}")

# Chain 1 stationarity check (for the record): drifts even after "arrival".
v1 = z_all[1, arrivals[1]:, J]
n1 = len(v1)
print(f"chain 1 post-arrival (n={n1}): first-third mean={v1[:n1//3].mean():.4f}  "
      f"last-third mean={v1[-n1//3:].mean():.4f}  -> DRIFTING, chain 1 EXCLUDED "
      "entirely per task directive.")

USE_CHAINS = {3: arrivals[3], 5: arrivals[5], 6: arrivals[6]}
compact_segs, compact_ids = [], []
for c, a in USE_CHAINS.items():
    seg = z_all[c, a:, :]
    compact_segs.append(seg)
    compact_ids.append((c, a, z_all.shape[1] - 1, seg.shape[0]))
z_compact_pool = np.concatenate(compact_segs, axis=0)
print("z_compact stationary segments used (chain, first_draw, last_draw, n):")
for row in compact_ids:
    print("  ", row)
print(f"z_compact pool: {z_compact_pool.shape[0]} draws total")
print(f"z_compact pool mean z[{J}] = {z_compact_pool[:, J].mean():.6f}")
print(f"z_sharp  pool mean z[{J}] = {z_sharp_pool[:, J].mean():.6f}")

print()
print("=" * 78)
print("STEP 3: representative draw per cluster = highest log_prob under PLAIN "
      "model, subsample n=1000")
print("=" * 78)
rng = np.random.default_rng(0)


def pick_representative(pool, name, n_sub=1000, chunk=20):
    n = pool.shape[0]
    idx = rng.choice(n, size=min(n_sub, n), replace=False)
    sub = pool[idx]
    lp_list, chi2_list = [], []
    for start in range(0, sub.shape[0], chunk):
        zj = jnp.asarray(sub[start:start + chunk])
        lp_c, chi2_c = pm.log_prob(zj)
        lp_list.append(np.asarray(lp_c))
        chi2_list.append(np.asarray(chi2_c))
    lp = np.concatenate(lp_list)
    red_chi2 = np.concatenate(chi2_list)
    best_local = int(np.argmax(lp))
    print(f"[{name}] subsample n={sub.shape[0]}  "
          f"lp range=[{lp.min():.3f}, {lp.max():.3f}]  "
          f"best lp={lp[best_local]:.6f}  best red_chi2={red_chi2[best_local]:.6f}  "
          f"pool_idx={idx[best_local]}")
    return sub[best_local], float(lp[best_local]), float(red_chi2[best_local])


z_sharp_rep, lp_sharp_rep, chi2_sharp_rep = pick_representative(z_sharp_pool, "sharp")
z_compact_rep, lp_compact_rep, chi2_compact_rep = pick_representative(z_compact_pool, "compact")
print(f"z_sharp_rep[{J}]   = {z_sharp_rep[J]:.6f}")
print(f"z_compact_rep[{J}] = {z_compact_rep[J]:.6f}")

print()
print("=" * 78)
print("STEP 4: z -> params conversion (model.bijector.forward then model.to_params)")
print("=" * 78)


def z_to_params(z):
    zj = jnp.asarray(z)
    x = pm.bij.forward(zj)
    return model.to_params(x)


params_sharp = z_to_params(z_sharp_rep)
params_compact = z_to_params(z_compact_rep)

# Inspect what the light params dict actually carries for src4 (plane 3, light 0)
# and src5 (plane 3, light 1) -- the CAUTION check: do lstsq components get amp keys?
for tag, p in (("sharp", params_sharp), ("compact", params_compact)):
    l0 = p["planes"][3]["light"][0]
    l1 = p["planes"][3]["light"][1]
    print(f"[{tag}] planes/3/light/0 (src4) keys: {sorted(l0.keys())}")
    print(f"[{tag}] planes/3/light/1 (src5) keys: {sorted(l1.keys())}")

print()
print("=" * 78)
print("STEP 5: build all-factor-1 AdaptiveGrid for band 4-5 (verify uniformity)")
print("=" * 78)
cfg = ds4_5.sim_config
image = np.asarray(ds4_5.image)
error_map = np.asarray(ds4_5.error_map)
mask = np.asarray(ds4_5.mask)
H, W = image.shape
factor_map_all1 = np.ones((H, W), dtype=np.float64)
grid = AdaptiveGrid(factor_map_all1)
tiers = {f: int((grid.factor_map == f).sum()) for f in ALLOWED_FACTORS
         if (grid.factor_map == f).any()}
print(f"band 4-5 shape = {(H, W)}, factor-map tier histogram = {tiers}")
assert tiers == {1.0: H * W}, f"grid is NOT uniform factor-1 everywhere! {tiers}"
print("CONFIRMED: factor map is 1.0 at every one of", H * W, "pixels "
      "(this IS plain supersample=1).")
print("grid repr:", repr(grid))

seen = ds4_5.resolve_sees(model)
print("seen (identity-resolved) light components for band 4-5:",
      [c.profile for c in seen])
print("component depths (n_layers, use_lstsq basis size):",
      [(c.profile.__class__.__name__, getattr(c.profile, "depth", None),
        c.profile.use_lstsq) for c in seen])

adaptive_sim = AdaptiveSceneSimulator(model, cfg, grid, sees=seen)
print("AdaptiveSceneSimulator built OK. n_points =", adaptive_sim.n_points,
      " (== H*W =", H * W, "for factor-1-everywhere)")
assert adaptive_sim.n_points == H * W

print()
print("=" * 78)
print("STEP 6: attempt compare_to_reference -- CHECK lstsq COMPATIBILITY")
print("=" * 78)

results = {}
for tag, params in (("sharp", params_sharp), ("compact", params_compact)):
    for psf_mode in ("bin_first", "subgrid"):
        key = f"{tag}/{psf_mode}"
        try:
            rep = compare_to_reference(adaptive_sim, params, error_map=error_map,
                                        psf_mode=psf_mode)
            print(f"[{key}] SUCCESS. keys = {list(rep.keys())}")
            if "max_abs_delta_over_sigma" in rep:
                print(f"[{key}] max_abs_delta_over_sigma = "
                      f"{rep['max_abs_delta_over_sigma']:.6f}")
            results[key] = rep
        except Exception as e:
            print(f"[{key}] FAILED: {type(e).__name__}: {e}")
            print("---- full traceback ----")
            traceback.print_exc()
            print("-------------------------")
            results[key] = {"error": f"{type(e).__name__}: {e}"}

print()
print("=" * 78)
print("SUMMARY")
print("=" * 78)
for k, v in results.items():
    if "error" in v:
        print(f"{k}: FAILED -- {v['error']}")
    else:
        keys = list(v.keys())
        print(f"{k}: SUCCESS -- keys={keys}")

np.savez(f"{OUT_DIR}/diag_sys2_ss1_falsifier_state.npz",
         z_sharp_rep=z_sharp_rep, z_compact_rep=z_compact_rep,
         lp_sharp_rep=lp_sharp_rep, lp_compact_rep=lp_compact_rep,
         chi2_sharp_rep=chi2_sharp_rep, chi2_compact_rep=chi2_compact_rep,
         z_sharp_pool_size=z_sharp_pool.shape[0],
         z_compact_pool_size=z_compact_pool.shape[0],
         z_compact_mean_37=z_compact_pool[:, J].mean(),
         z_sharp_mean_37=z_sharp_pool[:, J].mean())
print(f"\nsaved representative-draw state to {OUT_DIR}/diag_sys2_ss1_falsifier_state.npz")
print("DONE")
