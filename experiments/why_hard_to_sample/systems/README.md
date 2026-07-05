# T0/T1 per-system targets (`why_hard_to_sample/systems/`)

Two pre-registered systems (`docs/logs/why-hard-to-sample.md`, T0 seed-variance
and T1 Gaussian-clone). Each subdir has a `system.py` exposing `load_target()`
returning `(model_seq, qz, z_center, dim, param_names)`, consumed via
`common.load_target(data_dir)` (dispatches to `system.py` if present, else the
legacy carousel `build_model.py`+`z_best.npy` path).

Models replicate the human's modeling notebooks EXACTLY
(`experiments/TestNewAPI/TestSersic60.ipynb`, `.../TestSersiclets.ipynb`).
All data/PSF/results paths are absolute into the MAIN checkout
(`/global/homes/l/linusu/GIGALens-Code`, gigalens src at
`/global/homes/l/linusu/gigalens/src`); every missing input raises.

Verified live (CPU, shifter `docker:ghcr.io/nvidia/jax:jax-2026-04-13`,
`JAX_ENABLE_X64=1`): both modules build, `log_prob(z_center)[0]` is finite
float64, `param_names` recover 22/20 unique sorted keys, and **`stable_hash(qz)`
equals the reference MCLMC stage's recorded `upstream_hashes.qz`** (the pipeline's
own content hash) — i.e. qz is byte-identical to the reference run's qz.

---

## sys60 — simulated Sérsic "system 60"

- **dim:** 22 (scene `ProbModel(mode="forward")` → Sérsic **amplitudes sampled**,
  not lstsq-marginalized).
- **Sampled parameters (sorted, sampler column order):**
  ```
  planes/0/light/0/Ie        planes/0/light/0/R_sersic  planes/0/light/0/center_x
  planes/0/light/0/center_y  planes/0/light/0/e1        planes/0/light/0/e2
  planes/0/light/0/n_sersic  planes/0/mass/0/center_x   planes/0/mass/0/center_y
  planes/0/mass/0/e1         planes/0/mass/0/e2         planes/0/mass/0/gamma
  planes/0/mass/0/theta_E    planes/0/mass/1/gamma1     planes/0/mass/1/gamma2
  planes/1/light/0/Ie        planes/1/light/0/R_sersic  planes/1/light/0/center_x
  planes/1/light/0/center_y  planes/1/light/0/e1        planes/1/light/0/e2
  planes/1/light/0/n_sersic
  ```
  (plane 0 = lens: EPL + Shear mass, Sérsic lens light; plane 1 = source Sérsic.)
- **qz:** the reference **SVI-stage Gaussian** (`tfd.MultivariateNormalTriL`),
  rebuilt from
  `results/testsys60/svi/arrays.npz` keys `qz_loc (22,)`, `qz_scale_tril (22,22)`
  (both float64). SVI is **not** re-run. `stable_hash(qz)=e59573bb9e715b71` ==
  `results/testsys60/mclmc/manifest.json` `upstream_hashes.qz` (verified).
  `z_center` = `qz.loc` (SVI mean).
- **Reference run:** `results/testsys60/` — pipeline MAP → SVI → MCLMC,
  8 chains, **20000 burn-in / 20000 results**, seed 0, `debug=True`,
  n_params=22. MCLMC stage `status="ran"`.
- **Truth (for T4 freeze-at-truth):**
  `data/simulated_systems/100SystemsStandardParams.yaml`, **index i=60**
  (notebook cell 3: `params_lists_to_jax(yaml)` then `jax.tree.map(a[60], …)`).
  Observed image: `data/simulated_systems/100SystemsStandard80px.npz`, index 60.
  PSF: `<gigalens_src>/gigalens/assets/psf.npy`. Truth is NOT passed to the
  pipeline in the notebook (pure MAP→SVI→MCLMC).

## vela01_cam12_rep03_a0.500_f814w — simulated Vela source

- **dim:** 20 (scene `ProbModel(mode="lstsq")` → source `EllipticalSersiclets`
  amplitudes **lstsq-marginalized**; only non-linear params sampled). Built via
  registered builder `epl_shear_sersic_elliptical_sersiclets(system, n_max=5)`.
- **Sampled parameters (sorted, sampler column order):**
  ```
  planes/0/light/0/R_sersic  planes/0/light/0/center_x  planes/0/light/0/center_y
  planes/0/light/0/e1        planes/0/light/0/e2        planes/0/light/0/n_sersic
  planes/0/mass/0/center_x   planes/0/mass/0/center_y   planes/0/mass/0/e1
  planes/0/mass/0/e2         planes/0/mass/0/gamma      planes/0/mass/0/theta_E
  planes/0/mass/1/gamma1     planes/0/mass/1/gamma2     planes/1/light/0/beta
  planes/1/light/0/center_x  planes/1/light/0/center_y  planes/1/light/0/e1
  planes/1/light/0/e2        planes/1/light/0/n_sersic
  ```
- **qz:** the reference **truth-bootstrap MAP ball** (`PartialTruthBootstrapQzStage`,
  `free=("source",)`, `diag_scale=1e-6` → diagonal scale 1e-3), a
  `tfd.MultivariateNormalTriL` rebuilt from
  `results/test_new_api_elliptical_sersiclets/vela01_cam12_rep03_a0.500_f814w/bootstrap_map/arrays.npz`
  keys `qz_loc (20,)`, `qz_scale_tril (20,20)` (float64). The bootstrap MAP is
  **not** re-run. `stable_hash(qz)=1d80d19891ea0749` == the MCLMC stage's
  `upstream_hashes.qz` (verified). `z_center` = `qz.loc` (truth-bootstrapped center).
- **Reference run:** `results/test_new_api_elliptical_sersiclets/vela01_cam12_rep03_a0.500_f814w/`
  — pipeline PartialTruthBootstrapQzStage → MCLMC, 8 chains,
  **5000 burn-in / 5000 results**, seed 0, `debug=True`, n_params=20.
- **Truth (for T4 freeze-at-truth):**
  `data/vela_sim_systems/vela01_cam12_rep03_a0.500_f814w/true_params` (pickle,
  3-group nested params), loaded by `from_vela_dir` into `system.truth_x`
  (bootstrap config `truth_hash=be69fea07af651fb`). Source assets:
  `data/vela_sources/vela01_cam12_a0.500_f814w/` (rep-less; PSF + metadata).

---

## MCLMC config: notebook vs. harness `StandardMCLMCConfig`

`MCLMCStage` (pipeline.py:1696) passes `qz` straight to `MCLMC_JIT`; both runs
use `n_hmc=8`, `desired_energy_variance=5e-4`, `regularize_mass_matrix=True`,
`frac_tune1/2/3 = 0.2/0.6/0.2`, `init_L=init_step_size=None` — **identical** to
`StandardMCLMCConfig` (8 chains / 5e-4 / regularize=True; frac_tunes are
`MCLMC_JIT` defaults). `progress_bar`/`debug` differ but affect only output
capture, not sampling. **Only genuine difference: budget** — the *reference*
runs used 20000/20000 (sys60) and 5000/5000 (vela); the harness standard config
is 2000/2000. T0/T1 arms will therefore sample at the harness budget, not the
reference budget (expected — the harness is the single source of the sampler
budget across all arms).

## Discrepancies / notes

- **Vela `system_id` label:** the notebook passes `system_id="vela01_cam12_rep00"`
  to `from_vela_dir` even though the data/results dirs are `rep03`. This is a
  label only (it appears in the bootstrap manifest `system_id`) and is
  replicated verbatim; it does not change data paths (those are `rep03`).
- **Stale stage dirs:** both results trees contain many `*.stale-*` stage dirs
  from prior re-runs. `load_target()` reads only the current (non-stale)
  `svi/`, `bootstrap_map/` dirs referenced by `pipeline.json`; provenance is
  confirmed by the qz content-hash match, not by mtime.
- sys60's lens light and source light are BOTH `SersicEllipse(use_lstsq=False)`,
  so both carry a sampled `Ie` (hence dim 22, two `.../Ie` params).
