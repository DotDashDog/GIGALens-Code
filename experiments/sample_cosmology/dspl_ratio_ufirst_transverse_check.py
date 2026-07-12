import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
from scipy.stats import norm
import dspl_ratio_ufirst as ru
import dspl_ratio_coords as rc

R2_TRUTH = 1.3241652127406303
SIG_GRID = 1.3241652127406303e-3   # grid display sigma (sigma_frac=0.001)
SIG_EFF = 6.7e-4                    # measured effective likelihood width (Run A)

# --- Run D: u per sample, ANALYTIC from z1 (no solves) ---
model, *_, grouped = ru.build_grouped_model_ufirst()
bij = grouped.experimental_default_event_space_bijector()
i_u = model.z_param_names.index("cosmo/Om0")   # z1 column of the grouped entry
with np.load("/global/homes/l/linusu/GIGALens-Code/results/sample_cosmology/dspl_ratio_ufirst/mclmc/arrays.npz") as d:
    z = np.asarray(d["samples_z"])
u_d = np.asarray(bij._u_from_z1(jnp.asarray(z[..., i_u].ravel())))
print(f"Run D u: mean={u_d.mean():.7f} std={u_d.std():.3e} n={u_d.size}")
print(f"  offset from r2_truth: {u_d.mean()-R2_TRUTH:+.3e} "
      f"= {(u_d.mean()-R2_TRUTH)/SIG_EFF:+.2f} sig_eff = {(u_d.mean()-R2_TRUTH)/SIG_GRID:+.2f} sig_grid")
print(f"  std / sig_eff = {u_d.std()/SIG_EFF:.2f}   std / sig_grid = {u_d.std()/SIG_GRID:.2f}")
q = np.quantile((u_d - u_d.mean())/u_d.std(), [0.023, 0.159, 0.5, 0.841, 0.977])
print(f"  standardized quantiles (expect -2,-1,0,1,2 if Gaussian): {np.round(q,2)}")

# --- Run A: r2 per sample, analytic from its z column (UniformBij NormalCDF) ---
import dspl_free_r2 as fr2
m_a = fr2.build_r2_model()
i_r = m_a.z_param_names.index("planes/2/geometry/deflection_ratio")
with np.load("/global/homes/l/linusu/GIGALens-Code/results/sample_cosmology/dspl_free_r2/mclmc/arrays.npz") as d:
    z_a = np.asarray(d["samples_z"])
r2_a = fr2.R2_PRIOR_LOW + (fr2.R2_PRIOR_HIGH - fr2.R2_PRIOR_LOW) * norm.cdf(z_a[..., i_r].ravel())
print(f"Run A r2: mean={r2_a.mean():.7f} std={r2_a.std():.3e} n={r2_a.size}")
print(f"  Run D - Run A: mean diff = {u_d.mean()-r2_a.mean():+.3e} "
      f"({(u_d.mean()-r2_a.mean())/SIG_EFF:+.2f} sig_eff); std ratio = {u_d.std()/r2_a.std():.3f}")

# --- fraction of Run D samples inside the plotted 68% band, per side ---
# grid prob ~ exp(-(r2-r2_truth)^2/2 sig_grid^2); the plotted mass_levels are
# density thresholds; convert each to a |r2-r2_truth| half-width:
g = np.load("/global/homes/l/linusu/GIGALens-Code/results/sample_cosmology/dspl_cosmology_newapi/def_ratio_grid.npz")
prob, r2g, lv = np.asarray(g["prob"]), np.asarray(g["r2_grid"]), np.asarray(g["mass_levels"])
pmax = prob.max()
for name, level in zip(["99.7%","95.5%","68%"], lv):
    half = SIG_GRID*np.sqrt(2*np.log(pmax/level)) if level<pmax else 0.0
    dev = u_d - R2_TRUTH
    inside = np.mean(np.abs(dev) < half)
    outer = np.mean(dev > half); inner = np.mean(dev < -half)
    print(f"  {name} band half-width={half:.3e}: inside={inside:.3f} outer-spill={outer:.4f} inner-spill={inner:.4f}")
