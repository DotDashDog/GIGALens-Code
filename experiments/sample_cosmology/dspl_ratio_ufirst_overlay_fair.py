import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import dspl_ratio_ufirst as ru

U_HAT, SIG_HAT = 1.3235776, 5.564e-4   # Run A r2 posterior mean/std (independent)

model, *_ , grouped = ru.build_grouped_model_ufirst()
bij = grouped.experimental_default_event_space_bijector()
i1 = model.z_param_names.index("cosmo/Om0")
i2 = model.z_param_names.index("cosmo/w0")
with np.load("/global/homes/l/linusu/GIGALens-Code/results/sample_cosmology/dspl_ratio_ufirst/mclmc/arrays.npz") as d:
    z = np.asarray(d["samples_z"]).reshape(-1, 21)
rng = np.random.default_rng(1)
idx = rng.choice(z.shape[0], size=10000, replace=False)
zc = jnp.asarray(z[idx][:, [i1, i2]])
th = np.asarray(bij.forward(zc))
om_s, w_s = th[:, 0], th[:, 1]

g = np.load("/global/homes/l/linusu/GIGALens-Code/results/sample_cosmology/dspl_cosmology_newapi/def_ratio_grid.npz")
OM, W, R2 = np.asarray(g["Om0_mesh"]), np.asarray(g["w0_mesh"]), np.asarray(g["r2_grid"])
prob = np.exp(-0.5*((R2-U_HAT)/SIG_HAT)**2); prob /= prob.sum()
# mass levels 99.7/95.5/68
s = np.sort(prob.ravel())[::-1]; c = np.cumsum(s)
levels = [s[np.searchsorted(c, m)] for m in (0.997, 0.955, 0.68)]
fig, ax = plt.subplots(figsize=(8,6))
ax.contour(OM, W, prob, levels=sorted(levels),
           colors=["#bbbbbb","#888888","#333333"], linewidths=1.0)
ax.scatter(om_s, w_s, s=1.5, alpha=0.15, color="tab:red", rasterized=True,
           label="Run D samples (10k of 80k)")
ax.set_xlabel("Om0"); ax.set_ylabel("w0"); ax.set_xlim(0, 1.0); ax.set_ylim(-2, -0.35)
ax.set_title("Run D samples vs bands at Run A's independent (mean, sigma):\n"
             f"u_hat={U_HAT}, sigma={SIG_HAT:.2e} (grid display sigma was 2.5x wider)")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
out = "/global/homes/l/linusu/GIGALens-Code/results/sample_cosmology/dspl_ratio_ufirst/ratio_ufirst_overlay_fair.png"
fig.savefig(out, dpi=140)
print("wrote", out)
