# Cells for gen_improved_sersic_ersatz_carousel.ipynb -- paste, don't import.
# Replaces the commented-out `build(...)` / `real_simulators_for_model(...)` block and
# the hand-written `src_planes` list. Cells 0-3 (jax x64, imports) stay as they are.

# ================================================================== cell 4 (replaces)
# `p` is read and CHECKED, not just loaded: every plane's name must be the one its
# redshift implies. `model` is the fully-fixed truth -- its Sersics are use_lstsq=False,
# so they carry the Ie the simulation needs; the fitting model's do not.
from ersatz_truth import truth_params, build_truth_model, simulate_ersatz

p = truth_params("improved_sersic_carousel.json")
model = build_truth_model(p)

model.validate_params(p)                    # p is exactly this model's §5 layout
assert len(model.z_param_names) == 0        # truth renders one scene; it samples nothing


# ================================================================== cell 6 (tune Ie here)
# Only Ie is left to tune. Edit, re-run this cell and cell 8, look, repeat -- `model`
# reads `p` at simulate time, so it does NOT need rebuilding after an Ie change.
p["planes"]["source3"]["light"]["source3"]["Ie"] = 0.13
# ...


# ================================================================== cell 7/8 (unchanged idea)
from gigalens_research.plotting import plot_scene
from gigalens.jax.scene_simulator import SceneSimulator

cfg_k = SimulatorConfig(delta_pix=0.2, num_pix=300, supersample=1, kernel=None,
                       likelihood_precision="float64")
sims_nopsf = [SceneSimulator(model, cfg_k, sees=pl.light)
              for pl in model.planes if pl.has_light]
figs = plot_scene(model, sims_nopsf, p, with_curves=False)
plt.show()


# ================================================================== cell 10 (replaces)
# One render + one noise realisation per source plane, each with ITS OWN real cutout's
# PSF, background RMS and exposure time -- looked up through the plane's redshift, never
# by list position (plane order and source number agree for only three of nine).
#
# supersample=16 OOMs on a login-node GPU at 300x300; use an allocation for 16, or drop
# to 4 while tuning Ie (JAX_PLATFORMS=cpu also works, ~1 min/plane).
obs = simulate_ersatz(model, p, "real_cutouts", seed=0, supersample=16)

import copy
import ersatz_carousel_prior_improved as FIT   # the model you FIT with (lstsq, no Ie)

# Ascending redshift, so it lines up with `obs`; the assert is the guard, not the comment.
src_planes = [FIT.source1_2, FIT.source3, FIT.source4_5, FIT.source9, FIT.source7,
              FIT.source6, FIT.source12_13, FIT.source8, FIT.source11]
assert [e.plane for e in obs] == [pl.name for pl in src_planes], \
    [(e.plane, pl.name) for e, pl in zip(obs, src_planes) if e.plane != pl.name]

conservative_snr_levels = ((20.0, 8.0), (10.0, 4.0), (7.0, 2.0), (-np.inf, 1.0))
snr_levels = ((15.0, 8.0), (8.0, 4.0), (6.0, 2.0), (2.0, 1.0), (1.0, 0.5), (-np.inf, 0.25))

datasets = []
for i, (e, fit_plane) in enumerate(zip(obs, src_planes)):
    cfg = copy.deepcopy(e.sim_config)
    cfg.supersample = 1                      # the FIT grid, not the render grid
    datasets.append(AdaptiveImageData(
        e.image, cfg, exp_time=e.exp_time, background_rms=e.background_rms,
        sees=fit_plane.light,                # by identity: the fitting model's Components
        snr_levels=conservative_snr_levels if i in (4, 5, 6, 8) else snr_levels))


# ================================================================== cell 11 (replaces)
filter_i = list(range(len(datasets)))
filtered_datasets = [datasets[i] for i in filter_i]

model_filtered = LensModel(
    [FIT.lens_plane, *[src_planes[i] for i in filter_i]],
    cosmo=FIT.cosmo, unconstrain="gaussian",
)
for d in filtered_datasets:
    plot_factor_map(d.adaptive_grid)

prob_model = ProbModel(model_filtered, filtered_datasets, mode="lstsq")


# ================================================================== cell 13/19 (unchanged)
# `remove_key(p, 'Ie')` is still needed wherever p meets the FITTING model: its Sersics
# are use_lstsq=True and have no Ie parameter at all.
#   rep = diagnose_undersampling(prob_model, remove_key(p, 'Ie'), reference_supersample=16, ...)
#   z_truth = prob_model.unconstrained(remove_key(p, 'Ie'))
