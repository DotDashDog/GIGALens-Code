"""Single-system SersicShapelets run at campaign settings, with plots.

Runs ONE Vela system through the registered builder + pipeline
(SersicShapeletsBootstrapQzStage -> MCLMC) at the campaign's sampler settings,
then writes a full_report (image / convergence / source / corner / z-scores /
source-comparison) and the MCLMC tuning diagnostics.
"""
import json
import os

import jax
jax.config.update("jax_enable_x64", True)
import matplotlib
matplotlib.use("Agg")
import numpy as np

import gigalens_research.simtests.experiments  # registers builders/pipelines
from gigalens_research.simtests.system import from_vela_dir
from gigalens_research.simtests.registry import (
    get_inference_builder, get_pipeline_builder, get_metric,
)
from gigalens_research.inference_utils.pipeline import InferenceContext, Pipeline
from gigalens_research.inference_utils.truth_diagnostics import truth_source_from_light_model
from gigalens_research.simulations import load_vela_source
from gigalens_research.plotting import PosteriorReport, PipelineReport

HOME = os.path.expanduser("~")
DATA = os.path.join(HOME, "GIGALens-Code", "data")
OUT = os.path.join(HOME, "GIGALens-Code", "results", "sersicshapelets_single")
PLOTS = os.path.join(OUT, "plots")
os.makedirs(PLOTS, exist_ok=True)

SYS_ID = "vela01_cam12_rep00"
N_MAX = 5

src_dir = os.path.join(DATA, "vela_sources", "vela01_cam12_a0.500_f814w")
sys_dir = os.path.join(DATA, "vela_sim_systems", f"{SYS_ID}_a0.500_f814w")
delta_pix = float(json.load(open(os.path.join(src_dir, "metadata.json")))
                  ["instrument_pixel_scale_arcsec"])

system = from_vela_dir(
    system_dir=sys_dir, source_dir=src_dir, system_id=SYS_ID,
    delta_pix=delta_pix, num_pix=200, supersample=1,
    background_rms=0.002, exp_time=2000.0,
)
print(f"[run] system {SYS_ID}: image {system.observed_image.shape}, delta_pix={delta_pix}")

# Campaign sampler settings (matches campaign.yaml), with MCLMC debug for plots.
kw = dict(
    n_max=N_MAX,
    bootstrap_map_steps=200, bootstrap_map_n_samples=50,
    n_chains=8, num_burnin_steps=4000, num_results=4000,
    desired_energy_variance=5e-4,
    mclmc_debug=True,
)

model_seq = get_inference_builder("epl_shear_sersic_sersicshapelets")(system, n_max=N_MAX)
ctx = InferenceContext.from_modelling_sequence(model_seq)
print(f"[run] ctx hash: {ctx.hash()}")

pipeline = Pipeline(ctx, seed=0)
for s in get_pipeline_builder("map_bootstrap_mclmc_sersicshapelets")(system, **kw):
    pipeline.add(s)
print(f"[run] stages: {[s.instance_name for s in pipeline.stages]}")

pipeline.run(out_dir=os.path.join(OUT, SYS_ID), resume=True, verbose=True)

post = pipeline.posterior("mclmc")
print(f"[run] max_rhat={get_metric('max_rhat')(post, system):.3f}  "
      f"min_ess={get_metric('min_ess')(post, system):.1f}")
print("[run] mass z-scores:",
      {k: round(v, 2) for k, v in get_metric('mass_zscores')(post, system).items()})

# ---- Plots --------------------------------------------------------------
# Truth source: the actual Vela galaxy rendered at its truth centre.
vela = load_vela_source(src_dir)
truth_fn = truth_source_from_light_model(vela.light, system.truth_x[2][0])

report = PosteriorReport(
    post,
    prefix=f"{SYS_ID} (n_max={N_MAX}) ",
    truth_x=system.truth_x,
    truth_source_fn=truth_fn,
)
figs = report.full_report(save_dir=PLOTS, z_score_group="mass")
print(f"[run] full_report panels saved: {sorted(figs)} -> {PLOTS}")

# MCLMC tuning diagnostics (needs the debug=True capture above).
pr = PipelineReport(pipeline)
dfig = pr.diagnostics("mclmc")
dpath = os.path.join(PLOTS, "mclmc_diagnostics.png")
dfig.savefig(dpath, bbox_inches="tight", dpi=130)
print(f"[run] mclmc diagnostics saved -> {dpath}")
print("[run] DONE")
