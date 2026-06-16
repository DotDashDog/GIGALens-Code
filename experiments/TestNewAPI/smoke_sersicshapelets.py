"""Single-system smoke test of the SersicShapelets simtests pipeline.

Drives ONE Vela system through the registered inference builder
(``epl_shear_sersic_sersicshapelets``) and pipeline
(``map_bootstrap_mclmc_sersicshapelets``) with reduced settings, exercising the
SersicShapeletsBootstrapQzStage -> MCLMC path end-to-end. Not a science run.
"""
import json
import os

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np

import gigalens_research.simtests.experiments  # registers builders/pipelines
from gigalens_research.simtests.system import from_vela_dir
from gigalens_research.simtests.registry import (
    get_inference_builder, get_pipeline_builder, get_metric,
)
from gigalens_research.inference_utils.pipeline import InferenceContext, Pipeline

HOME = os.path.expanduser("~")
DATA = os.path.join(HOME, "GIGALens-Code", "data")
OUT = os.path.join(HOME, "GIGALens-Code", "results", "smoke_sersicshapelets")

src_dir = os.path.join(DATA, "vela_sources", "vela01_cam12_a0.500_f814w")
sys_dir = os.path.join(DATA, "vela_sim_systems", "vela01_cam12_rep00_a0.500_f814w")
delta_pix = float(json.load(open(os.path.join(src_dir, "metadata.json")))
                  ["instrument_pixel_scale_arcsec"])

system = from_vela_dir(
    system_dir=sys_dir, source_dir=src_dir, system_id="vela01_cam12_rep00",
    delta_pix=delta_pix, num_pix=200, supersample=1,
    background_rms=0.002, exp_time=2000.0,
)
print(f"[smoke] system loaded: image {system.observed_image.shape}, delta_pix={delta_pix}")

N_MAX = 5
# Reduced kwargs for a fast, light smoke test (not science settings).
kw = dict(
    n_max=N_MAX,
    bootstrap_map_steps=50, bootstrap_map_n_samples=20,
    n_chains=2, num_burnin_steps=200, num_results=200,
    desired_energy_variance=5e-4,
)

model_seq = get_inference_builder("epl_shear_sersic_sersicshapelets")(system, n_max=N_MAX)
ctx = InferenceContext.from_modelling_sequence(model_seq)
print(f"[smoke] ctx hash: {ctx.hash()}")

stages = get_pipeline_builder("map_bootstrap_mclmc_sersicshapelets")(system, **kw)
pipeline = Pipeline(ctx, seed=0)
for s in stages:
    pipeline.add(s)
print(f"[smoke] stages: {[s.instance_name for s in pipeline.stages]}")

artifacts = pipeline.run(out_dir=os.path.join(OUT, "vela01_cam12_rep00"),
                         resume=False, verbose=True)
print(f"[smoke] pipeline finished. artifacts: {list(artifacts) if hasattr(artifacts, '__iter__') else artifacts}")

# Posterior diagnostics + truth recovery on shared params.
post = pipeline.posterior() if hasattr(pipeline, "posterior") else None
if post is not None:
    print(f"[smoke] max_rhat={get_metric('max_rhat')(post, system):.3f}  "
          f"min_ess={get_metric('min_ess')(post, system):.1f}")
    z = get_metric('mass_zscores')(post, system)
    print("[smoke] mass z-scores:", {k: round(v, 2) for k, v in z.items()})
print("[smoke] DONE")
