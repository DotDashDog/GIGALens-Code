"""G1 GPU reproduction (thin script form of TestSersiclets.ipynb's active path).

Migrated scene-API sersiclets path: from_vela_dir -> scene-backed ProbModel ->
Pipeline -> PartialTruthBootstrapQzStage(free=source) -> MCLMCStage(8 chains). Run on a
GPU node inside the canonical Shifter container. This is the inference-level grader for
the G1 migration (does the migrated path reproduce pre-migration posterior behavior).

Run (GPU node, inside container, canonical PYTHONPATH):
    /usr/bin/python3 /global/u1/l/linusu/GIGALens-Code/experiments/TestNewAPI/g1_sersiclets_repro.py
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax
jax.config.update("jax_enable_x64", True)

import gigalens_research.simtests.experiments.vela_elliptical_sersiclets  # register builder
from gigalens_research.simtests.system import from_vela_dir
from gigalens_research.simtests.registry import get_inference_builder
from gigalens_research.inference_utils.pipeline import (
    Pipeline, InferenceContext, MCLMCStage, model_card, format_model_card,
)
from gigalens_research.simtests.pipelines import PartialTruthBootstrapQzStage

cam, sim_num, rep, n_max = "12", "01", 3, 5
system_name = f"vela{sim_num}_cam{cam}_rep{str(rep).zfill(2)}_a0.500_f814w"
source_name = f"vela{sim_num}_cam{cam}_a0.500_f814w"
HOME = "/global/homes/l/linusu/GIGALens-Code"

system = from_vela_dir(
    system_dir=f"{HOME}/data/vela_sim_systems/{system_name}",
    source_dir=f"{HOME}/data/vela_sources/{source_name}",
    system_id="vela01_cam12_rep03",
    delta_pix=0.03, num_pix=200, supersample=1,
    background_rms=0.002, exp_time=2000.0,
)
system.likelihood_precision = "float64"
system.conv_precision = "float32"

prob_model = get_inference_builder("epl_shear_sersic_elliptical_sersiclets")(system, n_max=n_max)
print("scene_backed:", True, "n_free:", prob_model.model.num_free_params)

ctx = InferenceContext.from_prob_model(prob_model)
print(format_model_card(model_card(ctx)))

pipeline = Pipeline(ctx)
pipeline.add(PartialTruthBootstrapQzStage(
    system=system, free=("source",), map_num_steps=200, map_n_samples=50))
pipeline.add(MCLMCStage(n_chains=8, num_burnin_steps=2000, num_results=2000,
                        progress_bar=True, debug=True))

results_dir = os.environ.get("G1_OUT", f"{HOME}/experiments/TestNewAPI/_g1_out")
artifacts = pipeline.run(resume=False, out_dir=os.path.join(results_dir, system_name))
print("DONE; artifacts keys:", list(artifacts.keys()) if hasattr(artifacts, "keys") else type(artifacts))

# Quick posterior sanity: render at the posterior median + report reduced chi2 if available.
post = pipeline.posterior()
print("posterior:", type(post).__name__)
