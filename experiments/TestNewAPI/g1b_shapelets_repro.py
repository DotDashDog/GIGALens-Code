"""G1b GPU reproduction: epl_shear_sersic_shapelets (lstsq mode), thin script form of the sersiclets repro.

Migrated scene-API path: from_vela_dir -> scene-backed ProbModel -> Pipeline ->
PartialTruthBootstrapQzStage(free=source) -> MCLMCStage(8x2000+2000) on the real system.
Inference-level grader for the G1b builder migration (posterior reproduces pre-migration).

Run (GPU node, inside the canonical Shifter container, canonical PYTHONPATH):
    /usr/bin/python3 experiments/TestNewAPI/g1b_shapelets_repro.py
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
import jax
jax.config.update("jax_enable_x64", True)

import gigalens_research.simtests.experiments  # ensure all built-ins registered
from gigalens_research.simtests.system import from_vela_dir
from gigalens_research.simtests.registry import get_inference_builder
from gigalens_research.inference_utils.pipeline import (
    Pipeline, InferenceContext, MCLMCStage, model_card, format_model_card,
)
from gigalens_research.simtests.pipelines import PartialTruthBootstrapQzStage

cam, sim_num, rep = "12", "01", 3
n_max = 5
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

prob_model = get_inference_builder("epl_shear_sersic_shapelets")(system, n_max=n_max)
print("builder=epl_shear_sersic_shapelets scene_backed:", True,
      "mode:", prob_model.mode, "n_free:", prob_model.model.num_free_params)

ctx = InferenceContext.from_prob_model(prob_model)
print(format_model_card(model_card(ctx)))

pipeline = Pipeline(ctx)
pipeline.add(PartialTruthBootstrapQzStage(
    system=system, free=("source",), map_num_steps=200, map_n_samples=50))
pipeline.add(MCLMCStage(n_chains=8, num_burnin_steps=2000, num_results=2000,
                        progress_bar=True, debug=True))

results_dir = os.environ.get("G1B_OUT", f"{HOME}/experiments/TestNewAPI/_g1b_out_shapelets")
artifacts = pipeline.run(resume=False, out_dir=os.path.join(results_dir, system_name))
print("DONE; posterior:", type(pipeline.posterior()).__name__)
