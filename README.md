# GIGALens-Code

Research code for Bayesian strong-lens modeling experiments built on top of
the `gigalens` package. The repository contains reusable research utilities,
experiment scripts/notebooks, local input data, generated modeling results, and
paper/reference material.

This is a working research repo, not a polished package release. The main goal
of the current layout is to keep reusable code, experiment drivers, input data,
and generated outputs separated enough that results can be reproduced and code
can be refactored safely.

## Layout

- `src/gigalens_research/`: importable research package.
- `src/gigalens_research/inference/`: alternate inference algorithms and sampler
  implementations, including MCLMC/LAPS-related code.
- `src/gigalens_research/inference_utils/`: wrappers and utilities around
  inference workflows, such as pipeline result containers and diagnostics.
- `src/gigalens_research/plotting/`: plotting helpers for images, residuals,
  loss histories, and corner plots.
- `src/gigalens_research/voronoi_src/`: experimental pixelized source-modeling
  utilities.
- `experiments/`: runnable scripts and notebooks for specific research
  questions.
- `data/`: local input data used by experiments.
- `results/`: generated outputs from experiment runs.
- `papers/`: papers and references relevant to the project.
- `attic/`: old or inactive code kept for reference during the migration.
- `docs/`: setup notes and operational documentation.

## Environment

The canonical runtime is the JAX 2026 Shifter container plus the pinned conda
environment and sidecar described in `docs/env_setup.md`.

After activating the runtime, both source repos should be installed editable:

```bash
pip install --no-deps -e ~/gigalens
pip install --no-deps -e ~/GIGALens-Code
```

The `--no-deps` flag is intentional: dependencies are provided by the pinned
container/conda/sidecar stack, and pip should not upgrade JAX, TFP, NumPy, or
related packages.

## Common Imports

```python
from gigalens_research.inference import MCLMC
from gigalens_research.inference_utils import PipelineConfig, run_pipeline
from gigalens_research.plotting import plot_image_results, cornerplot_results
```

## Current Experiments

- `experiments/shapelets_systematics/`: Vela simulated-system shapelet
  systematics experiments. Inputs live under `data/vela_sim_systems`, and
  generated modeling outputs live under `results/shapelets_systematics`.
- `experiments/hundred_systems_GL2/`: hundred-system simulated-lens experiments.
- `experiments/benchmarking/`: timing and performance comparison scripts.
- `experiments/profiling/`: JAX/GPU profiling drivers.

## Notes

- Treat `data/` as input state and `results/` as generated state.
- Keep reusable code in `src/gigalens_research/`; keep one-off analysis and run
  drivers under `experiments/`.
- Prefer clear experiment names and config files over duplicating scripts with
  small parameter changes.
