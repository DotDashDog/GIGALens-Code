"""Baseline MAP -> Bridge(diag_qz) -> MCLMC run on the ADAPTIVE system-2 model,
sharded across all visible GPUs.

Mirrors ``experiments/sim_carousel/debug_carousel_1_3_4_5_9.ipynb`` cell 12
EXACTLY (same stage classes, same stage kwargs, same pipeline seed=42), but:

  - imports the model from ``carousel_model_s2.build()`` (this worktree's
    verified, unmodified extraction of that notebook's model-building cells)
    instead of re-running the notebook's own cells 0-9;
  - runs with ``resume=False`` (a FRESH run, never resuming/mutating a prior
    on-disk stage);
  - writes to a NEW output directory, never the archive.

Sharding
--------
This script does **not** implement its own sharding: the library already
shards both ``ModellingSequence.MAP`` (in ``gigalens.jax.inference``) and
``MCLMCStage``/``MCLMC_JIT`` (in ``gigalens_research.inference.mclmc``)
across every JAX-visible device, automatically, keyed off
``len(jax.devices())``:

  - ``gigalens/src/gigalens/jax/inference.py:59`` builds a module-level
    ``mesh = jax.make_mesh((len(jax.devices()),), ('device',))`` at import
    time; ``ModellingSequence.MAP`` (same file, ~line 127-176) floor-divides
    ``n_samples`` by the device count and runs the optimizer step under
    ``shard_map(..., mesh=mesh, in_specs=P('device'), ...)``.
  - ``gigalens_research/inference/mclmc.py:134-138,488,496`` computes
    ``num_devices = len(jax.devices())`` *inside* ``full_mclmc_with_adapt_sharded``
    (called on every ``MCLMCStage.run``), floor-divides ``n_chains`` by it,
    builds its own ``jax.make_mesh((num_devices,), ('device',))``, and runs
    the sampling scan under ``shard_map`` the same way.

So "using all 4 GPUs" reduces to: launch this process so that
``jax.devices()`` reports 4 GPUs (see the module docstring's launch note
below), and DO NOT set ``CUDA_VISIBLE_DEVICES``/``--gpus-per-node`` in a way
that hides any of them. No ``jax.distributed.initialize()`` is needed here
(that is only for MULTI-NODE runs, i.e. ``jax.process_count() > 1``; see
``gigalens/demo-multinode/demo_multinode.py:3`` vs.
``gigalens/src/gigalens/jax/inference.py:57-58``, which only warns when
``process_count() > 1``) -- this is a single-process, 4-local-GPU run.

n_samples=128 and n_chains=8 both divide evenly by 4 GPUs (32/device and
2/device respectively), so the library's floor-division shard split does not
truncate either count.

If ``jax.device_count()`` does NOT equal the expected device count at
runtime, this script raises rather than silently proceeding on fewer
devices -- pass ``--allow-device-mismatch`` to override (loudly warns, does
not hide the mismatch).

Launch (PYTHONPATH + allocation already set up; see
``docs/logs`` / the gpu-launch memory for the exact recipe)::

    cd <worktree>/experiments/flow_precond
    /usr/bin/python3 s2_baseline_run.py --out-dir /path/to/new/out/dir

Dry run (prints the model card, builds the pipeline, then exits BEFORE any
heavy compute -- safe to use for an import/config sanity check, including on
CPU)::

    python3 s2_baseline_run.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Safety: the human's irreplaceable archive. Never write here, under any
# override. Resolved against carousel_model_s2.S2_ROOT (the same constant
# that module uses to locate the archive read-only), not re-typed, so this
# can't drift out of sync with that module.
# ---------------------------------------------------------------------------

_THIS_DIR = Path(__file__).resolve().parent

DEFAULT_OUT_DIR = os.environ.get(
    "S2_BASELINE_OUT_DIR",
    "/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/debug_carousel"
    "/1_2_3_4_5_9_adaptive",
)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir", default=DEFAULT_OUT_DIR,
        help="Output directory for the pipeline run (default: %(default)s, "
             "or $S2_BASELINE_OUT_DIR).",
    )
    p.add_argument(
        "--expected-devices", type=int, default=4,
        help="Number of JAX devices this run expects to shard across "
             "(default: 4).",
    )
    p.add_argument(
        "--allow-device-mismatch", action="store_true",
        help="Proceed even if jax.device_count() != --expected-devices "
             "(still prints a loud warning; does not hide the mismatch).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Build the model + pipeline and print the model card, then "
             "exit before running any stage (no MAP/MCLMC compute).",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    # carousel_model_s2 sets jax_enable_x64=True at ITS import time, before
    # importing any gigalens module -- import it first so that ordering is
    # preserved (gigalens.jax.inference's module-level mesh/shard_map setup
    # must see x64 already configured).
    import carousel_model_s2 as s2

    out_dir = Path(args.out_dir).resolve()
    archive_dir = s2.S2_ROOT.resolve()

    # --- Guard 1: never point at (or inside) the human's archive. ---
    if out_dir == archive_dir or archive_dir in out_dir.parents:
        raise RuntimeError(
            f"REFUSING to run: out_dir {out_dir} is the (or is inside the) "
            f"human's irreplaceable archive {archive_dir}. Choose a "
            f"different --out-dir / $S2_BASELINE_OUT_DIR."
        )

    # --- Guard 2: refuse if out_dir already contains ANY manifest -- the
    # top-level pipeline.json (a completed run) OR any per-stage
    # manifest.json (a partial run that crashed before pipeline.json was
    # written). Either means this out_dir has been used before.
    existing_manifests = []
    top_manifest = out_dir / "pipeline.json"
    if top_manifest.exists():
        existing_manifests.append(top_manifest)
    if out_dir.exists():
        existing_manifests.extend(sorted(out_dir.glob("*/manifest.json")))
    if existing_manifests:
        raise RuntimeError(
            f"REFUSING to run: out_dir {out_dir} already contains "
            f"{[str(p) for p in existing_manifests]} -- this out_dir has "
            f"already been used for a (possibly partial) pipeline run. "
            f"This script only does FRESH runs (resume=False); pick a new "
            f"--out-dir rather than risk clobbering/mixing with a prior run."
        )

    import jax
    import tensorflow_probability.substrates.jax as tfp
    import jax.numpy as jnp
    from gigalens.jax.experimental.adaptive_supersample import AdaptiveImageData
    import gigalens.jax.inference as ginf  # exposes the module-level `mesh`
    from gigalens_research.inference_utils import (
        InferenceContext, Pipeline, MAPStage, BridgeStage, MCLMCStage,
    )

    tfd = tfp.distributions

    # --- Device / sharding check: loud, never silent. ---
    devices = jax.devices()
    n_dev = len(devices)
    if n_dev != args.expected_devices:
        banner = (
            "\n" + "!" * 78 + "\n"
            f"!!! DEVICE COUNT MISMATCH: expected {args.expected_devices} GPU(s), "
            f"jax.devices() reports {n_dev}: {devices}\n"
            f"!!! MAP/MCLMC will shard across only {n_dev} device(s), NOT "
            f"{args.expected_devices}.\n"
            + "!" * 78 + "\n"
        )
        print(banner, file=sys.stderr)
        if not args.allow_device_mismatch:
            raise RuntimeError(
                f"jax.device_count() == {n_dev}, expected "
                f"{args.expected_devices}. Pass --allow-device-mismatch to "
                f"proceed anyway (this will NOT silently hide the mismatch; "
                f"it is printed above and will be printed again in the "
                f"model card)."
            )

    # --- Build the (unmodified, verified) adaptive system-2 model. ---
    model_seq, prob_model = s2.build()
    model = prob_model.model

    ctx = InferenceContext.from_modelling_sequence(model_seq)

    # --- Construct the stages exactly as notebook cell 12 does. ---
    map_stage = MAPStage(num_steps=2000, n_samples=128)

    def make_diag_qz(z_best):
        return tfd.MultivariateNormalDiag(
            loc=jnp.asarray(z_best),
            scale_diag=jnp.full(z_best.shape[-1], 1e-2),
        )

    bridge_stage = BridgeStage(
        name="diag_qz",
        version="v1",
        requires=("z_best",),
        produces=("qz",),
        fn=make_diag_qz,
    )

    mclmc_stage = MCLMCStage(
        n_chains=8,
        num_burnin_steps=10000,
        num_results=10000,
        debug=True,
        progress_bar=True,
        regularize_mass_matrix=True,
    )

    # -----------------------------------------------------------------
    # MODEL CARD -- printed before any heavy compute.
    # -----------------------------------------------------------------
    print("=" * 78)
    print("S2 BASELINE RUN -- MODEL CARD (pre-compute)")
    print("=" * 78)
    print(f"jax version        : {jax.__version__}")
    print(f"jax.devices()       : {devices}")
    print(f"jax.device_count()  : {n_dev}  (expected {args.expected_devices})")
    print(f"jax.process_count() : {jax.process_count()}  "
          f"(>1 would need jax.distributed.initialize(); not called here)")
    print(f"gigalens.jax.inference.mesh (MAP/SVI): {ginf.mesh}  "
          f"(module-level, built at import time from len(jax.devices()))")
    print("gigalens_research MCLMCStage mesh: built PER-CALL inside "
          "full_mclmc_with_adapt_sharded as "
          "jax.make_mesh((len(jax.devices()),), ('device',)); not "
          "constructed until MCLMCStage.run() executes.")
    print(f"XLA_FLAGS (respected, not overridden): "
          f"{os.environ.get('XLA_FLAGS', '<unset>')!r}")
    print(f"jax_enable_x64      : {bool(jax.config.jax_enable_x64)}")
    print(f"out_dir             : {out_dir}")
    print(f"archive (READ-ONLY, never written): {archive_dir}")
    print(f"pipeline seed       : 42")
    print()
    print(f"model.num_free_params : {model.num_free_params}")
    print(f"num datasets          : {len(prob_model.datasets)}")
    for i, d in enumerate(prob_model.datasets):
        is_adaptive = isinstance(d, AdaptiveImageData)
        print(f"  dataset[{i}]: class={type(d).__name__} "
              f"is_AdaptiveImageData={is_adaptive} "
              f"image.shape={tuple(d.image.shape)}")
        if not is_adaptive:
            raise RuntimeError(
                f"dataset[{i}] is {type(d).__name__}, not AdaptiveImageData "
                f"-- carousel_model_s2.build() is expected to build "
                f"AdaptiveImageData datasets only (per its own docstring). "
                f"Refusing to proceed with an unexpected dataset class."
            )
    print()
    print("Stage configs (as constructed, via each stage's own "
          "config_hash_data()):")
    print(f"  MAPStage    : {map_stage.config_hash_data()}")
    print(f"  BridgeStage : name={bridge_stage.instance_name!r} "
          f"{bridge_stage.config_hash_data()}  "
          f"requires={bridge_stage.requires} produces={bridge_stage.produces}")
    print(f"  MCLMCStage  : {mclmc_stage.config_hash_data()}")
    print("=" * 78)

    if args.dry_run:
        print("[--dry-run] Exiting before pipeline.run() -- no MAP/MCLMC "
              "compute was executed.")
        return

    # -----------------------------------------------------------------
    # Build and run the pipeline (FRESH: resume=False).
    # -----------------------------------------------------------------
    pipeline = Pipeline(ctx, seed=42)
    pipeline.add(map_stage)
    pipeline.add(bridge_stage)
    pipeline.add(mclmc_stage)

    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = pipeline.run(out_dir=str(out_dir), resume=False)
    print(f"[s2_baseline_run] DONE. Artifacts written under {out_dir}: "
          f"{sorted(artifacts.keys())}")


if __name__ == "__main__":
    main()
