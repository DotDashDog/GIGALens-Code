"""Fix 0 — precision-mechanics spike for the mixed-precision (float64-likelihood) plan.

Validates, with jax_enable_x64=True:
  (a) the simulator coordinate grids + basis + conv stay float32 (so memory ~ float32 baseline),
  (b) an explicit .astype(float64) yields TRUE float64 (no silent truncation),
  (c) float32 inputs are preserved through ops (only explicit casts / float64 literals promote).

NO behavior change to the library; read-only dtype inspection. Run on CPU (dtype semantics are
device-independent); a separate --device gpu pass checks memory at n_max=25.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

home = os.path.expanduser("~/")
for _p in [os.path.join(home, "sidecar_jax_upgrade"),
           os.path.join(home, "gigalens/src"),
           os.path.join(home, "GIGALens-Code/src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-max", type=int, default=25)
    args = ap.parse_args()

    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    print(f"[fix0] jax {jax.__version__}  x64={jax.config.jax_enable_x64}  devices={jax.devices()}")

    # ---- (c) bare dtype mechanics ----
    a32 = jnp.ones((3,), dtype=jnp.float32)
    print("[fix0] mechanics:")
    print(f"    float32 + float32         -> {(a32 + a32).dtype}   (expect float32)")
    print(f"    float32 * python 2.0      -> {(a32 * 2.0).dtype}   (weak float: expect float32)")
    print(f"    float32 + jnp.float64 lit -> {(a32 + jnp.float64(2.0)).dtype}   (expect float64: promotion)")
    print(f"    float32 @ float32         -> {(a32 @ a32).dtype}   (expect float32)")
    print(f"    float32.astype(float64)   -> {a32.astype(jnp.float64).dtype}   (expect float64: TRUE x64)")
    big = jnp.arange(40000, dtype=jnp.float32)
    s32 = jnp.sum(big * big)
    s64 = jnp.sum((big * big).astype(jnp.float64))
    print(f"    sum f32 dtype={s32.dtype}  sum-after-cast dtype={s64.dtype}  (expect float32, float64)")

    # ---- (a) real model: coordinate grid + basis + conv dtypes ----
    import gigalens.jax.simulator as sim
    from gigalens_research.simtests.system import from_vela_dir
    from gigalens_research.simtests.experiments import vela_shapelets  # noqa: F401
    from gigalens_research.simtests.registry import get_inference_builder

    data_dir = os.path.join(home, "GIGALens-Code/data")
    system_name = "vela01_cam12_rep03_a0.500_f814w"
    system = from_vela_dir(
        system_dir=os.path.join(data_dir, "vela_sim_systems", system_name),
        source_dir=os.path.join(data_dir, "vela_sources", system_name),
        system_id="vela01_cam12_rep03",
        delta_pix=0.03, num_pix=200, supersample=1,
        background_rms=0.002, exp_time=2000.0,
    )
    model_seq = get_inference_builder("epl_shear_sersic_shapelets")(system, n_max=args.n_max)
    lens_sim = sim.LensSimulator(model_seq.phys_model, model_seq.sim_config, bs=1)

    print("[fix0] simulator grid dtypes under x64:")
    for name in ["img_X", "img_Y", "flat_kernel"]:
        v = getattr(lens_sim, name, None)
        print(f"    {name:12s} dtype={getattr(v, 'dtype', None)}  shape={getattr(v, 'shape', None)}")

    pm = model_seq.prob_model
    # logp dtype end-to-end (no library change yet => everything float32 unless a grid promoted)
    z = jnp.zeros((1, 17), dtype=jnp.float32)
    lp = pm.log_prob(lens_sim, z)[0]
    print(f"[fix0] log_prob output dtype={jnp.asarray(lp).dtype}  value={float(np.asarray(lp)):.6g}")

    # basis/conv dtype via the lstsq path internals: call simulate to get a model image
    im = lens_sim.lstsq_simulate(pm.bij.forward(list(z.T)), pm.observed_image, pm.err_map)[0]
    print(f"[fix0] lstsq model image dtype={im.dtype}  shape={im.shape}")
    print(f"[fix0] observed_image dtype={pm.observed_image.dtype}  err_map dtype={pm.err_map.dtype}")

    # ---- verdict ----
    grids_f32 = all(getattr(getattr(lens_sim, n, None), "dtype", None) == jnp.float32
                    for n in ["img_X", "img_Y"])
    print(f"\n[fix0] VERDICT: coordinate grids float32 under x64 = {grids_f32}")
    print("[fix0]   -> if True: mixed precision feasible (basis stays f32; promote only at casts).")
    print("[fix0]   -> if False: grids promoted to f64 => basis 2x memory => need explicit f32 pin or escalate.")


if __name__ == "__main__":
    main()
