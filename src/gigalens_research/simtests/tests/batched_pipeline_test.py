"""Certification tests for the batched MAP -> SVI -> MCLMC pipeline.

The batched stages mirror the solo implementations' math and RNG streams, but
XLA compiles different programs (vmap fusion; no shard_map) and the samplers
are chaotic, so trajectories diverge from reassociation roundoff and bitwise
equality is unattainable BY CONSTRUCTION. Certification is therefore:

1. optimizer stages reach the same optimum (MAP best-lp gap, SVI surrogate
   agreement in posterior-scale units), and
2. MCLMC posteriors agree within Monte-Carlo error (mean shifts in posterior-sd
   units bounded by ~few / sqrt(ESS); sd ratios near 1), with comparable
   adapted step_size / L.

Thresholds were calibrated on the measured solo-vs-batched gaps (see the
printed report; values chosen with ~3x margin over observed) — loose enough
not to flake on sampler noise, tight enough that a wrong mass matrix, a
mis-swapped data row, or a broken adaptation stage fails immediately.

KNOWN FAILURE, 2026-09-04 (UNCERTIFIED — for Linus, not silently patched).
On the Perlmutter login node, against the gigalens ``multiplicity-term`` branch,
this test FAILS at the MAP gate: "system 0: MAP optima differ by -2.897 lp
units" (threshold 2.0). What is established:

* the batched side did not change — ``batched_map`` / ``batched_svi`` /
  ``batched_mclmc`` are byte-identical to main apart from the deleted
  ``batched_map_anneal`` and the mirrored admissible-init redraw (a measured
  no-op here: the gap is identical to 4 decimals with and without it);
* the solo side did NOT change numerically: gigalens ``MAP`` on the
  ``multiplicity-term`` branch was pinned BIT-IDENTICAL to its base commit
  (fd63b1d) on a point-source model — 64 particles x 300 Adam steps, same
  seed, identical ``z_best`` and log-prob to the last bit. The init redraw is
  a no-op when every prior draw is finite, and the scan body is untouched;
* the gap is system- and context-dependent, not a fixed offset. A MAP-only
  probe on this fixture measured gaps of -0.17 and -3.42 (dataset at
  ``lt_search_window`` 6) and -0.30 and +1.05 (window 12, the new default),
  while the same system inside the full test reads -2.90 — i.e. the quantity
  the 2.0 threshold gates ranges over ~4.5 lp units here;
* the decisive control — this test on unmodified ``main`` with the base
  gigalens — could NOT be completed: the login node killed it (OOM) twice.
  The threshold was calibrated on Lawrencium (2026-07-24) and this test has
  never passed on Perlmutter, so a pre-existing platform/JAX-version
  sensitivity of the chaotic 400-step Adam comparison is the open hypothesis.

Re-deriving the threshold on a compute node (or replacing the MAP gate with
the basin-identity check it is really trying to assert) is a certification
decision, so it is left to the human rather than loosened here.
"""
import os
import sys
import tempfile

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Small-but-real budgets: enough steps for every tuning phase (windows, sync,
# L adaptation) to execute, small enough for a CPU test run.
MAP_KW = dict(num_steps=400, n_samples=256, map_lr=3e-3, map_clip_norm=1.0)
SVI_KW = dict(num_steps=400, n_vi=128, init_scales=1e-3, svi_lr=1e-4)
MC_KW = dict(n_chains=4, num_burnin_steps=2000, num_results=3000,
             desired_energy_variance=5e-4, frac_tune1=0.2, frac_tune2=0.6,
             frac_tune3=0.2)
SEEDS = {"map": (11, 21), "svi": (12, 22), "mclmc": (13, 23)}


def _solo_pipeline(prob_model, seed_map, seed_svi, seed_mclmc):
    import jax.numpy as jnp
    import optax
    from gigalens.jax.inference import MAP as _MAP, SVI as _SVI
    from gigalens_research.inference import MCLMC_JIT

    map_opt = optax.chain(optax.zero_nans(),
                          optax.clip_by_global_norm(MAP_KW["map_clip_norm"]),
                          optax.adam(MAP_KW["map_lr"]))
    samples, lps, chisqs = _MAP(
        prob_model,
        optimizer=map_opt, start=None, n_samples=MAP_KW["n_samples"],
        num_steps=MAP_KW["num_steps"], seed=seed_map,
        output_type="best_step", pbar_interval=0)
    best = int(np.nanargmax(np.asarray(lps)))
    z_best = np.asarray(samples)[best]

    qz_raw, _ = _SVI(
        prob_model,
        start=jnp.asarray(z_best), optimizer=optax.adabelief(
            SVI_KW["svi_lr"], b1=0.95, b2=0.99),
        n_vi=SVI_KW["n_vi"], init_scales=SVI_KW["init_scales"],
        num_steps=SVI_KW["num_steps"], seed=seed_svi, pbar_interval=0)
    # The real pipeline round-trips qz through saved numpy arrays
    # (SVIStage arrays -> MCLMCStage derive_artifacts); mirror that, both for
    # fidelity and because the live shard_map-sharded arrays trip JAX 0.10's
    # closing-over check inside MCLMC_JIT.
    import tensorflow_probability.substrates.jax as tfp
    qz = tfp.distributions.MultivariateNormalTriL(
        loc=jnp.asarray(np.asarray(qz_raw.loc)),
        scale_tril=jnp.asarray(np.asarray(qz_raw.scale_tril)))

    samples_z = MCLMC_JIT(
        prob_model=prob_model, qz=qz, n_hmc=MC_KW["n_chains"],
        num_burnin_steps=MC_KW["num_burnin_steps"],
        num_results=MC_KW["num_results"],
        desired_energy_variance=MC_KW["desired_energy_variance"],
        frac_tune1=MC_KW["frac_tune1"], frac_tune2=MC_KW["frac_tune2"],
        frac_tune3=MC_KW["frac_tune3"], regularize_mass_matrix=True,
        progress_bar=False, seed=seed_mclmc)
    return {"z_best": z_best, "lp_best": float(np.asarray(lps)[best]),
            "qz_loc": np.asarray(qz.loc),
            "qz_scale_tril": np.asarray(qz.scale_tril),
            "samples_z": np.asarray(samples_z)}


def test_batched_pipeline_matches_solo():
    import jax.numpy as jnp
    try:
        from batched_ps_test import _generate, _build_all
    except ImportError:
        from gigalens_research.simtests.tests.batched_ps_test import (
            _generate, _build_all,
        )
    from gigalens_research.simtests.experiments.batched_point_source import (
        BatchedPointSourceProb,
    )
    from gigalens_research.simtests.experiments.batched_pipeline import (
        batched_map, batched_svi, batched_mclmc,
    )

    with tempfile.TemporaryDirectory() as td:
        systems = _generate(td, n_systems=2, seed=31)
        seqs = _build_all(systems)      # ProbModels
        bp = BatchedPointSourceProb.from_probs(seqs)

        # The builder returns the scene ProbModel directly, which is what both
        # the batched prob and the solo free-function stages consume.
        solo = [_solo_pipeline(seqs[i], SEEDS["map"][i], SEEDS["svi"][i],
                               SEEDS["mclmc"][i]) for i in range(2)]

        # Stage-isolated comparison: each batched stage gets the SAME input the
        # solo stage got (the solo predecessor's output). Comparing chained
        # outputs instead would mostly measure the predecessor stages'
        # trajectory divergence along flat degeneracy directions (measured:
        # a 0.75-lp-equivalent MAP z_best difference dominates the SVI loc
        # comparison), not the stage under test. The full chained pipeline is
        # exercised end-to-end against real campaign posteriors separately
        # (docs/logs/batched-point-source.md, phase-B validation).
        bmap = batched_map(bp, np.asarray(SEEDS["map"]),
                           num_steps=MAP_KW["num_steps"],
                           n_samples=MAP_KW["n_samples"],
                           map_lr=MAP_KW["map_lr"],
                           map_clip_norm=MAP_KW["map_clip_norm"])
        z_best_solo = np.stack([s["z_best"] for s in solo])
        bsvi = batched_svi(bp, z_best_solo, np.asarray(SEEDS["svi"]),
                           num_steps=SVI_KW["num_steps"], n_vi=SVI_KW["n_vi"],
                           init_scales=SVI_KW["init_scales"],
                           svi_lr=SVI_KW["svi_lr"])
        bmc = batched_mclmc(bp, np.stack([s["qz_loc"] for s in solo]),
                            np.stack([s["qz_scale_tril"] for s in solo]),
                            np.asarray(SEEDS["mclmc"]), **MC_KW)

        n_converged_compared = [0]
        for i, prob in enumerate(seqs):
            print(f"--- system {i} ---")
            # MAP: same optimum quality, judged by the SOLO model's own lp.
            lp_b = float(np.asarray(
                prob.log_prob(jnp.asarray(bmap["z_best"][i])[None])[0])[0])
            lp_s = solo[i]["lp_best"]
            print(f"MAP lp: solo {lp_s:.3f} batched {lp_b:.3f} gap {lp_b - lp_s:+.4f}")
            assert abs(lp_b - lp_s) < 2.0, \
                f"system {i}: MAP optima differ by {lp_b - lp_s:+.3f} lp units"

            # SVI: surrogate agreement in units of the surrogate's own sd.
            sd_s = np.sqrt(np.diag(
                solo[i]["qz_scale_tril"] @ solo[i]["qz_scale_tril"].T))
            dloc = (bsvi["qz_loc"][i] - solo[i]["qz_loc"]) / sd_s
            sd_b = np.sqrt(np.diag(
                bsvi["qz_scale_tril"][i] @ bsvi["qz_scale_tril"][i].T))
            print(f"SVI |dloc|_max {np.abs(dloc).max():.3f} qz-sd; "
                  f"sd ratio [{(sd_b / sd_s).min():.3f}, {(sd_b / sd_s).max():.3f}]")
            assert np.abs(dloc).max() < 1.0, f"system {i}: SVI loc off by >1 qz-sd"
            assert np.all((sd_b / sd_s > 0.5) & (sd_b / sd_s < 2.0)), \
                f"system {i}: SVI sd ratio outside [0.5, 2]"

            # MCLMC: posterior moments within MC error. The comparison is only
            # meaningful if BOTH runs converged — report R-hat/ESS for both so
            # a failure can be attributed (unconverged reference vs real
            # fidelity problem).
            import tensorflow_probability.substrates.jax as tfp

            def _health(chains):        # (C, N, P)
                s = jnp.asarray(np.swapaxes(chains, 0, 1))
                rh = float(np.asarray(tfp.mcmc.potential_scale_reduction(
                    s, independent_chain_ndims=1)).max())
                es = float(np.asarray(tfp.mcmc.effective_sample_size(
                    s, cross_chain_dims=1)).min())
                return rh, es

            rh_s, ess_s = _health(solo[i]["samples_z"])
            rh_b, ess_b = _health(bmc["samples_z"][i])
            s_s = solo[i]["samples_z"].reshape(-1, solo[i]["samples_z"].shape[-1])
            s_b = bmc["samples_z"][i].reshape(-1, bmc["samples_z"].shape[-1])
            sd = s_s.std(axis=0)
            dmean = (s_b.mean(axis=0) - s_s.mean(axis=0)) / sd
            ratio = s_b.std(axis=0) / sd
            print(f"MCLMC |dmean|_max {np.abs(dmean).max():.3f} sd; "
                  f"sd ratio [{ratio.min():.3f}, {ratio.max():.3f}]; "
                  f"nonan {bmc['results_nonan_frac'][i]:.3f}; "
                  f"solo R-hat {rh_s:.3f}/ESS {ess_s:.0f} | "
                  f"batched R-hat {rh_b:.3f}/ESS {ess_b:.0f}; "
                  f"batched ss {bmc['final_step_size'][i, 0]:.4f} "
                  f"L {bmc['final_L'][i]:.3f}")
            assert bmc["results_nonan_frac"][i] > 0.99, \
                f"system {i}: NaN-rejection rate {1 - bmc['results_nonan_frac'][i]:.2%}"
            assert np.isfinite(s_b).all(), f"system {i}: non-finite batched draws"
            # Moment comparison is only meaningful when BOTH runs converged at
            # this deliberately small test budget (measured: a hard generated
            # system reached solo R-hat 3.2 / ESS 6 — comparing moments there
            # compares noise to noise). Gate, don't pretend.
            if rh_s > 1.2 or rh_b > 1.2:
                print(f"    (system {i}: unconverged at test budget on "
                      f"{'solo' if rh_s > 1.2 else 'batched'} side — moment "
                      "comparison skipped; full-budget validation lives in the "
                      "phase-B real-data check)")
                continue
            n_converged_compared[0] += 1
            assert np.abs(dmean).max() < 0.5, \
                f"system {i}: MCLMC mean shifted {np.abs(dmean).max():.2f} sd"
            assert np.all((ratio > 0.6) & (ratio < 1.67)), \
                f"system {i}: MCLMC sd ratio outside [0.6, 1.67]: {ratio}"
        assert n_converged_compared[0] >= 1, \
            "no system converged in both paths — the MCLMC moment comparison " \
            "never ran; raise the test budget rather than passing vacuously"
    print("  batched pipeline == solo (MAP optimum, SVI surrogate, "
          "MCLMC posterior): OK")


def main():
    print("Running batched pipeline certification test...")
    test_batched_pipeline_matches_solo()
    print("\nBatched pipeline certification passed.")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    sys.path.insert(0, os.path.expanduser("~/gigalens/src"))
    sys.path.insert(0, os.path.expanduser("~/GIGALens-Code/src"))
    main()
