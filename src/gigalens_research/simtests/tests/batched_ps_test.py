"""Equivalence tests for ``BatchedPointSourceProb`` (batched SBC log-prob).

The batched path must be EXACTLY the solo path with a system axis vmapped over
it — same float64 ops, same solve, same normalization. These tests generate a
tiny quad campaign, build the solo models, and assert value- and gradient-level
agreement between ``BatchedPointSourceProb`` and each solo ``ProbModel``, plus
the raise-never-default guards on non-batchable inputs.
"""
import os
import sys
import tempfile

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# The batched program is the SAME float64 code path, but XLA compiles the
# vmapped computation with different fusion/FMA reassociation than the solo
# one, and the iterated Levenberg solve amplifies that roundoff (the gigalens
# module docstring documents the amplification). Measured batched-vs-solo
# disagreement: <= 2e-8 relative, including at hostile off-prior probes.
# These tolerances sit well above that and far below anything dynamically
# meaningful (MCLMC's energy-error tolerance is 5e-4); a data-swap bug would
# show as O(1) shifts, not 1e-8.
RTOL = 1e-6
ATOL = 1e-5


def _generate(tmpdir, n_systems=3, seed=11):
    import gigalens_research.simtests.experiments.lenstronomy_point_source  # noqa: F401 (registers)
    from gigalens_research.simtests.config import CampaignSpec
    from gigalens_research.simtests.generate import generate_campaign
    from gigalens_research.simtests.system import load_manifest, System

    spec = CampaignSpec.from_dict({
        "name": "batched_ps_test",
        "seed": seed,
        "plugins": ["gigalens_research.simtests.experiments.lenstronomy_point_source"],
        "dataset": {
            "generator": "lenstronomy_point_source",
            "n_systems": n_systems,
            "multiplicity": "quad",
        },
        "inference": {
            "builder": "epl_shear_point_source_obs",
            "pipeline": "map_svi_mclmc",
        },
        "sweep": [{}],
    })
    dataset_dir = generate_campaign(spec, tmpdir)
    m = load_manifest(dataset_dir)
    return [System.load(dataset_dir, sid) for sid in m["system_ids"]]


def _build_all(systems, **kw):
    from gigalens_research.simtests.experiments.lenstronomy_point_source import (
        build_epl_shear_point_source_obs,
    )
    seqs = [build_epl_shear_point_source_obs(s, **kw) for s in systems]
    return [getattr(s, "prob_model", s) for s in seqs]


def _z_points(probs, systems, n_random=4, seed=3):
    """Per-system z probes: the truth point (typical, converged-solve region)
    plus shared random unconstrained draws (arbitrary, possibly hostile)."""
    import jax.numpy as jnp
    from gigalens_research.simtests.experiments.lenstronomy_point_source import (
        _truth_unique,
    )

    scene = probs[0].model
    p_dim = len(scene.z_param_names)
    rng = np.random.default_rng(seed)
    zr = rng.normal(size=(n_random, p_dim)) * 0.7
    zs = []
    for prob, system in zip(probs, systems):
        unique = _truth_unique(prob.model, system.truth_x)
        z_truth = np.asarray(prob.model.bijector.inverse(unique)).reshape(1, -1)
        zs.append(np.concatenate([z_truth, zr], axis=0))
    return jnp.asarray(np.stack(zs))          # (S, 1 + n_random, P)


def test_log_prob_matches_solo():
    """Batched (S, C, P) log_prob/log_like == per-system solo results."""
    import jax.numpy as jnp
    from gigalens_research.simtests.experiments.batched_point_source import (
        BatchedPointSourceProb,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        systems = _generate(tmpdir)
        probs = _build_all(systems)
        bp = BatchedPointSourceProb.from_probs(probs)

        # The test is vacuous if the systems share data — assert heterogeneity.
        x = np.asarray(bp.data["x"])
        assert not np.allclose(x[0], x[1]), "systems have identical image positions"

        z = _z_points(probs, systems)
        lp_b, red_b = bp.log_prob(z)
        ll_b, _ = bp.log_like(z)
        for i, prob in enumerate(probs):
            lp_s, red_s = prob.log_prob(z[i])
            ll_s, _ = prob.log_like(z[i])
            np.testing.assert_allclose(np.asarray(lp_b[i]), np.asarray(lp_s),
                                       rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(np.asarray(red_b[i]), np.asarray(red_s),
                                       rtol=RTOL, atol=ATOL)
            np.testing.assert_allclose(np.asarray(ll_b[i]), np.asarray(ll_s),
                                       rtol=RTOL, atol=ATOL)
        # Finiteness is only guaranteed at the truth points (z index 0); the
        # random probes may legitimately land outside prior support in BOTH
        # paths (equivalence there is covered by the allclose above, which
        # treats matching non-finite lanes as equal).
        assert np.all(np.isfinite(np.asarray(lp_b[:, 0]))), \
            "non-finite log_prob at a truth point"

        # Cross-check the swap actually matters: system 0's data must NOT
        # reproduce system 1's solo values (different observations).
        lp_01 = np.asarray(probs[0].log_prob(z[1])[0])
        assert not np.allclose(np.asarray(lp_b[1]), lp_01, rtol=1e-6), \
            "system 1's batched lane returned system 0's likelihood — swap inert"

        # Gradients (what the samplers consume) must match too.
        import jax
        g_b = jax.grad(lambda zz: bp.log_prob(zz)[0].sum())(z)
        for i, prob in enumerate(probs):
            g_s = jax.grad(lambda zz: prob.log_prob(zz)[0].sum())(z[i])
            np.testing.assert_allclose(np.asarray(g_b[i]), np.asarray(g_s),
                                       rtol=1e-6, atol=1e-4)

        # And the whole thing must be jit-compatible.
        lp_j, _ = jax.jit(bp.log_prob)(z)
        np.testing.assert_allclose(np.asarray(lp_j), np.asarray(lp_b),
                                   rtol=RTOL, atol=ATOL)
    print("  batched log_prob == solo (values, reduced chi2, grads, jit): OK")


def test_anchor_variant_matches_solo():
    """Equivalence holds with the source-plane anchor on (traced branch)."""
    from gigalens_research.simtests.experiments.batched_point_source import (
        BatchedPointSourceProb,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        systems = _generate(tmpdir, n_systems=2, seed=23)
        probs = _build_all(systems, src_anchor_sigma=0.004)
        bp = BatchedPointSourceProb.from_probs(probs)
        z = _z_points(probs, systems, n_random=2)
        lp_b, _ = bp.log_prob(z)
        for i, prob in enumerate(probs):
            lp_s, _ = prob.log_prob(z[i])
            np.testing.assert_allclose(np.asarray(lp_b[i]), np.asarray(lp_s),
                                       rtol=RTOL, atol=ATOL)
    print("  batched == solo with src_anchor_sigma on: OK")


def test_guards_refuse_non_batchable():
    """Static structure mismatches must raise, never batch silently."""
    from gigalens_research.simtests.experiments.batched_point_source import (
        BatchedPointSourceProb,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        systems = _generate(tmpdir, n_systems=2, seed=42)
        base = _build_all(systems)

        # Different solver knob (newton_steps enters the traced solve as a
        # Python constant — mixed values would silently mis-solve one lane).
        other = _build_all(systems[1:], newton_steps=5)
        try:
            BatchedPointSourceProb.from_probs([base[0], other[0]])
        except ValueError as e:
            assert "not batchable" in str(e), e
        else:
            raise AssertionError("mixed newton_steps was not refused")

        # Mixed anchor on/off selects different trace-time branches.
        anchored = _build_all(systems[1:], src_anchor_sigma=0.004)
        try:
            BatchedPointSourceProb.from_probs([base[0], anchored[0]])
        except ValueError as e:
            assert "not batchable" in str(e), e
        else:
            raise AssertionError("mixed anchor on/off was not refused")

        # Mixed channel configuration (td off drops the cosmology and shrinks
        # the free-parameter list).
        no_td = _build_all(systems[1:], fit_td=False)
        try:
            BatchedPointSourceProb.from_probs([base[0], no_td[0]])
        except ValueError as e:
            assert ("not batchable" in str(e) or "free-parameter" in str(e)), e
        else:
            raise AssertionError("mixed fit_td was not refused")
    print("  non-batchable guards (newton_steps, anchor, channels): OK")


def main():
    print("Running batched point-source tests...")
    test_log_prob_matches_solo()
    test_anchor_variant_matches_solo()
    test_guards_refuse_non_batchable()
    print("\nAll batched point-source tests passed.")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.abspath(os.path.join(here, "../../../../")))
    sys.path.insert(0, os.path.expanduser("~/gigalens/src"))
    sys.path.insert(0, os.path.expanduser("~/GIGALens-Code/src"))
    main()
