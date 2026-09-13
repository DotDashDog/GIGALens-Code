"""The discrete multiplicity term, end to end: SELECTION == TARGET.

The whole point of the operator-based selection is that the truncation which
chose the dataset's truths is the truncation the inference target carries. This
test generates a tiny `double` dataset -- so the generator runs its
count-operator check on every candidate draw -- rebuilds the ProbModel with
``multiplicity: from_dataset``, and pins the three things that would silently
break that identity:

* the rebuilt ProbModel is ``discontinuous`` (the term is actually present, and
  therefore SVI/MCLMC will refuse it -- that refusal is what forced the
  ``map_mams`` pipeline);
* the term's count AT THE TRUTH equals the persisted observed image count for
  every system, and the log-prob there is finite. If selection and target used
  different operators, an accepted truth could sit outside its own target's
  support -- a -inf at the truth, which no sampler can recover from and which
  SBC would read as catastrophic miscalibration;
* the manifest reports ``operator_mismatch`` -- the residual disagreement
  between lenstronomy's floored count and the gigalens operator IS the residual
  SBC truncation mismatch, so it must be measured, not assumed.
"""
import os
import tempfile

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Cheap operator knobs: this test certifies the WIRING (same operator on both
# sides), not the operator's resolution -- gigalens' own
# tests/unit/test_point_source_multiplicity.py pins that against lenstronomy.
COUNT_FLOOR = {"mu_min": 0.1, "grid_n": 64, "inner_grid_n": 32}
N_SYSTEMS = 3
SEED = 17


def _generate(tmpdir):
    import gigalens_research.simtests.experiments.lenstronomy_point_source  # noqa: F401 (registers)
    from gigalens_research.simtests.config import CampaignSpec
    from gigalens_research.simtests.generate import generate_campaign
    from gigalens_research.simtests.system import load_manifest, System

    spec = CampaignSpec.from_dict({
        "name": "multiplicity_selection_test",
        "seed": SEED,
        "plugins": [
            "gigalens_research.simtests.experiments.lenstronomy_point_source"],
        "dataset": {
            "generator": "lenstronomy_point_source",
            "n_systems": N_SYSTEMS,
            "multiplicity": "double",
            "count_floor": dict(COUNT_FLOOR),
        },
        "inference": {
            "builder": "epl_shear_point_source_obs",
            "pipeline": "map_mams",
        },
        "sweep": [{}],
    })
    dataset_dir = generate_campaign(spec, tmpdir)
    manifest = load_manifest(dataset_dir)
    systems = [System.load(dataset_dir, sid) for sid in manifest["system_ids"]]
    return manifest, systems


def test_selection_operator_is_the_target_operator():
    import jax.numpy as jnp
    from gigalens_research.simtests.experiments.lenstronomy_point_source import (
        _truth_unique, build_epl_shear_point_source_obs,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        manifest, systems = _generate(tmpdir)

        # The manifest must MEASURE the residual selection/target mismatch.
        selection = manifest["extra"]["selection"]
        assert "operator_mismatch" in selection["rejects"], selection
        assert "operator_mismatch_rate" in selection, selection
        assert selection["operator_checks"] >= N_SYSTEMS, selection
        print(f"[multiplicity] operator checks {selection['operator_checks']}, "
              f"mismatches {selection['operator_mismatch']} "
              f"(rate {selection['operator_mismatch_rate']})")

        # The persisted count_floor is the operator the builder rebuilds.
        import json
        cfg = json.loads(systems[0].truth_assets["ps_config"])
        assert cfg["count_floor"]["mu_min"] == COUNT_FLOOR["mu_min"], cfg["count_floor"]
        assert cfg["lt_search_window"] == 12.0, cfg["lt_search_window"]

        for system in systems:
            # `multiplicity` defaults to "from_dataset" because the dataset
            # carries a persisted operator -- that default is part of the pin.
            prob = build_epl_shear_point_source_obs(system)
            assert prob.discontinuous, (
                "the multiplicity term must mark the ProbModel discontinuous; "
                "without it SVI/MCLMC would silently accept a walled target")
            mult_terms = [t for t in prob.terms
                          if getattr(t, "discontinuous", False)]
            assert len(mult_terms) == 1, [type(t).__name__ for t in prob.terms]
            term = mult_terms[0]

            obs = json.loads(system.truth_assets["ps_obs"])
            n_obs = int(obs["n_images"])
            assert n_obs == 2, n_obs                     # the `double` selection
            assert term.n_observed == n_obs

            unique = _truth_unique(prob.model, system.truth_x)
            assert unique is not None
            z_truth = jnp.asarray(
                np.asarray(prob.model.bijector.inverse(unique)).reshape(1, -1))

            count = int(np.asarray(
                term.count(prob.model.to_params(unique))).reshape(-1)[0])
            assert count == n_obs, (
                f"{system.system_id}: the count operator returns {count} at the "
                f"truth but the dataset selected {n_obs} observed images — "
                f"selection and target disagree")

            lp = float(np.asarray(prob.log_prob(z_truth)[0]).reshape(-1)[0])
            assert np.isfinite(lp), (
                f"{system.system_id}: log_prob at the truth is {lp}; an accepted "
                f"truth outside its own target's support means the selection "
                f"operator is not the target's operator")
            print(f"[multiplicity] {system.system_id}: count {count}, "
                  f"log_prob(truth) {lp:.3f}")


if __name__ == "__main__":
    test_selection_operator_is_the_target_operator()
    print("PASS")
