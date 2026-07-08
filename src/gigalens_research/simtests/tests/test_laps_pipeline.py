"""CPU tests for the LAPS transferability battery plumbing:

1. The ``map_svi_hmc_laps`` pipeline builder (``simtests/pipelines.py``):
   registration, stage order/names, certified-preset preservation, and the
   ``laps_num_chains`` / ``laps_keep`` / ``laps_seed`` /
   ``laps_cold_min_survivors`` kwargs.
2. The ``gl2_existing`` generator's pure ``system_ids`` subset-selection logic
   (``_resolve_selected_indices``) and manifest filtering — exercised WITHOUT
   touching the real npz/yaml/PSF assets (those are integration-tested by the
   existing ``gl2_existing`` campaigns, not here).
3. ``experiments/laps_transfer/campaign.yaml`` parses through ``CampaignSpec``
   with the expected resolved fields.
4. ``experiments/laps_transfer/analyze.py``'s pure/cheap metric functions
   (Tier 1 / Tier 2 / Tier 3), plus a fabricated-on-disk-run-dir smoke test of
   the ``posterior_from_disk`` / ``diagnostics_from_disk`` loading path those
   functions consume.

No GPU, no real data; mirrors the style of ``simtests/tests/smoke_test.py``
and ``inference/tests/test_laps_stage.py``.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest
import tensorflow_probability.substrates.jax as tfp

import gigalens_research.simtests.experiments  # noqa: F401  (registers all built-ins)
from gigalens_research.simtests.registry import get_pipeline_builder

tfd = tfp.distributions


# --------------------------------------------------------------------------- #
# 1 -- map_svi_hmc_laps pipeline builder                                     #
# --------------------------------------------------------------------------- #


def test_map_svi_hmc_laps_registered():
    from gigalens_research.simtests.registry import list_registered
    reg = list_registered()
    assert "map_svi_hmc_laps" in reg["pipeline_builders"]


def test_map_svi_hmc_laps_stage_order_and_names():
    build = get_pipeline_builder("map_svi_hmc_laps")
    stages = build(system=None)
    assert len(stages) == 5
    assert [s.instance_name for s in stages] == [
        "map", "svi", "hmc", "laps_warm", "laps_cold",
    ]
    assert [type(s).__name__ for s in stages] == [
        "MAPStage", "SVIStage", "HMCStage", "LAPSStage", "LAPSStage",
    ]


def test_map_svi_hmc_laps_certified_presets_untouched_by_default():
    build = get_pipeline_builder("map_svi_hmc_laps")
    stages = build(system=None)
    warm, cold = stages[3], stages[4]

    assert warm.init == "warm"
    assert warm.config == dict(
        init_mode="warm",
        num_chains=128,
        num_unadjusted_steps=300,
        num_adjusted_steps=248,
        early_stop=False,
        track_chains=True,
        p2_resample_at_chunk=None,
        p2_resample_min_survivors=32,
        p2_resample_mode="replace",
    )
    assert warm.requires == ("qz",)

    assert cold.init == "cold"
    assert cold.config == dict(
        init_mode="prior",
        num_chains=128,
        num_unadjusted_steps=300,
        num_adjusted_steps=248,
        early_stop=False,
        track_chains=True,
        p2_resample_at_chunk=13,
        p2_resample_min_survivors=24,
        p2_resample_mode="replace",
    )
    assert cold.requires == ()

    # keep=4 (the default) lands in extra_kwargs on BOTH arms, not self.config
    # (it is not a named LAPSStage preset field -- it's forwarded verbatim).
    assert warm.extra_kwargs == {"p2_keep_per_chain": 4}
    assert cold.extra_kwargs == {"p2_keep_per_chain": 4}


def test_laps_keep_override():
    build = get_pipeline_builder("map_svi_hmc_laps")
    stages = build(system=None, laps_keep=7)
    warm, cold = stages[3], stages[4]
    assert warm.extra_kwargs == {"p2_keep_per_chain": 7}
    assert cold.extra_kwargs == {"p2_keep_per_chain": 7}


def test_laps_num_chains_overrides_both_arms():
    build = get_pipeline_builder("map_svi_hmc_laps")
    stages = build(system=None, laps_num_chains=256)
    warm, cold = stages[3], stages[4]
    assert warm.config["num_chains"] == 256
    assert cold.config["num_chains"] == 256
    # Untouched certified fields still come from the presets.
    assert warm.config["num_adjusted_steps"] == 248
    assert cold.config["p2_resample_at_chunk"] == 13


def test_laps_seed_passes_through_to_effective_seed():
    build = get_pipeline_builder("map_svi_hmc_laps")
    stages = build(system=None, laps_seed=99)
    map_stage, svi_stage, hmc_stage, warm, cold = stages
    # LAPS arms re-seeded independently of the pipeline-wide seed...
    assert warm.effective_seed(pipeline_seed=0) == 99
    assert cold.effective_seed(pipeline_seed=0) == 99
    # ...while the front end still falls back to the pipeline-wide seed (its
    # cache is untouched by laps_seed).
    assert map_stage.effective_seed(pipeline_seed=0) == 0
    assert svi_stage.effective_seed(pipeline_seed=0) == 0
    assert hmc_stage.effective_seed(pipeline_seed=0) == 0


def test_laps_seed_absent_falls_back_to_pipeline_seed():
    build = get_pipeline_builder("map_svi_hmc_laps")
    stages = build(system=None)
    warm, cold = stages[3], stages[4]
    assert warm.effective_seed(pipeline_seed=3) == 3
    assert cold.effective_seed(pipeline_seed=3) == 3


def test_laps_cold_min_survivors_forwarded_to_cold_arm_only():
    build = get_pipeline_builder("map_svi_hmc_laps")
    stages = build(system=None, laps_cold_min_survivors=40)
    warm, cold = stages[3], stages[4]
    assert cold.config["p2_resample_min_survivors"] == 40
    # Warm arm's resample_min_survivors (inert -- resampler is off) is untouched.
    assert warm.config["p2_resample_min_survivors"] == 32


def test_map_svi_hmc_laps_reuses_map_svi_hmc_construction():
    """The MAP/SVI/HMC stages must be byte-identical (same class, same
    config_hash_data) to the standalone ``map_svi_hmc`` builder's output, for
    the same kwargs -- i.e. this builder truly reuses it rather than
    duplicating the construction."""
    from gigalens_research.simtests.pipelines import build_map_svi_hmc

    kwargs = dict(map_num_steps=10, svi_num_steps=20, hmc_n_hmc=8,
                  hmc_num_results=15, hmc_num_burnin=5)
    plain = build_map_svi_hmc(system=None, **kwargs)
    combo = get_pipeline_builder("map_svi_hmc_laps")(system=None, **kwargs)

    assert [type(s).__name__ for s in combo[:3]] == [type(s).__name__ for s in plain]
    for a, b in zip(plain, combo[:3]):
        assert a.config_hash_data() == b.config_hash_data()


# --------------------------------------------------------------------------- #
# 2 -- gl2_existing "system_ids" subset selection (pure logic only)          #
# --------------------------------------------------------------------------- #


def test_resolve_selected_indices_default_is_full_range():
    from gigalens_research.simtests.experiments.gl2_sersic import _resolve_selected_indices
    assert _resolve_selected_indices(None, 10) == list(range(10))


def test_resolve_selected_indices_subset():
    from gigalens_research.simtests.experiments.gl2_sersic import _resolve_selected_indices
    assert _resolve_selected_indices([0, 1, 9], 100) == [0, 1, 9]
    # Non-int-friendly input (e.g. numpy ints / floats from YAML) coerces to int.
    assert _resolve_selected_indices([0.0, 3], 10) == [0, 3]


def test_resolve_selected_indices_out_of_range_raises():
    from gigalens_research.simtests.experiments.gl2_sersic import _resolve_selected_indices
    with pytest.raises(ValueError):
        _resolve_selected_indices([0, 100], 100)
    with pytest.raises(ValueError):
        _resolve_selected_indices([-1], 10)


def test_manifest_filtering_contains_only_selected_ids():
    """Manifest.system_ids reflects ONLY the selected subset (pure I/O, no npz)."""
    from gigalens_research.simtests.experiments.gl2_sersic import _resolve_selected_indices
    from gigalens_research.simtests.system import write_manifest, load_manifest

    n_systems_total = 12
    n_digits = len(str(n_systems_total - 1))  # width from the FULL dataset
    selected = _resolve_selected_indices([2, 5, 7], n_systems_total)
    system_ids = [f"sys_{i:0{n_digits}d}" for i in selected]
    assert system_ids == ["sys_02", "sys_05", "sys_07"]

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = os.path.join(tmpdir, "dataset")
        write_manifest(
            dataset_dir, generator="gl2_existing", seed=0,
            system_ids=system_ids, dataset_hash="abc123",
            extra={"system_ids_filter": selected},
        )
        m = load_manifest(dataset_dir)
        assert m["n_systems"] == 3
        assert m["system_ids"] == ["sys_02", "sys_05", "sys_07"]
        assert m["extra"]["system_ids_filter"] == [2, 5, 7]


def test_enumerate_runs_sees_only_the_subset():
    """enumerate_runs must see exactly the 10 selected systems (not the full 100)."""
    from gigalens_research.simtests.config import CampaignSpec
    from gigalens_research.simtests.run import enumerate_runs
    from gigalens_research.simtests.system import write_manifest

    n_digits = 3  # matches a 100-system full dataset's width
    selected = list(range(10))
    system_ids = [f"sys_{i:0{n_digits}d}" for i in selected]

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = os.path.join(tmpdir, "dataset")
        write_manifest(dataset_dir, generator="gl2_existing", seed=0,
                       system_ids=system_ids, dataset_hash="abc",
                       extra={"system_ids_filter": selected})
        spec = CampaignSpec.from_dict({
            "name": "laps_transfer_test",
            "seed": 0,
            "dataset": {"generator": "gl2_existing"},
            "inference": {"builder": "epl_shear_sersic_sersic",
                          "pipeline": "map_svi_hmc_laps"},
            "sweep": [{}],
        })
        runs = enumerate_runs(spec, dataset_dir)
        assert len(runs) == 10
        assert [sid for sid, _ in runs] == [f"sys_{i:03d}" for i in range(10)]


# --------------------------------------------------------------------------- #
# 3 -- experiments/laps_transfer/campaign.yaml parses as expected             #
# --------------------------------------------------------------------------- #


def _campaign_yaml_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    # here = <root>/src/gigalens_research/simtests/tests -> <root>/experiments/laps_transfer
    repo_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
    candidate = os.path.join(repo_root, "experiments", "laps_transfer", "campaign.yaml")
    if os.path.exists(candidate):
        return candidate
    # Fallback: walk up from `here` looking for an experiments/laps_transfer dir
    # (robust to the package being installed/vendored at a different depth).
    d = here
    for _ in range(8):
        cand = os.path.join(d, "experiments", "laps_transfer", "campaign.yaml")
        if os.path.exists(cand):
            return cand
        d = os.path.dirname(d)
    return candidate


def test_laps_transfer_campaign_yaml_parses():
    from gigalens_research.simtests.config import CampaignSpec

    path = _campaign_yaml_path()
    assert os.path.exists(path), f"expected campaign yaml at {path}"
    spec = CampaignSpec.from_yaml(path)

    assert spec.name == "laps_transfer_v1"
    assert spec.seed == 0
    assert spec.output_dir == os.path.expanduser(
        "~/GIGALens-Code/simtests_results/laps_transfer_v1"
    )
    assert spec.dataset.generator == "gl2_existing"
    assert spec.dataset.extra["system_ids"] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    assert spec.dataset.extra["delta_pix"] == 0.065
    assert spec.dataset.extra["num_pix"] == 80
    assert spec.dataset.extra["supersample"] == 2
    assert spec.dataset.extra["noise_kind"] == "forward"
    assert spec.dataset.extra["background_rms"] == 0.2
    assert spec.dataset.extra["exp_time"] == 100

    assert spec.inference.builder == "epl_shear_sersic_sersic"
    assert spec.inference.pipeline == "map_svi_hmc_laps"
    pk = spec.inference.pipeline_kwargs
    assert pk["map_num_steps"] == 1000
    assert pk["map_n_samples"] == 2000
    assert pk["svi_num_steps"] == 5000
    assert pk["svi_n_vi"] == 1000
    assert pk["hmc_n_hmc"] == 64
    assert pk["hmc_num_results"] == 1500
    assert pk["hmc_num_burnin"] == 500
    assert pk["hmc_init_eps"] == 0.3
    assert pk["hmc_init_l"] == 3
    assert pk["laps_keep"] == 4

    assert spec.sweep_points == [{}]
    assert spec.execution.systems_per_task == 1
    assert "all_zscores" in spec.metrics
    assert "mass_zscores" in spec.metrics


def test_laps_transfer_campaign_yaml_pipeline_builds_five_stages():
    """End-to-end sanity: the campaign's (builder, pipeline_kwargs) actually
    build a 5-stage pipeline through the registries, exactly as run.py does."""
    from gigalens_research.simtests.config import CampaignSpec
    from gigalens_research.simtests.registry import get_pipeline_builder

    spec = CampaignSpec.from_yaml(_campaign_yaml_path())
    build = get_pipeline_builder(spec.inference.pipeline)
    stages = build(system=None, **spec.effective_pipeline_kwargs({}))
    assert [s.instance_name for s in stages] == [
        "map", "svi", "hmc", "laps_warm", "laps_cold",
    ]
    map_stage = stages[0]
    assert map_stage.num_steps == 1000
    assert map_stage.n_samples == 2000
    warm, cold = stages[3], stages[4]
    assert warm.extra_kwargs == {"p2_keep_per_chain": 4}
    assert cold.extra_kwargs == {"p2_keep_per_chain": 4}


# --------------------------------------------------------------------------- #
# 4 -- experiments/laps_transfer/analyze.py                                  #
# --------------------------------------------------------------------------- #


def _analyze_module():
    """Import ``experiments/laps_transfer/analyze.py`` (a standalone script,
    not part of the installed package) by adding its directory to sys.path."""
    campaign_yaml = _campaign_yaml_path()
    analyze_dir = os.path.dirname(campaign_yaml)
    if analyze_dir not in sys.path:
        sys.path.insert(0, analyze_dir)
    import analyze as _analyze_mod
    return _analyze_mod


def test_analyze_imports_and_has_help(capsys):
    """``python analyze.py --help`` must not require jax/gigalens at import
    time (heavy imports are deferred into main())."""
    analyze = _analyze_module()
    with pytest.raises(SystemExit) as exc:
        analyze._parse_args(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "campaign_yaml" in captured.out


def test_tier1_metrics_and_flags():
    analyze = _analyze_module()
    diag = {
        "p1_nan_frac": np.array([0.0, 0.1, 0.05]),
        "p2_settled_accept": np.array([0.5, 0.6, 0.62]),
        "p2_frozen": np.array([False, False, True, True, True]),
        "p1_step_size": np.array([0.1, 0.2]),
        "p2_step_size": np.array([0.05, 0.06]),
    }
    metadata = {
        "resample_info": {
            "chunk": 13, "skipped": False, "mode": "replace",
            "n_survivors": 40, "n_stragglers": 88, "cut": -12.3, "eps0_rs": 0.01,
        },
    }
    m = analyze._tier1_metrics(diag, metadata)
    assert m["max_p1_nan_frac"] == pytest.approx(0.1)
    assert m["final_p2_accept"] == pytest.approx(0.62)
    assert m["p2_frozen_count"] == 3
    assert m["stepsize_nonfinite"] is False
    assert m["resample_n_survivors"] == 40

    flags = analyze._tier1_flags(m, "laps_cold")
    # 40 >= 31 (the certified demo-lens minimum across seeds: 31/47/51) -> clean.
    assert flags == []

    # Non-finite step-size history -> FAIL, regardless of arm.
    bad_diag = dict(diag, p2_step_size=np.array([0.1, np.nan]))
    m_bad = analyze._tier1_metrics(bad_diag, metadata)
    assert m_bad["stepsize_nonfinite"] is True
    assert any("FAIL" in f and "non-finite" in f for f in analyze._tier1_flags(m_bad, "laps_warm"))

    # Skipped resample on cold -> FAIL.
    skipped_meta = {"resample_info": dict(metadata["resample_info"], skipped=True)}
    m_skip = analyze._tier1_metrics(diag, skipped_meta)
    assert any("FAIL" in f and "skipped" in f for f in analyze._tier1_flags(m_skip, "laps_cold"))

    # Below the certified demo-lens minimum (31) -> WARN (canary, not verdict;
    # an actual guard trip surfaces as resample_skipped -> FAIL above).
    low_meta = {"resample_info": dict(metadata["resample_info"], n_survivors=28)}
    m_low = analyze._tier1_metrics(diag, low_meta)
    low_flags = analyze._tier1_flags(m_low, "laps_cold")
    assert any("WARN" in f and "31" in f for f in low_flags)
    assert not any("FAIL" in f for f in low_flags)


def test_tier2_pooled_flags():
    analyze = _analyze_module()
    rng = np.random.default_rng(0)
    good_z = rng.normal(size=2000).tolist()  # a well-calibrated posterior
    pooled = analyze._tier2_pooled_flags({"hmc": good_z, "laps_warm": [], "laps_cold": [float("nan")]})
    assert 0.55 < pooled["hmc"]["frac_1sigma"] < 0.80
    assert pooled["hmc"]["frac_2sigma"] > 0.90
    assert pooled["hmc"]["flags"] == []
    assert pooled["laps_warm"]["frac_1sigma"] is None
    assert "WARN" in pooled["laps_warm"]["flags"][0]
    assert pooled["laps_cold"]["frac_1sigma"] is None  # all-NaN input


def test_tier3_metrics_and_flags_identical_arms_are_clean():
    """LAPS == HMC (same samples) -> zero offset, unit width ratio, full core
    fraction, and no bad-logp chains."""
    analyze = _analyze_module()
    dim = 3
    rng = np.random.default_rng(1)

    class _FakeProbModel:
        def log_prob(self, z):
            z = np.asarray(z)
            return (-0.5 * np.sum(z ** 2, axis=-1), None)

    class _FakeCtx:
        prob_model = _FakeProbModel()

    from gigalens_research.inference_utils.posterior import SamplerPosterior

    samples = rng.normal(size=(8, 50, dim))
    hmc_post = SamplerPosterior(_FakeCtx(), samples)
    laps_samples = samples[:, -4:, :]  # (chains, keep=4, dim), same distribution
    laps_post = SamplerPosterior(_FakeCtx(), laps_samples)

    t3 = analyze._tier3_metrics(_FakeCtx(), hmc_post, laps_post)
    assert t3["max_abs_offset"] < 1.0  # same-distribution samples: no real bias
    assert 0.5 < t3["min_width_ratio"] <= t3["max_width_ratio"] < 2.0
    assert t3["core_fraction"] == 1.0
    assert t3["core_fraction_box"] == 1.0
    assert t3["n_bad_logp_chains"] == 0
    assert set(t3["offset"].keys()) == {f"z{i}" for i in range(dim)}

    flags = analyze._tier3_flags(t3, "laps_cold")
    assert not any("FAIL" in f for f in flags)


def test_tier3_flags_catch_a_displaced_outlier_chain():
    analyze = _analyze_module()
    t3 = {
        "max_abs_offset": 0.05, "min_width_ratio": 0.95, "max_width_ratio": 1.05,
        "core_fraction": 0.9,  # one of ten chains left the core
        "core_fraction_box": 1.0,
        "n_bad_logp_chains": 2,
    }
    flags = analyze._tier3_flags(t3, "laps_cold")
    assert any("FAIL" in f and "core_fraction" in f for f in flags)
    assert any("FAIL" in f and "logp" in f for f in flags)  # laps_cold -> FAIL, not WARN


def test_tier3_box_core_catches_low_dim_lightswap_excursion():
    """A chain ~10 sigma off in a FEW params (light-swap-like) passes the L2
    rms core (diluted: rms ~ sqrt(4*100/22) ~ 4.3 < 6 at d=22) but must FAIL
    the per-param L-inf box core (grader DC-T1 finding)."""
    analyze = _analyze_module()
    dim = 22
    rng = np.random.default_rng(2)

    class _FakeProbModel:
        def log_prob(self, z):
            # Flat fake logp: keeps the logp gate out of this test's way so
            # it isolates the geometry (box vs rms) distinction.
            return (np.zeros(np.asarray(z).shape[0]), None)

    class _FakeCtx:
        prob_model = _FakeProbModel()

    from gigalens_research.inference_utils.posterior import SamplerPosterior

    hmc = rng.normal(size=(8, 200, dim))
    laps = rng.normal(size=(10, 4, dim)) * 0.9
    laps[0, :, :4] += 10.0  # one chain, 10 sigma off in 4 of 22 params
    t3 = analyze._tier3_metrics(
        _FakeCtx(), SamplerPosterior(_FakeCtx(), hmc), SamplerPosterior(_FakeCtx(), laps))
    assert t3["core_fraction"] == 1.0        # L2 rms is blind to this
    assert t3["core_fraction_box"] == 0.9    # L-inf box catches it
    assert t3["max_abs_chain_z"] > 6.0
    flags = analyze._tier3_flags(t3, "laps_cold")
    assert any("FAIL" in f and "box" in f for f in flags)


def test_write_csv_smoke(tmp_path):
    analyze = _analyze_module()
    rows = [
        {"system_id": "sys_000", "arm": "hmc", "status": "ok", "z_frac_1sigma": 0.7},
        {"system_id": "sys_000", "arm": "laps_warm", "status": "missing"},
    ]
    out_csv = str(tmp_path / "summary.csv")
    analyze._write_csv(out_csv, rows)
    with open(out_csv) as f:
        text = f.read()
    assert "system_id" in text.splitlines()[0]
    assert "sys_000" in text


def test_fabricated_run_dir_loads_via_posterior_and_diagnostics_from_disk(tmp_path):
    """Fabricate a tiny on-disk run dir (arrays.npz + manifest.json +
    diagnostics.npz) mimicking the real ``LAPSStage``/``HMCStage`` layout
    (see ``inference_utils/pipeline.py``'s ``_save_stage``), then smoke the
    loading path ``analyze_system`` uses: ``posterior_from_disk`` +
    ``diagnostics_from_disk`` feeding ``_tier1_metrics``/``_tier3_metrics``.
    """
    from gigalens_research.inference_utils.pipeline import (
        InferenceContext, posterior_from_disk, diagnostics_from_disk,
    )

    analyze = _analyze_module()
    dim = 3
    rng = np.random.default_rng(2)

    class _FakeProbModel:
        def log_prob(self, z):
            z = np.asarray(z)
            return (-0.5 * np.sum(z ** 2, axis=-1), None)

    class _FakeModelSeq:
        scene_model = None
        prob_model = _FakeProbModel()

    ctx = InferenceContext(
        phys_model=None, prob_model=_FakeModelSeq.prob_model,
        sim_config=None, model_seq=_FakeModelSeq(),
    )

    run_dir = str(tmp_path / "runs" / "sys_000" / "default")

    def _write_stage(name, cls_name, samples_z, metadata, diagnostics=None):
        stage_dir = os.path.join(run_dir, name)
        os.makedirs(stage_dir, exist_ok=True)
        np.savez(os.path.join(stage_dir, "arrays.npz"), samples_z=samples_z)
        if diagnostics:
            np.savez(os.path.join(stage_dir, "diagnostics.npz"), **diagnostics)
        manifest = {
            "stage": name, "class": cls_name, "schema_version": 1,
            "input_hash": "deadbeef", "metadata": metadata,
        }
        with open(os.path.join(stage_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f)

    hmc_samples = rng.normal(size=(6, 40, dim))
    _write_stage("hmc", "HMCStage", hmc_samples, {"num_chains": 6})

    laps_samples = hmc_samples[:, -4:, :]
    _write_stage(
        "laps_cold", "LAPSStage", laps_samples,
        metadata={
            "init": "cold", "init_mode": "prior",
            "resample_info": {"chunk": 13, "skipped": False, "mode": "replace",
                              "n_survivors": 50, "n_stragglers": 78, "cut": -10.0,
                              "eps0_rs": 0.02},
        },
        diagnostics={
            "p1_nan_frac": np.array([0.0, 0.02]),
            "p2_settled_accept": np.array([0.5, 0.55]),
            "p2_frozen": np.array([False, True, True]),
            "p1_step_size": np.array([0.1, 0.12]),
            "p2_step_size": np.array([0.05, 0.05]),
        },
    )

    hmc_post = posterior_from_disk(run_dir, "hmc", ctx)
    laps_post = posterior_from_disk(run_dir, "laps_cold", ctx)
    laps_diag = diagnostics_from_disk(run_dir, "laps_cold", ctx)
    with open(os.path.join(run_dir, "laps_cold", "manifest.json")) as f:
        laps_meta = json.load(f)["metadata"]

    t1 = analyze._tier1_metrics(dict(laps_diag.arrays), laps_meta)
    assert t1["resample_n_survivors"] == 50
    # 50 >= 31 (certified demo-lens minimum across seeds: 31/47/51) -> clean.
    flags = analyze._tier1_flags(t1, "laps_cold")
    assert flags == []

    t3 = analyze._tier3_metrics(ctx, hmc_post, laps_post)
    assert t3["core_fraction"] == 1.0
    assert t3["n_bad_logp_chains"] == 0
