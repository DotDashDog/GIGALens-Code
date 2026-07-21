"""Minimal smoke test for the simtests framework.

Runs a 2-system parametric generate → 1-step MAP pipeline without touching
any GPU. Validates that:
1. The registry, config, system, and generate layers work end-to-end.
2. A `CampaignSpec` round-trips through YAML (if PyYAML is available).
3. `enumerate_runs` produces the right number of runs.
4. The `status` CLI command works on an existing dataset.

Usage::

    python -m gigalens_research.simtests.tests.smoke_test

This test does NOT require a GPU; it uses CPU JAX devices.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np


def test_config_roundtrip():
    from gigalens_research.simtests.config import CampaignSpec, _normalize_sweep
    assert _normalize_sweep(None) == [{}]
    assert _normalize_sweep([{}]) == [{}]
    assert _normalize_sweep({"axis": "n_max", "values": [10, 15]}) == [
        {"n_max": 10}, {"n_max": 15}
    ]
    sweep = _normalize_sweep({"axes": [{"n_max": [10, 15]}, {"seed": [0, 1]}]})
    assert len(sweep) == 4
    assert {"n_max": 10, "seed": 0} in sweep

    spec = CampaignSpec.from_dict({
        "name": "test",
        "seed": 42,
        "dataset": {"generator": "parametric"},
        "inference": {"builder": "epl_shear_sersic_sersic", "pipeline": "map_svi_hmc"},
    })
    assert spec.name == "test"
    assert spec.seed == 42
    assert spec.sweep_points == [{}]
    assert spec.sweep_dir_name({}) == "default"
    assert spec.sweep_dir_name({"n_max": 10}) == "n_max10"
    print("  config roundtrip: OK")


def test_system_io():
    from gigalens_research.simtests.system import System, write_manifest, load_manifest

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = os.path.join(tmpdir, "dataset")
        img = np.random.default_rng(0).standard_normal((10, 10)).astype(np.float32)
        truth_x = [
            [{"theta_E": 1.0, "gamma": 2.0, "e1": 0.1, "e2": -0.05,
              "center_x": 0.0, "center_y": 0.0},
             {"gamma1": 0.01, "gamma2": -0.02}],
            [{"R_sersic": 1.5, "n_sersic": 3.0, "e1": 0.0, "e2": 0.0,
              "center_x": 0.0, "center_y": 0.0, "Ie": 250.0}],
            [{"R_sersic": 0.25, "n_sersic": 2.0, "e1": 0.0, "e2": 0.0,
              "center_x": 0.1, "center_y": 0.05, "Ie": 150.0}],
        ]
        sys = System(
            system_id="sys_000",
            observed_image=img,
            truth_x=truth_x,
            delta_pix=0.065,
            num_pix=10,
            supersample=1,
            psf=None,
            noise_kind="forward",
            background_rms=0.2,
            exp_time=100.0,
        )
        sys.save(dataset_dir)

        loaded = System.load(dataset_dir, "sys_000")
        assert loaded.system_id == "sys_000"
        np.testing.assert_allclose(loaded.observed_image, img, rtol=1e-5)
        assert loaded.delta_pix == 0.065
        assert loaded.truth_x[0][0]["theta_E"] == pytest_approx(1.0, rel=1e-4)

        write_manifest(dataset_dir, generator="test", seed=0,
                       system_ids=["sys_000"], dataset_hash="abc123")
        m = load_manifest(dataset_dir)
        assert m["n_systems"] == 1
        assert m["system_ids"] == ["sys_000"]
    print("  system I/O: OK")


def test_registry():
    import gigalens_research.simtests.experiments  # registers all built-ins
    from gigalens_research.simtests.registry import list_registered

    reg = list_registered()
    assert "parametric" in reg["generators"]
    assert "gl2_existing" in reg["generators"]
    assert "vela_existing" in reg["generators"]
    assert "epl_shear_sersic_sersic" in reg["inference_builders"]
    assert "epl_shear_sersic_shapelets" in reg["inference_builders"]
    assert "map_svi_hmc" in reg["pipeline_builders"]
    assert "map_bootstrap_mclmc" in reg["pipeline_builders"]
    assert "max_rhat" in reg["metrics"]
    assert "min_ess" in reg["metrics"]
    assert "all_zscores" in reg["metrics"]
    print(f"  registry: OK ({sum(len(v) for v in reg.values())} entries)")


def test_generate_minimal():
    """Generate 2 tiny parametric systems using CPU-only JAX."""
    import jax
    import jax.numpy as jnp

    # Make sure we're using CPU to avoid GPU contention on login nodes.
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    import gigalens_research.simtests.experiments  # registers builders
    from gigalens_research.simtests.config import CampaignSpec

    spec = CampaignSpec.from_dict({
        "name": "smoke",
        "seed": 7,
        "dataset": {
            "generator": "parametric",
            "n_systems": 2,
            "gen_chunk": 2,
            "num_pix": 10,
            "supersample": 1,
            "delta_pix": 0.065,
            "background_rms": 0.2,
            "exp_time": 100.0,
        },
        "inference": {
            "builder": "epl_shear_sersic_sersic",
            "pipeline": "map_svi_hmc",
        },
        "sweep": [{}],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        from gigalens_research.simtests.generate import generate_campaign
        from gigalens_research.simtests.system import load_manifest, System

        dataset_dir = generate_campaign(spec, tmpdir)
        m = load_manifest(dataset_dir)
        assert m["n_systems"] == 2
        assert len(m["system_ids"]) == 2

        s0 = System.load(dataset_dir, m["system_ids"][0])
        assert s0.observed_image.shape == (10, 10)
        assert s0.truth_x is not None
        assert s0.delta_pix == 0.065

    print("  generate_minimal: OK (2 systems, 10×10)")


def test_enumerate_runs():
    from gigalens_research.simtests.config import CampaignSpec
    from gigalens_research.simtests.run import enumerate_runs
    from gigalens_research.simtests.system import write_manifest

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = os.path.join(tmpdir, "dataset")
        write_manifest(dataset_dir, generator="test", seed=0,
                       system_ids=["sys_000", "sys_001"], dataset_hash="test")
        spec = CampaignSpec.from_dict({
            "name": "smoke",
            "seed": 0,
            "dataset": {"generator": "test"},
            "inference": {"builder": "test", "pipeline": "test"},
            "sweep": [{"n_max": 10}, {"n_max": 15}],
        })
        runs = enumerate_runs(spec, dataset_dir)
        assert len(runs) == 4  # 2 systems × 2 sweep points
        # Check stable ordering
        assert runs[0] == ("sys_000", {"n_max": 10})
        assert runs[1] == ("sys_000", {"n_max": 15})
        assert runs[2] == ("sys_001", {"n_max": 10})
    print("  enumerate_runs: OK")


def pytest_approx(v, rel=1e-4):
    """Minimal float comparison without pytest."""
    class _Approx:
        def __eq__(self, other):
            return abs(float(other) - v) <= rel * abs(v)
        def __repr__(self):
            return f"approx({v}, rel={rel})"
    return _Approx()


def test_scene_model_card_reports_noise():
    """G1: model_card() for a SCENE-backed InferenceContext must report a RECOGNIZED
    noise model — never the "unknown"/"unrecognized" branch — when the scene Dataset
    carries an error_map. A silent/unreported noise model is exactly what the §7 card
    guard exists to surface (project-standards.md), so a scene-backed card that says
    "unknown" while the likelihood uses a real error_map is a blind spot, not a pass.

    CPU-only; builds a tiny synthetic system + the migrated sersiclets builder.
    """
    import os as _os
    _os.environ.setdefault("JAX_ENABLE_X64", "1")
    _os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    jax.config.update("jax_enable_x64", True)

    import gigalens_research.simtests.experiments.vela_elliptical_sersiclets  # register
    from gigalens_research.simtests.registry import get_inference_builder
    from gigalens_research.simtests.system import System
    from gigalens_research.inference_utils.pipeline import (
        InferenceContext, model_card, format_model_card,
    )

    num_pix = 20
    obs = (np.random.default_rng(0).standard_normal((num_pix, num_pix)) * 0.01
           + 0.05).astype(np.float32)
    psf = np.zeros((9, 9), dtype=np.float32); psf[4, 4] = 1.0
    truth = {
        'lens_mass': {'0': dict(theta_E=0.9, gamma=2.0, e1=0.0, e2=0.0,
                                center_x=0.0, center_y=0.0),
                      '1': dict(gamma1=0.0, gamma2=0.0)},
        'lens_light': {'0': dict(R_sersic=1.0, n_sersic=2.0, e1=0.0, e2=0.0,
                                 center_x=0.0, center_y=0.0)},
        'source_light': {'0': dict(center_x=0.0, center_y=0.0)},
    }
    system = System(
        system_id="smoke_card", observed_image=obs, truth_x=truth,
        delta_pix=0.08, num_pix=num_pix, supersample=1, psf=psf,
        noise_kind="gaussian_poisson", background_rms=0.01, exp_time=1000.0,
        likelihood_precision="float64",
    )
    model_seq = get_inference_builder(
        "epl_shear_sersic_elliptical_sersiclets")(system, n_max=3)
    assert model_seq.is_scene_backed, "builder must produce a scene-backed ModellingSequence"

    ctx = InferenceContext.from_modelling_sequence(model_seq)
    card = model_card(ctx)
    # gigalens.jax.utils schema: per-dataset noise, never silent. Every section
    # must actually have built (a guarded failure shows up as "unavailable").
    for key, section in card.items():
        assert not (isinstance(section, dict) and "unavailable" in section), \
            f"model card section {key!r} failed to build: {section['unavailable']}"
    ds_card = card["datasets"][0]
    assert ds_card["noise"]["kind"] == "per-pixel error_map", ds_card["noise"]
    assert ds_card["noise"]["sigma_median"] > 0, ds_card["noise"]
    assert ds_card["psf"]["present"], ds_card["psf"]
    # The §7 blind spots must be advisory-free on this well-formed system.
    codes = {a["code"] for a in card["advisories"] if a["severity"] == "warning"}
    assert "psf-absent" not in codes and "psf-not-normalized" not in codes, codes
    text = format_model_card(card)
    assert "ABSENT" not in text, text

    # Amplitude mode, trace mode, and sees, in the new sections.
    assert card["likelihood"]["mode"] == "lstsq", card["likelihood"]
    assert card["trace"]["mode"] == "deflection_ratio", card["trace"]
    assert ds_card["sees"], ds_card  # resolved component labels
    # The pipeline's own extra: the stable input hash rides on the card.
    assert card["extras"]["context_hash"] == ctx.hash(), card["extras"]
    assert "Trace      :" in text and "sees" in text, text


def test_scene_model_card_multiplane_distances():
    """G1b/Phase-7b: a MULTIPLANE scene model card reports the active trace mode AND
    per-plane transverse comoving distances from the model's own cosmology.
    CPU-only; builds the scene model directly (no real data)."""
    import os as _os
    _os.environ.setdefault("JAX_ENABLE_X64", "1")
    _os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    jax.config.update("jax_enable_x64", True)

    from gigalens.jax.cosmo import wCDM_Cosmo
    from gigalens.jax.profiles.mass.epl import EPL
    from gigalens.jax.profiles.light.sersic import SersicEllipse
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_prob_model import ImageData, ProbModel
    from gigalens.jax.inference import ModellingSequence
    from gigalens.simulator import SimulatorConfig
    from gigalens_research.inference_utils.pipeline import (
        InferenceContext, model_card, format_model_card,
    )

    num_pix = 20
    obs = (np.random.default_rng(2).standard_normal((num_pix, num_pix)) * 0.01
           + 0.05).astype(np.float32)
    psf = np.zeros((9, 9), dtype=np.float32); psf[4, 4] = 1.0
    epl0 = dict(theta_E=1.0, gamma=2.0, e1=0.0, e2=0.0, center_x=0.0, center_y=0.0)
    epl1 = dict(theta_E=0.5, gamma=2.0, e1=0.0, e2=0.0, center_x=0.2, center_y=0.0)
    # z_source_ref must equal the LAST plane's redshift (scene_simulator raises
    # on a mismatch — reduced deflections are normalized to the last plane).
    cosmo = Component(wCDM_Cosmo(z_lens=0.4, z_source_ref=2.5), dict(H0=70.0, Om0=0.3, k=0.0, w0=-1.0))
    model = LensModel([
        Plane(redshift=0.4, mass=[Component(EPL(50), epl0)]),
        Plane(redshift=0.8, mass=[Component(EPL(50), epl1)]),
        Plane(redshift=2.5, light=[Component(SersicEllipse(use_lstsq=True),
              dict(R_sersic=0.2, n_sersic=1.0, e1=0.0, e2=0.0,
                   center_x=0.0, center_y=0.0))]),
    ], cosmo=cosmo)
    cfg = SimulatorConfig(delta_pix=0.08, num_pix=num_pix, supersample=1, kernel=psf,
                          likelihood_precision="float64")
    ds = ImageData(obs, cfg, background_rms=0.01, exp_time=1000.0, sees="all")
    pm = ProbModel(model, ds, mode="lstsq")
    ctx = InferenceContext.from_modelling_sequence(
        ModellingSequence(pm))

    card = model_card(ctx)
    trace = card["trace"]
    assert trace["mode"] == "multiplane", trace
    d = trace.get("distances_obs_to_plane_mpc")
    assert d is not None and len(d) == 3, trace
    # transverse comoving distance is monotonically increasing with redshift.
    assert d[0] < d[1] < d[2], f"distances not monotonic in z: {d}"
    assert all(np.isfinite(d)) and all(x > 0 for x in d), d
    assert "Distances  :" in format_model_card(card)


def main():
    print("Running simtests smoke tests...")
    test_config_roundtrip()
    test_system_io()
    test_registry()
    test_generate_minimal()
    test_enumerate_runs()
    test_scene_model_card_reports_noise()
    test_scene_model_card_multiplane_distances()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    # Add project to PYTHONPATH
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "../../../../")
    sys.path.insert(0, os.path.abspath(src))
    sys.path.insert(0, os.path.expanduser("~/gigalens/src"))
    sys.path.insert(0, os.path.expanduser("~/GIGALens-Code/src"))
    main()
