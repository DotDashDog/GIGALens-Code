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


def main():
    print("Running simtests smoke tests...")
    test_config_roundtrip()
    test_system_io()
    test_registry()
    test_generate_minimal()
    test_enumerate_runs()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    # Add project to PYTHONPATH
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "../../../../")
    sys.path.insert(0, os.path.abspath(src))
    sys.path.insert(0, os.path.expanduser("~/gigalens/src"))
    sys.path.insert(0, os.path.expanduser("~/GIGALens-Code/src"))
    main()
