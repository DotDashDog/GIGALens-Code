"""Tests for the sampler-benchmark campaign machinery (CPU-only).

Covers:
1. Config layer: the reserved ``pipeline`` sweep key, kwarg stripping,
   ``seed_mode`` validation, and per-system seed derivation.
2. The benchmark builder family: registered names, stage sequences, the
   16-chain defaults, the required (never-defaulted) burn-in kwargs, and the
   diagonal-qz bridge (scale + dtype handling).
3. Trunk/tail orchestration units: ``_split_trunk`` and ``_trunk_digest``
   (notably: a ``[MAP]`` trunk and a ``[MAP, SVI]`` trunk with the same MAP
   config must share one digest — that IS the same-MAP guarantee).
4. End-to-end trunk sharing on a tiny CPU-generated dataset: two pipeline
   variants of the same system share one trunk directory, the second variant
   loads MAP from cache, and per-system seeds differ across systems but not
   across variants.

Usage::

    python -m gigalens_research.simtests.tests.sampler_matrix_test

This test does NOT require a GPU; it forces CPU JAX devices.
"""
from __future__ import annotations

import json
import os
import tempfile

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np


# ---------------------------------------------------------------------------
# 1. Config layer
# ---------------------------------------------------------------------------


def _spec(sweep, seed_mode="campaign", seed=0):
    from gigalens_research.simtests.config import CampaignSpec
    return CampaignSpec.from_dict({
        "name": "t",
        "seed": seed,
        "seed_mode": seed_mode,
        "dataset": {"generator": "test"},
        "inference": {"builder": "b", "pipeline": "map_svi_hmc",
                      "pipeline_kwargs": {"map_num_steps": 11}},
        "sweep": sweep,
    })


def test_pipeline_sweep_key():
    spec = _spec([{"pipeline": "map_mclmc"}, {}])

    assert spec.pipeline_for({"pipeline": "map_mclmc"}) == "map_mclmc"
    assert spec.pipeline_for({}) == "map_svi_hmc"

    # Reserved key names the run dir directly; other keys append as before.
    assert spec.sweep_dir_name({"pipeline": "map_mclmc"}) == "map_mclmc"
    assert spec.sweep_dir_name({"pipeline": "map_mclmc", "n_max": 10}) == \
        "map_mclmc_n_max10"
    assert spec.sweep_dir_name({"n_max": 10}) == "n_max10"
    assert spec.sweep_dir_name({}) == "default"

    # The key is stripped from builder kwargs but the base kwargs survive.
    kw = spec.effective_pipeline_kwargs({"pipeline": "map_mclmc"})
    assert "pipeline" not in kw
    assert kw["map_num_steps"] == 11
    print("  pipeline sweep key: OK")


def test_seed_modes():
    from gigalens_research.simtests.config import CampaignSpec, derive_system_seed

    legacy = _spec([{}], seed_mode="campaign", seed=3)
    assert legacy.run_seed("sys_000") == 3
    assert legacy.run_seed("sys_042") == 3

    per_sys = _spec([{}], seed_mode="per_system", seed=3)
    s0 = per_sys.run_seed("sys_000")
    s1 = per_sys.run_seed("sys_001")
    assert s0 != s1
    # Deterministic and stable across processes (sha256-derived, not hash()).
    assert s0 == derive_system_seed(3, "sys_000")
    assert derive_system_seed(3, "sys_000") == derive_system_seed(3, "sys_000")
    assert derive_system_seed(4, "sys_000") != s0

    try:
        CampaignSpec.from_dict({
            "name": "t", "seed": 0, "seed_mode": "bogus",
            "dataset": {"generator": "g"},
            "inference": {"builder": "b", "pipeline": "p"},
        })
    except ValueError:
        pass
    else:
        raise AssertionError("seed_mode='bogus' should raise ValueError")
    print("  seed modes: OK")


# ---------------------------------------------------------------------------
# 2. Benchmark builder family
# ---------------------------------------------------------------------------

_BURNIN_KW = dict(hmc_num_burnin=5, nuts_num_burnin=5,
                  mclmc_num_burnin=5, mams_num_burnin=5)


def test_family_registered():
    import gigalens_research.simtests.pipelines  # noqa: F401  (registers)
    from gigalens_research.simtests.registry import list_registered

    reg = set(list_registered()["pipeline_builders"])
    for sampler in ("hmc", "nuts", "mclmc", "mams"):
        assert f"map_svi_{sampler}" in reg, f"map_svi_{sampler} missing"
        assert f"map_{sampler}" in reg, f"map_{sampler} missing"
    assert "map_bootstrap_mclmc" in reg  # untouched legacy builder
    print("  family registered: OK")


def test_family_stage_sequences():
    from gigalens_research.inference_utils.pipeline import (
        BridgeStage, HMCStage, MAMSStage, MAPStage, MCLMCStage, NUTSStage,
        SVIStage,
    )
    from gigalens_research.simtests.registry import get_pipeline_builder

    terminal = {"hmc": HMCStage, "nuts": NUTSStage,
                "mclmc": MCLMCStage, "mams": MAMSStage}
    for sampler, cls in terminal.items():
        full = get_pipeline_builder(f"map_svi_{sampler}")(None, **_BURNIN_KW)
        assert [type(s) for s in full] == [MAPStage, SVIStage, cls], sampler

        short = get_pipeline_builder(f"map_{sampler}")(None, **_BURNIN_KW)
        assert [type(s) for s in short] == [MAPStage, BridgeStage, cls], sampler
        assert short[1].produces == ("qz",)

        # Fixed benchmark budget: 16 chains everywhere by default.
        chains = getattr(full[-1], "n_hmc", None) or getattr(full[-1], "n_chains")
        assert chains == 16, f"{sampler}: default chains {chains} != 16"
    print("  family stage sequences: OK")


def test_burnin_required():
    from gigalens_research.simtests.registry import get_pipeline_builder

    for name, missing in [
        ("map_svi_hmc", "hmc_num_burnin"),
        ("map_svi_nuts", "nuts_num_burnin"),
        ("map_svi_mclmc", "mclmc_num_burnin"),
        ("map_svi_mams", "mams_num_burnin"),
        ("map_nuts", "nuts_num_burnin"),
        ("map_mclmc", "mclmc_num_burnin"),
        ("map_mams", "mams_num_burnin"),
    ]:
        kw = {k: v for k, v in _BURNIN_KW.items() if k != missing}
        try:
            get_pipeline_builder(name)(None, **kw)
        except KeyError as exc:
            assert missing in str(exc), (name, exc)
        else:
            raise AssertionError(f"{name} without {missing} should raise KeyError")
    print("  burn-in required: OK")


def test_diag_qz_bridge():
    import jax.numpy as jnp
    from gigalens_research.simtests.pipelines import _diag_qz_bridge

    bridge = _diag_qz_bridge({"bridge_qz_scale": 0.5})
    assert "0.5" in bridge.version  # scale participates in the cache hash

    z = np.arange(4, dtype=np.float32)
    qz = bridge.fn(z)
    np.testing.assert_allclose(np.asarray(qz.mean()), z, rtol=1e-6)
    np.testing.assert_allclose(np.asarray(qz.stddev()), 0.5, rtol=1e-6)
    # dtype of scale follows loc (x64 trap)
    assert qz.stddev().dtype == jnp.asarray(z).dtype

    default = _diag_qz_bridge({})
    np.testing.assert_allclose(np.asarray(default.fn(z).stddev()), 1e-2, rtol=1e-6)
    print("  diag qz bridge: OK")


# ---------------------------------------------------------------------------
# 3. Trunk/tail units
# ---------------------------------------------------------------------------


def test_split_trunk():
    from gigalens_research.inference_utils.pipeline import (
        BridgeStage, MAPStage, MCLMCStage, SVIStage,
    )
    from gigalens_research.simtests.registry import get_pipeline_builder
    from gigalens_research.simtests.run import _split_trunk

    full = get_pipeline_builder("map_svi_mclmc")(None, **_BURNIN_KW)
    trunk, tail = _split_trunk(full)
    assert [type(s) for s in trunk] == [MAPStage, SVIStage]
    assert [type(s) for s in tail] == [MCLMCStage]

    short = get_pipeline_builder("map_mclmc")(None, **_BURNIN_KW)
    trunk, tail = _split_trunk(short)
    assert [type(s) for s in trunk] == [MAPStage]
    assert [type(s) for s in tail] == [BridgeStage, MCLMCStage]

    # A pipeline led by a custom stage has an empty trunk (legacy behavior).
    import gigalens_research.simtests.pipelines as p
    trunk, tail = _split_trunk(p.build_map_bootstrap_mclmc(_FakeSystem()))
    assert trunk == []
    assert len(tail) == 2
    print("  split trunk: OK")


class _FakeSystem:
    system_id = "sys_fake"
    truth_x = None


def test_trunk_digest():
    from gigalens_research.simtests.registry import get_pipeline_builder
    from gigalens_research.simtests.run import _split_trunk, _trunk_digest

    full_trunk, _ = _split_trunk(
        get_pipeline_builder("map_svi_mclmc")(None, **_BURNIN_KW))
    short_trunk, _ = _split_trunk(
        get_pipeline_builder("map_mclmc")(None, **_BURNIN_KW))

    # THE invariant: [MAP] and [MAP, SVI] trunks with the same MAP config
    # share one digest (=> one directory => one MAP computation).
    assert _trunk_digest("ctx0", full_trunk, 1) == _trunk_digest("ctx0", short_trunk, 1)

    # Anything that changes the MAP inputs separates the digests.
    other_map, _ = _split_trunk(
        get_pipeline_builder("map_mclmc")(None, map_num_steps=7, **_BURNIN_KW))
    assert _trunk_digest("ctx0", other_map, 1) != _trunk_digest("ctx0", short_trunk, 1)
    assert _trunk_digest("ctx1", short_trunk, 1) != _trunk_digest("ctx0", short_trunk, 1)
    assert _trunk_digest("ctx0", short_trunk, 2) != _trunk_digest("ctx0", short_trunk, 1)
    print("  trunk digest: OK")


# ---------------------------------------------------------------------------
# 4. End-to-end trunk sharing (tiny CPU dataset, cheap bridge tails)
# ---------------------------------------------------------------------------


def _register_e2e_builders():
    """Two tiny pipeline variants sharing a MAP trunk; tails are pure bridges
    so the test exercises the ORCHESTRATION (trunk reuse, seeding, caching)
    without paying for a sampler."""
    from gigalens_research.inference_utils.pipeline import (
        BridgeStage, MAPStage, SVIStage,
    )
    from gigalens_research.simtests.pipelines import _diag_qz_bridge
    from gigalens_research.simtests.registry import (
        _PIPELINE_BUILDERS, register_pipeline_builder,
    )

    if "_t_map_svi" in _PIPELINE_BUILDERS:
        return

    def _tiny_map():
        return MAPStage(num_steps=2, n_samples=4, pbar_interval=0)

    def _tail():
        return BridgeStage(name="tailmark", version="v1", requires=("qz",),
                           produces=("tail_ok",),
                           fn=lambda qz: np.array([1.0]))

    @register_pipeline_builder("_t_map_svi")
    def _t_map_svi(system, **kw):
        return [_tiny_map(),
                SVIStage(num_steps=3, n_vi=8, pbar_interval=0),
                _tail()]

    @register_pipeline_builder("_t_map_bridge")
    def _t_map_bridge(system, **kw):
        return [_tiny_map(), _diag_qz_bridge(kw), _tail()]


def test_trunk_sharing_end_to_end():
    import gigalens_research.simtests.experiments  # registers generators
    from gigalens_research.simtests.config import CampaignSpec
    from gigalens_research.simtests.generate import generate_campaign
    from gigalens_research.simtests.run import run_campaign
    from gigalens_research.simtests.system import load_manifest

    _register_e2e_builders()

    spec = CampaignSpec.from_dict({
        "name": "trunkshare",
        "seed": 7,
        "seed_mode": "per_system",
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
        "inference": {"builder": "epl_shear_sersic_sersic",
                      "pipeline": "_t_map_svi"},
        "sweep": [{"pipeline": "_t_map_svi"}, {"pipeline": "_t_map_bridge"}],
        "metrics": [],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_dir = generate_campaign(spec, tmpdir)
        sids = load_manifest(dataset_dir)["system_ids"]
        run_campaign(spec, tmpdir, verbose=False)

        records = {}
        for sid in sids:
            # Exactly one shared trunk dir per system, holding map/ and svi/.
            trunk_root = os.path.join(tmpdir, "runs", sid, "trunk")
            digests = sorted(os.listdir(trunk_root))
            assert len(digests) == 1, (sid, digests)
            trunk_dir = os.path.join(trunk_root, digests[0])
            assert os.path.isdir(os.path.join(trunk_dir, "map"))
            assert os.path.isdir(os.path.join(trunk_dir, "svi"))

            for variant in ("_t_map_svi", "_t_map_bridge"):
                vdir = os.path.join(tmpdir, "runs", sid, variant)
                # Trunk stages must NOT be duplicated into variant dirs.
                assert not os.path.exists(os.path.join(vdir, "map")), variant
                with open(os.path.join(vdir, "run.json")) as f:
                    rec = json.load(f)
                assert rec["status"] == "ok", rec.get("error_traceback", rec)
                assert rec["trunk"]["trunk_dir"] == trunk_dir
                records[(sid, variant)] = rec

            # Variant 2's [MAP]-only trunk must have LOADED the cached MAP
            # (identical bytes), not recomputed it.
            map_status = records[(sid, "_t_map_bridge")]["trunk"]["stages"]["map"]["status"]
            assert map_status == "loaded", map_status

        # Per-system seeds: differ across systems, identical across variants.
        assert records[(sids[0], "_t_map_svi")]["seed"] == \
            records[(sids[0], "_t_map_bridge")]["seed"]
        assert records[(sids[0], "_t_map_svi")]["seed"] != \
            records[(sids[1], "_t_map_svi")]["seed"]

        # Resume: a second pass reruns nothing (bridge tails always re-run,
        # but MAP/SVI must load).
        run_campaign(spec, tmpdir, verbose=False, skip_existing=False)
        for sid in sids:
            trunk_root = os.path.join(tmpdir, "runs", sid, "trunk")
            digests = sorted(os.listdir(trunk_root))
            assert len(digests) == 1  # no churn, still one trunk
            with open(os.path.join(tmpdir, "runs", sid, "_t_map_svi",
                                   "run.json")) as f:
                rec = json.load(f)
            for stage, meta in rec["trunk"]["stages"].items():
                assert meta["status"] == "loaded", (stage, meta)

    print("  trunk sharing end-to-end: OK")


# ---------------------------------------------------------------------------


def main():
    print("[sampler_matrix_test]")
    test_pipeline_sweep_key()
    test_seed_modes()
    test_family_registered()
    test_family_stage_sequences()
    test_burnin_required()
    test_diag_qz_bridge()
    test_split_trunk()
    test_trunk_digest()
    test_trunk_sharing_end_to_end()
    print("[sampler_matrix_test] all OK")


if __name__ == "__main__":
    main()
