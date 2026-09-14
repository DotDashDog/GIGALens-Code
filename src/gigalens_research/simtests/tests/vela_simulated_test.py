"""Tests for the v2 ``vela_simulated`` generator: prior factory, PSF, noise,
calibration / cut-off helpers, and a CPU end-to-end run on a synthetic source.

Run::

    python -m pytest src/gigalens_research/simtests/tests/vela_simulated_test.py -q
"""
from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from gigalens_research.simtests.experiments import vela_simulated as vs


# ---------------------------------------------------------------------------
# Truth prior spec + factory
# ---------------------------------------------------------------------------

def test_prior_override_merges_and_records():
    spec = vs.resolve_truth_prior_spec(
        {"lens_mass": {"0": {"theta_E": {"dist": "LogNormal", "median": 1.1, "sigma": 0.2}}}})
    assert spec["lens_mass"]["0"]["theta_E"] == {"dist": "LogNormal", "median": 1.1, "sigma": 0.2}
    # untouched leaves keep the baseline
    assert spec["lens_mass"]["0"]["gamma"] == vs.TRUTH_PRIOR_BASELINE["lens_mass"]["0"]["gamma"]
    assert vs.TRUTH_PRIOR_BASELINE["lens_mass"]["0"]["theta_E"]["median"] == 1.25  # not mutated


@pytest.mark.parametrize("bad", [
    {"lens_mas": {}},
    {"lens_mass": {"7": {}}},
    {"lens_mass": {"0": {"theta_e": {"dist": "Normal", "loc": 0, "scale": 1}}}},
    {"lens_mass": {"0": {"theta_E": {"dist": "Gamma", "a": 1}}}},
    {"lens_mass": {"0": {"theta_E": {"dist": "Normal", "loc": 0}}}},
    {"lens_mass": {"0": {"theta_E": {"dist": "Normal", "loc": 0, "scale": 1, "extra": 2}}}},
])
def test_prior_override_typo_guard(bad):
    with pytest.raises(ValueError):
        vs.resolve_truth_prior_spec(bad)


def test_prior_samples_and_legacy_conversion():
    import jax
    jax.config.update("jax_enable_x64", True)
    spec = vs.resolve_truth_prior_spec(
        {"lens_light": {"0": {"n_sersic": {"dist": "Fixed", "value": 4.0}}}})
    prior = vs.build_truth_prior(spec)
    sample = prior.sample(seed=jax.random.PRNGKey(3))
    legacy = vs._sample_to_legacy(sample)
    assert len(legacy) == 3 and len(legacy[0]) == 2 and len(legacy[1]) == 1 and len(legacy[2]) == 1
    assert set(legacy[0][0]) == {"theta_E", "gamma", "e1", "e2", "center_x", "center_y"}
    assert set(legacy[0][1]) == {"gamma1", "gamma2"}
    assert legacy[1][0]["n_sersic"] == 4.0
    assert legacy[2][0]["amp"] == 1.0
    assert 1.0 <= legacy[0][0]["gamma"] <= 3.0
    assert all(isinstance(v, float) for grp in legacy for d in grp for v in d.values())


# ---------------------------------------------------------------------------
# Noise
# ---------------------------------------------------------------------------

def test_derive_background_rms_hand_check():
    d = vs.derive_background_rms(sky_mag_arcsec2=22.3, zeropoint_ab=25.94, exp_time=2400,
                                 n_reads=4, read_noise_e=4.5, dark_e_per_s=0.0075,
                                 native_pixel_arcsec=0.05, delta_pix=0.03)
    sky = 10 ** (-0.4 * (22.3 - 25.94)) * 0.05 ** 2
    sig = np.sqrt(sky * 2400 + 0.0075 * 2400 + 4 * 4.5 ** 2) / 2400
    assert np.isclose(d["sky_cps_per_native_pixel"], sky)
    assert np.isclose(d["sigma_cps_per_native_pixel"], sig)
    assert np.isclose(d["background_rms"], sig * 0.6)
    assert 0.003 < d["background_rms"] < 0.006


def test_resolve_noise_legacy_and_instrument():
    bkg, t, meta = vs.resolve_noise({"background_rms": 0.005, "exp_time": 2000.0}, 0.03, 1.52e-7)
    assert (bkg, t, meta["kind"]) == (0.005, 2000.0, "explicit")
    bkg, t, meta = vs.resolve_noise({"noise": {
        "kind": "instrument", "sky_mag_arcsec2": 22.3, "exp_time": 2400, "n_reads": 4,
        "read_noise_e": 4.5, "dark_e_per_s": 0.0075, "native_pixel_arcsec": 0.05}}, 0.03, 1.52e-7)
    assert np.isclose(meta["zeropoint_ab"], -2.5 * np.log10(1.52e-7 / 3631), atol=1e-9)
    assert t == 2400 and 0.003 < bkg < 0.006
    with pytest.raises(ValueError):
        vs.resolve_noise({}, 0.03, 1.52e-7)
    with pytest.raises(ValueError):
        vs.resolve_noise({"noise": {"kind": "instrument", "sky_mag_arcsec2": 22.3}}, 0.03, 1.52e-7)


# ---------------------------------------------------------------------------
# PSF
# ---------------------------------------------------------------------------

def test_gaussian_psf_normalised_and_fwhm_measured():
    k = vs._gaussian_psf(0.13, 0.03)
    assert k.shape[0] % 2 == 1 and np.isclose(k.sum(), 1.0)
    assert abs(vs.measure_fwhm(k, 0.03) - 0.13) < 0.01


def test_resample_oversampled_psf_preserves_fwhm():
    # synthetic 4x-oversampled Gaussian "ePSF" at 0.05" native, FWHM 0.09"
    spacing = 0.05 / 4
    n = 101
    c = (n - 1) / 2
    yy, xx = np.indices((n, n))
    sig = 0.09 / 2.3548 / spacing
    epsf = np.exp(-((yy - c) ** 2 + (xx - c) ** 2) / (2 * sig ** 2))
    k = vs.resample_oversampled_psf(epsf, spacing, 0.03, 33, 6)
    assert k.shape == (33, 33) and np.isclose(k.sum(), 1.0)
    # pixel-averaging a Gaussian over 0.03" pixels broadens it by < 3%
    assert abs(vs.measure_fwhm(k, 0.03) - 0.09) / 0.09 < 0.05
    assert np.argmax(k) == (33 * 33) // 2
    with pytest.raises(ValueError):
        vs.resample_oversampled_psf(epsf, spacing, 0.03, 32, 6)


def test_build_psf_requires_kind_and_keys():
    with pytest.raises(ValueError):
        vs.build_psf({}, 0.03, "f814w", "/nonexistent")
    with pytest.raises(ValueError):
        vs.build_psf({"kind": "gaussian"}, 0.03, "f814w", "/nonexistent")
    k, meta = vs.build_psf({"kind": "gaussian", "fwhm_arcsec": 0.1, "size_pix": 25}, 0.03, "f814w", "/x")
    assert k.shape == (25, 25) and meta["kind"] == "gaussian" and abs(meta["fwhm_arcsec_measured"] - 0.1) < 0.01


# ---------------------------------------------------------------------------
# Calibration + cut-off helpers
# ---------------------------------------------------------------------------

def test_flux_outside_and_border():
    canvas = np.zeros((40, 40))
    canvas[10:30, 10:30] = 1.0      # everything inside the centred 20x20 cutout
    assert vs.flux_outside_cutout(canvas, 20) == 0.0
    canvas[0, 0] = 400.0 / 3        # 1/4 of the flux outside
    assert np.isclose(vs.flux_outside_cutout(canvas, 20), 0.25)
    img = np.zeros((10, 10))
    img[5, 5] = 9.0
    img[0, 7] = 2.0
    assert vs.border_max(img, 3) == 2.0
    with pytest.raises(ValueError):
        vs.flux_outside_cutout(canvas, 21)


def test_calibrate_amp():
    assert np.isclose(vs.calibrate_amp(ratio=0.5, lens_flux=1000.0, source_flux_unit_amp=50.0), 10.0)
    with pytest.raises(ValueError):
        vs.calibrate_amp(ratio=0.5, lens_flux=0.0, source_flux_unit_amp=1.0)


def test_stdpsf_filename_by_detector():
    assert vs._stdpsf_filename("ACSWFC", "f814w", "SM4") == "STDPSF_ACSWFC_F814W_SM4.fits"
    assert vs._stdpsf_filename("WFC3IR", "f140w", None) == "STDPSF_WFC3IR_F140W.fits"
    with pytest.raises(ValueError):  # ACS library is split by era
        vs._stdpsf_filename("ACSWFC", "f814w", None)
    with pytest.raises(ValueError):  # WFC3 libraries are not
        vs._stdpsf_filename("WFC3IR", "f140w", "SM4")
    assert vs._stdpsf_detector("f140w") == "WFC3IR" and vs._stdpsf_detector("f814w") == "ACSWFC"


def test_calibration_new_modes_resolve_and_hand_checks():
    c = vs._resolve_calibration({"calibration": {"unlensed_ab_mag": 24.0}})
    assert c["mode"] == "unlensed_ab_mag" and c["unlensed_ab_mag"] == {"dist": "Fixed", "value": 24.0}
    c = vs._resolve_calibration({"calibration": {"unlensed_ab_mag": {"dist": "Normal", "loc": 24.0, "scale": 0.5}}})
    assert c["unlensed_ab_mag"]["dist"] == "Normal"
    with pytest.raises(ValueError):  # dist typo guard applies here too
        vs._resolve_calibration({"calibration": {"unlensed_ab_mag": {"dist": "Nrmal", "loc": 24.0, "scale": 0.5}}})
    with pytest.raises(ValueError):  # peak_sb needs n_brightest_pix
        vs._resolve_calibration({"calibration": {"peak_sb": {"sb_mag_arcsec2": 21.0}}})
    with pytest.raises(ValueError):  # exactly one mode
        vs._resolve_calibration({"calibration": {"peak_sb": {"sb_mag_arcsec2": 21.0, "n_brightest_pix": 7},
                                                 "unlensed_ab_mag": 24.0}})
    c = vs._resolve_calibration({"calibration": {"peak_sb": {"sb_mag_arcsec2": 21.0, "n_brightest_pix": 7}}})
    assert c["mode"] == "peak_sb" and c["peak_sb"]["n_brightest_pix"] == 7
    assert c["peak_sb"]["sb_mag_arcsec2"] == {"dist": "Fixed", "value": 21.0}

    zp = vs.ab_zeropoint_from_photfnu(9.52e-8)          # WFC3/IR F140W PHOTFNU
    assert abs(zp - 26.45) < 0.01
    amp = vs.calibrate_amp_unlensed_mag(ab_mag=24.0, zeropoint_ab=26.45, unlensed_flux_unit_amp=2.0)
    assert np.isclose(amp, 10 ** (-0.4 * (24.0 - 26.45)) / 2.0)   # 9.55 cps / 2 cps
    img = np.zeros((10, 10))
    img.flat[:3] = [5.0, 3.0, 1.0]
    assert np.isclose(vs.peak_surface_brightness(img, 2, 0.5), 4.0 / 0.25)
    assert np.isclose(vs.peak_surface_brightness(img, 3, 1.0), 3.0)
    with pytest.raises(ValueError):
        vs.peak_surface_brightness(img, 0, 1.0)
    amp = vs.calibrate_amp_peak_sb(sb_mag_arcsec2=21.0, zeropoint_ab=26.45, peak_sb_unit_amp=100.0)
    assert np.isclose(amp, 10 ** (-0.4 * (21.0 - 26.45)) / 100.0)
    with pytest.raises(ValueError):
        vs.calibrate_amp_peak_sb(sb_mag_arcsec2=21.0, zeropoint_ab=26.45, peak_sb_unit_amp=0.0)


def test_calibration_and_cutoff_blocks_are_strict():
    with pytest.raises(ValueError):
        vs._resolve_calibration({})
    with pytest.raises(ValueError):
        vs._resolve_calibration({"calibration": {"source_to_lens_flux_ratio": 0.5, "source_flux_scale": 1.0}})
    c = vs._resolve_calibration({"source_flux_scale": 2.0})
    assert c == {"mode": "scale", "source_flux_scale": 2.0, "measure_in": "cutout"}
    with pytest.raises(ValueError):
        vs._resolve_cutoff({}, 4)
    assert vs._resolve_cutoff({"cutoff": None}, 4) is None
    with pytest.raises(ValueError):
        vs._resolve_cutoff({"cutoff": {"canvas_factor": 2}}, 4)


def test_preprocess_source_crop_and_recenter():
    n = 101
    img = np.zeros((n, n))
    img[70, 70] = 10.0              # main blob off-centre
    img[10, 10] = 1.0               # far companion
    out, info = vs.preprocess_source(img, 0.01, crop_radius_arcsec=0.2, recenter=True)
    assert np.isclose(info["crop_flux_removed_frac"], 1.0 / 11.0)
    assert out.sum() == 10.0 and out[50, 50] == 10.0
    assert info["recenter_shift_arcsec"] == [-0.2, -0.2]
    assert np.allclose(info["centroid_offset_arcsec_after"], 0.0)
    out2, info2 = vs.preprocess_source(img, 0.01, crop_radius_arcsec=None, recenter=False)
    assert np.array_equal(out2, img) and "crop_flux_removed_frac" not in info2


# ---------------------------------------------------------------------------
# End-to-end on a synthetic "pristine" source (CPU, no download)
# ---------------------------------------------------------------------------

class _Spec:
    def __init__(self, extra):
        self.extra = extra


def _write_synthetic_source(root, name):
    """A verified-marker source dir: Gaussian blob, 200 px at 0.02"/px (4" FOV)."""
    n = 200
    c = (n - 1) / 2
    yy, xx = np.indices((n, n))
    img = 5000.0 * np.exp(-((yy - c) ** 2 + (xx - c - 15) ** 2) / (2 * 8.0 ** 2))  # nJy/px
    sdir = os.path.join(root, name)
    os.makedirs(sdir)
    np.save(os.path.join(sdir, "source_image.npy"), img)
    json.dump({
        "source_builder": vs._SOURCE_BUILDER, "source_extname": "IMAGE_PRISTINE",
        "source_pixel_scale_arcsec": 0.02, "instrument_pixel_scale_arcsec": 0.05,
        "photfnu_Jy": 1.52e-7, "redshift": 1.5,
    }, open(os.path.join(sdir, "metadata.json"), "w"))


@pytest.mark.slow
def test_end_to_end_synthetic_source():
    from gigalens_research.simtests.system import System, load_manifest

    with tempfile.TemporaryDirectory() as tmp:
        src_root = os.path.join(tmp, "sources")
        _write_synthetic_source(src_root, "vela99_cam12_a0.400_f814w")
        extra = {
            "scale_factor": "a0.400", "vela_ids": ["99"], "n_reps": 2,
            "num_pix": 48, "supersample": 2, "source_root": src_root, "datadir": tmp,
            "delta_pix": 0.065,   # override the synthetic mock's 0.05" TPIX (drizzle scale)
            "source_crop_radius_arcsec": 1.0, "source_recenter": True,
            "psf": {"kind": "gaussian", "fwhm_arcsec": 0.1, "size_pix": 11},
            "noise": {"kind": "explicit", "background_rms": 0.005, "exp_time": 2000},
            "calibration": {"source_to_lens_flux_ratio": 0.5},
            "cutoff": {"canvas_factor": 2, "max_flux_outside": 0.05, "max_border_sb_sigma": 3.0,
                       "max_redraws": 20},
            "truth_prior": {"lens_mass": {"0": {"theta_E": {"dist": "LogNormal", "median": 0.6, "sigma": 0.1}}},
                            "lens_light": {"0": {"R_sersic": {"dist": "LogNormal", "median": 0.5, "sigma": 0.1}}}},
        }
        ds = os.path.join(tmp, "dataset")
        vs.generate_vela_simulated(_Spec(extra), ds, seed=1)

        man = load_manifest(ds)
        assert man["n_systems"] == 2
        ex = man["extra"]
        assert ex["generator_version"] == 3 and ex["psf"]["kind"] == "gaussian"
        assert ex["delta_pix"] == 0.065 and ex["delta_pix_source"] == "config"
        assert ex["mock_instrument_pixel_arcsec"] == 0.05
        assert ex["truth_prior"]["lens_mass"]["0"]["theta_E"]["median"] == 0.6
        assert ex["source_preprocessing"]["per_source"]["vela99"]["recenter"] is True
        for sid in man["system_ids"]:
            m = ex["per_system"][sid]
            assert abs(m["source_to_lens_ratio_cutout"] - 0.5) < 1e-6
            assert m["flux_outside_frac"] <= 0.05 and m["border_sb_sigma"] <= 3.0
            sysobj = System.load(ds, sid)
            assert sysobj.observed_image.shape == (48, 48) and sysobj.psf.shape == (11, 11)
            assert float(sysobj.delta_pix) == 0.065
            assert m["calibration_mode"] == "ratio" and m["calibration_target"] is None
            assert np.isfinite(m["lens_ab_mag_cutout"]) and np.isfinite(m["peak_sb_mag_arcsec2"])
            assert m["peak_sb_n_pix"] == 7
            # lensed-source and lens magnitudes must reproduce the 0.5 flux ratio
            assert np.isclose(m["source_ab_mag_lensed_cutout"] - m["lens_ab_mag_cutout"],
                              -2.5 * np.log10(0.5), atol=1e-6)
            assert float(sysobj.truth_x[2][0]["amp"]) == pytest.approx(m["amp"])
            gen = json.load(open(os.path.join(ds, "systems", sid, "generation.json")))
            assert gen["metrics"]["n_redraws"] == len(gen["rejections"])
            noiseless = np.load(os.path.join(ds, "systems", sid, "noiseless_image.npy"))
            lens_only = np.load(os.path.join(ds, "systems", sid, "lens_light_only.npy"))
            src = noiseless - lens_only
            assert np.isclose(src.sum() / lens_only.sum(), 0.5, atol=1e-3)


@pytest.mark.slow
def test_end_to_end_peak_sb_mode():
    """peak_sb calibration: the realised peak SB equals the sampled target exactly."""
    from gigalens_research.simtests.system import load_manifest

    with tempfile.TemporaryDirectory() as tmp:
        src_root = os.path.join(tmp, "sources")
        _write_synthetic_source(src_root, "vela99_cam12_a0.400_f814w")
        extra = {
            "scale_factor": "a0.400", "vela_ids": ["99"], "n_reps": 2,
            "num_pix": 48, "supersample": 2, "source_root": src_root, "datadir": tmp,
            "psf": {"kind": "gaussian", "fwhm_arcsec": 0.1, "size_pix": 11},
            "noise": {"kind": "explicit", "background_rms": 0.005, "exp_time": 2000},
            "calibration": {"peak_sb": {"sb_mag_arcsec2": {"dist": "Normal", "loc": 21.0, "scale": 0.5},
                                        "n_brightest_pix": 5}},
            "cutoff": None,
            "truth_prior": {"lens_mass": {"0": {"theta_E": {"dist": "LogNormal", "median": 0.6, "sigma": 0.1}}}},
        }
        ds = os.path.join(tmp, "dataset")
        vs.generate_vela_simulated(_Spec(extra), ds, seed=3)
        ex = load_manifest(ds)["extra"]
        assert ex["calibration"]["mode"] == "peak_sb"
        targets = []
        for sid, m in ex["per_system"].items():
            assert m["peak_sb_n_pix"] == 5
            assert abs(m["peak_sb_mag_arcsec2"] - m["calibration_target"]) < 1e-6
            zp = m["zeropoint_ab"]
            # unlensed magnitude is consistent with the stored unlensed flux
            assert np.isclose(m["source_ab_mag_unlensed"], zp - 2.5 * np.log10(m["source_flux_unlensed_cps"]))
            targets.append(m["calibration_target"])
        assert targets[0] != targets[1]   # sampled per system, not shared
