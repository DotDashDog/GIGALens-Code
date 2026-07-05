"""Per-system target for the T13' RE-SIMULATED sys60 data (observed = d' rendered at
supersample=16, same truth + same recovered noise realization). See checkpoint T13' in
docs/logs/why-hard-to-sample.md.

Same model as systems/sys60/system.py (== TestSersic60.ipynb cells 3-4: EPL(50)+Shear
mass, two non-lstsq SersicEllipse, forward mode, 22 sampled params) EXCEPT:

  * OBSERVED IMAGE is loaded from resim/sys60_ss16/observed_ss16.npz (key "observed").
    RAISES if that file is missing -- t13_resim.py MUST run first (it produces d' only
    after GATE A + GATE B pass). The original data/simulated_systems/*.npz is untouched.

  * qz is NOT hash-pinned to the old reference SVI Gaussian. The DATA CHANGED (d' != the
    original observed), so the SVI posterior is necessarily different per model arm; each
    arm fits its OWN SVI on d' and passes its persisted qz here explicitly. `load_target`
    therefore REQUIRES `qz_arrays` (path to that arm's svi/arrays.npz) and RAISES without
    it -- there is no correct default qz for the re-simulated data.

  * `load_target(supersample=None, qz_arrays=None)`: `supersample` overrides ONLY the
    SimulatorConfig factor (as in systems/sys60); the MODEL ARM chooses 2 or 4.

STRICT SEPARATION: this module reads d' from resim/ (relative to its own location) and
never writes anything. All new products live under resim/sys60_ss16/.
"""
from __future__ import annotations

import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# resim/sys60_ss16 lives two levels up from systems/sys60_ss16data/
_RESIM_NPZ = os.path.abspath(
    os.path.join(_HERE, "..", "..", "resim", "sys60_ss16", "observed_ss16.npz"))
_GIGALENS_SRC = "/global/homes/l/linusu/gigalens/src"
_PSF = os.path.join(_GIGALENS_SRC, "gigalens/assets/psf.npy")
_EXPECTED_DIM = 22


def _require(path: str, what: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[sys60_ss16] missing {what}: {path}")
    return path


def _recover_param_names(prob_model, dim):
    """C-8-safe param names: zero-probe the bijector and SORT the output dict's keys
    (identical to systems/sys60/system.py). Never positional names."""
    probe = np.zeros((1, dim))
    out = prob_model.bij.forward(list(probe.T))
    if not isinstance(out, dict) or not out or any(
            isinstance(v, (dict, list)) for v in out.values()):
        raise TypeError(
            f"[sys60_ss16] bij.forward did not return a flat scalar-key dict: {type(out)}")
    names = sorted(str(k) for k in out.keys())
    if len(names) != dim or len(set(names)) != len(names):
        raise ValueError(
            f"[sys60_ss16] recovered {len(names)} unique names != dim {dim}: {names}")
    return names


def build_modelling_sequence(supersample=None):
    """Build the scene-API ModellingSequence on the re-simulated data d' WITHOUT a qz
    (used by the arm's MAP->SVI pipeline stage, which PRODUCES the qz). Returns
    (model_seq, dim, param_names). supersample=None -> notebook value 2; an int overrides
    ONLY SimulatorConfig.supersample."""
    import jax
    jax.config.update("jax_enable_x64", True)   # sampling / likelihood need float64
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions

    from gigalens.simulator import SimulatorConfig
    from gigalens.jax.profiles.light import sersic
    from gigalens.jax.profiles.mass import epl, shear
    from gigalens.jax.scene import Component, Plane, LensModel
    from gigalens.jax.scene_prob_model import Dataset, ProbModel
    from gigalens.jax.inference import ModellingSequence

    # --- re-simulated observed image d' (RAISE if t13_resim not run) ---------
    _require(_RESIM_NPZ, "re-simulated d' (run t13_resim.py first)")
    f = np.load(_RESIM_NPZ)
    if "observed" not in f.files:
        raise KeyError(f"[sys60_ss16] 'observed' not in {_RESIM_NPZ}; keys={f.files}")
    observed_img = jnp.asarray(np.asarray(f["observed"], dtype=np.float64))

    # --- model (MODELING prior; identical to systems/sys60/system.py) --------
    epl0 = Component(epl.EPL(50), dict(
        theta_E=tfd.LogNormal(jnp.log(1.25), 0.4),
        gamma=tfd.TruncatedNormal(2, 0.5, 1, 3),
        e1=tfd.Normal(0, 0.2),
        e2=tfd.Normal(0, 0.2),
        center_x=tfd.Normal(0, 0.06),
        center_y=tfd.Normal(0, 0.06),
    ))
    shear0 = Component(shear.Shear(), dict(
        gamma1=tfd.Normal(0, 0.1), gamma2=tfd.Normal(0, 0.1)))
    lens_light = Component(sersic.SersicEllipse(use_lstsq=False), dict(
        R_sersic=tfd.LogNormal(jnp.log(1.6), 0.25),
        n_sersic=tfd.Uniform(0.5, 8),
        e1=tfd.TruncatedNormal(0, 0.1, -0.15, 0.15),
        e2=tfd.TruncatedNormal(0, 0.1, -0.15, 0.15),
        center_x=tfd.Normal(0, 0.02),
        center_y=tfd.Normal(0, 0.02),
        Ie=tfd.LogNormal(jnp.log(300.0), 0.5),
    ))
    source_light = Component(sersic.SersicEllipse(use_lstsq=False), dict(
        R_sersic=tfd.LogNormal(jnp.log(0.25), 0.25),
        n_sersic=tfd.Uniform(0.5, 8),
        e1=tfd.TruncatedNormal(0, 0.3, -0.5, 0.5),
        e2=tfd.TruncatedNormal(0, 0.3, -0.5, 0.5),
        center_x=tfd.Normal(0, 0.5),
        center_y=tfd.Normal(0, 0.5),
        Ie=tfd.LogNormal(jnp.log(150.0), 0.9),
    ))

    model = LensModel([
        Plane(mass=[epl0, shear0], light=[lens_light]),
        Plane(light=[source_light]),
    ])

    kernel = np.load(_require(_PSF, "PSF")).astype(np.float64)
    _ss = 2 if supersample is None else int(supersample)
    sim_config = SimulatorConfig(delta_pix=0.065, num_pix=80, supersample=_ss, kernel=kernel)
    ds = Dataset(observed_img, sim_config,
                 sees=[lens_light, source_light],
                 background_rms=0.2, exp_time=100)
    prob_model = ProbModel(model, ds, mode="forward")
    model_seq = ModellingSequence(prob_model)

    param_names = _recover_param_names(prob_model, _EXPECTED_DIM)  # validates dim==22
    return model_seq, _EXPECTED_DIM, param_names


def load_target(supersample=None, qz_arrays=None):
    """Return (model_seq, qz, z_center, dim, param_names) for the re-simulated sys60.

    supersample : None -> 2 (notebook value); int overrides SimulatorConfig.supersample.
    qz_arrays   : REQUIRED path to THIS arm's SVI stage arrays.npz (keys qz_loc,
                  qz_scale_tril). RAISES if not provided -- the re-simulated data has no
                  correct default qz (data changed; not hash-pinned to any old reference).
    """
    if qz_arrays is None:
        raise ValueError(
            "[sys60_ss16] load_target requires qz_arrays=<arm svi/arrays.npz>: the "
            "re-simulated data d' has a DIFFERENT posterior per model arm, so each arm "
            "must pass its OWN SVI qz. (Pipeline/MAP->SVI uses build_modelling_sequence, "
            "which needs no qz.)")
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    tfd = tfp.distributions

    model_seq, dim, param_names = build_modelling_sequence(supersample=supersample)

    a = np.load(_require(qz_arrays, "arm SVI arrays.npz"))
    for k in ("qz_loc", "qz_scale_tril"):
        if k not in a.files:
            raise KeyError(f"[sys60_ss16] '{k}' not in {qz_arrays}; keys={a.files}")
    qz_loc = np.asarray(a["qz_loc"], dtype=np.float64)
    qz_scale_tril = np.asarray(a["qz_scale_tril"], dtype=np.float64)
    if qz_loc.shape != (_EXPECTED_DIM,):
        raise ValueError(f"[sys60_ss16] qz_loc shape {qz_loc.shape} != ({_EXPECTED_DIM},)")
    if qz_scale_tril.shape != (_EXPECTED_DIM, _EXPECTED_DIM):
        raise ValueError(f"[sys60_ss16] qz_scale_tril shape {qz_scale_tril.shape} != "
                         f"({_EXPECTED_DIM},{_EXPECTED_DIM})")
    qz = tfd.MultivariateNormalTriL(
        loc=jnp.asarray(qz_loc), scale_tril=jnp.asarray(qz_scale_tril))
    z_center = np.asarray(qz_loc)
    return model_seq, qz, z_center, dim, param_names
