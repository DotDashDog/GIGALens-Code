"""A/B check for SimulatorConfig.conv_precision on the lstsq_simulate hot path.

Compares full float64 (conv_precision=None) vs float64-everywhere-but-conv-in-f32
(conv_precision="float32") on:
  - the reconstructed image and lstsq coefficients,
  - value AND grad of a float64 chi-squared reduction wrt the nonlinear params,
  - wall-clock time of value_and_grad.

Mirrors TestEllipticalShapelets: EPL+Shear lens, SersicEllipse lens light,
SersicShapelets(n_max=25, use_lstsq=True) source, 200x200 image, 25x25 PSF,
supersample=1.
"""
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from gigalens.model import PhysicalModel
from gigalens.simulator import SimulatorConfig
from gigalens.jax.simulator import LensSimulator
from gigalens.jax.profiles.mass import epl, shear
from gigalens.jax.profiles.light import sersic
from gigalens.jax.profiles.light.sersic_shapelets import SersicShapelets

N_MAX = 25
NUM_PIX = 200
DELTA_PIX = 0.05
KSIZE = 25
SEED = 0

rng = np.random.default_rng(SEED)

# --- PSF: a normalized Gaussian 25x25 kernel ---
ax = np.arange(KSIZE) - (KSIZE - 1) / 2
gx, gy = np.meshgrid(ax, ax)
psf = np.exp(-(gx ** 2 + gy ** 2) / (2 * 2.0 ** 2))
psf = (psf / psf.sum()).astype(np.float64)

# --- observed data + noise map ---
observed_image = jnp.asarray(rng.normal(0, 1, (NUM_PIX, NUM_PIX)), dtype=jnp.float64)
err_map = jnp.asarray(np.full((NUM_PIX, NUM_PIX), 0.1), dtype=jnp.float64)

phys_model = PhysicalModel(
    [epl.EPL(), shear.Shear()],
    [sersic.SersicEllipse(use_lstsq=True)],
    [SersicShapelets(n_max=N_MAX, use_lstsq=True)],
)


def build_sim(conv_precision):
    sim_config = SimulatorConfig(
        delta_pix=DELTA_PIX,
        num_pix=NUM_PIX,
        supersample=1,
        kernel=psf,
        likelihood_precision="float64",
        conv_precision=conv_precision,
    )
    return LensSimulator(phys_model, sim_config, bs=1)


# --- nonlinear params (lstsq solves the amplitudes) ---
def f64(v):
    return jnp.asarray(v, dtype=jnp.float64)


lens_params = [
    dict(theta_E=f64(1.25), gamma=f64(2.0), e1=f64(0.05), e2=f64(-0.03),
         center_x=f64(0.0), center_y=f64(0.0)),
    dict(gamma1=f64(0.02), gamma2=f64(-0.01)),
]
lens_light_params = [
    dict(R_sersic=f64(1.0), n_sersic=f64(4.0), e1=f64(0.05), e2=f64(0.02),
         center_x=f64(0.0), center_y=f64(0.0)),
]
source_light_params = [
    dict(R_sersic=f64(0.25), n_sersic=f64(1.5), e1=f64(0.1), e2=f64(-0.05),
         center_x=f64(0.05), center_y=f64(-0.02), beta=f64(0.15)),
]
params = (lens_params, lens_light_params, source_light_params)


def chi2(sim, p):
    recon, _ = sim.lstsq_simulate(p, observed_image, err_map)
    resid = (recon.astype(jnp.float64) - observed_image) / err_map
    return jnp.sum(resid ** 2)


# Differentiate wrt a representative scalar nonlinear param (source beta) and the
# lens theta_E, to exercise the conv VJP.
def loss_fn(sim):
    def loss(beta, theta_E):
        lp = [dict(lens_params[0], theta_E=theta_E), lens_params[1]]
        sp = [dict(source_light_params[0], beta=beta)]
        return chi2(sim, (lp, lens_light_params, sp))
    return loss


def run(conv_precision, label):
    sim = build_sim(conv_precision)
    recon, coeffs = sim.lstsq_simulate(params, observed_image, err_map)
    recon.block_until_ready()

    vg = jax.jit(jax.value_and_grad(loss_fn(sim), argnums=(0, 1)))
    val, grads = vg(f64(0.15), f64(1.25))
    val.block_until_ready()

    # timing (median of several calls)
    ts = []
    for _ in range(7):
        t0 = time.perf_counter()
        v, g = vg(f64(0.15), f64(1.25))
        v.block_until_ready()
        g[0].block_until_ready()
        ts.append(time.perf_counter() - t0)
    tmed = float(np.median(ts)) * 1e3

    print(f"[{label}] recon.dtype={recon.dtype} coeffs.dtype={coeffs.dtype}")
    print(f"[{label}] chi2={float(val):.10e}  grad_beta={float(grads[0]):.8e}  "
          f"grad_thetaE={float(grads[1]):.8e}")
    print(f"[{label}] value_and_grad median: {tmed:.2f} ms")
    return dict(recon=np.asarray(recon), coeffs=np.asarray(coeffs),
               val=float(val), grad=np.asarray([float(grads[0]), float(grads[1])]),
               t=tmed)


print(f"jax x64={jax.config.jax_enable_x64}  device={jax.devices()[0]}")
print(f"n_max={N_MAX} -> {(N_MAX+1)*(N_MAX+2)//2} shapelet components + sersic; "
      f"image {NUM_PIX}x{NUM_PIX}, psf {KSIZE}x{KSIZE}\n")

a = run(None, "f64 (baseline)")
print()
b = run("float32", "f64 + conv32")
print()

recon_rel = np.max(np.abs(a["recon"] - b["recon"])) / (np.max(np.abs(a["recon"])) + 1e-30)
coeff_rel = np.max(np.abs(a["coeffs"] - b["coeffs"])) / (np.max(np.abs(a["coeffs"])) + 1e-30)
val_rel = abs(a["val"] - b["val"]) / (abs(a["val"]) + 1e-30)
grad_rel = np.max(np.abs(a["grad"] - b["grad"]) / (np.abs(a["grad"]) + 1e-30))

print("=== baseline vs conv32 ===")
print(f"max rel diff recon   : {recon_rel:.3e}")
print(f"max rel diff coeffs  : {coeff_rel:.3e}")
print(f"rel diff chi2 (val)  : {val_rel:.3e}")
print(f"max rel diff grad    : {grad_rel:.3e}")
print(f"speedup (t_base/t_new): {a['t']/b['t']:.2f}x  ({a['t']:.1f} -> {b['t']:.1f} ms)")
