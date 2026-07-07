"""DC-7.1 arm A3: the 100x anisotropic Gaussian bed (known-answer control) with
the F1 lever ON (velocity_init="gradient"). F1 must NOT break the already-working
cold-start contraction (falsifier: any dim stuck-wide/over-heated that the
baseline bed contracted). Identical to aniso_bed.py except the lever + OUT path.
"""
import os
import jax, jax.numpy as jnp, numpy as np
from gigalens_research.inference.laps_late_adjusted import LAPS_late_adjusted
print("devices", jax.devices()[:1], "...", len(jax.devices()), "total")

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "diag_levers"); os.makedirs(OUT, exist_ok=True)

d = 8
sig = np.geomspace(1e-3, 0.1, d)          # per-dim true std: 100x anisotropy (lens-like)
sig_j = jnp.asarray(sig)
def logp(x):
    return -0.5 * jnp.sum((x / sig_j)**2, axis=-1)

key = jax.random.key(0)
init = 0.05 * jax.random.normal(key, (512, d))

STEPS = 3000
res = LAPS_late_adjusted(logp, dim=d, num_chains=512,
                         num_unadjusted_steps=STEPS, num_adjusted_steps=1,
                         init_positions=init, early_stop=False,
                         phase2_enabled=False, chunk_size=100, seed=0,
                         velocity_init="gradient")
var = np.asarray(res.p1_obs_sq) - np.asarray(res.p1_obs_mean)**2   # (T1,d)
std = np.sqrt(np.clip(var, 0, None))
final = std[-1]
f0 = 2*STEPS//3
slopes = np.array([np.polyfit(np.arange(f0, STEPS), np.log(std[f0:, i]+1e-30), 1)[0]
                   for i in range(d)])
print(f"switch_index_paper={int(res.switch_index_paper)}  phase1_len={int(res.phase1_len)}  (T1={STEPS})")
print(f"{'dim':>3} {'true_sig':>9} {'init_std0':>9} {'final_std':>9} {'final/true':>10} {'logslope_f3':>12} {'verdict':>16}")
for i in range(d):
    ratio = final[i]/sig[i]
    verdict = 'contracted' if ratio < 2 else ('OVER-HEATED' if slopes[i] > 0 else 'stuck-wide')
    print(f"{i:>3} {sig[i]:9.2e} {std[0,i]:9.2e} {final[i]:9.2e} {ratio:10.1f} {slopes[i]:12.2e} {verdict:>16}")
np.savez(os.path.join(OUT, "aniso_bed_f1.npz"), std=std, sig=sig, slopes=slopes)
print("BED DONE")
