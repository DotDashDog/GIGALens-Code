import jax, jax.numpy as jnp, numpy as np
from gigalens_research.inference.laps_late_adjusted import LAPS_late_adjusted
print("devices", jax.devices()[:1], "...", len(jax.devices()), "total")

d = 8
sig = np.geomspace(1e-3, 0.1, d)          # per-dim true std: 100x anisotropy (lens-like)
sig_j = jnp.asarray(sig)
def logp(x):                               # anisotropic Gaussian, batched (n,d)->(n,)
    return -0.5 * jnp.sum((x / sig_j)**2, axis=-1)

# "prior": over-dispersed isotropic init at ~0.05 (mimics the lens prior draws' start scale)
key = jax.random.key(0)
init = 0.05 * jax.random.normal(key, (512, d))

STEPS = 3000
res = LAPS_late_adjusted(logp, dim=d, num_chains=512,
                         num_unadjusted_steps=STEPS, num_adjusted_steps=1,
                         init_positions=init, early_stop=False,
                         phase2_enabled=False, chunk_size=100, seed=0)
var = np.asarray(res.p1_obs_sq) - np.asarray(res.p1_obs_mean)**2   # (T1,d)
std = np.sqrt(np.clip(var, 0, None))
final = std[-1]
# slope of log std over final third
f0 = 2*STEPS//3
slope = np.polyfit(np.arange(f0, STEPS), np.log(std[f0:]+1e-30), 1)[:, ]  # per-dim
slopes = np.array([np.polyfit(np.arange(f0, STEPS), np.log(std[f0:, i]+1e-30), 1)[0] for i in range(d)])
print(f"switch_index_paper={int(res.switch_index_paper)}  phase1_len={int(res.phase1_len)}  (T1={STEPS})")
print(f"{'dim':>3} {'true_sig':>9} {'init_std0':>9} {'final_std':>9} {'final/true':>10} {'logslope_f3':>12} {'verdict':>16}")
for i in range(d):
    ratio = final[i]/sig[i]
    verdict = 'contracted' if ratio < 2 else ('OVER-HEATED' if slopes[i] > 0 else 'stuck-wide')
    print(f"{i:>3} {sig[i]:9.2e} {std[0,i]:9.2e} {final[i]:9.2e} {ratio:10.1f} {slopes[i]:12.2e} {verdict:>16}")
np.savez("/global/homes/l/linusu/.claude/jobs/cf0ab128/tmp/aniso_bed.npz",
         std=std, sig=sig, slopes=slopes)
print("BED DONE")
