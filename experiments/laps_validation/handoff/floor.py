"""LAPS Phase-1 switch floor measurement.

Reproduces the paper's delta (Eqs 10-11) at exact equilibrium and tests the
resolution: fresh-iid-per-step gives the maximal (decorrelated) floor
sqrt(Var_p[f]/M)/E_p[f]; a persistent (autocorrelated) ensemble suppresses the
window std below that because a finite window of a slow series does not span the
full marginal spread.

f(x) = x_i^2, x ~ N(0,I_d). Var_p[x^2]=2, E_p[x^2]=1 -> iid floor = sqrt(2/M).
"""
import numpy as np

rng = np.random.default_rng(0)

def delta_window(m):
    # m: (T, d) per-step ensemble means of the observable
    mu = m.mean(axis=0)
    sig = m.std(axis=0, ddof=1)
    d = sig / mu                       # (d,) Eq 10
    return d

def iid_floor(M, d=5, T=100, nwin=400):
    """Fresh iid N(0,I) ensemble each step (zero autocorrelation)."""
    per_dim = []
    sw_max = []
    se_max = []                        # standard-error variant: sigma/sqrt(T)/mu
    for _ in range(nwin):
        x = rng.standard_normal((T, M, d))
        m = (x**2).mean(axis=1)        # (T,d) E_rho[x_i^2] per step
        dd = delta_window(m)
        per_dim.append(dd.mean())
        sw_max.append(dd.max())
        mu = m.mean(axis=0); sig = m.std(axis=0, ddof=1)
        se_max.append((sig/np.sqrt(T)/mu).max())
    return np.mean(per_dim), np.mean(sw_max), np.mean(se_max)

def persistent_floor(M, rho_step, d=5, T=100, nwin=200):
    """AR(1) / OU persistent ensemble at equilibrium: each particle marginally
    N(0,I), lag-1 autocorrelation rho_step. Demonstrates window-std suppression."""
    sw_max = []
    a = rho_step; b = np.sqrt(1 - a*a)
    for _ in range(nwin):
        x = rng.standard_normal((M, d))            # stationary start
        m = np.empty((T, d))
        for s in range(T):
            x = a * x + b * rng.standard_normal((M, d))
            m[s] = (x**2).mean(axis=0)
        sw_max.append(delta_window(m).max())
    return np.mean(sw_max)

print("=== (a) paper delta = window-std/window-mean, fresh iid per step ===")
print(f"{'M':>7} {'theory sqrt(2/M)':>17} {'meas per-dim':>13} {'meas max_i(d=5)':>16} {'(b) /sqrtT max':>15}")
for M in (512, 4096, 16384):
    pd, mx, se = iid_floor(M)
    print(f"{M:>7} {np.sqrt(2/M):>17.4f} {pd:>13.4f} {mx:>16.4f} {se:>15.5f}")

print("\n=== persistent (autocorrelated) ensemble at M=4096, T=100 ===")
print(f"{'rho_step':>9} {'tau=-1/ln rho':>14} {'tau/T':>7} {'delta_max floor':>16}")
for rho in (0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99):
    tau = np.inf if rho == 0 else -1/np.log(rho)
    fl = persistent_floor(4096, rho)
    print(f"{rho:>9.2f} {tau:>14.2f} {tau/100:>7.3f} {fl:>16.4f}")

print("\n=== M needed for iid max_i(d=5) floor < 0.01 (extrapolate) ===")
for M in (16384, 32768, 65536):
    pd, mx, se = iid_floor(M, nwin=150)
    print(f"M={M:>6}  max_i delta floor = {mx:.4f}")
