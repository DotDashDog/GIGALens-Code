import numpy as np, sys, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag")
from build_model import prob_model
names=list(np.load("/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag/names.npy"))
RUN="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/just_map/"
D=np.load(RUN+"mclmc/diagnostics.npz")
sz=D['samples_z']; C,N,P=sz.shape
imm=D['inverse_mass_matrix'][0,-1]      # final mass matrix (PxP)
xi=D['xi'][:,-N:]
# hard prior bounds per param name (from notebook)
B={'n_sersic':(0.1,10)}
def bounds(nm):
    if 'n_sersic' in nm: return (0.1,10)
    if 'mass/0/e' in nm: return (-0.2,0.2)         # NFW e1,e2
    if nm.endswith('/e1') or nm.endswith('/e2'): return (-0.3,0.3)  # source & EPL ellipt, shear
    if nm.endswith('/gamma'): return (1,3)
    if 'mass/3/gamma' in nm: return (-0.3,0.3)
    return None
# constrained values
s=prob_model.bij.forward(list(jnp.asarray(sz.reshape(-1,P)).T))
con={n:np.asarray(s[n]).reshape(C,N) for n in names}
zvar=sz.reshape(-1,P).var(0)            # unconstrained variance (what Welford/M sees)
immdiag=np.diag(imm)
print("Top 8 params by MASS-MATRIX diagonal (the directions M makes biggest):")
for i in np.argsort(immdiag)[::-1][:8]:
    nm=names[i]; b=bounds(nm); cv=con[nm].ravel()
    prox=""
    if b is not None:
        # fractional distance of median to nearest bound (0=on bound)
        d=min(np.median(cv)-b[0], b[1]-np.median(cv))/((b[1]-b[0])/2)
        prox=f" | bound{b} median={np.median(cv):.3f} dist_to_bound_frac={d:.2f}" + ("  <== RAILING" if d<0.15 else "")
    print(f"  {i:2d} {nm:24s} M_diag={immdiag[i]:.3e} zvar={zvar[i]:.3e}{prox}")
# eigenvector of largest M eigenvalue
w,V=np.linalg.eigh(imm); 
print(f"\nLargest M eigenvalue={w[-1]:.3e}; dominant params in its eigenvector:")
top=np.argsort(np.abs(V[:,-1]))[::-1][:5]
for i in top: print(f"   {i:2d} {names[i]:24s} loading={V[i,-1]:+.2f}")
# does xi blow up when boundary params are extreme? use n_sersic(0) & EPL_Le e2(22) & src9 e2(1)
print("\ncorr(log10 xi, |z-med|/std) for boundary-railing params:")
flat=sz.reshape(-1,P); med=flat.mean(0); sd=flat.std(0)+1e-12
lxi=np.log10(np.clip(xi,1e-12,None)).ravel()
for i in [0,1,2,22,16,17]:
    dev=np.abs(flat[:,i]-med[i])/sd[i]
    print(f"   {i:2d} {names[i]:24s} corr={np.corrcoef(lxi,dev)[0,1]:+.3f}")
