import numpy as np, sys, jax, jax.numpy as jnp, warnings
jax.config.update("jax_enable_x64", True); warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag")
import build_model as bm
WD="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag/"
RUN="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/just_map/"
D=np.load(RUN+"mclmc/diagnostics.npz"); flat=D['samples_z'].reshape(-1,32); mu=flat.mean(0)
vstar=np.load(WD+"CB.npz")['vstar']; s=(flat-mu)@vstar
# 9 typical-set points spread along the slow direction (ordered for interpretability only)
oo=np.argsort(s); pick=oo[np.linspace(0,len(oo)-1,9).astype(int)]
pts=flat[pick]; spos=s[pick]
print("point slow-coords:", np.round(spos,3))

def hessian_fn(lp):
    grad=jax.grad(lp)
    def H(z):
        hvp=lambda v: jax.jvp(grad,(z,),(v,))[1]
        I=jnp.eye(32); cols=[jax.vmap(hvp)(I[k:k+8]) for k in range(0,32,8)]
        Hm=jnp.concatenate(cols,0); return 0.5*(Hm+Hm.T)
    return jax.jit(H)
pm64=bm.make_prob_model("float64"); pm32=bm.make_prob_model("float32")
lp64=lambda z: pm64.log_prob(z)[0]
lp32=lambda z: pm32.log_prob(z)[0]
H64f=hessian_fn(lp64); H32f=hessian_fn(lp32)
# Gaussian control with SAME Sigma, same autodiff path
Sig=np.cov(flat.T); Pinv=jnp.asarray(np.linalg.inv(Sig+1e-10*np.eye(32))); muj=jnp.asarray(mu)
lpG=lambda z: -0.5*jnp.dot(z-muj, Pinv@(z-muj)); HGf=hessian_fn(lpG)

H64=[np.asarray(H64f(jnp.asarray(p))) for p in pts]
H32=[np.asarray(H32f(jnp.asarray(p))) for p in pts]
HG =[np.asarray(HGf(jnp.asarray(p))) for p in pts]
# finiteness/symmetry
fin=all(np.isfinite(h).all() for h in H64); sym=max(np.abs(h-h.T).max()/np.abs(h).max() for h in H64)
print(f"H64 finite={fin}  max rel-asymmetry={sym:.1e}")

def topk_Q(H,k):
    w,V=np.linalg.eigh(H); idx=np.argsort(np.abs(w))[::-1][:k]; return V[:,idx]
def maxangle(Qi,Qj):
    sv=np.clip(np.linalg.svd(Qi.T@Qj,compute_uv=False),-1,1); return np.degrees(np.arccos(sv.min()))

for k in [4,6,8]:
    # precision stability per point (float32 vs float64 top-k subspace)
    prec=[maxangle(topk_Q(H64[i],k),topk_Q(H32[i],k)) for i in range(len(pts))]
    # rotation vs separation, real vs Gaussian
    real=[]; ctrl=[]; sep=[]
    for i in range(len(pts)):
        for j in range(i+1,len(pts)):
            real.append(maxangle(topk_Q(H64[i],k),topk_Q(H64[j],k)))
            ctrl.append(maxangle(topk_Q(HG[i],k),topk_Q(HG[j],k)))
            sep.append(abs(spos[i]-spos[j]))
    real=np.array(real);ctrl=np.array(ctrl);sep=np.array(sep)
    print(f"\n--- top-{k} stiff subspace ---")
    print(f"  precision stability (f32 vs f64) median={np.median(prec):.1f} deg  max={np.max(prec):.1f} deg")
    print(f"  GAUSSIAN-CONTROL rotation: median={np.median(ctrl):.2f} deg  max={np.max(ctrl):.2f} deg  (noise floor)")
    print(f"  REAL-MODEL rotation:       median={np.median(real):.1f} deg  max={np.max(real):.1f} deg")
    print(f"  REAL at max separation: {real[np.argmax(sep)]:.1f} deg;  monotonic? corr(angle,sep)={np.corrcoef(real,sep)[0,1]:+.2f}")
    if k==6:
        plt.figure(figsize=(7,5))
        plt.scatter(sep,real,label="real model",c='tab:red')
        plt.scatter(sep,ctrl,label="Gaussian control (floor)",c='tab:gray')
        plt.axhline(np.median(prec),ls='--',c='b',label=f"precision floor {np.median(prec):.0f}°")
        plt.xlabel("separation along slow dir |Δs|"); plt.ylabel("max principal angle of top-6 stiff subspace (deg)")
        plt.legend(); plt.title("Hessian eigenframe rotation along ridge"); plt.savefig(WD+"hess_rotation.png",dpi=95)
print("saved hess_rotation.png")
np.savez(WD+"hess.npz", H64=np.array(H64), H32=np.array(H32), HG=np.array(HG), spos=spos, pts=pts)
