import numpy as np, sys, jax, jax.numpy as jnp, warnings
jax.config.update("jax_enable_x64", True); warnings.filterwarnings("ignore")
sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag")
import build_model as bm
WD="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag/"
names=list(np.load(WD+"names.npy"))
H=np.load(WD+"hess.npz"); pts=H['pts']; spos=H['spos']; Hz=H['H64']   # z-space Hessians + points
pm=bm.make_prob_model("float64"); bij=pm.bij
lp=lambda z: pm.log_prob(z)[0]
def fldj(zv):
    zl=[zv[i] for i in range(32)]
    ld=bij.forward_log_det_jacobian(zl, event_ndims=[0]*32)
    return sum(jnp.sum(x) for x in jax.tree.leaves(ld))
def to_x(zv):                      # physical flat vector (names order)
    d=bij.forward([zv[i] for i in range(32)]); return jnp.array([d[n] for n in names])
def phys(xf):                      # physical log-posterior as fn of flat x
    d={names[i]:xf[i] for i in range(32)}
    zl=bij.inverse(d); zv=jnp.array([zl[i] for i in range(32)])
    return lp(zv)-fldj(zv)
# ---- VALIDATION GATE ----
z0=jnp.asarray(pts[4]); x0=to_x(z0)
rt=float(jnp.max(jnp.abs(jnp.array([bij.inverse({names[i]:x0[i] for i in range(32)})[i] for i in range(32)])-z0)))
lhs=float(lp(z0)); rhs=float(phys(x0)+fldj(z0))
print(f"VALIDATION: round-trip |inverse(forward(z))-z| max = {rt:.2e}")
print(f"VALIDATION: lp(z) = {lhs:.4f}  vs  phys(x)+fldj(z) = {rhs:.4f}  diff={abs(lhs-rhs):.2e}")
# sign control: if I had used lp+fldj, this check would fail by 2*fldj
print(f"  (fldj(z0)={float(fldj(z0)):.2f}; a sign error would show diff ~ {abs(2*float(fldj(z0))):.1f})")
ok = rt<1e-5 and abs(lhs-rhs)<1e-3
print(f"VALIDATION {'PASSED' if ok else 'FAILED'} -- {'proceeding' if ok else 'NOT trusting physical Hessian'}")
if not ok: sys.exit(0)

# ---- physical Hessians at same points, rotation test ----
gradp=jax.grad(phys)
def Hphys(xf):
    hvp=lambda v: jax.jvp(gradp,(xf,),(v,))[1]
    I=jnp.eye(32); cols=[jax.vmap(hvp)(I[k:k+8]) for k in range(0,32,8)]
    M=jnp.concatenate(cols,0); return 0.5*(M+M.T)
Hphys=jax.jit(Hphys)
xpts=[to_x(jnp.asarray(p)) for p in pts]
Hx=[np.asarray(Hphys(x)) for x in xpts]
print(f"\nphys Hessian finite={all(np.isfinite(h).all() for h in Hx)}")
def topk(Hm,k):
    w,V=np.linalg.eigh(Hm); return V[:,np.argsort(np.abs(w))[::-1][:k]]
def ang(Qi,Qj):
    sv=np.clip(np.linalg.svd(Qi.T@Qj,compute_uv=False),-1,1); return np.degrees(np.arccos(sv.min()))
print(f"\n{'k':>3} {'z-rotation(median/max)':>26} {'PHYS-rotation(median/max)':>28}")
for k in [4,6,8]:
    rz=[];rx=[]
    for i in range(len(pts)):
        for j in range(i+1,len(pts)):
            rz.append(ang(topk(Hz[i],k),topk(Hz[j],k)))
            rx.append(ang(topk(Hx[i],k),topk(Hx[j],k)))
    rz=np.array(rz);rx=np.array(rx)
    print(f"{k:3d} {np.median(rz):10.1f}/{np.max(rz):<14.1f} {np.median(rx):12.1f}/{np.max(rx):<14.1f}")
print("\n=> phys-rotation ~ z-rotation: curvature is PHYSICAL; phys-rotation ~0: bijector-induced")
