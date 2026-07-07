import os, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import sys; sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/minimal_case_recheck_plots/_scripts")
from build_model import build
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from gigalens.jax.profiles.mass.nfw import NFW_ELLIPSE
from gigalens.jax.profiles.mass.shear import Shear
EXP="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel"
OUT=f"{EXP}/minimal_case_recheck_plots"
pm=build()
zmap=np.asarray(np.load(f"{EXP}/messy_tests/minimal_case/map/arrays.npz")['z_best']).reshape(-1)
x=pm.bij.forward(list(zmap.reshape(-1,1)))
def g(k): return float(np.reshape(np.asarray(x[k]),()))
# discover keys
keys=list(x.keys())
# NFW params and shear
import re
def find(sub):
    for k in keys:
        if sub in k: return k
    return None
P={}
for nm in ['Rs','alpha_Rs','e1','e2','center_x','center_y']:
    P[nm]=g([k for k in keys if k.endswith('/mass/0/'+nm)][0])
sg1=g([k for k in keys if k.endswith('/mass/1/gamma1')][0])
sg2=g([k for k in keys if k.endswith('/mass/1/gamma2')][0])
print("NFW MAP phys:",P,"shear",sg1,sg2)
nfw=NFW_ELLIPSE(); sh=Shear()
def alpha(theta):
    X,Y=theta[...,0],theta[...,1]
    ax1,ay1=nfw.deriv(X,Y,P['Rs'],P['alpha_Rs'],P['e1'],P['e2'],P['center_x'],P['center_y'])
    ax2,ay2=sh.deriv(X,Y,sg1,sg2)
    return jnp.stack([ax1+ax2,ay1+ay2],-1)
def beta(theta): return theta-alpha(theta)
# jacobian A=dbeta/dtheta on grid; detA via jacfwd per point
jb=jax.jit(jax.vmap(jax.jacfwd(beta)))
# image-plane grid around lens center (NFW cx~5.04 cy~3.89), source near 4.4,3.9
gx=np.linspace(-30,40,500); gy=np.linspace(-31,39,500)
GX,GY=np.meshgrid(gx,gy,indexing='xy')
TH=jnp.asarray(np.stack([GX.ravel(),GY.ravel()],-1))
A=np.asarray(jb(TH)).reshape(GY.size//gx.size if False else gy.size,gx.size,2,2) if False else np.asarray(jb(TH))
det=(A[:,0,0]*A[:,1,1]-A[:,0,1]*A[:,1,0]).reshape(gy.size,gx.size)
# source positions of grid
B=np.asarray(jax.vmap(beta)(TH)).reshape(gy.size,gx.size,2)
# extract critical curve contour det=0, map to source plane
fig,ax=plt.subplots(1,2,figsize=(14,6.4))
cs=ax[0].contour(GX,GY,det,levels=[0],colors='r')
pr=det
print("detA range: %.4g .. %.4g ; sign-change=%s"%(det.min(),det.max(),det.min()<0<det.max()),flush=True)
ax[0].set_title("image plane: critical curve (red)"); ax[0].set_xlabel("x"); ax[0].set_ylabel("y"); ax[0].set_aspect('equal')
ax[0].plot(P['center_x'],P['center_y'],'kx')
# caustic: map each critical-curve vertex through beta
caustic_pts=[]
for seg in cs.allsegs[0]:
    bs=np.asarray(beta(jnp.asarray(seg)))
    caustic_pts.append(bs)
    ax[1].plot(bs[:,0],bs[:,1],'r-',lw=1)
ax[1].set_title("source plane: caustic (red) + src4 cx modes"); ax[1].set_xlabel("beta_x"); ax[1].set_ylabel("beta_y"); ax[1].set_aspect('equal')
# src4 cy phys ~ col10 mean
sz=np.load(f"{EXP}/messy_tests/minimal_case/mclmc/arrays.npz")['samples_z'].reshape(-1,14)
cy4=sz[:,10].mean()
modeL,modeU=4.3345,4.4486
ax[1].plot([modeL],[cy4],'bo',ms=9,label=f'lower cx={modeL:.3f}')
ax[1].plot([modeU],[cy4],'go',ms=9,label=f'upper cx={modeU:.3f}')
# also src5 position
ax[1].plot([sz[:,12].mean()],[sz[:,13].mean()],'m^',ms=8,label='src5')
ax[1].legend(fontsize=8)

plt.tight_layout(); plt.savefig(f"{OUT}/R_caustic_check.png",dpi=120)
allc=np.vstack(caustic_pts) if caustic_pts else np.zeros((0,2))
np.savez(f"{OUT}/R_caustic_check.npz",caustic=allc,modeL=modeL,modeU=modeU,cy4=cy4,
         nfw=P,shear=(sg1,sg2))
# crude: distance from each mode point to nearest caustic vertex; and is the segment between modes crossing a caustic?
def mindist(px,py):
    if len(allc)==0: return np.nan
    return np.min(np.hypot(allc[:,0]-px,allc[:,1]-py))
print("min dist lower-mode to caustic:",mindist(modeL,cy4))
print("min dist upper-mode to caustic:",mindist(modeU,cy4))
# sample along segment between modes, count caustic crossings via point-in-... use winding of nearest
seg_x=np.linspace(modeL,modeU,200)
# distance profile along segment
d=[mindist(sx,cy4) for sx in seg_x]
print("caustic-distance along mode segment: min %.4f at x=%.4f (modes at %.3f,%.3f)"%(np.min(d),seg_x[np.argmin(d)],modeL,modeU))
print("saved R_caustic_check.png")
