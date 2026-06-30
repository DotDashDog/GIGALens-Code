import numpy as np, sys, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag")
from build_model import prob_model
RUN="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/just_map/"
sz=np.load(RUN+"mclmc/diagnostics.npz")['samples_z']; C,N,P=sz.shape
flat=sz.reshape(-1,P)
s=prob_model.bij.forward(list(jnp.asarray(flat).T))
g=lambda k: np.asarray(s[k]).reshape(C,N)
e1=g('planes/0/mass/1/e1'); e2=g('planes/0/mass/1/e2')
te=g('planes/0/mass/1/theta_E'); ga=g('planes/0/mass/1/gamma')
cx=g('planes/0/mass/1/center_x'); cy=g('planes/0/mass/1/center_y')
e1f=g('planes/0/mass/2/e1'); e2f=g('planes/0/mass/2/e2')
g1=g('planes/0/mass/3/gamma1'); g2=g('planes/0/mass/3/gamma2')
grpH = sz[:,:,23].mean(1) > -3.65
md=lambda a,m: (np.median(a[m]), a[m].std())
for lbl,m in [(f"Mode A chains{list(np.where(grpH)[0])}",grpH),
              (f"Mode B chains{list(np.where(~grpH)[0])}",~grpH)]:
    emag=np.sqrt(e1[m]**2+e2[m]**2); pa=0.5*np.degrees(np.arctan2(e2[m],e1[m]))
    print(f"\n{lbl}   (n_chains={m.sum()})")
    print(f"  EPL_Le e1 = {md(e1,m)[0]:+.4f} ± {md(e1,m)[1]:.4f}")
    print(f"  EPL_Le e2 = {md(e2,m)[0]:+.4f} ± {md(e2,m)[1]:.4f}")
    print(f"  EPL_Le |e|= {np.median(emag):.4f}    PA = {np.median(pa):+.1f}°")
    print(f"  EPL_Le theta_E={md(te,m)[0]:.3f}  gamma={md(ga,m)[0]:.3f}  cx={md(cx,m)[0]:.3f} cy={md(cy,m)[0]:.3f}")
    print(f"  EPL_Lf e1={md(e1f,m)[0]:+.3f} e2={md(e2f,m)[0]:+.3f}   shear g1={md(g1,m)[0]:+.3f} g2={md(g2,m)[0]:+.3f}")
print(f"\n(prior on EPL_Le e1,e2 = TruncatedNormal(0, 0.1, -0.3, 0.3))")
