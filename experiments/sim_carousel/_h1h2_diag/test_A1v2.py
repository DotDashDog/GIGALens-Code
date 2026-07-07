import numpy as np, sys, jax, jax.numpy as jnp, warnings
jax.config.update("jax_enable_x64", True); warnings.filterwarnings("ignore")
sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag")
import build_model as bm
RUN="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/just_map/"
sz=np.load(RUN+"mclmc/diagnostics.npz")['samples_z']; flat=sz.reshape(-1,32)
rng=np.random.default_rng(1)
pts=jnp.asarray(flat[rng.choice(len(flat),12,replace=False)])   # typical set
pm32=bm.make_prob_model("float32"); pm64=bm.make_prob_model("float64")
def vg(pm):
    f=lambda z: pm.log_prob(z)[0]
    return jax.jit(jax.vmap(jax.value_and_grad(f)))
vg32=vg(pm32); vg64=vg(pm64)
v32,g32=vg32(pts); v64,g64=vg64(pts)
v32=np.asarray(v32);g32=np.asarray(g32);v64=np.asarray(v64);g64=np.asarray(g64)
print("=== DIRECT conv-precision noise (float32-conv vs float64-conv), typical set ===")
print(f"  |logp_f32 - logp_f64|:  median={np.median(np.abs(v32-v64)):.3e}  max={np.abs(v32-v64).max():.3e}")
gn=np.linalg.norm(g64,axis=1)
gerr=np.linalg.norm(g32-g64,axis=1)
print(f"  |grad_f32 - grad_f64|/|grad|: median={np.median(gerr/gn):.3e}  max={np.max(gerr/gn):.3e}")
print(f"  (|grad| median={np.median(gn):.2e})")
# how big is the logp noise vs the controller's target? dim*dev=32*5e-4=0.016 -> dE target ~0.13
print(f"  controller per-step energy-error target |dE| ~ sqrt(dim*dev)={np.sqrt(32*5e-4):.3f};")
print(f"     conv-noise in logp (median {np.median(np.abs(v32-v64)):.3e}) is this fraction of it: "
      f"{np.median(np.abs(v32-v64))/np.sqrt(32*5e-4):.2f}")

# FD-vs-AD floor for each precision, along gradient direction (Dad=|g|, large denom)
def fd_floor(pm, vgf):
    f=jax.jit(jax.vmap(lambda z: pm.log_prob(z)[0]))
    hs=np.geomspace(1e-3,1e-7,12); out=[]
    vv,gg=vgf(pts); gg=np.asarray(gg)
    for i,z0 in enumerate(np.asarray(pts)):
        v=gg[i]/np.linalg.norm(gg[i]); Dad=float(gg[i]@v)  # = |g|
        best=np.inf
        for h in hs:
            fp=float(f(jnp.asarray((z0+h*v)[None]))[0]); fm=float(f(jnp.asarray((z0-h*v)[None]))[0])
            best=min(best, abs((fp-fm)/(2*h)-Dad)/abs(Dad))
        out.append(best)
    return np.array(out)
fl32=fd_floor(pm32,vg32); fl64=fd_floor(pm64,vg64)
print(f"\n=== FD-vs-AD min rel-error along grad dir (smooth float64 ~1e-8 expected) ===")
print(f"  conv=float32 (NOTEBOOK): median={np.median(fl32):.2e}  max={fl32.max():.2e}")
print(f"  conv=float64 (control):  median={np.median(fl64):.2e}  max={fl64.max():.2e}")
