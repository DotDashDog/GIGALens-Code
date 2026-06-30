import numpy as np, sys, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
sys.path.insert(0,"/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/_h1h2_diag")
from build_model import prob_model
RUN="/global/u1/l/linusu/GIGALens-Code/experiments/sim_carousel/messy_tests/just_map/"
sz=np.load(RUN+"mclmc/diagnostics.npz")['samples_z']; flat=sz.reshape(-1,32)
s=prob_model.bij.forward(list(jnp.asarray(flat[:5]).T))
print("type:", type(s))
def show(o,p=""):
    if isinstance(o,dict):
        for k,v in o.items(): show(v,p+f"['{k}']")
    elif isinstance(o,(list,tuple)):
        for i,v in enumerate(o): show(v,p+f"[{i}]")
    else:
        import numpy as np
        try: sh=np.asarray(o).shape
        except: sh="?"
        print(f"  {p}: shape={sh}")
show(s)
