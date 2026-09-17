import numpy as np, jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from gigalens_research.simulations.image_based_light import ImageBasedLight
a=np.zeros((5,5)); a[3,1]=1.0   # row 3, col 1  -> numpy/imshow: y=+1 above centre, x=-1 left of centre
L=ImageBasedLight(a, 1.0)
for (x,y) in [(1.0,-1.0),(-1.0,1.0)]:
    print(f'light(x={x:+.1f}, y={y:+.1f}) =', np.asarray(L.light(jnp.array([x]),jnp.array([y]),0.0,0.0,1.0)))
print('=> the nonzero pixel img[3,1] sits at (x=+1,y=-1) in scene coords, i.e. axis0 is x: the lensed source is the TRANSPOSE of what imshow(img, origin="lower") shows.')
