
# Shard_map Explained
Shard_map’s behavior is determined by input_spec and output_spec. First, you define your function on a block of data. Shard_map takes in the data, splits/shards it into chunks as specified by input_spec, and passes it into the original function. The function does computation on each chunk and returns the result. Output_spec doesn’t do change inside the body of the function, instead dealing with how to put the results of the different function evaluations back together.

If output_spec has the same sharded dimensions as input_spec, it just reassembles the array the same way it was split up for input_spec. If it has less, then the result is assumed to be replicated across the axes that were sharded in input_spec but are not in output_spec (e.g. because there was a psum or pmean operation done along one of these axes), and only one of the final copies across that axis is returned from shard_map. If out_spec has more sharded axes than in_spec, shard_map will effectively tile the result of the function evaluation across the axes that weren’t sharded in the input but are in the output.

To make it a little clearer, some simple examples are below:

## Examples:
```python
import jax
import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=6' # Use 6 CPU devices
import jax.numpy as jnp

from jax.sharding import Mesh, PartitionSpec as P
# Mesh along one axis
mesh = jax.make_mesh((6,), ('ax1',))

def f(x):
    return x

a = jnp.arange(12)
res = jax.shard_map(f, mesh=mesh, in_specs=P('ax1'), out_specs=P('ax1'))(a)
print(res)
# works like np.concat([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])
# res is [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

res2 = jax.shard_map(f, mesh=mesh, in_specs=P('ax1'), out_specs=P(), check_vma=False)(a)
print(res2)
# res2 is [0, 1]
# Because shard_map has been told that f(block 0) and f(block n) should be the same, it just returns f(block 0).
# It would normally throw an error to warn you about losing the information, (which is good), but we can disable it by setting check_vma=False.


a = jnp.arange(12).reshape(12, 1) #* Add a new axis. The input must have as many dimensions as the larger between in_specs and out_specs

mesh = jax.make_mesh((3,2), ('ax1', 'ax2'))

res3 = jax.shard_map(f, mesh=mesh, in_specs=P('ax1'), out_specs=P('ax1', 'ax2'))(a)
print(res3)
# res3 is [[ 0,  0], [ 1,  1], [ 2,  2], [ 3,  3], [ 4,  4], [ 5,  5], [ 6,  6], [ 7,  7], [ 8,  8], [ 9,  9], [10, 10], [11, 11]]
# f(block 0) is [0, 1, 2, 3]. f(block 1) is [4, 5, 6, 7]. f(block 2) is [8, 9, 10, 11].
# They are concatenated along the 1st axis and then tiled twics along the 2nd axis.
```


