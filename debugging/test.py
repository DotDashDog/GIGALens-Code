import jax
import pickle

z_sn = pickle.load(open("GIGALens-Code/z_singlenode.pkl", "rb"))


z_mn = pickle.load(open("GIGALens-Code/z_multinode.pkl", "rb"))
print(z_sn.shape)
print(z_mn.shape)
# print(jax.numpy.sum(jax.numpy.abs(z_sn-z_mn)))

# cov_sn = pickle.load(open("GIGALens-Code/cov_singlenode.pkl", "rb"))
# cov_mn = pickle.load(open("GIGALens-Code/cov_multinode.pkl", "rb"))
# print(jax.numpy.sum(jax.numpy.abs(cov_sn-cov_mn)))

# mean_sn = pickle.load(open("GIGALens-Code/mean_singlenode.pkl", "rb"))
# mean_mn = pickle.load(open("GIGALens-Code/mean_multinode.pkl", "rb"))
# print(jax.numpy.sum(jax.numpy.abs(mean_sn-mean_mn)))