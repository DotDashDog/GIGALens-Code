import functools

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp
from jax import jit
from lenstronomy.LightModel.Profiles.shapelets import Shapelets as LenstronomyShapelets

import gigalens.profile


_PI_NEG_QUARTER = float(jnp.pi) ** -0.25


def _phi_basis_1d(x, n_max):
    """Compute the orthonormal Hermite (shapelet) basis ``phi_n(x)`` for
    ``n = 0, ..., n_max`` using the numerically stable three-term recurrence
    on the *normalized* functions:

        phi_0(x) = pi^(-1/4) * exp(-x^2/2)
        phi_1(x) = sqrt(2) * x * phi_0(x)
        phi_n(x) = sqrt(2/n) * x * phi_{n-1}(x) - sqrt((n-1)/n) * phi_{n-2}(x)

    Unlike the textbook approach (compute H_n(x) via its recurrence, then
    multiply by the tiny prefactor 1/sqrt(2^n * n! * sqrt(pi)) and the
    Gaussian envelope exp(-x^2/2)), this recurrence keeps every intermediate
    value bounded uniformly in n and x (|phi_n(x)| <= O(1)), so it does not
    overflow in float32 for any n_max or x, and its VJP is finite everywhere.
    """
    phi = jnp.empty((n_max + 1, *x.shape), dtype=x.dtype)
    phi = phi.at[0].set(_PI_NEG_QUARTER * jnp.exp(-x ** 2 / 2))
    if n_max >= 1:
        phi = phi.at[1].set(jnp.sqrt(2.0) * x * phi[0])
    for n in range(2, n_max + 1):
        phi = phi.at[n].set(
            jnp.sqrt(2.0 / n) * x * phi[n - 1]
            - jnp.sqrt((n - 1) / n) * phi[n - 2]
        )
    return phi


class Sersiclets(gigalens.profile.LightProfile):
    _name = "SHAPELETS"
    _params = ["beta", "n_sersic","center_x", "center_y"]

    def __init__(self, n_max, use_lstsq=False, interpolate=True):
        super(Shapelets, self).__init__(use_lstsq=use_lstsq)
        del self._params[-1]  # Deletes the amp parameter, to be added again later below with numbering convention
        self.n_layers = int((n_max + 1) * (n_max + 2) / 2)
        self.n_max = n_max
        self.interpolate = interpolate
        n1 = 0
        n2 = 0
        herm_X = []
        herm_Y = []
        self.N1 = []
        self.N2 = []
        decimal_places = len(str(self.n_layers))
        self._amp_names = []
        for i in range(self.n_layers):
            self._params.append(f"amp{str(i).zfill(decimal_places)}")
            self._amp_names.append(f"amp{str(i).zfill(decimal_places)}")
            self.N1.append(n1)
            self.N2.append(n2)
            herm_X.append(LenstronomyShapelets().phi_n(n1, jnp.linspace(-5, 5, 6000)))
            herm_Y.append(LenstronomyShapelets().phi_n(n2, jnp.linspace(-5, 5, 6000)))
            if n1 == 0:
                n1 = n2 + 1
                n2 = 0
            else:
                n1 -= 1
                n2 += 1
        self.depth = len(self._amp_names)
        self.herm_X = jax.tree.map(lambda x : x.astype('float32'), herm_X)
        self.herm_Y = jax.tree.map(lambda x : x.astype('float32'), herm_Y)

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y, center_x, center_y, beta, **amp):
        x = (x - center_x) / beta
        y = (y - center_y) / beta
        if self.interpolate:
            ret = tfp.math.interp_regular_1d_grid(x, -5., 5., self.herm_X, fill_value_below=0., fill_value_above=0.)
            ret = ret * tfp.math.interp_regular_1d_grid(y, -5., 5., self.herm_Y, fill_value_below=0.,
                                                        fill_value_above=0.)
            if self.use_lstsq:
                return ret
            else:
                ret = jnp.einsum('i...j,ij->i...j', ret, jnp.stack([amp[x] for x in self._amp_names], axis=0))
                return jnp.sum(ret, axis=0)
        else:
            XX = _phi_basis_1d(x, self.n_max)
            YY = _phi_basis_1d(y, self.n_max)
            if self.use_lstsq:
                return XX[self.N1, ...] * YY[self.N2, ...]
            else:
                return jnp.einsum('ij,i...j->...j', jnp.stack([amp[x] for x in self._amp_names], axis=0),
                                  XX[self.N1, ...] * YY[self.N2, ...])


class ShapeletsFast(gigalens.profile.LightProfile):
    """Optimized Shapelets that interpolates only n_max+1 unique basis functions
    instead of n_layers redundant copies, then uses index gather to assemble
    the full set of 2D shapelet components."""
    _name = "SHAPELETS"
    _params = ["beta", "center_x", "center_y"]

    def __init__(self, n_max, use_lstsq=False, interpolate=True):
        super(ShapeletsFast, self).__init__(use_lstsq=use_lstsq)
        del self._params[-1]
        self.n_layers = int((n_max + 1) * (n_max + 2) / 2)
        self.n_max = n_max
        self.interpolate = interpolate
        n1 = 0
        n2 = 0
        N1 = []
        N2 = []
        decimal_places = len(str(self.n_layers))
        self._amp_names = []
        for i in range(self.n_layers):
            self._params.append(f"amp{str(i).zfill(decimal_places)}")
            self._amp_names.append(f"amp{str(i).zfill(decimal_places)}")
            N1.append(n1)
            N2.append(n2)
            if n1 == 0:
                n1 = n2 + 1
                n2 = 0
            else:
                n1 -= 1
                n2 += 1
        self.N1 = jnp.array(N1)
        self.N2 = jnp.array(N2)
        self.depth = len(self._amp_names)

        if self.interpolate:
            grid = jnp.linspace(-5, 5, 6000)
            ls = LenstronomyShapelets()
            unique_phi = [ls.phi_n(n, grid) for n in range(n_max + 1)]
            self.unique_phi = jnp.stack(unique_phi, axis=0).astype('float32')

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y, center_x, center_y, beta, **amp):
        x = (x - center_x) / beta
        y = (y - center_y) / beta
        if self.interpolate:
            phi_x = tfp.math.interp_regular_1d_grid(
                x, -5., 5., self.unique_phi, fill_value_below=0., fill_value_above=0.
            )
            phi_y = tfp.math.interp_regular_1d_grid(
                y, -5., 5., self.unique_phi, fill_value_below=0., fill_value_above=0.
            )
            ret = phi_x[self.N1] * phi_y[self.N2]
            if self.use_lstsq:
                return ret
            else:
                ret = jnp.einsum('i...j,ij->i...j', ret, jnp.stack([amp[x] for x in self._amp_names], axis=0))
                return jnp.sum(ret, axis=0)
        else:
            XX = _phi_basis_1d(x, self.n_max)
            YY = _phi_basis_1d(y, self.n_max)
            if self.use_lstsq:
                return XX[self.N1, ...] * YY[self.N2, ...]
            else:
                return jnp.einsum('ij,i...j->...j', jnp.stack([amp[x] for x in self._amp_names], axis=0),
                                  XX[self.N1, ...] * YY[self.N2, ...])
