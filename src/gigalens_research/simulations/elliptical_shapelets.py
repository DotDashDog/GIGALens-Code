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

def _transform_e1e2_product_average(x, y, e1, e2, center_x, center_y):
    """FROM LENSTRONOMY: Maps the coordinates x, y with eccentricities e1 e2 into a new elliptical
    coordinate system such that R = sqrt(R_major * R_minor)

    :param x: x-coordinate
    :param y: y-coordinate
    :param e1: eccentricity
    :param e2: eccentricity
    :param center_x: center of distortion
    :param center_y: center of distortion
    :return: distorted coordinates x', y'
    """
    x_shift = x - center_x
    y_shift = y - center_y

    norm = jnp.sqrt(jnp.maximum(jnp.abs(1 - e1**2 - e2**2), 0.000001))
    x_ = ((1 - e1) * x_shift - e2 * y_shift) / norm
    y_ = (-e2 * x_shift + (1 + e1) * y_shift) / norm
    return x_, y_

class EllipticalShapelets(gigalens.profile.LightProfile):
    """Optimized Shapelets that interpolates only n_max+1 unique basis functions
    instead of n_layers redundant copies, then uses index gather to assemble
    the full set of 2D shapelet components."""
    _name = "ELLIPTICALSHAPELETS"
    _params = ["beta", "e1", "e2", "center_x", "center_y"]

    def __init__(self, n_max, use_lstsq=False):
        super(EllipticalShapelets, self).__init__(use_lstsq=use_lstsq)
        del self._params[-1]
        self.n_layers = int((n_max + 1) * (n_max + 2) / 2)
        self.n_max = n_max
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

    @functools.partial(jit, static_argnums=(0,))
    def light(self, x, y, e1, e2,center_x, center_y, beta, **amp):
        _x, _y = _transform_e1e2_product_average(x, y, e1, e2, center_x, center_y)

        _x = _x / beta
        _y = _y / beta

        XX = _phi_basis_1d(_x, self.n_max)
        YY = _phi_basis_1d(_y, self.n_max)
        if self.use_lstsq:
            return XX[self.N1, ...] * YY[self.N2, ...]
        else:
            return jnp.einsum('ij,i...j->...j', jnp.stack([amp[x] for x in self._amp_names], axis=0),
                                XX[self.N1, ...] * YY[self.N2, ...])
