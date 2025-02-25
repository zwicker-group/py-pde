"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from scipy import fftpack, stats

from pde import CartesianGrid, UnitGrid
from pde.tools.spectral import make_colored_noise


def spectral_density(data, dx=1.0):
    """Calculate the power spectral density of a field.

    Args:
        data (:class:`~numpy.ndarray`):
            Data of which the power spectral density will be calculated
        dx (float or list): The discretizations of the grid either as a single
            number or as an array with a value for each dimension

    Returns:
        A tuple with two arrays containing the magnitudes of the wave vectors
        and the associated density, respectively.
    """
    dim = len(data.shape)
    dx = np.broadcast_to(dx, (dim,))

    # prepare wave vectors
    k2s = 0
    for i in range(dim):
        k = fftpack.fftfreq(data.shape[i], dx[i])
        k2s = np.add.outer(k2s, k**2)

    res = fftpack.fftn(data)

    return np.sqrt(k2s), np.abs(res) ** 2


def test_colored_noise(rng):
    """Test the implementation of the colored noise."""
    grid = UnitGrid([64, 64], periodic=True)
    for exponent in [0, -1, 2]:
        scale = rng.uniform(1, 10)
        noise = make_colored_noise(grid.shape, grid.discretization[0], exponent, scale)
        x = noise()
        msg = f"Colored noise with exp={exponent} is not normal distributed"
        assert stats.normaltest(x.flat).pvalue > 2e-5, msg


def test_noise_scaling(rng):
    """Compare the noise strength (in terms of the spectral density of two different
    noise sources that should be equivalent)"""
    # create a grid
    x, w = 2 + 10 * rng.random(2)
    size = rng.integers(128, 256)
    grid = CartesianGrid([[x, x + w]], size, periodic=True)

    # colored noise
    noise_colored = make_colored_noise(grid.shape, grid.discretization[0], exponent=2)

    # divergence of white noise
    shape = (grid.dim,) + grid.shape
    div = grid.make_operator("divergence", bc="auto_periodic_neumann")

    def noise_div():
        return div(rng.normal(size=shape))

    def get_noise(noise_func):
        k, density = spectral_density(data=noise_func(), dx=grid.discretization[0])
        assert k[0] == 0
        assert density[0] == pytest.approx(0)
        return np.log(density[1])  # log of spectral density

    # calculate spectral densities of the two noises
    result = []
    for noise_func in [noise_colored, noise_div]:
        # average spectral density of longest length scale
        mean = np.mean([get_noise(noise_func=noise_func) for _ in range(64)])
        result.append(mean)

    np.testing.assert_allclose(*result, rtol=0.5)
