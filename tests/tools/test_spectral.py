"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest
from scipy import fftpack, stats

from pde import CartesianGrid, UnitGrid
from pde.tools.spectral import make_colored_noise, make_correlated_noise


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

    return np.sqrt(k2s), np.abs(res) ** 2 / len(data)


def test_colored_noise(rng):
    """Test the implementation of the colored noise."""
    grid = CartesianGrid([[0, 32], [0, 32]], [16, 64], periodic=True)
    for exponent in [0, -1, 2]:
        scale = rng.uniform(1, 10)
        noise = make_colored_noise(
            grid.shape, dx=grid.discretization[0], exponent=exponent
        )
        x = scale * noise()
        msg = f"Colored noise with exp={exponent} is not normal distributed"
        assert stats.normaltest(x.flat).pvalue > 2e-5, msg

    # check scaling of almost uncorrelated case
    s = 5
    n1 = s * make_colored_noise(grid.shape, dx=grid.discretization, exponent=1e-10)()
    n2 = s * rng.normal(size=grid.shape)

    assert stats.ks_2samp(n1.flat, n2.flat).pvalue > 0.05
    # compare Laplacian of field, which should be uncorrelated
    laplace = grid.make_operator("laplace", bc="periodic")
    assert stats.ks_2samp(laplace(n1).flat, laplace(n2).flat).pvalue > 0.05


def test_correlated_noise(rng):
    """Test the implementation of the correlated noise."""
    grid = CartesianGrid([[0, 32], [0, 32]], [16, 64], periodic=True)
    for corr_length in [0, 0.1, 1]:
        scale = rng.uniform(1, 10)
        noise = make_correlated_noise(
            grid.shape,
            correlation="gaussian",
            discretization=grid.discretization[0],
            length_scale=corr_length,
        )
        x = scale * noise()
        msg = f"Colored noise with corr_length={corr_length} is not normal distributed"
        assert stats.normaltest(x.flat).pvalue > 2e-5, msg

    # check scaling of almost uncorrelated case
    s = 5
    n = make_correlated_noise(grid.shape, correlation="gaussian", length_scale=1e-10)
    n1 = s * n()
    n2 = s * rng.normal(size=grid.shape)

    assert stats.ks_2samp(n1.flat, n2.flat).pvalue > 0.05
    # compare Laplacian of field, which should be uncorrelated
    laplace = grid.make_operator("laplace", bc="periodic")
    assert stats.ks_2samp(laplace(n1).flat, laplace(n2).flat).pvalue > 0.05


def test_colored_noise_scaling(rng):
    """Compare the noise strength (in terms of the spectral density of two different
    noise sources that should be equivalent)"""
    # create a grid
    x, w = 2 + 10 * rng.random(2)
    size = rng.integers(128, 256)
    grid = CartesianGrid([[x, x + w]], size, periodic=True)

    # colored noise
    noise_colored = make_colored_noise(
        grid.shape, dx=grid.discretization[0], exponent=2
    )

    # divergence of white noise
    shape = (grid.dim,) + grid.shape
    div = grid.make_operator("divergence", bc="auto_periodic_neumann")

    def noise_div():
        return div(rng.normal(size=shape)) / (2 * np.pi)

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


def test_correlated_noise_scaling(rng):
    """Compare the noise strength (in terms of the spectral density of two different
    noise sources that should be equivalent)"""
    # create a grid
    x, w = 2 + 10 * rng.random(2)
    size = rng.integers(128, 256)
    grid = CartesianGrid([[x, x + w]], size, periodic=True)

    corr_length = 2
    noise_corr = make_correlated_noise(
        grid.shape,
        correlation="gaussian",
        discretization=grid.discretization,
        length_scale=corr_length,
    )

    # get spectral density
    k, density = spectral_density(data=noise_corr(), dx=grid.discretization[0])
    assert k[0] == 0
    assert density[0] == pytest.approx(0)

    # compare to expectation
    expect = np.exp(-0.5 * ((corr_length * k) ** 2))
    i = (expect > 1e-15) & (density > 1e-15)
    x, y = np.log(density[i]), np.log(expect[i])
    assert np.mean((x - y) ** 2) < 0.02 * np.mean(x**2)


def test_correlated_noise_cosine(rng):
    """Test noise with cosine correlation function."""
    grid = CartesianGrid([[0, 100]], 516, periodic=True)
    length_scale = 20
    noise_corr = make_correlated_noise(
        grid.shape,
        correlation="cosine",
        discretization=grid.discretization,
        length_scale=length_scale,
        rng=rng,
    )

    # get spectral density
    k, density = spectral_density(data=noise_corr(), dx=grid.discretization[0])
    k_max = k[np.argmax(density)]
    assert k_max == pytest.approx(1 / length_scale)
