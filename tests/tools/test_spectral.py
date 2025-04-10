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

    res = fftpack.fftn(data) / data.size

    return np.sqrt(k2s), np.abs(res) ** 2


def test_colored_noise(rng):
    """Test the implementation of the colored noise."""
    grid = CartesianGrid([[0, 32], [0, 32]], [16, 64], periodic=True)
    for exponent in [0, -1, 2]:
        scale = rng.uniform(1, 10)
        noise = make_colored_noise(
            grid.shape, dx=grid.discretization[0], exponent=exponent, rng=rng
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


@pytest.mark.parametrize("correlation", ["none", "gaussian", "power law"])
def test_correlated_gaussian_statistics(correlation, rng):
    """Test whether each points in the noise field overall have Gaussian statistics."""
    grid = CartesianGrid([[0, 20], [0, 20]], [32, 32], periodic=True)

    noise = make_correlated_noise(grid.shape, correlation=correlation, rng=rng)
    samples = np.ravel([noise() for _ in range(1000)])

    pvalue = stats.kstest(samples, "norm").pvalue
    assert pvalue > 0.05, f"DISTRIBUTION: {samples.mean():.5g} ± {samples.std():.5g}"


def test_gaussian_correlation(rng):
    """Test the implementation of the Gaussian correlated noise."""
    noise = make_correlated_noise(
        (32, 32), correlation="gaussian", length_scale=1e-10, rng=rng
    )
    n1 = noise()
    n2 = rng.normal(size=(32, 32))

    assert stats.ks_2samp(n1.flat, n2.flat).pvalue > 0.05
    # compare Laplacian of field, which should be uncorrelated
    laplace = UnitGrid([32, 32], periodic=True).make_operator("laplace", bc="periodic")
    assert stats.ks_2samp(laplace(n1).flat, laplace(n2).flat).pvalue > 0.05

    # create a grid
    x, w = 2 + 10 * rng.random(2)
    size = rng.integers(128, 256)
    grid = CartesianGrid([[x, x + w]], size, periodic=True)
    grid = CartesianGrid([[0, 12.8]], 128, periodic=True)
    dx = grid.discretization[0]

    corr_length = 5
    noise_corr = make_correlated_noise(
        grid.shape,
        correlation="gaussian",
        discretization=dx,
        length_scale=corr_length,
        rng=rng,
    )

    # get spectral density
    k = spectral_density(noise_corr(), dx=dx)[0]
    density = np.mean(
        [spectral_density(noise_corr(), dx=dx)[1] for _ in range(128)],
        axis=0,
    )
    assert k[0] == 0
    assert density[0] == pytest.approx(0)

    # compare to expectation
    expect = np.exp(-0.5 * ((corr_length * k) ** 2)) * corr_length / np.sqrt(2 * np.pi)
    i = (expect > 1e-15) & (density > 1e-15)  # remove super small values
    i[0] = False
    np.testing.assert_allclose(
        np.log(density[i]), np.log(expect[i]), atol=0.5, rtol=0.1
    )


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


def test_cosine_correlation(rng):
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
