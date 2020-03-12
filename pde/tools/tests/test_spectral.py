'''
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

import pytest
import numpy as np
from scipy import stats, fftpack

from ..spectral import make_colored_noise
from ...grids import UnitGrid, CartesianGrid



def spectral_density(data, dx=1.):
    """ calculate the power spectral density of a field
    
    Args:
        data (:class:`numpy.ndarray`):
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

    return np.sqrt(k2s), np.abs(res)**2
    

                            
def test_colored_noise():
    """ test the implementation of the colored noise """
    grid = UnitGrid([64, 64], periodic=True)
    for exponent in [0, -1, 2]:
        scale = np.random.uniform(1, 10)
        noise = make_colored_noise(grid.shape, grid.discretization,
                                   exponent, scale)
        x = noise()
        assert stats.normaltest(x.flat).pvalue > 2e-5, \
               f'Colored noise with exp={exponent} is not normal distributed'
        
        
        
def test_noise_scaling():
    """ compare the noise strength (in terms of the spectral density of
    two different noise sources that should be equivalent) """
    # create a grid
    x, w = 2 + 10 * np.random.random(2)
    size = np.random.randint(128, 256)
    grid = CartesianGrid([[x, x + w]], size, periodic=True)
    
    # colored noise
    noise_colored = make_colored_noise(grid.shape, grid.discretization,
                                       exponent=2)
    
    # divergence of white noise
    shape = (grid.dim, ) + grid.shape
    div = grid.get_operator('divergence', bc='natural')
    
    def noise_div():
        return div(np.random.randn(*shape))
    
    # calculate spectral densities of the two noises
    result = []
    for noise_func in [noise_colored, noise_div]:
        def get_noise():
            k, density = spectral_density(data=noise_func(),
                                          dx=grid.discretization)
            assert k[0] == 0
            assert density[0] == pytest.approx(0)
            return np.log(density[1])  # log of spectral density
        
        # average spectral density of longest length scale
        mean = np.mean([get_noise() for _ in range(64)])
        result.append(mean)
        
    np.testing.assert_allclose(*result, rtol=0.5)
