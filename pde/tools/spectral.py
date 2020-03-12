'''
Functions making use of spectral decompositions


.. autosummary::
   :nosignatures:

   make_colored_noise


.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''


from typing import Callable, Tuple

import numpy as np

try:
    from pyfftw.interfaces.numpy_fft import rfftn as np_rfftn
    from pyfftw.interfaces.numpy_fft import irfftn as np_irfftn
except ImportError:
    from numpy.fft import rfftn as np_rfftn
    from numpy.fft import irfftn as np_irfftn



def make_colored_noise(shape: Tuple[int, ...],
                       dx=1.,
                       exponent: float = 0,
                       scale: float = 1) -> Callable:
    r""" Return a function creating an array of random values that obey
    
    .. math::
        \langle c(\boldsymbol k) c(\boldsymbol k’) \rangle =
            \Gamma^2 |\boldsymbol k|^\nu \delta(\boldsymbol k-\boldsymbol k’)
            
    in spectral space on a Cartesian grid. The special case :math:`\nu = 0`
    corresponds to white noise.
    
    Args:
        shape (tuple of ints):
            Number of supports points in each spatial dimension. The number of
            the list defines the spatial dimension.
        dx (float or list of floats):
            Discretization along each dimension. A uniform discretization in
            each direction can be indicated by a single number.
        exponent:
            Exponent :math:`\nu` of the power spectrum
        scale:
            Scaling factor :math:`\Gamma` determining noise strength
        
    Returns:
        callable: a function returning a random realization
    """
    # extract some information about the grid    
    dim = len(shape)
    dx = np.broadcast_to(dx, (dim,))

    if exponent == 0:
        # fast case of white noise
        def noise_normal():
            """ return array of colored noise """
            return scale * np.random.randn(*shape)
        return noise_normal

    # deal with colored noise in the following

    # prepare wave vectors
    k2s = 0
    for i in range(dim):
        if i == dim - 1:
            k = np.fft.rfftfreq(shape[i], dx[i])
        else:
            k = np.fft.fftfreq(shape[i], dx[i])
        k2s = np.add.outer(k2s, k**2)

    # scaling of all modes with k != 0
    k2s.flat[0] = 1  # type: ignore
    scaling = 2 * np.pi * scale * k2s ** (exponent / 4)
    scaling.flat[0] = 0
    
    # TODO: accelerate the FFT using the pyfftw package

    def noise_colored():
        """ return array of colored noise """
        # random field
        arr = np.random.randn(*shape)
        
        # forward transform
        arr = np_rfftn(arr)
        
        # scale according to frequency
        arr *= scaling

        # backwards transform
        arr = np_irfftn(arr, shape)
        return arr
        
    return noise_colored

