"""
Functions making use of spectral decompositions


.. autosummary::
   :nosignatures:

   make_colored_noise


.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""


from typing import Callable, Tuple

import numpy as np

try:
    from pyfftw.interfaces.numpy_fft import irfftn as np_irfftn
    from pyfftw.interfaces.numpy_fft import rfftn as np_rfftn
except ImportError:
    from numpy.fft import irfftn as np_irfftn
    from numpy.fft import rfftn as np_rfftn


def make_colored_noise(
    shape: Tuple[int, ...],
    dx=1.0,
    exponent: float = 0,
    scale: float = 1,
    rng: np.random.Generator = None,
) -> Callable[[], np.ndarray]:
    r"""Return a function creating an array of random values that obey

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
        rng (:class:`~numpy.random.Generator`):
            Random number generator (default: :func:`~numpy.random.default_rng()`)

    Returns:
        callable: a function returning a random realization
    """
    if rng is None:
        rng = np.random.default_rng()

    # extract some information about the grid
    dim = len(shape)
    dx = np.broadcast_to(dx, (dim,))

    if exponent == 0:
        # fast case of white noise
        def noise_normal():
            """return array of colored noise"""
            return scale * rng.normal(size=shape)

        return noise_normal

    # deal with colored noise in the following

    # prepare wave vectors
    k2s = np.array(0)
    for i in range(dim):
        if i == dim - 1:
            k = np.fft.rfftfreq(shape[i], dx[i])
        else:
            k = np.fft.fftfreq(shape[i], dx[i])
        k2s = np.add.outer(k2s, k**2)

    # scaling of all modes with k != 0
    k2s.flat[0] = 1
    scaling = 2 * np.pi * scale * k2s ** (exponent / 4)
    scaling.flat[0] = 0

    # TODO: accelerate the FFT using the pyfftw package

    def noise_colored() -> np.ndarray:
        """return array of colored noise"""
        # random field
        arr: np.ndarray = rng.normal(size=shape)  # type: ignore

        # forward transform
        arr = np_rfftn(arr)

        # scale according to frequency
        arr *= scaling

        # backwards transform
        arr = np_irfftn(arr, shape)
        return arr

    return noise_colored
