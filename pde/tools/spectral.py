"""Functions making use of spectral decompositions.

.. autosummary::
   :nosignatures:

   make_colored_noise
   make_correlated_noise

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging
import warnings
from typing import Callable, Literal

import numpy as np

from .typing import NumberOrArray

try:
    from pyfftw.interfaces.numpy_fft import irfftn as np_irfftn
    from pyfftw.interfaces.numpy_fft import rfftn as np_rfftn
except ImportError:
    from numpy.fft import irfftn as np_irfftn
    from numpy.fft import rfftn as np_rfftn


_logger = logging.getLogger(__name__)
""":class:`logging.Logger`: Logger instance."""


def _make_isotropic_correlated_noise(
    shape: tuple[int, ...],
    corr_spectrum: Callable[[np.ndarray], np.ndarray],
    *,
    discretization: NumberOrArray = 1.0,
    rng: np.random.Generator | None = None,
) -> Callable[[], np.ndarray]:
    r"""Return a function creating an array of random values with spatial correlations.

    An ensemble average (calling the returned function) multiple times produces normal
    distributed variables for each entry in the field. Different entries of the same
    field have a correlation function defined by `corr_spectrum`.

    Args:
        shape (tuple of ints):
            Number of supports points in each spatial dimension. The number of the list
            defines the spatial dimension.
        corr_spectrum (callable):
            Implementation of the Fourier transform of the correlation function, i.e.,
            the square root of its power spectrum. The arguments of the function are the
            squared wave vectors, ensuring isotropy.
        discretization (float or list of floats):
            Discretization along each dimension. A uniform discretization in each
            direction can be indicated by a single number.
        rng (:class:`~numpy.random.Generator`):
            Random number generator (default: :func:`~numpy.random.default_rng()`)

    Returns:
        callable: a function returning a random realization
    """
    rng = np.random.default_rng(rng)

    # extract some information about the grid
    dim = len(shape)
    dx_arr = np.broadcast_to(discretization, (dim,))

    # prepare wave vectors
    k2s = np.array(0)
    for i in range(dim):
        if i == dim - 1:
            k = np.fft.rfftfreq(shape[i], dx_arr[i])
        else:
            k = np.fft.fftfreq(shape[i], dx_arr[i])
        k2s = np.add.outer(k2s, k**2)

    # scaling of all modes with k != 0
    k2s.flat[0] = 1
    scaling = np.sqrt(corr_spectrum(k2s))
    scaling.flat[0] = 0
    scaling /= scaling.mean()  # normalize scaling to correct the variance

    def noise_corr() -> np.ndarray:
        """Return array of correlated noise."""
        # initialize uncorrelated random field
        arr: np.ndarray = rng.normal(size=shape)
        # forward transform to frequency space
        arr = np_rfftn(arr, s=shape, axes=range(dim))
        # scale frequency according to transformed correlation function
        arr *= scaling
        # backward transform to return to real space
        return np_irfftn(arr, s=shape, axes=range(dim))  # type: ignore

    return noise_corr


CorrelationType = Literal["none", "gaussian", "power law", "cosine"]


def make_correlated_noise(
    shape: tuple[int, ...],
    correlation: CorrelationType,
    *,
    discretization: NumberOrArray = 1.0,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> Callable[[], np.ndarray]:
    r"""Return a function creating random values with specified spatial correlations.

    The returned field :math:`f` generally obeys a selected correlation function
    :math:`C(k)`. In Fourier space, we thus have

    .. math::
        \langle f(\boldsymbol k) f(\boldsymbol k’) \rangle =
            C(|\boldsymbol k|) \delta(\boldsymbol k-\boldsymbol k’)

    For simplicity, the correlations respect periodic boundary conditions.

    Args:
        shape (tuple of ints):
            Number of supports points in each spatial dimension. The number of the list
            defines the spatial dimension.
        correlation (str):
            Selects the correlation function used to make the correlated noise. Many of
            the options (described below) support additional parameters that can be
            supplied as keyword arguments.
        discretization (float or list of floats):
            Discretization along each dimension. A uniform discretization in each
            direction can be indicated by a single number.
        rng (:class:`~numpy.random.Generator`):
            Random number generator (default: :func:`~numpy.random.default_rng()`)
        **kwargs:
            Additional parameters can affect details of the correlation function

    .. table:: Supported correlation functions
        :widths: 20 80

        ================= ==============================================================
        Identifier        Correlation function
        ================= ==============================================================
        :code:`none`      No correlation, :math:`C(k) = \delta(k)`

        :code:`gaussian`  :math:`C(k) = \exp(\frac12 k^2 \lambda^2)` with the length
                          scale :math:`\lambda` set by argument :code:`length_scale`.

        :code:`power law` :math:`C(k) = k^{\nu/2}` with exponent :math:`\nu` set by
                          argument :code:`exponent`.

        :code:`cosine`    :math:`C(k) = \exp\bigl(-s^2(\lambda k - 1)^2\bigr)` with the
                          length scale :math:`\lambda` set by argument
                          :code:`length_scale`, whereas the sharpness parameter
                          :math:`s` is set by :code:`sharpness` and defaults to 10.
        ================= ==============================================================

    Returns:
        callable: a function returning a random realization
    """
    rng = np.random.default_rng(rng)

    if correlation == "none":
        # no correlation
        corr_spectrum = None

    elif correlation == "gaussian":
        # gaussian correlation function with length scale `length_scale`
        length_scale = kwargs.pop("length_scale", 1)
        if length_scale == 0:
            corr_spectrum = None
        else:

            def corr_spectrum(k2s):
                """Fourier transform of a Gaussian function."""
                return np.exp(-0.5 * length_scale**2 * k2s)

    elif correlation == "power law":
        # power law correlation function with `exponent`
        exponent = kwargs.pop("exponent", 0)
        if exponent == 0:
            corr_spectrum = None
        else:

            def corr_spectrum(k2s):
                """Fourier transform of a power law."""
                return k2s ** (exponent / 4)

    elif correlation == "cosine":
        # cosine correlation function with a dominant mode scale `length_scale`
        length_scale = kwargs.pop("length_scale", 1)
        sharpness = kwargs.pop("sharpness", 10)
        sharpness2 = sharpness**2

        def corr_spectrum(k2s):
            """Fourier transform of a function that has a dominant harmonic mode."""
            return np.exp(-sharpness2 * (length_scale * np.sqrt(k2s) - 1) ** 2)

    else:
        raise ValueError(f"Unknown correlation `{correlation}`")

    if kwargs:
        _logger.warning("Unused arguments: %s", kwargs.keys())

    if corr_spectrum is None:
        # fast case of uncorrelated white noise
        def noise_normal():
            """Return array of colored noise."""
            return rng.normal(size=shape)

        return noise_normal

    else:
        return _make_isotropic_correlated_noise(
            shape, corr_spectrum=corr_spectrum, discretization=discretization, rng=rng
        )


def make_colored_noise(
    shape: tuple[int, ...],
    dx=1.0,
    exponent: float = 0,
    scale: float = 1,
    rng: np.random.Generator | None = None,
) -> Callable[[], np.ndarray]:
    r"""Return a function creating an array of random values that obey.

    .. math::
        \langle c(\boldsymbol k) c(\boldsymbol k’) \rangle =
            \Gamma^2 |\boldsymbol k|^\nu \delta(\boldsymbol k-\boldsymbol k’)

    in spectral space on a Cartesian grid. The special case :math:`\nu = 0` corresponds
    to white noise. For simplicity, the correlations respect periodic boundary
    conditions.

    Args:
        shape (tuple of ints):
            Number of supports points in each spatial dimension. The number of the list
            defines the spatial dimension.
        dx (float or list of floats):
            Discretization along each dimension. A uniform discretization in each
            direction can be indicated by a single number.
        exponent:
            Exponent :math:`\nu` of the power spectrum
        scale:
            Scaling factor :math:`\Gamma` determining noise strength
        rng (:class:`~numpy.random.Generator`):
            Random number generator (default: :func:`~numpy.random.default_rng()`)

    Returns:
        callable: a function returning a random realization
    """
    # deprecated since 2025-04-04
    warnings.warn(
        "`make_colored_noise` is deprecated. Use `make_correlated_noise` with "
        "correlation='power law' instead",
        DeprecationWarning,
    )
    rng = np.random.default_rng(rng)

    # extract some information about the grid
    dim = len(shape)
    dx = np.broadcast_to(dx, (dim,))

    if exponent == 0:
        # fast case of white noise
        def noise_normal():
            """Return array of colored noise."""
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
    scaling = scale * k2s ** (exponent / 4)
    scaling.flat[0] = 0

    def noise_colored() -> np.ndarray:
        """Return array of colored noise."""
        # random field
        arr: np.ndarray = rng.normal(size=shape)

        # forward transform
        arr = np_rfftn(arr)

        # scale according to frequency
        arr *= scaling

        # backwards transform
        return np_irfftn(arr, s=shape, axes=range(dim))  # type: ignore

    return noise_colored
