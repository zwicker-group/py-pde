"""
Auxiliary mathematical functions

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Tuple

import numpy as np


class SmoothData1D:
    """allows smoothing data in 1d using a Gaussian kernel of defined width

    The data is given a pairs of `x` and `y`, the assumption being that there is
    an underlying relation `y = f(x)`.
    """

    sigma_auto_scale: float = 10
    """ float: scale for setting automatic values for sigma """

    def __init__(self, x, y, sigma: float = None):
        """initialize with data

        Args:
            x:
                List of x values
            y:
                List of y values
            sigma (float):
                The size of the smoothing window in units of `x`. If omitted, the
                average distance of x values multiplied by `sigma_auto_scale` is used.
        """
        self.x = np.ravel(x)
        self.y = np.ravel(y)
        if self.x.shape != self.y.shape:
            raise ValueError("`x` and `y` must have equal number of elements")

        # only take finite values
        idx = np.isfinite(self.y)
        if not np.all(idx):
            self.x = self.x[idx]
            self.y = self.y[idx]

        if sigma is None:
            self.sigma = float(self.sigma_auto_scale * self.x.ptp() / len(self.x))
        else:
            self.sigma = sigma

    @property
    def bounds(self) -> Tuple[float, float]:
        """return minimal and maximal `x` values"""
        return float(self.x.min()), float(self.x.max())

    def __contains__(self, x: float) -> bool:
        """checks whether the value `x` is contain in the range of x-values"""
        return self.x.min() <= x <= self.x.max()  # type: ignore

    def __call__(self, xs):
        """return smoothed y values for the positions given in `xs`

        Args:
            xs: a list of x-values

        Returns:
            :class:`~numpy.ndarray`: The associated y-values
        """
        xs = np.asanyarray(xs)
        shape = xs.shape
        xs = np.ravel(xs)
        scale = 0.5 * self.sigma**-2

        # determine the weights of all input points
        with np.errstate(under="ignore"):
            weight = np.exp(-scale * (self.x[:, None] - xs[None, :]) ** 2)
            weight_sum = weight.sum(axis=0)
            i = weight_sum > 0
            weight[..., i] /= weight_sum[i]

        result = self.y @ weight
        return result.reshape(shape)
