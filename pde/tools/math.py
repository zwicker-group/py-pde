"""Auxiliary mathematical functions.

.. autosummary::
   :nosignatures:

   SmoothData1D
   OnlineStatistics

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import math
from typing import Any

import numba as nb
import numpy as np
from numba.experimental import jitclass

from .typing import ArrayLike


class SmoothData1D:
    """Allows smoothing data in 1d using a Gaussian kernel of defined width.

    The data is given a pairs of `x` and `y`, the assumption being that there is
    an underlying relation `y = f(x)`.
    """

    sigma_auto_scale: float = 10
    """float: scale for setting automatic values for sigma"""

    def __init__(self, x, y, sigma: float | None = None):
        """Initialize with data.

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
            self.sigma = float(self.sigma_auto_scale * np.ptp(self.x) / len(self.x))
        else:
            self.sigma = sigma

    @property
    def bounds(self) -> tuple[float, float]:
        """Return minimal and maximal `x` values."""
        return float(self.x.min()), float(self.x.max())

    def __contains__(self, x: float) -> bool:
        """Checks whether the value `x` is contain in the range of x-values."""
        return self.x.min() <= x <= self.x.max()  # type: ignore

    def __call__(self, xs: ArrayLike) -> np.ndarray:
        """Return smoothed y values for the positions given in `xs`

        Args:
            xs (list of :class:`~numpy.ndarray`): the x-values

        Returns:
            :class:`~numpy.ndarray`: The associated function values
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
        return result.reshape(shape)  # type: ignore

    def derivative(self, xs: ArrayLike) -> np.ndarray:
        """Return the derivative of the smoothed values for the positions `xs`

        Note that this value

        Args:
            xs (list of :class:`~numpy.ndarray`): the x-values

        Returns:
            :class:`~numpy.ndarray`: The associated values of the derivative
        """
        xs = np.asanyarray(xs)
        shape = xs.shape
        xs = np.ravel(xs)
        scale = 0.5 * self.sigma**-2

        # determine the weights of all input points
        with np.errstate(under="ignore"):
            ws = np.exp(-scale * (self.x[:, None] - xs[None, :]) ** 2)
            # first axis varies internal points where we know values
            # second axis gives positions at which we want to evaluate values
            ws_sum = ws.sum(axis=0)
            i = ws_sum > 0
            ws[..., i] /= ws_sum[i]

        diff = 2 * scale * (self.x[:, None] - xs[None, :])
        result = self.y @ (diff * ws) - (self.y @ ws) * (diff * ws).sum(axis=0)
        return result.reshape(shape)  # type: ignore


@jitclass(
    [
        ("min", nb.double),
        ("max", nb.double),
        ("mean", nb.double),
        ("_mean2", nb.double),
        ("count", nb.uint),
    ]
)
class OnlineStatistics:
    """Class for using an online algorithm for calculating statistics."""

    mean: float
    """float: recorded mean"""
    count: int
    """int: recorded number of items"""

    def __init__(self) -> None:
        self.min = np.inf
        self.max = -np.inf
        self.mean = 0
        self._mean2: float = 0
        self.count = 0

    @property
    def var(self) -> float:
        """float: recorded variance"""
        DDOF = 0
        if self.count <= DDOF:
            return math.nan
        else:
            return self._mean2 / (self.count - DDOF)

    @property
    def std(self) -> float:
        """float: recorded standard deviation"""
        return np.sqrt(self.var)  # type: ignore

    def add(self, value: float) -> None:
        """Add a value to the accumulator.

        Args:
            value (float): The value to add
        """
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        delta = value - self.mean
        self.count += 1
        self.mean += delta / self.count
        self._mean2 += delta * (value - self.mean)

    def to_dict(self) -> dict[str, Any]:
        """Return the information as a dictionary."""
        return {
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "std": self.std,
            "count": self.count,
        }
