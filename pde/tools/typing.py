"""Provides support for mypy type checking of the package.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Protocol, Union

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from ..grids.base import GridBase

# types for single numbers:
Real = Union[int, float]  # a real number (no complex number allowed)
Number = Union[Real, complex, np.number]  # any number, including complex numbers

# array types:
NumericArray = np.ndarray[Any, np.dtype[np.number]]  # array of numbers (incl complex)
NumberOrArray = Union[Number, NumericArray]  # number or array of numbers (incl complex)
# a floating number or an array of floating (no integers and no complex numbers)
FloatingArray = np.ndarray[Any, np.dtype[np.floating]]
FloatOrArray = Union[float, np.ndarray[Any, np.dtype[np.floating]]]

# miscellaneous types:
BackendType = Literal["auto", "numpy", "numba"]


class OperatorType(Protocol):
    """An operator that acts on an array."""

    def __call__(self, arr: NumericArray, out: NumericArray) -> None:
        """Evaluate the operator."""


class OperatorFactory(Protocol):
    """A factory function that creates an operator for a particular grid."""

    def __call__(self, grid: GridBase, **kwargs) -> OperatorType:
        """Create the operator."""


class CellVolume(Protocol):
    def __call__(self, *args: int) -> float:
        """Calculate the volume of the cell at the given position."""


class VirtualPointEvaluator(Protocol):
    def __call__(self, arr: NumericArray, idx: tuple[int, ...], args=None) -> float:
        """Evaluate the virtual point at the given position."""


class GhostCellSetter(Protocol):
    def __call__(self, data_full: NumericArray, args=None) -> None:
        """Set the ghost cells."""


class StepperHook(Protocol):
    def __call__(
        self, state_data: NumericArray, t: float, post_step_data: NumericArray
    ) -> None:
        """Function analyzing and potentially modifying the current state."""
