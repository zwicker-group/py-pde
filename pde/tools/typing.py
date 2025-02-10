"""Provides support for mypy type checking of the package.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, Union

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from ..grids.base import GridBase

Real = Union[int, float]
Number = Union[Real, complex]
NumberOrArray = Union[Number, np.ndarray]
FloatNumerical = Union[float, np.ndarray]
BackendType = Literal["auto", "numpy", "numba"]


class OperatorType(Protocol):
    """An operator that acts on an array."""

    def __call__(self, arr: np.ndarray, out: np.ndarray) -> None:
        """Evaluate the operator."""


class OperatorFactory(Protocol):
    """A factory function that creates an operator for a particular grid."""

    def __call__(self, grid: GridBase, **kwargs) -> OperatorType:
        """Create the operator."""


class CellVolume(Protocol):
    def __call__(self, *args: int) -> float:
        """Calculate the volume of the cell at the given position."""


class VirtualPointEvaluator(Protocol):
    def __call__(self, arr: np.ndarray, idx: tuple[int, ...], args=None) -> float:
        """Evaluate the virtual point at the given position."""


class GhostCellSetter(Protocol):
    def __call__(self, data_full: np.ndarray, args=None) -> None:
        """Set the ghost cells."""


class StepperHook(Protocol):
    def __call__(
        self, state_data: np.ndarray, t: float, post_step_data: np.ndarray
    ) -> None:
        """Function analyzing and potentially modifying the current state."""
