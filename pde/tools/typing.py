"""
Provides support for mypy type checking of the package

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, Union

import numpy as np
from numpy.typing import ArrayLike  # @UnusedImport

if TYPE_CHECKING:
    from ..grids.base import GridBase

Real = Union[int, float]
Number = Union[Real, complex]
NumberOrArray = Union[Number, np.ndarray]
FloatNumerical = Union[float, np.ndarray]
BackendType = Literal["auto", "numpy", "numba"]


class OperatorType(Protocol):
    """an operator that acts on an array"""

    def __call__(self, arr: np.ndarray, out: np.ndarray) -> None:
        """evaluate the operator"""


class OperatorFactory(Protocol):
    """a factory function that creates an operator for a particular grid"""

    def __call__(self, grid: GridBase, **kwargs) -> OperatorType:
        """create the operator"""


class CellVolume(Protocol):
    def __call__(self, *args: int) -> float:
        """calculate the volume of the cell at the given position"""


class VirtualPointEvaluator(Protocol):
    def __call__(self, arr: np.ndarray, idx: tuple[int, ...], args=None) -> float:
        """evaluate the virtual point at the given position"""


class AdjacentEvaluator(Protocol):
    def __call__(
        self, arr_1d: np.ndarray, i_point: int, bc_idx: tuple[int, ...]
    ) -> float:
        """evaluate the values at adjecent points"""


class GhostCellSetter(Protocol):
    def __call__(self, data_full: np.ndarray, args=None) -> None:
        """set the ghost cells"""
