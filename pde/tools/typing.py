"""
Provides support for mypy type checking of the package

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import TYPE_CHECKING, Protocol, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike  # @UnusedImport

if TYPE_CHECKING:
    from ..grids.base import GridBase  # @UnusedImport

Real = Union[int, float]
Number = Union[Real, complex]
NumberOrArray = Union[Number, np.ndarray]
FloatNumerical = Union[float, np.ndarray]


class OperatorType(Protocol):
    """an operator that acts on an array"""

    def __call__(self, arr: np.ndarray, out: np.ndarray) -> None:
        ...


class OperatorFactory(Protocol):
    """a factory function that creates an operator for a particular grid"""

    def __call__(self, grid: "GridBase", **kwargs) -> OperatorType:
        ...


class AdjacentEvaluator(Protocol):
    def __call__(
        self, arr_1d: np.ndarray, i_point: int, bc_idx: Tuple[int, ...]
    ) -> float:
        ...


class CellVolume(Protocol):
    def __call__(self, *args: int) -> float:
        ...


class GhostCellSetter(Protocol):
    def __call__(self, data_full: np.ndarray, args=None) -> None:
        ...


class VirtualPointEvaluator(Protocol):
    def __call__(self, arr: np.ndarray, idx: Tuple[int, ...], args=None) -> float:
        ...
