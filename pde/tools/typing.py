"""
Provides support for mypy type checking of the package

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Any, Protocol, Sequence, Tuple, Union

import numpy as np

Real = Union[int, float]
Number = Union[Real, complex]
NumberOrArray = Union[Number, np.ndarray]
FloatNumerical = Union[float, np.ndarray]
ArrayLike = Union[NumberOrArray, Sequence[NumberOrArray], Sequence[Sequence[Any]]]


class OperatorType(Protocol):
    def __call__(self, arr: np.ndarray, out: np.ndarray) -> None:
        pass


class CellVolume(Protocol):
    def __call__(self, *args: int) -> float:
        pass


class GhostCellSetter(Protocol):
    def __call__(self, data_all: np.ndarray, args=None) -> None:
        pass


class AdjacentEvaluator(Protocol):
    def __call__(
        self, arr_1d: np.ndarray, i_point: int, bc_idx: Tuple[int, ...]
    ) -> float:
        pass


class VirtualPointEvaluator(Protocol):
    def __call__(self, arr: np.ndarray, idx: Tuple[int, ...], args=None) -> float:
        pass
