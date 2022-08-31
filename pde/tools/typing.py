"""
Provides support for mypy type checking of the package

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from typing import Callable, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike  # @UnusedImport

Real = Union[int, float]
Number = Union[Real, complex]
NumberOrArray = Union[Number, np.ndarray]
FloatNumerical = Union[float, np.ndarray]


try:
    from typing import Protocol

except ImportError:
    # Protocol is not defined (likely because python version <= 3.7
    # We demand python version 3.8 since 2022-08-23
    # This fallback can thus be removed after 2022-02-23
    OperatorType = Callable[[np.ndarray, np.ndarray], None]
    AdjacentEvaluator = Callable[[np.ndarray, int, Tuple[int, ...]], float]
    CellVolume = Callable[..., float]
    GhostCellSetter = Callable[..., None]
    VirtualPointEvaluator = Callable[..., float]

else:
    # Protocol is defined

    class OperatorType(Protocol):  # type: ignore
        def __call__(self, arr: np.ndarray, out: np.ndarray) -> None:
            pass

    class AdjacentEvaluator(Protocol):  # type: ignore
        def __call__(
            self, arr_1d: np.ndarray, i_point: int, bc_idx: Tuple[int, ...]
        ) -> float:
            pass

    class CellVolume(Protocol):  # type: ignore
        def __call__(self, *args: int) -> float:
            pass

    class GhostCellSetter(Protocol):  # type: ignore
        def __call__(self, data_full: np.ndarray, args=None) -> None:
            pass

    class VirtualPointEvaluator(Protocol):  # type: ignore
        def __call__(self, arr: np.ndarray, idx: Tuple[int, ...], args=None) -> float:
            pass
