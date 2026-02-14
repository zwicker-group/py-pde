"""Provides support for mypy type checking of the package.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, TypeVar

import numpy as np
from numpy.typing import ArrayLike  # noqa: F401

if TYPE_CHECKING:
    from torch import Tensor

    from ..fields import DataFieldBase, FieldCollection
    from ..grids.base import GridBase

# types for single numbers:
Real = int | float  # a real number (no complex number allowed)
Number = Real | complex | np.number  # any number, including complex numbers

# array types:
NumericArray = np.ndarray[Any, np.dtype[np.number]]  # array of numbers (incl complex)
NumberOrArray = Number | NumericArray  # number or array of numbers (incl complex)
# a floating number or an array of floating (no integers and no complex numbers)
FloatingArray = np.ndarray[Any, np.dtype[np.floating]]
FloatOrArray = float | np.ndarray[Any, np.dtype[np.floating]]

# generic array types that work for various fields or arrays
TField = TypeVar("TField", "FieldCollection", "DataFieldBase")
TArray = TypeVar("TArray", NumericArray, "Tensor")

# miscellaneous types:
BackendType = Literal["scipy", "numpy", "numba", "numba_mpi", "torch"]


class OperatorInfo(NamedTuple):
    """Stores information about an operator."""

    factory: OperatorFactory
    rank_in: int
    rank_out: int
    name: str = ""  # attach a unique name to help caching


class OperatorImplType(Protocol):
    """An operator that acts on an array."""

    def __call__(self, arr: TArray, out: TArray) -> None:
        """Evaluate the operator.
        
        Args:
            arr: Input array
            out: Output array
        """


class OperatorFactory(Protocol):
    """A factory function that creates an operator for a particular grid."""

    def __call__(self, grid: GridBase, **kwargs) -> OperatorImplType:
        """Create the operator.
        
        Args:
            grid: The grid for which the operator is created
            **kwargs: Additional keyword arguments
        """


class OperatorType(Protocol):
    """An operator that acts on an array."""

    def __call__(
        self,
        arr: TArray,
        out: TArray | None = None,
        args: dict[str, Any] | None = None,
    ) -> TArray:
        """Evaluate the operator.
        
        Args:
            arr: Input array
            out: Output array (optional)
            args: Additional arguments (optional)
        """


class CellVolume(Protocol):
    def __call__(self, *args: int) -> float:
        """Calculate the volume of the cell at the given position.
        
        Args:
            *args: Position indices
        """


class VirtualPointEvaluator(Protocol):
    def __call__(self, arr: NumericArray, idx: tuple[int, ...], args=None) -> float:
        """Evaluate the virtual point at the given position.
        
        Args:
            arr: Data array
            idx: Index tuple
            args: Additional arguments (optional)
        """


class GhostCellSetter(Protocol):
    def __call__(self, data_full: NumericArray, args=None) -> None:
        """Set the ghost cells.
        
        Args:
            data_full: Full data array including ghost cells
            args: Additional arguments (optional)
        """


class DataSetter(Protocol):
    def __call__(
        self, data_full: NumericArray, data_valid: NumericArray, args=None
    ) -> None:
        """Set the valid data cells (and potentially BCs).
        
        Args:
            data_full: Full data array including ghost cells
            data_valid: Valid data array
            args: Additional arguments (optional)
        """


class StepperHook(Protocol):
    def __call__(
        self, state_data: NumericArray, t: float, post_step_data: NumericArray
    ) -> None:
        """Function analyzing and potentially modifying the current state.
        
        Args:
            state_data: Current state data
            t: Current time
            post_step_data: Data after the step
        """
