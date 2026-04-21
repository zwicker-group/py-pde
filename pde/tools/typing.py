"""Provides support for mypy type checking of the package.

.. autosummary::
   :nosignatures:

   OperatorInfo
   OperatorImplType
   OperatorFactory
   OperatorType
   CellVolume
   VirtualPointEvaluator
   GhostCellSetter
   DataSetter
   StepperHook

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, TypeVar, Union

import numpy as np
from numpy.typing import ArrayLike  # noqa: F401

if TYPE_CHECKING:
    from jax import Array
    from torch import Tensor

    from ..fields import DataFieldBase, FieldCollection
    from ..grids.base import GridBase

# types for single numbers:
Real = int | float  # a real number (no complex number allowed)
Number = Real | complex | np.number  # any number, including complex numbers

# array types:
NumericArray = np.ndarray[Any, np.dtype[np.number]]  # array of int, real, or complex
NumberOrArray = Number | NumericArray  # number or array of numbers (incl complex)
InexactArray = np.ndarray[Any, np.dtype[np.inexact]]  # array of real or complex numbers
# a floating number or an array of floating (no integers and no complex numbers)
FloatingArray = np.ndarray[Any, np.dtype[np.floating]]  # array of real numbers
FloatOrArray = float | np.ndarray[Any, np.dtype[np.floating]]

# generic array types that work for various fields or arrays
TField = TypeVar("TField", "FieldCollection", "DataFieldBase")
# the following generic array type also supports torch.Tensor and jax.Array
NativeArray = Union[NumericArray, "Tensor", "Array"]
TNativeArray = TypeVar("TNativeArray", NumericArray, "Tensor", "Array")

# generic array types that work for various fields or arrays
TFunc = TypeVar("TFunc", bound=Callable)


class OperatorInfo(NamedTuple):
    """Stores information about an operator."""

    factory: OperatorFactory
    rank_in: int
    rank_out: int
    name: str = ""  # attach a unique name to help caching


# operators act on an array and either return result or write it into supplied array
OperatorImplType = (
    Callable[[TNativeArray], TNativeArray]
    | Callable[[TNativeArray, TNativeArray], TNativeArray]
)
BinaryOperatorImplType = Callable[
    [TNativeArray, TNativeArray, TNativeArray | None], TNativeArray
]


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
        arr: TNativeArray,
        out: TNativeArray | None = None,
        args: dict[str, Any] | None = None,
    ) -> TNativeArray:
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


class PostStepHook(Protocol):
    def __call__(self, state_data: TNativeArray, t: float) -> TNativeArray:
        """Function analyzing and potentially modifying the current state.

        Args:
            state_data: Current state data
            t: Current time

        Returns:
            state_data: Can be the same arrays as the input
        """


class StepperHook(Protocol):
    def __call__(
        self, state_data: TNativeArray, t: float, post_step_data: Any
    ) -> tuple[TNativeArray, Any]:
        """Function analyzing and potentially modifying the current state.

        Args:
            state_data: Current state data
            t: Current time
            post_step_data: Data to be stored for later analysis

        Returns:
            tuple(state_data, post_step_data): Can be the same arrays as the input
        """


class StepperType(Protocol):
    """General stepping-function type working with py-pde fields.

    Instances of this protocol are typically created by solver objects.
    """

    def __call__(self, state: TField, t_start: float, t_end: float) -> float:
        """Advance the state given as a field.

        Args:
            state (:class:`~pde.fields.base.FieldBase`):
                The state, which will be updated in-place
            t_start (float):
                Initial time point
            t_end (float):
                Desired final time point

        Returns:
            float: the actual final time point
        """


class InnerStepperType(Protocol):
    """General backend-level stepping-function type working with numpy arrays."""

    def __call__(self, state_data: NumericArray, t_start: float, t_end: float) -> float:
        """Advance the state given as a numpy array.

        Args:
            state_data (:class:`~numpy.ndarray`):
                The state, which will be updated in-place
            t_start (float):
                Initial time point
            t_end (float):
                Desired final time point

        Returns:
            float: the actual final time point
        """
