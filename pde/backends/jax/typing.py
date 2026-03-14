"""Provides support for mypy type checking of the module.

.. autosummary::
   :nosignatures:

   JaxOperatorImplType

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from numpy.typing import ArrayLike  # noqa: F401

if TYPE_CHECKING:
    from jax import Array


class JaxOperatorImplType(Protocol):
    """An operator that acts on an array."""

    def __call__(self, arr: Array) -> Array:
        """Evaluate the operator.

        Args:
            arr: Input array

        Returns:
            Output array
        """


class JaxDataSetter(Protocol):
    def __call__(self, data_valid: Array, args=None) -> Array:
        """Set the valid data cells (and potentially BCs).

        Args:
            data_valid: Valid data array
            args: Additional arguments (optional)

        Returns:
            Full data array including ghost cells
        """


class JaxGhostCellSetter(Protocol):
    def __call__(self, data_full: Array, args=None) -> Array:
        """Set the ghost cells.

        Args:
            data_full: Full data array including ghost cells
            args: Additional arguments (optional)
        """


class VirtualPointEvaluator(Protocol):
    def __call__(self, arr: Array, idx: tuple[int, ...], args=None) -> float:
        """Evaluate the virtual point at the given position.

        Args:
            arr: Data array
            idx: Index tuple
            args: Additional arguments (optional)
        """
