"""Provides support for mypy type checking of the module.

.. autosummary::
   :nosignatures:

   JaxOperatorType
   JaxDataSetter
   JaxGhostCellSetter
   JaxVirtualPointEvaluator

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from jax import Array


class JaxOperatorType(Protocol):
    """An operator that acts on an array."""

    def __call__(
        self,
        arr: Array,
        args: dict[str, Any] | None = None,
    ) -> Array:
        """Evaluate the operator.

        Args:
            arr: Input array
            args: Additional arguments (optional)

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


class JaxVirtualPointEvaluator(Protocol):
    def __call__(self, arr: Array, idx: tuple[int, ...], args=None) -> Array:
        """Evaluate the virtual point at the given position.

        Args:
            arr: Data array
            idx: Index tuple
            args: Additional arguments (optional)
        """
