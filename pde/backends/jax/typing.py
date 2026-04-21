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
    def __call__(self, arr: Array, idx: tuple[int | slice, ...], args=None) -> Array:
        """Evaluate the virtual point at the given position.

        Args:
            arr: Data array
            idx: Index tuple
            args: Additional arguments (optional)
        """


class JaxInnerStepperType(Protocol):
    """General backend-level stepping-function type working with jax arrays."""

    def __call__(
        self, state_data: Array, t_start: float, t_end: float
    ) -> tuple[Array, float]:
        """Advance the state given as a jax array.

        Args:
            state_data (:class:`~jax.Array`):
                The current state
            t_start (float):
                Initial time point
            t_end (float):
                Desired final time point

        Returns:
            tuple of the state and time at the final point
        """
