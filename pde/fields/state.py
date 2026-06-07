"""Defines base class of fields or collections, which are discretized on grids.

.. autosummary::
   :nosignatures:

   StateBase

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from typing_extensions import Self

if TYPE_CHECKING:
    import logging
    from collections.abc import Callable

    from numpy.typing import DTypeLike

    from ..tools.typing import NumericArray, TState


class StateBase(metaclass=ABCMeta):
    """Abstract base class for describing numerical data."""

    _logger: logging.Logger  # logger instance to output information

    def __init__(self, data: NumericArray):
        """
        Args:
            data (:class:`~numpy.ndarray`):
                Data that is stored in this field
        """
        self._data = data

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_cache_methods", None)  # delete method cache if present
        return state

    @property
    def data(self) -> NumericArray:
        """:class:`~numpy.ndarray`: discretized data at the support points."""
        return self._data

    @data.setter
    def data(self, data: NumericArray) -> None:
        """Set the valid data of the field.

        Args:
            data:
                The value of the new data
        """
        self._data[:] = data

    @property
    def writeable(self) -> bool:
        """bool: whether the field data can be changed or not"""
        return self._data.flags.writeable

    @writeable.setter
    def writeable(self, value: bool) -> None:
        """Set whether the field data can be changed or not."""
        self._data.flags.writeable = value

    @abstractmethod
    def copy(
        self: TState, *, label: str | None = None, dtype: DTypeLike | None = None
    ) -> TState:
        """Return a new field with the data (but not the grid) copied.

        Args:
            label (str, optional):
                Name of the returned field
            dtype (numpy dtype):
                The data type of the field. If omitted, it will be determined from
                `data` automatically or the dtype of the current field is used.

        Returns:
            :class:`DataFieldBase`: A copy of the current field
        """

    @property
    def dtype(self) -> DTypeLike:
        """:class:`~DTypeLike`: the numpy dtype of the underlying data."""
        # this property is necessary to support np.iscomplexobj for DataFieldBases
        return self.data.dtype

    @property
    def is_complex(self) -> bool:
        """bool: whether the field contains real or complex data"""
        return np.iscomplexobj(self.data)

    @property
    def attributes(self) -> dict[str, Any]:
        """dict: describes the state of the instance (without the data)"""
        return {"class": self.__class__.__name__, "dtype": self.dtype}

    # @property
    # def attributes_serialized(self) -> dict[str, str]:
    #     """dict: serialized version of the attributes"""
    #     results = {}
    #     for key, value in self.attributes.items():
    #         if key == "grid":
    #             results[key] = value.state_serialized
    #         elif key == "dtype":
    #             results[key] = json.dumps(value.str)
    #         else:
    #             results[key] = json.dumps(value)
    #     return results

    # @classmethod
    # def unserialize_attributes(cls, attributes: dict[str, str]) -> dict[str, Any]:
    #     """Unserializes the given attributes.

    #     Args:
    #         attributes (dict):
    #             The serialized attributes

    #     Returns:
    #         dict: The unserialized attributes
    #     """
    #     # base class was chosen => select correct class from attributes
    #     class_name = json.loads(attributes["class"])

    #     if class_name == cls.__name__:
    #         msg = f"Cannot reconstruct abstract class `{class_name}`"
    #         raise RuntimeError(msg)

    #     # call possibly overwritten classmethod from subclass
    #     return cls._subclasses[class_name].unserialize_attributes(attributes)

    def __eq__(self, other):
        """Test fields for equality, ignoring the label."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return np.array_equal(self.data, other.data)

    def _unary_operation(self, op: Callable) -> Self:
        """Perform an unary operation on this field.

        Args:
            op (callable):
                A function calculating the result

        Returns:
            :class:`FieldBase`: An field that contains the result of the operation.
        """
        return self.__class__(data=op(self.data))

    @property
    def real(self) -> Self:
        """:class:`FieldBase`: Real part of the field."""
        return self._unary_operation(np.real)

    @property
    def imag(self) -> Self:
        """:class:`FieldBase`: Imaginary part of the field."""
        return self._unary_operation(np.imag)

    def conjugate(self) -> Self:
        """Returns complex conjugate of the field.

        Returns:
            :class:`FieldBase`: the complex conjugated field
        """
        return self._unary_operation(np.conjugate)

    def __neg__(self):
        """Return the negative of the current field.

        :class:`FieldBase`: The negative of the current field
        """
        return self._unary_operation(np.negative)

    @abstractmethod
    def plot(self, *args, **kwargs):
        """Visualize the field."""

    def split_mpi(
        self, decomposition: Literal["auto"] | int | list[int] = "auto"
    ) -> Self:
        """Splits the field onto subgrids in an MPI run.

        In a normal serial simulation, the method simply returns the field itself. In
        contrast, in an MPI simulation, the field provided on the main node is split
        onto all nodes using the given decomposition. The field data provided on all
        other nodes is not used.

        Args:
            decomposition (list of ints):
                Number of subdivision in each direction. Should be a list of length
                `grid.num_axes` specifying the number of nodes for this axis. If one
                value is `-1`, its value will be determined from the number of available
                nodes. The default value `auto` tries to determine an optimal
                decomposition by minimizing communication between nodes.

        Returns:
            :class:`FieldBase`: The part of the field that corresponds to the subgrid
            associated with the current MPI node.
        """
        raise NotImplementedError
