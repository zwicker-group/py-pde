"""Defines base class of fields or collections, which are discretized on grids.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import json
import logging
import warnings
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar

import numpy as np
from numpy.typing import DTypeLike

from ..grids.base import GridBase
from ..tools.plotting import napari_add_layers, napari_viewer
from ..tools.typing import NumberOrArray

if TYPE_CHECKING:
    from .scalar import ScalarField


_base_logger = logging.getLogger(__name__.rsplit(".", 1)[0])
""":class:`logging.Logger`: Base logger for fields."""

TField = TypeVar("TField", bound="FieldBase")


class RankError(TypeError):
    """Error indicating that the field has the wrong rank."""


class FieldBase(metaclass=ABCMeta):
    """Abstract base class for describing (discretized) fields."""

    _subclasses: dict[str, type[FieldBase]] = {}  # all classes inheriting from this
    _grid: GridBase  # the grid on which the field is defined
    __data_full: np.ndarray  # the data on the grid including ghost points
    _data_valid: np.ndarray  # the valid data without ghost points
    _label: str | None  # name of the field
    _logger: logging.Logger  # logger instance to output information

    def __init__(
        self,
        grid: GridBase,
        data: np.ndarray,
        *,
        label: str | None = None,
    ):
        """
        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                Grid defining the space on which this field is defined
            data (:class:`~numpy.ndarray`, optional):
                Field values at the support points of the grid and the ghost cells
            label (str, optional):
                Name of the field
        """
        self._grid = grid
        self._data_full = data
        self.label = label

    def __init_subclass__(cls, **kwargs):
        """Initialize class-level attributes of subclasses."""
        super().__init_subclass__(**kwargs)

        # create logger for this specific field class
        cls._logger = _base_logger.getChild(cls.__qualname__)

        # register all subclasses to reconstruct them later
        if cls is not FieldBase:
            if cls.__name__ in cls._subclasses:
                warnings.warn(f"Redefining class {cls.__name__}")
            cls._subclasses[cls.__name__] = cls

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state.pop("_cache_methods", None)  # delete method cache if present
        return state

    @property
    def data(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: discretized data at the support points."""
        return self._data_valid

    @data.setter
    def data(self, value: NumberOrArray) -> None:
        """Set the valid data of the field.

        Args:
            value:
                The value of the valid data. If a scalar is supplied all data points get
                the same value. The value of ghost cells are not changed.
        """
        if isinstance(value, FieldBase):
            # copy data into current field
            self.assert_field_compatible(value, accept_scalar=True)
            self._data_valid[...] = value.data
        else:
            self._data_valid[...] = value

    @property
    def _idx_valid(self) -> tuple[slice, ...]:
        """tuple: slices to extract valid data from full data"""
        idx_comp = (slice(None),) * (self.__data_full.ndim - self.grid.num_axes)
        return idx_comp + self.grid._idx_valid

    @property
    def _data_full(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: the full data including ghost cells."""
        return self.__data_full

    @_data_full.setter
    def _data_full(self, value: NumberOrArray) -> None:
        """Set the full data including ghost cells.

        Args:
            value:
                The value of the full data including those for ghost cells. If a scalar
                is supplied all data points get the same value.
        """
        if not self.writeable:
            raise ValueError("assignment destination is read-only")

        if np.isscalar(value):
            # supplied value is a scalar
            self.__data_full[...] = value

        elif isinstance(value, np.ndarray):
            # check the shape of the supplied array
            if value.shape[-self.grid.num_axes :] != self.grid._shape_full:
                raise ValueError(
                    f"Supplied data has wrong shape: {value.shape} is not compatible "
                    f"with {self.grid._shape_full}"
                )
            # actually set the data
            self.__data_full = value

        else:
            raise TypeError(f"Cannot set field values to {value}")

        # set reference to valid data
        self._data_valid = self.__data_full[self._idx_valid]

    @property
    def _data_flat(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: flat version of discretized data with ghost
        cells."""
        # flatten the first dimension of the internal data by creating a view and then
        # setting the new shape. This disallows accidental copying of the data
        data_flat = self._data_full.view()
        data_flat.shape = (-1, *self.grid._shape_full)
        return data_flat

    @_data_flat.setter
    def _data_flat(self, value: np.ndarray) -> None:
        """Set the full data including ghost cells from a flattened array."""
        # simply set the data -> this might need to be overwritten
        self._data_full = value

    @property
    def writeable(self) -> bool:
        """bool: whether the field data can be changed or not"""
        return not hasattr(self, "_data_full") or self._data_full.flags.writeable

    @writeable.setter
    def writeable(self, value: bool) -> None:
        """Set whether the field data can be changed or not."""
        self._data_full.flags.writeable = value
        self._data_valid.flags.writeable = value

    @property
    def label(self) -> str | None:
        """str: the name of the field"""
        return self._label

    @label.setter
    def label(self, value: str | None = None):
        """Set the new label of the field."""
        if value is None or isinstance(value, str):
            self._label = value
        else:
            raise TypeError("Label must be a string or None")

    @classmethod
    def from_state(
        cls, attributes: dict[str, Any], data: np.ndarray | None = None
    ) -> FieldBase:
        """Create a field from given state.

        Args:
            attributes (dict):
                The attributes that describe the current instance
            data (:class:`~numpy.ndarray`, optional):
                Data values at the support points of the grid defining the field

        Returns:
            :class:`FieldBase`: The field created from the state
        """
        # base class was chosen => select correct class from attributes
        class_name = attributes.pop("class")

        if class_name == cls.__name__:
            raise RuntimeError(f"Cannot reconstruct abstract class `{class_name}`")

        # call possibly overwritten classmethod from subclass
        return cls._subclasses[class_name].from_state(attributes, data)

    @classmethod
    def from_file(cls, filename: str) -> FieldBase:
        """Create field from data stored in a file.

        Field can be written to a file using :meth:`FieldBase.to_file`.

        Example:
            Write a field to a file and then read it back:

            .. code-block:: python

                field = pde.ScalarField(...)
                field.write_to("test.hdf5")

                field_copy = pde.FieldBase.from_file("test.hdf5")

        Args:
            filename (str):
                Path to the file being read

        Returns:
            :class:`FieldBase`: The field with the appropriate sub-class
        """
        import h5py

        from .collection import FieldCollection

        with h5py.File(filename, "r") as fp:
            if "class" in fp.attrs:
                # this should be a field collection
                assert json.loads(fp.attrs["class"]) == "FieldCollection"
                obj = FieldCollection._from_hdf_dataset(fp)

            elif len(fp) == 1:
                # a single field is stored in the data
                dataset = fp[list(fp.keys())[0]]  # retrieve only dataset
                obj = cls._from_hdf_dataset(dataset)  # type: ignore

            else:
                raise RuntimeError(
                    "Multiple data fields were found in the file but no "
                    "`FieldCollection` is expected."
                )
        return obj

    @classmethod
    def _from_hdf_dataset(cls, dataset) -> FieldBase:
        """Construct a field by reading data from an hdf5 dataset."""
        # copy attributes from hdf
        attributes = dict(dataset.attrs)

        # determine class
        class_name = json.loads(attributes.pop("class"))
        field_cls = cls._subclasses[class_name]

        # unserialize the attributes
        attributes = field_cls.unserialize_attributes(attributes)
        return field_cls.from_state(attributes, data=dataset)

    @property
    def grid(self) -> GridBase:
        """:class:`~pde.grids.base,GridBase`: The grid on which the field is defined."""
        return self._grid

    def to_file(self, filename: str, **kwargs) -> None:
        r"""Store field in a file.

        The extension of the filename determines what format is being used. If it ends
        in `.h5` or `.hdf`, the Hierarchical Data Format is used. The other supported
        format are images, where only the most typical formats are supported.

        To load the field back from the file, you may use :meth:`FieldBase.from_file`.

        Example:
            Write a field to a file and then read it back:

            .. code-block:: python

                field = pde.ScalarField(...)
                field.write_to("test.hdf5")

                field_copy = pde.FieldBase.from_file("test.hdf5")

        Args:
            filename (str):
                Path where the data is stored
            \**kwargs:
                Additional parameters may be supported for some formats
        """
        extension = Path(filename).suffix.lower()

        if extension in {".hdf", ".hdf5", ".he5", ".h5"}:
            # save data in hdf5 format
            import h5py

            with h5py.File(filename, "w") as fp:
                self._write_hdf_dataset(fp, **kwargs)

        elif extension in {".png", ".jpg", ".jpeg", ".tif", ".pdf", ".svg"}:
            # save data as an image
            self._write_to_image(filename, **kwargs)

        else:
            raise ValueError(f"Do not know how to save data to `*{extension}`")

    def _write_hdf_dataset(self, hdf_path, key: str = "data") -> None:
        """Write data to a given hdf5 path `hdf_path`"""
        # write the data
        dataset = hdf_path.create_dataset(key, data=self.data)

        # write attributes
        for key, value in self.attributes_serialized.items():
            dataset.attrs[key] = value

    def _write_to_image(self, filename: str, **kwargs):
        """Write data to image.

        Args:
            filename (str): The path to the image that will be created
        """
        raise NotImplementedError(f"Cannot save {self.__class__.__name__} as an image")

    @abstractmethod
    def copy(
        self: TField, *, label: str | None = None, dtype: DTypeLike | None = None
    ) -> TField:
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

    def assert_field_compatible(
        self, other: FieldBase, accept_scalar: bool = False
    ) -> None:
        """Checks whether `other` is compatible with the current field.

        Args:
            other (FieldBase):
                The other field this one is compared to
            accept_scalar (bool, optional):
                Determines whether it is acceptable that `other` is an instance of
                :class:`~pde.fields.ScalarField`.
        """
        from .scalar import ScalarField

        # check whether they are the same class
        is_scalar = accept_scalar and isinstance(other, ScalarField)
        class_compatible = self.__class__ == other.__class__ or is_scalar
        if not class_compatible:
            raise TypeError(f"Fields {self} and {other} are incompatible")

        # check whether the associated grids are identical
        if not self.grid.compatible_with(other.grid):
            raise ValueError(f"Grids {self.grid} and {other.grid} are incompatible")

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
        return {
            "class": self.__class__.__name__,
            "grid": self.grid,
            "label": self.label,
            "dtype": self.dtype,
        }

    @property
    def attributes_serialized(self) -> dict[str, str]:
        """dict: serialized version of the attributes"""
        results = {}
        for key, value in self.attributes.items():
            if key == "grid":
                results[key] = value.state_serialized
            elif key == "dtype":
                results[key] = json.dumps(value.str)
            else:
                results[key] = json.dumps(value)
        return results

    @classmethod
    def unserialize_attributes(cls, attributes: dict[str, str]) -> dict[str, Any]:
        """Unserializes the given attributes.

        Args:
            attributes (dict):
                The serialized attributes

        Returns:
            dict: The unserialized attributes
        """
        # base class was chosen => select correct class from attributes
        class_name = json.loads(attributes["class"])

        if class_name == cls.__name__:
            raise RuntimeError(f"Cannot reconstruct abstract class `{class_name}`")

        # call possibly overwritten classmethod from subclass
        return cls._subclasses[class_name].unserialize_attributes(attributes)

    def __eq__(self, other):
        """Test fields for equality, ignoring the label."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.grid == other.grid and np.array_equal(self.data, other.data)

    def _unary_operation(self: TField, op: Callable) -> TField:
        """Perform an unary operation on this field.

        Args:
            op (callable):
                A function calculating the result

        Returns:
            :class:`FieldBase`: An field that contains the result of the operation.
        """
        return self.__class__(grid=self.grid, data=op(self.data), label=self.label)

    @property
    def real(self: TField) -> TField:
        """:class:`FieldBase`: Real part of the field."""
        return self._unary_operation(np.real)

    @property
    def imag(self: TField) -> TField:
        """:class:`FieldBase`: Imaginary part of the field."""
        return self._unary_operation(np.imag)

    def conjugate(self: TField) -> TField:
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

    def _binary_operation(
        self, other, op: Callable, scalar_second: bool = True
    ) -> FieldBase:
        """Perform a binary operation between this field and `other`

        Args:
            other (number of FieldBase):
                The second term of the operator
            op (callable):
                A binary function calculating the result
            scalar_second (bool):
                Flag determining whether the second operator must be a scalar

        Returns:
            :class:`FieldBase`: An field that contains the result of the operation. If
            `scalar_second == True`, the type of FieldBase is the same as `self`
        """
        # determine the dtype of the output

        if isinstance(other, FieldBase):
            # right operator is a field
            from .scalar import ScalarField

            # determine the dtype of the result of the operation
            dtype = np.result_type(self.data, other.data)

            if scalar_second:
                # right operator must be a scalar or scalar field
                if not isinstance(other, ScalarField):
                    raise TypeError("Right operator must be a scalar field")
                self.grid.assert_grid_compatible(other.grid)
                result: FieldBase = self.copy(dtype=dtype)

            elif isinstance(self, ScalarField):
                # left operator is a scalar field (right can be tensor)
                self.grid.assert_grid_compatible(other.grid)
                result = other.copy(dtype=dtype)

            else:
                # left operator is tensor and right one might be anything
                self.assert_field_compatible(other, accept_scalar=True)
                result = self.copy(dtype=dtype)

            op(self.data, other.data, out=result.data)

        else:
            # the second operator is a number or a numpy array
            dtype = np.result_type(self.data, other)
            result = self.copy(dtype=dtype)
            op(self.data, other, out=result.data)

        return result

    def _binary_operation_inplace(
        self: TField, other, op_inplace: Callable, scalar_second: bool = True
    ) -> TField:
        """Perform an in-place binary operation between this field and `other`

        Args:
            other (number of FieldBase):
                The second term of the operator
            op_inplace (callable):
                A binary function storing its result in the first argument
            scalar_second (bool):
                Flag determining whether the second operator must be a scalar.

        Returns:
            :class:`FieldBase`: The field `self` with updated data
        """
        if isinstance(other, FieldBase):
            # right operator is a field
            from .scalar import ScalarField

            if scalar_second:
                # right operator must be a scalar
                if not isinstance(other, ScalarField):
                    raise TypeError("Right operator must be a scalar field")
                self.grid.assert_grid_compatible(other.grid)
            else:
                # left operator is tensor and right one might be anything
                self.assert_field_compatible(other, accept_scalar=True)

            # operators only affect the valid data and do not touch the ghost cells
            op_inplace(self.data, other.data, out=self.data)

        else:
            # the second operator is a number or a numpy array
            op_inplace(self.data, other, out=self.data)

        return self

    def __add__(self, other) -> FieldBase:
        """Add two fields."""
        return self._binary_operation(other, np.add, scalar_second=False)

    __radd__ = __add__

    def __iadd__(self: TField, other) -> TField:
        """Add `other` to the current field."""
        return self._binary_operation_inplace(other, np.add, scalar_second=False)

    def __sub__(self, other) -> FieldBase:
        """Subtract two fields."""
        return self._binary_operation(other, np.subtract, scalar_second=False)

    def __rsub__(self, other) -> FieldBase:
        """Subtract two fields."""
        return self._binary_operation(
            other, lambda x, y, out: np.subtract(y, x, out=out), scalar_second=False
        )

    def __isub__(self: TField, other) -> TField:
        """Add `other` to the current field."""
        return self._binary_operation_inplace(other, np.subtract, scalar_second=False)

    def __mul__(self, other) -> FieldBase:
        """Multiply field by value."""
        return self._binary_operation(other, np.multiply, scalar_second=False)

    __rmul__ = __mul__

    def __imul__(self: TField, other) -> TField:
        """Multiply field by value."""
        return self._binary_operation_inplace(other, np.multiply, scalar_second=False)

    def __truediv__(self, other) -> FieldBase:
        """Divide field by value."""
        return self._binary_operation(other, np.true_divide, scalar_second=True)

    def __rtruediv__(self, other) -> FieldBase:
        """Divide field by value."""

        def rdivision(x, y, **kwargs):
            return np.true_divide(y, x, **kwargs)

        return self._binary_operation(other, rdivision, scalar_second=True)

    def __itruediv__(self: TField, other) -> TField:
        """Divide field by value."""
        return self._binary_operation_inplace(other, np.true_divide, scalar_second=True)

    def __pow__(self, exponent: float) -> FieldBase:
        """Raise data of the field to a certain power."""
        if not np.isscalar(exponent):
            raise NotImplementedError("Only scalar exponents are supported")
        return self._binary_operation(exponent, np.power, scalar_second=True)

    def __ipow__(self: TField, exponent: float) -> TField:
        """Raise data of the field to a certain power in-place."""
        if not np.isscalar(exponent):
            raise NotImplementedError("Only scalar exponents are supported")
        self.data **= exponent
        return self

    def apply(
        self: TField,
        func: Callable | str,
        out: TField | None = None,
        *,
        label: str | None = None,
        evaluate_args: dict[str, Any] | None = None,
    ) -> TField:
        """Applies a function/expression to the data and returns it as a field.

        Args:
            func (callable or str):
                The (vectorized) function being applied to the data or an expression
                that can be parsed using sympy (:func:`~pde.tools.expressions.evaluate`
                is used in this case). The local field values can be accessed using the
                field labels for a field collection and via the variable `c` otherwise.
            out (FieldBase, optional):
                Optional field into which the data is written
            label (str, optional):
                Name of the returned field
            evaluate_args (dict):
                Additional arguments passed to :func:`~pde.tools.expressions.evaluate`.
                Only used when `func` is a string.

        Returns:
            :class:`FieldBase`: Field with new data. Identical to `out` if given.
        """
        from .datafield_base import DataFieldBase  # avoid circular import

        if isinstance(func, str):
            # function is given as an expression that will be evaluated
            from ..tools.expressions import evaluate
            from .collection import FieldCollection

            if evaluate_args is None:
                evaluate_args = {}
            if isinstance(self, DataFieldBase):
                fields = {"c": self}
                if self.label is not None:
                    fields[self.label] = self
                result = evaluate(func, fields, **evaluate_args)
            elif isinstance(self, FieldCollection):
                result = evaluate(func, self, **evaluate_args)
            else:
                raise TypeError("self must be DataFieldBase or FieldCollection")

            if out is None:
                out = result  # type: ignore
            else:
                result.assert_field_compatible(out)
                out.data[...] = result.data

        elif callable(func):
            # function should directly be applied to the data
            if out is None:
                out = self.copy(label=label)
            else:
                self.assert_field_compatible(out)
            out.data[...] = func(self.data)

        else:
            raise TypeError("`func` must be string or callable")

        if not isinstance(out, FieldBase):
            raise TypeError("`out` must be of type `FieldBase`")
        if label:
            out.label = label
        return out  # type: ignore

    @abstractmethod
    def get_line_data(
        self, scalar: str = "auto", extract: str = "auto"
    ) -> dict[str, Any]:
        """Return data for a line plot of the field.

        Args:
            scalar (str or int):
                The method for extracting scalars as described in
                :meth:`DataFieldBase.to_scalar`.
            extract (str):
                The method used for extracting the line data. See the docstring
                of the grid method `get_line_data` to find supported values.

        Returns:
            dict: Information useful for performing a line plot of the field
        """

    @abstractmethod
    def get_image_data(self) -> dict[str, Any]:
        r"""Return data for plotting an image of the field.

        Args:
            scalar (str or int):
                The method for extracting scalars as described in
                :meth:`DataFieldBase.to_scalar`.
            transpose (bool):
                Determines whether the transpose of the data should is plotted
            \**kwargs:
                Additional parameters are forwarded to `grid.get_image_data`

        Returns:
            dict: Information useful for plotting an image of the field
        """

    @abstractmethod
    def plot(self, *args, **kwargs):
        """Visualize the field."""

    @abstractmethod
    def _get_napari_data(self, **kwargs) -> dict[str, dict[str, Any]]:
        """Returns data for plotting this field using :mod:`napari`"""

    def plot_interactive(
        self, viewer_args: dict[str, Any] | None = None, **kwargs
    ) -> None:
        """Create an interactive plot of the field using :mod:`napari`

        For a detailed description of the launched program, see the
        `napari webpage <http://napari.org/>`_.

        Args:
            viewer_args (dict):
                Arguments passed to :class:`napari.viewer.Viewer` to affect the viewer.
            **kwargs:
                Extra arguments passed to the plotting function
        """
        if viewer_args is None:
            viewer_args = {}

        if self.grid.num_axes == 1:
            raise RuntimeError(
                "Interactive plotting needs at least 2 spatial dimensions"
            )

        with napari_viewer(self.grid, **viewer_args) as viewer:
            napari_add_layers(viewer, self._get_napari_data(**kwargs))

    def split_mpi(
        self: TField, decomposition: Literal["auto"] | int | list[int] = "auto"
    ) -> TField:
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
        from ..grids._mesh import GridMesh
        from ..tools import mpi

        if not mpi.parallel_run:
            return self
        if self.grid._mesh is not None:
            raise RuntimeError("Cannot split an already split field")

        # create the grid mesh using the decomposition information
        mesh = GridMesh.from_grid(self.grid, decomposition)
        # do the actual splitting
        return mesh.split_field_mpi(self)
