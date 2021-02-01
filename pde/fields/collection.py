"""
Defines a collection of fields to represent multiple fields defined on a common
grid.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import json
import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np

from ..grids.base import GridBase
from ..tools.docstrings import fill_in_docstring
from ..tools.misc import Number, number_array
from ..tools.plotting import PlotReference, plot_on_figure
from .base import DataFieldBase, FieldBase, OptionalArrayLike
from .scalar import ScalarField


class FieldCollection(FieldBase):
    """ Collection of fields defined on the same grid """

    def __init__(
        self,
        fields: Sequence[DataFieldBase],
        data: OptionalArrayLike = None,
        *,
        copy_fields: bool = False,
        label: Optional[str] = None,
        labels: List[Optional[str]] = None,
        dtype=None,
    ):
        """
        Args:
            fields:
                Sequence of the individual fields
            data (:class:`numpy.ndarray`):
                Data of the fields. If `None`, the data is instead taken from
                the individual fields given by `fields`.
            copy_fields (bool):
                Flag determining whether the individual fields given in `fields`
                are copied. Note that fields are always copied if some of the supplied
                fields are identical.
            label (str):
                Label of the field collection
            labels (list of str):
                Labels of the individual fields. If omitted, the labels from the
                `fields` argument are used.
            dtype (numpy dtype):
                The data type of the field. All the numpy dtypes are supported. If
                omitted, it will be determined from `data` automatically.

        Warning:
            If `data` is given and :code:`copy_fields == False`, the data in the
            individual fields is overwritten by the associated `data`.
        """
        if isinstance(fields, FieldCollection):
            # support assigning a field collection for convenience
            fields = fields.fields

        if len(fields) == 0:
            raise ValueError("At least one field must be defined")

        # check if grids are compatible
        grid = fields[0].grid
        if any(grid != f.grid for f in fields[1:]):
            grids = [f.grid for f in fields]
            raise RuntimeError(f"Grids are incompatible: {grids}")

        # check whether some fields are identical
        if not copy_fields and len(fields) != len(set(id(field) for field in fields)):
            self._logger = logging.getLogger(self.__class__.__name__)
            self._logger.warning("Creating a copy of identical fields in collection")
            copy_fields = True

        # create the list of underlying fields
        if copy_fields:
            self._fields = [field.copy() for field in fields]
        else:
            self._fields = fields  # type: ignore

        # extract data from individual fields
        fields_data: List[np.ndarray] = []
        self._slices: List[slice] = []
        dof = 0  # count local degrees of freedom
        for field in self.fields:
            if not isinstance(field, DataFieldBase):
                raise RuntimeError(
                    "Individual fields must be of type DataFieldBase. Field "
                    "collections cannot be nested."
                )
            start = len(fields_data)
            this_data = field._data_flat
            fields_data.extend(this_data)
            self._slices.append(slice(start, len(fields_data)))
            dof += len(this_data)

        # combine into one data field
        data_shape = (dof,) + grid.shape
        if data is None:
            # the data is taken from the individual fields
            data_arr = number_array(fields_data, dtype=dtype, copy=False)

        else:
            # the data is taken from the supplied data argument
            data_arr = number_array(data, dtype=dtype, copy=False)
            if data_arr.shape != data_shape:
                data_arr = np.array(np.broadcast_to(data_arr, data_shape))
        assert data_arr.shape == data_shape

        # initialize the class
        super().__init__(grid, data_arr, label=label)

        # link the data of the original fields back to self._data if they were
        # not copied
        if not copy_fields:
            for i, field in enumerate(self.fields):
                field_shape = field.data.shape
                field._data_flat = self.data[self._slices[i]]

                # check whether the field data is based on our data field
                assert field.data.shape == field_shape
                assert field.data.base is self.data

        if labels is not None:
            self.labels = labels  # type: ignore

    def __repr__(self):
        """ return instance as string """
        fields = []
        for f in self.fields:
            name = f.__class__.__name__
            if f.label:
                fields.append(f'{name}(..., label="{f.label}")')
            else:
                fields.append(f"{name}(...)")
        return f"{self.__class__.__name__}({', '.join(fields)})"

    def __len__(self):
        """ return the number of stored fields """
        return len(self.fields)

    def __iter__(self) -> Iterator[DataFieldBase]:
        """ return iterator over the actual fields """
        return iter(self.fields)

    def __getitem__(self, index: Union[int, str]) -> DataFieldBase:
        """ return a specific field """
        if isinstance(index, int):
            # simple numerical index
            return self.fields[index]

        elif isinstance(index, str):
            # index specifying the label of the field
            for field in self.fields:
                if field.label == index:
                    return field
            raise KeyError(f"No field with name {index}")

        else:
            raise TypeError(f"Unsupported index {index}")

    def __setitem__(self, index: int, value):
        """ set the value of a specific field """
        # We need to load the field and set data explicitly
        # WARNING: Do not use `self.fields[index] = value`, since this would
        # break the connection between the data fields
        if isinstance(index, int):
            # simple numerical index
            self.fields[index].data = value

        elif isinstance(index, str):
            # index specifying the label of the field
            for field in self.fields:
                if field.label == index:
                    field.data = value
                    break
            else:
                raise KeyError(f"No field with name {index}")

        else:
            raise TypeError(f"Unsupported index {index}")

    @property
    def fields(self) -> List[DataFieldBase]:
        """ list: the fields of this collection """
        return self._fields

    @property
    def labels(self) -> "_FieldLabels":
        """:class:`_FieldLabels`: the labels of all fields

        Note:
            The attribute returns a special class :class:`_FieldLabels` to allow
            specific manipulations of the field labels. The returned object behaves
            much like a list, but assigning values will modify the labels of the fields
            in the collection.
        """
        return _FieldLabels(self)

    @labels.setter
    def labels(self, values: List[Optional[str]]):
        """ sets the labels of all fields """
        if len(values) != len(self):
            raise ValueError("Require a label for each field")
        for field, value in zip(self.fields, values):
            field.label = value

    def __eq__(self, other):
        """ test fields for equality, ignoring the label """
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.fields == other.fields

    @classmethod
    def from_state(
        cls, attributes: Dict[str, Any], data: np.ndarray = None
    ) -> "FieldCollection":
        """create a field collection from given state.

        Args:
            attributes (dict):
                The attributes that describe the current instance
            data (:class:`numpy.ndarray`, optional):
                Data values at support points of the grid defining all fields
        """
        if "class" in attributes:
            class_name = attributes.pop("class")
            assert class_name == cls.__name__

        # restore the individual fields (without data)
        fields = [
            FieldBase.from_state(field_state)
            for field_state in attributes.pop("fields")
        ]

        return cls(fields, data=data, **attributes)  # type: ignore

    @classmethod
    def _from_hdf_dataset(cls, dataset) -> "FieldCollection":
        """ construct the class by reading data from an hdf5 dataset """
        # copy attributes from hdf
        attributes = dict(dataset.attrs)

        # determine class
        class_name = json.loads(attributes.pop("class"))
        assert class_name == cls.__name__

        # determine the fields
        field_attrs = json.loads(attributes.pop("fields"))
        fields = [
            DataFieldBase._from_hdf_dataset(dataset[f"field_{i}"])
            for i in range(len(field_attrs))
        ]

        # unserialize remaining attributes
        attributes = cls.unserialize_attributes(attributes)
        return cls(fields, **attributes)  # type: ignore

    def _write_hdf_dataset(self, hdf_path):
        """ write data to a given hdf5 path `hdf_path` """
        # write attributes of the collection
        for key, value in self.attributes_serialized.items():
            hdf_path.attrs[key] = value

        # write individual fields
        for i, field in enumerate(self.fields):
            field._write_hdf_dataset(hdf_path, f"field_{i}")

    def assert_field_compatible(self, other: FieldBase, accept_scalar: bool = False):
        """checks whether `other` is compatible with the current field

        Args:
            other (FieldBase): Other field this is compared to
            accept_scalar (bool, optional): Determines whether it is acceptable
                that `other` is an instance of
                :class:`~pde.fields.ScalarField`.
        """
        super().assert_field_compatible(other, accept_scalar=accept_scalar)

        # check whether all sub fields are compatible
        if isinstance(other, FieldCollection):
            for f1, f2 in zip(self, other):
                f1.assert_field_compatible(f2, accept_scalar=accept_scalar)

    @classmethod
    @fill_in_docstring
    def from_scalar_expressions(
        cls,
        grid: GridBase,
        expressions: Sequence[str],
        *,
        label: str = None,
        labels: Optional[Sequence[str]] = None,
        dtype=None,
    ) -> "FieldCollection":
        """create a field collection on a grid from given expressions

        Warning:
            {WARNING_EXEC}

        Args:
            grid (:class:`~pde.grids.GridBase`):
                Grid defining the space on which this field is defined
            expressions (list of str):
                A list of mathematical expression, one for each field in the
                collection. The expressions determine the values as a function
                of the position on the grid. The expressions may contain
                standard mathematical functions and they may depend on the axes
                labels of the grid.
            label (str, optional):
                Name of the whole collection
            labels (list of str, optional):
                Names of the individual fields
            dtype (numpy dtype):
                The data type of the field. All the numpy dtypes are supported. If
                omitted, it will be determined from `data` automatically.
        """
        if isinstance(expressions, str):
            expressions = [expressions]
        if labels is None:
            labels = [None] * len(expressions)  # type: ignore

        # evaluate all expressions at all points
        fields = [
            ScalarField.from_expression(grid, expression, label=labels[i], dtype=dtype)
            for i, expression in enumerate(expressions)
        ]

        # create vector field from the data
        return cls(fields=fields, label=label)  # lgtm [py/call-to-non-callable]

    @classmethod
    def scalar_random_uniform(
        cls,
        num_fields: int,
        grid: GridBase,
        vmin: float = 0,
        vmax: float = 1,
        label: Optional[str] = None,
        labels: Optional[Sequence[str]] = None,
    ):
        """create scalar fields with random values between `vmin` and `vmax`

        Args:
            num_fields (int): The number of fields to create
            grid (:class:`~pde.grids.GridBase`):
                Grid defining the space on which the fields are defined
            vmin (float): Smallest random value
            vmax (float): Largest random value
            label (str, optional): Name of the field collection
            labels (list of str, optional): Names of the individual fields
        """
        if labels is None:
            labels = [None] * num_fields  # type: ignore
        return cls(
            [
                ScalarField.random_uniform(grid, vmin, vmax, label=labels[i])
                for i in range(num_fields)
            ],
            label=label,
        )

    @property
    def attributes(self) -> Dict[str, Any]:
        """ dict: describes the state of the instance (without the data) """
        results = super().attributes
        del results["grid"]
        results["fields"] = [f.attributes for f in self.fields]
        return results

    @property
    def attributes_serialized(self) -> Dict[str, str]:
        """ dict: serialized version of the attributes """
        results = {}
        for key, value in self.attributes.items():
            if key == "fields":
                fields = [f.attributes_serialized for f in self.fields]
                results[key] = json.dumps(fields)
            else:
                results[key] = json.dumps(value)
        return results

    @classmethod
    def unserialize_attributes(cls, attributes: Dict[str, str]) -> Dict[str, Any]:
        """unserializes the given attributes

        Args:
            attributes (dict):
                The serialized attributes

        Returns:
            dict: The unserialized attributes
        """
        results = {}
        for key, value in attributes.items():
            if key == "fields":
                results[key] = [
                    FieldBase.unserialize_attributes(attrs)
                    for attrs in json.loads(value)
                ]
            else:
                results[key] = json.loads(value)
        return results

    def copy(
        self, data: OptionalArrayLike = None, *, label: str = None, dtype=None
    ) -> "FieldCollection":
        """return a copy of the data, but not of the grid

        Args:
            data (:class:`numpy.ndarray`, optional):
                Data values at the support points of the grid that define the
                field. Note that the data is not copied but used directly.
            label (str, optional):
                Name of the copied field
            dtype (numpy dtype):
                The data type of the field. If omitted, it will be determined from
                `data` automatically.
        """
        if label is None:
            label = self.label
        fields = [f.copy() for f in self.fields]
        # if data is None, the data of the individual fields is copied in their
        # copy() method above. The underlying data is therefore independent from
        # the current field
        return self.__class__(
            fields, data=data, copy_fields=False, label=label, dtype=dtype
        )

    def interpolate_to_grid(
        self,
        grid: GridBase,
        method: str = "numba",
        fill: Number = None,
        label: Optional[str] = None,
    ) -> "FieldCollection":
        """interpolate the data of this field collection to another grid.

        Args:
            grid (:class:`~pde.grids.GridBase`):
                The grid of the new field onto which the current field is
                interpolated.
            method (str):
                Specifies interpolation method, e.g., 'numba', 'scipy_linear',
                'scipy_nearest' .
            fill (Number, optional):
                Determines how values out of bounds are handled. If `None`, a
                `ValueError` is raised when out-of-bounds points are requested.
                Otherwise, the given value is returned.
            label (str, optional):
                Name of the returned field collection

        Returns:
            FieldCollection: Interpolated data
        """
        if label is None:
            label = self.label
        fields = [
            f.interpolate_to_grid(grid, method=method, fill=fill) for f in self.fields
        ]
        return self.__class__(fields, label=label)

    def smooth(
        self,
        sigma: float = 1,
        out: Optional["FieldCollection"] = None,
        label: str = None,
    ) -> "FieldCollection":
        """applies Gaussian smoothing with the given standard deviation

        This function respects periodic boundary conditions of the underlying
        grid, using reflection when no periodicity is specified.

        sigma (float):
            Gives the standard deviation of the smoothing in real length units
            (default: 1)
        out (FieldCollection, optional):
            Optional field into which the smoothed data is stored
        label (str, optional):
            Name of the returned field

        Returns:
            Field collection with smoothed data, stored at `out` if given.
        """
        # allocate memory for storing output
        if out is None:
            out = self.copy(label=label)
        else:
            self.assert_field_compatible(out)
            if label:
                out.label = label

        # apply Gaussian smoothing for each axis
        for f_in, f_out in zip(self, out):
            f_in.smooth(sigma=sigma, out=f_out)

        return out

    @property
    def integrals(self):
        """ return the integrals of all fields """
        return [field.integral for field in self]

    def get_line_data(  # type: ignore
        self,
        index: int = 0,
        scalar: str = "auto",
        extract: str = "auto",
    ) -> Dict[str, Any]:
        r"""return data for a line plot of the field

        Args:
            index (int):
                Index of the field whose data is returned
            scalar (str or int):
                The method for extracting scalars as described in
                :meth:`DataFieldBase.to_scalar`.
            extract (str):
                The method used for extracting the line data. See the docstring
                of the grid method `get_line_data` to find supported values.

        Returns:
            dict: Information useful for performing a line plot of the field
        """
        return self[index].get_line_data(scalar=scalar, extract=extract)

    def get_image_data(self, index: int = 0, **kwargs) -> Dict[str, Any]:
        r"""return data for plotting an image of the field

        Args:
            index (int): Index of the field whose data is returned
            \**kwargs: Arguments forwarded to the `get_image_data` method

        Returns:
            dict: Information useful for plotting an image of the field
        """
        return self[index].get_image_data(**kwargs)

    def _update_plot(self, reference: List[PlotReference]) -> None:
        """update a plot collection with the current field values

        Args:
            reference (list of :class:`PlotReference`):
                All references of the plot to update
        """
        for field, ref in zip(self.fields, reference):
            field._update_plot(ref)

    @plot_on_figure(update_method="_update_plot")
    def plot(
        self,
        kind: Union[str, Sequence[str]] = "auto",
        resize_fig=None,
        figsize="auto",
        arrangement="horizontal",
        fig=None,
        subplot_args=None,
        **kwargs,
    ) -> List[PlotReference]:
        r"""visualize all the fields in the collection

        Args:
            kind (str or list of str):
                Determines the kind of the visualizations. Supported values are `image`,
                `line`, `vector`, or `interactive`. Alternatively, `auto` determines the
                best visualization based on each field itself. Instead of a single value
                for all fields, a list with individual values can be given.
            resize_fig (bool):
                Whether to resize the figure to adjust to the number of panels
            figsize (str or tuple of numbers):
                Determines the figure size. The figure size is unchanged if the string
                `default` is passed. Conversely, the size is adjusted automatically when
                `auto` is passed. Finally, a specific figure size can be specified using
                two values, using :func:`matplotlib.figure.Figure.set_size_inches`.
            arrangement (str):
                Determines how the subpanels will be arranged. The default value
                `horizontal` places all subplots next to each other. The alternative
                value `vertical` puts them below each other.
            {PLOT_ARGS}
            subplot_args (list):
                Additional arguments for the specific subplots. Should be a list with a
                dictionary of arguments for each subplot. Supplying an empty allows to
                keep the default setting of specific subplots.
            \**kwargs:
                All additional keyword arguments are forwarded to the actual plotting
                function of all subplots.

        Returns:
            List of :class:`PlotReference`: Instances that contain information
            to update all the plots with new data later.
        """
        if resize_fig is not None:
            # Deprecated this argument on 2021-02-01
            import warnings

            warnings.warn(
                "`resize_fig` is a deprecated argument. Use `figsize` directly",
                DeprecationWarning,
            )

            if resize_fig is True:
                figsize = "auto"
            elif resize_fig is False:
                figsize = "default"
            else:
                raise ValueError

        # set the size of the figure
        if figsize == "default":
            pass  # just leave the figure size at its default value

        elif figsize == "auto":
            # adjust the size of the figure
            if arrangement == "horizontal":
                fig.set_size_inches((4 * len(self), 3), forward=True)
            elif arrangement == "vertical":
                fig.set_size_inches((4, 3 * len(self)), forward=True)

        else:
            # assume that an actual tuple is given
            fig.set_size_inches(figsize, forward=True)

        # create all the subpanels
        if arrangement == "horizontal":
            (axs,) = fig.subplots(1, len(self), squeeze=False)
        elif arrangement == "vertical":
            axs = fig.subplots(len(self), 1, squeeze=False)
            axs = [a[0] for a in axs]  # transpose
        else:
            raise ValueError(f"Unknown arrangement `{arrangement}`")

        if subplot_args is None:
            subplot_args = [{}] * len(self)

        if isinstance(kind, str):
            kind = [kind] * len(self.fields)

        # plot all the elements onto the respective axes
        reference = [
            field.plot(kind=knd, ax=ax, action="create", **kwargs, **sp_args)
            for field, knd, ax, sp_args in zip(self.fields, kind, axs, subplot_args)
        ]

        # return the references for all subplots
        return reference

    def _get_napari_data(self, **kwargs) -> Dict[str, Dict[str, Any]]:
        r"""returns data for plotting all fields

        Args:
            \**kwargs: all arguments are forwarded to `_get_napari_layer_data`

        Returns:
            dict: all the information necessary to plot all fields
        """
        result = {}
        for i, field in enumerate(self, 1):
            name = f"Field {i}" if field.label is None else field.label
            result[name] = field._get_napari_layer_data(**kwargs)
        return result


class _FieldLabels:
    """ helper class that allows manipulating all labels of field collections"""

    def __init__(self, collection: FieldCollection):
        """
        Args:
            collection (:class:`pde.fields.collection.FieldCollection`):
                The field collection that these labels are associated with
        """
        self.collection = collection

    def __repr__(self):
        return repr(list(self))

    def __str__(self):
        return str(list(self))

    def __len__(self):
        return len(self.collection)

    def __eq__(self, other):
        return list(self) == list(other)

    def __iter__(self):
        for field in self.collection:
            yield field.label

    def __getitem__(
        self, index: Union[int, slice]
    ) -> Union[Optional[str], List[Optional[str]]]:
        """ return one or many labels of a field in the collection """
        if isinstance(index, int):
            return self.collection[index].label
        elif isinstance(index, slice):
            return list(self)[index]
        else:
            raise TypeError("Unsupported index type")

    def __setitem__(
        self, index: Union[int, slice], value: Union[Optional[str], List[Optional[str]]]
    ):
        """ change one or many labels of a field in the collection """
        if isinstance(index, int):
            assert isinstance(value, str)
            self.collection.fields[index].label = value
        elif isinstance(index, slice):
            fields = self.collection.fields[index]
            if value is None or isinstance(value, str):
                value = [value] * len(fields)
            if len(fields) != len(value):
                raise ValueError("Require a label for each field")
            for field, label in zip(fields, value):
                field.label = label
        else:
            raise TypeError("Unsupported index type")
