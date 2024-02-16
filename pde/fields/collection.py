"""
Defines a collection of fields to represent multiple fields defined on a common grid.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import json
import logging
import warnings
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Callable, Literal, overload

import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from numpy.typing import DTypeLike

try:
    from matplotlib.colormaps import get_cmap
except ImportError:
    from matplotlib.cm import get_cmap

from ..grids.base import GridBase
from ..tools.docstrings import fill_in_docstring
from ..tools.misc import Number, number_array
from ..tools.plotting import PlotReference, plot_on_axes, plot_on_figure
from ..tools.typing import NumberOrArray
from .base import DataFieldBase, FieldBase
from .scalar import ScalarField


class FieldCollection(FieldBase):
    """Collection of fields defined on the same grid

    Note:
        All fields in a collection must have the same data type. This might lead to
        up-casting, where for instance a combination of a real-valued and a
        complex-valued field will be both stored as complex fields.
    """

    def __init__(
        self,
        fields: Sequence[DataFieldBase] | Mapping[str, DataFieldBase],
        *,
        copy_fields: bool = False,
        label: str | None = None,
        labels: list[str | None] | _FieldLabels | None = None,
        dtype: DTypeLike = None,
    ):
        """
        Args:
            fields (sequence or mapping of :class:`DataFieldBase`):
                Sequence or mapping of the individual fields. If a mapping is used, the
                keys set the names of the individual fields.
            copy_fields (bool):
                Flag determining whether the individual fields given in `fields` are
                copied. Note that fields are always copied if some of the supplied
                fields are identical. If fields are copied the original fields will be
                left untouched. Conversely, if `copy_fields == False`, the original
                fields are modified so their data points to the collection. It is thus
                basically impossible to have fields that are linked to multiple
                collections at the same time.
            label (str):
                Label of the field collection
            labels (list of str):
                Labels of the individual fields. If omitted, the labels from the
                `fields` argument are used.
            dtype (numpy dtype):
                The data type of the field. All the numpy dtypes are supported. If
                omitted, it will be determined from `data` automatically.
        """
        if isinstance(fields, FieldCollection):
            # support assigning a field collection for convenience
            fields = fields.fields
        elif isinstance(fields, Mapping):
            # support setting fields using a mapping
            if labels is not None:
                self._logger = logging.getLogger(self.__class__.__name__)
                self._logger.warning("`labels` argument is ignored")
            labels = list(fields.keys())
            fields = list(fields.values())

        if len(fields) == 0:
            raise ValueError("At least one field must be defined")

        # check if grids are compatible
        grid = fields[0].grid
        if any(grid != f.grid for f in fields[1:]):
            grids = [f.grid for f in fields]
            raise RuntimeError(f"Grids are incompatible: {grids}")

        # check whether some fields are identical
        if not copy_fields and len(fields) != len({id(field) for field in fields}):
            self._logger = logging.getLogger(self.__class__.__name__)
            self._logger.warning("Creating a copy of identical fields in collection")
            copy_fields = True

        # create the list of underlying fields
        if copy_fields:
            self._fields = [field.copy() for field in fields]
        else:
            self._fields = fields  # type: ignore

        # extract data from individual fields
        fields_data: list[np.ndarray] = []
        self._slices: list[slice] = []
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

        # initialize the data from the individual fields
        data_arr = number_array(fields_data, dtype=dtype, copy=False)

        # initialize the class
        super().__init__(grid, data_arr, label=label)

        if not copy_fields:
            # link the data of the original fields back to self._data
            for i, field in enumerate(self.fields):
                field_shape = field.data.shape
                field._data_flat = self._data_full[self._slices[i]]

                # check whether the field data is based on our data field
                assert field.data.shape == field_shape
                assert np.may_share_memory(field._data_full, self._data_full)

        if labels is not None:
            self.labels = labels  # type: ignore

    def __repr__(self):
        """return instance as string"""
        fields = []
        for f in self.fields:
            name = f.__class__.__name__
            if f.label:
                fields.append(f'{name}(..., label="{f.label}")')
            else:
                fields.append(f"{name}(...)")
        return f"{self.__class__.__name__}({', '.join(fields)})"

    def __len__(self):
        """return the number of stored fields"""
        return len(self.fields)

    def __iter__(self) -> Iterator[DataFieldBase]:
        """return iterator over the actual fields"""
        return iter(self.fields)

    @overload
    def __getitem__(self, index: int | str) -> DataFieldBase: ...

    @overload
    def __getitem__(self, index: slice) -> FieldCollection: ...

    def __getitem__(self, index: int | str | slice) -> DataFieldBase | FieldCollection:
        """returns one or many fields from the collection

        If `index` is an integer or string, the field at this position or with this
        label is returned, respectively. If `index` is a :class:`slice`, a collection is
        returned. In this case the field data is copied.
        """
        if isinstance(index, int):
            # simple numerical index -> return single field
            return self.fields[index]

        elif isinstance(index, str):
            # index specifying the label of the field -> return a single field
            for field in self.fields:
                if field.label == index:
                    return field
            raise KeyError(f"No field with name `{index}`")

        elif isinstance(index, slice):
            # range of indices -> collection is returned
            return FieldCollection(self.fields[index], copy_fields=True)

        else:
            raise TypeError(f"Unsupported index `{index}`")

    def __setitem__(self, index: int | str, value: NumberOrArray):
        """set the value of a specific field

        Args:
            index (int or str):
                Determines which field is updated. If `index` is an integer it specifies
                the position of the field that will be updated. If `index` is a string,
                the first field with this name will be updated.
            value (float or :class:`~numpy.ndarray`):
                The updated value(s) of the chosen field.
        """
        # We need to load the field and set data explicitly
        # WARNING: Do not use `self.fields[index] = value`, since this would
        # break the connection between the data fields
        if isinstance(index, int):
            # simple numerical index
            self.fields[index].data = value  # type: ignore

        elif isinstance(index, str):
            # index specifying the label of the field
            for field in self.fields:
                if field.label == index:
                    field.data = value  # type: ignore
                    break  # indicates that a field has been found
            else:
                raise KeyError(f"No field with name `{index}`")

        else:
            raise TypeError(f"Unsupported index `{index}`")

    @property
    def fields(self) -> list[DataFieldBase]:
        """list: the fields of this collection"""
        # return shallow copy of list so the internal list is not modified accidentially
        return self._fields[:]

    @property
    def labels(self) -> _FieldLabels:
        """:class:`_FieldLabels`: the labels of all fields

        Note:
            The attribute returns a special class :class:`_FieldLabels` to allow
            specific manipulations of the field labels. The returned object behaves
            much like a list, but assigning values will modify the labels of the fields
            in the collection.
        """
        return _FieldLabels(self)

    @labels.setter
    def labels(self, values: list[str | None]):
        """sets the labels of all fields"""
        if len(values) != len(self):
            raise ValueError("Require a label for each field")
        for field, value in zip(self.fields, values):
            field.label = value

    def __eq__(self, other):
        """test fields for equality, ignoring the label"""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.fields == other.fields

    @classmethod
    def from_state(
        cls, attributes: dict[str, Any], data: np.ndarray | None = None
    ) -> FieldCollection:
        """create a field collection from given state.

        Args:
            attributes (dict):
                The attributes that describe the current instance
            data (:class:`~numpy.ndarray`, optional):
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

        # create the collection
        collection = cls(fields, **attributes)  # type: ignore

        if data is not None:
            collection.data = data  # set the data of all fields

        return collection

    @classmethod
    def from_data(
        cls,
        field_classes,
        grid: GridBase,
        data: np.ndarray,
        *,
        with_ghost_cells: bool = True,
        label: str | None = None,
        labels: list[str | None] | _FieldLabels | None = None,
        dtype: DTypeLike = None,
    ):
        """create a field collection from classes and data

        Args:
            field_classes (list):
                List of the classes that define the individual fields
            data (:class:`~numpy.ndarray`, optional):
                Data values of all fields at support points of the grid
            grid (:class:`~pde.grids.base.GridBase`):
                Grid defining the space on which this field is defined.
            with_ghost_cells (bool):
                Indicates whether the ghost cells are included in data
            label (str):
                Label of the field collection
            labels (list of str):
                Labels of the individual fields. If omitted, the labels from the
                `fields` argument are used.
            dtype (numpy dtype):
                The data type of the field. All the numpy dtypes are supported. If
                omitted, it will be determined from `data` automatically.
        """
        # extract data from individual fields
        fields = []
        start = 0
        for field_class in field_classes:
            if not issubclass(field_class, DataFieldBase):
                raise RuntimeError("Individual fields must be of type DataFieldBase.")
            field = field_class(grid)
            end = start + grid.num_axes**field.rank
            if with_ghost_cells:
                field._data_flat = data[start:end]
            else:
                field.data.flat = data[start:end].flat
            fields.append(field)
            start = end

        return cls(fields, copy_fields=False, label=label, labels=labels, dtype=dtype)

    @classmethod
    def _from_hdf_dataset(cls, dataset) -> FieldCollection:
        """construct the class by reading data from an hdf5 dataset"""
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
        """write data to a given hdf5 path `hdf_path`"""
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
        user_funcs: dict[str, Callable] | None = None,
        consts: dict[str, NumberOrArray] | None = None,
        label: str | None = None,
        labels: Sequence[str] | None = None,
        dtype: DTypeLike = None,
    ) -> FieldCollection:
        """create a field collection on a grid from given expressions

        Warning:
            {WARNING_EXEC}

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                Grid defining the space on which this field is defined
            expressions (list of str):
                A list of mathematical expression, one for each field in the collection.
                The expressions determine the values as a function of the position on
                the grid. The expressions may contain standard mathematical functions
                and they may depend on the axes labels of the grid.
                More information can be found in the
                :ref:`expression documentation <documentation-expressions>`.
            user_funcs (dict, optional):
                A dictionary with user defined functions that can be used in the
                expression
            consts (dict, optional):
                A dictionary with user defined constants that can be used in the
                expression. The values of these constants should either be numbers or
                :class:`~numpy.ndarray`.
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
            ScalarField.from_expression(
                grid,
                expression,
                user_funcs=user_funcs,
                consts=consts,
                label=sublabel,
                dtype=dtype,
            )
            for expression, sublabel in zip(expressions, labels)
        ]

        # create vector field from the data
        return cls(fields=fields, label=label)

    @classmethod
    def scalar_random_uniform(
        cls,
        num_fields: int,
        grid: GridBase,
        vmin: float = 0,
        vmax: float = 1,
        *,
        label: str | None = None,
        labels: Sequence[str] | None = None,
        rng: np.random.Generator | None = None,
    ) -> FieldCollection:
        """create scalar fields with random values between `vmin` and `vmax`

        Args:
            num_fields (int):
                The number of fields to create
            grid (:class:`~pde.grids.base.GridBase`):
                Grid defining the space on which the fields are defined
            vmin (float):
                Lower bound. Can be complex to create complex fields
            vmax (float):
                Upper bound. Can be complex to create complex fields
            label (str, optional):
                Name of the field collection
            labels (list of str, optional):
                Names of the individual fields
            rng (:class:`~numpy.random.Generator`):
                Random number generator (default: :func:`~numpy.random.default_rng()`)
        """
        if labels is None:
            labels = [None] * num_fields  # type: ignore
        return cls(
            [
                ScalarField.random_uniform(grid, vmin, vmax, label=labels[i], rng=rng)
                for i in range(num_fields)
            ],
            label=label,
        )

    @property
    def attributes(self) -> dict[str, Any]:
        """dict: describes the state of the instance (without the data)"""
        results = super().attributes

        # store the attributes of the individual fields in a separate attribute
        results["fields"] = [f.attributes for f in self.fields]

        # the grid information does not need to be stored since it is included in the
        # attributes of the individual fields
        del results["grid"]

        return results

    @property
    def attributes_serialized(self) -> dict[str, str]:
        """dict: serialized version of the attributes"""
        results = {}
        for key, value in self.attributes.items():
            if key == "fields":
                fields = [f.attributes_serialized for f in self.fields]
                results[key] = json.dumps(fields)
            elif key == "dtype":
                results[key] = json.dumps(value.str)
            else:
                results[key] = json.dumps(value)
        return results

    @classmethod
    def unserialize_attributes(cls, attributes: dict[str, str]) -> dict[str, Any]:
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
        self: FieldCollection,
        *,
        label: str | None = None,
        dtype: DTypeLike = None,
    ) -> FieldCollection:
        """return a copy of the data, but not of the grid

        Args:
            label (str, optional):
                Name of the returned field
            dtype (numpy dtype):
                The data type of the field. If omitted, it will be determined from
                `data` automatically.
        """
        if label is None:
            label = self.label
        fields = [f.copy() for f in self.fields]

        # create the collection from the copied fields
        return self.__class__(fields, copy_fields=False, label=label, dtype=dtype)

    def append(
        self,
        *fields: DataFieldBase | FieldCollection,
        label: str | None = None,
    ) -> FieldCollection:
        """create new collection with appended field(s)

        Args:
            fields (`FieldCollection` or `DataFieldBase`):
                A sequence of single fields or collection of fields that will be
                appended to the fields in the current collection. The data of all fields
                will be copied.
            label (str):
                Label of the new field collection. If omitted, the current label is used

        Returns:
            :class:`~pde.fields.collection.FieldCollection`: A new field collection,
            which combines the current one with fields given by `fields`.
        """
        # copy fields and labels
        _fields, _labels = self.fields[:], list(self.labels)
        for field in fields:
            if isinstance(field, FieldCollection):
                _fields.extend(field.fields)
                _labels.extend(field.labels)
            else:
                _fields.append(field)
                _labels.append(field.label)

        return FieldCollection(
            _fields,
            copy_fields=True,
            label=self.label if label is None else label,
            labels=_labels,
        )

    def _unary_operation(self: FieldCollection, op: Callable) -> FieldCollection:
        """perform an unary operation on this field collection

        Args:
            op (callable):
                A function calculating the result

        Returns:
            FieldBase: An field that contains the result of the operation.
        """
        fields = [fields._unary_operation(op) for fields in self.fields]
        return self.__class__(fields, label=self.label)

    def interpolate_to_grid(
        self,
        grid: GridBase,
        *,
        fill: Number | None = None,
        label: str | None = None,
    ) -> FieldCollection:
        """interpolate the data of this field collection to another grid.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid of the new field onto which the current field is
                interpolated.
            fill (Number, optional):
                Determines how values out of bounds are handled. If `None`, a
                `ValueError` is raised when out-of-bounds points are requested.
                Otherwise, the given value is returned.
            label (str, optional):
                Name of the returned field collection

        Returns:
            :class:`~pde.fields.coolection.FieldCollection`: Interpolated data
        """
        if label is None:
            label = self.label
        fields = [f.interpolate_to_grid(grid, fill=fill) for f in self.fields]
        return self.__class__(fields, label=label)

    def smooth(
        self,
        sigma: float = 1,
        *,
        out: FieldCollection | None = None,
        label: str | None = None,
    ) -> FieldCollection:
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
    def integrals(self) -> list:
        """integrals of all fields"""
        return [field.integral for field in self]

    @property
    def averages(self) -> list:
        """averages of all fields"""
        return [field.average for field in self]

    @property
    def magnitudes(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: scalar magnitudes of all fields"""
        return np.array([field.magnitude for field in self])

    def get_line_data(  # type: ignore
        self,
        index: int = 0,
        scalar: str = "auto",
        extract: str = "auto",
    ) -> dict[str, Any]:
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

    def get_image_data(self, index: int = 0, **kwargs) -> dict[str, Any]:
        r"""return data for plotting an image of the field

        Args:
            index (int): Index of the field whose data is returned
            \**kwargs: Arguments forwarded to the `get_image_data` method

        Returns:
            dict: Information useful for plotting an image of the field
        """
        return self[index].get_image_data(**kwargs)

    def _get_merged_image_data(
        self,
        colors: list[str] | None = None,
        projection: Literal["max", "mean", "min", "product", "sum"] = "min",
        *,
        background_color: str = "w",
        inverse_projection: bool = False,
        transpose: bool = False,
        vmin: float | list[float | None] | None = None,
        vmax: float | list[float | None] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """obtain data required for a merged plot

        Args:
            colors (list):
                Colors used for each color channel. This can either be a matplotlib
                colormap used for mapping the channels or a single matplotlib color used
                to interpolate between the background.
            projection (str):
                Defines a projection determining how different colors are merged.
                Possible options are "max", "mean", "min", "product", and "sum".
            inverse_projection (bool):
                Inverses colors before applying the projection. Can be useful for dark
                color maps and black backgrounds.
            background_color (str):
                Defines the background color corresponding to vanishing values. Not used
                for colormaps specified in `colors`.
            transpose (bool):
                Determines whether the transpose of the data is plotted
            vmin, vmax (float, list of float):
                Define the data range that the color chanels cover. By default, they
                cover the complete value range of the supplied data.

        Returns:
            tuple: a :class:`~numpy.ndarray` of the merged data together with a dict of
            additional information, e.g., about the extent and the axes.
        """
        num_fields = len(self)
        if colors is None:
            colors = [f"C{i}" for i in range(num_fields)]
        if not hasattr(vmin, "__iter__"):
            vmin = [vmin] * num_fields
        if not hasattr(vmax, "__iter__"):
            vmax = [vmax] * num_fields

        # compile image data for all channels
        data = []
        for i, f in enumerate(self):
            field_data = f.get_image_data(transpose=transpose)
            norm = Normalize(vmin=vmin[i], vmax=vmax[i], clip=True)  # type: ignore
            try:
                cmap = get_cmap(colors[i])
            except ValueError:
                cmap = LinearSegmentedColormap.from_list(
                    "", [background_color, colors[i]]
                )
            m = cm.ScalarMappable(norm=norm, cmap=cmap)
            data.append(m.to_rgba(field_data["data"].T))
        arr = np.array(data)

        # combine the images
        if inverse_projection:
            arr = 1 - arr
        if projection == "max":
            rgb_arr = np.max(arr, axis=0)
        elif projection == "mean":
            rgb_arr = np.mean(arr, axis=0)
        elif projection == "min":
            rgb_arr = np.min(arr, axis=0)
        elif projection == "product":
            rgb_arr = np.prod(arr, axis=0)
        elif projection == "sum":
            rgb_arr = np.sum(arr, axis=0)
        else:
            raise ValueError(f"Undefined projection `{projection}`")
        if inverse_projection:
            rgb_arr = 1 - rgb_arr

        return rgb_arr, field_data

    def _update_merged_image_plot(self, reference: PlotReference) -> None:
        """update an merged image plot with the current field values

        Args:
            reference (:class:`PlotReference`):
                The reference to the plot that is updated
        """
        # obtain image data
        data_args = reference.parameters.copy()
        data_args.pop("kind")
        rgb_arr, _ = self._get_merged_image_data(**data_args)
        # update the axes image
        reference.element.set_data(rgb_arr)

    @plot_on_axes(update_method="_update_merged_image_plot")
    def _plot_merged_image(
        self,
        ax,
        colors: list[str] | None = None,
        projection: Literal["max"] = "max",
        inverse_projection: bool = False,
        background_color: str = "w",
        transpose: bool = False,
        vmin: float | list[float | None] | None = None,
        vmax: float | list[float | None] | None = None,
        **kwargs,
    ) -> PlotReference:
        r"""visualize fields by mapping to different color chanels in a 2d density plot

        Args:
            ax (:class:`matplotlib.axes.Axes`):
                Figure axes to be used for plotting.
            colors (list):
                Colors used for each color channel. This can either be a matplotlib
                colormap used for mapping the channels or a single matplotlib color used
                to interpolate between the background.
            projection (str):
                Defines a projection determining how different colors are merged.
                Possible options are "max", "mean", "min", "product", and "sum".
            inverse_projection (bool):
                Inverses colors before applying the projection. Can be useful for dark
                color maps and black backgrounds.
            background_color (str):
                Defines the background color corresponding to vanishing values. Not used
                for colormaps specified in `colors`.
            transpose (bool):
                Determines whether the transpose of the data is plotted
            vmin, vmax (float, list of float):
                Define the data range that the color chanels cover. By default, they
                cover the complete value range of the supplied data.
            \**kwargs:
                Additional keyword arguments that affect the image. Non-Cartesian grids
                might support `performance_goal` to influence how an image is created
                from raw data. Finally, remaining arguments are passed to
                :func:`matplotlib.pyplot.imshow` to affect the appearance.

        Returns:
            :class:`PlotReference`: Instance that contains information to update the
            plot with new data later.
        """
        rgba_arr, data = self._get_merged_image_data(
            colors,
            projection,
            inverse_projection=inverse_projection,
            background_color=background_color,
            transpose=transpose,
            vmin=vmin,
            vmax=vmax,
        )

        # plot the image
        kwargs.setdefault("origin", "lower")
        kwargs.setdefault("interpolation", "none")
        axes_image = ax.imshow(rgba_arr, extent=data["extent"], **kwargs)

        # set some default properties
        ax.set_xlabel(data["label_x"])
        ax.set_ylabel(data["label_y"])
        ax.set_title(self.label)

        # store parameters of the plot that are necessary for updating
        parameters = {
            "kind": "merged_image",
            "transpose": transpose,
            "vmin": vmin,
            "vmax": vmax,
        }
        return PlotReference(ax, axes_image, parameters)

    @plot_on_axes(update_method="_update_rgb_image_plot")
    def _plot_rgb_image(
        self,
        ax,
        transpose: bool = False,
        vmin: float | list[float | None] | None = None,
        vmax: float | list[float | None] | None = None,
        **kwargs,
    ) -> PlotReference:
        r"""visualize fields by mapping to different color chanels in a 2d density plot

        Args:
            ax (:class:`matplotlib.axes.Axes`):
                Figure axes to be used for plotting.
            transpose (bool):
                Determines whether the transpose of the data is plotted
            vmin, vmax (float, list of float):
                Define the data range that the color chanels cover. By default, they
                cover the complete value range of the supplied data.
            \**kwargs:
                Additional keyword arguments that affect the image. Non-Cartesian grids
                might support `performance_goal` to influence how an image is created
                from raw data. Finally, remaining arguments are passed to
                :func:`matplotlib.pyplot.imshow` to affect the appearance.

        Returns:
            :class:`PlotReference`: Instance that contains information to update the
            plot with new data later.
        """
        # since 2024-01-25
        warnings.warn(
            "`rgb_image` is deprecated in favor of `merged`", DeprecationWarning
        )
        return self._plot_merged_image(  # type: ignore
            ax=ax,
            colors="rgb",
            background_color="k",
            projection="max",
            transpose=transpose,
            vmin=vmin,
            vmax=vmax,
            **kwargs,
        )

    def _update_plot(self, reference: list[PlotReference]) -> None:
        """update a plot collection with the current field values

        Args:
            reference (list of :class:`PlotReference`):
                All references of the plot to update
        """
        if reference[0].parameters.get("kind", None) == "merged_image":
            self._update_merged_image_plot(reference[0])
        else:
            for field, ref in zip(self.fields, reference):
                field._update_plot(ref)

    @plot_on_figure(update_method="_update_plot")
    def plot(
        self,
        kind: str | Sequence[str] = "auto",
        figsize="auto",
        arrangement="horizontal",
        fig=None,
        subplot_args=None,
        **kwargs,
    ) -> list[PlotReference]:
        r"""visualize all the fields in the collection

        Args:
            kind (str or list of str):
                Determines the kind of the visualizations. Supported values are `image`,
                `line`, `vector`, `interactive`, or `merged`. Alternatively, `auto`
                determines the best visualization based on each field itself. Instead of
                a single value for all fields, a list with individual values can be
                given, unless `merged` is chosen.
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
                dictionary of arguments for each subplot. Supplying an empty dict allows
                to keep the default setting of specific subplots.
            \**kwargs:
                All additional keyword arguments are forwarded to the actual plotting
                function of all subplots.

        Returns:
            List of :class:`PlotReference`: Instances that contain information
            to update all the plots with new data later.
        """
        if kind in {"merged", "rgb", "rgb_image", "rgb-image"}:
            num_panels = 1
        else:
            num_panels = len(self)

        # set the size of the figure
        if figsize == "default":
            pass  # just leave the figure size at its default value

        elif figsize == "auto":
            # adjust the size of the figure
            if arrangement == "horizontal":
                fig.set_size_inches((4 * num_panels, 3), forward=True)
            elif arrangement == "vertical":
                fig.set_size_inches((4, 3 * num_panels), forward=True)

        else:
            # assume that an actual tuple is given
            fig.set_size_inches(figsize, forward=True)

        # create all the subpanels
        if arrangement == "horizontal":
            (axs,) = fig.subplots(1, num_panels, squeeze=False)
        elif arrangement == "vertical":
            axs = fig.subplots(num_panels, 1, squeeze=False)
            axs = [a[0] for a in axs]  # transpose
        else:
            raise ValueError(f"Unknown arrangement `{arrangement}`")

        if subplot_args is None:
            subplot_args = [{}] * num_panels

        if kind in {"merged"}:
            # plot a single RGB representation
            reference = [
                self._plot_merged_image(
                    ax=axs[0], action="none", **kwargs, **subplot_args[0]
                )
            ]

        elif kind in {"rgb", "rgb_image", "rgb-image"}:
            # plot a single RGB representation
            reference = [
                self._plot_rgb_image(
                    ax=axs[0], action="none", **kwargs, **subplot_args[0]
                )
            ]

        else:
            # plot all the elements onto the respective axes
            if isinstance(kind, str):
                kind = [kind] * num_panels
            reference = [
                field.plot(kind=knd, ax=ax, action="none", **kwargs, **sp_args)
                for field, knd, ax, sp_args in zip(  # @UnusedVariable
                    self.fields, kind, axs, subplot_args
                )
            ]

        # return the references for all subplots
        return reference

    def _get_napari_data(self, **kwargs) -> dict[str, dict[str, Any]]:
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
    """helper class that allows manipulating all labels of field collections"""

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

    def __iter__(self) -> Iterator[str | None]:
        for field in self.collection:
            yield field.label

    def __getitem__(self, index: int | slice) -> str | None | list[str | None]:
        """return one or many labels of a field in the collection"""
        if isinstance(index, int):
            return self.collection[index].label
        elif isinstance(index, slice):
            return list(self)[index]
        else:
            raise TypeError("Unsupported index type")

    def __setitem__(self, index: int | slice, value: None | str | list[str | None]):
        """change one or many labels of a field in the collection"""
        if isinstance(index, int):
            self.collection.fields[index].label = value  # type: ignore
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

    def index(self, label: str) -> int:
        """return the index in the field labels where a certain label is stored

        Args:
            label (str):
                The label that is sought

        Returns:
            int: The index in the list (or `ValueError` if it cannot be found)
        """
        for i, field in enumerate(self.collection):
            if field.label == label:
                return i
        raise ValueError(f"Label `{label}` is not present in the collection")
