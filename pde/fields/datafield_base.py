"""Defines base class of single fields with arbitrary rank.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import functools
import json
import warnings
from abc import ABCMeta, abstractmethod
from inspect import isabstract
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeVar

import numba as nb
import numpy as np
from numba.extending import overload, register_jitable
from numpy.typing import DTypeLike

from ..grids.base import DimensionError, DomainError, GridBase, discretize_interval
from ..grids.boundaries.axes import BoundariesData
from ..tools.cache import cached_method
from ..tools.docstrings import fill_in_docstring
from ..tools.misc import Number, number_array
from ..tools.numba import get_common_numba_dtype, jit, make_array_constructor
from ..tools.plotting import PlotReference, plot_on_axes
from ..tools.spectral import CorrelationType, make_correlated_noise
from ..tools.typing import ArrayLike, NumberOrArray
from .base import FieldBase, RankError

if TYPE_CHECKING:
    from .scalar import ScalarField


TDataField = TypeVar("TDataField", bound="DataFieldBase")


class DataFieldBase(FieldBase, metaclass=ABCMeta):
    """Abstract base class for describing fields of single entities."""

    rank: int  # the rank of the tensor field

    def __init__(
        self,
        grid: GridBase,
        data: ArrayLike | str | None = "zeros",
        *,
        label: str | None = None,
        dtype: DTypeLike | None = None,
        with_ghost_cells: bool = False,
    ):
        """
        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                Grid defining the space on which this field is defined.
            data (Number or :class:`~numpy.ndarray`, optional):
                Field values at the support points of the grid. The flag
                `with_ghost_cells` determines whether this data array contains values
                for the ghost cells, too. The resulting field will contain real data
                unless the `data` argument contains complex values. Special values are
                "zeros" or None, initializing the field with zeros, and "empty", just
                allocating memory with unspecified values.
            label (str, optional):
                Name of the field
            dtype (numpy dtype):
                The data type of the field. If omitted, it will be determined from
                `data` automatically.
            with_ghost_cells (bool):
                Indicates whether the ghost cells are included in data
        """
        if isinstance(data, self.__class__):
            # special case where a DataFieldBase is supplied
            data_arr = number_array(data._data_full, dtype=dtype, copy=True)
            super().__init__(grid, data=data_arr, label=label)

        elif with_ghost_cells:
            # use full data without copying (unless necessary)
            if data is None or isinstance(data, str):
                raise ValueError("`data` must be supplied if with_ghost_cells==True")
            data_arr = number_array(data, dtype=dtype, copy=None)
            super().__init__(grid, data=data_arr, label=label)

        else:
            # ghost cells are not supplied => allocate full array and write valid data
            full_shape = (grid.dim,) * self.rank + tuple(s + 2 for s in grid.shape)

            if data is None:
                # fill full data with zeros by default
                data_arr = np.zeros(full_shape, dtype=dtype)
                super().__init__(grid, data=data_arr, label=label)

            elif isinstance(data, str):
                # allocate empty data
                if data == "empty":
                    data_arr = np.empty(full_shape, dtype=dtype)
                elif data == "zeros":
                    data_arr = np.zeros(full_shape, dtype=dtype)
                elif data == "ones":
                    data_arr = np.ones(full_shape, dtype=dtype)
                else:
                    raise ValueError(f"Unknown data '{data}'")
                super().__init__(grid, data=data_arr, label=label)

            elif isinstance(data, DataFieldBase):
                # copy the full data from the supplied field
                grid.assert_grid_compatible(data.grid)
                data_arr = number_array(data._data_full, dtype=dtype, copy=True)
                super().__init__(grid, data=data_arr, label=label)

            else:
                # initialize empty data and set the valid data
                data_arr = number_array(data, dtype=dtype, copy=None)
                empty_data = np.empty(full_shape, dtype=data_arr.dtype)
                super().__init__(grid, data=empty_data, label=label)
                self.data = data_arr

    def __repr__(self) -> str:
        """Return instance as string."""
        class_name = self.__class__.__name__
        result = f"{class_name}(grid={self.grid!r}, data={self.data}"
        if self.label:
            result += f', label="{self.label}"'
        if self.dtype != np.double:
            result += f', dtype="{self.dtype}"'
        return result + ")"

    def __str__(self) -> str:
        """Return instance as string."""
        result = (
            f"{self.__class__.__name__}(grid={self.grid}, data=Array{self.data.shape}"
        )
        if self.dtype != np.double:
            result = result[:-1] + f', dtype="{self.dtype}")'
        if self.label:
            result += f', label="{self.label}"'
        return result + ")"

    @classmethod
    def random_uniform(
        cls: type[TDataField],
        grid: GridBase,
        vmin: float = 0,
        vmax: float = 1,
        *,
        label: str | None = None,
        dtype: DTypeLike | None = None,
        rng: np.random.Generator | None = None,
    ) -> TDataField:
        """Create field with uncorrelated uniform distributed random values.

        A complex field is returned when `vmin` or `vmax` is a complex number. In this
        case, the real and imaginary parts of these arguments are used to determine
        the distribution of the real and imaginary parts of the resulting field.
        Consequently, :code:`ScalarField.random_uniform(grid, 0, 1 + 1j)` creates a
        complex field where the real and imaginary parts are chosen from a standard
        uniform distribution.

        Real and imaginary parts of fields, all components of vector and tensor fields,
        as well as all spatial positions are always uncorrelated.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                Grid defining the space on which this field is defined
            vmin (float):
                Lower bound for the random values
            vmax (float):
                Upper bound for the random values
            label (str, optional):
                Name of the returned field
            dtype (numpy dtype):
                The data type of the field. If omitted, it defaults to `double` if both
                `vmin` and `vmax` are real, otherwise it is `complex`.
            rng (:class:`~numpy.random.Generator`):
                Random number generator (default: :func:`~numpy.random.default_rng()`)
        """
        rng = np.random.default_rng(rng)

        # determine the shape of the data array
        shape = (grid.dim,) * cls.rank + grid.shape

        if np.iscomplexobj(vmin) or np.iscomplexobj(vmax):
            # create complex random numbers for the field
            real_part = rng.uniform(np.real(vmin), np.real(vmax), size=shape)
            imag_part = rng.uniform(np.imag(vmin), np.imag(vmax), size=shape)
            data: np.ndarray = real_part + 1j * imag_part
        else:
            # create real random numbers for the field
            data = rng.uniform(vmin, vmax, size=shape)

        return cls(grid, data=data, label=label, dtype=dtype)

    @classmethod
    def random_normal(
        cls: type[TDataField],
        grid: GridBase,
        mean: float = 0,
        std: float = 1,
        *,
        scaling: Literal["none", "physical"] = "none",
        correlation: CorrelationType = "none",
        label: str | None = None,
        dtype: DTypeLike | None = None,
        rng: np.random.Generator | None = None,
        **kwargs,
    ) -> TDataField:
        r"""Creates Gaussian random field with normal distributed random values.

        A complex field is returned when either `mean` or `std` is a complex number. In
        this case, the real and imaginary parts of these arguments are used to determine
        the distribution of the real and imaginary parts of the resulting field.
        Consequently, :code:`ScalarField.random_normal(grid, 0, 1 + 1j)` creates a
        complex field where the real and imaginary parts are chosen from a standard
        normal distribution.

        Real and imaginary parts of fields, as well as all components of vector and
        tensor fields, are always uncorrelated. Correlations in spatial positions are
        supported through the `correlation` argument. If set, the returned field
        :math:`f` obeys the selected correlation function :math:`C(k)` (see table below
        for details). In Fourier space, we thus have

        .. math::
            \langle f(\boldsymbol k) f(\boldsymbol k’) \rangle =
                C(|\boldsymbol k|) \delta(\boldsymbol k-\boldsymbol k’)

        For simplicity, the correlations respect periodic boundary conditions.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                Grid defining the space on which this field is defined
            mean (float):
                Mean of the Gaussian distribution
            std (float):
                Standard deviation of the Gaussian distribution.
            scaling (str):
                Determines how the values are scaled. Possible choices are 'none'
                (values are drawn from a normal distribution with given mean and
                standard deviation) or 'physical' (the variance of the random number is
                scaled by the inverse volume of the grid cell; this is for instance
                useful for concentration fields, which vary less in larger volumes).
            correlation (str):
                Selects the correlation function used to make the correlated noise. Many
                of the options (described below) support additional parameters that can
                be supplied as keyword arguments.
            label (str, optional):
                Name of the returned field
            dtype (numpy dtype):
                The data type of the field. If omitted, it defaults to `double` if both
                `mean` and `std` are real, otherwise it is `complex`.
            rng (:class:`~numpy.random.Generator`):
                Random number generator (default: :func:`~numpy.random.default_rng()`)
            **kwargs:
                Additional parameters can affect details of the correlation function.

        .. table:: Supported correlation functions
            :widths: 20 80

            ================= ==========================================================
            Identifier        Correlation function
            ================= ==========================================================
            :code:`none`      No correlation, :math:`C(k) = \delta(k)`

            :code:`gaussian`  :math:`C(k) = \exp(\frac12 k^2 \lambda^2)` with the length
                              scale :math:`\lambda` set by argument :code:`length_scale`.

            :code:`power law` :math:`C(k) = k^{\nu/2}` with exponent :math:`\nu` set by
                              argument :code:`exponent`.

            :code:`cosine`    :math:`C(k) = \exp\bigl(-s^2(\lambda k - 1)^2\bigr)` with
                              the length scale :math:`\lambda` set by argument
                              :code:`length_scale`, whereas the sharpness parameter
                              :math:`s` is set by :code:`sharpness` and defaults to 10.
            ================= ==========================================================
        """
        rng = np.random.default_rng(rng)

        # create a function for creating a single noise field
        make_scalar_field = make_correlated_noise(
            grid.shape,
            correlation=correlation,
            discretization=grid.discretization,
            rng=rng,
            **kwargs,
        )

        if cls.rank == 0:
            make_random_field = make_scalar_field
        else:
            tensor_shape = (grid.dim,) * cls.rank

            def make_random_field() -> np.ndarray:
                """Helper function that creates a single tensor field."""
                out = np.empty(tensor_shape + grid.shape)
                print(out.shape, tensor_shape, grid)
                for idx in np.ndindex(tensor_shape):
                    out[idx] = make_scalar_field()
                return out  # type: ignore

        # create random fields with correct mean and standard deviation
        if scaling == "none":
            scale = 1
        elif scaling == "physical":
            scale = 1 / np.sqrt(grid.cell_volumes)
        else:
            raise ValueError(f"Unknown noise scaling {scaling}")

        if np.iscomplexobj(mean) or np.iscomplexobj(std):
            # create complex random numbers for the field
            real_part = np.real(mean) + np.real(std) * scale * make_random_field()
            imag_part = np.imag(mean) + np.imag(std) * scale * make_random_field()
            data: np.ndarray = real_part + 1j * imag_part
        else:
            # create real random numbers for the field
            data = mean + std * scale * make_random_field()

        return cls(grid, data=data, label=label, dtype=dtype)

    @classmethod
    def random_harmonic(
        cls: type[TDataField],
        grid: GridBase,
        modes: int = 3,
        harmonic=np.cos,
        axis_combination=np.multiply,
        *,
        label: str | None = None,
        dtype: DTypeLike | None = None,
        rng: np.random.Generator | None = None,
    ) -> TDataField:
        r"""Create a random field build from harmonics.

        The resulting fields will be highly correlated in space and can thus
        serve for testing differential operators.

        With the default settings, the resulting field :math:`c_i(\mathbf{x})`
        is given by

        .. math::
            c_i(\mathbf{x}) = \prod_{\alpha=1}^N \sum_{j=1}^M a_{ij\alpha}
                \cos\left(\frac{2 \pi x_\alpha}{j L_\alpha}\right) \;,

        where :math:`N` is the number of spatial dimensions, each with length
        :math:`L_\alpha`, :math:`M` is the number of modes given by `modes`, and
        :math:`a_{ij\alpha}` are random amplitudes, chosen from a uniform
        distribution over the interval [0, 1].

        Note that the product could be replaced by a sum when
        `axis_combination = numpy.add` and the :math:`\cos()` could be any other
        function given by the parameter `harmonic`.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                Grid defining the space on which this field is defined
            modes (int):
                Number :math:`M` of harmonic modes
            harmonic (callable):
                Determines which harmonic function is used. Typical values are
                :func:`numpy.sin` and :func:`numpy.cos`, which basically relate
                to different boundary conditions applied at the grid boundaries.
            axis_combination (callable):
                Determines how values from different axis are combined. Typical
                choices are :func:`numpy.multiply` and :func:`numpy.add`
                resulting in products and sums of the values along axes,
                respectively.
            label (str, optional):
                Name of the returned field
            dtype (numpy dtype):
                The data type of the field. If omitted, it defaults to `double`.
            rng (:class:`~numpy.random.Generator`):
                Random number generator (default: :func:`~numpy.random.default_rng()`)
        """
        rng = np.random.default_rng(rng)

        tensor_shape = (grid.dim,) * cls.rank

        data = np.empty(tensor_shape + grid.shape)
        # determine random field for each component
        for index in np.ndindex(*tensor_shape):
            data_axis = []
            # random harmonic function along each axis
            for i in range(len(grid.axes)):
                # choose wave vectors
                ampl = rng.random(size=modes)  # amplitudes
                x = discretize_interval(0, 2 * np.pi, grid.shape[i])[0]
                data_axis.append(
                    sum(a * harmonic(n * x) for n, a in enumerate(ampl, 1))
                )
            # full dataset is product of values along axes
            data[index] = functools.reduce(axis_combination.outer, data_axis)

        return cls(grid, data=data, label=label, dtype=dtype)

    @classmethod
    def random_colored(
        cls: type[TDataField],
        grid: GridBase,
        exponent: float = 0,
        scale: float = 1,
        *,
        label: str | None = None,
        dtype: DTypeLike | None = None,
        rng: np.random.Generator | None = None,
    ) -> TDataField:
        r"""Create a field of random values with colored noise.

        The spatially correlated values obey

        .. math::
            \langle c_i(\boldsymbol k) c_j(\boldsymbol k’) \rangle =
                \Gamma^2 |\boldsymbol k|^\nu \delta_{ij}
                \delta(\boldsymbol k - \boldsymbol k’)

        in spectral space, where :math:`\boldsymbol k` is the wave vector. The special
        case :math:`\nu = 0` corresponds to white noise. Note that the spatial
        correlations always assume periodic boundary conditions (even if the underlying
        grid does not) and that the components of tensor fields are uncorrelated.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                Grid defining the space on which this field is defined
            exponent (float):
                Exponent :math:`\nu` of the power spectrum
            scale (float):
                Scaling factor :math:`\Gamma` determining noise strength
            label (str, optional):
                Name of the returned field
            dtype (numpy dtype):
                The data type of the field. If omitted, it defaults to `double`.
            rng (:class:`~numpy.random.Generator`):
                Random number generator (default: :func:`~numpy.random.default_rng()`)
        """
        # deprecated since 2025-04-04
        warnings.warn(
            "`random_colored` method is deprecated. Use `random_normal` with "
            "correlation='power law' instead",
            DeprecationWarning,
        )

        # get function making colored noise
        from ..tools.spectral import make_colored_noise

        make_scalar_field = make_colored_noise(
            grid.shape, dx=grid.discretization, exponent=exponent, rng=rng
        )

        # create random fields for each tensor component
        tensor_shape = (grid.dim,) * cls.rank
        data = np.empty(tensor_shape + grid.shape)
        # determine random field for each component
        for index in np.ndindex(*tensor_shape):
            data[index] = scale * make_scalar_field()

        return cls(grid, data=data, label=label, dtype=dtype)

    @classmethod
    def get_class_by_rank(cls, rank: int) -> type[DataFieldBase]:
        """Return a :class:`DataFieldBase` subclass describing a field with a given
        rank.

        Args:
            rank (int): The rank of the tensor field

        Returns:
            The DataField class that corresponds to the rank
        """
        for field_cls in cls._subclasses.values():
            if (
                issubclass(field_cls, DataFieldBase)
                and not isabstract(field_cls)
                and field_cls.rank == rank
            ):
                return field_cls
        raise RuntimeError(f"Could not find field class for rank {rank}")

    @classmethod
    def from_state(
        cls: type[TDataField],
        attributes: dict[str, Any],
        data: np.ndarray | None = None,
    ) -> TDataField:
        """Create a field from given state.

        Args:
            attributes (dict):
                The attributes that describe the current instance
            data (:class:`~numpy.ndarray`, optional):
                Data values at the support points of the grid defining the field

        Returns:
            :class:`DataFieldBase`: The instance created from the stored state
        """
        if "class" in attributes:
            class_name = attributes.pop("class")
            assert class_name == cls.__name__

        # create the instance from the attributes
        return cls(attributes.pop("grid"), data=data, **attributes)

    def copy(
        self: TDataField,
        *,
        label: str | None = None,
        dtype: DTypeLike | None = None,
    ) -> TDataField:
        if label is None:
            label = self.label
        if dtype is None:
            dtype = self.dtype

        return self.__class__(
            self.grid,
            data=np.array(self._data_full, dtype=dtype, copy=True),
            label=label,
            dtype=dtype,
            with_ghost_cells=True,
        )

    @property
    def data_shape(self) -> tuple[int, ...]:
        """tuple: the shape of the data at each grid point"""
        return (self.grid.dim,) * self.rank

    @classmethod
    def unserialize_attributes(cls, attributes: dict[str, str]) -> dict[str, Any]:
        """Unserializes the given attributes.

        Args:
            attributes (dict):
                The serialized attributes

        Returns:
            dict: The unserialized attributes
        """
        results = {}
        for key, value in attributes.items():
            if key == "grid":
                results[key] = GridBase.from_state(value)
            else:
                results[key] = json.loads(value)
        return results

    def _write_to_image(self, filename: str, **kwargs) -> None:
        r"""Write data to image.

        Args:
            filename (str):
                The path to the image that will be created
            \**kwargs:
                Additional keyword arguments that affect the image. For
                instance, some fields support a `scalar` argument that
                determines how they are converted to a scalar. Non-Cartesian
                grids might support a `performance_goal` argument to influence
                how an image is created from the raw data. Finally, the
                remaining arguments are are passed to
                :func:`matplotlib.pyplot.imsave` to affect the appearance.
        """
        import matplotlib.pyplot as plt

        # obtain image data
        get_image_args = {}
        for key in ["performance_goal", "scalar"]:
            if key in kwargs:
                get_image_args[key] = kwargs.pop(key)
        img = self.get_image_data(**get_image_args)

        kwargs.setdefault("cmap", "gray")
        plt.imsave(filename, img["data"].T, origin="lower", **kwargs)

    @cached_method()
    def make_interpolator(
        self,
        *,
        fill: Number | None = None,
        with_ghost_cells: bool = False,
    ) -> Callable[[np.ndarray, np.ndarray], NumberOrArray]:
        r"""Returns a function that can be used to interpolate values.

        Args:
            fill (Number, optional):
                Determines how values out of bounds are handled. If `None`, a
                `ValueError` is raised when out-of-bounds points are requested.
                Otherwise, the given value is returned.
            with_ghost_cells (bool):
                Flag indicating that the interpolator should work on the full data array
                that includes values for the ghost points. If this is the case, the
                boundaries are not checked and the coordinates are used as is.

        Returns:
            A function which returns interpolated values when called with arbitrary
            positions within the space of the grid.
        """
        grid = self.grid
        num_axes = self.grid.num_axes
        data_shape = self.data_shape

        # convert `fill` to dtype of data
        if fill is not None:
            if self.rank == 0:
                fill = self.data.dtype.type(fill)
            else:
                fill = np.broadcast_to(fill, self.data_shape).astype(self.data.dtype)

        # create the method to interpolate data at a single point
        interpolate_single = grid._make_interpolator_compiled(
            fill=fill, with_ghost_cells=with_ghost_cells
        )

        # provide a method to access the current data of the field
        if with_ghost_cells:
            get_data_array = make_array_constructor(self._data_full)
        else:
            get_data_array = make_array_constructor(self.data)

        dim_error_msg = f"Dimension of point does not match axes count {num_axes}"

        @jit
        def interpolator(
            point: np.ndarray, data: np.ndarray | None = None
        ) -> np.ndarray:
            """Return the interpolated value at the position `point`

            Args:
                point (:class:`~numpy.ndarray`):
                    The list of points. This point coordinates should be given along the
                    last axis, i.e., the shape should be `(..., num_axes)`.
                data (:class:`~numpy.ndarray`, optional):
                    The discretized field values. If omitted, the data of the current
                    field is used, which should be the default. However, this option can
                    be useful to interpolate other fields defined on the same grid
                    without recreating the interpolator. If a data array is supplied, it
                    needs to be the full data if `with_ghost_cells == True`, and
                    otherwise only the valid data.

            Returns:
                :class:`~numpy.ndarray`: The interpolated values at the points
            """
            # check input
            point = np.atleast_1d(point)
            if point.shape[-1] != num_axes:
                raise DimensionError(dim_error_msg)
            point_shape = point.shape[:-1]

            if data is None:
                # reconstruct data field from memory address
                data = get_data_array()

            # interpolate at every valid point
            out = np.empty(data_shape + point_shape, dtype=data.dtype)
            for idx in np.ndindex(*point_shape):
                out[(...,) + idx] = interpolate_single(data, point[idx])

            return out  # type: ignore

        # store a reference to the data so it is not garbage collected too early
        interpolator._data = self.data

        return interpolator  # type: ignore

    @fill_in_docstring
    def interpolate(
        self,
        point: np.ndarray,
        *,
        bc: BoundariesData | None = None,
        fill: Number | None = None,
    ) -> np.ndarray:
        r"""Interpolate the field to points between support points.

        Args:
            point (:class:`~numpy.ndarray`):
                The points at which the values should be obtained. This is given in grid
                coordinates.
            bc:
                The boundary conditions applied to the field, which affects values close
                to the boundary. If omitted, the argument `fill` is used to determine
                values outside the domain.
                {ARG_BOUNDARIES_OPTIONAL}
            fill (Number, optional):
                Determines how values out of bounds are handled. If `None`, a
                `ValueError` is raised when out-of-bounds points are requested.
                Otherwise, the given value is returned.

        Returns:
            :class:`~numpy.ndarray`: the values of the field
        """
        if bc is not None:
            # impose boundary conditions and then interpolate using ghost cells
            self.set_ghost_cells(bc, set_corners=True)
            interpolator = self.make_interpolator(fill=fill, with_ghost_cells=True)

        else:
            # create an interpolator without imposing bounary conditions
            interpolator = self.make_interpolator(fill=fill, with_ghost_cells=False)

        # do the actual interpolation
        return interpolator(np.asarray(point))  # type: ignore

    @fill_in_docstring
    def interpolate_to_grid(
        self: TDataField,
        grid: GridBase,
        *,
        bc: BoundariesData | None = None,
        fill: Number | None = None,
        label: str | None = None,
    ) -> TDataField:
        """Interpolate the data of this field to another grid.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid of the new field onto which the current field is
                interpolated.
            bc:
                The boundary conditions applied to the field, which affects values close
                to the boundary. If omitted, the argument `fill` is used to determine
                values outside the domain.
                {ARG_BOUNDARIES_OPTIONAL}
            fill (Number, optional):
                Determines how values out of bounds are handled. If `None`, a
                `ValueError` is raised when out-of-bounds points are requested.
                Otherwise, the given value is returned.
            label (str, optional):
                Name of the returned field

        Returns:
            Field of the same rank as the current one.
        """
        raise NotImplementedError(f"Cannot interpolate {self.__class__.__name__}")

    def insert(self, point: np.ndarray, amount: ArrayLike) -> None:
        """Adds an (integrated) value to the field at an interpolated position.

        Args:
            point (:class:`~numpy.ndarray`):
                The point inside the grid where the value is added. This is
                given in grid coordinates.
            amount (Number or :class:`~numpy.ndarray`):
                The amount that will be added to the field. The value describes
                an integrated quantity (given by the field value times the
                discretization volume). This is important for consistency with
                different discretizations and in particular grids with
                non-uniform discretizations.
        """
        point = np.atleast_1d(point)
        amount = np.broadcast_to(amount, self.data_shape)
        grid = self.grid
        grid_dim = len(grid.axes)

        if point.size != grid_dim or point.ndim != 1:
            raise DimensionError(f"Dimension mismatch for point {point}")

        # determine the grid coordinates next to the chosen points
        low = np.array(grid.axes_bounds)[:, 0]
        c_l, d_l = np.divmod((point - low) / grid.discretization - 0.5, 1.0)
        c_l = c_l.astype(int)  # support points to the left of the chosen points
        w_l = 1 - d_l  # weights of the low point
        w_h = d_l  # weights of the high point

        # apply periodic boundary conditions to grid coordinates
        c_h = c_l + 1  # support points to the right of the chosen points
        for ax in np.flatnonzero(grid.periodic):
            c_l[..., ax] %= grid.shape[ax]
            c_h[..., ax] %= grid.shape[ax]

        # determine the valid points and the total weight in first iteration
        total_weight = 0.0
        cells = []
        for i in np.ndindex(*((2,) * grid_dim)):
            coords = np.choose(i, [c_l, c_h])
            if np.all(coords >= 0) and np.all(coords < grid.shape):
                weight = np.prod(np.choose(i, [w_l, w_h]))
                total_weight += weight
                cells.append((tuple(coords), weight))

        if total_weight == 0:
            raise DomainError("Point lies outside grid")

        # alter each point in second iteration
        for coords, weight in cells:
            chng = weight * amount / (total_weight * grid.cell_volumes[coords])
            self.data[(Ellipsis,) + coords] += chng

    @fill_in_docstring
    def get_boundary_values(
        self, axis: int, upper: bool, bc: BoundariesData | None = None
    ) -> NumberOrArray:
        """Get the field values directly on the specified boundary.

        Args:
            axis (int):
                The axis perpendicular to the boundary
            upper (bool):
                Whether the boundary is at the upper side of the axis
            bc:
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES_OPTIONAL}

        Returns:
            :class:`~numpy.ndarray`: The discretized values on the boundary
        """
        if bc is not None:
            self.set_ghost_cells(bc=bc)

        l_wall: list[slice | int] = [slice(1, -1)] * self.grid.num_axes
        l_ghost = l_wall.copy()
        if upper:
            l_wall[axis] = -2
            l_ghost[axis] = -1
        else:
            l_wall[axis] = 1
            l_ghost[axis] = 0
        i_wall = (...,) + tuple(l_wall)
        i_ghost = (...,) + tuple(l_ghost)

        return (self._data_full[i_wall] + self._data_full[i_ghost]) / 2  # type: ignore

    @fill_in_docstring
    def set_ghost_cells(self, bc: BoundariesData, *, args=None, **kwargs) -> None:
        r"""Set the boundary values on virtual points for all boundaries.

        Args:
            bc (str or list or tuple or dict):
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES}
            args:
                Additional arguments that might be supported by special boundary
                conditions.
            **kwargs:
                Some boundary conditions might provide additional control via keyword
                arguments, e.g., `set_corners` to control whether corner cells are set
                using interpolation.
        """
        bcs = self.grid.get_boundary_conditions(bc, rank=self.rank)
        bcs.set_ghost_cells(self._data_full, args=args, **kwargs)

    @property
    @abstractmethod
    def integral(self) -> NumberOrArray:
        """Integral of the scalar field over space."""

    @abstractmethod
    def to_scalar(
        self, scalar: str = "auto", *, label: str | None = None
    ) -> ScalarField:
        """Return scalar variant of the field."""

    @property
    def average(self) -> NumberOrArray:
        """Float or :class:`~numpy.ndarray`: the average of data.

        This is calculated by integrating each component of the field over space and
        dividing by the grid volume
        """
        return self.integral / self.grid.volume

    @property
    def fluctuations(self) -> NumberOrArray:
        """Float or :class:`~numpy.ndarray`: quantification of the average fluctuations.

        The fluctuations are defined as the standard deviation of the data scaled by the
        cell volume. This definition makes the fluctuations independent of the
        discretization. It corresponds to the physical scaling available in the
        :func:`~DataFieldBase.random_normal`.

        Returns:
            :class:`~numpy.ndarray`: A tensor with the same rank of the field,
            specifying the fluctuations of each component of the tensor field
            individually. Consequently, a simple scalar is returned for a
            :class:`~pde.fields.scalar.ScalarField`.
        """
        scaled_data = self.data * np.sqrt(self.grid.cell_volumes)
        axes = tuple(range(self.rank, self.data.ndim))
        return np.std(scaled_data, axis=axes)  # type: ignore

    @property
    def magnitude(self) -> float:
        """float: determine the (scalar) magnitude of the field

        This is calculated by getting a scalar field using the default arguments of the
        :func:`to_scalar` method, averaging the result over the whole grid, and taking
        the absolute value.
        """
        if self.rank == 0:
            return abs(self.average)  # type: ignore
        elif self.rank > 0:
            return abs(self.to_scalar().average)  # type: ignore
        else:
            raise AssertionError("Rank must be non-negative")

    @fill_in_docstring
    def apply_operator(
        self,
        operator: str,
        bc: BoundariesData | None,
        out: DataFieldBase | None = None,
        *,
        label: str | None = None,
        args: dict[str, Any] | None = None,
        **kwargs,
    ) -> DataFieldBase:
        r"""Apply a (differential) operator and return result as a field.

        Args:
            operator (str):
                An identifier determining the operator. Note that not all grids support
                the same operators.
            bc:
                Boundary conditions applied to the field before applying the operator.
                {ARG_BOUNDARIES_OPTIONAL}
            out (:class:`DataFieldBase`, optional):
                Optional field to which the  result is written.
            label (str, optional):
                Name of the returned field
            args (dict):
                Additional arguments for the boundary conditions
            **kwargs:
                Additional arguments affecting how the operator behaves.

        Returns:
            :class:`DataFieldBase`: Field data after applying the operator. This field
            is identical to `out` if this argument was specified.
        """
        # get information about the operator
        operator_info = self.grid._get_operator_info(operator)
        out_cls = self.get_class_by_rank(operator_info.rank_out)

        # prepare the output field
        if out is None:
            out = out_cls(self.grid, data="empty", label=label, dtype=self.dtype)
        elif not isinstance(out, out_cls):
            raise RankError(f"`out` must be a {out_cls.__name__}")
        else:
            self.grid.assert_grid_compatible(out.grid)
            if label is not None:
                out.label = label

        op_raw = self.grid.make_operator_no_bc(operator_info, **kwargs)
        if bc is not None:
            self.set_ghost_cells(bc, args=args)  # impose boundary conditions
        # apply the operator without imposing boundary conditions
        op_raw(self._data_full, out.data)

        return out

    def make_dot_operator(
        self, backend: Literal["numpy", "numba"] = "numba", *, conjugate: bool = True
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray | None], np.ndarray]:
        """Return operator calculating the dot product between two fields.

        This supports both products between two vectors as well as products
        between a vector and a tensor.

        Args:
            backend (str):
                Can be `numba` or `numpy`, deciding how the function is constructed
            conjugate (bool):
                Whether to use the complex conjugate for the second operand

        Returns:
            function that takes two instance of :class:`~numpy.ndarray`, which contain
            the discretized data of the two operands. An optional third argument can
            specify the output array to which the result is written.
        """
        dim = self.grid.dim
        num_axes = self.grid.num_axes

        @register_jitable
        def maybe_conj(arr: np.ndarray) -> np.ndarray:
            """Helper function implementing optional conjugation."""
            return arr.conjugate() if conjugate else arr

        def dot(
            a: np.ndarray, b: np.ndarray, out: np.ndarray | None = None
        ) -> np.ndarray:
            """Numpy implementation to calculate dot product between two fields."""
            rank_a = a.ndim - num_axes
            rank_b = b.ndim - num_axes
            if rank_a < 1 or rank_b < 1:
                raise TypeError("Fields in dot product must have rank >= 1")
            if a.shape[rank_a:] != b.shape[rank_b:]:
                raise ValueError("Shapes of fields are not compatible for dot product")

            if rank_a == 1 and rank_b == 1:  # result is scalar field
                return np.einsum("i...,i...->...", a, maybe_conj(b), out=out)

            elif rank_a == 2 and rank_b == 1:  # result is vector field
                return np.einsum("ij...,j...->i...", a, maybe_conj(b), out=out)

            elif rank_a == 1 and rank_b == 2:  # result is vector field
                return np.einsum("i...,ij...->j...", a, maybe_conj(b), out=out)

            elif rank_a == 2 and rank_b == 2:  # result is tensor-2 field
                return np.einsum("ij...,jk...->ik...", a, maybe_conj(b), out=out)

            else:
                raise TypeError(f"Unsupported shapes ({a.shape}, {b.shape})")

        if backend == "numpy":
            # return the bare dot operator without the numba-overloaded version
            return dot

        elif backend == "numba":
            # overload `dot` and return a compiled version

            def get_rank(arr: nb.types.Type | nb.types.Optional) -> int:
                """Determine rank of field with type `arr`"""
                arr_typ = arr.type if isinstance(arr, nb.types.Optional) else arr
                if not isinstance(arr_typ, (np.ndarray, nb.types.Array)):
                    raise nb.errors.TypingError(
                        f"Dot argument must be array, not  {arr_typ.__class__}"
                    )
                rank = arr_typ.ndim - num_axes
                if rank < 1:
                    raise nb.NumbaTypeError(
                        f"Rank={rank} too small for dot product. Use a normal product "
                        "instead."
                    )
                return rank  # type: ignore

            @overload(dot, inline="always")
            def dot_ol(
                a: np.ndarray, b: np.ndarray, out: np.ndarray | None = None
            ) -> np.ndarray:
                """Numba implementation to calculate dot product between two fields."""
                # get (and check) rank of the input arrays
                rank_a = get_rank(a)
                rank_b = get_rank(b)

                if rank_a == 1 and rank_b == 1:  # result is scalar field

                    @register_jitable
                    def calc(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> None:
                        out[:] = a[0] * maybe_conj(b[0])
                        for j in range(1, dim):
                            out[:] += a[j] * maybe_conj(b[j])

                elif rank_a == 2 and rank_b == 1:  # result is vector field

                    @register_jitable
                    def calc(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> None:
                        for i in range(dim):
                            out[i] = a[i, 0] * maybe_conj(b[0])
                            for j in range(1, dim):
                                out[i] += a[i, j] * maybe_conj(b[j])

                elif rank_a == 1 and rank_b == 2:  # result is vector field

                    @register_jitable
                    def calc(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> None:
                        for i in range(dim):
                            out[i] = a[0] * maybe_conj(b[0, i])
                            for j in range(1, dim):
                                out[i] += a[j] * maybe_conj(b[j, i])

                elif rank_a == 2 and rank_b == 2:  # result is tensor-2 field

                    @register_jitable
                    def calc(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> None:
                        for i in range(dim):
                            for j in range(dim):
                                out[i, j] = a[i, 0] * maybe_conj(b[0, j])
                                for k in range(1, dim):
                                    out[i, j] += a[i, k] * maybe_conj(b[k, j])

                else:
                    raise NotImplementedError("Inner product for these ranks")

                if isinstance(out, (nb.types.NoneType, nb.types.Omitted)):
                    # function is called without `out` -> allocate memory
                    rank_out = rank_a + rank_b - 2
                    a_shape = (dim,) * rank_a + self.grid.shape
                    b_shape = (dim,) * rank_b + self.grid.shape
                    out_shape = (dim,) * rank_out + self.grid.shape
                    dtype = get_common_numba_dtype(a, b)

                    def dot_impl(
                        a: np.ndarray, b: np.ndarray, out: np.ndarray | None = None
                    ) -> np.ndarray:
                        """Helper function allocating output array."""
                        assert a.shape == a_shape
                        assert b.shape == b_shape
                        out = np.empty(out_shape, dtype=dtype)
                        calc(a, b, out)
                        return out

                else:
                    # function is called with `out` argument -> reuse `out` array

                    def dot_impl(
                        a: np.ndarray, b: np.ndarray, out: np.ndarray | None = None
                    ) -> np.ndarray:
                        """Helper function without allocating output array."""
                        assert a.shape == a_shape
                        assert b.shape == b_shape
                        assert out.shape == out_shape  # type: ignore
                        calc(a, b, out)
                        return out  # type: ignore

                return dot_impl  # type: ignore

            @jit
            def dot_compiled(
                a: np.ndarray, b: np.ndarray, out: np.ndarray | None = None
            ) -> np.ndarray:
                """Numba implementation to calculate dot product between two fields."""
                return dot(a, b, out)

            return dot_compiled  # type: ignore

        else:
            raise ValueError(f"Unsupported backend `{backend}")

    def smooth(
        self: TDataField,
        sigma: float = 1,
        *,
        out: TDataField | None = None,
        label: str | None = None,
    ) -> TDataField:
        """Applies Gaussian smoothing with the given standard deviation.

        This function respects periodic boundary conditions of the underlying grid,
        using reflection when no periodicity is specified.

        Args:
            sigma (float):
                Gives the standard deviation of the smoothing in real length units
                (default: 1)
            out (FieldBase, optional):
                Optional field into which the smoothed data is stored. Setting this
                to the input field enables in-place smoothing.
            label (str, optional):
                Name of the returned field

        Returns:
            Field with smoothed data. This is stored at `out` if given.
        """
        from scipy import ndimage

        # allocate memory for storing output
        if out is None:
            out = self.__class__(self.grid, label=self.label)
        else:
            self.assert_field_compatible(out)

        # apply Gaussian smoothing for each axis
        data_in = self.data  # use the field data as input
        data_out = out.data  # write to the output
        for axis in range(-len(self.grid.axes), 0):
            sigma_dx = sigma / self.grid.discretization[axis]
            mode = "wrap" if self.grid.periodic[axis] else "reflect"
            ndimage.gaussian_filter1d(
                data_in, sigma=sigma_dx, axis=axis, output=data_out, mode=mode
            )
            data_in = data_out  # use this smoothed data as input for next axis

        # return the data in the correct field class
        if label:
            out.label = label
        return out

    def get_line_data(
        self, scalar: str = "auto", extract: str = "auto"
    ) -> dict[str, Any]:
        # turn field into scalar field
        scalar_data = self.to_scalar(scalar).data

        # extract the line data
        data = self.grid.get_line_data(scalar_data, extract=extract)
        if "label_y" in data and data["label_y"]:
            if self.label:
                data["label_y"] = f"{self.label} ({data['label_y']})"
        else:
            data["label_y"] = self.label
        return data

    def get_image_data(
        self, scalar: str = "auto", transpose: bool = False, **kwargs
    ) -> dict[str, Any]:
        # turn field into scalar field
        scalar_data = self.to_scalar(scalar).data

        # remove imaginary parts
        if self.is_complex:
            self._logger.warning("Only the absolute value of complex data is shown")
            scalar_data = abs(scalar_data)

        # extract the image data
        data = self.grid.get_image_data(scalar_data, **kwargs)
        data["title"] = self.label

        if transpose:
            # adjust image data such that the transpose is plotted
            data["x"], data["y"] = data["y"], data["x"]
            data["data"] = data["data"].T
            data["label_x"], data["label_y"] = data["label_y"], data["label_x"]
            data["extent"] = data["extent"][2:] + data["extent"][:2]

        return data

    def get_vector_data(self, transpose: bool = False, **kwargs) -> dict[str, Any]:
        r"""Return data for a vector plot of the field.

        Args:
            \**kwargs: Additional parameters are forwarded to
                `grid.get_image_data`

        Returns:
            dict: Information useful for plotting an vector field
        """
        raise NotImplementedError

    def _plot_line(
        self,
        ax,
        scalar: str = "auto",
        extract: str = "auto",
        ylabel: str | None = None,
        ylim: tuple[float, float] | None = None,
        **kwargs,
    ) -> PlotReference:
        r"""Visualize a field using a 1d line plot.

        Args:
            ax (:class:`matplotlib.axes.Axes`):
                Figure axes to be used for plotting.
            scalar (str or int):
                The method for extracting scalars as described in
                :meth:`DataFieldBase.to_scalar`.
            extract (str):
                The method used for extracting the line data.
            ylabel (str):
                Label of the y-axis. If omitted, the label is chosen automatically from
                the data field.
            ylim (tuple of float):
                Limits of the y-axis. If omitted, the data range is used
            \**kwargs:
                Additional arguments are passed to :func:`matplotlib.pyplot.plot`

        Returns:
            :class:`PlotReference`: Instance that contains information to update
            the plot with new data later.
        """
        # obtain data for the plot
        line_data = self.get_line_data(scalar=scalar, extract=extract)

        # warn if there is an imaginary part
        if np.any(np.iscomplex(line_data["data_y"])):
            self._logger.warning("Only the real part of the complex data is shown")

        # do the plot
        (line2d,) = ax.plot(line_data["data_x"], line_data["data_y"].real, **kwargs)

        # set some default properties
        ax.set_xlabel(line_data["label_x"])
        if ylabel is None:
            ylabel = line_data.get("label_y", self.label)
        if ylabel:
            ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(ylim)

        return PlotReference(
            ax,
            line2d,
            {"scalar": scalar, "extract": extract, "ylabel": ylabel, "ylim": ylim},
        )

    def _update_line_plot(self, reference: PlotReference) -> None:
        """Update a line plot with the current field values.

        Args:
            reference (:class:`PlotReference`):
                The reference to the plot that is updated
        """
        import matplotlib as mpl

        # obtain data for the plot
        scalar = reference.parameters.get("scalar", "auto")
        extract = reference.parameters.get("extract", "auto")
        line_data = self.get_line_data(scalar=scalar, extract=extract)

        line2d = reference.element
        if isinstance(line2d, mpl.lines.Line2D):
            # update old plot
            line2d.set_xdata(line_data["data_x"])
            line2d.set_ydata(line_data["data_y"].real)

        else:
            raise ValueError(f"Unsupported plot reference {reference}")

    def _plot_image(
        self,
        ax,
        colorbar: bool = True,
        scalar: str = "auto",
        transpose: bool = False,
        **kwargs,
    ) -> PlotReference:
        r"""Visualize a field using a 2d density plot.

        Args:
            ax (:class:`matplotlib.axes.Axes`):
                Figure axes to be used for plotting.
            colorbar (bool):
                Determines whether a colorbar is shown
            scalar (str or int):
                The method for extracting scalars as described in
                :meth:`DataFieldBase.to_scalar`.
            transpose (bool):
                Determines whether the transpose of the data is plotted
            \**kwargs:
                Additional keyword arguments that affect the image. Non-Cartesian grids
                might support `performance_goal` to influence how an image is created
                from raw data. Finally, remaining arguments are passed to
                :func:`matplotlib.pyplot.imshow` to affect the appearance.

        Returns:
            :class:`PlotReference`: Instance that contains information to update
            the plot with new data later.
        """
        # obtain image data with appropriate parameters
        data_kws = {}
        for arg in ["performance_goal", "scalar", "transpose"]:
            if arg in kwargs:
                data_kws[arg] = kwargs.pop(arg)
        data = self.get_image_data(scalar, transpose, **data_kws)

        # plot the image
        kwargs.setdefault("origin", "lower")
        kwargs.setdefault("interpolation", "none")
        axes_image = ax.imshow(data["data"].T, extent=data["extent"], **kwargs)

        # set some default properties
        ax.set_xlabel(data["label_x"])
        ax.set_ylabel(data["label_y"])
        ax.set_title(data.get("title", self.label))

        if colorbar:
            from ..tools.plotting import add_scaled_colorbar

            add_scaled_colorbar(axes_image, ax=ax)

        # store parameters of the plot that are necessary for updating
        parameters = {"scalar": scalar, "transpose": transpose}
        if "vmin" in kwargs:
            parameters["vmin"] = kwargs["vmin"]
        if "vmax" in kwargs:
            parameters["vmax"] = kwargs["vmax"]
        return PlotReference(ax, axes_image, parameters)

    def _update_image_plot(self, reference: PlotReference) -> None:
        """Update an image plot with the current field values.

        Args:
            reference (:class:`PlotReference`):
                The reference to the plot that is updated
        """
        # obtain image data
        p = reference.parameters
        data = self.get_image_data(
            scalar=p.get("scalar", "auto"), transpose=p.get("transpose", False)
        )

        # update the axes image
        reference.element.set_data(data["data"].T)

        # adjust the colorbar limits
        vmin = p["vmin"] if "vmin" in p else data["data"].min()
        vmax = p["vmax"] if "vmax" in p else data["data"].max()
        reference.element.set_clim(vmin, vmax)

    def _plot_vector(
        self,
        ax,
        *,
        method: Literal["quiver", "streamplot"] = "quiver",
        max_points: int | None = 16,
        **kwargs,
    ) -> PlotReference:
        r"""Visualize a field using a 2d vector plot.

        Args:
            ax (:class:`matplotlib.axes.Axes`):
                Figure axes to be used for plotting.
            method (str):
                Plot type that is used. This can be either `quiver` or `streamplot`.
            max_points (int):
                The maximal number of points that is used along each axis. This argument
                is only used for quiver plots. `None` indicates all points are used.
            \**kwargs:
                Additional keyword arguments are passed to
                :meth:`~pde.field.base.DataFieldBase.get_vector_data` and
                :func:`matplotlib.pyplot.quiver` or
                :func:`matplotlib.pyplot.streamplot`.

        Returns:
            :class:`PlotReference`: Instance that contains information to update
            the plot with new data later.
        """
        # store the parameters of this plot for later updating
        parameters: dict[str, Any] = {"method": method, "kwargs": kwargs}

        # obtain parameter used to extract vector data
        data_kws = {}
        for arg in ["performance_goal", "transpose"]:
            if arg in kwargs:
                data_kws[arg] = kwargs.pop(arg)

        if method == "quiver":
            # plot vector field using a quiver plot
            data_kws["max_points"] = max_points
            data = self.get_vector_data(**data_kws)
            element = ax.quiver(
                data["x"], data["y"], data["data_x"].T, data["data_y"].T, **kwargs
            )

        elif method == "streamplot":
            # plot vector field using a streamplot
            data = self.get_vector_data(**data_kws)
            element = ax.streamplot(
                data["x"], data["y"], data["data_x"].T, data["data_y"].T, **kwargs
            )

        else:
            raise ValueError(f"Vector plot `{method}` is not supported.")
        parameters["data_kws"] = data_kws  # save data parameters

        # set some default properties of the plot
        ax.set_aspect("equal")
        ax.set_xlabel(data["label_x"])
        ax.set_ylabel(data["label_y"])
        ax.set_title(data.get("title", self.label))

        return PlotReference(ax, element, parameters)

    def _update_vector_plot(self, reference: PlotReference) -> None:
        """Update a vector plot with the current field values.

        Args:
            reference (:class:`PlotReference`):
                The reference to the plot that is updated
        """
        # extract general parameters
        method = reference.parameters.get("method", "quiver")
        data_kws = reference.parameters.get("data_kws", {})

        if method == "quiver":
            # update the data of a quiver plot
            data = self.get_vector_data(**data_kws)
            reference.element.set_UVC(data["data_x"], data["data_y"])

        elif method == "streamplot":
            # update a streamplot by redrawing it completely
            ax = reference.ax
            kwargs = reference.parameters.get("kwargs", {})
            data = self.get_vector_data(**data_kws)
            # remove old streamplot
            ax.cla()
            # update with new streamplot
            reference.element = ax.streamplot(
                data["x"], data["y"], data["data_x"].T, data["data_y"].T, **kwargs
            )

        else:
            raise ValueError(f"Vector plot `{method}` is not supported.")

    def _update_plot(self, reference: PlotReference) -> None:
        """Update a plot with the current field values.

        Args:
            reference (:class:`PlotReference`):
                The reference to the plot to updated
        """
        import matplotlib as mpl

        # update the plot based on the given reference
        el = reference.element
        if isinstance(el, mpl.lines.Line2D):
            self._update_line_plot(reference)
        elif isinstance(el, mpl.image.AxesImage):
            self._update_image_plot(reference)
        elif isinstance(el, (mpl.quiver.Quiver, mpl.streamplot.StreamplotSet)):
            self._update_vector_plot(reference)
        else:
            raise ValueError(f"Unknown plot element {el.__class__.__name__}")

    @plot_on_axes(update_method="_update_plot")
    def plot(self, kind: str = "auto", **kwargs) -> PlotReference:
        r"""Visualize the field.

        Args:
            kind (str):
                Determines the visualizations. Supported values are `image`,
                `line`, `vector`, or `interactive`. Alternatively, `auto`
                determines the best visualization based on the field itself.
            {PLOT_ARGS}
            \**kwargs:
                All additional keyword arguments are forwarded to the actual
                plotting function determined by `kind`.

        Returns:
            :class:`~pde.tools.plotting.PlotReference`: Instance that contains
            information to update the plot with new data later.

        Tip:
            Typical additional arguments for the various plot kinds include

            * :code:`kind == "line"`:

              - `scalar`: Sets method for extracting scalars as described in
                :meth:`DataFieldBase.to_scalar`.
              - `extract`: Method used for extracting the line data.
              - `ylabel`: Label of the y-axis.
              - `ylim`: Data limits of the y-axis.
              - Additional arguments are passed to :func:`matplotlib.pyplot.plot`

            * :code:`kind == "image"`:

              - `colorbar`: Determines whether a colorbar is shown
              - `scalar`: Sets method for extracting scalars as described in
                 :meth:`DataFieldBase.to_scalar`.
              - `transpose` Determines whether the transpose of the data is plotted
              - Most remaining arguments are passed to :func:`matplotlib.pyplot.imshow`

            * :code:`kind == `"vector"`:

              - `method` Can be either `quiver` or `streamplot`
              - `transpose` Determines whether the transpose of the data is plotted
              - `max_points` Sets max. number of points along each axis in quiver plots
              - Additional arguments are passed to :func:`matplotlib.pyplot.quiver` or
                :func:`matplotlib.pyplot.streamplot`.
        """
        # determine the correct kind of plotting
        if kind == "auto":
            # determine best plot for this field
            if (
                isinstance(self, DataFieldBase)
                and self.rank == 1
                and self.grid.dim == 2
            ):
                kind = "vector"
            elif self.grid.num_axes == 1:
                kind = "line"
            else:
                kind = "image"

        elif kind == "quiver":
            kind = "vector"
            kwargs["method"] = "quiver"

        elif kind == "streamplot":
            kind = "vector"
            kwargs["method"] = "streamplot"

        # do the actual plotting
        if kind == "image":
            reference = self._plot_image(**kwargs)
        elif kind == "line":
            reference = self._plot_line(**kwargs)
        elif kind == "vector":
            reference = self._plot_vector(**kwargs)
        else:
            raise ValueError(
                f"Unsupported plot `{kind}`. Possible choices are `image`, `line`, "
                "`vector`, or `auto`."
            )

        return reference

    def _get_napari_layer_data(
        self, scalar: str = "auto", args: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Returns data for plotting on a single napari layer.

        Args:
            scalar (str):
                Indicates how the scalar field is generated; see `to_scalar`
            args (dict):
                Additional arguments returned in the result, which affect how the layer
                is shown.

        Returns:
            dict: all the information necessary to plot this field
        """
        result = {} if args is None else args.copy()

        result.setdefault("scale", self.grid.discretization)
        result.setdefault("rgb", False)
        result["type"] = "image"
        result["data"] = self.to_scalar(scalar).data
        return result

    def _get_napari_data(self, **kwargs) -> dict[str, dict[str, Any]]:
        r"""Returns data for plotting this field using :mod:`napari`

        Args:
            \**kwargs: all arguments are forwarded to `_get_napari_layer_data`

        Returns:
            dict: all the information necessary to plot this field
        """
        name = "Field" if self.label is None else self.label
        return {name: self._get_napari_layer_data(**kwargs)}
