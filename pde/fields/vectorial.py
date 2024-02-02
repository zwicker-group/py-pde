"""
Defines a vectorial field over a grid

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable, Literal

import numba as nb
import numpy as np
from numba.extending import overload, register_jitable
from numpy.typing import DTypeLike

from ..grids.base import DimensionError, GridBase
from ..grids.boundaries.axes import BoundariesData
from ..grids.cartesian import CartesianGrid
from ..tools.docstrings import fill_in_docstring
from ..tools.misc import Number, get_common_dtype
from ..tools.numba import get_common_numba_dtype, jit
from ..tools.typing import NumberOrArray
from .base import DataFieldBase
from .scalar import ScalarField

if TYPE_CHECKING:
    from .tensorial import Tensor2Field


class VectorField(DataFieldBase):
    """Vector field discretized on a grid

    Warning:
        Components of the vector field are given in the local basis. While the local
        basis is identical to the global basis in Cartesian coordinates, the local basis
        depends on position in curvilinear coordinate systems. Moreover, the field
        always contains all components, even if the underlying grid assumes symmetries.
    """

    rank = 1

    @classmethod
    def from_scalars(
        cls,
        fields: list[ScalarField],
        *,
        label: str | None = None,
        dtype: DTypeLike | None = None,
    ) -> VectorField:
        """create a vector field from a list of ScalarFields

        Note that the data of the scalar fields is copied in the process

        Args:
            fields (list):
                The list of (compatible) scalar fields
            label (str, optional):
                Name of the returned field
            dtype (numpy dtype):
                The data type of the field. If omitted, it will be determined from
                `data` automatically.

        Returns:
            :class:`VectorField`: the resulting vector field
        """
        grid = fields[0].grid

        if grid.dim != len(fields):
            raise DimensionError(
                f"Grid dimension and number of scalar fields differ ({grid.dim} != "
                f"{len(fields)})"
            )

        data = []
        for field in fields:
            assert field.grid.compatible_with(grid)
            data.append(field.data)

        return cls(grid, data, label=label, dtype=dtype)

    @classmethod
    @fill_in_docstring
    def from_expression(
        cls,
        grid: GridBase,
        expressions: Sequence[str],
        *,
        user_funcs: dict[str, Callable] | None = None,
        consts: dict[str, NumberOrArray] | None = None,
        label: str | None = None,
        dtype: DTypeLike | None = None,
    ) -> VectorField:
        """create a vector field on a grid from given expressions

        Warning:
            {WARNING_EXEC}

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                Grid defining the space on which this field is defined
            expressions (list of str):
                A list of mathematical expression, one for each component of the vector
                field. The expressions determine the values as a function of the
                position on the grid. The expressions may contain standard mathematical
                functions and they may depend on the axes labels of the grid.
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
                Name of the field
            dtype (numpy dtype):
                The data type of the field. If omitted, it will be determined from
                `data` automatically.
        """
        from ..tools.expressions import ScalarExpression

        if isinstance(expressions, str) or len(expressions) != grid.dim:
            axes_names = grid.axes + grid.axes_symmetric
            raise DimensionError(
                f"Expected {grid.dim} expressions for the coordinates {axes_names}."
            )

        # obtain the coordinates of the grid points
        points = [grid.cell_coords[..., i] for i in range(grid.num_axes)]

        # evaluate all vector components at all points
        data = []
        for expression in expressions:
            expr = ScalarExpression(
                expression=expression,
                signature=grid.axes,
                user_funcs=user_funcs,
                consts=consts,
                repl=grid.c._axes_alt_repl,
            )
            values = np.broadcast_to(expr(*points), grid.shape)
            data.append(values)

        # create vector field from the data
        return cls(grid=grid, data=data, label=label, dtype=dtype)

    def __getitem__(self, key: int | str) -> ScalarField:
        """extract a component of the VectorField"""
        axis = self.grid.get_axis_index(key)
        comp_name = self.grid.c.axes[axis]
        if self.label:
            label = self.label + f"_{comp_name}"
        else:
            label = f"{comp_name} component"
        return ScalarField(
            self.grid, data=self._data_full[axis], label=label, with_ghost_cells=True
        )

    def __setitem__(self, key: int | str, value: NumberOrArray | ScalarField):
        """set a component of the VectorField"""
        idx = self.grid.get_axis_index(key)
        if isinstance(value, ScalarField):
            self.grid.assert_grid_compatible(value.grid)
            self.data[idx] = value.data
        else:
            self.data[idx] = value

    def dot(
        self,
        other: VectorField | Tensor2Field,
        out: ScalarField | VectorField | None = None,
        *,
        conjugate: bool = True,
        label: str = "dot product",
    ) -> ScalarField | VectorField:
        """calculate the dot product involving a vector field

        This supports the dot product between two vectors fields as well as the
        product between a vector and a tensor. The resulting fields will be a
        scalar or vector, respectively.

        Args:
            other (VectorField or Tensor2Field):
                the second field
            out (ScalarField or VectorField, optional):
                Optional field to which the result is written.
            conjugate (bool):
                Whether to use the complex conjugate for the second operand
            label (str, optional):
                Name of the returned field

        Returns:
            :class:`~pde.fields.scalar.ScalarField` or
            :class:`~pde.fields.vectorial.VectorField`: result of applying the operator
        """
        from .tensorial import Tensor2Field  # @Reimport

        # check input
        self.grid.assert_grid_compatible(other.grid)
        if isinstance(other, VectorField):
            result_type = ScalarField
        elif isinstance(other, Tensor2Field):
            result_type = VectorField  # type: ignore
        else:
            raise TypeError("Second term must be a vector or tensor field")

        if out is None:
            out = result_type(self.grid, dtype=get_common_dtype(self, other))
        else:
            assert isinstance(out, result_type), f"`out` must be {result_type}"
            self.grid.assert_grid_compatible(out.grid)

        # calculate the result
        other_data = other.data.conjugate() if conjugate else other.data
        np.einsum("i...,i...->...", self.data, other_data, out=out.data)
        if label is not None:
            out.label = label

        return out

    __matmul__ = dot  # support python @-syntax for matrix multiplication

    def outer_product(
        self,
        other: VectorField,
        out: Tensor2Field | None = None,
        *,
        label: str | None = None,
    ) -> Tensor2Field:
        """calculate the outer product of this vector field with another

        Args:
            other (:class:`~pde.fields.vectorial.VectorField`):
                The second vector field
            out (:class:`~pde.fields.tensorial.Tensor2Field`, optional):
                Optional tensorial field to which the  result is written.
            label (str, optional):
                Name of the returned field

        Returns:
            :class:`~pde.fields.tensorial.Tensor2Field`: result of the operation
        """
        from .tensorial import Tensor2Field  # @Reimport

        self.assert_field_compatible(other)

        if out is None:
            out = Tensor2Field(self.grid)
        else:
            self.grid.assert_grid_compatible(out.grid)

        # calculate the result
        np.einsum("i...,j...->ij...", self.data, other.data, out=out.data)
        if label is not None:
            out.label = label

        return out

    def make_outer_prod_operator(
        self, backend: Literal["numpy", "numba"] = "numba"
    ) -> Callable[[np.ndarray, np.ndarray, np.ndarray | None], np.ndarray]:
        """return operator calculating the outer product of two vector fields

        Warning:
            This function does not check types or dimensions.

        Args:
            backend (str):
                The backend (e.g., 'numba' or 'scipy') used for this operator.

        Returns:
            function that takes two instance of :class:`~numpy.ndarray`, which contain
            the discretized data of the two operands. An optional third argument can
            specify the output array to which the result is written. Note that the
            returned function is jitted with numba for speed.
        """

        def outer(
            a: np.ndarray, b: np.ndarray, out: np.ndarray | None = None
        ) -> np.ndarray:
            """calculate the outer product using numpy"""
            return np.einsum("i...,j...->ij...", a, b, out=out)

        if backend == "numpy":
            # return the bare dot operator without the numba-overloaded version
            return outer

        elif backend == "numba":
            # overload `outer` with a numba-compiled version

            dim = self.grid.dim
            num_axes = self.grid.num_axes

            def check_rank(arr: nb.types.Type | nb.types.Optional) -> None:
                """determine rank of field with type `arr`"""
                arr_typ = arr.type if isinstance(arr, nb.types.Optional) else arr
                if not isinstance(arr_typ, (np.ndarray, nb.types.Array)):
                    raise nb.errors.TypingError(
                        f"Arguments must be array, not  {arr_typ.__class__}"
                    )
                assert arr_typ.ndim == 1 + num_axes

            # create the inner function calculating the outer product
            @register_jitable
            def calc(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> np.ndarray:
                """calculate outer product between fields `a` and `b`"""
                for i in range(0, dim):
                    for j in range(0, dim):
                        out[i, j, :] = a[i] * b[j]
                return out

            @overload(outer, inline="always")
            def outer_ol(
                a: np.ndarray, b: np.ndarray, out: np.ndarray | None = None
            ) -> np.ndarray:
                """numba implementation to calculate outer product between two fields"""
                # get (and check) rank of the input arrays
                check_rank(a)
                check_rank(b)
                in_shape = (dim,) + self.grid.shape
                out_shape = (dim, dim) + self.grid.shape

                if isinstance(out, (nb.types.NoneType, nb.types.Omitted)):
                    # function is called without `out` -> allocate memory
                    dtype = get_common_numba_dtype(a, b)

                    def outer_impl(
                        a: np.ndarray, b: np.ndarray, out: np.ndarray | None = None
                    ) -> np.ndarray:
                        """helper function allocating output array"""
                        assert a.shape == b.shape == in_shape
                        out = np.empty(out_shape, dtype=dtype)
                        calc(a, b, out)
                        return out

                else:
                    # function is called with `out` argument -> reuse `out` array

                    def outer_impl(
                        a: np.ndarray, b: np.ndarray, out: np.ndarray | None = None
                    ) -> np.ndarray:
                        """helper function without allocating output array"""
                        # check input
                        assert a.shape == b.shape == in_shape
                        assert out.shape == out_shape  # type: ignore
                        calc(a, b, out)
                        return out  # type: ignore

                return outer_impl  # type: ignore

            @jit
            def outer_compiled(
                a: np.ndarray, b: np.ndarray, out: np.ndarray | None = None
            ) -> np.ndarray:
                """numba implementation to calculate outer product between two fields"""
                return outer(a, b, out)

            return outer_compiled  # type: ignore

        else:
            raise ValueError(f"Unsupported backend `{backend}")

    @fill_in_docstring
    def divergence(
        self, bc: BoundariesData | None, out: ScalarField | None = None, **kwargs
    ) -> ScalarField:
        """apply divergence operator and return result as a field

        Args:
            bc:
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES_OPTIONAL}
            out (ScalarField, optional):
                Optional scalar field to which the  result is written.
            label (str, optional):
                Name of the returned field
            **kwargs:
                Additional arguments affecting how the operator behaves.

        Returns:
            :class:`~pde.fields.scalar.ScalarField`: Divergence of the field
        """
        return self.apply_operator("divergence", bc=bc, out=out, **kwargs)  # type: ignore

    @fill_in_docstring
    def gradient(
        self,
        bc: BoundariesData | None,
        out: Tensor2Field | None = None,
        **kwargs,
    ) -> Tensor2Field:
        r"""apply vector gradient operator and return result as a field

        The vector gradient field is a tensor field :math:`t_{\alpha\beta}` that
        specifies the derivatives of the vector field :math:`v_\alpha` with respect to
        all coordinates :math:`x_\beta`.

        Args:
            bc:
                The boundary conditions applied to the field. Boundary conditions need
                to determine all components of the vector field.
                {ARG_BOUNDARIES_OPTIONAL}
            out (VectorField, optional):
                Optional vector field to which the result is written.
            label (str, optional):
                Name of the returned field
            **kwargs:
                Additional arguments affecting how the operator behaves.

        Returns:
            :class:`~pde.fields.tensorial.Tensor2Field`: Gradient of the field
        """
        return self.apply_operator("vector_gradient", bc=bc, out=out, **kwargs)  # type: ignore

    @fill_in_docstring
    def laplace(
        self, bc: BoundariesData | None, out: VectorField | None = None, **kwargs
    ) -> VectorField:
        r"""apply vector Laplace operator and return result as a field

        The vector Laplacian is a vector field :math:`L_\alpha` containing the second
        derivatives of the vector field :math:`v_\alpha` with respect to the coordinates
        :math:`x_\beta`:

        .. math::
            L_\alpha = \sum_\beta
                \frac{\partial^2 v_\alpha}{\partial x_\beta \partial x_\beta}

        Args:
            bc:
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES_OPTIONAL}
            out (VectorField, optional):
                Optional vector field to which the  result is written.
            label (str, optional):
                Name of the returned field
            **kwargs:
                Additional arguments affecting how the operator behaves.

        Returns:
            :class:`~pde.fields.vectorial.VectorField`: Laplacian of the field
        """
        return self.apply_operator("vector_laplace", bc=bc, out=out, **kwargs)  # type: ignore

    @property
    def integral(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: integral of each component over space"""
        return self.grid.integrate(self.data)  # type: ignore

    def to_scalar(
        self,
        scalar: str | int = "auto",
        *,
        label: str | None = "scalar `{scalar}`",
    ) -> ScalarField:
        """return scalar variant of the field

        Args:
            scalar (str):
                Choose the method to use. Possible  choices are `norm`, `max`, `min`,
                `squared_sum`, `norm_squared`, or an integer specifying which component
                is returned (indexing starts at `0`). The default value `auto` picks the
                method automatically: The first (and only) component is returned for
                real fields on one-dimensional spaces, while the norm of the vector is
                returned otherwise.
            label (str, optional):
                Name of the returned field

        Returns:
            :class:`pde.fields.scalar.ScalarField`:
                The scalar field after applying the operation
        """
        if scalar == "auto":
            if self.grid.dim > 1 or np.iscomplexobj(self.data):
                scalar = "norm"
            else:
                scalar = 0  # return the field unchanged

        if isinstance(scalar, int):
            data = self.data[scalar]

        elif scalar == "norm":
            data = np.linalg.norm(self.data, axis=0)

        elif scalar == "max":
            data = np.max(self.data, axis=0)

        elif scalar == "min":
            data = np.min(self.data, axis=0)

        elif scalar == "squared_sum":
            data = np.sum(self.data**2, axis=0)

        elif scalar == "norm_squared":
            data = np.sum(self.data * self.data.conjugate(), axis=0)

        else:
            raise ValueError(f"Unknown method `{scalar}` for `to_scalar`")

        if label is not None:
            label = label.format(scalar=scalar)

        return ScalarField(self.grid, data, label=label)

    def get_vector_data(
        self, transpose: bool = False, max_points: int | None = None, **kwargs
    ) -> dict[str, Any]:
        r"""return data for a vector plot of the field

        Args:
            transpose (bool):
                Determines whether the transpose of the data should be plotted.
            max_points (int):
                The maximal number of points that is used along each axis. This
                option can be used to sub-sample the data.
            \**kwargs:
                Additional parameters forwarded to `grid.get_image_data`

        Returns:
            dict: Information useful for plotting an vector field
        """
        if self.is_complex:
            self._logger.warning("Only the real part of complex data is shown")

        # extract the image data
        data = self.grid.get_vector_data(self.data.real, **kwargs)
        data["title"] = self.label

        # transpose the data if requested
        if transpose:
            data["x"], data["y"] = data["y"], data["x"]
            data["data_x"], data["data_y"] = data["data_y"].T, data["data_x"].T
            data["label_x"], data["label_y"] = data["label_y"], data["label_x"]
            data["extent"] = data["extent"][2:] + data["extent"][:2]

        # reduce the sampling of the vector points
        if max_points is not None:
            shape = data["data_x"].shape
            for axis, size in enumerate(shape):
                if size > max_points:
                    # sub-sample the data
                    idx_f = np.linspace(0, size - 1, max_points)
                    idx_i = np.round(idx_f).astype(int)

                    data["data_x"] = np.take(data["data_x"], idx_i, axis=axis)
                    data["data_y"] = np.take(data["data_y"], idx_i, axis=axis)
                    if axis == 0:
                        data["x"] = data["x"][idx_i]
                    elif axis == 1:
                        data["y"] = data["y"][idx_i]
                    else:
                        raise RuntimeError("Only supports 2d grids")

        data["shape"] = data["data_x"].shape
        data["size"] = data["data_x"].size

        return data

    @fill_in_docstring
    def interpolate_to_grid(
        self: VectorField,
        grid: GridBase,
        *,
        bc: BoundariesData | None = None,
        fill: Number | None = None,
        label: str | None = None,
    ) -> VectorField:
        """interpolate the data of this vector field to another grid.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid of the new field onto which the current field is interpolated.
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
        if self.grid.dim != grid.dim:
            raise DimensionError(
                f"Incompatible grid dimensions ({self.grid.dim:d} != {grid.dim:d})"
            )

        # determine the points at which data needs to be calculated
        if isinstance(grid, CartesianGrid):
            # convert Cartesian coordinates to coordinates in current grid
            points = self.grid.c.pos_from_cart(grid.cell_coords)
            points_grid_sym = self.grid._coords_symmetric(points)
            # interpolate the data to the grid; this gives the vector in the grid basis
            data_grid = self.interpolate(points_grid_sym, bc=bc, fill=fill)
            # convert the vector to the cartesian basis
            data = self.grid._vector_to_cartesian(points, data_grid)

        elif (
            self.grid.__class__ is grid.__class__
            and self.grid.num_axes == grid.num_axes
        ):
            # convert within the same grid class
            points = grid.cell_coords
            # vectors are already given in the correct basis
            data = self.interpolate(points, bc=bc, fill=fill)

        else:
            # this type of interpolation is not supported
            grid_in = self.grid.__class__.__name__
            grid_out = grid.__class__.__name__
            raise NotImplementedError(f"Can't interpolate from {grid_in} to {grid_out}")

        return self.__class__(grid, data, label=label)

    def _get_napari_layer_data(  # type: ignore
        self, max_points: int | None = None, args: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """returns data for plotting on a single napari layer

        Args:
            max_points (int):
                The maximal number of points that is used along each axis. This option
                can be used to subsample the data.
            args (dict):
                Additional arguments returned in the result, which affect how the layer
                is shown.

        Returns:
            dict: all the information necessary to plot this field
        """
        result = {} if args is None else args.copy()

        # extract the vector components in the format required by napari
        data = self.get_vector_data(max_points=max_points)
        vectors = np.empty((data["size"], 2, 2))
        xs, ys = np.meshgrid(data["x"], data["y"], indexing="ij")
        vectors[:, 0, 0] = xs.flat
        vectors[:, 0, 1] = ys.flat
        vectors[:, 1, 0] = data["data_x"].flat
        vectors[:, 1, 1] = data["data_y"].flat

        result["type"] = "vectors"
        result["data"] = vectors
        return result
