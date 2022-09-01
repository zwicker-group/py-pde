"""
Defines a vectorial field over a grid

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

import numba as nb
import numpy as np
from numpy.typing import DTypeLike

try:
    from numba.core.extending import register_jitable
except ImportError:
    # assume older numba module structure
    from numba.extending import register_jitable

from ..grids.base import DimensionError, GridBase
from ..tools.docstrings import fill_in_docstring
from ..tools.misc import get_common_dtype
from ..tools.numba import get_common_numba_dtype
from ..tools.typing import NumberOrArray
from .base import DataFieldBase
from .scalar import ScalarField

if TYPE_CHECKING:
    from ..grids.boundaries.axes import BoundariesData  # @UnusedImport
    from .tensorial import Tensor2Field  # @UnusedImport


class VectorField(DataFieldBase):
    """Vector field discretized on a grid"""

    rank = 1

    @classmethod
    def from_scalars(
        cls, fields: List[ScalarField], *, label: str = None, dtype: DTypeLike = None
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
        label: str = None,
        dtype: DTypeLike = None,
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
        points = {name: grid.cell_coords[..., i] for i, name in enumerate(grid.axes)}

        # evaluate all vector components at all points
        data = []
        for expression in expressions:
            expr = ScalarExpression(expression=expression, signature=grid.axes)
            values = np.broadcast_to(expr(**points), grid.shape)
            data.append(values)

        # create vector field from the data
        return cls(grid=grid, data=data, label=label, dtype=dtype)

    def __getitem__(self, key: Union[int, str]) -> ScalarField:
        """extract a component of the VectorField"""
        return ScalarField(
            self.grid,
            data=self._data_full[self.grid.get_axis_index(key)],
            with_ghost_cells=True,
        )

    def __setitem__(
        self, key: Union[int, str], value: Union[NumberOrArray, ScalarField]
    ):
        """set a component of the VectorField"""
        idx = self.grid.get_axis_index(key)
        if isinstance(value, ScalarField):
            self.grid.assert_grid_compatible(value.grid)
            self.data[idx] = value.data
        else:
            self.data[idx] = value

    def dot(
        self,
        other: Union[VectorField, "Tensor2Field"],
        out: Optional[Union[ScalarField, VectorField]] = None,
        *,
        conjugate: bool = True,
        label: str = "dot product",
    ) -> Union[ScalarField, VectorField]:
        """calculate the dot product involving a vector field

        This supports the dot product between two vectors fields as well as the
        product between a vector and a tensor. The resulting fields will be a
        scalar or vector, respectively.

        Args:
            other (VectorField or Tensor2Field):
                the second field
            out (ScalarField or VectorField, optional):
                Optional field to which the  result is written.
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

    def make_dot_operator(
        self, backend: str = "numba", *, conjugate: bool = True
    ) -> Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray]:
        """return operator calculating the dot product involving vector fields

        This supports both products between two vectors as well as products between a
        vector and a tensor.

        Warning:
            This function does not check types or dimensions.

        Args:
            backend (str):
                The backend (e.g., 'numba' or 'scipy') used for this operator.
            conjugate (bool):
                Whether to use the complex conjugate for the second operand

        Returns:
            function that takes two instance of :class:`~numpy.ndarray`, which contain
            the discretized data of the two operands. An optional third argument can
            specify the output array to which the result is written. Note that the
            returned function is jitted with numba for speed.
        """
        dim = self.grid.dim

        if backend == "numba":
            # create the dot product using a numba compiled function

            if conjugate:
                # create inner function calculating the dot product using conjugate

                @register_jitable
                def calc(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> np.ndarray:
                    """calculate dot product between fields `a` and `b`"""
                    out[:] = a[0] * b[0].conjugate()  # overwrite potential data in out
                    for i in range(1, dim):
                        out[:] += a[i] * b[i].conjugate()
                    return out

            else:
                # create the inner function calculating the dot product

                @register_jitable
                def calc(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> np.ndarray:
                    """calculate dot product between fields `a` and `b`"""
                    out[:] = a[0] * b[0]  # overwrite potential data in out
                    for i in range(1, dim):
                        out[:] += a[i] * b[i]
                    return out

            # build the outer function with the correct signature
            if nb.config.DISABLE_JIT:  # @UndefinedVariable

                def dot(
                    a: np.ndarray, b: np.ndarray, out: np.ndarray = None
                ) -> np.ndarray:
                    """wrapper deciding whether the underlying function is called
                    with or without `out`."""
                    if out is None:
                        out = np.empty(b.shape[1:], dtype=get_common_dtype(a, b))
                    return calc(a, b, out)  # type: ignore

            else:

                @nb.generated_jit
                def dot(
                    a: np.ndarray, b: np.ndarray, out: np.ndarray = None
                ) -> np.ndarray:
                    """wrapper deciding whether the underlying function is called
                    with or without `out`."""
                    if isinstance(a, nb.types.Number):
                        # simple scalar call -> do not need to allocate anything
                        raise RuntimeError("Dot needs to be called with fields")

                    elif isinstance(out, (nb.types.NoneType, nb.types.Omitted)):
                        # function is called without `out`
                        dtype = get_common_numba_dtype(a, b)

                        def f_with_allocated_out(
                            a: np.ndarray, b: np.ndarray, out: np.ndarray
                        ) -> np.ndarray:
                            """helper function allocating output array"""
                            out = np.empty(b.shape[1:], dtype=dtype)
                            return calc(a, b, out=out)  # type: ignore

                        return f_with_allocated_out  # type: ignore

                    else:
                        # function is called with `out` argument
                        return calc  # type: ignore

        elif backend == "numpy":
            # create the dot product using basic numpy functions

            def calc(
                a: np.ndarray, b: np.ndarray, out: np.ndarray = None
            ) -> np.ndarray:
                """inner function doing the actual calculation of the dot product"""
                if a.shape == b.shape:
                    # dot product between vector and vector
                    if out is None:
                        # TODO: Remove this construct once we make numpy 1.20 a minimal
                        # requirement. Earlier version of numpy do not support out=None
                        # correctly and we thus had to use this work-around
                        return np.einsum("i...,i...->...", a, b)  # type: ignore
                    else:
                        return np.einsum("i...,i...->...", a, b, out=out)

                elif a.shape == b.shape[1:]:
                    # dot product between vector and tensor
                    if out is None:
                        # TODO: Remove this construct once we make numpy 1.20 a minimal
                        # requirement. Earlier version of numpy do not support out=None
                        # correctly and we thus had to use this work-around
                        return np.einsum("i...,ij...->j...", a, b)  # type: ignore
                    else:
                        return np.einsum("i...,ij...->j...", a, b, out=out)

                else:
                    raise ValueError(f"Unsupported shapes ({a.shape}, {b.shape})")

            if conjugate:
                # create inner function calculating the dot product using conjugate

                def dot(
                    a: np.ndarray, b: np.ndarray, out: np.ndarray = None
                ) -> np.ndarray:
                    """dot product with conjugated second operand"""
                    return calc(a, b.conjugate(), out=out)  # type: ignore

            else:
                dot = calc

        else:
            raise ValueError(f"Undefined backend `{backend}")

        return dot

    def outer_product(
        self, other: VectorField, out: "Tensor2Field" = None, *, label: str = None
    ) -> "Tensor2Field":
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
        self, backend: str = "numba"
    ) -> Callable[[np.ndarray, np.ndarray, Optional[np.ndarray]], np.ndarray]:
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
        dim = self.grid.dim

        if backend == "numba":
            # create the dot product using a numba compiled function

            # create the inner function calculating the dot product
            @register_jitable
            def calc(a: np.ndarray, b: np.ndarray, out: np.ndarray) -> np.ndarray:
                """calculate dot product between fields `a` and `b`"""
                for i in range(0, dim):
                    for j in range(0, dim):
                        out[i, j, :] = a[i] * b[j]
                return out

            # build the outer function with the correct signature
            if nb.config.DISABLE_JIT:

                def outer(
                    a: np.ndarray, b: np.ndarray, out: np.ndarray = None
                ) -> np.ndarray:
                    """wrapper deciding whether the underlying function is called
                    with or without `out`."""
                    if out is None:
                        out = np.empty(
                            (len(a),) + b.shape, dtype=get_common_dtype(a, b)
                        )
                    return calc(a, b, out)  # type: ignore

            else:

                @nb.generated_jit
                def outer(
                    a: np.ndarray, b: np.ndarray, out: np.ndarray = None
                ) -> np.ndarray:
                    """wrapper deciding whether the underlying function is called
                    with or without `out`."""
                    if isinstance(a, nb.types.Number):
                        # simple scalar call -> do not need to allocate anything
                        raise RuntimeError("Dot needs to be called with fields")

                    elif isinstance(out, (nb.types.NoneType, nb.types.Omitted)):
                        # function is called without `out`
                        dtype = get_common_numba_dtype(a, b)

                        def f_with_allocated_out(
                            a: np.ndarray, b: np.ndarray, out: np.ndarray
                        ) -> np.ndarray:
                            """helper function allocating output array"""
                            out = np.empty((len(a),) + b.shape, dtype=dtype)
                            return calc(a, b, out=out)  # type: ignore

                        return f_with_allocated_out  # type: ignore

                    else:
                        # function is called with `out` argument
                        return calc  # type: ignore

        elif backend == "numpy":
            # create the dot product using basic numpy functions

            def outer(
                a: np.ndarray, b: np.ndarray, out: np.ndarray = None
            ) -> np.ndarray:
                """calculates the outer product between two vector fields"""
                if out is None:
                    # TODO: Remove this construct once we make numpy 1.20 a minimal
                    # requirement. Earlier version of numpy do not support out=None
                    # correctly and we thus had to use this work-around
                    return np.einsum("i...,j...->ij...", a, b)  # type: ignore
                else:
                    return np.einsum("i...,j...->ij...", a, b, out=out)

        else:
            raise ValueError(f"Undefined backend `{backend}")

        return outer

    @fill_in_docstring
    def divergence(
        self, bc: Optional[BoundariesData], out: Optional[ScalarField] = None, **kwargs
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
            :class:`~pde.fields.scalar.ScalarField`: result of applying the operator
        """
        return self._apply_operator("divergence", bc=bc, out=out, **kwargs)  # type: ignore

    @fill_in_docstring
    def gradient(
        self,
        bc: Optional[BoundariesData],
        out: Optional["Tensor2Field"] = None,
        **kwargs,
    ) -> "Tensor2Field":
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
            :class:`~pde.fields.tensorial.Tensor2Field`: result of applying the operator
        """
        return self._apply_operator("vector_gradient", bc=bc, out=out, **kwargs)  # type: ignore

    @fill_in_docstring
    def laplace(
        self, bc: Optional[BoundariesData], out: Optional[VectorField] = None, **kwargs
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
            :class:`~pde.fields.vectorial.VectorField`: result of applying the operator
        """
        return self._apply_operator("vector_laplace", bc=bc, out=out, **kwargs)  # type: ignore

    @property
    def integral(self) -> np.ndarray:
        """:class:`~numpy.ndarray`: integral of each component over space"""
        return self.grid.integrate(self.data)  # type: ignore

    def to_scalar(
        self,
        scalar: Union[str, int] = "auto",
        *,
        label: Optional[str] = "scalar `{scalar}`",
    ) -> ScalarField:
        """return a scalar field by applying `method`

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
            :class:`pde.fields.scalar.ScalarField`: the scalar field after
            applying the operation
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
        self, transpose: bool = False, max_points: int = None, **kwargs
    ) -> Dict[str, Any]:
        r"""return data for a vector plot of the field

        Args:
            transpose (bool):
                Determines whether the transpose of the data should be plotted.
            max_points (int):
                The maximal number of points that is used along each axis. This
                option can be used to sub-sample the data.
            \**kwargs: Additional parameters are forwarded to
                `grid.get_image_data`

        Returns:
            dict: Information useful for plotting an vector field
        """
        # TODO: Handle Spherical and Cartesian grids, too. This could be
        # implemented by adding a get_vector_data method to the grids
        if self.grid.dim == 2:
            vx = self[0].get_image_data(**kwargs)
            vy = self[1].get_image_data(**kwargs)
            data = vx  # use one of the fields to extract basic information
            data["data_x"] = vx.pop("data")
            data["data_y"] = vy["data"]
            data["title"] = self.label

        else:
            raise NotImplementedError("Only supports 2d grids")

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

    def _get_napari_layer_data(  # type: ignore
        self, max_points: int = None, args: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """returns data for plotting on a single napari layer

        Args:
            max_points (int):
                The maximal number of points that is used along each axis. This
                option can be used to subsample the data.
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
