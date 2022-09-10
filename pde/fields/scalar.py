"""
Defines a scalar field over a grid

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import numbers
from pathlib import Path
from typing import List  # @UnusedImport
from typing import TYPE_CHECKING, Callable, Dict, Optional, Sequence, Union

import numpy as np
from numpy.typing import DTypeLike

from ..grids import CartesianGrid, UnitGrid
from ..grids.base import DomainError, GridBase
from ..tools.docstrings import fill_in_docstring
from ..tools.misc import Number
from .base import DataFieldBase

if TYPE_CHECKING:
    from ..grids.boundaries.axes import BoundariesData  # @UnusedImport
    from .vectorial import VectorField  # @UnusedImport


class ScalarField(DataFieldBase):
    """Scalar field discretized on a grid"""

    rank = 0

    @classmethod
    @fill_in_docstring
    def from_expression(
        cls,
        grid: GridBase,
        expression: str,
        *,
        label: str = None,
        dtype: DTypeLike = None,
    ) -> ScalarField:
        """create a scalar field on a grid from a given expression

        Warning:
            {WARNING_EXEC}

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                Grid defining the space on which this field is defined
            expression (str):
                Mathematical expression for the scalar value as a function of the
                position on the grid. The expression may contain standard mathematical
                functions and it may depend on the axes labels of the grid.
                More information can be found in the
                :ref:`expression documentation <documentation-expressions>`.
            label (str, optional):
                Name of the field
            dtype (numpy dtype):
                The data type of the field. If omitted, it will be determined from
                `data` automatically.
        """
        from ..tools.expressions import ScalarExpression

        expr = ScalarExpression(expression=expression, signature=grid.axes)
        points = {name: grid.cell_coords[..., i] for i, name in enumerate(grid.axes)}

        try:
            # try evaluating the expression using a vectorized call
            data = expr(**points)
        except ValueError:
            # if this fails, evaluate expression point-wise
            data = np.empty(grid.shape)
            for cells in np.ndindex(*grid.shape):
                data[cells] = expr(grid.cell_coords[cells])

        return cls(grid=grid, data=data, label=label, dtype=dtype)

    @classmethod
    def from_image(
        cls, path: Union[Path, str], bounds=None, periodic=False, *, label: str = None
    ) -> ScalarField:
        """create a scalar field from an image

        Args:
            path (:class:`Path` or str):
                The path to the image file
            bounds (tuple, optional):
                Gives the coordinate range for each axis. This should be two tuples of
                two numbers each, which mark the lower and upper bound for each axis.
            periodic (bool or list):
                Specifies which axes possess periodic boundary conditions. This is
                either a list of booleans defining periodicity for each individual axis
                or a single boolean value specifying the same periodicity for all axes.
            label (str, optional):
                Name of the field
        """
        from matplotlib.pyplot import imread

        # read image and convert to grayscale
        data = imread(str(path))
        if data.ndim == 2:
            pass  # is already gray scale
        elif data.ndim == 3:
            # convert to gray scale using ITU-R 601-2 luma transform:
            weights = np.array([0.299, 0.587, 0.114])
            data = data[..., :3] @ weights
        else:
            raise RuntimeError(f"Image data has wrong shape: {data.shape}")

        # transpose data to use mathematical conventions for axes
        data = data.T[:, ::-1]

        # determine the associated grid
        if bounds is None:
            grid: GridBase = UnitGrid(data.shape, periodic=periodic)
        else:
            grid = CartesianGrid(bounds, data.shape, periodic=periodic)

        return cls(grid, data, label=label)

    @DataFieldBase._data_flat.setter  # type: ignore
    def _data_flat(self, value):
        """set the data from a value from a collection"""
        self._data_full = value[0]

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """support unary numpy ufuncs, like np.sin, but also np.multiply"""
        if method == "__call__":
            # only support unary functions in simple calls

            # check the input
            arrs = []
            for arg in inputs:
                if isinstance(arg, numbers.Number):
                    arrs.append(arg)
                elif isinstance(arg, np.ndarray):
                    if arg.shape != self.data.shape:
                        raise RuntimeError("Data shapes incompatible")
                    arrs.append(arg)
                elif isinstance(arg, self.__class__):
                    self.assert_field_compatible(arg)
                    arrs.append(arg.data)
                else:
                    # unsupported type
                    return NotImplemented

            if "out" in kwargs:
                # write to given field
                out = kwargs.pop("out")[0]
                self.assert_field_compatible(out)
                kwargs["out"] = (out.data,)
                ufunc(*arrs, **kwargs)
                return out
            else:
                # return new field
                return self.__class__(self.grid, data=ufunc(*arrs, **kwargs))
        else:
            return NotImplemented

    @fill_in_docstring
    def laplace(
        self,
        bc: Optional[BoundariesData],
        out: Optional[ScalarField] = None,
        **kwargs,
    ) -> ScalarField:
        """apply Laplace operator and return result as a field

        Args:
            bc:
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES_OPTIONAL}
            out (ScalarField, optional):
                Optional scalar field to which the  result is written.
            label (str, optional):
                Name of the returned field
            backend (str):
                The backend (e.g., 'numba' or 'scipy') used for this operator.

        Returns:
            :class:`~pde.fields.scalar.ScalarField`: the Laplacian of the field
        """
        return self._apply_operator("laplace", bc=bc, out=out, **kwargs)  # type: ignore

    @fill_in_docstring
    def gradient_squared(
        self,
        bc: Optional[BoundariesData],
        out: Optional[ScalarField] = None,
        **kwargs,
    ) -> ScalarField:
        r"""apply squared gradient operator and return result as a field

        This evaluates :math:`|\nabla \phi|^2` for the scalar field :math:`\phi`

        Args:
            bc:
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES_OPTIONAL}
            out (ScalarField, optional):
                Optional vector field to which the result is written.
            label (str, optional):
                Name of the returned field
            central (bool):
                Determines whether a central difference approximation is used for the
                gradient operator or not. If not, the squared gradient is calculated as
                the mean of the squared values of the forward and backward derivatives,
                which thus includes the value at a support point in the result at the
                same point.

        Returns:
            :class:`~pde.fields.scalar.ScalarField`: the squared gradient of the field
        """
        return self._apply_operator("gradient_squared", bc=bc, out=out, **kwargs)  # type: ignore

    @fill_in_docstring
    def gradient(
        self,
        bc: Optional[BoundariesData],
        out: Optional["VectorField"] = None,
        **kwargs,
    ) -> "VectorField":
        """apply gradient operator and return result as a field

        Args:
            bc:
                The boundary conditions applied to the field.
                {ARG_BOUNDARIES_OPTIONAL}
            out (VectorField, optional):
                Optional vector field to which the result is written.
            label (str, optional):
                Name of the returned field

        Returns:
            :class:`~pde.fields.vectorial.VectorField`: result of applying the operator
        """
        return self._apply_operator("gradient", bc=bc, out=out, **kwargs)  # type: ignore

    @property
    def integral(self) -> Number:
        """Number: integral of the scalar field over space"""
        return self.grid.integrate(self.data)  # type: ignore

    def project(
        self,
        axes: Union[str, Sequence[str]],
        method: str = "integral",
        label: str = None,
    ) -> ScalarField:
        """project scalar field along given axes

        Args:
            axes (list of str):
                The names of the axes that are removed by the projection
                operation. The valid names for a given grid are the ones in
                the :attr:`GridBase.axes` attribute.
            method (str):
                The projection method. This can be either 'integral' to
                integrate over the removed axes or 'average' to perform an
                average instead.
            label (str, optional):
                The label of the returned field

        Returns:
            :class:`~pde.fields.scalar.ScalarField`: The projected data in a scalar
            field with a subgrid of the original grid.
        """
        if isinstance(axes, str):
            axes = [axes]
        if any(ax not in self.grid.axes for ax in axes):
            raise ValueError(
                f"The axes {axes} are not all contained in {self.grid} with axes "
                f"{self.grid.axes}"
            )

        # determine the axes after projection
        ax_all = range(self.grid.num_axes)
        ax_remove = tuple(self.grid.axes.index(ax) for ax in axes)
        ax_retain = tuple(sorted(set(ax_all) - set(ax_remove)))

        # determine the new grid
        sliced_grid = self.grid.slice(ax_retain)

        # calculate the new data
        if method == "integral":
            subdata = self.grid.integrate(self.data, axes=ax_remove)

        elif method == "average" or method == "mean":
            integrals = self.grid.integrate(self.data, axes=ax_remove)
            volumes = self.grid.integrate(1, axes=ax_remove)
            subdata = integrals / volumes

        else:
            raise ValueError(f"Unknown projection method `{method}`")

        # create the new field instance
        return self.__class__(grid=sliced_grid, data=subdata, label=label)

    def slice(
        self, position: Dict[str, float], *, method: str = "nearest", label: str = None
    ) -> ScalarField:
        """slice data at a given position

        Args:
            position (dict):
                Determines the location of the slice using a dictionary
                supplying coordinate values for a subset of axes. Axes not
                mentioned in the dictionary are retained and form the slice.
                For instance, in a 2d Cartesian grid, `position = {'x': 1}`
                slices along the y-direction at x=1. Additionally, the special
                positions 'low', 'mid', and 'high' are supported to reference
                relative positions along the axis.
            method (str):
                The method used for slicing. `nearest` takes data from cells
                defined on the grid.
            label (str, optional):
                The label of the returned field

        Returns:
            :class:`~pde.fields.scalar.ScalarField`: The sliced data in a scalar field
            with a subgrid of the original grid.
        """
        grid = self.grid

        # parse the positions and determine the axes to remove
        ax_remove, pos_values = [], np.zeros(grid.num_axes)
        for ax, pos in position.items():
            # check the axis
            try:
                i = grid.axes.index(ax)
            except ValueError:
                raise ValueError(
                    f"The axes {ax} is not contained in "
                    f"{self.grid} with axes {self.grid.axes}"
                )
            ax_remove.append(i)

            # check the position
            if isinstance(pos, str):
                if pos in {"min", "low", "lower"}:
                    pos_values[i] = grid.axes_coords[i][0]
                elif pos in {"max", "high", "upper"}:
                    pos_values[i] = grid.axes_coords[i][-1]
                elif pos in {"mid", "middle", "center"}:
                    pos_values[i] = np.mean(grid.axes_bounds[i])
                else:
                    raise ValueError(f"Unknown position `{pos}`")
            else:
                pos_values[i] = float(pos)

        # determine the axes left after slicing and the new grid
        ax_all = range(grid.num_axes)
        ax_retain = tuple(sorted(set(ax_all) - set(ax_remove)))
        sliced_grid = grid.slice(ax_retain)

        # obtain the sliced data
        if method == "nearest":
            idx: List[Union[int, slice]] = []
            for i in range(grid.num_axes):
                if i in ax_remove:
                    pos = pos_values[i]
                    axis_bounds = grid.axes_bounds[i]
                    if pos < axis_bounds[0] or pos > axis_bounds[1]:
                        raise DomainError(
                            f"Position {grid.axes[i]} = {pos} is outside the domain"
                        )
                    # add slice that is closest to pos
                    idx.append(int(np.argmin((grid.axes_coords[i] - pos) ** 2)))
                else:
                    idx.append(slice(None))
            subdata = self.data[tuple(idx)]

        else:
            raise ValueError(f"Unknown slicing method `{method}`")

        # create the new field instance
        return self.__class__(grid=sliced_grid, data=subdata, label=label)

    def to_scalar(
        self, scalar: Union[str, Callable] = "auto", *, label: Optional[str] = None
    ) -> ScalarField:
        """return a modified scalar field by applying method `scalar`

        Args:
            scalar (str or callable):
                Determines the method used for obtaining the scalar. If this is a
                callable, it is simply applied to self.data and a new scalar field with
                this data is returned. Alternatively, pre-defined methods can be
                selected using strings. Here, `abs` and `norm` denote the norm of each
                entry of the field, while `norm_squared` returns the squared norm. The
                default `auto` is to return a (unchanged) copy of a real field and the
                norm of a complex field.
            label (str, optional):
                Name of the returned field

        Returns:
            :class:`~pde.fields.scalar.ScalarField`: Scalar field after applying the
            operation
        """
        if callable(scalar):
            data = scalar(self.data)
        elif scalar == "auto":
            if np.iscomplexobj(self.data):
                data = np.abs(self.data)
            else:
                data = self.data
        elif scalar == "abs" or scalar == "norm":
            data = np.abs(self.data)
        elif scalar == "norm_squared":
            data = self.data * self.data.conjugate()
        else:
            raise ValueError(f"Unknown method `{scalar}` for `to_scalar`")

        return ScalarField(grid=self.grid, data=data, label=label)
