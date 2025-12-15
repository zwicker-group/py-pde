"""Compiled functions for dealing with grids.

.. autosummary::
   :nosignatures:

   get_grid_numba_type
   make_cell_volume_getter
   make_interpolation_axis_data
   make_single_interpolator

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba.extending import register_jitable

from ...grids import DomainError, GridBase
from .utils import jit

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...tools.typing import (
        CellVolume,
        FloatingArray,
        Number,
        NumberOrArray,
        NumericArray,
    )


def get_grid_numba_type(grid: GridBase, rank: int = 0) -> str:
    """Return numba type corresponding to a particular grid.

    Args:
        grid (GridBase):
            The grid for which we determine the type
        rank (int):
            The rank of the data stored in the grid

    Returns:
        _type_: _description_
    """
    dim = grid.num_axes + rank
    return "f8[" + ", ".join([":"] * dim) + "]"


def make_cell_volume_getter(grid: GridBase, *, flat_index: bool = False) -> CellVolume:
    """Return a compiled function returning the volume of a grid cell.

    Args:
        grid (:class:`~pde.grid.base.GridBase`):
            Grid for which the function is defined
        flat_index (bool):
            When True, cell_volumes are indexed by a single integer into the
            flattened array.

    Returns:
        function: returning the volume of the chosen cell
    """
    if grid.cell_volume_data is not None and all(
        np.isscalar(d) for d in grid.cell_volume_data
    ):
        # all cells have the same volume
        cell_volume = np.prod(grid.cell_volume_data)  # type: ignore

        @jit
        def get_cell_volume(*args) -> float:
            return cell_volume  # type: ignore

    else:
        # some cells have a different volume
        cell_volumes = grid.cell_volumes

        if flat_index:

            @jit
            def get_cell_volume(idx: int) -> float:
                return cell_volumes.flat[idx]  # type: ignore

        else:

            @jit
            def get_cell_volume(*args) -> float:
                return cell_volumes[args]  # type: ignore

    return get_cell_volume  # type: ignore


def make_interpolation_axis_data(
    grid: GridBase,
    axis: int,
    *,
    with_ghost_cells: bool = False,
    cell_coords: bool = False,
) -> Callable[[float], tuple[int, int, float, float]]:
    """Factory for obtaining interpolation information.

    Args:
        grid (:class:`~pde.grid.base.GridBase`):
            Grid for which the interpolator is defined
        axis (int):
            The axis along which interpolation is performed
        with_ghost_cells (bool):
            Flag indicating that the interpolator should work on the full data array
            that includes values for the ghost points. If this is the case, the
            boundaries are not checked and the coordinates are used as is.
        cell_coords (bool):
            Flag indicating whether points are given in cell coordinates or actual
            point coordinates.

    Returns:
        callable: A function that is called with a coordinate value for the axis.
        The function returns the indices of the neighboring support points as well
        as the associated weights.
    """
    # obtain information on how this axis is discretized
    size = grid.shape[axis]
    periodic = grid.periodic[axis]
    lo = grid.axes_bounds[axis][0]
    dx = grid.discretization[axis]

    @register_jitable
    def get_axis_data(coord: float) -> tuple[int, int, float, float]:
        """Determines data for interpolating along one axis."""
        # determine the index of the left cell and the fraction toward the right
        if cell_coords:
            c_l, d_l = divmod(coord, 1.0)
        else:
            c_l, d_l = divmod((coord - lo) / dx - 0.5, 1.0)

        # determine the indices of the two cells whose value affect interpolation
        if periodic:
            # deal with periodic domains, which is easy
            c_li = int(c_l) % size  # left support point
            c_hi = (c_li + 1) % size  # right support point

        elif with_ghost_cells:
            # deal with edge cases using the values of ghost cells
            if -0.5 <= c_l + d_l <= size - 0.5:  # in bulk part of domain
                c_li = int(c_l)  # left support point
                c_hi = c_li + 1  # right support point
            else:
                return -42, -42, 0.0, 0.0  # indicates out of bounds

        else:
            # deal with edge cases using nearest-neighbor interpolation at boundary
            if 0 <= c_l + d_l < size - 1:  # in bulk part of domain
                c_li = int(c_l)  # left support point
                c_hi = c_li + 1  # right support point
            elif size - 1 <= c_l + d_l <= size - 0.5:  # close to upper boundary
                c_li = c_hi = int(c_l)  # both support points close to boundary
                # This branch also covers the special case, where size == 1 and data
                # is evaluated at the only support point (c_l == d_l == 0.)
            elif -0.5 <= c_l + d_l <= 0:  # close to lower boundary
                c_li = c_hi = int(c_l) + 1  # both support points close to boundary
            else:
                return -42, -42, 0.0, 0.0  # indicates out of bounds

        # determine the weights of the two cells
        w_l, w_h = 1 - d_l, d_l
        # set small weights to zero. If this is not done, invalid data at the corner
        # of the grid (where two rows of ghost cells intersect) could be accessed.
        # If this random data is very large, e.g., 1e100, it contributes
        # significantly, even if the weight is low, e.g., 1e-16.
        if w_l < 1e-15:
            w_l = 0
        if w_h < 1e-15:
            w_h = 0

        # shift points to allow accessing data with ghost points
        if with_ghost_cells:
            c_li += 1
            c_hi += 1

        return c_li, c_hi, w_l, w_h

    return get_axis_data  # type: ignore


def make_single_interpolator(
    grid: GridBase,
    *,
    fill: Number | None = None,
    with_ghost_cells: bool = False,
    cell_coords: bool = False,
) -> Callable[[NumericArray, FloatingArray], NumericArray]:
    """Return a compiled function for linear interpolation on the grid.

    Args:
        grid (:class:`~pde.grid.base.GridBase`):
            Grid for which the interpolator is defined
        fill (Number, optional):
            Determines how values out of bounds are handled. If `None`, `ValueError`
            is raised when out-of-bounds points are requested. Otherwise, the given
            value is returned.
        with_ghost_cells (bool):
            Flag indicating that the interpolator should work on the full data array
            that includes values for the ghost points. If this is the case, the
            boundaries are not checked and the coordinates are used as is.
        cell_coords (bool):
            Flag indicating whether points are given in cell coordinates or actual
            point coordinates.

    Returns:
        callable: A function which returns interpolated values when called with
        arbitrary positions within the space of the grid. The signature of this
        function is (data, point), where `data` is the numpy array containing the
        field data and position denotes the position in grid coordinates.
    """
    args = {"with_ghost_cells": with_ghost_cells, "cell_coords": cell_coords}

    if grid.num_axes == 1:
        # specialize for 1-dimensional interpolation
        data_x = make_interpolation_axis_data(grid=grid, axis=0, **args)

        @jit
        def interpolate_single(
            data: NumericArray, point: FloatingArray
        ) -> NumberOrArray:
            """Obtain interpolated value of data at a point.

            Args:
                data (:class:`~numpy.ndarray`):
                    A 1d array of valid values at the grid points
                point (:class:`~numpy.ndarray`):
                    Coordinates of a single point in the grid coordinate system

            Returns:
                :class:`~numpy.ndarray`: The interpolated value at the point
            """
            c_li, c_hi, w_l, w_h = data_x(float(point[0]))

            if c_li == -42:  # out of bounds
                if fill is None:  # outside the domain
                    print("POINT", point)
                    msg = "Point lies outside the grid domain"
                    raise DomainError(msg)
                return fill

            # do the linear interpolation
            return w_l * data[..., c_li] + w_h * data[..., c_hi]

    elif grid.num_axes == 2:
        # specialize for 2-dimensional interpolation
        data_x = make_interpolation_axis_data(grid=grid, axis=0, **args)
        data_y = make_interpolation_axis_data(grid=grid, axis=1, **args)

        @jit
        def interpolate_single(
            data: NumericArray, point: FloatingArray
        ) -> NumberOrArray:
            """Obtain interpolated value of data at a point.

            Args:
                data (:class:`~numpy.ndarray`):
                    A 2d array of valid values at the grid points
                point (:class:`~numpy.ndarray`):
                    Coordinates of a single point in the grid coordinate system

            Returns:
                :class:`~numpy.ndarray`: The interpolated value at the point
            """
            # determine surrounding points and their weights
            c_xli, c_xhi, w_xl, w_xh = data_x(float(point[0]))
            c_yli, c_yhi, w_yl, w_yh = data_y(float(point[1]))

            if c_xli == -42 or c_yli == -42:  # out of bounds
                if fill is None:  # outside the domain
                    print("POINT", point)
                    msg = "Point lies outside the grid domain"
                    raise DomainError(msg)
                return fill

            # do the linear interpolation
            return (
                w_xl * w_yl * data[..., c_xli, c_yli]
                + w_xl * w_yh * data[..., c_xli, c_yhi]
                + w_xh * w_yl * data[..., c_xhi, c_yli]
                + w_xh * w_yh * data[..., c_xhi, c_yhi]
            )

    elif grid.num_axes == 3:
        # specialize for 3-dimensional interpolation
        data_x = make_interpolation_axis_data(grid=grid, axis=0, **args)
        data_y = make_interpolation_axis_data(grid=grid, axis=1, **args)
        data_z = make_interpolation_axis_data(grid=grid, axis=2, **args)

        @jit
        def interpolate_single(
            data: NumericArray, point: FloatingArray
        ) -> NumberOrArray:
            """Obtain interpolated value of data at a point.

            Args:
                data (:class:`~numpy.ndarray`):
                    A 2d array of valid values at the grid points
                point (:class:`~numpy.ndarray`):
                    Coordinates of a single point in the grid coordinate system

            Returns:
                :class:`~numpy.ndarray`: The interpolated value at the point
            """
            # determine surrounding points and their weights
            c_xli, c_xhi, w_xl, w_xh = data_x(float(point[0]))
            c_yli, c_yhi, w_yl, w_yh = data_y(float(point[1]))
            c_zli, c_zhi, w_zl, w_zh = data_z(float(point[2]))

            if c_xli == -42 or c_yli == -42 or c_zli == -42:  # out of bounds
                if fill is None:  # outside the domain
                    print("POINT", point)
                    msg = "Point lies outside the grid domain"
                    raise DomainError(msg)
                return fill

            # do the linear interpolation
            return (
                w_xl * w_yl * w_zl * data[..., c_xli, c_yli, c_zli]
                + w_xl * w_yl * w_zh * data[..., c_xli, c_yli, c_zhi]
                + w_xl * w_yh * w_zl * data[..., c_xli, c_yhi, c_zli]
                + w_xl * w_yh * w_zh * data[..., c_xli, c_yhi, c_zhi]
                + w_xh * w_yl * w_zl * data[..., c_xhi, c_yli, c_zli]
                + w_xh * w_yl * w_zh * data[..., c_xhi, c_yli, c_zhi]
                + w_xh * w_yh * w_zl * data[..., c_xhi, c_yhi, c_zli]
                + w_xh * w_yh * w_zh * data[..., c_xhi, c_yhi, c_zhi]
            )

    else:
        msg = f"Compiled interpolation not implemented for dimension {grid.num_axes}"
        raise NotImplementedError(msg)

    return interpolate_single  # type: ignore
