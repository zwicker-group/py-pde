r"""This module contains classes for handling a single boundary of a non-periodic axis.
Since an axis has two boundary, we simply distinguish them by a boolean flag `upper`,
which is `True` for the side of the axis with the larger coordinate.

The module currently supports the following standard boundary conditions:

* :class:`~pde.grids.boundaries.local.DirichletBC`:
  Imposing the value of a field at the boundary
* :class:`~pde.grids.boundaries.local.NeumannBC`:
  Imposing the derivative of a field in the outward normal direction at the boundary
* :class:`~pde.grids.boundaries.local.MixedBC`:
  Imposing the derivative of a field in the outward normal direction proportional to its
  value at the boundary
* :class:`~pde.grids.boundaries.local.CurvatureBC`:
  Imposing the second derivative (curvature) of a field at the boundary

There are also variants of these boundary conditions that only affect the normal
components of a vector or tensor field:
:class:`~pde.grids.boundaries.local.NormalDirichletBC`,
:class:`~pde.grids.boundaries.local.NormalNeumannBC`,
:class:`~pde.grids.boundaries.local.NormalMixedBC`, and
:class:`~pde.grids.boundaries.local.NormalCurvatureBC`.

Finally, there are more specialized classes, which offer greater flexibility, but might
also require a slightly deeper understanding for proper use:

* :class:`~pde.grids.boundaries.local.ExpressionValueBC`:
  Imposing the value of a field at the boundary based on a mathematical expression or a
  python function.
* :class:`~pde.grids.boundaries.local.ExpressionDerivativeBC`:
  Imposing the derivative of a field in the outward normal direction at the boundary
  based on a mathematical expression or a python function.
* :class:`~pde.grids.boundaries.local.ExpressionMixedBC`:
  Imposing a mixed (Robin) boundary condition using mathematical expressions or python
  functions.
* :class:`~pde.grids.boundaries.local.UserBC`:
  Allows full control for setting virtual points, values, or derivatives. The boundary
  conditions are never enforced automatically. It is thus the user's responsibility to
  ensure virtual points are set correctly before operators are applied. To set boundary
  conditions a dictionary :code:`{TARGET: value}` must be supplied as argument `args` to
  :meth:`set_ghost_cells` or the numba equivalent. Here, `TARGET` determines how the
  `value` is interpreted and what boundary condition is actually enforced: the value of
  the virtual points directly (`virtual_point`), the value of the field at the boundary
  (`value`) or the outward derivative of the field at the boundary (`derivative`).

Note that derivatives are generally given in the direction of the outward normal vector,
such that positive derivatives correspond to a function that increases across the
boundary.


**Inheritance structure of the classes:**

.. inheritance-diagram:: pde.grids.boundaries.local
   :parts: 1

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging
import warnings
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, Literal, TypeVar, Union

import numpy as np
from typing_extensions import Self

from ...tools.cache import cached_method
from ...tools.docstrings import fill_in_docstring
from ...tools.misc import number
from ..base import GridBase, PeriodicityError

if TYPE_CHECKING:
    from collections.abc import Callable

    from ...tools.typing import (
        NumberOrArray,
        NumericArray,
    )
    from .._mesh import GridMesh

_logger = logging.getLogger(__name__)
""":class:`logging.Logger`: Logger instance."""

BoundaryData = Union[dict, str, "BCBase"]


class BCDataError(ValueError):
    """Exception that signals that incompatible data was supplied for the BC."""


def _get_arr_1d(
    arr, idx: tuple[int, ...], axis: int
) -> tuple[NumericArray, int, tuple]:
    """Extract the 1d array along axis at point idx.

    Args:
        arr (:class:`~numpy.ndarray`): The full data array
        idx (tuple): An index into the data array
        axis (int): The axis along which the 1d array will be extracted

    Returns:
        tuple: a tuple (arr_1d, i, bc_i), where `arr_1d` is the 1d array, `i` is
        the index `i` into this array marking the current point and `bc_i` are
        the remaining components of `idx`, which locate the point in the
        orthogonal directions. Consequently, `i = idx[axis]` and
        `arr[..., idx] == arr_1d[..., i]`.
    """
    dim = len(idx)
    # extract the correct indices
    if dim == 1:
        i = idx[0]
        bc_idx: tuple = (...,)
        arr_1d = arr

    elif dim == 2:
        if axis == 0:
            i, y = idx
            bc_idx = (..., y)
            arr_1d = arr[..., :, y]
        elif axis == 1:
            x, i = idx
            bc_idx = (..., x)
            arr_1d = arr[..., x, :]

    elif dim == 3:
        if axis == 0:
            i, y, z = idx
            bc_idx = (..., y, z)
            arr_1d = arr[..., :, y, z]
        elif axis == 1:
            x, i, z = idx
            bc_idx = (..., x, z)
            arr_1d = arr[..., x, :, z]
        elif axis == 2:
            x, y, i = idx
            bc_idx = (..., x, y)
            arr_1d = arr[..., x, y, :]

    else:
        raise NotImplementedError

    return arr_1d, i, bc_idx


# define generic type variable of type BCBase
TBC = TypeVar("TBC", bound="BCBase")


class BCBase(metaclass=ABCMeta):
    """Represents a single boundary in an BoundaryPair instance."""

    names: list[str]
    """list: identifiers used to specify the given boundary class"""
    homogeneous: bool
    """bool: determines whether the boundary condition depends on space"""
    normal: bool = False
    """bool: determines whether the boundary condition only affects normal components.

    If this flag is `False`, boundary conditions must specify values for all components
    of the field. If `True`, only the normal components at the boundary are specified.
    """
    _axes_indices: str = "αβγδ"
    """ str: indices used to indicate arbitrary axes in boundary conditions"""

    _subclasses: dict[str, type[BCBase]] = {}  # all classes inheriting from this
    _conditions: dict[str, type[BCBase]] = {}  # mapping from all names to classes

    def __init__(self, grid: GridBase, axis: int, upper: bool, *, rank: int = 0):
        """
        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Flag indicating whether this boundary condition is associated with the
                upper side of an axis or not. In essence, this determines the direction
                of the local normal vector of the boundary.
            rank (int):
                The tensorial rank of the field for this boundary condition
        """
        self.grid = grid
        self.axis = axis
        self.upper = upper
        self.rank = rank

        # Get the shape of the values imposed at the boundary. These are the shape of
        # the tensor field unless only the normal component is specified
        if self.rank == 0:
            self.normal = False

        if self.normal:
            self._shape_tensor = (self.grid.dim,) * (self.rank - 1)
        else:
            self._shape_tensor = (self.grid.dim,) * self.rank

        # get the shape of the data at the boundary
        self._shape_boundary = (
            self.grid.shape[: self.axis] + self.grid.shape[self.axis + 1 :]
        )

    def __init_subclass__(cls, **kwargs):
        """Register all subclasses to reconstruct them later."""
        super().__init_subclass__(**kwargs)

        if cls is not BCBase:
            if cls.__name__ in cls._subclasses:
                warnings.warn(f"Redefining class {cls.__name__}", stacklevel=2)
            cls._subclasses[cls.__name__] = cls

            if hasattr(cls, "names"):
                for name in cls.names:
                    cls._conditions[name] = cls

    @property
    def periodic(self) -> bool:
        """bool: whether the boundary condition is periodic"""
        # we determine the periodicity of the boundary condition from the condition
        # itself so we can check for consistency against the grid periodicity
        return isinstance(self, _PeriodicBC)

    @property
    def axis_coord(self) -> float:
        """float: value of the coordinate that defines this boundary condition"""
        if self.upper:
            return self.grid.axes_bounds[self.axis][1]
        return self.grid.axes_bounds[self.axis][0]

    def _field_repr(self, field_name: str) -> str:
        """Return representation of the field to which the condition is applied.

        Args:
            field_name (str): Symbol of the field variable

        Returns:
            str: A field with indices denoting which components will be modified
        """
        axis_name = self.grid.axes[self.axis]
        if self.normal:
            assert self.rank > 0
            field_indices = self._axes_indices[: self.rank - 1] + axis_name
        else:
            field_indices = self._axes_indices[: self.rank]
        if field_indices:
            return f"{field_name}_{field_indices}"
        return f"{field_name}"

    def get_mathematical_representation(self, field_name: str = "C") -> str:
        """Return mathematical representation of the boundary condition."""
        raise NotImplementedError

    @classmethod
    def get_help(cls) -> str:
        """Return information on how boundary conditions can be set."""
        types = ", ".join(
            f"'{subclass.names[0]}'"
            for subclass in cls._subclasses.values()
            if hasattr(subclass, "names")
        )
        return (
            f"Possible types of boundary conditions are {types}. "
            "Values can be set using {'type': TYPE, 'value': VALUE}. "
            "Here, VALUE can be a scalar number, a vector for tensorial boundary "
            "conditions, or a string, which can be interpreted as a sympy expression. "
            "In the latter case, the names of the axes not associated with this "
            "boundary can be used as variables to describe inhomogeneous boundary "
            "conditions."
        )

    def _repr_value(self) -> list[str]:
        return []

    def __repr__(self):
        args = [f"grid={self.grid}", f"axis={self.axis}", f"upper={self.upper}"]
        if self.rank != 0:
            args.append(f"rank={self.rank}")
        args += self._repr_value()
        return f"{self.__class__.__name__}({', '.join(args)})"

    def __str__(self):
        args = [f"axis={self.axis}", "upper" if self.upper else "lower"]
        if self.rank != 0:
            args.append(f"rank={self.rank}")
        args += self._repr_value()
        return f"{self.__class__.__name__}({', '.join(args)})"

    def __eq__(self, other):
        """Checks for equality neglecting the `upper` property."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.__class__ == other.__class__
            and self.grid == other.grid
            and self.axis == other.axis
            and self.homogeneous == other.homogeneous
            and self.rank == other.rank
            and self.normal == other.normal
        )

    def __ne__(self, other):
        return not self == other

    @classmethod
    def from_str(
        cls,
        grid: GridBase,
        axis: int,
        upper: bool,
        condition: str,
        *,
        rank: int = 0,
        **kwargs,
    ) -> BCBase:
        r"""Creates boundary from a given string identifier.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Indicates whether this boundary condition is associated with the
                upper or lower side of the axis.
            condition (str):
                Identifies the boundary condition
            rank (int):
                The tensorial rank of the field for this boundary condition
            \**kwargs:
                Additional arguments passed to the constructor
        """
        # extract the class
        try:
            boundary_class = cls._conditions[condition]
        except KeyError:
            raise BCDataError(
                f"Boundary condition `{condition}` not defined. " + cls.get_help()
            ) from None

        # create the actual class
        return boundary_class(grid=grid, axis=axis, upper=upper, rank=rank, **kwargs)

    @classmethod
    def from_dict(
        cls,
        grid: GridBase,
        axis: int,
        upper: bool,
        data: dict[str, Any],
        *,
        rank: int = 0,
    ) -> BCBase:
        """Create boundary from data given in dictionary.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Indicates whether this boundary condition is associated with the
                upper or lower side of the axis.
            data (dict):
                The dictionary defining the boundary condition
            rank (int):
                The tensorial rank of the field for this boundary condition
        """
        data = data.copy()  # need to make a copy since we modify it below

        # parse all possible variants that could be given
        if "type" in data:
            # type is given (optionally with a value)
            b_type = data.pop("type")
            return cls.from_str(grid, axis, upper, condition=b_type, rank=rank, **data)

        if len(data) == 1:
            # only a single items is given
            b_type, b_value = data.popitem()
            return cls.from_str(
                grid, axis, upper, condition=b_type, rank=rank, value=b_value, **data
            )

        msg = f"Boundary conditions `{list(data)}` are not supported."
        raise BCDataError(msg)

    @classmethod
    def from_data(
        cls,
        grid: GridBase,
        axis: int,
        upper: bool,
        data: BoundaryData,
        *,
        rank: int = 0,
    ) -> BCBase:
        """Create boundary from some data.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Indicates whether this boundary condition is associated with the
                upper or lower side of the axis.
            data (str or dict):
                Data that describes the boundary
            rank (int):
                The tensorial rank of the field for this boundary condition

        Returns:
            :class:`~pde.grids.boundaries.local.BCBase`: the instance created
            from the data

        Throws:
            ValueError if `data` cannot be interpreted as a boundary condition
        """
        # check all different data formats
        if isinstance(data, BCBase):
            # already in the correct format
            if data.grid._mesh is not None:
                # we need to exclude this case since otherwise we get into a rabbit hole
                # where it is not clear what grid boundary conditions belong to. The
                # idea is that users only create boundary conditions for the full grid
                # and that the splitting onto subgrids is only done once, automatically,
                # and without involving calls to `from_data`
                msg = "Cannot create MPI subgrid BC from data"
                raise ValueError(msg)

            if data.grid != grid or data.axis != axis or data.rank != rank:
                msg = f"Incompatible: {data!r} & {grid=}, {axis=}, {rank=})"
                raise ValueError(msg)
            bc = data.copy(upper=upper)

        elif isinstance(data, dict):
            # create from dictionary
            bc = cls.from_dict(grid, axis, upper=upper, data=data, rank=rank)

        elif isinstance(data, str):
            # create a specific condition given by a string
            bc = cls.from_str(grid, axis, upper=upper, condition=data, rank=rank)

        elif data is None:
            msg = (
                "Unspecified condition for boundary "
                f"{grid.axes[axis]}{'-+'[int(upper)]}"
            )
            raise BCDataError(msg)

        else:
            raise BCDataError(f"Unsupported BC format: `{data}`. " + cls.get_help())

        # check consistency
        if bc.periodic != grid.periodic[axis]:
            msg = "Periodicity of conditions must match grid"
            raise PeriodicityError(msg)
        return bc

    def to_subgrid(self, subgrid: GridBase) -> Self:
        """Converts this boundary condition to one valid for a given subgrid.

        Args:
            subgrid (:class:`GridBase`):
                Grid of the new boundary conditions

        Returns:
            :class:`BCBase`: Boundary conditions valid on the subgrid
        """
        msg = "Boundary condition cannot be transferred to subgrid"
        raise NotImplementedError(msg)

    def check_value_rank(self, rank: int) -> None:
        """Check whether the values at the boundaries have the correct rank.

        Args:
            rank (int):
                The tensorial rank of the field for this boundary condition

        Throws:
            RuntimeError: if the value does not have rank `rank`
        """
        if self.rank != rank:
            msg = f"Expected rank {rank}, but boundary condition had rank {self.rank}."
            raise RuntimeError(msg)

    def copy(self, upper: bool | None = None, rank: int | None = None) -> Self:
        raise NotImplementedError

    def get_sparse_matrix_data(
        self, idx: tuple[int, ...]
    ) -> tuple[float, dict[int, float]]:
        raise NotImplementedError

    def get_virtual_point(self, arr, idx: tuple[int, ...] | None = None) -> float:
        raise NotImplementedError

    @abstractmethod
    def set_ghost_cells(self, data_full: NumericArray, *, args=None) -> None:
        """Set the ghost cell values for this boundary.

        Args:
            data_full (:class:`~numpy.ndarray`):
                The full field data including ghost points
            args (:class:`~numpy.ndarray`):
                Determines what boundary conditions are set. `args` should be set to
                :code:`{TARGET: value}`. Here, `TARGET` determines how the `value` is
                interpreted and what boundary condition is actually enforced: the value
                of the virtual points directly (`virtual_point`), the value of the field
                at the boundary (`value`) or the outward derivative of the field at the
                boundary (`derivative`).
        """

    def _get_value_cell_index(self, with_ghost_cells: bool) -> int:
        """Determine index of the cell from which field value is read.

        Args:
            with_ghost_cells (bool):
                Determines whether the index is supposed to be into an array with ghost
                cells or not
        """
        if self.upper:
            if with_ghost_cells:
                return self.grid.shape[self.axis]
            return self.grid.shape[self.axis] - 1
        if with_ghost_cells:
            return 1
        return 0


class _MPIBC(BCBase):
    """Represents a boundary that is exchanged with another MPI process."""

    homogeneous = False

    def __init__(
        self,
        mesh: GridMesh,
        axis: int,
        upper: bool,
        *,
        rank: int = 0,
        node_id: int | None = None,
    ):
        """
        Args:
            mesh (:class:`~pde.grids._mesh.GridMesh`):
                Grid mesh describing the distributed MPI nodes
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Flag indicating whether this boundary condition is associated with the
                upper side of an axis or not. In essence, this determines the direction
                of the local normal vector of the boundary.
            rank (int):
                The tensorial rank of the field for this boundary condition
            node_id (int):
                The MPI node (the subgrid ID) this BC is associated with
        """
        super().__init__(mesh[node_id], axis, upper, rank=rank)
        neighbor_id = mesh.get_neighbor(axis, upper, node_id=node_id)
        if neighbor_id is None:
            msg = "No neighboring cell for this boundary"
            raise RuntimeError(msg)
        self._neighbor_id = neighbor_id
        self._mpi_flag = mesh.get_boundary_flag(self._neighbor_id, upper)

        # determine indices for reading and writing data
        idx: list[Any] = [slice(1, -1)] * self.grid.num_axes
        idx[self.axis] = -2 if self.upper else 1  # read valid data
        self._idx_read = (Ellipsis, *idx)
        idx[self.axis] = -1 if self.upper else 0  # write ghost cells
        self._idx_write = (Ellipsis, *idx)

    def _repr_value(self):
        return [f"neighbor={self._neighbor_id}"]

    def __eq__(self, other):
        """Checks for equality neglecting the `upper` property."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.__class__ == other.__class__
            and self.grid == other.grid
            and self.axis == other.axis
            and self.rank == other.rank
            and self._neighbor_id == other._neighbor_id
        )

    def get_mathematical_representation(self, field_name: str = "C") -> str:
        """Return mathematical representation of the boundary condition."""
        axis_name = self.grid.axes[self.axis]
        return f"MPI @ {axis_name}={self.axis_coord}"

    def send_ghost_cells(self, data_full: NumericArray, *, args=None) -> None:
        """mpi_send the ghost cell values for this boundary.

        Args:
            data_full (:class:`~numpy.ndarray`):
                The full field data including ghost points
        """
        from ...tools.mpi import mpi_send

        mpi_send(data_full[self._idx_read], self._neighbor_id, self._mpi_flag)

    def set_ghost_cells(self, data_full: NumericArray, *, args=None) -> None:
        from ...tools.mpi import mpi_recv

        mpi_recv(data_full[self._idx_write], self._neighbor_id, self._mpi_flag)


class UserBC(BCBase):
    """Represents a boundary whose virtual point are set by the user.

    Boundary conditions will only be set when a dictionary :code:`{TARGET: value}` is
    supplied as argument `args` to :meth:`set_ghost_cells` or the numba equivalent.
    Here, `TARGET` determines how the `value` is interpreted and what boundary condition
    is actually enforced: the value of the virtual points directly (`virtual_point`),
    the value of the field at the boundary (`value`) or the outward derivative of the
    field at the boundary (`derivative`).

    Warning:
        This implies that the boundary conditions are never enforced automatically,
        e.g., when evaluating an operator. It is thus the user's responsibility to
        ensure virtual points are set correctly before operators are applied.
    """

    names = ["user"]

    def get_mathematical_representation(self, field_name: str = "C") -> str:
        """Return mathematical representation of the boundary condition."""
        axis_name = self.grid.axes[self.axis]
        return f"user-controlled  @ {axis_name}={self.axis_coord}"

    def copy(self, upper: bool | None = None, rank: int | None = None) -> Self:
        """Return a copy of itself, but with a reference to the same grid."""
        return self.__class__(
            grid=self.grid,
            axis=self.axis,
            upper=self.upper if upper is None else upper,
            rank=self.rank if rank is None else rank,
        )

    def to_subgrid(self, subgrid: GridBase) -> Self:
        """Converts this boundary condition to one valid for a given subgrid.

        Args:
            subgrid (:class:`GridBase`):
                Grid of the new boundary conditions

        Returns:
            :class:`BCBase`: Boundary conditions valid on the subgrid
        """
        # use `issubclass`, so that `self.grid` could be `UnitGrid`, while `subgrid` is
        # `CartesianGrid`
        assert issubclass(self.grid.__class__, subgrid.__class__)
        return self.__class__(
            grid=subgrid, axis=self.axis, upper=self.upper, rank=self.rank
        )

    def set_ghost_cells(self, data_full: NumericArray, *, args=None) -> None:
        if args is None:
            # usual case where set_ghost_cells is called automatically. In our case,
            # won't do anything since we expect the user to call the function manually
            # with the user data provided as the argument.
            return

        if any(t in args for t in ["virtual_point", "value", "derivative"]):
            # ghost cells will only be set if any of the above keys were supplied

            # prepare the array of slices to index bcs
            offset = data_full.ndim - self.grid.num_axes  # additional data axes
            idx_offset = [slice(None)] * offset
            idx_valid = [slice(1, -1)] * self.grid.num_axes
            idx_write: list[slice | int] = idx_offset + idx_valid  # type: ignore
            idx_write[offset + self.axis] = -1 if self.upper else 0
            idx_read = idx_write[:]
            idx_read[offset + self.axis] = -2 if self.upper else 1

            if self.normal:
                assert offset > 0
                idx_write[offset - 1] = self.axis
                idx_read[offset - 1] = self.axis

            # get values right next to the boundary
            bndry_values = data_full[tuple(idx_read)]

            # calculate the virtual points
            if "virtual_point" in args:
                data_full[tuple(idx_write)] = args["virtual_point"]
            elif "value" in args:
                data_full[tuple(idx_write)] = 2 * args["value"] - bndry_values
            elif "derivative" in args:
                dx = self.grid.discretization[self.axis]
                data_full[tuple(idx_write)] = dx * args["derivative"] + bndry_values
            else:
                raise RuntimeError
        # else: no-op for the default case where BCs are not set by user


ExpressionBCTargetType = Literal["value", "derivative", "mixed", "virtual_point"]


class ExpressionBC(BCBase):
    """Represents a boundary whose virtual point is calculated from an expression.

    The expression is given as a string that can be parsed by :mod:`sympy` or as a
    function. The expression can contain typical mathematical operators and may depend
    on the value at the last support point next to the boundary (`value`), spatial
    coordinates defined by the grid marking the boundary point (e.g., `x` or `r`), and
    time `t`.
    """

    names = ["virtual_point"]

    @fill_in_docstring
    def __init__(
        self,
        grid: GridBase,
        axis: int,
        upper: bool,
        *,
        rank: int = 0,
        value: float | str | Callable = 0,
        const: float | str | Callable = 0,
        target: ExpressionBCTargetType = "virtual_point",
        user_funcs: dict[str, Callable] | None = None,
        value_cell: int | None = None,
    ):
        r"""
        Warning:
            {WARNING_EXEC}

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Flag indicating whether this boundary condition is associated with the
                upper side of an axis or not. In essence, this determines the direction
                of the local normal vector of the boundary.
            rank (int):
                The tensorial rank of the field for this boundary condition
            value (float or str or callable):
                An expression that determines the value of the boundary condition.
                Alternatively, this can be a function with signature `(value, dx,
                *coords, t)` that determines the value of `target` from the field value
                `value` (the value of the adjacent cell unless `value_cell` is
                specified), the spatial discretization `dx` in the direction
                perpendicular to the wall, the spatial coordinates of the wall point,
                and time `t`. Ideally, this function should be numba-compilable since
                simulations might otherwise be very slow.
            const (float or str or callable):
                An expression similar to `value`, which is only used for mixed (Robin)
                boundary conditions. Note that the implementation currently does not
                support that one argument is given as a callable function while the
                other is defined via an expression, so both need to have the same type.
            target (str):
                Selects which value is actually set. Possible choices include `value`,
                `derivative`, `mixed`, and `virtual_point`.
            user_funcs (dict, optional):
                A dictionary with user defined functions that can be used in the
                expression
            value_cell (int):
                Determines which cells is read to determine the field value that is used
                as `value` in the expression or the function call. The default (`None`)
                specifies the adjacent cell.
        """
        super().__init__(grid, axis, upper, rank=rank)
        self.value_cell = value_cell

        if self.rank != 0:
            msg = "Expression boundary conditions only work for scalar conditions"
            raise NotImplementedError(msg)

        # store data for later use
        self._input: dict[str, Any] = {
            "value_expr": value,
            "const_expr": const,
            "target": target,
            "user_funcs": user_funcs,
        }
        signature = ["value", "dx", *grid.axes, "t"]

        if callable(value) or callable(const):
            # the coefficients are given as functions
            self._is_func = True
        else:
            # the coefficients are expressions or constant values
            self._is_func = False
            if target == "virtual_point":
                expression = value
            elif target == "value":
                # Dirichlet boundary condition
                expression = f"2 * ({value}) - value"
            elif target == "derivative":
                # Neumann boundary condition
                expression = f"dx * ({value}) + value"
            elif target == "mixed":
                # special case of a Robin boundary condition, which also uses `const`
                enumerator = f"2 * dx * ({const}) + (2 - ({value}) * dx) * value"
                expression = f"({enumerator}) / (({value}) * dx + 2)"
            else:
                msg = f"Unknown target `{target}` for expression"
                raise ValueError(msg)

            # parse this expression
            from pde.tools.expressions import ScalarExpression

            self._func_expression = ScalarExpression(
                expression,
                signature=signature,
                user_funcs=user_funcs,
                repl=grid.c._axes_alt_repl,
            )

        # quickly check whether the expression was parsed correctly
        try:
            self._make_function()(*self._test_values)
        except Exception as err:
            if self._is_func:
                msg = f"Could not evaluate BC function. Expected signature {signature}."
                raise BCDataError(msg) from err
            msg = (
                f"Could not evaluate BC expression `{expression}` with signature "
                f"{signature}."
            )
            raise BCDataError(msg) from err

    @property
    def _test_values(self) -> tuple[float, ...]:
        """tuple: suitable values with which the user expression can be tested"""
        test_values = [
            np.zeros((self.grid.dim,) * self.rank),
            self.grid.discretization[self.axis],
        ]
        bc_coords = self.grid._boundary_coordinates(self.axis, self.upper)
        test_values.extend(np.moveaxis(bc_coords, -1, 0))
        test_values.append(0)
        return tuple(test_values)

    def _prepare_function(self, func: Callable | str | complex) -> Callable:
        """Helper function that compiles a single function given as a parameter."""
        if not callable(func):
            # the function is just a number, which we also support
            func_value = number(func)

            def value_func(*args):
                return func_value

            return value_func

        return func

    @cached_method()
    def _make_function(self) -> Callable:
        """Returns function that evaluates the value of the virtual point."""
        if not self._is_func:
            return self._func_expression
        # `value` is a callable function
        target = self._input["target"]
        value_func = self._prepare_function(self._input["value_expr"])

        if target == "virtual_point":
            return value_func

        if target == "value":
            # Dirichlet boundary condition

            def virtual_from_value(adjacent_value, *args):
                return 2 * value_func(adjacent_value, *args) - adjacent_value

            return virtual_from_value

        if target == "derivative":
            # Neumann boundary condition

            def virtual_from_derivative(adjacent_value, dx, *args):
                return dx * value_func(adjacent_value, dx, *args) + adjacent_value

            return virtual_from_derivative

        if target == "mixed":
            # special case of a Robin boundary condition, which also uses `const`
            const_func = self._prepare_function(self._input["const_expr"])

            def virtual_from_mixed(adjacent_value, dx, *args):
                value_dx = dx * value_func(adjacent_value, dx, *args)
                const_value = const_func(adjacent_value, dx, *args)
                expr_A = 2 * dx / (value_dx + 2) * const_value
                expr_B = (value_dx - 2) / (value_dx + 2)
                return expr_A - expr_B * adjacent_value

            return virtual_from_mixed

        msg = f"Unknown target `{target}` for expression"
        raise ValueError(msg)

    def _repr_value(self):
        if self._input["target"] == "mixed":
            # treat the mixed case separately
            res = [
                f'target="{self._input["target"]}", '
                f'value="{self._input["value_expr"]}", '
                f'const="{self._input["const_expr"]}"'
            ]
        elif self._is_func:
            res = [f"{self._input['target']}=<function>"]
        else:
            res = [f'{self._input["target"]}="{self._input["value_expr"]}"']
        if self.value_cell is not None:
            res.append(f", value_cell={self.value_cell}")
        return res

    def get_mathematical_representation(self, field_name: str = "C") -> str:
        """Return mathematical representation of the boundary condition."""
        axis_name = self.grid.axes[self.axis]
        target = self._input["target"]

        if self._is_func:
            value_expr = "<function>"
        else:
            value_expr = self._input["value_expr"]
        const_expr = self._input["const_expr"]

        field = self._field_repr(field_name)
        if target == "virtual_point":
            return f"{field} = {value_expr}   @ virtual point"
        if target == "value":
            return f"{field} = {value_expr}   @ {axis_name}={self.axis_coord}"
        if target == "derivative":
            sign = " " if self.upper else "-"
            return (
                f"{sign}∂{field}/∂{axis_name} = {value_expr}"
                f"   @ {axis_name}={self.axis_coord}"
            )
        if target == "mixed":
            sign = " " if self.upper else "-"
            return (
                f"{sign}∂{field}/∂{axis_name} + ({value_expr})*{field} = "
                f"{const_expr}   @ {axis_name}={self.axis_coord}"
            )
        msg = f"Unsupported target `{target}`"
        raise NotImplementedError(msg)

    def __eq__(self, other):
        """Checks for equality neglecting the `upper` property."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            super().__eq__(other)
            and self._input == other._input
            and self.value_cell == other.value_cell
        )

    def copy(
        self: ExpressionBC, upper: bool | None = None, rank: int | None = None
    ) -> ExpressionBC:
        """Return a copy of itself, but with a reference to the same grid."""
        return self.__class__(
            grid=self.grid,
            axis=self.axis,
            upper=self.upper if upper is None else upper,
            rank=self.rank if rank is None else rank,
            value=self._input["value_expr"],
            const=self._input["const_expr"],
            target=self._input["target"],
            user_funcs=self._input["user_funcs"],
            value_cell=self.value_cell,
        )

    def to_subgrid(self: ExpressionBC, subgrid: GridBase) -> ExpressionBC:
        """Converts this boundary condition to one valid for a given subgrid.

        Args:
            subgrid (:class:`GridBase`):
                Grid of the new boundary conditions

        Returns:
            :class:`BCBase`: Boundary conditions valid on the subgrid
        """
        # use `issubclass`, so that `self.grid` could be `UnitGrid`, while `subgrid` is
        # `CartesianGrid`
        assert issubclass(self.grid.__class__, subgrid.__class__)

        if self.value_cell is not None:
            msg = "Custom value indices are not supported"
            raise NotImplementedError(msg)

        bc = self.__class__(
            grid=subgrid,
            axis=self.axis,
            upper=self.upper,
            rank=self.rank,
            value=self._input["value_expr"],
            const=self._input["const_expr"],
            target=self._input["target"],
            user_funcs=self._input["user_funcs"],
            value_cell=self.value_cell,
        )

        # The following call raise error when `value_cell` index is not in `subgrid`.
        bc._get_value_cell_index(with_ghost_cells=False)
        return bc

    def get_sparse_matrix_data(
        self, idx: tuple[int, ...]
    ) -> tuple[float, dict[int, float]]:
        raise NotImplementedError

    def get_virtual_point(self, arr, idx: tuple[int, ...] | None = None) -> float:
        raise NotImplementedError

    def _get_value_cell_index(self, with_ghost_cells: bool) -> int:
        if self.value_cell is None:
            # pick adjacent cell by default
            return super()._get_value_cell_index(with_ghost_cells)
        if self.value_cell >= 0:
            # positive indexing
            idx = int(self.value_cell)
            if idx >= self.grid.shape[self.axis]:
                size = self.grid.shape[self.axis]
                msg = f"Index {self.value_cell} out of bounds ({size=})"
                raise IndexError(msg)
            return idx + 1 if with_ghost_cells else idx
        # self.value_cell < 0:
        # negative indexing
        idx = int(self.value_cell)
        if idx < -self.grid.shape[self.axis]:
            size = self.grid.shape[self.axis]
            msg = f"Index {self.value_cell} out of bounds ({size=})"
            raise IndexError(msg)
        return idx - 1 if with_ghost_cells else idx

    def set_ghost_cells(self, data_full: NumericArray, *, args=None) -> None:
        dx = self.grid.discretization[self.axis]

        # prepare the array of slices to index bcs
        offset = data_full.ndim - self.grid.num_axes  # additional data axes
        idx_offset = [slice(None)] * offset
        idx_valid = [slice(1, -1)] * self.grid.num_axes
        idx_write: list[slice | int] = idx_offset + idx_valid  # type: ignore
        idx_write[offset + self.axis] = -1 if self.upper else 0
        idx_read = idx_write[:]
        idx_read[offset + self.axis] = self._get_value_cell_index(with_ghost_cells=True)

        if self.normal:
            assert offset > 0
            idx_write[offset - 1] = self.axis
            idx_read[offset - 1] = self.axis

        # prepare the arguments
        values = data_full[tuple(idx_read)]
        coords = self.grid._boundary_coordinates(axis=self.axis, upper=self.upper)
        coords = np.moveaxis(coords, -1, 0)  # point coordinates to first axis

        if args is None:
            if not self._is_func and self._func_expression.depends_on("t"):
                msg = (
                    "Require value for `t` for time-dependent BC. The value must be "
                    "passed explicitly via `args` when calling a differential operator."
                )
                raise RuntimeError(msg)
            t = 0.0
        else:
            t = float(args["t"])

        # calculate the virtual points
        data_full[tuple(idx_write)] = self._make_function()(values, dx, *coords, t)


class ExpressionValueBC(ExpressionBC):
    """Represents a boundary whose value is calculated from an expression.

    The expression is given as a string and will be parsed by :mod:`sympy`. The
    expression can contain typical mathematical operators and may depend on the value
    at the last support point next to the boundary (`value`), spatial coordinates
    defined by the grid marking the boundary point (e.g., `x` or `r`), and time `t`.
    """

    names = ["value_expression", "value_expr"]

    def __init__(
        self,
        grid: GridBase,
        axis: int,
        upper: bool,
        *,
        rank: int = 0,
        value: float | str | Callable = 0,
        target: ExpressionBCTargetType = "value",
        user_funcs: dict[str, Callable] | None = None,
        value_cell: int | None = None,
    ):
        super().__init__(
            grid,
            axis,
            upper,
            rank=rank,
            value=value,
            target=target,
            user_funcs=user_funcs,
            value_cell=value_cell,
        )

    __init__.__doc__ = ExpressionBC.__init__.__doc__


class ExpressionDerivativeBC(ExpressionBC):
    """Represents a boundary whose outward derivative is calculated from an expression.

    The expression is given as a string and will be parsed by :mod:`sympy`. The
    expression can contain typical mathematical operators and may depend on the value
    at the last support point next to the boundary (`value`), spatial coordinates
    defined by the grid marking the boundary point (e.g., `x` or `r`), and time `t`.
    """

    names = ["derivative_expression", "derivative_expr"]

    def __init__(
        self,
        grid: GridBase,
        axis: int,
        upper: bool,
        *,
        rank: int = 0,
        value: float | str | Callable = 0,
        target: ExpressionBCTargetType = "derivative",
        user_funcs: dict[str, Callable] | None = None,
        value_cell: int | None = None,
    ):
        super().__init__(
            grid,
            axis,
            upper,
            rank=rank,
            value=value,
            target=target,
            user_funcs=user_funcs,
            value_cell=value_cell,
        )

    __init__.__doc__ = ExpressionBC.__init__.__doc__


class ExpressionMixedBC(ExpressionBC):
    """Represents a boundary whose outward derivative is calculated from an expression.

    The expression is given as a string and will be parsed by :mod:`sympy`. The
    expression can contain typical mathematical operators and may depend on the value
    at the last support point next to the boundary (`value`), spatial coordinates
    defined by the grid marking the boundary point (e.g., `x` or `r`), and time `t`.
    """

    names = ["mixed_expression", "mixed_expr", "robin_expression", "robin_expr"]

    def __init__(
        self,
        grid: GridBase,
        axis: int,
        upper: bool,
        *,
        rank: int = 0,
        value: float | str | Callable = 0,
        const: float | str | Callable = 0,
        target: ExpressionBCTargetType = "mixed",
        user_funcs: dict[str, Callable] | None = None,
        value_cell: int | None = None,
    ):
        super().__init__(
            grid,
            axis,
            upper,
            rank=rank,
            value=value,
            const=const,
            target=target,
            user_funcs=user_funcs,
            value_cell=value_cell,
        )

    __init__.__doc__ = ExpressionBC.__init__.__doc__


class ConstBCBase(BCBase):
    """Base class representing a boundary whose virtual point is set from constants."""

    _value: NumericArray

    value_is_linked: bool
    """ bool: flag that indicates whether the value associated with this
    boundary condition is linked to :class:`~numpy.ndarray` managed by external
    code. """

    @fill_in_docstring
    def __init__(
        self,
        grid: GridBase,
        axis: int,
        upper: bool,
        *,
        rank: int = 0,
        value: float | NumericArray | str = 0,
    ):
        """
        Warning:
            {WARNING_EXEC} However, the function is safe when `value` cannot be
            an arbitrary string.

        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Flag indicating whether this boundary condition is associated with the
                upper side of an axis or not. In essence, this determines the direction
                of the local normal vector of the boundary.
            rank (int):
                The tensorial rank of the field for this boundary condition
            normal (bool):
                Flag indicating whether the condition is only applied in the normal
                direction.
            value (float or str or :class:`~numpy.ndarray`):
                a value stored with the boundary condition. The interpretation of this
                value depends on the type of boundary condition. If value is a single
                value (or tensor in case of tensorial boundary conditions), the same
                value is applied to all points. Inhomogeneous boundary conditions are
                possible by supplying an expression as a string, which then may depend
                on the axes names of the respective grid.
        """
        super().__init__(grid, axis, upper, rank=rank)
        self.value = value

    def __eq__(self, other):
        """Checks for equality neglecting the `upper` property."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return super().__eq__(other) and np.array_equal(self.value, other.value)

    @property
    def value(self) -> NumericArray:
        return self._value

    @value.setter
    @fill_in_docstring
    def value(self, value: float | NumericArray | str = 0):
        """Set the value of this boundary condition.

        Warning:
            {WARNING_EXEC}

        Args:
            value (float or str or array):
                a value stored with the boundary condition. The interpretation
                of this value depends on the type of boundary condition.
        """
        self._value = self._parse_value(value)

        if self._value.shape == self._shape_tensor:
            # value does not depend on space
            self.homogeneous = True
        elif self._value.shape == self._shape_tensor + self._shape_boundary:
            # inhomogeneous field
            self.homogeneous = False
        else:
            msg = (
                f"Dimensions {self._value.shape} of the value are incompatible with "
                f"rank {self.rank} and spatial dimensions {self._shape_boundary}"
            )
            raise ValueError(msg)

        self.value_is_linked = False

    def _repr_value(self):
        if self.value_is_linked:
            return [f"value=<linked: {self.value.ctypes.data}>"]
        if np.array_equal(self.value, 0):
            return []
        return [f"value={self.value!r}"]

    def __str__(self):
        if hasattr(self, "names"):
            if np.array_equal(self.value, 0):
                return f'"{self.names[0]}"'
            if self.value_is_linked:
                return (
                    f'{{"type": "{self.names[0]}", '
                    f'"value": <linked: {self.value.ctypes.data}>}}'
                )
            return f'{{"type": "{self.names[0]}", "value": {self.value}}}'
        return self.__repr__()

    @fill_in_docstring
    def _parse_value(self, value: float | NumericArray | str) -> NumericArray:
        """Parses a boundary value.

        Warning:
            {WARNING_EXEC}

        Args:
            value (array-like or str):
                The value given as a array of tensorial character and optionally
                dependent on space (along the boundary). Alternatively, a string
                can specify a mathematical expression that can optionally depend
                on the coordinates along the boundary. This expression is only
                supported for scalar boundary conditions.

        Returns:
            :class:`~numpy.ndarray`: The value at the boundary
        """
        if isinstance(value, str):
            # inhomogeneous value given by an expression
            if len(self._shape_tensor) > 0:
                msg = "Expressions for boundaries are only supported for scalar values."
                raise NotImplementedError(msg)

            from ...tools.expressions import ScalarExpression

            # determine which coordinates are allowed to vary
            axes_ids = list(range(self.axis)) + list(
                range(self.axis + 1, self.grid.num_axes)
            )

            # parse the expression with the correct variables
            bc_vars = [self.grid.axes[i] for i in axes_ids]
            expr = ScalarExpression(
                value, self.grid.axes, repl=self.grid.c._axes_alt_repl
            )

            if axes_ids:
                # extended boundary

                # get the coordinates at each point of the boundary
                bc_coords = np.meshgrid(
                    *[self.grid.axes_coords[i] for i in axes_ids], indexing="ij"
                )

                # determine the value at each of these points. Note that we here
                # iterate explicitly over all points because the expression might
                # not depend on some of the variables, but we still want the array
                # to contain a value at each boundary point
                result = np.empty_like(bc_coords[0])
                coords: dict[str, float] = dict.fromkeys(self.grid.axes, 0)
                # set the coordinate of this BC
                coords[self.grid.axes[self.axis]] = self.axis_coord
                for idx in np.ndindex(*result.shape):
                    for i, name in enumerate(bc_vars):
                        coords[name] = bc_coords[i][idx]
                    result[idx] = expr(**coords)

            else:
                # point boundary
                result = np.array(expr(self.axis_coord))

        elif np.isscalar(value):
            # scalar value applied to all positions
            result = np.broadcast_to(value, self._shape_tensor)

        else:
            # assume tensorial and/or inhomogeneous values
            value = np.asarray(value)

            if value.ndim == 0:
                # value is a scalar
                result = np.broadcast_to(value, self._shape_tensor)
            elif value.shape == self._shape_tensor + self._shape_boundary:
                # inhomogeneous field with all tensor components
                result = value
            elif value.shape == self._shape_tensor:
                # homogeneous field with all tensor components
                result = value
            else:
                msg = (
                    f"Dimensions {value.shape} of the given value are incompatible "
                    f"with the expected shape {self._shape_tensor} of the boundary "
                    f"value and its spatial dimensions {self._shape_boundary}. "
                    f"(rank={self.rank}, normal={self.normal})"
                )
                raise ValueError(msg)

        # check consistency
        if np.any(np.isnan(result)):
            _logger.warning("In valid values in %s", self)

        return result

    def link_value(self, value: NumericArray):
        """Link value of this boundary condition to external array."""
        assert value.data.c_contiguous

        shape = self._shape_tensor + self._shape_boundary
        if value.shape != shape:
            msg = (
                f"The shape of the value, {value.shape}, is incompatible with the "
                f"expected shape for this boundary condition, {shape}"
            )
            raise ValueError(msg)
        self._value = value
        self.homogeneous = False
        self.value_is_linked = True

    def copy(
        self: ConstBCBase,
        upper: bool | None = None,
        rank: int | None = None,
        value: float | NumericArray | str | None = None,
    ) -> ConstBCBase:
        """Return a copy of itself, but with a reference to the same grid."""
        obj = self.__class__(
            grid=self.grid,
            axis=self.axis,
            upper=self.upper if upper is None else upper,
            rank=self.rank if rank is None else rank,
            value=self.value if value is None else value,
        )
        if self.value_is_linked:
            obj.link_value(self.value)
        return obj

    def to_subgrid(self: ConstBCBase, subgrid: GridBase) -> ConstBCBase:
        """Converts this boundary condition to one valid for a given subgrid.

        Args:
            subgrid (:class:`GridBase`):
                Grid of the new boundary conditions

        Returns:
            :class:`ConstBCBase`: Boundary conditions valid on the subgrid
        """
        # use `issubclass`, so that `self.grid` could be `UnitGrid`, while `subgrid` is
        # `CartesianGrid`
        assert issubclass(self.grid.__class__, subgrid.__class__)
        if self.value_is_linked or not self.homogeneous:
            msg = "Cannot transfer complicated BC to subgrid"
            raise NotImplementedError(msg)

        return self.__class__(
            grid=subgrid,
            axis=self.axis,
            upper=self.upper,
            rank=self.rank,
            value=self.value,
        )


class ConstBC1stOrderBase(ConstBCBase):
    """Represents a single boundary in an BoundaryPair instance."""

    @abstractmethod
    def get_virtual_point_data(self) -> tuple[Any, Any, int]:
        """Return data suitable for calculating virtual points.

        Returns:
            tuple: the data structure associated with this virtual point
        """

    def get_sparse_matrix_data(
        self, idx: tuple[int, ...]
    ) -> tuple[float, dict[int, float]]:
        """Sets the elements of the sparse representation of this condition.

        Args:
            idx (tuple):
                The index of the point that must lie on the boundary condition

        Returns:
            float, dict: A constant value and a dictionary with indices and
            factors that can be used to calculate this virtual point
        """
        data = self.get_virtual_point_data()

        if self.homogeneous:
            const = data[0]
            factor = data[1]
        else:
            # obtain index of the boundary point
            idx_c = list(idx)
            del idx_c[self.axis]
            const = data[0][tuple(idx_c)]
            factor = data[1][tuple(idx_c)]

        return const, {data[2]: factor}

    def get_virtual_point(self, arr, idx: tuple[int, ...] | None = None) -> float:
        """Calculate the value of the virtual point outside the boundary.

        Args:
            arr (array):
                The data values associated with the grid
            idx (tuple):
                The index of the point to evaluate. This is a tuple of length
                `grid.num_axes` with the either -1 or `dim` as the entry for the
                axis associated with this boundary condition. Here, `dim` is the
                dimension of the axis. The index is optional if dim == 1.

        Returns:
            float: Value at the virtual support point
        """
        if idx is None:
            if self.grid.num_axes == 1:
                idx = (self.grid.shape[0] if self.upper else -1,)
            else:
                msg = "Index `idx` can only be deduced for grids with a single axis."
                raise ValueError(msg)

        # extract the 1d array
        arr_1d, _, bc_idx = _get_arr_1d(arr, idx, axis=self.axis)

        # calculate necessary constants
        const, factor, index = self.get_virtual_point_data()

        if self.homogeneous:
            return const + factor * arr_1d[..., index]  # type: ignore
        return const[bc_idx] + factor[bc_idx] * arr_1d[..., index]  # type: ignore

    def set_ghost_cells(self, data_full: NumericArray, *, args=None) -> None:
        # calculate necessary constants
        const, factor, index = self.get_virtual_point_data()

        # prepare the array of slices to index bcs
        offset = data_full.ndim - self.grid.num_axes  # additional data axes
        idx_offset = [slice(None)] * offset
        idx_valid = [slice(1, -1)] * self.grid.num_axes
        idx_write: list[slice | int] = idx_offset + idx_valid  # type: ignore
        idx_write[offset + self.axis] = -1 if self.upper else 0
        idx_read = idx_write[:]
        idx_read[offset + self.axis] = index + 1

        if self.normal:
            assert offset > 0
            idx_write[offset - 1] = self.axis
            idx_read[offset - 1] = self.axis

        if self.homogeneous and not np.isscalar(factor):
            # add dimension to const so it can be broadcasted to shape of data_full
            for _ in range(self.grid.num_axes - 1):
                factor = factor[..., np.newaxis]
                const = const[..., np.newaxis]

        # calculate the virtual points
        data_full[tuple(idx_write)] = const + factor * data_full[tuple(idx_read)]


class _PeriodicBC(ConstBC1stOrderBase):
    """Represents one part of a boundary condition."""

    def __init__(
        self,
        grid: GridBase,
        axis: int,
        upper: bool,
        *,
        flip_sign: bool = False,
    ):
        """
        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Flag indicating whether this boundary condition is associated with the
                upper side of an axis or not. In essence, this determines the direction
                of the local normal vector of the boundary.
            flip_sign (bool):
                Impose different signs on the two sides of the boundary
        """
        super().__init__(grid, axis, upper)
        self.flip_sign = flip_sign

    def __str__(self):
        return '"periodic"'

    def copy(self: _PeriodicBC, upper: bool | None = None) -> _PeriodicBC:  # type: ignore
        """Return a copy of itself, but with a reference to the same grid."""
        return self.__class__(
            grid=self.grid,
            axis=self.axis,
            upper=self.upper if upper is None else upper,
            flip_sign=self.flip_sign,
        )

    def to_subgrid(self: _PeriodicBC, subgrid: GridBase) -> _PeriodicBC:
        """Converts this boundary condition to one valid for a given subgrid.

        Args:
            subgrid (:class:`GridBase`):
                Grid of the new boundary conditions

        Returns:
            :class:`ConstBCBase`: Boundary conditions valid on the subgrid
        """
        # use `issubclass`, so that `self.grid` could be `UnitGrid`, while `subgrid` is
        # `CartesianGrid`
        assert issubclass(self.grid.__class__, subgrid.__class__)
        if self.value_is_linked or not self.homogeneous:
            msg = "Cannot transfer complicated BC to subgrid"
            raise NotImplementedError(msg)

        return self.__class__(
            grid=subgrid, axis=self.axis, upper=self.upper, flip_sign=self.flip_sign
        )

    def get_mathematical_representation(self, field_name: str = "C") -> str:
        """Return mathematical representation of the boundary condition."""
        if self.upper:
            other_coord = self.grid.axes_bounds[self.axis][0]
        else:
            other_coord = self.grid.axes_bounds[self.axis][1]

        axis_name = self.grid.axes[self.axis]
        if self.flip_sign:
            return (
                f"{field_name}({axis_name}={self.axis_coord})"
                f" = -{field_name}({axis_name}={other_coord})"
            )
        return (
            f"{field_name}({axis_name}={self.axis_coord})"
            f" = {field_name}({axis_name}={other_coord})"
        )

    def get_virtual_point_data(self) -> tuple[Any, Any, int]:
        index = 0 if self.upper else self.grid.shape[self.axis] - 1
        value = -1 if self.flip_sign else 1
        return (0.0, value, index)


class DirichletBC(ConstBC1stOrderBase):
    """Represents a boundary condition imposing the value."""

    names = ["value", "dirichlet"]  # identifiers for this boundary condition

    def get_mathematical_representation(self, field_name: str = "C") -> str:
        """Return mathematical representation of the boundary condition."""
        axis_name = self.grid.axes[self.axis]
        field = self._field_repr(field_name)
        return f"{field} = {self.value}   @ {axis_name}={self.axis_coord}"

    def get_virtual_point_data(self) -> tuple[Any, Any, int]:
        const: NumberOrArray = 2 * self.value
        index = self.grid.shape[self.axis] - 1 if self.upper else 0
        factor = -np.ones_like(const)
        return (const, factor, index)


class NeumannBC(ConstBC1stOrderBase):
    """Represents a boundary condition imposing the derivative in the outward normal
    direction of the boundary."""

    names = ["derivative", "neumann"]  # identifiers for this boundary condition

    def get_mathematical_representation(self, field_name: str = "C") -> str:
        """Return mathematical representation of the boundary condition."""
        sign = " " if self.upper else "-"
        axis_name = self.grid.axes[self.axis]
        deriv = f"∂{self._field_repr(field_name)}/∂{axis_name}"
        return f"{sign}{deriv} = {self.value}   @ {axis_name}={self.axis_coord}"

    def get_virtual_point_data(self) -> tuple[Any, Any, int]:
        dx = self.grid.discretization[self.axis]
        const = dx * self.value
        index = self.grid.shape[self.axis] - 1 if self.upper else 0
        factor = np.ones_like(const)
        return (const, factor, index)


class MixedBC(ConstBC1stOrderBase):
    r"""represents a mixed (or Robin) boundary condition imposing a derivative
    in the outward normal direction of the boundary that is given by an affine
    function involving the actual value:

    .. math::
        \partial_n c + \gamma c = \beta

    Here, :math:`c` is the field to which the condition is applied,
    :math:`\gamma` quantifies the influence of the field and :math:`\beta` is
    the constant term. Note that :math:`\gamma = 0` corresponds
    to Dirichlet conditions imposing :math:`\beta` as the derivative.
    Conversely,  :math:`\gamma \rightarrow \infty` corresponds to imposing a
    zero value on :math:`c`.

    This condition can be enforced by using one of the following variants

    .. code-block:: python

        bc = {"mixed": VALUE}
        bc = {"type": "mixed", "value": VALUE, "const": CONST}

    where `VALUE` corresponds to :math:`\gamma` and `CONST` to :math:`\beta`.
    """

    names = ["mixed", "robin"]

    def __init__(
        self,
        grid: GridBase,
        axis: int,
        upper: bool,
        *,
        rank: int = 0,
        value: float | NumericArray | str = 0,
        const: float | NumericArray | str = 0,
    ):
        r"""
        Args:
            grid (:class:`~pde.grids.base.GridBase`):
                The grid for which the boundary conditions are defined
            axis (int):
                The axis to which this boundary condition is associated
            upper (bool):
                Flag indicating whether this boundary condition is associated
                with the upper side of an axis or not. In essence, this
                determines the direction of the local normal vector of the
                boundary.
            rank (int):
                The tensorial rank of the field for this boundary condition
            value (float or str or array):
                The parameter :math:`\gamma` quantifying the influence of the field onto
                its normal derivative. If `value` is a single value (or tensor in case
                of tensorial boundary conditions), the same value is applied to all
                points.  Inhomogeneous boundary conditions are possible by supplying an
                expression as a string, which then may depend on the axes names of the
                respective grid.
            const (float or :class:`~numpy.ndarray` or str):
                The parameter :math:`\beta` determining the constant term for
                the boundary condition. Supports the same input as `value`.
        """
        super().__init__(grid, axis, upper, rank=rank, value=value)
        self.const = self._parse_value(const)

    def __eq__(self, other):
        """Checks for equality neglecting the `upper` property."""
        if not isinstance(other, self.__class__):
            return NotImplemented
        return super().__eq__(other) and self.const == other.const

    def copy(
        self: MixedBC,
        upper: bool | None = None,
        rank: int | None = None,
        value: float | NumericArray | str | None = None,
        const: float | NumericArray | str | None = None,
    ) -> MixedBC:
        """Return a copy of itself, but with a reference to the same grid."""
        obj = self.__class__(
            grid=self.grid,
            axis=self.axis,
            upper=self.upper if upper is None else upper,
            rank=self.rank if rank is None else rank,
            value=self.value if value is None else value,
            const=self.const if const is None else const,
        )
        if self.value_is_linked:
            obj.link_value(self.value)
        return obj

    def to_subgrid(self: MixedBC, subgrid: GridBase) -> MixedBC:
        """Converts this boundary condition to one valid for a given subgrid.

        Args:
            subgrid (:class:`GridBase`):
                Grid of the new boundary conditions

        Returns:
            :class:`ConstBCBase`: Boundary conditions valid on the subgrid
        """
        # use `issubclass`, so that `self.grid` could be `UnitGrid`, while `subgrid` is
        # `CartesianGrid`
        assert issubclass(self.grid.__class__, subgrid.__class__)
        if self.value_is_linked or not self.homogeneous:
            msg = "Cannot transfer complicated BC to subgrid"
            raise NotImplementedError(msg)

        return self.__class__(
            grid=subgrid,
            axis=self.axis,
            upper=self.upper,
            rank=self.rank,
            value=self.value,
            const=self.const,
        )

    def get_mathematical_representation(self, field_name: str = "C") -> str:
        """Return mathematical representation of the boundary condition."""
        sign = "" if self.upper else "-"
        axis_name = self.grid.axes[self.axis]
        field_repr = self._field_repr(field_name)
        deriv = f"∂{field_repr}/∂{axis_name}"
        return (
            f"{sign}{deriv} + {self.value} * {field_repr} = {self.const}"
            f"  @ {axis_name}={self.axis_coord}"
        )

    def get_virtual_point_data(self) -> tuple[Any, Any, int]:
        # calculate values assuming finite factor
        dx = self.grid.discretization[self.axis]
        with np.errstate(invalid="ignore"):
            const = np.asarray(2 * dx * self.const / (2 + dx * self.value))
            factor = np.asarray((2 - dx * self.value) / (2 + dx * self.value))

        # correct at places of infinite values
        const[~np.isfinite(factor)] = 0
        factor[~np.isfinite(factor)] = -1
        index = self.grid.shape[self.axis] - 1 if self.upper else 0
        return (const, factor, index)


class ConstBC2ndOrderBase(ConstBCBase):
    """Abstract base class for boundary conditions of 2nd order."""

    @abstractmethod
    def get_virtual_point_data(self) -> tuple[Any, Any, int, Any, int]:
        """Return data suitable for calculating virtual points.

        Returns:
            tuple: the data structure associated with this virtual point
        """

    def get_sparse_matrix_data(
        self, idx: tuple[int, ...]
    ) -> tuple[float, dict[int, float]]:
        """Sets the elements of the sparse representation of this condition.

        Args:
            idx (tuple):
                The index of the point that must lie on the boundary condition

        Returns:
            float, dict: A constant value and a dictionary with indices and
            factors that can be used to calculate this virtual point
        """
        data = self.get_virtual_point_data()

        if self.homogeneous:
            const = data[0]
            factor1 = data[1]
            factor2 = data[3]
        else:
            # obtain index of the boundary point
            idx_c = list(idx)
            del idx_c[self.axis]
            bc_idx = tuple(idx_c)
            const = data[0][bc_idx]
            factor1 = data[1][bc_idx]
            factor2 = data[3][bc_idx]

        return const, {data[2]: factor1, data[4]: factor2}

    def get_virtual_point(self, arr, idx: tuple[int, ...] | None = None) -> float:
        """Calculate the value of the virtual point outside the boundary.

        Args:
            arr (array):
                The data values associated with the grid
            idx (tuple):
                The index of the point to evaluate. This is a tuple of length
                `grid.num_axes` with the either -1 or `dim` as the entry for the
                axis associated with this boundary condition. Here, `dim` is the
                dimension of the axis. The index is optional if dim == 1.

        Returns:
            float: Value at the virtual support point
        """
        if idx is None:
            if self.grid.num_axes == 1:
                idx = (self.grid.shape[0] if self.upper else -1,)
            else:
                msg = "Index can only be deduced for grids with a single axis."
                raise ValueError(msg)

        # extract the 1d array
        arr_1d, _, bc_idx = _get_arr_1d(arr, idx, axis=self.axis)

        # calculate necessary constants
        data = self.get_virtual_point_data()

        if self.homogeneous:
            return (  # type: ignore
                data[0]
                + data[1] * arr_1d[..., data[2]]
                + data[3] * arr_1d[..., data[4]]
            )
        return (  # type: ignore
            data[0][bc_idx]
            + data[1][bc_idx] * arr_1d[..., data[2]]
            + data[3][bc_idx] * arr_1d[..., data[4]]
        )

    def set_ghost_cells(self, data_full: NumericArray, *, args=None) -> None:
        # calculate necessary constants
        data = self.get_virtual_point_data()

        # prepare the array of slices to index bcs
        offset = data_full.ndim - self.grid.num_axes  # additional data axes
        idx_offset = [slice(None)] * offset
        idx_valid = [slice(1, -1)] * self.grid.num_axes
        idx_write: list[slice | int] = idx_offset + idx_valid  # type: ignore
        idx_write[offset + self.axis] = -1 if self.upper else 0
        idx_1 = idx_write[:]
        idx_1[offset + self.axis] = data[2] + 1
        idx_2 = idx_write[:]
        idx_2[offset + self.axis] = data[4] + 1

        if self.normal:
            assert offset > 0
            idx_write[offset - 1] = self.axis
            idx_1[offset - 1] = self.axis
            idx_2[offset - 1] = self.axis

        # # add dimension to const until it can be broadcasted to shape of data_full
        const, factor1, factor2 = data[0], data[1], data[3]
        if self.homogeneous:
            if not np.isscalar(factor1):
                for _ in range(self.grid.num_axes - 1):
                    const = const[..., np.newaxis]
            if not np.isscalar(factor1):
                for _ in range(self.grid.num_axes - 1):
                    factor1 = factor1[..., np.newaxis]
            if not np.isscalar(factor2):
                for _ in range(self.grid.num_axes - 1):
                    factor2 = factor2[..., np.newaxis]

        # calculate the virtual points
        data_full[tuple(idx_write)] = (
            const
            + factor1 * data_full[tuple(idx_1)]
            + factor2 * data_full[tuple(idx_2)]
        )


class CurvatureBC(ConstBC2ndOrderBase):
    """Represents a boundary condition imposing the 2nd normal derivative at the
    boundary."""

    names = ["curvature", "second_derivative", "extrapolate"]  # identifiers for this BC

    def get_mathematical_representation(self, field_name: str = "C") -> str:
        """Return mathematical representation of the boundary condition."""
        sign = " " if self.upper else "-"
        axis_name = self.grid.axes[self.axis]
        deriv = f"∂²{self._field_repr(field_name)}/∂{axis_name}²"
        return f"{sign}{deriv} = {self.value}   @ {axis_name}={self.axis_coord}"

    def get_virtual_point_data(
        self,
    ) -> tuple[NumericArray, NumericArray, int, NumericArray, int]:
        """Return data suitable for calculating virtual points.

        Returns:
            tuple: the data structure associated with this virtual point
        """
        size = self.grid.shape[self.axis]
        dx = self.grid.discretization[self.axis]

        if size < 2:
            msg = "Need at least 2 support points to use curvature boundary condition"
            raise RuntimeError(msg)

        value = np.asarray(self.value * dx**2)
        f1 = np.full_like(value, 2.0)
        f2 = np.full_like(value, -1.0)
        if self.upper:
            i1, i2 = size - 1, size - 2
        else:
            i1, i2 = 0, 1
        return (value, f1, i1, f2, i2)


class NormalDirichletBC(DirichletBC):
    """Represents a boundary condition imposing the value on normal components."""

    names = ["normal_value", "normal_dirichlet", "dirichlet_normal"]
    normal = True


class NormalNeumannBC(NeumannBC):
    """Represents a boundary condition imposing the derivative of normal components in
    the outward normal direction of the boundary."""

    names = ["normal_derivative", "normal_neumann", "neumann_normal"]
    normal = True


class NormalMixedBC(MixedBC):
    r"""represents a mixed (or Robin) boundary condition setting the derivative of the
    normal components in the outward normal direction of the boundary using an affine
    function involving the actual value:

    .. math::
        \partial_n c + \gamma c = \beta

    Here, :math:`c` is the field to which the condition is applied,
    :math:`\gamma` quantifies the influence of the field and :math:`\beta` is
    the constant term. Note that :math:`\gamma = 0` corresponds
    to Dirichlet conditions imposing :math:`\beta` as the derivative.
    Conversely,  :math:`\gamma \rightarrow \infty` corresponds to imposing a
    zero value on :math:`c`.

    This condition can be enforced by using one of the following variants

    .. code-block:: python

        bc = {"mixed": VALUE}
        bc = {"type": "mixed", "value": VALUE, "const": CONST}

    where `VALUE` corresponds to :math:`\gamma` and `CONST` to :math:`\beta`.
    """

    names = ["normal_mixed", "normal_robin"]
    normal = True


class NormalCurvatureBC(CurvatureBC):
    """Represents a boundary condition imposing the 2nd normal derivative onto the
    normal components at the boundary."""

    names = ["normal_curvature"]  # identifiers for this boundary condition
    normal = True


def registered_boundary_condition_classes() -> dict[str, type[BCBase]]:
    """Returns all boundary condition classes that are currently defined.

    Returns:
        dict: a dictionary with the names of the boundary condition classes
    """
    return {
        cls_name: cls
        for cls_name, cls in BCBase._subclasses.items()
        if not ("Base" in cls_name or cls_name.startswith("_"))  # skip internal classes
    }


def registered_boundary_condition_names() -> dict[str, type[BCBase]]:
    """Returns all named boundary conditions that are currently defined.

    Returns:
        dict: a dictionary with the names of the boundary conditions that can be used
    """
    return dict(BCBase._conditions.items())
