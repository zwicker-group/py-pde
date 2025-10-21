"""Defines base class of backends that implement computations.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Callable, Literal

from ..fields import DataFieldBase
from ..grids import BoundariesBase, GridBase
from ..pdes.base import PDEBase
from ..solvers.base import SolverBase
from ..tools.typing import (
    DataSetter,
    FloatingArray,
    GhostCellSetter,
    Number,
    NumberOrArray,
    NumericArray,
    OperatorFactory,
    OperatorInfo,
    TField,
)

_base_logger = logging.getLogger(__name__.rsplit(".", 1)[0])
""":class:`logging.Logger`: Base logger for backends."""


class BackendBase:
    """Basic backend from which all other backends inherit."""

    _logger: logging.Logger  # logger instance to output information

    def __init__(self, name: str = "numpy"):
        self.name = name

    def __init_subclass__(cls, **kwargs) -> None:
        """Initialize class-level attributes of subclasses."""
        super().__init_subclass__(**kwargs)
        # create logger for this specific field class
        cls._logger = _base_logger.getChild(cls.__qualname__)

    def register_operator(
        self,
        grid_cls: type[GridBase],
        name: str,
        factory_func: OperatorFactory | None = None,
        **kwargs,
    ):
        """Register an operator for a particular grid.

        Example:
            The method can either be used directly:

            .. code-block:: python

                backend.register_operator(grid_class, "operator", make_operator)

            or as a decorator for the factory function:

            .. code-block:: python

                @backend.register_operator(grid_class, "operator")
                def make_operator(grid: GridBase): ...

        Args:
            grid (:class:`~pde.grid.base.GridBase`):
                Grid for which the operator is defined
            name (str):
                The name of the operator to register
            factory_func (callable):
                A function with signature ``(grid: GridBase, **kwargs)``, which takes
                a grid object and optional keyword arguments and returns an
                implementation of the given operator. This implementation is a function
                that takes a :class:`~numpy.ndarray` of discretized values as arguments
                and returns the resulting discretized data in a :class:`~numpy.ndarray`
                after applying the operator.
            rank_in (int):
                The rank of the input field for the operator
            rank_out (int):
                The rank of the field that is returned by the operator
        """
        from .registry import backends

        backends.register_operator(self.name, grid_cls, name, factory_func, **kwargs)

    def get_registered_operators(self, grid_id: GridBase | type[GridBase]) -> set[str]:
        """Returns all operators defined for a grid.

        Args:
            grid (:class:`~pde.grid.base.GridBase` or its type):
                Grid for which the operator need to be returned
        """
        from . import backends

        grid_cls = grid_id if inspect.isclass(grid_id) else grid_id.__class__

        # get all operators registered on the class
        operators = set()
        # add all custom defined operators
        classes = inspect.getmro(grid_cls)[:-1]  # type: ignore
        for cls in classes:
            operators |= set(backends._operators[self.name][cls].keys())

        return operators

    def get_operator_info(
        self, grid: GridBase, operator: str | OperatorInfo
    ) -> OperatorInfo:
        """Return an operator for a particular grid.

        Args:
            grid (:class:`~pde.grid.base.GridBase`):
                Grid for which the operator is needed
            operator (str):
                Identifier for the operator. Some examples are 'laplace', 'gradient', or
                'divergence'. The registered operators for this grid can be obtained
                from the :attr:`~pde.grids.base.GridBase.operators` attribute.

        Returns:
            :class:`~pde.grids.base.OperatorInfo`: information for the operator
        """
        if isinstance(operator, OperatorInfo):
            return operator
        assert isinstance(operator, str)

        from . import backends

        # look for defined operators on all parent grid classes (except `object`)
        classes = inspect.getmro(grid.__class__)[:-1]
        for cls in classes:
            if operator in backends._operators[self.name][cls]:
                return backends._operators[self.name][cls][operator]

        # throw an error since operator was not found
        raise NotImplementedError(
            f"Operator '{operator}' is not defined for backend {self.name}"
        )

    def make_ghost_cell_setter(self, boundaries: BoundariesBase) -> GhostCellSetter:
        """Return function that sets the ghost cells on a full array.

        Args:
            boundaries (:class:`~pde.grids.boundaries.axes.BoundariesBase`):
                Defines the boundary conditions for a particular grid, for which the
                setter should be defined.

        Returns:
            Callable with signature :code:`(data_full: NumericArray, args=None)`, which
            sets the ghost cells of the full data, potentially using additional
            information in `args` (e.g., the time `t` during solving a PDE)
        """
        raise NotImplementedError(
            f"Ghost cell setter not defined for backend {self.name}"
        )

    def make_data_setter(
        self, grid: GridBase, bcs: BoundariesBase | None = None
    ) -> DataSetter:
        """Create a function to set the valid part of a full data array.

        Args:
            bcs (:class:`~pde.grids.boundaries.axes.BoundariesBase`, optional):
                If supplied, the returned function also enforces boundary conditions by
                setting the ghost cells to the correct values
            backend (str):
                The backend to use for making the operator

        Returns:
            callable:
                Takes two numpy arrays, setting the valid data in the first one, using
                the second array. The arrays need to be allocated already and they need
                to have the correct dimensions, which are not checked. If `bcs` are
                given, a third argument is allowed, which sets arguments for the BCs.
        """
        raise NotImplementedError(f"Data setter not defined for backend {self.name}")

    def make_operator(
        self,
        grid: GridBase,
        operator: str | OperatorInfo,
        bcs: BoundariesBase,
        **kwargs,
    ) -> Callable[[NumericArray, NumericArray | None, Any], NumericArray]:
        """Return a compiled function applying an operator with boundary conditions.

        Args:
            operator (str):
                Identifier for the operator. Some examples are 'laplace', 'gradient', or
                'divergence'. The registered operators for this grid can be obtained
                from the :attr:`~pde.grids.base.GridBase.operators` attribute.
            bcs (:class:`~pde.grids.boundaries.axes.BoundariesBase`, optional):
                The boundary conditions used before the operator is applied
            **kwargs:
                Specifies extra arguments influencing how the operator is created.

        The returned function takes the discretized data on the grid as an input and
        returns the data to which the operator `operator` has been applied. The function
        only takes the valid grid points and allocates memory for the ghost points
        internally to apply the boundary conditions specified as `bc`. Note that the
        function supports an optional argument `out`, which if given should provide
        space for the valid output array without the ghost cells. The result of the
        operator is then written into this output array.

        The function also accepts an optional parameter `args`, which is forwarded to
        `set_ghost_cells`. This allows setting boundary conditions based on external
        parameters, like time. Note that since the returned operator will always be
        compiled by Numba, the arguments need to be compatible with Numba. The
        following example shows how to pass the current time `t`:

        Returns:
            callable: the function that applies the operator. This function has the
            signature (arr: NumericArray, out: NumericArray = None, args=None).
        """
        raise NotImplementedError(f"Operators not defined for backend {self.name}")

    def make_inner_prod_operator(
        self, field: DataFieldBase, *, conjugate: bool = True
    ) -> Callable[[NumericArray, NumericArray, NumericArray | None], NumericArray]:
        """Return operator calculating the dot product between two fields.

        This supports both products between two vectors as well as products
        between a vector and a tensor.

        Args:
            field (:class:`~pde.fields.datafield_base.DataFieldBase`):
                Field for which the inner product is defined
            conjugate (bool):
                Whether to use the complex conjugate for the second operand

        Returns:
            function that takes two instance of :class:`~numpy.ndarray`, which contain
            the discretized data of the two operands. An optional third argument can
            specify the output array to which the result is written.
        """
        raise NotImplementedError(f"Inner product not defined for backend {self.name}")

    def make_outer_prod_operator(
        self, field: DataFieldBase
    ) -> Callable[[NumericArray, NumericArray, NumericArray | None], NumericArray]:
        """Return operator calculating the outer product between two fields.

        This supports typically only supports products between two vector fields.

        Args:
            field (:class:`~pde.fields.datafield_base.DataFieldBase`):
                Field for which the outer product is defined
            conjugate (bool):
                Whether to use the complex conjugate for the second operand

        Returns:
            function that takes two instance of :class:`~numpy.ndarray`, which contain
            the discretized data of the two operands. An optional third argument can
            specify the output array to which the result is written.
        """
        raise NotImplementedError(f"Outer product not defined for backend {self.name}")

    def make_interpolator(
        self,
        field: DataFieldBase,
        *,
        fill: Number | None = None,
        with_ghost_cells: bool = False,
    ) -> Callable[[FloatingArray, NumericArray], NumberOrArray]:
        r"""Returns a function that can be used to interpolate values.

        Args:
            field (:class:`~pde.fields.datafield_base.DataFieldBase`):
                Field for which the interpolator is defined
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
        raise NotImplementedError(f"Interpolator not defined for backend {self.name}")

    def make_pde_rhs(
        self, eq: PDEBase, state: TField, **kwargs
    ) -> Callable[[NumericArray, float], NumericArray]:
        """Return a function for evaluating the right hand side of the PDE.

        Args:
            eq (:class:`~pde.pdes.base.PDEBase`):
                The object describing the differential equation
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which information can be extracted

        Returns:
            Function returning deterministic part of the right hand side of the PDE
        """
        raise NotImplementedError(
            f"PDE right hand side not defined for backend {self.name}"
        )

    def make_sde_rhs(
        self, eq: PDEBase, state: TField, **kwargs
    ) -> Callable[[NumericArray, float], tuple[NumericArray, NumericArray]]:
        """Return a function for evaluating the right hand side of the SDE.

        Args:
            eq (:class:`~pde.pdes.base.PDEBase`):
                The object describing the differential equation
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which information can be extracted

        Returns:
            Function returning deterministic part of the right hand side of the PDE
            together with a noise realization.
        """
        raise NotImplementedError(
            f"SDE right hand side not defined for backend {self.name}"
        )

    def make_inner_stepper(
        self,
        solver: SolverBase,
        stepper_style: Literal["fixed", "adaptive"],
        state: TField,
        dt: float,
    ) -> Callable:
        """Return a stepper function using an explicit scheme.

        Args:
            solver (:class:`~pde.solvers.base.SolverBase`):
                The solver instance, which determines how the stepper is constructed
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            stepper_style (str):
                Determines how the stepper is expected to work

        Returns:
            Function that can be called to advance the `state` from time `t_start` to
            time `t_end`. The function call signature is `(state: numpy.ndarray,
            t_start: float, t_end: float)`
        """
        raise NotImplementedError(f"Steppers are not defined for backend {self.name}")
