"""Defines base class of backends that implement computations.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import inspect
import logging
from abc import abstractmethod
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypeVar

from ..tools.typing import (
    DataSetter,
    FloatingArray,
    GhostCellSetter,
    Number,
    NumberOrArray,
    NumericArray,
    OperatorFactory,
    OperatorInfo,
    OperatorType,
    TField,
)

if TYPE_CHECKING:
    from ..fields import DataFieldBase
    from ..grids import BoundariesBase, GridBase
    from ..pdes.base import PDEBase
    from ..solvers.base import SolverBase
    from ..tools.expressions import ExpressionBase

_base_logger = logging.getLogger(__name__.rsplit(".", 1)[0])
""":class:`logging.Logger`: Base logger for backends."""

TFunc = TypeVar("TFunc", bound=Callable)


class BackendBase:
    """Basic backend from which all other backends inherit."""

    _logger: logging.Logger  # logger instance to output information
    _operators: dict[type[GridBase], dict[str, OperatorInfo]]
    """dict: all operators registered for all backends"""

    def __init__(self, name: str = ""):
        self.name = name
        self._operators = defaultdict(dict)

    def __init_subclass__(cls, **kwargs) -> None:
        """Initialize class-level attributes of subclasses."""
        super().__init_subclass__(**kwargs)
        # create logger for this specific field class
        cls._logger = _base_logger.getChild(cls.__qualname__)

    def compile_function(self, func: TFunc) -> TFunc:
        """General method that compiles a user function.

        Args:
            func (callable):
                The function that needs to be compiled for this backend
        """
        msg = f"Compiling functions is not supported by backend `{self.name}`"
        raise NotImplementedError(msg)

    def register_operator(
        self,
        grid_cls: type[GridBase],
        name: str,
        factory_func: OperatorFactory | None = None,
        *,
        rank_in: int = 0,
        rank_out: int = 0,
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

        def register_operator(factor_func_arg: OperatorFactory):
            """Helper function to register the operator."""
            self._operators[grid_cls][name] = OperatorInfo(
                factory=factor_func_arg, rank_in=rank_in, rank_out=rank_out, name=name
            )
            return factor_func_arg

        if factory_func is None:
            # method is used as a decorator, so return the helper function
            return register_operator
        # method is used directly
        register_operator(factory_func)
        return None

    def get_registered_operators(self, grid_id: GridBase | type[GridBase]) -> set[str]:
        """Returns all operators defined for a grid.

        Args:
            grid (:class:`~pde.grid.base.GridBase` or its type):
                Grid for which the operator need to be returned
        """
        grid_cls = grid_id if inspect.isclass(grid_id) else grid_id.__class__

        # get all operators registered on the class
        operators = set()
        # add all custom defined operators
        classes = inspect.getmro(grid_cls)[:-1]  # type: ignore
        for cls in classes:
            operators |= set(self._operators[cls].keys())

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

        # look for defined operators on all parent grid classes (except `object`)
        classes = inspect.getmro(grid.__class__)[:-1]
        for cls in classes:
            if operator in self._operators[cls]:
                return self._operators[cls][operator]

        # throw an error since operator was not found
        msg = (
            f"Backend `{self.name}` does not define operator '{operator}' for grid "
            f"`{grid.__class__.__name__}`. Defined operators are: "
            f"{sorted(self.get_registered_operators(cls))}."
        )
        raise NotImplementedError(msg)

    @abstractmethod
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
        msg = f"Ghost cell setter not defined for backend {self.name}"
        raise NotImplementedError(msg)

    def make_data_setter(
        self, grid: GridBase, bcs: BoundariesBase | None = None
    ) -> DataSetter:
        """Create a function to set the valid part of a full data array.

        Args:
            grid (:class:`~pde.grid.base.GridBase`):
                Grid for which the data setter is defined
            bcs (:class:`~pde.grids.boundaries.axes.BoundariesBase`, optional):
                If supplied, the returned function also enforces boundary conditions by
                setting the ghost cells to the correct values

        Returns:
            callable:
                Takes two numpy arrays, setting the valid data in the first one, using
                the second array. The arrays need to be allocated already and they need
                to have the correct dimensions, which are not checked. If `bcs` are
                given, a third argument is allowed, which sets arguments for the BCs.
        """
        msg = f"Data setter not defined for backend {self.name}"
        raise NotImplementedError(msg)

    def make_integrator(
        self, grid: GridBase
    ) -> Callable[[NumericArray], NumberOrArray]:
        """Return function that integrates discretized data over a grid.

        If this function is used in a multiprocessing run (using MPI), the integrals are
        performed on all subgrids and then accumulated. Each process then receives the
        same value representing the global integral.

        Args:
            grid (:class:`~pde.grid.base.GridBase`):
                Grid for which the operator is needed

        Returns:
            A function that takes a numpy array and returns the integral with the
            correct weights given by the cell volumes.
        """
        msg = f"Integrator not defined for backend {self.name}"
        raise NotImplementedError(msg)

    def make_operator(
        self,
        grid: GridBase,
        operator: str | OperatorInfo,
        bcs: BoundariesBase,
        **kwargs,
    ) -> OperatorType:
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
        parameters, like time.

        Returns:
            callable: the function that applies the operator. This function has the
            signature (arr: NumericArray, out: NumericArray = None, args=None).
        """
        msg = f"Operators not defined for backend {self.name}"
        raise NotImplementedError(msg)

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
        msg = f"Inner product not defined for backend {self.name}"
        raise NotImplementedError(msg)

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
        msg = f"Outer product not defined for backend {self.name}"
        raise NotImplementedError(msg)

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
        msg = f"Interpolator not defined for backend {self.name}"
        raise NotImplementedError(msg)

    def make_noise_realization(
        self, eq: PDEBase, state: TField
    ) -> Callable[[NumericArray, float], NumericArray | None]:
        """Return a function for evaluating the noise term of the PDE.

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted
            noise (float or :class:`~numpy.ndarray` or None):
                Variance of the additive Gaussian white noise

        Returns:
            Function calculating noise
        """
        msg = f"Noise terms not defined for backend {self.name}"
        raise NotImplementedError(msg)

    def make_pde_rhs(
        self, eq: PDEBase, state: TField
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
        msg = f"PDE right hand side not defined for backend {self.name}"
        raise NotImplementedError(msg)

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
        msg = f"Steppers are not defined for backend {self.name}"
        raise NotImplementedError(msg)

    def make_expression_function(
        self,
        expression: ExpressionBase,
        *,
        single_arg: bool = False,
        user_funcs: dict[str, Callable] | None = None,
    ) -> Callable[..., NumberOrArray]:
        """Return a function evaluating an expression for a particular backend.

        Args:
            expression (:class:`~pde.tools.expression.ExpressionBase`):
                The expression that is converted to a function
            single_arg (bool):
                Determines whether the returned function accepts all variables in a
                single argument as an array or whether all variables need to be
                supplied separately.
            user_funcs (dict):
                Additional functions that can be used in the expression.

        Returns:
            function: the function
        """
        msg = f"Expressions are not supported by backend {self.name}"
        raise NotImplementedError(msg)

    def make_mpi_synchronizer(
        self, operator: int | str = "MAX"
    ) -> Callable[[float], float]:
        """Return function that synchronizes values between multiple MPI processes.

        Warning:
            The default implementation does not synchronize anything. This is simply a
            hook, which can be used by backends that support MPI

        Args:
            operator (str or int):
                Flag determining how the value from multiple nodes is combined.
                Possible values include "MAX", "MIN", and "SUM".

        Returns:
            Function that can be used to synchronize values across nodes
        """

        def synchronize_value(value: float) -> float:
            return value

        return synchronize_value
