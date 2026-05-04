"""Defines base class of backends that implement computations.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import inspect
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Generic, TypeVar, overload

from ..tools.config import _OMITTED, Config, ConfigLike
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
    TNativeArray,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from numpy.typing import DTypeLike

    from ..fields import DataFieldBase
    from ..grids import BoundariesBase, GridBase
    from ..pdes.base import PDEBase
    from ..solvers.base import SolverBase
    from ..tools.expressions import ExpressionBase
    from ..tools.typing import (
        BinaryOperatorImplType,
        OperatorImplType,
        StepperType,
        TFunc,
    )

_base_logger = logging.getLogger(__name__.rsplit(".", 1)[0])
""":class:`logging.Logger`: Base logger for backends."""

_RESERVED_BACKEND_NAMES: set[str] = {
    "auto",
    "best",
    "config",
    "default",
    "none",
    "undetermined",
    "unknown",
}
TValue = TypeVar("TValue")


class BackendBase(Generic[TNativeArray]):
    """Basic backend from which all other backends inherit.

    The generic type parameter `TNativeArray` determines the type of the native data
    representation of the backend.
    """

    implementation: str = "undefined"
    """str: The name of the python module that is used to implement this backend. This
    information can be used to distinguish the general implementation of backends."""

    copy_data: bool = False
    """bool: Flag indicating whether data needs to be copied between numpy's
    representation on CPU and a native device."""

    config_inheritance: list[str] = []
    """list: Additional backends that are queried for configuration parameters."""

    config: Config
    """dict: Configuration options of this backend."""

    _logger: logging.Logger  # logger instance to output information
    _operators: dict[type[GridBase], dict[str, OperatorInfo]]
    """dict: All operators registered for this backend class.

    Operators are registered for each grid class individually. Note that operators are
    registered on the backend classes, so that we can use inheritance to find operators
    defined on parent classes.
    """

    def __init__(self, config: ConfigLike | None, *, name: str | None = None):
        """Initialize the backend.

        Args:
            config (:class:`~pde.tools.config.Config`):
                Configuration data for the backend
            name (str):
                The name of the backend
        """
        if config is None:
            self.config = Config(mode="insert")
        elif isinstance(config, Config):
            self.config = config
        else:
            self.config = Config(config)

        if name is None:
            name = self.__class__.__name__  # extract a default name
        if name in _RESERVED_BACKEND_NAMES:
            self._logger.warning("Backend uses reserved name.")
        self.name = name

    @classmethod
    def from_args(
        cls, config: ConfigLike | None, args: str = "", *, name: str | None = None
    ):
        """Initialize backend with extra arguments.

        Args:
            config (:class:`~pde.tools.config.Config`):
                Configuration data for the backend
            args (str):
                Additional arguments that determine how the backend is initialized
            name (str):
                The name of the backend
        """
        raise NotImplementedError

    def __init_subclass__(cls, **kwargs) -> None:
        """Initialize class-level attributes of subclasses.

        Args:
            **kwargs: Additional arguments for subclass initialization
        """
        super().__init_subclass__(**kwargs)
        # create logger for this specific field class
        cls._logger = _base_logger.getChild(cls.__qualname__)
        cls._operators = defaultdict(dict)

    def __repr__(self) -> str:
        """Return concise string representation of this backend."""
        return f"{self.__class__.__name__}(name={self.name!r})"

    @property
    def info(self) -> dict[str, Any]:
        """dict: relevant information about the backend"""
        return {"name": self.name, "implementation": self.implementation}

    def _config_parameter(self, key: str, default: Any = _OMITTED) -> Any:
        """Returns the value of a configuration option, respecting inheritance

        Args:
            key (str):
                The name of the configuration parameter
            default
        """
        if key in self.config:
            return self.config[key]

        from .registry import backend_registry

        for name in self.config_inheritance:
            backend = backend_registry.get_backend(name)
            if key in backend.config:
                return backend.config[key]

        if default is _OMITTED:
            msg = f"Parameter `{key}` cannot be found"
            raise KeyError(msg)
        return default

    @overload
    def numpy_to_native(self, value: NumericArray) -> TNativeArray: ...
    @overload
    def numpy_to_native(self, value: TValue) -> TValue: ...

    def numpy_to_native(self, value: Any) -> Any:
        """Convert values from numpy to native representation.

        Args:
            value: The value to convert from numpy representation
        """
        return value

    @overload
    def native_to_numpy(self, value: TNativeArray) -> NumericArray: ...
    @overload
    def native_to_numpy(self, value: TValue) -> TValue: ...

    def native_to_numpy(self, value: Any) -> Any:
        """Convert native values to numpy representation.

        Args:
            value: The value to convert to numpy representation
        """
        return value

    def compile_function(self, func: TFunc) -> TFunc:
        """General method that compiles a user function.

        Args:
            func (callable):
                The function that needs to be compiled for this backend
        """
        msg = f"Compiling functions is not supported by backend `{self.name}`"
        raise NotImplementedError(msg)

    def _apply_function(
        self, func: Callable, *values: NumericArray, **kwargs
    ) -> NumericArray:
        r"""Apply a native function to numpy data and return result.

        Args:
            func (callable):
                The function defined in the native space of the backend
            *values (:class:`~numpy.ndarray`):
                The array data that is fed to the function
            **kwargs:
                Additional arguments that are forwarded to the function call

        Returns:
            :class:`~numpy.ndarray`: The result as a numpy array
        """
        values_native = [self.numpy_to_native(value) for value in values]
        res_native = func(*values_native, **kwargs)
        return self.native_to_numpy(res_native)

    def _apply_operator(
        self, func: Callable, *values: NumericArray, out: NumericArray, **kwargs
    ) -> None:
        r"""Apply a native operator to numpy data and store result in `out`.

        Args:
            func (callable):
                The operator defined in the native space of the backend
            *values (:class:`~numpy.ndarray`):
                The array data that is fed to the function
            out (:class:`~numpy.ndarray`):
                The array to which the results are written
            *args, **kwargs:
                Additional arguments that are forwarded to the function call
        """
        raise NotImplementedError

    @classmethod
    def register_operator(
        cls,
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

                BackendClass.register_operator(grid_class, "operator", make_operator)

            or as a decorator for the factory function:

            .. code-block:: python

                @BackendClass.register_operator(grid_class, "operator")
                def make_operator(grid: GridBase): ...

        Args:
            grid_cls (:class:`~pde.grid.base.GridBase`):
                Grid class for which the operator is defined
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
            """Helper function to register the operator.

            Args:
                factor_func_arg (OperatorFactory):
                    The operator factory function to register
            """
            cls._operators[grid_cls][name] = OperatorInfo(
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
            grid_id (:class:`~pde.grid.base.GridBase` or its type):
                Grid or grid class for which the operators need to be returned
        """
        # determine all classes that are relevant to the grid
        grid_cls = grid_id if inspect.isclass(grid_id) else grid_id.__class__
        grid_classes = inspect.getmro(grid_cls)[:-1]  # type: ignore

        # get all operators registered on the class from all relevant backends and grids
        operators = set()
        for backend_cls in inspect.getmro(self.__class__)[:-1]:
            operators_dict = getattr(backend_cls, "_operators", {})
            for grid in grid_classes:
                if grid in operators_dict:
                    operators |= set(operators_dict[grid].keys())

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

        # determine all classes that are relevant to the grid
        grid_classes = inspect.getmro(grid.__class__)[:-1]

        # look for operators on all parent backend and grid classes (except `object`)
        for backend_cls in inspect.getmro(self.__class__)[:-1]:
            operators_dict = getattr(backend_cls, "_operators", {})
            for grid_cls in grid_classes:
                if grid_cls in operators_dict and operator in operators_dict[grid_cls]:
                    return operators_dict[grid_cls][operator]  # type: ignore

        # throw an error since operator was not found
        msg = (
            f"Backend `{self.name}` does not define operator '{operator}' for grid "
            f"`{grid.__class__.__name__}`. Defined operators are: "
            f"{sorted(self.get_registered_operators(grid))}."
        )
        raise NotImplementedError(msg)

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

    def make_integrator(self, grid: GridBase) -> Callable[[TNativeArray], TNativeArray]:
        """Return function that integrates discretized data over a grid.

        Note that this function takes and returns data in the native representation of
        the backend. If this function is used in a multiprocessing run (using MPI), the
        integrals are performed on all subgrids and then accumulated. Each process then
        receives the same value representing the global integral.

        Args:
            grid (:class:`~pde.grid.base.GridBase`):
                Grid for which the operator is needed

        Returns:
            A function that takes a numpy array and returns the integral with the
            correct weights given by the cell volumes.
        """
        msg = f"Integrator not defined for backend {self.name}"
        raise NotImplementedError(msg)

    def make_operator_no_bc(
        self,
        grid: GridBase,
        operator: str | OperatorInfo,
        *,
        dtype: DTypeLike | None = None,
        **kwargs,
    ) -> OperatorImplType:
        """Return a compiled function applying an operator without boundary conditions.

        A function that takes the discretized full data as an input and an array of
        valid data points to which the result of applying the operator is written.

        Note:
            The resulting function does not check whether the ghost cells of the input
            array have been supplied with sensible values. It is the responsibility of
            the user to set the values of the ghost cells beforehand. Use this function
            only if you absolutely know what you're doing. In all other cases,
            :meth:`make_operator` is probably the better choice.

        Args:
            grid (:class:`~pde.grid.base.GridBase`):
                Grid for which the operator is needed
            operator (str):
                Identifier for the operator. Some examples are 'laplace', 'gradient', or
                'divergence'. The registered operators for this grid can be obtained
                from the :attr:`~pde.grids.base.GridBase.operators` attribute.
            dtype (numpy dtype):
                The data type of the field.
            **kwargs:
                Specifies extra arguments influencing how the operator is created.

        Returns:
            callable: the function that applies the operator. This function has the
            signature (arr: NumericArray, out: NumericArray), so they `out` array need
            to be supplied explicitly.
        """
        # determine the operator for the chosen backend
        operator_info = self.get_operator_info(grid, operator)
        return operator_info.factory(grid, **kwargs)

    def make_operator(
        self,
        grid: GridBase,
        operator: str | OperatorInfo,
        *,
        bcs: BoundariesBase,
        dtype: DTypeLike | None = None,
        **kwargs,
    ) -> OperatorType:
        """Return a compiled function applying an operator with boundary conditions.

        Args:
            grid (:class:`~pde.grid.base.GridBase`):
                Grid for which the operator is needed
            operator (str):
                Identifier for the operator. Some examples are 'laplace', 'gradient', or
                'divergence'. The registered operators for this grid can be obtained
                from the :attr:`~pde.grids.base.GridBase.operators` attribute.
            bcs (:class:`~pde.grids.boundaries.axes.BoundariesBase`):
                The boundary conditions used before the operator is applied
            dtype (numpy dtype):
                The data type of the field.
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
    ) -> BinaryOperatorImplType:
        """Return operator calculating the dot product between two fields.

        This supports both products between two vectors as well as products
        between a vector and a tensor.

        Args:
            field (:class:`~pde.fields.datafield_base.DataFieldBase`):
                Field for which the inner product is defined
            conjugate (bool):
                Whether to use the complex conjugate for the second operand

        Returns:
            Function that takes two instance of native data arrays, which contain the
            discretized data of the two operands. An optional third argument can specify
            the output array to which the result is written.
        """
        msg = f"Inner product not defined for backend {self.name}"
        raise NotImplementedError(msg)

    def make_outer_prod_operator(self, field: DataFieldBase) -> BinaryOperatorImplType:
        """Return operator calculating the outer product between two fields.

        This supports typically only supports products between two vector fields.

        Args:
            field (:class:`~pde.fields.datafield_base.DataFieldBase`):
                Field for which the outer product is defined

        Returns:
            Function that takes two instance of native data arrays, which contain the
            discretized data of the two operands. An optional third argument can specify
            the output array to which the result is written.
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

    def make_pde_rhs(
        self,
        eq: PDEBase,
        state: TField,
    ) -> Callable[[TNativeArray, float], TNativeArray]:
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

    def make_expression_function(
        self,
        expression: ExpressionBase,
        *,
        single_arg: bool = False,
        user_funcs: dict[str, Callable] | None = None,
    ) -> Callable[..., NumberOrArray]:
        """Return a function evaluating an expression.

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
        self, operator: int | str = "MAX", mpi_run: bool = False
    ) -> Callable[[float], float]:
        """Return function that synchronizes values between multiple MPI processes.

        Warning:
            The default implementation does not synchronize anything. This is simply a
            hook, which can be used by backends that support MPI

        Args:
            operator (str or int):
                Flag determining how the value from multiple nodes is combined.
                Possible values include "MAX", "MIN", and "SUM".
            mpi_run (bool):
                Whether MPI is actually used. If `False`, the method returns a no-op.

        Returns:
            Function that can be used to synchronize values across nodes
        """
        from ..tools import mpi

        if not mpi_run or mpi.size == 1:
            # serial run, which does not require synchronization

            def synchronize_value(value: float) -> float:
                return value

        else:
            # parallel run, which requires synchronization

            def synchronize_value(value: float) -> float:
                """Return error synchronized across all cores."""
                return mpi.mpi_allreduce(value, operator=operator)  # type: ignore

        return synchronize_value

    def make_gaussian_noise(
        self, field: TField, *, rng: np.random.Generator
    ) -> Callable[[], TNativeArray]:
        """Create a function generating Gaussian white noise.

        This noise is already scaled to respect different cell volumes of the grid.

        Args:
            field (:class:`~pde.fields.base.FieldBase`):
                An example for the field from which the grid and other information can
                be extracted
            rng (:class:`~numpy.random.Generator`):
                Random number generator (default: :func:`~numpy.random.default_rng()`).
        """
        raise NotImplementedError

    def make_stepper(self, solver: SolverBase, state: TField) -> StepperType:
        """Create a field-based stepping function for a given solver.

        Args:
            solver (:class:`~pde.solvers.base.SolverBase`):
                The solver instance, which determines how the stepper is constructed
            state (:class:`~pde.fields.base.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted

        Returns:
            Function that can be called to advance the `state` from time `t_start` to
            time `t_end`.
        """
        if self.copy_data:
            self._logger.warning(
                "Backend requires that data is copied, so it likely needs to implement "
                "its own `make_stepper` method to control data."
            )

        inner_stepper = solver._make_inner_stepper(state)

        def stepper(state: TField, t_start: float, t_end: float) -> float:
            """Advance `state` by executing the backend-level stepping function."""
            # call the backend-level stepping function with field data directly
            return inner_stepper(state.data, t_start, t_end)

        return stepper  # type: ignore
