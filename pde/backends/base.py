"""Defines base class of backends that implement computations.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import inspect
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Callable, NamedTuple

import numpy as np

from ..fields.datafield_base import DataFieldBase
from ..grids.base import GridBase
from ..grids.boundaries.axes import BoundariesBase
from ..tools.typing import DataSetter, GhostCellSetter, NumericArray, OperatorFactory

_base_logger = logging.getLogger(__name__.rsplit(".", 1)[0])
""":class:`logging.Logger`: Base logger for backends."""


class OperatorInfo(NamedTuple):
    """Stores information about an operator."""

    factory: OperatorFactory
    rank_in: int
    rank_out: int
    name: str = ""  # attach a unique name to help caching


class BackendBase(ABC):
    """Basic backend from which all other backends inherit."""

    _operators: dict[type[GridBase], dict[str, OperatorInfo]]
    _logger: logging.Logger  # logger instance to output information

    def __init__(self, name: str = "numpy"):
        self.name = name
        self._operators = defaultdict(dict)

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
        else:
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
        raise NotImplementedError(f"Operator '{operator}' is not defined.")

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

    @abstractmethod
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
        raise NotImplementedError

    def make_inner_prod_operator(
        self, field: DataFieldBase, *, conjugate: bool = True
    ) -> Callable[[NumericArray, NumericArray, NumericArray | None], NumericArray]:
        """Return operator calculating the dot product between two fields.

        This supports both products between two vectors as well as products
        between a vector and a tensor.

        Args:
            conjugate (bool):
                Whether to use the complex conjugate for the second operand

        Returns:
            function that takes two instance of :class:`~numpy.ndarray`, which contain
            the discretized data of the two operands. An optional third argument can
            specify the output array to which the result is written.
        """
        raise NotImplementedError
