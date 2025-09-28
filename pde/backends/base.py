"""Defines base class of backends that implement computations.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import inspect
import logging
import warnings
from abc import ABCMeta, abstractmethod
from collections import defaultdict
from typing import NamedTuple

from ..grids.base import GridBase
from ..grids.boundaries.axes import BoundariesBase
from ..tools.typing import GhostCellSetter, NumericArray, OperatorFactory

_base_logger = logging.getLogger(__name__.rsplit(".", 1)[0])
""":class:`logging.Logger`: Base logger for backends."""


class OperatorInfo(NamedTuple):
    """Stores information about an operator."""

    factory: OperatorFactory
    rank_in: int
    rank_out: int
    name: str = ""  # attach a unique name to help caching


class BackendBase(metaclass=ABCMeta):
    """Abstract base class for describing backends."""

    _operators: dict[type[GridBase], dict[str, OperatorInfo]]
    _logger: logging.Logger  # logger instance to output information

    def __init__(self, name: str):
        self.name = name
        self._operators = defaultdict(dict)

    def __init_subclass__(cls, **kwargs) -> None:
        """Initialize class-level attributes of subclasses."""
        super().__init_subclass__(**kwargs)
        # create logger for this specific field class
        cls._logger = _base_logger.getChild(cls.__qualname__)

    @abstractmethod
    def load_backend(self) -> None: ...

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

        def ghost_cell_setter(data_full: NumericArray, args=None) -> None:
            """Default implementation that simply uses the python interface."""
            boundaries.set_ghost_cells(data_full, *args)

        return ghost_cell_setter
