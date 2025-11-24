"""Defines base class of backends that implement computations.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import importlib
import logging
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

from .. import config
from ..grids import GridBase
from ..tools.typing import OperatorFactory, OperatorInfo
from .base import BackendBase

_RESERVED_NAMES = {
    "auto",
    "best",
    "config",
    "default",
    "none",
    "undetermined",
    "unknown",
}
_logger = logging.getLogger(__name__)
""":class:`logging.Logger`: Logger instance."""


class BackendRegistry:
    """Class handling all backends."""

    _backends: dict[str, str | BackendBase]
    """dict: all backends, either as a reference to a package or as an object"""
    _hooks: dict[str, dict[str, dict[str, Any]]]
    """dict: all hooks registered for all backends"""
    _operators: dict[str, dict[type[GridBase], dict[str, OperatorInfo]]]
    """dict: all operators registered for all backends"""

    def __init__(self):
        self._backends = {}
        self._operators = defaultdict(lambda: defaultdict(dict))

    def register_package(self, package_path: str, name: str) -> None:
        """Register a backend python package (without loading it yet)

        Args:
            package_path (str):
                Import path for the package
            name (str):
                Name of the backend
        """
        if name in _RESERVED_NAMES:
            _logger.warning("Reserved backend name `%s` should not be used.", name)
        if name in self._backends:
            if isinstance(self._backends[name], str):
                _logger.info("Redefining backend `%s`", name)
            else:
                raise RuntimeError("Cannot register package for loaded backend")
        self._backends[name] = package_path

    def add(self, backend: BackendBase) -> None:
        """Add a loaded backend object.

        This object can replace a previously registered python package.

        Args:
            backend (:class:`~pde.backends.base.BackendBase`):
                Implementation of the backend
        """
        if backend.name in _RESERVED_NAMES:
            _logger.warning(
                "Reserved backend name `%s` should not be used.", backend.name
            )
        if backend.name in self._backends:
            _logger.info("Reloading backend `%s`", backend.name)
        self._backends[backend.name] = backend

    def __getitem__(self, backend: str | BackendBase) -> BackendBase:
        """Return backend object, potentially loading the respective package.

        As a special case, we also allow full backend objects, which are simply
        returned. This is a simple way to allow providing full backend objects in places
        where we otherwise would expect a backend name.
        """
        if isinstance(backend, BackendBase):
            return backend
        name = str(backend)  # if it's not a class, it needs to be a backend name

        # handle special names
        if name == "config":
            name = config["default_backend"]

        # get the backend from the registry
        backend_obj = self._backends.get(name, None)
        if backend_obj is None:
            backends = ", ".join(self._backends.keys())
            raise KeyError(f"Backend `{name}` not in [{backends}]")

        # load the backend from a python package if necessary
        if isinstance(backend_obj, str):
            importlib.import_module(backend_obj)
            backend_obj = self._backends[name]

        assert isinstance(backend_obj, BackendBase)
        return backend_obj

    def __contains__(self, name: str) -> bool:
        return name == "config" or name in self._backends

    def __iter__(self) -> Iterator[str]:
        """Iterate over the defined backends."""
        return self._backends.keys().__iter__()

    def register_operator(
        self,
        backend: str,
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

                backends.register_operator("numba", grid_cls, "operator", make_operator)

            or as a decorator for the factory function:

            .. code-block:: python

                @backend.register_operator("numba", grid_cls, "operator")
                def make_operator(grid: GridBase): ...

        Args:
            backend (str):
                Name of the backend for which we register the option
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
            self._operators[backend][grid_cls][name] = OperatorInfo(
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


# initiate the backend registry – there should only be one instance of this class
backends = BackendRegistry()
