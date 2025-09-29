"""Defines base class of backends that implement computations.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

from .. import config
from .base import BackendBase

_RESERVED_NAMES = {"default", "auto", "config", "none", "best"}
_logger = logging.getLogger(__name__)
""":class:`logging.Logger`: Logger instance."""


class BackendRegistry:
    """Class handling all backends."""

    _backends: dict[str, BackendBase | str]

    def __init__(self):
        self._backends = {}

    def register_package(self, package_path: str, name: str) -> None:
        """Register a backend python package (without loading it yet)

        Args:
            package_path (str):
                Import path for the package
            name (str):
                Name of the backend
        """
        if name in _RESERVED_NAMES:
            _logger.warning("Reserved name `%s` should not be used.", name)
        if name in self._backends:
            if isinstance(self._backends[name], str):
                _logger.info("Redefining backend %s", name)
            else:
                raise RuntimeError("Cannot register package for loaded backend")
        self._backends[name] = package_path

    def add(self, backend: BackendBase) -> None:
        """Add a backend object to the registry.

        Args:
            name (str):
                Name of the backend
            backend (:class:`~pde.backends.base.BackendBase`):
                Implementation of the backend
        """
        if backend.name in _RESERVED_NAMES:
            _logger.warning("Reserved name `%s` should not be used.", backend.name)
        if backend.name in self._backends:
            _logger.info("Reloading backend %s", backend.name)
        self._backends[backend.name] = backend

    def __getitem__(self, backend: str | BackendBase) -> BackendBase:
        """Load a backend."""
        if isinstance(backend, BackendBase):
            return backend
        name = str(backend)

        # handle special names
        if name == "config":
            name = config["default_backend"]

        # load the backend
        backend_obj = self._backends.get(name, None)
        if backend_obj is None:
            backends = ", ".join(self._backends.keys())
            raise KeyError(f"Backend `{name}` not in [{backends}]")

        if isinstance(backend_obj, str):
            # load the backend package
            import importlib

            importlib.import_module(backend_obj)
            backend_obj = self._backends[name]

        assert isinstance(backend_obj, BackendBase)
        return backend_obj

    def __contains__(self, name: str) -> bool:
        return name == "config" or name in self._backends

    def __iter__(self) -> Iterator[str]:
        """Iterate over the defined backends."""
        return self._backends.keys().__iter__()


# initiate the backend registry – there should only be one instance of this class
backends = BackendRegistry()
