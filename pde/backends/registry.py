"""Defines base class of backends that implement computations.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

from .. import config
from ..tools.config import Config
from .base import _RESERVED_BACKEND_NAMES, BackendBase

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from ..tools.config import Parameter


_logger = logging.getLogger(__name__)
""":class:`logging.Logger`: Logger instance."""


class BackendRegistry:
    """Class handling all backends and their configurations."""

    _backends: dict[str, str | BackendBase]
    """dict: all backends, either as a reference to a package or as an object"""
    _configs: dict[str, Config]
    """dict: configurations of all backends"""

    def __init__(self):
        self._backends = {}
        self._configs = {}

    def register_package(
        self,
        name: str,
        package_path: str,
        *,
        config: list[Parameter] | None = None,
    ) -> None:
        """Register a backend python package (without loading it yet)

        Args:
            name (str):
                Name of the backend
            package_path (str):
                Import path for the package
            config (list):
                Configuration options for the package
        """
        if name in _RESERVED_BACKEND_NAMES:
            _logger.warning("Reserved backend name `%s` should not be used.", name)
        if name in self._backends:
            if isinstance(self._backends[name], str):
                _logger.info("Redefining backend `%s`", name)
            else:
                msg = "Cannot register package for loaded backend"
                raise RuntimeError(msg)

        self._backends[name] = package_path
        self._configs[name] = Config(config)

    def add(self, backend: BackendBase) -> None:
        """Add a loaded backend object.

        This object can replace a previously registered python package.

        Args:
            backend (:class:`~pde.backends.base.BackendBase`):
                Implementation of the backend
        """
        if backend.name in _RESERVED_BACKEND_NAMES:
            _logger.warning(
                "Reserved backend name `%s` should not be used.", backend.name
            )
        if backend.name in self._backends:
            _logger.info("Reloading backend `%s`", backend.name)
        self._backends[backend.name] = backend
        if backend.name not in self._configs:
            self._configs[backend.name] = Config()
        backend.config = self._configs[backend.name]

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
        if name == "default":
            name = config["default_backend"]

        # get the backend from the registry
        backend_obj = self._backends.get(name, None)
        if backend_obj is None:
            backends = ", ".join(self._backends.keys())
            msg = f"Backend `{name}` not in [{backends}]"
            raise KeyError(msg)

        # load the backend from a python package if necessary
        if isinstance(backend_obj, str):
            importlib.import_module(backend_obj)
            backend_obj = self._backends[name]

        assert isinstance(backend_obj, BackendBase)
        return backend_obj

    def __contains__(self, name: str) -> bool:
        return name == "default" or name in self._backends

    def __iter__(self) -> Iterator[str]:
        """Iterate over the defined backends."""
        return self._backends.keys().__iter__()


# initiate the backend registry - there should only be one instance of this class
backends = BackendRegistry()


def load_default_config(module_path: str | Path) -> list[Parameter] | None:
    """Load a default configuration from a module.

    Args:
        module_path (str):
            String to the module to be loaded
    """
    module_name = (
        str(module_path).replace(".", "_").replace("/", "_").replace("\\", "_")
    )

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None:
        _logger.warning("Could not load module `%s`", module_path)
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    try:
        return module.DEFAULT_CONFIG  # type: ignore
    except AttributeError:
        _logger.warning("Configuration module had no variable `DEFAULT_CONFIG`")
        return None


def registered_backends() -> list[str]:
    """Returns all registered backends."""
    return sorted(backends._backends)
