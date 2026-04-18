"""Defines the registry for managing backends.

.. autosummary::
   :nosignatures:

   BackendRegistry

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import contextlib
import importlib
import logging
from typing import TYPE_CHECKING

from .. import config as global_config
from ..tools.config import Config, ConfigLike
from .base import _RESERVED_BACKEND_NAMES, BackendBase

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from ..tools.config import Parameter


_logger = logging.getLogger(__name__)
""":class:`logging.Logger`: Logger instance."""


class BackendRegistry:
    """Class handling all backends and their configurations.

    Backends can exist in three different states in registry:
    * Registered meta-information on how to load a backend package
    * Loaded backend module, so the class is available
    * Fully instantiated :class:`~pde.backends.base.BackendBase` classes
    """

    _packages: dict[str, str]
    """dict: backends whose packages have been registered"""
    _configs: dict[str, Config]
    """dict: configurations of backend classes"""
    _classes: dict[str, type[BackendBase]]
    """dict: backends whose classes have been defined"""
    _backends: dict[str, BackendBase]
    """dict: backends that have been instantiated"""

    def __init__(self):
        self._packages = {}
        self._configs = {}
        self._classes = {}
        self._backends = {}

    def register_package(
        self,
        name: str,
        package_path: str,
        *,
        config: ConfigLike | None = None,
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
        if name in self._packages:
            _logger.info("Redefining backend `%s`", name)
        self._packages[name] = package_path
        self._configs[name] = Config(config)

    def register_class(self, name: str, cls: type[BackendBase]):
        """Register a backend class.

        Args:
            name (str):
                Name of the backend
            cls (subclass of :class:`~pde.backend.base.BackendBase`):
                The class for creating a backend
        """
        if name in _RESERVED_BACKEND_NAMES:
            _logger.warning("Reserved backend name `%s` should not be used.", name)
        if name in self._classes:
            _logger.info("Redefining backend `%s`", name)
        self._classes[name] = cls

    def register_backend(
        self, backend: BackendBase, *, link_config: bool = False
    ) -> None:
        """Register a loaded backend object.

        Args:
            backend (:class:`~pde.backends.base.BackendBase`):
                Implementation of the backend
            link_config (bool):
                If True, the configuration of `backend` is linked with the global
                configuration, so that both show consistent values.
        """
        if backend.name in _RESERVED_BACKEND_NAMES:
            _logger.warning(
                "Reserved backend name `%s` should not be used.", backend.name
            )
        if backend.name in self._backends:
            if isinstance(self._backends[backend.name], str):
                _logger.info("Loading backend `%s`", backend.name)
            else:
                _logger.info("Reloading backend `%s`", backend.name)
        self._backends[backend.name] = backend

        if link_config:
            self._configs[backend.name] = backend.config

    def get_config(self, name: str) -> Config:
        """Get configuration of a particular backend.

        An empty configuration is returned if nothing was found.

        Args:
            name (str):
                Name of the backend

        Returns:
            :class:`~pde.tools.config.Config`: the configuration
        """
        try:
            return self._configs[name]
        except KeyError:
            return Config()

    def _get_class(self, name: str) -> type[BackendBase]:
        """Get the class associated with a particular backend.

        Args:
            name (str):
                The name of the backend class to load
        """
        if name not in self._classes:
            # determine the backend information to load it
            try:
                package_path = self._packages[name]
            except KeyError as err:
                backends = ", ".join(self._packages.keys())
                msg = f"Backend `{name}` not in [{backends}]"
                raise KeyError(msg) from err

            # load the backend from a python package, which should register its class
            importlib.import_module(package_path)
            if name not in self._classes:
                msg = f"Backend `{name}` was loaded, but did not register its class."
                raise RuntimeError(msg)

        return self._classes[name]

    def get_backend(
        self, name: str, *, config: ConfigLike | None = None, **kwargs
    ) -> BackendBase:
        """Return backend object, potentially loading the respective package.

        Args:
            name (str):
                Name of the backend to be loaded.
            config (dict):
                Configuration options for this specific backend
            **kwargs:
                Additional options of the backend

        Returns:
            :class:`~pde.backends.base.BackendBase`: An instance of the backend with
            the particular configuration
        """
        # handle special names
        if name == "default":
            name = global_config["default_backend"]

        # check whether the precise backend has been instantiated already
        if name not in self._backends:
            # create backend from the class definition
            parts = name.split(":", 1)
            if len(parts) == 2:
                cls_name, args = parts
            elif len(parts) == 1:
                cls_name, args = parts[0], None
            else:
                raise RuntimeError
            cls = self._get_class(cls_name)
            if config is None:
                config = self.get_config(name)

            if args:
                backend_obj = cls.from_args(config, args, name=name, **kwargs)
                self.register_backend(backend_obj, link_config=False)
            else:
                backend_obj = cls(config, name=name, **kwargs)
                self.register_backend(backend_obj, link_config=True)

        return self._backends[name]

    def __getitem__(self, name: str) -> BackendBase:
        """Return backend object, potentially loading the respective package."""
        return self.get_backend(name)

    def __contains__(self, name: str) -> bool:
        return name == "default" or name in self._backends

    def __iter__(self) -> Iterator[str]:
        """Iterate over the names of the defined backends."""
        return self._backends.keys().__iter__()

    def values(self) -> Iterator[BackendBase]:
        """Iterate over all backends that can be imported."""
        for name in self:
            with contextlib.suppress(ImportError):
                yield self[name]


# initiate the backend registry - there should only be one instance of this class
backend_registry = BackendRegistry()


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


def get_backend(backend: str | BackendBase) -> BackendBase:
    """Return backend specified by string of instance.

    Args:
        backend (str or :class:`~pde.backends.base.BackendBase`):
            Backend specified by name given as a string. If the string contains a colon,
            the first part determines the backend, whereas the second part can be used
            to convey additional information. For example, :code:`torch:cuda` may load a
            torch backend an use a cuda device. As a special case, we also allow full
            backend objects, which are simply returned. This is a simple way to allow
            providing full backend objects in places where we otherwise would expect a
            backend name.
    """
    if isinstance(backend, BackendBase):
        # backend is already initialized
        return backend

    if isinstance(backend, str):
        # backend is given by name
        return backend_registry.get_backend(backend)

    raise TypeError


def registered_backends() -> list[str]:
    """Returns all registered backends."""
    return sorted(set(backend_registry._packages) | set(backend_registry._classes))
