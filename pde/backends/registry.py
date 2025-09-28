"""Defines base class of backends that implement computations.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import warnings
from collections import UserDict
from collections.abc import Iterator

from .. import config
from .base import BackendBase

_RESERVED_NAMES = {"default", "auto", "config", "none", "best"}


class BackendRegistry:
    """Class handling all backends."""

    _backends: dict[str, BackendBase]

    def __init__(self):
        self._backends = {}

    def add(self, backend: BackendBase) -> None:
        """
        Args:
            name (str):
                Name of the backend
        """
        if backend.name in _RESERVED_NAMES:
            warnings.warn("Backend {name} is a reserved name and should not be used.")
        if backend.name in self._backends:
            warnings.warn(f"Redefining backend {backend.name}")
        self._backends[backend.name] = backend

    def __getitem__(self, backend: str | BackendBase) -> BackendBase:
        if isinstance(backend, BackendBase):
            return backend
        name = str(backend)

        # handle special names
        if name == "config":
            name = config["default_backend"]

        # load the backend
        try:
            return self._backends[name]
        except KeyError:
            backends = ", ".join(self._backends.keys())
            raise KeyError(f"Backend `{name}` not in [{backends}]") from None

    def __contains__(self, name: str) -> bool:
        return name == "config" or name in self._backends

    def __iter__(self) -> Iterator[BackendBase]:
        return self._backends.values().__iter__()


# initiate the backend registry – there should only be one instance of this class
backends = BackendRegistry()
