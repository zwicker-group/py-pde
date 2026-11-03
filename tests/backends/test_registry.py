"""Test the backend registry.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import pytest

from pde import config
from pde.backends import (
    BackendBase,
    available_backends,
    backend_registry,
    registered_backends,
)
from pde.backends.registry import BackendRegistry
from pde.tools.misc import module_available


def test_backend_availability():
    """Test whether the availability of the standard backends is detected."""
    # backends whose packages are hard dependencies of `py-pde`
    assert backend_registry.is_available("numpy")
    assert backend_registry.is_available("numba")

    # optional backends are only available when their package is installed
    for name in ["jax", "torch"]:
        assert backend_registry.is_available(name) == module_available(name)

    # additional information after a colon does not affect the check
    assert backend_registry.is_available("numba:parallel")

    # the default backend must obviously be usable
    assert backend_registry.is_available("default")

    # unknown backends are simply reported as being unavailable
    assert not backend_registry.is_available("not_a_backend")


def test_available_backends():
    """Test the list of available backends."""
    available = available_backends()
    assert "numpy" in available
    assert set(available) <= set(registered_backends())
    assert all(backend_registry.is_available(name) for name in available)


def test_backend_availability_custom():
    """Test the availability of custom backends."""
    registry = BackendRegistry()
    with config():  # registering a package also adds a node to the global config
        registry.register_package("dummy", "not.a.module", requires=["not_a_module"])
        assert not registry.is_available("dummy")

        registry.register_package("simple", "not.a.module")
        assert registry.is_available("simple")  # no requirements => always available

        # instantiated backends are available, even without a registered package
        class MyBackend(BackendBase): ...

        registry.register_backend(MyBackend({}, name="my_backend"))
        assert registry.is_available("my_backend")

        with pytest.raises(RuntimeError):
            registry.register_package("dummy", "not.a.module")
