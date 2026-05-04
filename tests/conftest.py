"""This file is used to configure the test environment when running py.test.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import logging
import platform

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pde import config
from pde.backends import BackendBase, backend_registry, get_backend
from pde.backends.numba.utils import random_seed
from pde.tools.misc import module_available

# ensure we use the Agg backend, so figures are not displayed
plt.switch_backend("agg")

_logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def create_clean_session():
    """Run some initial commands."""
    if module_available("torch"):
        import torch

        torch._dynamo.reset()


@pytest.fixture(autouse=True)
def _setup_and_teardown():
    """Helper function adjusting environment before and after tests."""
    # raise all underflow errors
    np.seterr(all="raise", under="ignore")

    # run the actual test
    yield

    # clean up open matplotlib figures after the test
    plt.close("all")

    if module_available("torch"):
        import torch

        # Clean up torch cache of the compiler since otherwise some tests might fail
        # with an exception due to too many recompilations.
        torch.compiler.reset()


@pytest.fixture(autouse=False, name="rng")
def init_random_number_generators():
    """Get a random number generator and set the seed of the random number generator.

    The function returns an instance of :func:`~numpy.random.default_rng()` and
    initializes the default generators of both :mod:`numpy` and :mod:`numba`.
    """
    random_seed()
    return np.random.default_rng(0)


# try registering specific torch backends for various devices
if module_available("torch"):
    from pde.backends.torch import TorchBackend

    # TODO: Test mps device once torch-mps works well again
    for device in ["cpu", "cuda"]:  # , "mps"]:
        try:
            backend: BackendBase = TorchBackend(
                config["backend"]["torch"].copy(), name=f"torch-{device}", device=device
            )
        except RuntimeError:
            _logger.info("Torch device `%s` is unavailable", device)
        else:
            backend_registry.register_backend(backend)

if module_available("jax"):
    from pde.backends.jax import JaxBackend

    for device in ["cpu", "cuda"]:
        try:
            backend = JaxBackend(
                config["backend"]["jax"].copy(), name=f"jax-{device}", device=device
            )
        except RuntimeError:
            _logger.info("Jax device `%s` is unavailable", device)
        else:
            backend_registry.register_backend(backend)


@pytest.fixture
def backend(request):
    """Fixture that sets up the backend.

    This fixture can generate special backends with custom configurations and it makes
    sure that backends are available. If they are not, the respective test is skipped
    automatically.
    """
    if request.param == "numba":
        # the numba backend
        if not module_available("numba"):
            pytest.skip("`numba` is not available")
        backend = get_backend("numba")

    elif request.param.startswith("jax"):
        # a jax backend, which might possibly include a device
        if not module_available("jax"):
            pytest.skip("`jax` is not available")
        try:
            backend = get_backend(request.param)
        except KeyError as err:
            pytest.skip(str(err))

    elif request.param.startswith("torch"):
        # a torch backend, which might possibly include a device
        if not module_available("torch"):
            pytest.skip("`torch` is not available")
        if platform.system() == "Windows":
            pytest.skip("Skip `torch` tests on Windows")

        try:
            backend = get_backend(request.param)
        except KeyError as err:
            pytest.skip(str(err))

    else:
        # try loading a generic backend by name
        backend = get_backend(request.param)

    return backend


def pytest_configure(config):
    """Add markers to the configuration."""
    config.addinivalue_line("markers", "interactive: test is interactive")
    config.addinivalue_line("markers", "multiprocessing: test requires multiprocessing")
    config.addinivalue_line("markers", "slow: test runs slowly")


def pytest_addoption(parser):
    """Pytest hook to add command line options parsed by pytest."""
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="also run tests marked by `slow`",
    )
    parser.addoption(
        "--runinteractive",
        action="store_true",
        default=False,
        help="also run tests marked by `interactive`",
    )
    parser.addoption(
        "--use_mpi",
        action="store_true",
        default=False,
        help="only run tests marked by `multiprocessing`",
    )


def pytest_collection_modifyitems(config, items):
    """Pytest hook to filter a collection of tests."""
    # parse options provided to py.test
    running_cov = config.getvalue("--cov")
    runslow = config.getoption("--runslow", default=False)
    runinteractive = config.getoption("--runinteractive", default=False)
    use_mpi = config.getoption("--use_mpi", default=False)

    # prepare markers
    skip_cov = pytest.mark.skip(reason="skipped during coverage run")
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_interactive = pytest.mark.skip(reason="need --runinteractive option to run")
    skip_serial = pytest.mark.skip(reason="serial test, but --use_mpi option was set")

    # check each test item
    for item in items:
        if "no_cover" in item.keywords and running_cov:
            # skip some tests when determining test coverage
            item.add_marker(skip_cov)
        if "slow" in item.keywords and not runslow:
            # skip slow tests unless they are specifically requested
            item.add_marker(skip_slow)
        if "interactive" in item.keywords and not runinteractive:
            # skip interactive tests unless they are specifically requested
            item.add_marker(skip_interactive)

        if use_mpi and "multiprocessing" not in item.keywords:
            # skip all non-MPI tests in an MPI test run
            item.add_marker(skip_serial)
