"""
This file is used to configure the test environment when running py.test

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pde.tools.misc import module_available


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    """helper function adjusting environment before and after tests"""
    # raise all underflow errors
    np.seterr(all="raise", under="ignore")

    # run the actual test
    yield

    # clean up open matplotlib figures after the test
    plt.close("all")


def pytest_configure(config):
    """add markers to the configuration"""
    config.addinivalue_line("markers", "interactive: test is interactive")
    config.addinivalue_line("markers", "multiprocessing: test requires multiprocessing")
    config.addinivalue_line("markers", "slow: test runs slowly")


def pytest_addoption(parser):
    """pytest hook to add command line options parsed by pytest"""
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
    """pytest hook to filter a collection of tests"""
    # parse options provided to py.test
    running_cov = config.getvalue("--cov")
    runslow = config.getoption("--runslow", default=False)
    runinteractive = config.getoption("--runinteractive", default=False)
    use_mpi = config.getoption("--use_mpi", default=False)
    has_numba_mpi = module_available("numba_mpi")

    # prepare markers
    skip_cov = pytest.mark.skip(reason="skipped during coverage run")
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_interactive = pytest.mark.skip(reason="need --runinteractive option to run")
    skip_serial = pytest.mark.skip(reason="serial test, but --use_mpi option was set")
    skip_mpi = pytest.mark.skip(reason="mpi test, but `numba_mpi` not available")

    # check each test item
    for item in items:
        if "no_cover" in item.keywords and running_cov:
            item.add_marker(skip_cov)
        if "slow" in item.keywords and not runslow:
            item.add_marker(skip_slow)
        if "interactive" in item.keywords and not runinteractive:
            item.add_marker(skip_interactive)

        if "multiprocessing" in item.keywords and not has_numba_mpi:
            item.add_marker(skip_mpi)
        if use_mpi and "multiprocessing" not in item.keywords:
            item.add_marker(skip_serial)
