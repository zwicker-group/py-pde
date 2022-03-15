"""
This file is used to configure the test environment when running py.test

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    """helper function adjusting environment before and after tests"""
    # raise all underflow errors
    np.seterr(all="raise", under="ignore")

    # run the actual test
    yield

    # clean up open matplotlib figures after the test
    plt.close("all")


def pytest_addoption(parser):
    """pytest hook to add command line options parsed by pytest"""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption(
        "--runinteractive",
        action="store_true",
        default=False,
        help="run interactive tests",
    )


def pytest_collection_modifyitems(config, items):
    """pytest hook to filter a collection of tests"""
    runslow = config.getoption("--runslow", default=False)
    runinteractive = config.getoption("--runinteractive", default=False)

    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    skip_interactive = pytest.mark.skip(reason="need --runinteractive option to run")
    for item in items:
        if "slow" in item.keywords and not runslow:
            item.add_marker(skip_slow)
        if "interactive" in item.keywords and not runinteractive:
            item.add_marker(skip_interactive)
