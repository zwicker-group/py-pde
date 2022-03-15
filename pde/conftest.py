"""
This file is used to configure the test environment when running py.test

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from pde import environment


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
    parser.addoption(
        "--showconfig",
        action="store_true",
        default=False,
        help="show configuration at beginning of the test run",
    )


def pytest_sessionstart(session):
    """pytest hook to display configuration at startup"""
    if session.config.getoption("--showconfig", default=False):
        terminal_reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        capture_manager = session.config.pluginmanager.get_plugin("capturemanager")
        with capture_manager.global_and_fixture_disabled():
            terminal_reporter.write(f"{'='*33} CONFIGURATION {'='*32}\n")

            for category, data in environment().items():
                if hasattr(data, "items"):
                    terminal_reporter.write(f"\n{category}:\n")
                    for key, value in data.items():
                        terminal_reporter.write(f"    {key}: {value}\n")
                else:
                    data_formatted = data.replace("\n", "\n    ")
                    terminal_reporter.write(f"{category}: {data_formatted}\n")


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
