"""
This file is used to configure the test environment when running py.test

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def suppressing():
    """ helper function adjusting message reporting for all tests """
    # raise all underflow errors
    np.seterr(all="raise", under="ignore")

    # run the actual test
    yield

    # clean up open matplotlib figures after the test
    plt.close("all")
