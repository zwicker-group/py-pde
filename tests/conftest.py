"""
This file is used to configure the test environment when running py.test
"""

import pytest
import numpy as np
import matplotlib.pyplot as plt


@pytest.fixture(scope="function", autouse=True)
def adjust_messages():
    """ helper function adjusting message reporting for all tests """
    # raise all underflow errors
    np.seterr(all="raise", under="ignore")

    # run the actual test    
    yield

    # clean up after the test
    plt.close("all")
