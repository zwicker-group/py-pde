import pytest
import numpy as np


@pytest.fixture(scope="session", autouse=True)
def adjust_messages():
    """ helper function adjusting message reporting for all tests """
    # raise all underflow errors
    np.seterr(all='raise', under='ignore')
