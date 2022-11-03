"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde.tools.mpi import mpi_allreduce, mpi_recv, mpi_send, rank, size


@pytest.mark.multiprocessing
def test_send_recv():
    """test basic send and receive"""
    if size == 1:
        pytest.skip("Run without multiprocessing")

    data = np.arange(5)
    if rank == 0:
        out = np.empty_like(data)
        mpi_recv(out, 1, 1)
        np.testing.assert_allclose(out, data)
    elif rank == 1:
        mpi_send(data, 0, 1)


@pytest.mark.multiprocessing
def test_allreduce():
    """test basic send and receive"""
    from numba_mpi import Operator

    data = np.arange(size)
    total = mpi_allreduce(data[rank])
    assert total == data.sum()

    total = mpi_allreduce(data[rank], int(Operator.MAX))
    assert total == data.max()
