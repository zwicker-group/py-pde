"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from pde.tools.mpi import mpi_allreduce, mpi_recv, mpi_send, rank, size


@pytest.mark.multiprocessing
def test_send_recv():
    """Test basic send and receive."""
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
@pytest.mark.parametrize("operator", ["MAX", "MIN", "SUM"])
def test_allreduce(operator, rng):
    """Test MPI allreduce function."""
    data = rng.uniform(size=size)
    result = mpi_allreduce(data[rank], operator=operator)

    if operator == "MAX":
        assert result == data.max()
    elif operator == "MIN":
        assert result == data.min()
    elif operator == "SUM":
        assert result == data.sum()
    else:
        raise NotImplementedError


@pytest.mark.multiprocessing
@pytest.mark.parametrize("operator", ["MAX", "MIN", "SUM"])
def test_allreduce_array(operator, rng):
    """Test MPI allreduce function."""
    data = rng.uniform(size=(size, 3))
    result = mpi_allreduce(data[rank], operator=operator)

    if operator == "MAX":
        np.testing.assert_allclose(result, data.max(axis=0))
    elif operator == "MIN":
        np.testing.assert_allclose(result, data.min(axis=0))
    elif operator == "SUM":
        np.testing.assert_allclose(result, data.sum(axis=0))
    else:
        raise NotImplementedError
