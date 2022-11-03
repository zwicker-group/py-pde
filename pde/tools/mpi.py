"""
Auxillary functions and variables for dealing with MPI multiprocessing


.. autosummary::
   :nosignatures:

   mpi_send
   mpi_recv
   mpi_allreduce

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import os
import sys
from numbers import Number
from typing import TYPE_CHECKING, Union

import numba as nb
import numpy as np
from numba.core import types
from numba.extending import register_jitable

from .numba import jit

if TYPE_CHECKING:
    from numba_mpi import Operator  # @UnusedImport

# Initialize assuming that we run serial code if `numba_mpi` is not available
initialized: bool = False
"""bool: Flag determining whether mpi was initialized (and is available)"""

size: int = 1
"""int: Total process count"""

rank: int = 0
"""int: ID of the current process"""

# read state of the current MPI node
try:
    import numba_mpi

except ImportError:
    # package `numba_mpi` could not be loaded
    if int(os.environ.get("PMI_SIZE", "1")) > 1:
        # environment variable indicates that we are in a parallel program
        sys.exit(
            "WARNING: Detected multiprocessing run, but could not load `numba_mpi`"
        )

else:
    # we have access to MPI
    initialized = numba_mpi.initialized()
    size = numba_mpi.size()
    rank = numba_mpi.rank()

parallel_run: bool = size > 1
"""bool: Flag indicating whether the current run is using multiprocessing"""

is_main: bool = rank == 0
"""bool: Flag indicating whether the current process is the main process (with ID 0)"""


@jit
def mpi_send(data, dest: int, tag: int) -> None:
    """send data to another MPI node

    Args:
        data: The data being send
        dest (int): The ID of the receiving node
        tag (int): A numeric tag identifying the message
    """
    status = numba_mpi.send(data, dest, tag)
    assert status == 0


@jit()
def mpi_recv(data, source, tag) -> None:
    """receive data from another MPI node

    Args:
        data: A buffer into which the received data is written
        dest (int): The ID of the sending node
        tag (int): A numeric tag identifying the message

    """
    status = numba_mpi.recv(data, source, tag)
    assert status == 0


@nb.generated_jit(nopython=True)
def mpi_allreduce(data, operator: Union[int, "Operator"] = None):
    """combines data from all MPI nodes

    Note that complex datatypes and user-defined functions are not properly supported.

    Args:
        data:
            Data being send from this node to all others
        operator:
            The operator used to combine all data. Possible options are summarized in
            the IntEnum :class:`numba_mpi.Operator`.

    Returns:
        The accumulated data
    """
    # the following definition of `allreduce` is a workaround that allows using the
    # numba_mpi.allreduce function without jitting. This workaround can be dropped once
    # numba_mpi.allreduce can be called without jitting in the future.
    if nb.config.DISABLE_JIT:

        def allreduce(sendobj, recvobj, operator=None):
            if operator is None:
                impl = numba_mpi.allreduce(sendobj, recvobj)
                return impl(sendobj, recvobj)
            else:
                impl = numba_mpi.allreduce(sendobj, recvobj, operator)
                return impl(sendobj, recvobj, operator)

    else:

        @register_jitable
        def allreduce(sendobj, recvobj, operator=None):
            if operator is None:
                return numba_mpi.allreduce(sendobj, recvobj)
            else:
                return numba_mpi.allreduce(sendobj, recvobj, operator)

    if isinstance(data, (types.Number, Number)):

        def impl(data, operator=None):
            """reduce a single number across all cores"""
            sendobj = np.array([data])
            recvobj = np.empty((1,), sendobj.dtype)
            status = allreduce(sendobj, recvobj, operator)
            assert status == 0
            return recvobj[0]

    elif isinstance(data, (types.Array, np.ndarray)):

        def impl(data, operator=None):
            """reduce an array across all cores"""
            recvobj = np.empty(data.shape, data.dtype)
            status = allreduce(data, recvobj, operator)
            assert status == 0
            return recvobj

    else:
        raise TypeError(f"Unsupported type {data.__class__.__name__}")

    if nb.config.DISABLE_JIT:
        return impl(data, operator)
    else:
        return impl
