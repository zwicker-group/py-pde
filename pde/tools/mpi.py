"""Auxiliary functions and variables for dealing with MPI multiprocessing.

Warning:
    These functions are mostly no-ops unless MPI is properly installed and python code
    was started using :code:`mpirun` or :code:`mpiexec`. Please refer to the
    documentation of your MPI distribution for details.

.. autosummary::
   :nosignatures:

   mpi_send
   mpi_recv
   mpi_allreduce

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numba_mpi import Operator

# Initialize assuming that we run serial code if `numba_mpi` is not available
initialized: bool = False
"""bool: Flag determining whether mpi was initialized (and is available)"""

size: int = 1
"""int: Total process count"""

rank: int = 0
"""int: ID of the current process"""

# read state of the current MPI node
try:
    from mpi4py import MPI
except ImportError:
    # package `numba_mpi` could not be loaded
    if int(os.environ.get("PMI_SIZE", "1")) > 1:
        # environment variable indicates that we are in a parallel program
        sys.exit(
            "WARNING: Detected multiprocessing run, but could not import python "
            "package `numba_mpi`"
        )
else:
    initialized = MPI.Is_initialized()
    size = MPI.COMM_WORLD.size
    rank = MPI.COMM_WORLD.rank

    class _OperatorRegistry:
        """Collection of operators that MPI supports."""

        _name_ids: dict[str, int]
        _ids_operators: dict[int, MPI.Op]

        def __init__(self):
            self._name_ids = {}
            self._ids_operators = {}

        def register(self, name: str, op: MPI.Op):
            op_id = int(MPI._addressof(op))
            self._name_ids[name] = op_id
            self._ids_operators[op_id] = op

        def id(self, name_or_id: int | str) -> int:
            if isinstance(name_or_id, int):
                return name_or_id
            return self._name_ids[name_or_id]

        def operator(self, name_or_id: int | str) -> MPI.Op:
            if isinstance(name_or_id, str):
                name_or_id = self._name_ids[name_or_id]
            return self._ids_operators[name_or_id]

        def __getattr__(self, name: str):
            try:
                return self._name_ids[name]
            except KeyError:
                msg = f"MPI operator `{name}` not registered"
                raise AttributeError(msg) from None

    Operator = _OperatorRegistry()
    Operator.register("MAX", MPI.MAX)
    Operator.register("MIN", MPI.MIN)
    Operator.register("SUM", MPI.SUM)

parallel_run: bool = size > 1
"""bool: Flag indicating whether the current run is using multiprocessing"""

is_main: bool = rank == 0
"""bool: Flag indicating whether the current process is the main process (with ID 0)"""


def mpi_send(data, dest: int, tag: int) -> None:
    """Send data to another MPI node.

    Args:
        data: The data being send
        dest (int): The ID of the receiving node
        tag (int): A numeric tag identifying the message
    """
    MPI.COMM_WORLD.send(data, dest=dest, tag=tag)


def mpi_recv(data, source, tag) -> None:
    """Receive data from another MPI node.

    Args:
        data: A buffer into which the received data is written
        dest (int): The ID of the sending node
        tag (int): A numeric tag identifying the message
    """
    data[...] = MPI.COMM_WORLD.recv(source=source, tag=tag)


def mpi_allreduce(data, operator):
    """Combines data from all MPI nodes.

    Note that complex datatypes and user-defined reduction operators are not properly
    supported in numba-compiled cases.

    Args:
        data:
            Data being send from this node to all others
        operator:
            The operator used to combine all data. Possible options are summarized in
            the IntEnum :class:`numba_mpi.Operator`.

    Returns:
        The accumulated data
    """
    if not parallel_run:
        # in a serial run, we can always return the value as is
        return data

    if isinstance(data, np.ndarray):
        # synchronize an array
        out = np.empty_like(data)
        MPI.COMM_WORLD.Allreduce(data, out, op=Operator.operator(operator))
        return out

    # synchronize a single value
    return MPI.COMM_WORLD.allreduce(data, op=Operator.operator(operator))
