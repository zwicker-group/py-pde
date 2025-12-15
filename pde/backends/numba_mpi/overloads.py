"""Defines functions overloads, so numba can use them.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import numba as nb
import numpy as np
from numba import types
from numba.extending import SentryLiteralArgs, overload, register_jitable

from ...tools.mpi import Operator, mpi_allreduce, mpi_recv, mpi_send, size

try:
    from numba.types import Literal
except ImportError:
    from numba.types.misc import Literal


@overload(mpi_send)
def ol_mpi_send(data, dest: int, tag: int):
    """Overload the `mpi_send` function."""
    import numba_mpi

    def impl(data, dest: int, tag: int) -> None:
        """Reduce a single number across all cores."""
        status = numba_mpi.send(data, dest, tag)
        assert status == 0

    return impl


@overload(mpi_recv)
def ol_mpi_recv(data, source: int, tag: int):
    """Overload the `mpi_recv` function."""
    import numba_mpi

    def impl(data, source: int, tag: int) -> None:
        """Receive data from another MPI node.

        Args:
            data: A buffer into which the received data is written
            dest (int): The ID of the sending node
            tag (int): A numeric tag identifying the message
        """
        status = numba_mpi.recv(data, source, tag)
        assert status == 0

    return impl


@overload(mpi_allreduce)
def ol_mpi_allreduce(data, operator):
    """Overload the `mpi_allreduce` function."""
    if size == 1:
        # We can simply return the value in a serial run

        def impl(data, operator):
            return data

        return impl

    # Conversely, in a parallel run, we need to use the correct reduction. Let's first
    # determine the operator, which must be given as a literal type
    SentryLiteralArgs(["operator"]).for_function(ol_mpi_allreduce).bind(data, operator)
    if isinstance(operator, Literal):
        # an operator is specified (using a literal value)
        if isinstance(operator.literal_value, str):
            # an operator is specified by it's name
            op_id = Operator.id(operator.literal_value)
        else:
            # assume an operator is specified by it's id
            op_id = int(operator.literal_value)
    elif isinstance(operator, nb.types.Integer):
        op_id = None  # use given value of operator
    else:
        msg = f"`operator` must be a literal type, not {operator}"
        raise TypeError(msg)

    import numba_mpi

    @register_jitable
    def _allreduce(sendobj, recvobj, operator) -> int:
        """Helper function that calls `numba_mpi.allreduce`"""
        if op_id is None:
            return numba_mpi.allreduce(sendobj, recvobj, operator)  # type: ignore
        return numba_mpi.allreduce(sendobj, recvobj, op_id)  # type: ignore

    if isinstance(data, types.Number):
        # implementation of the reduction for a single number

        def impl(data, operator):
            """Reduce a single number across all cores."""
            sendobj = np.array([data])
            recvobj = np.empty((1,), sendobj.dtype)
            status = _allreduce(sendobj, recvobj, operator)
            if status != 0:
                raise RuntimeError
            return recvobj[0]

    elif isinstance(data, types.Array):
        # implementation of the reduction for a numpy array

        def impl(data, operator):
            """Reduce an array across all cores."""
            recvobj = np.empty(data.shape, data.dtype)
            status = _allreduce(data, recvobj, operator)
            if status != 0:
                raise RuntimeError
            return recvobj

    else:
        msg = f"Unsupported type {data.__class__.__name__}"
        raise TypeError(msg)

    return impl
