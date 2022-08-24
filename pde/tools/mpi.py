"""
Auxillary functions and variables for dealing with MPI multiprocessing

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import os
import sys

# read state of the current MPI node
try:
    import numba_mpi

except ImportError:
    # package `numba_mpi` could not be loaded
    if os.environ.get("PMI_SIZE", 1) > 1:
        # environment variable indicates that we are in a parallel program
        sys.exit(
            "WARNING: Detected multiprocessing run, but could not load `numba_mpi`"
        )

    # assume that we run serial code if `numba_mpi` is not available
    size = 1
    rank = 0

else:
    # we have access to MPI
    size = numba_mpi.size()
    rank = numba_mpi.rank()

# set flag indicating whether the current process is the main process
is_main = rank == 0