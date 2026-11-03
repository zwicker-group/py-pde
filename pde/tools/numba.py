"""Helper functions for just-in-time compilation with numba.

.. autosummary::
   :nosignatures:

   numba_environment
   jit
   make_array_constructor
   numba_dict
   get_common_numba_dtype
   random_seed

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import warnings

# module deprecated since 2025-12-12
warnings.warn(
    "Module `pde.tools.numba` has been moved to `pde.backends.numba.utils`",
    DeprecationWarning,
    stacklevel=2,
)

from ..backends.numba.utils import (  # noqa: F401
    get_common_numba_dtype,
    jit,
    make_array_constructor,
    numba_dict,
    numba_environment,
    random_seed,
)
