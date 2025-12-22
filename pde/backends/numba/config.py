"""Defines the configuration for the numba backend.

This configuration file will be imported without importing the enclosed module (which
will instead be loaded on demand). Consequently, the module should use absolute
references to other modules and not import anything from the backend.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from pde.tools.config import Parameter

# define default parameter values
DEFAULT_CONFIG: list[Parameter] = [
    Parameter(
        "debug",
        False,
        bool,
        "Determines whether numba uses the debug mode for compilation. If enabled, "
        "this emits extra information that might be useful for debugging.",
    ),
    Parameter(
        "fastmath",
        True,
        bool,
        "Determines whether the fastmath flag is set during compilation. If enabled, "
        "some mathematical operations might be faster, but less precise. This flag "
        "does not affect infinity detection and NaN handling.",
    ),
    Parameter(
        "multithreading",
        "only_local",
        str,
        "Determines whether multiple threads are used in numba-compiled code. Enabling "
        "this option accelerates a small subset of operators applied to fields defined "
        "on large grids. Possible options are 'never' (disable multithreading), "
        "'only_local' (disable on HPC hardware), and 'always' (enable if number of "
        "grid points exceeds `numba.multithreading_threshold`)",
    ),
    Parameter(
        "multithreading_threshold",
        256**2,
        int,
        "Minimal number of support points of grids before multithreading is enabled in "
        "numba compilations. Has no effect when `numba.multithreading` is `False`.",
    ),
    Parameter(
        "use_spectral",
        False,
        bool,
        "Indicates whether a spectral implementation should be used where possible. "
        "Note that this option is only implemented for few operators.",
    ),
]
