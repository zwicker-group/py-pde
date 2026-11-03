"""Defines the configuration for the numba backend.

This configuration file will be imported without importing the enclosed module (which
will instead be loaded on demand). Consequently, the module should use absolute
references to other modules and not import anything from the backend.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from pde.tools.config import Parameter

# define default parameter values
DEFAULT_CONFIG: dict[str, Parameter] = {
    "debug": Parameter(
        value=False,
        cls=bool,
        description="Determines whether numba uses the debug mode for compilation. If "
        "enabled, this emits extra information that might be useful for debugging.",
    ),
    "fastmath": Parameter(
        value=True,
        cls=bool,
        description="Determines whether the fastmath flag is set during compilation. "
        "If enabled, some mathematical operations might be faster, but less precise. "
        "This flag does not affect infinity detection and NaN handling.",
    ),
    "multithreading": Parameter(
        value="only_local",
        cls=str,
        description="Determines whether multiple threads are used in numba-compiled "
        "code. Enabling this option accelerates a small subset of operators applied to "
        "fields defined on large grids. Possible options are 'never' (disable "
        "multithreading), 'only_local' (disable on HPC hardware), and 'always' (enable "
        "if number of grid points exceeds `multithreading_threshold`)",
    ),
    "multithreading_threshold": Parameter(
        value=256**2,
        cls=int,
        description="Minimal number of support points of grids before multithreading "
        "is enabled in numba compilations. Has no effect when multihreading is "
        "disabled via the `multithreading` option.",
    ),
    "use_spectral": Parameter(
        value=False,
        cls=bool,
        description="Indicates whether a spectral implementation should be used where "
        "possible. Note that this option is only implemented for few operators.",
    ),
}
