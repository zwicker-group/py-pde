"""Defines the configuration for the torch backend.

This configuration file will be imported without importing the enclosed module (which
will instead be loaded on demand). Consequently, the module should use absolute
references to other modules and not import anything from the backend.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from pde.tools.config import Parameter

# define default parameter values
DEFAULT_CONFIG: list[Parameter] = [
    Parameter(
        "device",
        "cpu",
        str,
        "Determines the torch device that is used for the torch backend. Common "
        "options include `cpu`, `cuda`, and more specific choices, like `cuda:0`. The "
        "special value `auto` chooses `cuda` if it is available, and falls back to "
        "`cpu` if not.",
    ),
    Parameter(
        "dtype_downcasting",
        True,
        bool,
        "Determines whether dtype downcasting is used automatically. A typical example "
        "are torch devices that only support float32, so the numpy arrays using float64"
        " need to be converted. If enabled, this happens automatically.",
    ),
    Parameter(
        "compile",
        True,
        bool,
        "Enables compilation in the `torch` backend.",
    ),
]
