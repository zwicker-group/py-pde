"""Defines the configuration for the torch backend.

This configuration file will be imported without importing the enclosed module (which
will instead be loaded on demand). Consequently, the module should use absolute
references to other modules and not import anything from the backend.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from pde.tools.config import Parameter

# define default parameter values
DEFAULT_CONFIG: list[Parameter] = []
