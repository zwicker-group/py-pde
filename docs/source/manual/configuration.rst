.. _configuration:

Configuration parameters
""""""""""""""""""""""""

Configuration parameters affect how the package behaves.
Parameters can generally be set using a dictionary-like interface in either the global
configuration :data:`~pde.config` or in backend-specific settings at
:data:`~pde.backends.base.BackendBase.config`.
To provide flexibility, the default backends (whose name is identical to the backend 
package, e.g. `numba`, `jax`, or `torch`) have a special behavior such that their
configuration is linked with the global configuration. Consequently, the following
commands all yield identical results:

.. code-block:: python

    from pde import config, get_backend

    get_backend("numba").config["debug"] = True
    config["backend"]["numba"]["debug"] = True
    config["backend.numba.debug"] = True

In contrast, custom backends (example: :code:`get_backend("torch:cuda")`) inherit the
global configuration, but modifying their configuration would not influence the global
configuration.

Here is a list of all global configuration options, which includes the backend-specific
parameters

.. package_configuration ::


.. tip::

    To disable parallel computing via multithreading in the package, the following code
    could be added to the start of the script:


    .. code-block:: python

        from pde import config
        config["backend.numba.multithreading"] = "never"

        # actual code using py-pde

    The default value `only_local` already disables multithreading when the package
    detects it is run in an HPC environment.