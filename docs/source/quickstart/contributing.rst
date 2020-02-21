Contributing code
^^^^^^^^^^^^^^^^^


Structure of the package
""""""""""""""""""""""""
The functionality of the :mod:`pde` package is split into multiple sub-package.
The domain, together with its symmetries, periodicities, and discretizations, is
described by classes defined in :mod:`~pde.grids`.
Discretized fields are represented by classes in :mod:`~pde.fields`, which have
methods for differential operators with various boundary conditions collected
in :mod:`~pde.grids.boundaries`.
The actual pdes are collected in :mod:`~pde.pdes` and the respective solvers
are defined in :mod:`~pde.solvers`.


Extending functionality
"""""""""""""""""""""""
All code is build on a modular basis, making it easy to introduce new classes
that integrate with the rest of the package. For instance, it is simple to
define a new partial differential equation by subclassing
:class:`~pde.pdes.base.PDEBase`.
Similarly, new grids can be introduced by subclassing
:class:`~pde.grids.base.GridBase`.
It is also possible to only use parts of the package, e.g., the discretized
differential operators from :mod:`~pde.grids.operators`.


Design choices
""""""""""""""
The data layout of field classes (subclasses of
:class:`~pde.fields.base.FieldBase`) was chosen to allow for a simple
decomposition of different fields and tensor components. Consequently, the data
is laid out in memory such that spatial indices are last. For instance, the data
of a vector field ``field`` defined on a 2d Cartesian grid will have three
dimensions and can be accessed as ``field.data[vector_component, x, y]``,
where ``vector_component`` is either 0 or 1.


Coding style
""""""""""""
In terms of coding style, we try to adhere to `PEP8
<https://www.python.org/dev/peps/pep-0008/>`_ and use `Google Style docstrings
<https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings>`_.
The docstring convention might be best `learned by example
<https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_.
The documentation, including the docstrings, are written using `reStructuredText
<https://de.wikipedia.org/wiki/ReStructuredText>`_, with examples in the
following `cheatsheet
<https://github.com/ralsina/rst-cheatsheet/blob/master/rst-cheatsheet.rst>`_.
To ensure the integrity of the code, we also try to provide many test functions,
which are typically contained in separate modules in sub-packages called
:mod:`tests`.
These tests can be ran using scripts in the :file:`tests` subfolder in the root
folder.
This folder also contain a script :file:`tests_types.sh`, which uses :mod:`mypy`
to check the consistency of the python type annotations.
We use these type annotations for additional documentation and they have also
already been useful for finding some bugs.

We also have some conventions that should make the package more consistent and
thus easier to use. For instance, we try to use ``properties`` instead of getter
and setter methods as often as possible.
Because we use a lot of :mod:`numba` just-in-time compilation to speed up computations,
we need to pass around (compiled) functions regularly. The names of the methods
and functions that make such functions, i.e. that return Callables, should start
with 'make_*' where the wildcard should describe the purpose of the function
being created. 


Running unit tests
""""""""""""""""""
The :mod:`pde` package contains several unit tests, typically contained in 
sub-module :mod:`tests` in the folder of a given module. These tests ensure that
basic functions work as expected, in particular when code is changed in future
versions. To run all tests, there are a few convenience scripts in the root
directory :file:`tests`. The most basic script is :file:`tests_run.sh`, which
uses :mod:`pytest` to run the tests in the sub-modules of the :mod:`pde`
package. Clearly, the python package :mod:`pytest` needs to be installed. There
are also additional scripts that for instance run tests in parallel (need the
python package :mod:`pytest-xdist` installed), measure test coverage (need
package :mod:`pytest-cov` installed), and make simple performance measurements.
Moreover, there is a script :file:`test_types.sh`, which uses :mod:`mypy` to
check the consistency of the python type annotations and there is a script
:file:`codestyle.sh`, which checks the coding style.

Before committing a change to the code repository, it is good practice to run
the tests, check the type annotations, and the coding style with the scripts
described above.

