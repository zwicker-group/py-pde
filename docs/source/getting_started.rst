Getting started
===============

Install from pip
^^^^^^^^^^^^^^^^

This `py-pde` package is developed for python 3.6+ and should run on all
common platforms.
The code is tested under Linux, Windows, and macOS.
Since the package is available on `pypi <https://pypi.org/project/py-pde/>`_,
the installation is in principle as simple as running

.. code-block:: bash

    pip install py-pde
    
    

In order to have all features of the package available, you might also want to 
install the following optional packages:

.. code-block:: bash

	pip install h5py pandas pyfftw tqdm

Moreover, :command:`ffmpeg` needs to be installed and for creating movies.    
    

Install from source
^^^^^^^^^^^^^^^^^^^
Installing from source can be necessary if the pypi installation does not work
or if the latest source code should be installed from github.


Required prerequisites
----------------------

The code builds on other python packages, which need to be installed for
`py-pde` to function properly.
The required packages are listed in the table below:

===========  ========= =========
Package      Version   Usage 
===========  ========= =========
matplotlib   >= 3.1.0  Visualizing results
numpy        >=1.16    Array library used for storing data
numba        >=0.43    Just-in-time compilation to accelerate numerics
scipy        >=1.2     Miscellaneous scientific functions
sympy        >=1.4     Dealing with user-defined mathematical expressions
===========  ========= =========

The simplest way to install these packages is to use the
:file:`requirements.txt` in the base folder:

.. code-block:: bash

    pip install -r requirements.txt
    

Alternatively, these package can be installed via your operating system's
package manager, e.g. using :command:`macports`, :command:`homebrew`, or
:command:`conda`.
The package versions given above are minimal requirements, although
this is not tested systematically. Generally, it should help to install the
latest version of the package.


Optional packages
-----------------

The following packages should be installed to use some miscellaneous features:

===========  =========
Package      Usage                                      
===========  =========
h5py         Storing data in the hierarchical file format
pandas       Handling tabular data
pyfftw       Faster Fourier transforms
tqdm         Display progress bars during calculations
===========  =========

For making movies, the :command:`ffmpeg` should be available.
Additional packages might be required for running the tests in the folder
:file:`tests` and to build the documentation in the folder :file:`docs`.
These packages are listed in the files :file:`requirements.txt` in the
respective folders.


Downloading `py-pde`
--------------------

The package can be simply checked out from
`github.com/zwicker-group/py-pde <https://github.com/zwicker-group/py-pde>`_.
To import the package from any python session, it might be convenient to include
the root folder of the package into the :envvar:`PYTHONPATH` environment variable.

This documentation can be built by calling the :command:`make html` in the
:file:`docs` folder.
The final documentation will be available in :file:`docs/build/html`.
Note that a LaTeX documentation can be build using :command:`make latexpdf`.

	
	
Package overview
^^^^^^^^^^^^^^^^

The main aim of the :mod:`pde` package is to simulate partial differential
equations in simple geometries.
Here, the time evolution of a PDE is determined using the method of lines by
explicitly discretizing space using fixed grids.
The differential operators are implemented using the `finite difference method
<https://en.wikipedia.org/wiki/Finite_difference_method>`_.
For simplicity, we consider only regular, orthogonal grids, where each axis has
a uniform discretization and all axes are (locally) orthogonal.
Currently, we support simulations on  
:class:`~pde.grids.cartesian.CartesianGrid`,
:class:`~pde.grids.spherical.PolarGrid`,
:class:`~pde.grids.spherical.SphericalGrid`, and 
:class:`~pde.grids.cylindrical.CylindricalGrid`,
with and without periodic boundaries where applicable.

Fields are defined by specifying values at the grid points using the classes
:class:`~pde.fields.scalar.ScalarField`,
:class:`~pde.fields.vectorial.VectorField`, and
:class:`~pde.fields.tensorial.Tensor2Field`.
These classes provide methods for applying differential operators to the fields, 
e.g., the result of applying the Laplacian to a scalar field is returned by
calling the method :meth:`~pde.fields.scalar.ScalarField.laplace`, which
returns another instance of :class:`~pde.fields.scalar.ScalarField`, whereas
:meth:`~pde.fields.scalar.ScalarField.gradient` returns a `VectorField`.
Combining these functions with ordinary arithmetics on fields allows to
represent the right hand side of many partial differential equations that appear
in physics.
Importantly, the differential operators work with flexible boundary conditions. 

The pde to solve are represented as a separate class inheriting from 
:class:`~pde.pdes.base.PDEBase`.
One example defined in this package is the diffusion equation implemented as
:class:`~pde.pdes.diffusion.DiffusionPDE`, but more specific situations need to
be implemented by the user.

The pdes are solved using solver classes, where a simple explicit solver is
implemented by :class:`~pde.solvers.explicit.ExplicitSolver`, but more advanced
implementations can be done. 
To obtain more details during the simulation, trackers can be attached to the
solver instance, which analyze intermediate states periodically. Typical
trackers include
:class:`~pde.trackers.trackers.ProgressTracker` (display simulation progress),
:class:`~pde.trackers.trackers.PlotTracker` (display images of the simulation),
and :class:`~pde.trackers.trackers.SteadyStateTracker` (aborting simulation when
a stationary state is reached).
Others can be found in the :mod:`~pde.trackers.trackers` module.
Moreover, we provide :class:`~pde.storage.memory.MemoryStorage` and
:class:`~pde.storage.file.FileStorage`, which can be used as trackers
to store the intermediate state to memory and to a file, respectively. 

