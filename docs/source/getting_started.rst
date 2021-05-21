Getting started
===============

This `py-pde` package is developed for python 3.7+ and should run on all
common platforms.
The code is tested under Linux, Windows, and macOS.


Install using pip
^^^^^^^^^^^^^^^^^

The package is available on `pypi <https://pypi.org/project/py-pde/>`_, so you should be
able to install it by running

.. code-block:: bash

    pip install py-pde
    
    

In order to have all features of the package available, you might also want to 
install the following optional packages:

.. code-block:: bash

	pip install h5py pandas pyfftw tqdm

Moreover, :command:`ffmpeg` needs to be installed and for creating movies.    
    

Install using conda
^^^^^^^^^^^^^^^^^^^

The `py-pde` package is also available on `conda <https://conda.io>`_ using the
`conda-forge` channel.
You can thus install it using

.. code-block:: bash

    conda install -c conda-forge py-pde
    
This installation includes all required dependencies to have all features of `py-pde`.


Install from source
^^^^^^^^^^^^^^^^^^^
Installing from source can be necessary if the pypi installation does not work
or if the latest source code should be installed from github.


Required prerequisites
----------------------

The code builds on other python packages, which need to be installed for
`py-pde` to function properly.
The required packages are listed in the table below:


.. csv-table:: 
   :file: _static/requirements_main.csv
   :header-rows: 1


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

.. csv-table:: 
   :file: _static/requirements_optional.csv
   :header-rows: 1

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
:class:`~pde.grids.spherical.PolarSymGrid`,
:class:`~pde.grids.spherical.SphericalSymGrid`, and 
:class:`~pde.grids.cylindrical.CylindricalSymGrid`,
with and without periodic boundaries where applicable.

Fields are defined by specifying values at the grid points using the classes
:class:`~pde.fields.scalar.ScalarField`,
:class:`~pde.fields.vectorial.VectorField`, and
:class:`~pde.fields.tensorial.Tensor2Field`.
These classes provide methods for applying differential operators to the fields, 
e.g., the result of applying the Laplacian to a scalar field is returned by
calling the method :meth:`~pde.fields.scalar.ScalarField.laplace`, which
returns another instance of :class:`~pde.fields.scalar.ScalarField`, whereas
:meth:`~pde.fields.scalar.ScalarField.gradient` returns a
:class:`~pde.fields.vectorial.VectorField`.
Combining these functions with ordinary arithmetics on fields allows to
represent the right hand side of many partial differential equations that appear
in physics.
Importantly, the differential operators work with flexible boundary conditions. 

The PDEs to solve are represented as a separate class inheriting from 
:class:`~pde.pdes.base.PDEBase`.
One example defined in this package is the diffusion equation implemented as
:class:`~pde.pdes.diffusion.DiffusionPDE`, but more specific situations need to
be implemented by the user.
Most notably, PDEs can be specified by their expression using the convenient
:class:`~pde.pdes.pde.PDE` class.

The PDEs are solved using solver classes, where a simple explicit solver is
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

