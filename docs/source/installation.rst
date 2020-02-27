Installation
############

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
    

Installing from source
^^^^^^^^^^^^^^^^^^^^^^
Installing from source can be necessary if the pypi installation does not work
or if the latest source code should be installed from github.


Prerequisites
-------------

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

These package can be installed via your operating system's package manager, e.g.
using :command:`macports`, :command:`homebrew`, :command:`conda`, or
:command:`pip`.
The package versions given above are minimal requirements, although
this is not tested systematically. Generally, it should help to install the
latest version of the package.  

Optionally, the following packages should be installed to use some miscellaneous
features:

===========  =========
Package      Usage                                      
===========  =========
h5py         Storing data in the hierarchical file format
pandas       Handling tabular data
pyfftw       Faster Fourier transforms
pytest       Running tests
sphinx       Building the documentation
tqdm         Display progress bars during calculations
===========  =========

Additionally, :command:`ffmpeg` should be installed for making movies and the
packages :mod:`sphinx-autodoc-annotation` and :mod:`sphinx_rtd_theme` are
required for building the documentation.


Downloading the package
-----------------------

The package can be simply checked out from
`github.com/zwicker-group/py-pde <https://github.com/zwicker-group/py-pde>`_.
To import the package from any python session, it might be convenient to include
the root folder of the package into the :envvar:`PYTHONPATH` environment variable.

This documentation can be built by calling the :command:`make html` in the
:file:`docs` folder.
The final documentation will be available in :file:`docs/build/html`.
Note that a LaTeX documentation can be build using :command:`make latexpdf`.



Optimizing performance
-----------------------

Factors influencing the performance of the package include the compiler used for
:mod:`numpy`, :mod:`scipy`, and of course :mod:`numba`.
Moreover, the BLAS and LAPACK libraries might make a difference.
The package has some basic support for multithreading, which can be accelerated
using the `Threading Building Blocks` library.
Finally, it can help to install the intel short vector math library (SVML).
However, this is not distributed with :command:`macports` and might thus be more
difficult to enable. 

Using :command:`macports`, one could for instance install the following variants
of typical packages

.. code-block:: bash

	port install py37-numpy +gcc8+openblas
	port install py37-scipy +gcc8+openblas
	port install py37-numba +tbb
