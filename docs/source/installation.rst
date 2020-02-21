Installation
############

This package is developed for python 3.6+ and might thus not work in earlier
python versions. 


Prerequisites
^^^^^^^^^^^^^

The code builds on other python packages, which need to be installed for this
package to function properly. The required packages are listed in the table below:

===========  =========
Package      Usage 
===========  =========
matplotlib   Visualizing results (version >= 3.1.0)
numpy        Array library used for manipulating data (version >=1.16)
numba        Just-in-time compilation to accelerate numerics (version >=0.43)
scipy        Miscellaneous scientific functions (version >=1.2)
sympy        Dealing with user-defined mathematical expressions (version >=1.4)
===========  =========

These package can be installed via your operating systems package manage, via
:command:`macports`, :command:`homebrew`, :command:`conda`, or :command:`pip`.
The package versions given in the brackets are minimal requirements, although
this is not tested systematically. Generally, it should help to install the
latest version of the package.  
Note that the last package is only available via github and just needs to be
checked out into a folder included in your :envvar:`PYTHONPATH`.


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


Installing the `py-pde` package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The package can be simply checked out from
`github.com/zwicker-group/py-pde <https://github.com/zwicker-group/py-pde>`_.
To import the package from any python session, it might be convenient to include the
root folder of the package into the :envvar:`PYTHONPATH` environment variable.

This documentation can be built by calling the :command:`make html` in the
:file:`docs` folder.
The final documentation will be available in :file:`docs/build/html`.
Note that a LaTeX documentation can be build using :command:`make latexpdf`.



Optimizing performance
^^^^^^^^^^^^^^^^^^^^^^

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
