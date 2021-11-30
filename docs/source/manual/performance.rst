Performance
^^^^^^^^^^^

Measuring performance
"""""""""""""""""""""

The performance of the `py-pde` package depends on many details and general 
statements are thus difficult to make.
However, since the core operators are just-in-time compiled using :mod:`numba`,
many operations of the package proceed at performances close to most compiled
languages.
For instance, a simple Laplace operator applied to fields defined on a Cartesian
grid has performance that is similar to the operators supplied by the popular
`OpenCV <https://opencv.org>`_ package.
The following figures illustrate this by showing the duration of evaluating the
Laplacian on grids of increasing number of support points for
two different boundary conditions (lower duration is better):


.. image:: /_images/performance_periodic.*
   :width: 49%

.. image:: /_images/performance_noflux.*
   :width: 49%
   
   
Note that the call overhead is lower in the `py-pde` package, so that the
performance on small grids is particularly good.
However, realistic use-cases probably need more complicated operations and it is
thus always necessary to profile the respective code.
This can be done using the function
:func:`~pde.tools.misc.estimate_computation_speed` or the traditional
:mod:`timeit`, :mod:`profile`, or even more sophisticated profilers like
`pyinstrument <https://github.com/joerick/pyinstrument>`_.


Improving performance
"""""""""""""""""""""

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