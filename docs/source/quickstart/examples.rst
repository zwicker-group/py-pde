Examples
^^^^^^^^

We here collect examples for using the package to demonstrate some of its
functionality. 


Basic simulation
""""""""""""""""


Basic simulations can be run as follows

.. include:: ../examples/simple.rst

The phase field at the final time step will then be contained in `result`, which
is an instance of :class:`~pde.fields.scalar.ScalarField`.
The result can for instance be visualized using :meth:`result.plot`, or the
discretized data can be accessed via :attr:`result.data`.


Tracking and storing results
""""""""""""""""""""""""""""

To show output during the simulation and also store the full time course of the
simulation, e.g. for later analysis, the package offers tracker classes,
which can be used like so 

.. include:: ../examples/trackers.rst

The time steps at which data was written is then available as
:attr:`storage.times` an the actual data is stored at :attr:`storage.data` for
each time step.


Imposing boundary conditions
""""""""""""""""""""""""""""

The default :class:`~pde.pdes.diffusion.DiffusionPDE` imposed vanishing
derivatives on the scalar field.
The boundary conditions can be altered when creating the instance like so

.. include:: ../examples/boundary_conditions.rst

Here, we consider a grid that is periodic in the `y`-direction and thus have
to set the associated boundary condition `bc_y` also to `periodic`. Since the 
grid is not periodic in the `x`-direction, we have more freedom for specifying
boundary conditions. In particular, we can choose different boundary conditions
for the two sides. In the particular example we impose that the gradient of the
field is `0.1` on the left side, while its value is fixed to `0` on
the right side. 


Make a movie
""""""""""""
The following example shows how to make a movie from a simulation:

.. include:: ../examples/make_movie.rst