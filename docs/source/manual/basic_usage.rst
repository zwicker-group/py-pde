Basic usage
^^^^^^^^^^^
We here describe the typical workflow to solve a PDE using `py-pde`.
Throughout this section, we assume that the package has been imported using
:code:`import pde`.


Defining the geometry
"""""""""""""""""""""
The state of the system is described in a discretized geometry, also known as a `grid`.
The package focuses on simple geometries, which work well for the employed finite
difference scheme.
Grids are defined by instance of various classes that capture the symmetries of the 
underlying space.
In particular, the package offers Cartesian grids of `1` to `3` dimensions via
:class:`~pde.grids.cartesian.CartesianGrid`, as well as curvilinear coordinate for
spherically symmetric systems in two dimension (:class:`~pde.grids.spherical.PolarSymGrid`)
and three dimensions (:class:`~pde.grids.spherical.SphericalSymGrid`), as well as the
special class :class:`~pde.grids.cylindrical.CylindricalSymGrid` for a cylindrical geometry
which is symmetric in the angle. 

All grids allow to set the size of the underlying geometry and the number of support
points along each axis, which determines the spatial resolution.
Moreover, most grids support periodic boundary conditions.
For example, a rectangular grid with one periodic boundary condition can be specified
as 

.. code-block:: python

    grid = pde.CartesianGrid([[0, 10], [0, 5]], [20, 10], periodic=[True, False])

This grid will have a rectangular shape of 10x5 with square unit cells of side length
`0.5`.
Note that the grid will only be periodic in the `x`-direction.


Initializing a field
""""""""""""""""""""
Fields specifying the values at the discrete points of the grid defined in the previous
section.
Most PDEs discussed in the package describe a scalar variable, which can be encoded th
class :class:`~pde.fields.scalar.ScalarField`.
However, tensors with rank 1 (vectors) and rank 2 are also supported using 
:class:`~pde.fields.vectorial.VectorField` and
:class:`~pde.fields.tensorial.Tensor2Field`, respectively.
In any case, a field is initialized using a pre-defined grid, e.g.,
:code:`field = pde.ScalarField(grid)`.
Optional values allow to set the value of the grid, as well as a label that is later
used in plotting, e.g., :code:`field1 = pde.ScalarField(grid, data=1, label="Ones")`.
Moreover, fields can be initialized randomly
(:code:`field2 = pde.ScalarField.random_normal(grid, mean=0.5)`) or from a mathematical
expression, which may depend on the coordinates of the grid
(:code:`field3 = pde.ScalarField.from_expression(grid, "x * y")`). 

All field classes support basic arithmetic operations and can be used much like
numpy arrays.
Moreover, they have methods for applying differential operators, 
e.g., the result of applying the Laplacian to a scalar field is returned by
calling the method :meth:`~pde.fields.scalar.ScalarField.laplace`, which
returns another instance of :class:`~pde.fields.scalar.ScalarField`, whereas
:meth:`~pde.fields.scalar.ScalarField.gradient` returns a
:class:`~pde.fields.vector.VectorField`.
Combining these functions with ordinary arithmetics on fields allows to
represent the right hand side of many partial differential equations that appear
in physics.
Importantly, the differential operators work with flexible boundary conditions. 


Specifying the PDE
""""""""""""""""""
PDEs are also instances of special classes and a number of classical PDEs are already
pre-defined in the module :mod:`pde.pdes`.
Moreover, the special class :class:`~pde.pdes.pde.PDE` allows defining PDEs by simply
specifying the expression on their right hand side.
To see how this works in practice, let us consider the `Kuramoto–Sivashinsky equation 
<https://en.wikipedia.org/wiki/Kuramoto–Sivashinsky_equation>`_,
:math:`\partial_t u = - \nabla^4 u - \nabla^2 u - \frac{1}{2} |\nabla u|^2`,
which describes the time evolution of a scalar field :math:`u`.
A simple implementation of this equation reads 

.. code-block:: python

    eq = pde.PDE({"u": "-gradient_squared(u) / 2 - laplace(u + laplace(u))"})

Here, the argument defines the evolution rate for all fields (in this case
only :math:`u`).
The expression on the right hand side can contain typical mathematical functions
and the operators defined by the package.


Running the simulation
""""""""""""""""""""""
To solve the PDE, we first need to generate an initial condition, i.e., the initial
values of the fields that are evolved forward in time by the PDE.
This field also defined the geometry on which the PDE is solved.
In the simplest case, the solution is then obtain by running

 .. code-block:: python

    result = eq.solve(field, t_range=10, dt=1e-2)

Here, `t_range` specifies the duration over which the PDE is considered and `dt`
specifies the time step.
The `result` field will be defined on the same grid as the initial condition `field`,
but instead contain the data value at the final time.
Note that all intermediate states are discarded in the simulation above and no
information about the dynamical evolution is retained.
To study the dynamics, one can either analyze the evolution on the fly or store its
state for subsequent analysis.
Both these tasks are achieved using :mod:`~pde.trackers`, which analyze the simulation
periodically.
For instance, to store the state for some time points in memory, one uses  
 
 .. code-block:: python

    storage = pde.MemoryStorage()
    result = eq.solve(field, t_range=10, dt=1e-3, tracker=["progress", storage.tracker(1)])

Note that we also included the special identifier :code:`"progress"` in the list of
trackers, which shows a progress bar during the simulation.
Another useful tracker is :code:`"plot"` which displays the state on the fly.


Analyzing the results
"""""""""""""""""""""
Sometimes is suffices to plot the final result, which can be done using
:code:`result.plot()`.
The final result can of course also be analyzed quantitatively, e.g., using
:attr:`result.average` to obtain its mean value.
If the intermediate states have been saved as indicated above, they can be analyzed
subsequently:

.. code-block:: python

    for time, field in storage.items():
        print(f"t={time}, field={field.magnitude}")

Moreover, a movie of the simulation can be created using
:code:`pde.movie(storage, filename=FILE)`, where `FILE` determines where the movie is
written.
