Overview
^^^^^^^^

The main aim of the :mod:`pde` package is to simulate partial
differential equations in simple geometries.
We here focus on the finite difference method, where fields are represented on
discretized grids.
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
and :class:`~pde.trackers.trackers.LengthScaleTracker` (calculating
typical length scales of the state over time).
Others can be found in the :mod:`~pde.trackers.trackers` module.
Moreover, we provide :class:`~pde.storage.memory.MemoryStorage` and
:class:`~pde.storage.file.FileStorage`, which can be used as trackers
to store the intermediate state to memory and to a file, respectively. 
