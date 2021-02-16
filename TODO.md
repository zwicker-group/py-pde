TODO
====
* Centrally manage requirements and write all files in a requirements.py
    requirements.txt, requirements_min.txt, setup.py, sphinx
* Automatically register derivative with respect to a single axis
    - pattern: d_*(field), e.g., d_x(field)
    - second derivatives, too? d2_*(field)
    - can we support this for all grids or just for Cartesian grids?
    - should also apply to vectors and tensors
    - would need a new method that generates the respective operators
* Allow creating ScalarFields from data points and from a python function 
* Indicate periodic boundary by dashed/dotted axes in image plot?
    - use ax.spines['top'].set_linestyle((0, (5, 10)))
* Add support for dtype=np.single
    - Add general support for dtype in classmethods (e.g. to create random fields)
* Improve interactive plotting:
    - allow displaying time somewhere (statusbar or extra widget)
    - Improve this display by adding a progress bar and support displaying extra text
    - allow displaying data from storage in n-d manner
* Add documentation entry for how to build expressions
    - Improved documentation on how to set boundary conditions
* Implement Gray Scott Model of Reaction Diffusion
* pde.PDE:
    - add test for dot operator and custom boundary conditions
* Plot tracker:
    - plot final state in the live view (finalize does not get the final state yet)
    - we could for instance have a flag on trackers, whether they are being handled a final time
    - an alternative would be to pass the final state to the `finalize` method 
* Think about logger names (add `pde.` before class name)
* Count the number of compilations and store it in the info field of the simulation
    - raise a warning when this number became too large in a simulation?
* Think about interface for changing boundary values in numba
    - We might need to support optional `bc` argument for operators
    - Try using https://cffi.readthedocs.io/en/latest/overview.html#purely-for-performance-api-level-out-of-line 
* Support 3d plots in plot (use for Laplace and Poisson eq) 
* Add Glossary or something to development guide
    - e.g., state = attributes + data
* Improve cell_volume and cell_volume_data of grids to be more useful
* Hide attributes in field classes that should not be overwritten
    - clarify in the description of grids and fields what fields are mutable
* Fix progress bar when starting from non-zero t_start?
* Extend methods `get_image_data` to allow different cuts, visualizations
  - use an interface similar to that of `get_line_data`
  - mimick this interface for plotting 3d droplets?
* Think about implementing vector differential operators more generally based
  on the scalar operators –> could this work for all grids?
* Add conservative Laplace operator for polar and cylindrical grid?
* Add tests:
    - update plotting of fields and field collections
	- general Trackers
	- Different intervals for trackers
	- Interpolating using boundary conditions
* Think about better interface to convert between different coordinate systems:
	- we have global cartesian coordinates, grid coordinates, and cell indices
	- all functions dealing with points or returning points should be able to
	  handle all coordinate types?!
	- what is the best interface?
	- can we just add an option coords='cells', coords='cartesian', coords='grid'
	  to methods that return points (or accept points)
	- there should also be a method `convert_coords(from, to)`



LOW-PRIORITY (Future ideas)
===========================
* Add adaptive Euler stepping
* Think about 2nd order BCs for spherical coordinates
* Think about hyperbolic equations:
    - Introducing "advection" operator that could either implement really simple
      Gudunov finite volume scheme or upwind finite difference scheme
    - Introduce gradient operator for given direction:
        https://en.wikipedia.org/wiki/Lax–Wendroff_method (Richtmyer or MacCormack)
* Think about implementing helmholtz solver
    - generally useful to discuss eigenvalues of laplace operator?
* Add method showing the boundary condition as a mathematical equation
* Consider using @numba.overload decorator instead of generated jit to support
	out=None idiom
* Ensure that stochastic simulations on a single core can be resumed from any
	stored state (this requires storing random seeds)
* Implement multiprocessing:
	- Separate CartesianGrid into different blocks
	  (we need to slice the whole grid at defined locations)
	- Communicate boundary values every time step using special boundary condition
	  (these BCs must follow from the slicing, periodic BCs can also be replaced
	  by the virtual BC)
	- This relies on being able to pass information into BCs  
	- Use mpi4py to pass information around
	- How would this play with numba? (we might need to only use numba for the
	  time stepping and do the rest in python)
	- Also look into ipyparallel, pyop
* Support more flexible boundary conditions
    - Think about implementing vectorial boundary conditions without creating three
    separate compiled functions
    - Add tests for setting vectorial boundary conditions
    - add gradient calculation to performance test 
* Support CUDA/threading using numba?
    - could we partition the calculation of the rhs of PDE and just exchange the
      boundary/interface values between threads?
    - this should be implemented in the Solver class, which needs to prepare
      special grids and duplicate the PDE class
    - This is easiest to implement for Cartesian grids and we might only focus on
      these (the axial direction of Cylindrical grids should also be supported)
* Implement on-the-fly performance testing to decided whether to use parallel=True
    - Store the information for a given grid size in a cache
    - This is similar to the wisdom created by FFTW
* Provide class SymmmetricTensorField, which should be more efficient than the
	full TensorField
* Implement bipolar/bispherical coordinate systems
* Rename SphericalGrid to SphericalSymmetricGrid
	- also CylindricalGrid -> CylindricalSymmetricGrid
	- this allows to use general spherical coordinates later

	
