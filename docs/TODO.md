TODO
====
* Change default plotting such that axes are reused (using `plt.gca()`) instead of created by default
* Think about incremental test run
    - should break at first failure
    - should restart at last failure
* Think about a way to save information about how the right hand side of the PDE
    - this should be forwarded to solver to store with the diagnostics
    - this could for instance store whether a staggered grid was used
* Indicate periodic boundary by dashed/dotted axes in image plot?
    - use ax.spines['top'].set_linestyle((0, (5, 10)))
* Improve interactive plotting:
    - allow displaying time somewhere (statusbar or extra widget)
    - Improve this display by adding a progress bar and support displaying extra text
    - allow displaying data from storage in n-d manner
* Plot tracker:
    - plot final state in the live view (finalize does not get the final state yet)
    - we could for instance have a flag on trackers, whether they are being handled a final time
    - an alternative would be to pass the final state to the `finalize` method 
* Think about logger names (add `pde.` before class name)
* Hide attributes in field classes that should not be overwritten
    - clarify in the description of grids and fields what fields are mutable
* Extend methods `get_image_data` to allow different cuts, visualizations
  - use an interface similar to that of `get_line_data`
  - mimic this interface for plotting 3d droplets?


LOW-PRIORITY (Future ideas)
===========================
* Think about hyperbolic equations:
    - Introducing "advection" operator that could either implement really simple
      Gudunov finite volume scheme or upwind finite difference scheme
    - Introduce gradient operator for given direction:
        https://en.wikipedia.org/wiki/Lax–Wendroff_method (Richtmyer or MacCormack)
* Think about implementing Helmholtz solver
    - generally useful to discuss eigenvalues of laplace operator?
* Consider using @numba.overload decorator instead of generated jit to support
	out=None idiom
* Ensure that stochastic simulations on a single core can be resumed from any
	stored state (this requires storing random seeds)
	- we now have support for random state in numpy implementations
	- it's not clear how to support numba, since the random state is not as accessible
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