---
title: 'py-pde: A Python package for solving partial differential equations'
tags:
  - Python
  - partial differential equation
  - dynamical systems
  - finite-difference
  - just-in-time compilation
authors:
  - name: David Zwicker
    orcid: 0000-0002-3909-3334
    affiliation: 1
affiliations:
 - name: Max Planck Institute for Dynamics and Self-Organization, Göttingen, Germany
   index: 1
date: 2 March 2020
bibliography: paper.bib
---

# Summary

Partial differential equations (PDEs) play a central role in describing the
dynamics of physical systems in research and in practical applications.
However, equations appearing in realistic scenarios are typically non-linear and
analytical solutions rarely exist.
Instead, such systems are solved by numerical integration to provide insight
into their behavior.
Moreover, such investigations can motivate approximative solutions, which might
then lead to analytical insight.

The `py-pde` python package presented in this paper allows researchers to
quickly and conveniently simulate and analyze PDEs of the general form
$$
	\partial_t u(\boldsymbol x, t) = \mathcal D[u(\boldsymbol x, t)] 
		+ \eta(u, \boldsymbol x, t) \;,
$$
where $\mathcal D$ is a (non-linear) differential operator that defines
the time evolution of a (set of) physical fields $u$ with possibly
tensorial character, which depend on spatial coordinates $\boldsymbol x$
and time $t$.
The framework also supports stochastic differential equations in the Itô
representation, indicated by the noise term $\eta$ in the equation above.

The main goal of the `py-pde` package is to provide a convenient way to analyze
general PDEs, while at the same time allowing for enough flexibility to easily
implement specialized code for particular cases.
Since the code is written in pure Python, it can be easily installed via pip by
simply calling `pip install py-pde`.
However, the central parts are just-in-time compiled using `numba` [@numba] for 
computational efficiency.
To improve user interaction further, some arguments accept mathematical
expressions that are parsed by `sympy` [@sympy] and are compiled optionally.
This approach lowers the barrier for new users while also providing speed and 
flexibility for advanced use cases.
Moreover, the package provides convenience functions for creating suitable 
initial conditions, for controlling what is analyzed as well as stored during a
simulation, and for visualizing the final results.
The `py-pde` package thus serves as a toolbox for exploring PDEs for researchers
as well as for students who want to delve into the fascinating world of
dynamical systems.


# Methods

The basic objects of the `py-pde` package are scalar and tensorial fields
defined on various discretized grids.
These grids can deal with periodic boundary conditions and they allow exploiting
spatial symmetries that might be present in the physical problem. 
For instance, the scalar field $f(z, r) = \sqrt{z} * e^{-r^2}$ in cylindrical
coordinates assuming azimuthal symmetry can be visualized using
```python
    grid = pde.CylindricalSymGrid(radius=5, bounds_z=[0, 1], shape=(32, 8))
    field = pde.ScalarField.from_expression(grid, 'sqrt(z) * exp(-r**2)')
    field.plot()
```
The package defines common differential operators that act directly on the
fields.
For instance, calling `field.gradient(bc='neumann')` returns a vector field on
the same cylindrical grid where the components correspond to the gradient of
`field` assuming Neumann boundary conditions.
Here, differential operators are evaluated using the finite difference method
(FDM) and the package supports various boundary conditions, which can be
separately specified per field and boundary.
The discretized fields are the foundation of the `py-pde` package and allow 
the comfortable construction of initial conditions, the visualization of final
results, and the detailed investigation of intermediate data.

The main part of the `py-pde` package provides the infrastructure for solving
partial differential equations.
Here, we use the method of lines by explicitly discretizing space using the
grid classes described above.
This reduces the PDEs to a set of ordinary differential equations, which can
be solved using standard methods.
For instance, the diffusion equation $\partial_t u = \nabla^2 u$ on the
cylindrical grid defined above can be solved by
```python
    eq = pde.DiffusionPDE()
    result = eq.solve(field, t_range=[0, 10])
```
Note that the partial differential equation is defined independent of the grid,
allowing use of the same implementation for various geometries.
The package provides simple implementations of standard PDEs, but extensions are
simple to realize.
In particular, the differential operator $\mathcal D$ can be implemented in pure
Python for initial testing, while a more specialized version compiled
with `numba` [@numba] might be added later for speed.
This approach allows fast testing of new PDEs while also providing an avenue
for efficient calculations later.

The flexibility of `py-pde` is one of its key features.
For instance, while the package implements forward and backward Euler methods as
well as a Runge-Kutta scheme, users might require more sophisticated solvers.
We already provide a wrapper for the excellent `scipy.integrate.solve_ivp` method
from the SciPy package [@SciPy2020] and further additions are straightforward.
Finally, the explicit Euler stepper provided by `py-pde` also supports
stochastic differential equations in the Itô representation.
The standard PDE classes support additive Gaussian white noise, but
alternatives, including multiplicative noise, can be specified in user-defined
classes.
This feature allows users to quickly test the effect of noise on 
dynamical systems without in-depth knowledge of the associated numerical
implementation.

Finally, the package provides many convenience methods that allow analyzing
simulations on the fly, storing data persistently, and visualizing the temporal
evolution of quantities of interest.
These features might be helpful even when not dealing with PDEs.
For instance, the result of applying differential operators on the discretized
fields can be visualized directly. 
Here, the excellent integration of `matplotlib` [@matplotlib] into 
Jupyter notebooks [@ipython] allows for an efficient workflow.

The `py-pde` package employs a consistent object-oriented approach, where each
component can be extended and some can even be used in isolation.
For instance, the numba-compiled finite-difference operators, which support
flexible boundary conditions, can be applied to `numpy.ndarrays` directly, e.g., 
in custom applications.
Generally, the just-in-time compilation provided by numba [@numba] allows for
numerically efficient code while making deploying code easy.
In particular, the package can be distributed to a cluster using `pip` without
worrying about setting paths or compiling source code. 

The `py-pde` package joins a long list of software packages that aid researchers
in analyzing PDEs.
Lately, there have been several attempts at simplifying the process of
translating the mathematical formulation of a PDE to a numerical implementation 
on the computer.
Most notably, the finite-difference approach has been favored by the packages
`scikit-finite-diff` [@Cellier2019] and `Devito` [@devito].
Conversely, finite-element and finite-volume methods provide more flexibility in
the geometries considered and have been used in major packages, including
`FEniCS` [@Fenics], `FiPy` [@FiPy], `pyclaw` [@pyclaw], and `SfePy` [@SfePy].
Finally, spectral methods are another popular approach for calculating
differentials of discretized fields, e.g., in the `dedalus project` [@dedalus].
While these methods could in principle also be implemented in `py-pde`, they are
limited to a small set of viable boundary conditions and are thus not the 
primary focus.
Instead, `py-pde` aims at providing a full toolchain for creating,
simulating, and analyzing PDEs and the associated fields.
While being useful in research, `py-pde` might thus also suitable for education.  


# Acknowledgements

I am thankful to Jan Kirschbaum, Ajinkya Kulkarni, Estefania Vidal, and Noah
Ziethen for discussions and critical testing of this package. 

# References
