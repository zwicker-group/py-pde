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
date: 20 February 2020
bibliography: paper.bib
---

# Summary

Partial differential equations (PDEs) play a central role in describing the
dynamics of physical systems.
Typical equations are non-linear, so analytical solutions rarely exist.
Instead, numerical integration of such equations is used to provide insight into
their behavior and to motivate approximative solutions, which might then lead to
analytical insight.

The `py-pde` python package presented in this paper allows researchers to
quickly and conveniently simulate and analyze PDEs of the general form
$$
	\partial_t u(\boldsymbol x, t) = \mathcal D[u(\boldsymbol x, t)] 
		+ \eta(u, \boldsymbol x, t) \;,
$$
where $\mathcal D$ is a (non-lienar) differential operator that defines
the time evolution of a (set of) physical fields $u$ with possibly
tensorial character, which depend on spatial coordinates $\boldsymbol x$
and time $t$.
The framework also supports stochastic differential equations, where the noise
is represented by $\eta$ above.

The main design goal of the package was to provide a convenient way to analyze
PDEs using general methods, while at the same time allowing for enough
flexibility to easily implement more specialized code.
Moreover, the package provides convenience functions for creating suitable 
initial conditions, for controlling what is analyzed and stored during a
simulation, and for visualizing final results.
The package thus serves as a toolbox for exploring PDEs both in the professional
context and for students, who want to delve into the facinating world of
dynamical systems.


# Methods

The basic object of the `py-pde` package are scalar and tensorial fields defined
on various discretzied grids.
These grids can deal with periodic boundary conditions and they allow exploiting
spatial symmetries that might be present in the physical problem. 
For instance, a scalar field $f(z, r) = \sqrt{z} * e^{-r^2}$ in cylindrical
coordiantes can be visualized using
```
grid = CylindricalGrid(radius=5, bounds_z=[0, 10], shape=(32, 64))
field = ScalarField.from_expression(grid, 'sqrt(z) * exp(-r**2)')
field.plot()
```
The package defines common differential operators that act directly on the
fields.
For instance, the call `field.gradient('neumann')` returns a vector field on the
same cylindrical grid where the differential operators is evaluated with Neumann
boundary conditions.
Here, differential operators are evaluated using teh finite difference method
(FDM) and the package supports various boundary conditions, which can be
specified per field and boundary separately.
The discretized fields are the foundation of the `py-pde` package and allow 
the comfortable construction of initial conditions, the visualization of final
results, and the detailed investigation of intermediate data.

The main part of the `py-pde` package provides the infrastructure for solving
partial differential equations.
For instance, solving the diffusion equation on the spherical grid defined
above simply becomes
```
result = DiffusionPDE().solve(field, t_range=[0, 10])
```
Here, we use the method of lines by explicitely discretizing space using the
grid classes described above.
This reduces the PDEs to a set of ordinary differential equations, which can
be solved using standard methods.
For convenience, forward and backward Euler methods as well as a Runge-Kutta
scheme are directly implemented in the package.
Moreover, a wrapper for the excellent `scipy.integrate.solve_ivp` method from
the scipy package [@SciPy2020] exists, which provides additional methods, in
particular for stiff problems.
Note that the explicit stepper provided by `py-pde` also supports stochastic
differential equations in the Itô representation.
This feature allows users to quickly test the effect of noise onto their
dynamical systems without in depth knowledge on the associated numerical
implementation.

Finally, the package provides many convenience methods that allow analyzing
simulations on the fly, storing data persistently or visualizing the temporal
evolution of quantities of interest.



* Plays well together with jupyter
* Implemented using OOP -> easy extensibility
* JIT using numba
* Written in pure python -> easy installation


<!-- Detailed use cases -->
* Just-in-time compiled inner loops using numba
* Object-oriented design for easy extensions
* Formulate PDEs independent of geometry
* Convenience functions for visualization and analysis


# Methods


 finite difference method (FDM) + method of lines

numpy [@numpy] and scipy for basic math [@SciPy2020]

numba for accelerating [@numba]
tqdm for displaying progress [@tqdm]
sympy for parsing expressions [@sympy]


# Related software


scikit-finite-diff: [@Cellier2019]

https://gitlab.com/celliern/scikit-fdiff
- limited to cartesian coordinates, no easy working with fields

Other methods:

finite volume method (FVM)
finite element method (FEM)
spectral methods such as the Fourier-spectral (could be included)


# Acknowledgements

I am thankful to Jan Kirschbaum, Ajinkya Kulkarni, Estefania Vidal, and Noah
Ziethen for discussions and critical testing of this package. 

# References