Mathematical basics
^^^^^^^^^^^^^^^^^^^

To solve partial differential equations (PDEs), the `py-pde` package provides
differential operators to express spatial derivatives.
These operators are implemented using the `finite difference method 
<https://en.wikipedia.org/wiki/Finite_difference_method>`_ to support various 
boundary conditions.
The time evolution of the PDE is then calculated using the method of lines by
explicitly discretizing space using the grid classes. This reduces the PDEs to
a set of ordinary differential equations, which can be solved using standard
methods as described below.


Spatial discretization
""""""""""""""""""""""


.. image:: /_images/discretization_cropped.*
   :alt: Schematic of the discretization scheme
   :width: 350px
   :class: float-right

The finite differences scheme used by `py-pde` is currently restricted to 
orthogonal coordinate systems with uniform discretization.
Because of the orthogonality, each axis of the grid can be discretized
independently.
For simplicity, we only consider uniform grids, where the support points  are
spaced equidistantly along a given axis, i.e., the discretization
:math:`\Delta x` is constant.
If a given axis covers values in a range
:math:`[x_\mathrm{min}, x_\mathrm{max}]`, a discretization with :math:`N`
support points can then be though of as covering the axis with :math:`N`
equal-sized boxes; see inset.
Field values are then specified for each box, i.e., the support points lie at
the centers of the box:

.. math::

        x_i &= x_\mathrm{min} + \left(i + \frac12\right) \Delta x
        \quad \text{for} \quad i = 0, \ldots, N - 1
    \\
        \Delta x &= \frac{x_\mathrm{max} - x_\mathrm{min}}{N}
        
which is also indicated in the inset.
Differential operators are implemented using the usual second-order central
difference.
This requires the introducing of virtual support points at :math:`x_{-1}` and
:math:`x_N`, which can be determined from the boundary conditions at
:math:`x=x_\mathrm{min}` and :math:`x=x_\mathrm{max}`, respectively. 


Temporal evolution
""""""""""""""""""
Once the fields have been discretized, the PDE reduces to a set of coupled
ordinary differential equations (ODEs), which can be solved using standard
methods.
This reduction is also known as the method of lines.
The `py-pde` package implements the simple Euler scheme and a more advanced
`Runge-Kutta scheme <https://en.wikipedia.org/wiki/Runge–Kutta_methods>`_ in 
the :class:`~pde.solvers.explicit.ExplicitSolver` class.   
For the simple implementations of these explicit methods, the user needs to
specify a time step, which will be kept fixed.
One problem with explicit solvers is that they require small time steps for some
PDEs, which are then often called 'stiff PDEs'.
Stiff PDEs can sometimes be solved more efficiently by using implicit methods.
This package provides a simple implementation of the `Backward Euler method
<https://en.wikipedia.org/wiki/Backward_Euler_method>`_ in the
:class:`~pde.solvers.implicit.ImplicitSolver` class.
Finally, more advanced methods are available by wrapping the
:func:`scipy.integrate.solve_ivp` in the
:class:`~pde.solvers.scipy.ScipySolver` class.
 
