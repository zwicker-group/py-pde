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



Curvilinear coordinates
"""""""""""""""""""""""
The package supports multiple curvilinear coordinate systems. They allow to exploit
symmetries present in physical systems. Consequently, many grids implemented in
`py-pde` inherently assume symmetry of the described fields. However, a drawback of
curvilinear coordinates are the fact that the basis vectors now depend on position,
which makes tensor fields less intuitive and complicates the expression of differential
operators. To avoid confusion, we here specify the used coordinate systems explictely:

Polar coordinates
-----------------
Polar coordinates describe points by a radius :math:`r` and an angle :math:`\phi` in a
two-dimensional coordinates system. They are defined by the transformation

.. math::
    \begin{cases}
        x = r \cos(\phi) &\\
        y = r \sin(\phi) &
    \end{cases}
    \text{for} \; r \in [0, \infty] \;
    \text{and} \; \phi \in [0, 2\pi)


The associated symmetric grid :class:`~pde.grids.spherical.PolarSymGrid` assumes that
fields only depend on the radial coordinate :math:`r`. Note that vector and tensor
fields can still have components in the polar direction. In particular, vector fields
still have two components: :math:`\vec v(r) = v_r(r) \vec e_r +  v_\phi(r) \vec e_\phi`. 


Spherical coordinates
---------------------
Spherical coordinates describe points by a radius :math:`r`, an azimuthal angle
:math:`\theta`, and a polar angle :math:`\phi`. The conversion to ordinary Cartesian
coordinates reads 

.. math::
    \begin{cases}
        x = r \sin(\theta) \cos(\phi) &\\
        y = r \sin(\theta) \sin(\phi) &\\
        z = r \cos(\theta)
    \end{cases}
    \text{for} \; r \in [0, \infty], \;
    \theta \in [0, \pi], \; \text{and} \;
    \phi \in [0, 2\pi)
    

The associated symmetric grid  :class:`~pde.grids.spherical.SphericalSymGrid`
assumes that fields only depend on the radial coordinate :math:`r`. Note that vector and
tensor fields can still have components in the two angular direction. 

.. warning::
   Not all results of differential operators on vectorial and tensorial fields can be
   expressed in terms of fields that only depend on the radial coordinate :math:`r`.
   In particular, the gradient of a vector field can only be calculated if the azimuthal
   component of the vector field vanishes. Similarly, the divergence of a tensor field
   can only be taken in special situations.

    

Cylindrical coordinates
----------------------- 
Cylindrical coordinates describe points by a radius :math:`r`, an axial coordinate
:math:`z`, and a polar angle :math:`\phi`. The conversion to ordinary Cartesian
coordinates reads 

.. math::
    \begin{cases}
        x = r \cos(\phi) &\\
        y = r  \sin(\phi) &\\
        z = z
    \end{cases}
    \text{for} \; r \in [0, \infty], 
    z \in \mathbb{R}, \; \text{and} \;
    \phi \in [0, 2\pi)


The associated symmetric grid  :class:`~pde.grids.cylindrical.CylindricalSymGrid`
assumes that fields only depend on the coordinates :math:`r` and :math:`z`. Vector and
tensor fields still specify all components in the three-dimensional space. 

.. warning::
   The order of components in the vector and tensor fields defined on cylindrical grids
   is different than in ordinary math. While it is common to use :math:`(r, \phi, z)`,
   we here use the order :math:`(r, z, \phi)`. It might thus be best to access
   components by name instead of index.

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
 
