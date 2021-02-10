Advanced usage
^^^^^^^^^^^^^^

Boundary conditions
"""""""""""""""""""
Boundary conditions can be specified for both sides of each axis individually.
For instance, one can enforce the value of a field to be `4` at the lower side and its
derivative (in the outward direction) to be `2` on the upper side using the following
code:

.. code-block:: python

    bc_lower = {'type': 'value', 'value': 4}
    bc_upper = {'type': 'derivative', 'value': 2}
    bc = [bc_lower, bc_upper]
    
    grid = pde.UnitGrid([16])
    field = pde.ScalarField(grid)
    field.laplace(bc)
    
Here, the Laplace operator applied to the field in the last line will respect
the boundary conditions.
Note that it suffices to give one condition if both sides of the axis require the same
condition.
For instance, to enforce a value of `3` on both side, one could simply use
:code:`bc = {'type': 'value', 'value': 3}`.

Boundary values that depend on space can be set by specifying a mathematical expression,
which may depend on the coordinates of all axes:

.. code-block:: python

    bc_x = [{"derivative": 0.1}, {"value": "sin(y / 2)"}]
    bc_y = {"value": "sqrt(1 + cos(x))"}
     
    grid = UnitGrid([32, 32])
    field = pde.ScalarField(grid)
    field.laplace(bc=[bc_x, bc_y])
    
.. warning::
    To interpret arbitrary expressions, the package uses :func:`exec`. It
    should therefore not be used in a context where malicious input could occur.
        
Inhomogeneous values can also be specified by directly supplying an array, whose shape
needs to be compatible with the boundary, i.e., it needs to have the same shape as the
grid but with the dimension of the axis along which the boundary is specified removed. 

One important aspect about boundary conditions is that they need to respect the
periodicity of the underlying grid.
For instance, in a 2d grid with one periodic axis, the following boundary condition
can be used:

.. code-block:: python

    grid = pde.UnitGrid([16, 16], periodic=[True, False])
    field = pde.ScalarField(grid)
    bc = ['periodic', {'derivative': 0}]
    field.laplace(bc)
    
For convenience, this typical situation can be described with the special boundary
condition `natural`, e.g., calling the Laplace operator using `field.laplace('natural')`
is identical to the example above. Alternatively, this condition can be called
`auto_periodic_neumann` to stress that this chooses between periodic and Neumann
boundary conditions automatically. Similarly, the special condition
`auto_periodic_dirichlet` enforces periodic boundary conditions or Dirichlet boundary
condition (vanishing value), depending on the periodicity of the underlying grid. 


Custom PDE classes
""""""""""""""""""
To implement a new PDE in a way that all of the machinery of `py-pde` can be
used, one needs to subclass :class:`~pde.pdes.base.PDEBase` and overwrite at 
least the :meth:`~pde.pdes.base.PDEBase.evolution_rate` method.
A simple implementation for the Kuramoto–Sivashinsky equation could read 

.. code-block:: python

    class KuramotoSivashinskyPDE(PDEBase):
        
        def evolution_rate(self, state, t=0):
            """ numpy implementation of the evolution equation """
            state_lapacian = state.laplace(bc='natural')
            return (- state_lapacian.laplace(bc='natural')
                    - state_lapacian
                    - 0.5 * state.gradient(bc='natural').to_scalar('squared_sum'))

A slightly more advanced example would allow for class attributes that for
instance define the boundary conditions and the diffusivity:

.. code-block:: python

    class KuramotoSivashinskyPDE(PDEBase):
        
        def __init__(self, diffusivity=1, bc='natural', bc_laplace='natural'):
            """ initialize the class with a diffusivity and boundary conditions
            for the actual field and its second derivative """
            self.diffusivity = diffusivity
            self.bc = bc
            self.bc_laplace = bc_laplace
        
        def evolution_rate(self, state, t=0):
            """ numpy implementation of the evolution equation """
            state_lapacian = state.laplace(bc=self.bc)
            state_gradient = state.gradient(bc=self.bc)
            return (- state_lapacian.laplace(bc=self.bc_laplace)
                    - state_lapacian
                    - 0.5 * self.diffusivity * (state_gradient @ state_gradient))

We here replaced the call to :code:`to_scalar('squared_sum')` by a 
dot product with itself (using the `@` notation), which is equivalent.
Note that the numpy implementation of the right hand side of the PDE is rather
slow since it runs mostly in pure python and constructs a lot of intermediate
field classes.
While such an implementation is helpful for testing initial ideas, actual
computations should be performed with compiled PDEs as described below.


Low-level operators 
"""""""""""""""""""
This section explains how to use the low-level version of the field operators.
This is necessary for the numba-accelerated implementations described above and
it might be necessary to use parts of the `py-pde` package in other packages.


Differential operators
**********************
Applying a differential operator to an instance of
:class:`~pde.fields.scalar.ScalarField` is a simple as calling
:code:`field.laplace(bc)`, where `bc` denotes the boundary conditions.
Calling this method returns another :class:`~pde.fields.scalar.ScalarField`,
which in this case contains the discretized Laplacian of the original field.
The equivalent call using the low-level interface is

.. code-block:: python
    
    apply_laplace = field.grid.get_operator('laplace', bc)
    
    laplace_data = apply_laplace(field.data)
    
Here, the first line creates a function :code:`apply_laplace` for the given grid
:code:`field.grid` and the boundary conditions `bc`.
This function can be applied to :class:`numpy.ndarray` instances, e.g.
:code:`field.data`.
Note that the result of this call is again a :class:`numpy.ndarray`.

Similarly, a gradient operator can be defined

.. code-block:: python
    
    grid = UnitGrid([6, 8])
    apply_gradient = grid.get_operator('gradient', bc='natural')
    
    data = np.random.random((6, 8))
    gradient_data = apply_gradient(data)
    assert gradient_data.shape == (2, 6, 8)

Note that this example does not even use the field classes. Instead, it directly
defines a `grid` and the respective gradient operator.
This operator is then applied to a random field and the resulting
:class:`numpy.ndarray` represents the 2-dimensional vector field.

The :code:`get_operator` method of the grids generally supports the following
differential operators: :code:`'laplacian'`, :code:`'gradient'`,
:code:`'gradient_squared'`, :code:`'divergence'`, :code:`'vector_gradient'`,
:code:`'vector_laplace'`, and :code:`'tensor_divergence'`.
However, a complete list of operators supported by a certain grid class can be
obtained from the class property :attr:`GridClass.operators`.
New operators can be added using the class method
:meth:`GridClass.register_operator`.
 

Field integration
*****************
The integral of an instance of :class:`~pde.fields.scalar.ScalarField` is
usually determined by accessing the property :code:`field.integral`.
Since the integral of a discretized field is basically a sum weighted by the
cell volumes, calculating the integral using only :mod:`numpy` is easy:


.. code-block:: python
    
    cell_volumes = field.grid.cell_volumes
    integral = (field.data * cell_volumes).sum()

Note that :code:`cell_volumes` is a simple number for Cartesian grids, but is
an array for more complicated grids, where the cell volume is not uniform.


Field interpolation
*******************
The fields defined in the `py-pde` package also support linear interpolation
by calling :code:`field.interpolate(point)`.
Similarly to the differential operators discussed above, this call can also be
translated to code that does not use the full package:

.. code-block:: python
    
    grid = UnitGrid([6, 8])
    interpolate = grid.make_interpolator_compiled(bc='natural')
    
    data = np.random.random((6, 8))
    value = interpolate(data, np.array([3.5, 7.9]))
    
We first create a function :code:`interpolate`, which is then used to
interpolate the field data at a certain point.
Note that the coordinates of the point need to be supplied as a
:class:`numpy.ndarray` and that only the interpolation at single points is
supported.
However, iteration over multiple points can be fast when the loop is compiled
with :mod:`numba`.


Inner products
**************
For vector and tensor fields, `py-pde` defines inner products that can be
accessed conveniently using the `@`-syntax: :code:`field1 @ field2` determines
the scalar product between the two fields.
The package also provides an implementation for an dot-operator:


.. code-block:: python
    
    grid = UnitGrid([6, 8])
    field1 = VectorField.random_normal(grid)
    field2 = VectorField.random_normal(grid)
    
    dot_operator = field1.get_dot_operator()
    
    result = dot_operator(field1.data, field2.data)
    assert result.shape == (6, 8)

Here, :code:`result` is the data of the scalar field resulting from the dot
product. 


Numba-accelerated PDEs
""""""""""""""""""""""
The compiled operators introduced in the previous section can be used to
implement a compiled method for the evolution rate of PDEs.
As an example, we now extend the class :class:`KuramotoSivashinskyPDE`
introduced above:


.. code-block:: python

    from pde.tools.numba import jit
    

    class KuramotoSivashinskyPDE(PDEBase):
        
        def __init__(self, diffusivity=1, bc='natural', bc_laplace='natural'):
            """ initialize the class with a diffusivity and boundary conditions
            for the actual field and its second derivative """
            self.diffusivity = diffusivity
            self.bc = bc
            self.bc_laplace = bc_laplace
        
        
        def evolution_rate(self, state, t=0):
            """ numpy implementation of the evolution equation """
            state_lapacian = state.laplace(bc=self.bc)
            state_gradient = state.gradient(bc='natural')
            return (- state_lapacian.laplace(bc=self.bc_laplace)
                    - state_lapacian
                    - 0.5 * self.diffusivity * (state_gradient @ state_gradient))
              
                
        def _make_pde_rhs_numba(self, state):
            """ the numba-accelerated evolution equation """
            # make class attributes locally available             
            diffusivity = self.diffusivity
    
            # create operators
            laplace_u = state.grid.get_operator('laplace', bc=self.bc)
            gradient_u = state.grid.get_operator('gradient', bc=self.bc)
            laplace2_u = state.grid.get_operator('laplace', bc=self.bc_laplace)
            dot = VectorField(state.grid).get_dot_operator()
    
            @jit
            def pde_rhs(state_data, t=0):
                """ compiled helper function evaluating right hand side """
                state_lapacian = laplace_u(state_data)
                state_grad = gradient_u(state_data)
                return (- laplace2_u(state_lapacian)
                        - state_lapacian
                        - diffusivity / 2 * dot(state_grad, state_grad))
            
            return pde_rhs
        
 
To activate the compiled implementation of the evolution rate, we simply have
to overwrite the :meth:`~pde.pdes.base.PDEBase._make_pde_rhs_numba` method.
This method expects an example of the state class (e.g., an instance of
:class:`~pde.fields.scalar.ScalarField`) and returns a function that calculates
the evolution rate.
The `state` argument is necessary to define the grid and the dimensionality of
the data that the returned function is supposed to be handling.
The implementation of the compiled function is split in several parts, where we 
first copy the attributes that are required by the implementation.
This is necessary, since :mod:`numba` freezes the values when compiling the
function, so that in the example above the diffusivity cannot be altered without
recompiling.
In the next step, we create all operators that we need subsequently.
Here, we use the boundary conditions defined by the class attributes, which
requires two different laplace operators, since their boundary conditions might
differ.
In the last step, we define the actual implementation of the evolution rate as
a local function that is compiled using the :code:`jit` decorator.
Here, we use the implementation shipped with `py-pde`, which sets some default
values.
However, we could have also used the usual numba implementation.
It is important that the implementation of the evolution rate only uses python
constructs that numba can compile.  

One advantage of the numba compiled implementation is that we can now use loops,
which will be much faster than their python equivalents.
For instance, we could have written the dot product in the last line as an
explicit loop:

 
.. code-block:: python

    [...]
                
        def _make_pde_rhs_numba(self, state):
            """ the numba-accelerated evolution equation """
            # make class attributes locally available             
            diffusivity = self.diffusivity
    
            # create operators
            laplace_u = state.grid.get_operator('laplace', bc=self.bc)
            gradient_u = state.grid.get_operator('gradient', bc=self.bc)
            laplace2_u = state.grid.get_operator('laplace', bc=self.bc_laplace)
            dot = VectorField(state.grid).get_dot_operator()
            dim = state.grid.dim
    
            @jit
            def pde_rhs(state_data, t=0):
                """ compiled helper function evaluating right hand side """
                state_lapacian = laplace_u(state_data)
                state_grad = gradient_u(state_data)
                result = - laplace2_u(state_lapacian) - state_lapacian
                
                for i in range(state_data.size):
                    for j in range(dim):
                        result.flat[i] -= diffusivity / 2 * state_grad[j].flat[i]**2
                        
                return result
            
            return pde_rhs
        
Here, we extract the total number of elements in the state using its
:attr:`size` attribute and we obtain the dimensionality of the space from the
grid attribute :attr:`dim`.
Note that we access numpy arrays using their :attr:`flat` attribute to provide
an implementation that works for all dimensions.     
        
        
Configuration parameters
""""""""""""""""""""""""

Configuration parameters affect how the package behaves.
They can be set using a dictionary-like interface of the configuration
:data:`~pde.config`, which can be imported from the base package.
Here is a list of all configuration options that can be adjusted in the package:

.. package_configuration ::


.. tip::
    
    To disable parallel computing in the package, the following code could be added to
    the start of the script:
    
    
    .. code-block:: python
    
        from pde import config
        config['numba.parallel'] = False
        
        # actual code using py-pde