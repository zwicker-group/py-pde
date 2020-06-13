'py-pde' python package
=========================

The `py-pde` python package provides methods and classes useful for solving
partial differential equations (PDEs) of the form

.. math::
	\partial_t u(\boldsymbol x, t) = \mathcal D[u(\boldsymbol x, t)] 
		+ \eta(u, \boldsymbol x, t) \;,

where :math:`\mathcal D` is a (non-linear) differential operator that defines
the time evolution of a (set of) physical fields :math:`u` with possibly
tensorial character, which depend on spatial coordinates :math:`\boldsymbol x`
and time :math:`t`.
The framework also supports stochastic differential equations in the Itô
representation, where the noise is represented by :math:`\eta` above.

The main audience for the package are researchers and students who want to
investigate the behavior of a PDE and get an intuitive understanding of the
role of the different terms and the boundary conditions.
To support this, `py-pde` evaluates PDEs using the methods of lines with a
finite-difference approximation of the differential operators.
Consequently, the mathematical operator :math:`\mathcal D` can be naturally
translated to a function evaluating the evolution rate of the PDE.



**Contents**

.. toctree-filt::
    :maxdepth: 2
    :numbered:

    getting_started
    :gallery:examples_gallery/index
    manual/index
    packages/pde
 


**Indices and tables**

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
