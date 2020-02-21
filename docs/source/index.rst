'py-pde' python package
=========================

The `py-pde` python package provides methods and classes useful for solving
partial differential equations (PDEs) of the form

.. math::
	\partial_t u(\boldsymbol x, t) = \mathcal D[u(\boldsymbol x, t)] 
		+ \eta(u, \boldsymbol x, t) \;,

where :math:`\mathcal D` is a (non-lienar) differential operator that defines
the time evolution of a (set of) physical fields :math:`u` with possibly
tensorial character, which depend on spatial coordinates :math:`\boldsymbol x`
and time :math:`t`.
The framework also supports stochastic differential equations, where the noise
is represented by :math:`\eta` above. Note that we here represent stochastic
differential equations in the Itô representation.


Contents
--------

.. toctree::
    :maxdepth: 4

    installation
    quickstart/quickstart
    packages/pde
 


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
