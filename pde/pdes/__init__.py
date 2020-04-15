"""
Package that defines PDEs describing physical systems.

The examples in this package are often simple version of classical PDEs to
demonstrate various aspects of the `py-pde` package. Clearly, not all extensions
to these PDEs can be covered here, but this should serve as a starting point for
custom investigations.

Publicly available methods should take fields with grid information and also
only return such methods. There might be corresponding private methods that
deal with raw data for faster simulations.


.. autosummary::
   :nosignatures:

   ~pde.PDE
   ~allen_cahn.AllenCahnPDE
   ~cahn_hilliard.CahnHilliardPDE
   ~diffusion.DiffusionPDE
   ~kpz_interface.KPZInterfacePDE
   ~kuramoto_sivashinsky.KuramotoSivashinskyPDE
   ~swift_hohenberg.SwiftHohenbergPDE
   ~wave.WavePDE

Additionally, we offer two solvers for typical elliptical PDEs:


.. autosummary::
   :nosignatures:

   ~laplace.solve_laplace_equation
   ~laplace.solve_poisson_equation

   
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de> 
"""

from .base import PDEBase
from .pde import PDE
from .allen_cahn import AllenCahnPDE
from .cahn_hilliard import CahnHilliardPDE
from .diffusion import DiffusionPDE
from .kpz_interface import KPZInterfacePDE
from .kuramoto_sivashinsky import KuramotoSivashinskyPDE
from .swift_hohenberg import SwiftHohenbergPDE
from .wave import WavePDE

from .laplace import solve_laplace_equation
from .laplace import solve_poisson_equation
