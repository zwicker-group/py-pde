'''
Defines fields, which contain the actual data stored on a discrete grid.

.. autosummary::
   :nosignatures:

   ~scalar.ScalarField
   ~vectorial.VectorField
   ~tensorial.Tensor2Field
   ~collection.FieldCollection

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
'''

from .scalar import ScalarField
from .vectorial import VectorField
from .tensorial import Tensor2Field
from .collection import FieldCollection
