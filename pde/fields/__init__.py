"""Defines fields, which contain the actual data stored on a discrete grid.

.. autosummary::
   :nosignatures:

   ~scalar.ScalarField
   ~vectorial.VectorField
   ~tensorial.Tensor2Field
   ~collection.FieldCollection


Inheritance structure of the classes:


.. inheritance-diagram:: scalar.ScalarField vectorial.VectorField tensorial.Tensor2Field
        collection.FieldCollection
   :parts: 1

The details of the classes are explained below:

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .base import FieldBase  # noqa: F401
from .collection import FieldCollection  # noqa: F401
from .datafield_base import DataFieldBase  # noqa: F401
from .scalar import ScalarField  # noqa: F401
from .tensorial import Tensor2Field  # noqa: F401
from .vectorial import VectorField  # noqa: F401
