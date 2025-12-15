"""Package defining fields, which contain the actual data stored on discrete grids.

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
from .collection import FieldCollection
from .datafield_base import DataFieldBase  # noqa: F401
from .scalar import ScalarField
from .tensorial import Tensor2Field
from .vectorial import VectorField

__all__ = ["FieldCollection", "ScalarField", "Tensor2Field", "VectorField"]
